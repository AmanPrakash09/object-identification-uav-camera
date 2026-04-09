from pathlib import Path
import time
import numpy as np
import torch
from rfdetr import RFDETRNano
import supervision as sv

from PIL import Image, ImageDraw, ImageFont

# ---- GStreamer capture ----
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# ---- Display ----
import pygame


# -----------------------------
# Config
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_PATH = SCRIPT_DIR / "visdrone_rfdetr_nano_best_ema.pth"
assert CKPT_PATH.exists(), f"Missing checkpoint: {CKPT_PATH}"

THRESHOLD = 0.30

# Run detection every N frames.
# For USB webcams, start with 1 for best quality/stability.
DETECT_EVERY_N = 1

# Filter tiny boxes to reduce flicker
MIN_BOX_AREA = 32 * 32

# Keep drawing tracks for this many frames after last association
DISPLAY_GRACE = 15  # tuned for ~30 FPS webcam

PHASE2_CLASSES = {
    1: "Human",
    2: "Bicycle",
    3: "Car",
    4: "Truck",
    5: "Bus",
    6: "Motorcycle",
}

TEMP_FRAME_PATH = SCRIPT_DIR / "_temp_frame.jpg"

# USB webcam config
USB_DEVICE = "/dev/video0"
W = 1280
H = 720
FPS = 30


# -----------------------------
# GStreamer pipeline helpers
# -----------------------------
def usb_gstreamer_pipeline_mjpeg(device="/dev/video0", width=1280, height=720, framerate=30):
    """
    USB webcam pipeline using MJPEG input.
    Many webcams support this efficiently at 720p/1080p.
    """
    return (
        f"v4l2src device={device} ! "
        f"image/jpeg, width=(int){width}, height=(int){height}, framerate=(fraction){framerate}/1 ! "
        f"jpegdec ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)RGB ! "
        f"appsink name=appsink drop=true max-buffers=1 sync=false"
    )


def usb_gstreamer_pipeline_raw(device="/dev/video0", width=1280, height=720, framerate=30):
    """
    USB webcam pipeline using raw V4L2 frames.
    Useful fallback if MJPEG isn't supported by the webcam.
    """
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, framerate=(fraction){framerate}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)RGB ! "
        f"appsink name=appsink drop=true max-buffers=1 sync=false"
    )


class GstCamera:
    def __init__(self, pipeline: str):
        Gst.init(None)
        self.pipeline_str = pipeline
        self.pipeline = Gst.parse_launch(pipeline)
        self.appsink = self.pipeline.get_by_name("appsink")
        if self.appsink is None:
            raise RuntimeError("appsink not found in pipeline")

        self.appsink.set_property("drop", True)
        self.appsink.set_property("max-buffers", 1)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"Failed to start GStreamer pipeline:\n{pipeline}")

    def read(self):
        sample = self.appsink.emit("try-pull-sample", 100_000_000)  # 0.1s
        if sample is None:
            return False, None

        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        w = s.get_value("width")
        h = s.get_value("height")

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return False, None

        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3))  # RGB
            return True, frame
        finally:
            buf.unmap(mapinfo)

    def release(self):
        self.pipeline.set_state(Gst.State.NULL)


def open_usb_camera(device="/dev/video0", width=1280, height=720, framerate=30):
    """
    Try MJPEG first, then fall back to raw.
    """
    pipelines = [
        usb_gstreamer_pipeline_mjpeg(device=device, width=width, height=height, framerate=framerate),
        usb_gstreamer_pipeline_raw(device=device, width=width, height=height, framerate=framerate),
    ]

    last_error = None
    for idx, pipeline in enumerate(pipelines, start=1):
        try:
            print(f"Trying USB webcam pipeline #{idx}...")
            cam = GstCamera(pipeline)

            # Probe a few frames to ensure it actually works
            for _ in range(10):
                ret, frame = cam.read()
                if ret and frame is not None:
                    print(f"USB webcam opened successfully with pipeline #{idx}.")
                    print(pipeline)
                    return cam

            cam.release()
            last_error = RuntimeError(f"Pipeline #{idx} started but produced no frames.")
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not open USB webcam {device}. Last error: {last_error}")


# -----------------------------
# Model init
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Device:", device)
print("Checkpoint:", CKPT_PATH)

model = RFDETRNano(pretrain_weights=str(CKPT_PATH))
model.model.model.to(device)
model.model.model.eval()

try:
    model.optimize_for_inference()
    print("Model optimized for inference.")
except Exception:
    pass


# -----------------------------
# ByteTrack (supervision)
# -----------------------------
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=60,              # shorter than CSI version since webcam is ~30 FPS
    minimum_matching_threshold=0.6,
    frame_rate=FPS
)

# Track metadata: track_id -> (class_id, confidence)
track_meta = {}

# Track last seen frame index: track_id -> frame_idx
track_last_seen = {}


def rfdetr_to_sv_detections(det, min_conf=0.30):
    xyxy = np.asarray(det.xyxy, dtype=np.float32)
    conf = np.asarray(det.confidence, dtype=np.float32)
    cls = np.asarray(det.class_id, dtype=np.int32)

    keep = conf >= float(min_conf)
    return sv.Detections(
        xyxy=xyxy[keep],
        confidence=conf[keep],
        class_id=cls[keep],
    )


def filter_small_boxes(dets: sv.Detections, min_area: int):
    if len(dets) == 0:
        return dets
    xyxy = dets.xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return dets[areas >= float(min_area)]


def build_tensors_from_detections(dets: sv.Detections) -> np.ndarray:
    if len(dets) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.hstack([
        dets.xyxy.astype(np.float32),
        dets.confidence.reshape(-1, 1).astype(np.float32),
    ])


def active_tracks_to_boxes_ids(tracker_obj):
    tracks = getattr(tracker_obj, "tracked_tracks", [])
    boxes, ids = [], []
    for t in tracks:
        boxes.append(np.array(t.tlbr, dtype=np.float32))
        ids.append(int(t.external_track_id))
    return boxes, ids


# -----------------------------
# Display (pygame)
# -----------------------------
def init_display(width: int, height: int):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("RF-DETR + ByteTrack (USB webcam)")
    return screen


def show_frame(screen, frame_rgb: np.ndarray):
    surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
    screen.blit(surf, (0, 0))
    pygame.display.flip()


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            return True
    return False


# -----------------------------
# Main loop
# -----------------------------
def run_live():
    cam = open_usb_camera(device=USB_DEVICE, width=W, height=H, framerate=FPS)
    screen = init_display(W, H)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    print(f"USB camera opened on {USB_DEVICE}. Press 'q' to quit.")
    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            if should_quit():
                break

            ret, frame_rgb = cam.read()
            if not ret:
                continue

            if frame_idx % DETECT_EVERY_N == 0:
                # RFDETR.predict() expects a path in your current setup
                Image.fromarray(frame_rgb).save(TEMP_FRAME_PATH, quality=90)

                det = model.predict(str(TEMP_FRAME_PATH), threshold=THRESHOLD)

                detections = rfdetr_to_sv_detections(det, min_conf=THRESHOLD)
                detections = filter_small_boxes(detections, MIN_BOX_AREA)

                # Keep same behavior as your original script
                _ = tracker.update_with_tensors(build_tensors_from_detections(detections))

                det_with_ids = tracker.update_with_detections(detections)
                if len(det_with_ids) > 0 and getattr(det_with_ids, "tracker_id", None) is not None:
                    for i in range(len(det_with_ids)):
                        tid = int(det_with_ids.tracker_id[i])
                        cls = int(det_with_ids.class_id[i])
                        conf = float(det_with_ids.confidence[i])

                        track_meta[tid] = (cls, conf)
                        track_last_seen[tid] = frame_idx

            # Draw active tracks using PIL
            img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(img)

            boxes, ids = active_tracks_to_boxes_ids(tracker)
            for box, tid in zip(boxes, ids):
                last = track_last_seen.get(tid, -10**9)
                if frame_idx - last > DISPLAY_GRACE:
                    continue

                x1, y1, x2, y2 = box.astype(int).tolist()
                cls, conf = track_meta.get(tid, (-1, 0.0))
                label = PHASE2_CLASSES.get(cls, f"class_{cls}") if cls != -1 else "unknown"
                text = f"ID {tid}: {label} {conf:.2f}"

                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                tx, ty = x1, max(0, y1 - th - 4)
                draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=(0, 255, 0))
                draw.text((tx + 3, ty + 2), text, fill=(0, 0, 0), font=font)

            out_rgb = np.array(img, dtype=np.uint8)

            if frame_idx % FPS == 0 and frame_idx > 0:
                elapsed = time.time() - t0
                fps_eff = frame_idx / max(elapsed, 1e-6)
                print(f"Live FPS: ~{fps_eff:.2f} | detect every {DETECT_EVERY_N} frames")

            show_frame(screen, out_rgb)
            frame_idx += 1

    finally:
        cam.release()
        pygame.quit()
        if TEMP_FRAME_PATH.exists():
            try:
                TEMP_FRAME_PATH.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    run_live()