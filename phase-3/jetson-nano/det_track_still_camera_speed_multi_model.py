from pathlib import Path
import time
import inspect
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

RGB_CKPT_PATH = SCRIPT_DIR / "visdrone_rfdetr_nano_best_ema.pth"
IR_CKPT_PATH = SCRIPT_DIR / "dronevehicle_rfdetr_nano_best_ema.pth"

assert RGB_CKPT_PATH.exists(), f"Missing checkpoint: {RGB_CKPT_PATH}"
assert IR_CKPT_PATH.exists(), f"Missing checkpoint: {IR_CKPT_PATH}"

THRESHOLD = 0.30

# Run detection every N frames.
# For USB webcams, start with 1 for best quality/stability.
DETECT_EVERY_N = 1

# Filter tiny boxes to reduce flicker
MIN_BOX_AREA = 32 * 32

# Keep drawing tracks for this many frames after last association
DISPLAY_GRACE = 15

# USB webcam config
USB_DEVICE = "/dev/video0"
W = 1280
H = 720
FPS = 30

TEMP_FRAME_PATH = SCRIPT_DIR / "_temp_frame.jpg"

PHASE2_CLASSES = {
    1: "Human",
    2: "Bicycle",
    3: "Car",
    4: "Truck",
    5: "Bus",
    6: "Motorcycle",
}

SPEED_ESTIMATION_CLASS_IDS = {1, 2, 3, 4, 5, 6}

# -----------------------------
# Speed-estimation config
# -----------------------------
CLASS_DIMENSION_PRIORS = {
    "Human": {
        "width_m": 0.50,
        "length_m": 0.50,
        "height_m": 1.70,
        "rep_point_mode": "bottom_center",
        "rep_y_offset_px": 8.0,
    },
    "Bicycle": {
        "width_m": 0.60,
        "length_m": 1.80,
        "height_m": 1.10,
        "rep_point_mode": "bottom_center",
        "rep_y_offset_px": 8.0,
    },
    "Car": {
        "width_m": 2.05,
        "length_m": 5.10,
        "height_m": 1.50,
        "rep_point_mode": "bottom_center",
        "rep_y_offset_px": 8.0,
    },
    "Truck": {
        "width_m": 2.50,
        "length_m": 7.00,
        "height_m": 3.20,
        "rep_point_mode": "bottom_center",
        "rep_y_offset_px": 8.0,
    },
    "Bus": {
        "width_m": 2.55,
        "length_m": 12.00,
        "height_m": 3.20,
        "rep_point_mode": "bottom_center",
        "rep_y_offset_px": 8.0,
    },
    "Motorcycle": {
        "width_m": 0.80,
        "length_m": 2.20,
        "height_m": 1.20,
        "rep_point_mode": "bottom_center",
        "rep_y_offset_px": 8.0,
    },
    "Other": {
        "width_m": 2.00,
        "length_m": 4.50,
        "height_m": 1.80,
        "rep_point_mode": "bottom_center",
        "rep_y_offset_px": 8.0,
    },
}

CLASS_PRIOR_ALIASES = {
    "person": "Human",
    "pedestrian": "Human",
    "people": "Human",
    "bike": "Bicycle",
    "bicycle": "Bicycle",
    "cyclist": "Bicycle",
    "car": "Car",
    "sedan": "Car",
    "van": "Car",
    "suv": "Car",
    "pickup": "Car",
    "truck": "Truck",
    "lorry": "Truck",
    "semi": "Truck",
    "bus": "Bus",
    "coach": "Bus",
    "motorcycle": "Motorcycle",
    "motorbike": "Motorcycle",
    "other": "Other",
    "motor": "Other",
    "vehicle": "Other",
}

MIN_REASONABLE_SPEED_KMH = 0.0
MAX_REASONABLE_SPEED_KMH = 250.0
SPEED_WINDOW = 5
MPP_WINDOW = 5
BORDER_MARGIN_PX = 20


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


def load_model(ckpt_path: Path):
    print("Loading checkpoint:", ckpt_path)
    model = RFDETRNano(pretrain_weights=str(ckpt_path))
    model.model.model.to(device)
    model.model.model.eval()

    try:
        model.optimize_for_inference()
        print("Model optimized for inference.")
    except Exception:
        pass

    return model


current_mode = "RGB"
current_ckpt_path = RGB_CKPT_PATH
model = load_model(current_ckpt_path)


# -----------------------------
# ByteTrack
# -----------------------------
def build_tracker(frame_rate):
    sig = inspect.signature(sv.ByteTrack.__init__)
    params = set(sig.parameters.keys())
    kwargs = {}

    if "track_activation_threshold" in params:
        kwargs["track_activation_threshold"] = 0.25
    if "lost_track_buffer" in params:
        kwargs["lost_track_buffer"] = 60
    if "minimum_matching_threshold" in params:
        kwargs["minimum_matching_threshold"] = 0.6
    if "frame_rate" in params:
        kwargs["frame_rate"] = int(round(frame_rate))

    if "track_thresh" in params:
        kwargs["track_thresh"] = 0.25
    if "track_buffer" in params:
        kwargs["track_buffer"] = 60
    if "match_thresh" in params:
        kwargs["match_thresh"] = 0.6

    try:
        tracker = sv.ByteTrack(**kwargs)
    except TypeError:
        tracker = sv.ByteTrack()

    if hasattr(tracker, "reset"):
        tracker.reset()

    return tracker


tracker = build_tracker(frame_rate=FPS)

# Track metadata
track_meta = {}       # track_id -> (class_id, label, confidence)
track_last_seen = {}  # track_id -> frame_idx
track_histories = {}  # track_id -> history dict


def reset_tracking_state():
    global tracker, track_meta, track_last_seen, track_histories
    tracker = build_tracker(frame_rate=FPS)
    track_meta = {}
    track_last_seen = {}
    track_histories = {}


# -----------------------------
# Detection helpers
# -----------------------------
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
# Speed-estimation helpers
# -----------------------------
def canonicalize_class_name(label: str) -> str:
    if label is None:
        return "Other"
    label = str(label).strip()
    if label in CLASS_DIMENSION_PRIORS:
        return label
    lowered = label.lower()
    if lowered in CLASS_PRIOR_ALIASES:
        return CLASS_PRIOR_ALIASES[lowered]
    return "Other"


def get_class_dimension_prior(label: str) -> dict:
    canonical = canonicalize_class_name(label)
    prior = CLASS_DIMENSION_PRIORS[canonical].copy()
    prior["canonical_label"] = canonical
    prior["requested_label"] = label
    return prior


def bbox_center_xyxy(box_xyxy):
    x1, y1, x2, y2 = map(float, box_xyxy)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def bbox_bottom_center_xyxy(box_xyxy, y_offset_px=0.0):
    x1, y1, x2, y2 = map(float, box_xyxy)
    cx = 0.5 * (x1 + x2)
    cy = y2 + float(y_offset_px)
    return np.array([cx, cy], dtype=np.float32)


def representative_point_xyxy(box_xyxy, class_label):
    prior = get_class_dimension_prior(class_label)
    mode = prior["rep_point_mode"]
    y_offset_px = prior["rep_y_offset_px"]

    if mode == "center":
        return bbox_center_xyxy(box_xyxy)
    if mode == "bottom_center":
        return bbox_bottom_center_xyxy(box_xyxy, y_offset_px=y_offset_px)
    raise ValueError(f"Unsupported representative point mode: {mode}")


def bbox_touches_border(box_xyxy, image_width, image_height, margin_px=20):
    x1, y1, x2, y2 = map(float, box_xyxy)
    return (
        x1 <= margin_px or
        y1 <= margin_px or
        x2 >= (image_width - margin_px) or
        y2 >= (image_height - margin_px)
    )


def moving_average_ignore_none(values, window=5):
    recent = [float(v) for v in values if v is not None]
    if len(recent) == 0:
        return None
    recent = recent[-int(window):]
    return float(np.mean(recent))


def moving_average_nan(values, window=5):
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        vals = values[lo:i + 1]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            out[i] = np.mean(vals)
    return out


def is_reasonable_speed_kmh(speed_kmh, min_kmh=0.0, max_kmh=250.0):
    if speed_kmh is None:
        return False
    return float(min_kmh) <= float(speed_kmh) <= float(max_kmh)


def make_empty_track_history():
    return {
        "frames": [],
        "boxes": [],
        "labels": [],
        "confidences": [],
        "image_points": [],
        "mpp_x": [],
        "mpp_y": [],
        "mpp_geom": [],
        "mpp_smoothed": [],
        "raw_speeds_kmh": [],
        "smoothed_speeds_kmh": [],
    }


def safe_bbox_dims(box_xyxy):
    x1, y1, x2, y2 = map(float, box_xyxy)
    w_px = max(float(x2 - x1), 1e-6)
    h_px = max(float(y2 - y1), 1e-6)
    return w_px, h_px


def compute_local_scale_from_box(box_xyxy, class_label):
    prior = get_class_dimension_prior(class_label)
    width_m = float(prior["width_m"])
    length_m = float(prior["length_m"])

    bbox_w_px, bbox_h_px = safe_bbox_dims(box_xyxy)

    mpp_x = width_m / bbox_w_px
    mpp_y = length_m / bbox_h_px
    mpp_geom = float(np.sqrt(max(mpp_x, 1e-12) * max(mpp_y, 1e-12)))

    return {
        "canonical_label": prior["canonical_label"],
        "width_m": width_m,
        "length_m": length_m,
        "bbox_width_px": bbox_w_px,
        "bbox_height_px": bbox_h_px,
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "mpp_geom": mpp_geom,
    }


def append_track_observation(track_history, frame_idx, box_xyxy, class_label, confidence, fps):
    rep_pt = representative_point_xyxy(box_xyxy, class_label)
    scale_info = compute_local_scale_from_box(box_xyxy, class_label)

    track_history["frames"].append(int(frame_idx))
    track_history["boxes"].append(np.asarray(box_xyxy, dtype=np.float32))
    track_history["labels"].append(class_label)
    track_history["confidences"].append(float(confidence))
    track_history["image_points"].append(rep_pt)

    track_history["mpp_x"].append(scale_info["mpp_x"])
    track_history["mpp_y"].append(scale_info["mpp_y"])
    track_history["mpp_geom"].append(scale_info["mpp_geom"])

    mpp_smoothed_arr = moving_average_nan(
        np.asarray(track_history["mpp_geom"], dtype=np.float64),
        window=MPP_WINDOW
    )
    current_mpp_smoothed = float(mpp_smoothed_arr[-1]) if np.isfinite(mpp_smoothed_arr[-1]) else None
    track_history["mpp_smoothed"].append(current_mpp_smoothed)

    raw_speed_kmh = None
    if len(track_history["image_points"]) >= 2:
        p1 = np.asarray(track_history["image_points"][-2], dtype=np.float64)
        p2 = np.asarray(track_history["image_points"][-1], dtype=np.float64)

        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        disp_px = float(np.sqrt(dx * dx + dy * dy))

        prev_mpp = track_history["mpp_smoothed"][-2]
        curr_mpp = track_history["mpp_smoothed"][-1]

        if prev_mpp is not None and curr_mpp is not None:
            local_mpp = 0.5 * (float(prev_mpp) + float(curr_mpp))
            dt = 1.0 / float(fps)
            raw_speed_kmh = (disp_px * local_mpp / dt) * 3.6

            if not is_reasonable_speed_kmh(
                raw_speed_kmh,
                min_kmh=MIN_REASONABLE_SPEED_KMH,
                max_kmh=MAX_REASONABLE_SPEED_KMH,
            ):
                raw_speed_kmh = None

    track_history["raw_speeds_kmh"].append(raw_speed_kmh)

    smoothed_speed_kmh = moving_average_ignore_none(
        track_history["raw_speeds_kmh"],
        window=SPEED_WINDOW
    )
    track_history["smoothed_speeds_kmh"].append(smoothed_speed_kmh)

    return {
        "rep_pt": rep_pt,
        "scale_info": scale_info,
        "raw_speed_kmh": raw_speed_kmh,
        "smoothed_speed_kmh": smoothed_speed_kmh,
    }


# -----------------------------
# Display (pygame)
# -----------------------------
def init_display(width: int, height: int):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("RF-DETR + ByteTrack + Speed Estimation (USB webcam)")
    return screen


def show_frame(screen, frame_rgb: np.ndarray):
    surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
    screen.blit(surf, (0, 0))
    pygame.display.flip()


def draw_overlay_message(frame_rgb: np.ndarray, message: str):
    """
    Draw a centered overlay message on top of the frame.
    """
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), message, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad_x = 30
    pad_y = 20
    box_w = tw + 2 * pad_x
    box_h = th + 2 * pad_y

    x1 = (img.width - box_w) // 2
    y1 = (img.height - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

    # Dim the background slightly
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([0, 0, img.width, img.height], fill=(0, 0, 0, 80))
    overlay_draw.rounded_rectangle([x1, y1, x2, y2], radius=16, fill=(0, 0, 0, 180))
    img = Image.alpha_composite(img.convert("RGBA"), overlay)

    draw = ImageDraw.Draw(img)
    draw.text(
        (x1 + pad_x, y1 + pad_y),
        message,
        fill=(255, 255, 255),
        font=font,
    )

    return np.array(img.convert("RGB"), dtype=np.uint8)


def handle_events():
    """
    Returns:
        "quit"   -> user requested quit
        "toggle" -> user pressed c to switch model
        None     -> no action
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return "quit"
            if event.key == pygame.K_c:
                return "toggle"
    return None


# -----------------------------
# Main loop
# -----------------------------
def run_live():
    global model, current_mode, current_ckpt_path

    cam = open_usb_camera(device=USB_DEVICE, width=W, height=H, framerate=FPS)
    screen = init_display(W, H)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    print(f"USB camera opened on {USB_DEVICE}. Press 'q' to quit.")
    print(f"Press 'c' to toggle between RGB and infrared models.")
    print(f"Current mode: {current_mode}")

    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            action = handle_events()
            if action == "quit":
                break

            ret, frame_rgb = cam.read()
            if not ret:
                continue

            # Handle model switching
            if action == "toggle":
                if current_mode == "RGB":
                    next_mode = "Infrared"
                    next_ckpt_path = IR_CKPT_PATH
                    overlay_text = "switching to infrared"
                else:
                    next_mode = "RGB"
                    next_ckpt_path = RGB_CKPT_PATH
                    overlay_text = "switching to RGB"

                # Pause inference and show overlay
                paused_frame = draw_overlay_message(frame_rgb, overlay_text)
                show_frame(screen, paused_frame)

                print(f"Switching model: {current_mode} -> {next_mode}")
                model = load_model(next_ckpt_path)
                current_mode = next_mode
                current_ckpt_path = next_ckpt_path

                # Reset tracking and speed histories when switching models
                reset_tracking_state()

                print(f"Switched to {current_mode} model ({current_ckpt_path.name})")
                # Resume normal loop on next frame
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

                        label = PHASE2_CLASSES.get(cls, f"class_{cls}")
                        track_meta[tid] = (cls, label, conf)
                        track_last_seen[tid] = frame_idx

            img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(img)

            # Draw current mode at top-left
            mode_text = f"Mode: {current_mode}"
            bbox = draw.textbbox((0, 0), mode_text, font=font)
            mw = bbox[2] - bbox[0]
            mh = bbox[3] - bbox[1]
            draw.rectangle([10, 10, 10 + mw + 6, 10 + mh + 4], fill=(255, 255, 0))
            draw.text((13, 12), mode_text, fill=(0, 0, 0), font=font)

            boxes, ids = active_tracks_to_boxes_ids(tracker)
            for box, tid in zip(boxes, ids):
                last = track_last_seen.get(tid, -10**9)
                if frame_idx - last > DISPLAY_GRACE:
                    continue

                x1, y1, x2, y2 = box.astype(int).tolist()
                cls, label, conf = track_meta.get(tid, (-1, "unknown", 0.0))

                speed_text = ""
                rep_pt = None

                if cls in SPEED_ESTIMATION_CLASS_IDS:
                    if not bbox_touches_border(box, W, H, margin_px=BORDER_MARGIN_PX):
                        if tid not in track_histories:
                            track_histories[tid] = make_empty_track_history()

                        result = append_track_observation(
                            track_history=track_histories[tid],
                            frame_idx=frame_idx,
                            box_xyxy=box,
                            class_label=label,
                            confidence=conf,
                            fps=FPS,
                        )

                        rep_pt = result["rep_pt"]
                        smooth_speed = result["smoothed_speed_kmh"]

                        if smooth_speed is not None:
                            speed_text = f" | {smooth_speed:.2f} km/h"

                text = f"ID {tid}: {label} {conf:.2f}{speed_text}"

                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

                if rep_pt is not None:
                    rx, ry = int(round(rep_pt[0])), int(round(rep_pt[1]))
                    draw.ellipse([rx - 3, ry - 3, rx + 3, ry + 3], fill=(255, 0, 0))

                    pts = np.array(track_histories[tid]["image_points"][-20:], dtype=np.float32)
                    if len(pts) >= 2:
                        pts = np.round(pts).astype(int)
                        for j in range(1, len(pts)):
                            p1 = tuple(pts[j - 1])
                            p2 = tuple(pts[j])
                            draw.line([p1, p2], fill=(0, 0, 255), width=2)

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
                print(
                    f"Live FPS: ~{fps_eff:.2f} | detect every {DETECT_EVERY_N} frames | mode: {current_mode}"
                )

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