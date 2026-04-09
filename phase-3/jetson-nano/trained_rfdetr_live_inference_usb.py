from pathlib import Path
import time
import math
import numpy as np
import torch
import cv2
from rfdetr import RFDETRNano
import supervision as sv

from PIL import Image, ImageDraw, ImageFont

# ---- GStreamer capture ----
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# ---- Display ----
import pygame


# ============================================================
# Config
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

RGB_CKPT_PATH = SCRIPT_DIR / "visdrone_rfdetr_nano_best_ema.pth"
IR_CKPT_PATH = SCRIPT_DIR / "dronevehicle_rfdetr_nano_best_ema.pth"

assert RGB_CKPT_PATH.exists(), f"Missing RGB checkpoint: {RGB_CKPT_PATH}"
assert IR_CKPT_PATH.exists(), f"Missing IR checkpoint: {IR_CKPT_PATH}"

THRESHOLD = 0.30
DETECT_EVERY_N = 1
MIN_BOX_AREA = 32 * 32
DISPLAY_GRACE = 15  # tuned for ~30 FPS operation

TEMP_FRAME_PATH = SCRIPT_DIR / "_temp_frame.jpg"

# Shared runtime size/FPS
W = 1280
H = 720
FPS = 30

# USB RGB webcam path
RGB_USB_DEVICE = "/dev/video0"

CAMERA_CONFIGS = {
    0: {
        "label": "USB RGB Webcam",
        "short_label": "RGB",
        "ckpt": RGB_CKPT_PATH,
        "source_type": "usb",
        "usb_device": RGB_USB_DEVICE,
        "usb_try_mjpeg_first": True,
    },
    1: {
        "label": "IMX462 IR (CAM1)",
        "short_label": "IR",
        "ckpt": IR_CKPT_PATH,
        "source_type": "csi",
        "sensor_id": 1,
    },
}

PHASE2_CLASSES = {
    1: "Human",
    2: "Bicycle",
    3: "Car",
    4: "Truck",
    5: "Bus",
    6: "Motorcycle",
}

# ============================================================
# Speed estimator config (adapted from notebook)
# ============================================================
ALPHA_LEFT_RIGHT = 0.90
ALPHA_UP_DOWN = 1.60

FINAL_BORDER_MARGIN_PX = 20
FINAL_MPP_WINDOW = 5
FINAL_SPEED_WINDOW = 5
FINAL_MAX_REASONABLE_SPEED_KMH = 250.0

FINAL_REP_Y_OFFSET_PX = 8.0

FINAL_EGO_FEATURE_MAX_CORNERS = 600
FINAL_EGO_FEATURE_QUALITY = 0.01
FINAL_EGO_FEATURE_MIN_DISTANCE = 7
FINAL_EGO_FEATURE_BLOCK_SIZE = 7
FINAL_EGO_LK_WIN_SIZE = (21, 21)
FINAL_EGO_LK_MAX_LEVEL = 3
FINAL_EGO_LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    30,
    0.01,
)
FINAL_EGO_MIN_VALID_TRACKS = 25
FINAL_EGO_DIRECTION_SMOOTH_WINDOW = 5
FINAL_EGO_MASK_DILATE_PX = 12
FINAL_EGO_BORDER_MARGIN_PX = 10

CLASS_DIMENSION_PRIORS = {
    "Human": {"width_m": 0.50, "length_m": 0.50},
    "Bicycle": {"width_m": 0.60, "length_m": 1.80},
    "Car": {"width_m": 2.05, "length_m": 5.10},
    "Truck": {"width_m": 2.50, "length_m": 7.00},
    "Bus": {"width_m": 2.55, "length_m": 12.00},
    "Motorcycle": {"width_m": 0.80, "length_m": 2.20},
    "Other": {"width_m": 2.00, "length_m": 4.50},
}


# ============================================================
# GStreamer pipeline
# ============================================================
def csi_gstreamer_pipeline(sensor_id=0, width=1280, height=720, framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)RGB ! "
        f"appsink name=appsink drop=true max-buffers=1 sync=false"
    )


def usb_gstreamer_pipeline_mjpeg(device="/dev/video0", width=1280, height=720, framerate=30):
    return (
        f"v4l2src device={device} ! "
        f"image/jpeg, width=(int){width}, height=(int){height}, framerate=(fraction){framerate}/1 ! "
        f"jpegdec ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)RGB ! "
        f"appsink name=appsink drop=true max-buffers=1 sync=false"
    )


def usb_gstreamer_pipeline_raw(device="/dev/video0", width=1280, height=720, framerate=30):
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


def open_usb_camera(device="/dev/video0", width=1280, height=720, framerate=30, try_mjpeg_first=True):
    pipelines = []
    if try_mjpeg_first:
        pipelines.append(usb_gstreamer_pipeline_mjpeg(device=device, width=width, height=height, framerate=framerate))
        pipelines.append(usb_gstreamer_pipeline_raw(device=device, width=width, height=height, framerate=framerate))
    else:
        pipelines.append(usb_gstreamer_pipeline_raw(device=device, width=width, height=height, framerate=framerate))
        pipelines.append(usb_gstreamer_pipeline_mjpeg(device=device, width=width, height=height, framerate=framerate))

    last_error = None
    for idx, pipeline in enumerate(pipelines, start=1):
        try:
            print(f"Trying USB webcam pipeline #{idx} for {device}...")
            cam = GstCamera(pipeline)

            # Probe a few frames to make sure the pipeline actually delivers
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


# ============================================================
# Model / tracker helpers
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Device:", device)


def load_model(ckpt_path: Path):
    print(f"Loading checkpoint: {ckpt_path}")
    model = RFDETRNano(pretrain_weights=str(ckpt_path))
    model.model.model.to(device)
    model.model.model.eval()

    try:
        model.optimize_for_inference()
        print("Model optimized for inference.")
    except Exception:
        pass

    return model


def make_tracker(frame_rate=30):
    return sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.6,
        frame_rate=int(round(frame_rate)),
    )


def open_camera(camera_id: int, width: int, height: int, fps: int):
    cfg = CAMERA_CONFIGS[camera_id]
    print(f"Opening camera {camera_id}: {cfg['label']}")

    if cfg["source_type"] == "usb":
        return open_usb_camera(
            device=cfg["usb_device"],
            width=width,
            height=height,
            framerate=fps,
            try_mjpeg_first=cfg.get("usb_try_mjpeg_first", True),
        )

    if cfg["source_type"] == "csi":
        return GstCamera(
            csi_gstreamer_pipeline(
                sensor_id=cfg["sensor_id"],
                width=width,
                height=height,
                framerate=fps,
                flip_method=0,
            )
        )

    raise ValueError(f"Unsupported source_type: {cfg['source_type']}")


# ============================================================
# Detection / tracking helpers
# ============================================================
def rfdetr_to_sv_detections(det, min_conf=0.30):
    xyxy = np.asarray(det.xyxy, dtype=np.float32)
    conf = np.asarray(det.confidence, dtype=np.float32)
    cls = np.asarray(det.class_id, dtype=np.int32)

    keep = conf >= float(min_conf)
    return sv.Detections(xyxy=xyxy[keep], confidence=conf[keep], class_id=cls[keep])


def filter_small_boxes(dets: sv.Detections, min_area: int):
    if len(dets) == 0:
        return dets
    xyxy = dets.xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return dets[areas >= float(min_area)]


def active_tracks_to_boxes_ids(tracker_obj):
    tracks = getattr(tracker_obj, "tracked_tracks", [])
    boxes, ids = [], []
    for t in tracks:
        boxes.append(np.array(t.tlbr, dtype=np.float32))
        ids.append(int(t.external_track_id))
    return boxes, ids


# ============================================================
# Speed-estimation helpers
# ============================================================
def canonicalize_class_name(label: str) -> str:
    if label is None:
        return "Other"
    label = str(label).strip()
    if label in CLASS_DIMENSION_PRIORS:
        return label

    lowered = label.lower()
    aliases = {
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
        "vehicle": "Other",
    }
    return aliases.get(lowered, "Other")


def bbox_touches_border(box_xyxy, image_width, image_height, margin_px=20):
    x1, y1, x2, y2 = map(float, box_xyxy)
    return (
        x1 <= margin_px or
        y1 <= margin_px or
        x2 >= (image_width - margin_px) or
        y2 >= (image_height - margin_px)
    )


def bbox_bottom_center_xyxy(box_xyxy, y_offset_px=0.0):
    x1, y1, x2, y2 = map(float, box_xyxy)
    cx = 0.5 * (x1 + x2)
    cy = y2 + float(y_offset_px)
    return np.array([cx, cy], dtype=np.float32)


def representative_point_xyxy(box_xyxy):
    return bbox_bottom_center_xyxy(box_xyxy, y_offset_px=FINAL_REP_Y_OFFSET_PX)


def safe_bbox_dims(box_xyxy):
    x1, y1, x2, y2 = map(float, box_xyxy)
    w_px = max(float(x2 - x1), 1e-6)
    h_px = max(float(y2 - y1), 1e-6)
    return w_px, h_px


def compute_local_scale_from_box(box_xyxy, class_label):
    canonical = canonicalize_class_name(class_label)
    prior = CLASS_DIMENSION_PRIORS[canonical]

    bbox_w_px, bbox_h_px = safe_bbox_dims(box_xyxy)
    mpp_x = float(prior["width_m"]) / bbox_w_px
    mpp_y = float(prior["length_m"]) / bbox_h_px
    mpp_geom = float(np.sqrt(max(mpp_x, 1e-12) * max(mpp_y, 1e-12)))

    return {
        "canonical_label": canonical,
        "bbox_width_px": bbox_w_px,
        "bbox_height_px": bbox_h_px,
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "mpp_geom": mpp_geom,
    }


def moving_average_ignore_none(values, window=5):
    recent = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(recent) == 0:
        return None
    return float(np.mean(recent[-int(window):]))


def make_object_mask(frame_shape, boxes_xyxy, dilate_px=12, border_margin_px=10):
    h, w = frame_shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)

    if border_margin_px > 0:
        mask[:border_margin_px, :] = 0
        mask[h - border_margin_px:, :] = 0
        mask[:, :border_margin_px] = 0
        mask[:, w - border_margin_px:] = 0

    for box in boxes_xyxy:
        x1, y1, x2, y2 = map(int, np.round(box))
        x1 = max(0, x1 - dilate_px)
        y1 = max(0, y1 - dilate_px)
        x2 = min(w - 1, x2 + dilate_px)
        y2 = min(h - 1, y2 + dilate_px)
        mask[y1:y2 + 1, x1:x2 + 1] = 0

    return mask


def detect_background_features(gray_frame, allowed_mask):
    pts = cv2.goodFeaturesToTrack(
        gray_frame,
        maxCorners=FINAL_EGO_FEATURE_MAX_CORNERS,
        qualityLevel=FINAL_EGO_FEATURE_QUALITY,
        minDistance=FINAL_EGO_FEATURE_MIN_DISTANCE,
        blockSize=FINAL_EGO_FEATURE_BLOCK_SIZE,
        mask=allowed_mask,
    )
    if pts is None:
        return np.zeros((0, 1, 2), dtype=np.float32)
    return pts.astype(np.float32)


def robust_unit_direction_from_flows(flow_vectors, min_valid_tracks=25):
    flow_vectors = np.asarray(flow_vectors, dtype=np.float32)
    if flow_vectors.ndim != 2 or flow_vectors.shape[0] == 0:
        return {"ok": False}

    mags = np.linalg.norm(flow_vectors, axis=1)
    finite = np.isfinite(mags)
    flow_vectors = flow_vectors[finite]
    mags = mags[finite]

    if len(flow_vectors) < min_valid_tracks:
        return {"ok": False}

    q10, q90 = np.percentile(mags, [10, 90])
    keep = (mags >= q10) & (mags <= q90)
    trimmed = flow_vectors[keep]
    if len(trimmed) < min_valid_tracks:
        trimmed = flow_vectors

    mean_flow = np.median(trimmed, axis=0).astype(np.float32)
    mean_dx = float(mean_flow[0])
    mean_dy = float(mean_flow[1])
    mag = float(np.sqrt(mean_dx**2 + mean_dy**2))

    if not np.isfinite(mag) or mag <= 1e-8:
        return {"ok": False}

    return {
        "ok": True,
        "n_used": int(len(trimmed)),
        "mean_dx": mean_dx,
        "mean_dy": mean_dy,
        "unit_dx": mean_dx / mag,
        "unit_dy": mean_dy / mag,
        "flow_mag_px": mag,
    }


def moving_average_unit_vectors(dx_list, dy_list, window=5):
    recent = []
    for dx, dy in zip(dx_list, dy_list):
        if dx is None or dy is None:
            continue
        if not (np.isfinite(dx) and np.isfinite(dy)):
            continue
        recent.append([float(dx), float(dy)])

    if len(recent) == 0:
        return None, None

    recent = np.asarray(recent[-int(window):], dtype=np.float64)
    mean_vec = recent.mean(axis=0)
    mag = float(np.linalg.norm(mean_vec))
    if mag <= 1e-8 or not np.isfinite(mag):
        return None, None

    return float(mean_vec[0] / mag), float(mean_vec[1] / mag)


def make_empty_track_history():
    return {
        "frames": [],
        "labels": [],
        "image_points": [],
        "mpp_geom": [],
        "mpp_smoothed": [],
        "vx_rel_mps_raw": [],
        "vy_rel_mps_raw": [],
        "speed_rel_kmh_raw": [],
        "vx_rel_mps_smooth": [],
        "vy_rel_mps_smooth": [],
        "speed_rel_kmh_smooth": [],
    }


def append_relative_velocity_observation(track_history, frame_idx, box_xyxy, class_label, fps):
    rep_pt = representative_point_xyxy(box_xyxy)
    scale_info = compute_local_scale_from_box(box_xyxy, class_label)

    track_history["frames"].append(int(frame_idx))
    track_history["labels"].append(class_label)
    track_history["image_points"].append(rep_pt)
    track_history["mpp_geom"].append(scale_info["mpp_geom"])

    current_mpp_smoothed = moving_average_ignore_none(
        track_history["mpp_geom"], window=FINAL_MPP_WINDOW
    )
    track_history["mpp_smoothed"].append(current_mpp_smoothed)

    vx_rel_mps_raw = None
    vy_rel_mps_raw = None
    speed_rel_kmh_raw = None

    if len(track_history["image_points"]) >= 2:
        p1 = np.asarray(track_history["image_points"][-2], dtype=np.float64)
        p2 = np.asarray(track_history["image_points"][-1], dtype=np.float64)

        prev_mpp = track_history["mpp_smoothed"][-2]
        curr_mpp = track_history["mpp_smoothed"][-1]

        if prev_mpp is not None and curr_mpp is not None:
            local_mpp = 0.5 * (float(prev_mpp) + float(curr_mpp))
            dt = 1.0 / float(fps)

            dx_px = float(p2[0] - p1[0])
            dy_px = float(p2[1] - p1[1])

            vx_rel_mps_raw = (dx_px * local_mpp) / dt
            vy_rel_mps_raw = (dy_px * local_mpp) / dt
            speed_rel_mps_raw = float(np.sqrt(vx_rel_mps_raw**2 + vy_rel_mps_raw**2))
            speed_rel_kmh_raw = 3.6 * speed_rel_mps_raw

            if not (0.0 <= speed_rel_kmh_raw <= FINAL_MAX_REASONABLE_SPEED_KMH):
                vx_rel_mps_raw = None
                vy_rel_mps_raw = None
                speed_rel_kmh_raw = None

    track_history["vx_rel_mps_raw"].append(vx_rel_mps_raw)
    track_history["vy_rel_mps_raw"].append(vy_rel_mps_raw)
    track_history["speed_rel_kmh_raw"].append(speed_rel_kmh_raw)

    vx_rel_mps_smooth = moving_average_ignore_none(
        track_history["vx_rel_mps_raw"], window=FINAL_SPEED_WINDOW
    )
    vy_rel_mps_smooth = moving_average_ignore_none(
        track_history["vy_rel_mps_raw"], window=FINAL_SPEED_WINDOW
    )

    speed_rel_kmh_smooth = None
    if vx_rel_mps_smooth is not None and vy_rel_mps_smooth is not None:
        speed_rel_kmh_smooth = 3.6 * float(np.sqrt(vx_rel_mps_smooth**2 + vy_rel_mps_smooth**2))

    track_history["vx_rel_mps_smooth"].append(vx_rel_mps_smooth)
    track_history["vy_rel_mps_smooth"].append(vy_rel_mps_smooth)
    track_history["speed_rel_kmh_smooth"].append(speed_rel_kmh_smooth)

    return {
        "rep_pt": rep_pt,
        "scale_info": scale_info,
        "vx_rel_mps_smooth": vx_rel_mps_smooth,
        "vy_rel_mps_smooth": vy_rel_mps_smooth,
        "speed_rel_kmh_smooth": speed_rel_kmh_smooth,
    }


def make_speed_state():
    return {
        "prev_gray": None,
        "prev_background_pts": None,
        "smooth_unit_dx_hist": [],
        "smooth_unit_dy_hist": [],
        "ego_bg_dx_smooth": None,
        "ego_bg_dy_smooth": None,
        "track_histories": {},
    }


def reset_speed_state():
    return make_speed_state()


# ============================================================
# Display helpers
# ============================================================
def init_display(width: int, height: int):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("RF-DETR + ByteTrack + Speed Estimation")
    return screen


def show_frame(screen, frame_rgb: np.ndarray):
    surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
    screen.blit(surf, (0, 0))
    pygame.display.flip()


def handle_events():
    """
    Returns:
        "quit"   -> exit app
        "toggle" -> switch camera + model
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


def update_window_title(camera_id: int, camera_speed_kmh: float):
    cam_label = CAMERA_CONFIGS[camera_id]["label"]
    ckpt_name = CAMERA_CONFIGS[camera_id]["ckpt"].name
    pygame.display.set_caption(
        f"RF-DETR + ByteTrack + Speed | {cam_label} | {ckpt_name} | camera speed={camera_speed_kmh:.2f} km/h"
    )


def draw_arrow(draw, origin, end, color=(255, 255, 0), width=4, head_len=12):
    x1, y1 = origin
    x2, y2 = end
    draw.line([x1, y1, x2, y2], fill=color, width=width)

    angle = math.atan2(y2 - y1, x2 - x1)
    left = (
        x2 - head_len * math.cos(angle - math.pi / 6),
        y2 - head_len * math.sin(angle - math.pi / 6),
    )
    right = (
        x2 - head_len * math.cos(angle + math.pi / 6),
        y2 - head_len * math.sin(angle + math.pi / 6),
    )
    draw.polygon([(x2, y2), left, right], fill=color)


def draw_camera_motion_indicator(draw, cam_unit_dx, cam_unit_dy, camera_speed_kmh, font):
    origin = np.array([70, 70], dtype=np.int32)
    arrow_len = 50

    if abs(float(camera_speed_kmh)) < 1e-6:
        draw.ellipse(
            [origin[0] - 8, origin[1] - 8, origin[0] + 8, origin[1] + 8],
            fill=(255, 255, 0),
            outline=(255, 255, 0),
        )
        draw.text((20, 20), "camera still", fill=(255, 255, 0), font=font)
        return

    if cam_unit_dx is not None and cam_unit_dy is not None:
        end = origin + np.round(
            arrow_len * np.array([cam_unit_dx, cam_unit_dy], dtype=np.float32)
        ).astype(np.int32)
        draw_arrow(draw, tuple(origin), tuple(end), color=(255, 255, 0), width=4, head_len=12)
        draw.text((20, 20), "camera ego", fill=(255, 255, 0), font=font)
        return

    draw.ellipse(
        [origin[0] - 8, origin[1] - 8, origin[0] + 8, origin[1] + 8],
        outline=(255, 255, 0),
        width=3,
    )
    draw.text((20, 20), "camera ego unavailable", fill=(255, 255, 0), font=font)


# ============================================================
# Prompt
# ============================================================
def prompt_camera_speed_kmh():
    while True:
        try:
            raw = input("Enter constant camera speed in km/h (example: 5): ").strip()
            value = float(raw)
            if value < 0:
                print("Please enter a non-negative speed.")
                continue
            return value
        except ValueError:
            print("Invalid number. Try again.")


# ============================================================
# Global runtime objects
# ============================================================
current_camera = 0
model = load_model(CAMERA_CONFIGS[current_camera]["ckpt"])
tracker = make_tracker(frame_rate=FPS)

track_meta = {}       # track_id -> (class_id, confidence)
track_last_seen = {}  # track_id -> frame_idx
speed_state = reset_speed_state()


# ============================================================
# Main loop
# ============================================================
def run_live():
    global current_camera, model, tracker, track_meta, track_last_seen, speed_state

    camera_speed_kmh = prompt_camera_speed_kmh()

    cam = open_camera(current_camera, W, H, FPS)
    screen = init_display(W, H)
    update_window_title(current_camera, camera_speed_kmh)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    print("Camera opened. Press 'c' to switch cameras/models, 'q' to quit.")
    print(f"Current camera: {CAMERA_CONFIGS[current_camera]['label']}")
    print(f"Current checkpoint: {CAMERA_CONFIGS[current_camera]['ckpt']}")
    print(f"Assumed constant camera speed: {camera_speed_kmh:.2f} km/h")

    frame_idx = 0
    t0 = time.time()

    while True:
        action = handle_events()
        if action == "quit":
            break

        elif action == "toggle":
            try:
                cam.release()
            except Exception:
                pass

            # Toggle between RGB USB camera and IR CSI camera
            current_camera = 1 if current_camera == 0 else 0

            cam = open_camera(current_camera, W, H, FPS)
            model = load_model(CAMERA_CONFIGS[current_camera]["ckpt"])

            tracker = make_tracker(frame_rate=FPS)
            track_meta.clear()
            track_last_seen.clear()
            speed_state = reset_speed_state()

            update_window_title(current_camera, camera_speed_kmh)

            print(f"Switched to camera: {CAMERA_CONFIGS[current_camera]['label']}")
            print(f"Switched to checkpoint: {CAMERA_CONFIGS[current_camera]['ckpt']}")

            # Small debounce / allow pipeline to settle
            time.sleep(0.2)
            continue

        ret, frame_rgb = cam.read()
        if not ret:
            continue

        # ----------------------------------------------------
        # 1) Detect + track
        # ----------------------------------------------------
        if frame_idx % DETECT_EVERY_N == 0:
            # Save temp frame for RFDETR.predict() (expects a path)
            Image.fromarray(frame_rgb).save(TEMP_FRAME_PATH, quality=90)

            det = model.predict(str(TEMP_FRAME_PATH), threshold=THRESHOLD)
            detections = rfdetr_to_sv_detections(det, min_conf=THRESHOLD)
            detections = filter_small_boxes(detections, MIN_BOX_AREA)

            det_with_ids = tracker.update_with_detections(detections)

            if len(det_with_ids) > 0 and getattr(det_with_ids, "tracker_id", None) is not None:
                for i in range(len(det_with_ids)):
                    tid = int(det_with_ids.tracker_id[i])
                    cls = int(det_with_ids.class_id[i])
                    conf = float(det_with_ids.confidence[i])

                    track_meta[tid] = (cls, conf)
                    track_last_seen[tid] = frame_idx

        # ----------------------------------------------------
        # 2) Live ego-direction estimation from background flow
        # ----------------------------------------------------
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        active_boxes, active_ids = active_tracks_to_boxes_ids(tracker)

        bg_mask = make_object_mask(
            frame_shape=frame_rgb.shape,
            boxes_xyxy=active_boxes,
            dilate_px=FINAL_EGO_MASK_DILATE_PX,
            border_margin_px=FINAL_EGO_BORDER_MARGIN_PX,
        )

        if speed_state["prev_gray"] is not None:
            need_new_points = (
                speed_state["prev_background_pts"] is None or
                len(speed_state["prev_background_pts"]) < FINAL_EGO_MIN_VALID_TRACKS
            )

            if need_new_points:
                prev_bg_mask = make_object_mask(
                    frame_shape=frame_rgb.shape,
                    boxes_xyxy=active_boxes,
                    dilate_px=FINAL_EGO_MASK_DILATE_PX,
                    border_margin_px=FINAL_EGO_BORDER_MARGIN_PX,
                )
                speed_state["prev_background_pts"] = detect_background_features(
                    speed_state["prev_gray"], prev_bg_mask
                )

            if (
                speed_state["prev_background_pts"] is not None and
                len(speed_state["prev_background_pts"]) > 0
            ):
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    speed_state["prev_gray"],
                    frame_gray,
                    speed_state["prev_background_pts"],
                    None,
                    winSize=FINAL_EGO_LK_WIN_SIZE,
                    maxLevel=FINAL_EGO_LK_MAX_LEVEL,
                    criteria=FINAL_EGO_LK_CRITERIA,
                )

                valid_next = None

                if next_pts is not None and status is not None:
                    good_prev = speed_state["prev_background_pts"][status.flatten() == 1].reshape(-1, 2)
                    good_next = next_pts[status.flatten() == 1].reshape(-1, 2)

                    filtered_prev = []
                    filtered_next = []

                    for p0, p1 in zip(good_prev, good_next):
                        x1i = int(round(p1[0]))
                        y1i = int(round(p1[1]))
                        if 0 <= x1i < W and 0 <= y1i < H and bg_mask[y1i, x1i] > 0:
                            filtered_prev.append(p0)
                            filtered_next.append(p1)

                    if len(filtered_prev) > 0:
                        filtered_prev = np.asarray(filtered_prev, dtype=np.float32)
                        filtered_next = np.asarray(filtered_next, dtype=np.float32)

                        flows = filtered_next - filtered_prev

                        robust = robust_unit_direction_from_flows(
                            flows,
                            min_valid_tracks=FINAL_EGO_MIN_VALID_TRACKS,
                        )

                        if robust.get("ok", False):
                            speed_state["smooth_unit_dx_hist"].append(robust["unit_dx"])
                            speed_state["smooth_unit_dy_hist"].append(robust["unit_dy"])

                            unit_dx_smooth, unit_dy_smooth = moving_average_unit_vectors(
                                speed_state["smooth_unit_dx_hist"],
                                speed_state["smooth_unit_dy_hist"],
                                window=FINAL_EGO_DIRECTION_SMOOTH_WINDOW,
                            )

                            speed_state["ego_bg_dx_smooth"] = unit_dx_smooth
                            speed_state["ego_bg_dy_smooth"] = unit_dy_smooth

                        valid_next = filtered_next.reshape(-1, 1, 2)
                    else:
                        valid_next = None

                speed_state["prev_background_pts"] = valid_next

        speed_state["prev_gray"] = frame_gray

        # background flow direction -> camera motion direction
        cam_unit_dx = None
        cam_unit_dy = None
        alpha_eff = None
        vx_cam_mps = None
        vy_cam_mps = None

        ego_bg_dx = speed_state["ego_bg_dx_smooth"]
        ego_bg_dy = speed_state["ego_bg_dy_smooth"]

        if (
            ego_bg_dx is not None and ego_bg_dy is not None and
            np.isfinite(ego_bg_dx) and np.isfinite(ego_bg_dy)
        ):
            cam_unit_dx = -float(ego_bg_dx)
            cam_unit_dy = -float(ego_bg_dy)

            alpha_eff = (
                float(ALPHA_LEFT_RIGHT) * abs(cam_unit_dx) +
                float(ALPHA_UP_DOWN) * abs(cam_unit_dy)
            )

            camera_speed_mps = float(camera_speed_kmh) / 3.6
            vx_cam_mps = alpha_eff * camera_speed_mps * cam_unit_dx
            vy_cam_mps = alpha_eff * camera_speed_mps * cam_unit_dy

        # ----------------------------------------------------
        # 3) Draw + live world-speed estimation
        # ----------------------------------------------------
        img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img)

        boxes, ids = active_tracks_to_boxes_ids(tracker)

        for box, tid in zip(boxes, ids):
            # Only draw if recently seen (prevents blinking)
            last = track_last_seen.get(tid, -10**9)
            if frame_idx - last > DISPLAY_GRACE:
                continue

            if bbox_touches_border(box, W, H, margin_px=FINAL_BORDER_MARGIN_PX):
                continue

            cls, conf = track_meta.get(tid, (-1, 0.0))
            label = PHASE2_CLASSES.get(cls, f"class_{cls}") if cls != -1 else "unknown"

            if tid not in speed_state["track_histories"]:
                speed_state["track_histories"][tid] = make_empty_track_history()

            result = append_relative_velocity_observation(
                track_history=speed_state["track_histories"][tid],
                frame_idx=frame_idx,
                box_xyxy=box,
                class_label=label,
                fps=FPS,
            )

            vx_rel = result["vx_rel_mps_smooth"]
            vy_rel = result["vy_rel_mps_smooth"]
            speed_rel = result["speed_rel_kmh_smooth"]

            speed_world = None
            if (
                vx_rel is not None and vy_rel is not None and
                vx_cam_mps is not None and vy_cam_mps is not None
            ):
                vx_world = float(vx_rel) + float(vx_cam_mps)
                vy_world = float(vy_rel) + float(vy_cam_mps)
                speed_world = 3.6 * float(np.sqrt(vx_world**2 + vy_world**2))
            else:
                speed_world = speed_rel

            x1, y1, x2, y2 = box.astype(int).tolist()
            cam_name = CAMERA_CONFIGS[current_camera]["short_label"]

            text = f"[{cam_name}] ID {tid}: {label}"
            if speed_world is not None and np.isfinite(speed_world):
                text += f" | {speed_world:.2f} km/h"

            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

            tb = draw.textbbox((0, 0), text, font=font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            tx, ty = x1, max(0, y1 - th - 4)

            draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=(0, 255, 0))
            draw.text((tx + 3, ty + 2), text, fill=(0, 0, 0), font=font)

            if speed_world is not None and np.isfinite(speed_world):
                rep = result["rep_pt"]
                rx, ry = int(round(rep[0])), int(round(rep[1]))
                draw.ellipse([rx - 2, ry - 2, rx + 2, ry + 2], fill=(255, 0, 0))

        # Draw camera ego-motion indicator
        draw_camera_motion_indicator(draw, cam_unit_dx, cam_unit_dy, camera_speed_kmh, font)

        # Extra status text
        status_lines = [
            f"Mode: {CAMERA_CONFIGS[current_camera]['label']}",
            f"Model: {CAMERA_CONFIGS[current_camera]['ckpt'].name}",
            f"Camera speed input: {camera_speed_kmh:.2f} km/h",
        ]
        y_text = H - 60
        for line in status_lines:
            draw.text((10, y_text), line, fill=(255, 255, 255), font=font_small)
            y_text += 16

        out_rgb = np.array(img, dtype=np.uint8)

        # FPS print
        if frame_idx % FPS == 0 and frame_idx > 0:
            elapsed = time.time() - t0
            fps_eff = frame_idx / max(elapsed, 1e-6)
            print(
                f"Live FPS: ~{fps_eff:.2f} | detect every {DETECT_EVERY_N} | "
                f"camera: {CAMERA_CONFIGS[current_camera]['label']} | "
                f"model: {CAMERA_CONFIGS[current_camera]['ckpt'].name} | "
                f"camera speed input: {camera_speed_kmh:.2f} km/h"
            )

        show_frame(screen, out_rgb)
        frame_idx += 1

    try:
        cam.release()
    except Exception:
        pass
    pygame.quit()

    if TEMP_FRAME_PATH.exists():
        try:
            TEMP_FRAME_PATH.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    run_live()