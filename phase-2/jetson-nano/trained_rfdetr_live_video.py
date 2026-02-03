from pathlib import Path
import time
import cv2
import numpy as np
import torch
from rfdetr import RFDETRNano
import supervision as sv

# -----------------------------
# 1) Paths + config
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent  # phase-2/jetson-nano
CKPT_PATH = SCRIPT_DIR / "visdrone_rfdetr_nano_best_ema.pth"

assert CKPT_PATH.exists(), f"Missing checkpoint: {CKPT_PATH}"

# Detection/tracking knobs (tune on-device)
THRESHOLD = 0.30
DETECT_EVERY_N = 5          # Run detector every N frames (try 5–10 on Orin Nano)
MIN_BOX_AREA = 24 * 24      # Filter tiny detections for stability

# Your Phase-2 class mapping (match training)
PHASE2_CLASSES = {
    1: "Human",
    2: "Bicycle",
    3: "Car",
    4: "Truck",
    5: "Bus",
    6: "Motorcycle",
}

# Temp path for RF-DETR predict() (expects an image path)
TEMP_FRAME_PATH = SCRIPT_DIR / "_temp_frame.jpg"

# -----------------------------
# 2) Camera pipeline
# -----------------------------
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# -----------------------------
# 3) Init model
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

# -----------------------------
# 4) Init ByteTrack (supervision 0.27.0 API)
# -----------------------------
# These names match supervision 0.27.0
tracker = sv.ByteTrack(
    track_activation_threshold=THRESHOLD,
    lost_track_buffer=30,
    minimum_matching_threshold=0.7,
    frame_rate=30
)

# Metadata memory: track_id -> (class_id, confidence)
track_meta = {}

def rfdetr_to_sv_detections(det, min_conf=0.30):
    xyxy = np.asarray(det.xyxy, dtype=np.float32)
    conf = np.asarray(det.confidence, dtype=np.float32)
    cls  = np.asarray(det.class_id, dtype=np.int32)

    keep = conf >= float(min_conf)
    xyxy = xyxy[keep]
    conf = conf[keep]
    cls  = cls[keep]

    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)

def filter_small_boxes(dets: sv.Detections, min_area: int):
    if len(dets) == 0:
        return dets
    xyxy = dets.xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    keep = areas >= float(min_area)
    return dets[keep]

def build_tensors_from_detections(dets: sv.Detections) -> np.ndarray:
    # Nx5 [x1,y1,x2,y2,score]
    if len(dets) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.hstack([dets.xyxy.astype(np.float32),
                      dets.confidence.reshape(-1, 1).astype(np.float32)])

def active_tracks_to_boxes_ids(tracker_obj):
    tracks = getattr(tracker_obj, "tracked_tracks", [])
    boxes, ids = [], []
    for t in tracks:
        boxes.append(np.array(t.tlbr, dtype=np.float32))
        ids.append(int(t.external_track_id))
    return boxes, ids

# -----------------------------
# 5) Main loop
# -----------------------------
def run_live():
    pipeline = gstreamer_pipeline(flip_method=0)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise RuntimeError("Could not open CSI camera. Check connections / sensor_id / permissions.")

    print("Camera opened successfully. Press 'q' to exit.")
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Update tracker every frame
        if frame_idx % DETECT_EVERY_N == 0:
            # Write temp frame for predict()
            cv2.imwrite(str(TEMP_FRAME_PATH), frame_bgr)

            det = model.predict(str(TEMP_FRAME_PATH), threshold=THRESHOLD)
            detections = rfdetr_to_sv_detections(det, min_conf=THRESHOLD)
            detections = filter_small_boxes(detections, MIN_BOX_AREA)

            # Advance tracker using tensors
            _ = tracker.update_with_tensors(build_tensors_from_detections(detections))

            # Get tracker_id mapping on detection frames
            det_with_ids = tracker.update_with_detections(detections)
            if len(det_with_ids) > 0 and getattr(det_with_ids, "tracker_id", None) is not None:
                for i in range(len(det_with_ids)):
                    tid = int(det_with_ids.tracker_id[i])
                    cls = int(det_with_ids.class_id[i])
                    conf = float(det_with_ids.confidence[i])
                    track_meta[tid] = (cls, conf)
        else:
            # No detections this frame → predict step
            _ = tracker.update_with_tensors(np.zeros((0, 5), dtype=np.float32))

        # Draw active tracks
        out = frame_bgr.copy()
        boxes, ids = active_tracks_to_boxes_ids(tracker)

        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = box.astype(int).tolist()
            cls, conf = track_meta.get(tid, (-1, 0.0))
            label = PHASE2_CLASSES.get(cls, f"class_{cls}") if cls != -1 else "unknown"
            text = f"ID {tid}: {label} {conf:.2f}"

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, text, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # FPS overlay
        if frame_idx % 30 == 0 and frame_idx > 0:
            elapsed = time.time() - t0
            fps_eff = frame_idx / max(elapsed, 1e-6)
            print(f"Live FPS: ~{fps_eff:.2f} | detect every {DETECT_EVERY_N} frames")

        cv2.imshow("RF-DETR + ByteTrack (Jetson)", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live()