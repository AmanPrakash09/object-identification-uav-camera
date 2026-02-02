from pathlib import Path
import time
import cv2
import torch
from rfdetr import RFDETRNano

# ------------------------------------------------------------
# Resolve paths relative to THIS script location (portable)
# ------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # .../phase-2/rfdetr
PHASE2_DIR = SCRIPT_DIR.parent                        # .../phase-2

# ---- CONFIG ----
CKPT_PATH = SCRIPT_DIR / "models" / "visdrone_rfdetr_nano_best_ema.pth"

VIDEO_DIR = SCRIPT_DIR / "videos" / "personal_simulation"
INPUT_VIDEO = VIDEO_DIR / "input.mp4"
OUTPUT_VIDEO = VIDEO_DIR / "output_annotated.mp4"

THRESHOLD = 0.30
DETECT_EVERY_N = 1   # 1 = detect every frame. Set to 5 or 10 to speed up.

# Phase-2 class names (match your training mapping)
PHASE2_CLASSES = {
    1: "Human",
    2: "Bicycle",
    3: "Car",
    4: "Truck",
    5: "Bus",
    6: "Other",
}

# ---- Checks ----
assert CKPT_PATH.exists(), f"Missing checkpoint: {CKPT_PATH}"
assert VIDEO_DIR.exists(), f"Missing video folder: {VIDEO_DIR}"
assert INPUT_VIDEO.exists(), f"Missing input video: {INPUT_VIDEO}"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Device:", device)
print("Checkpoint:", CKPT_PATH)
print("Input video:", INPUT_VIDEO)
print("Output video:", OUTPUT_VIDEO)
print("Threshold:", THRESHOLD, "| Detect every N:", DETECT_EVERY_N)

# ---- Load model ----
model = RFDETRNano(pretrain_weights=str(CKPT_PATH))
model.model.model.to(device)
model.model.model.eval()

# ---- Open video ----
cap = cv2.VideoCapture(str(INPUT_VIDEO))
assert cap.isOpened(), f"Could not open video: {INPUT_VIDEO}"

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

# ---- Writer ----
OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))
assert writer.isOpened(), f"Could not open writer: {OUTPUT_VIDEO}"

print(f"Video properties: {width}x{height} @ {fps:.2f} FPS | frames={total_frames}")

# ---- Process frames ----
frame_idx = 0
t0 = time.time()

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    annotated = frame_bgr

    # Detect every N frames (set N=1 for every frame)
    if frame_idx % DETECT_EVERY_N == 0:
        # RF-DETR predict() expects a file path
        temp_path = str(VIDEO_DIR / "_temp_frame.jpg")
        cv2.imwrite(temp_path, frame_bgr)

        det = model.predict(temp_path, threshold=THRESHOLD)

        # Draw detections
        for box, score, cls in zip(det.xyxy, det.confidence, det.class_id):
            x1, y1, x2, y2 = map(int, box.tolist())
            cls = int(cls)
            label = PHASE2_CLASSES.get(cls, f"class_{cls}")
            text = f"{label} {float(score):.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, text, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )

    writer.write(annotated)
    frame_idx += 1

    # periodic progress print
    if frame_idx % 60 == 0:
        elapsed = time.time() - t0
        fps_eff = frame_idx / max(elapsed, 1e-6)
        print(f"Processed {frame_idx} frames | ~{fps_eff:.2f} FPS")

cap.release()
writer.release()

elapsed = time.time() - t0
fps_eff = frame_idx / max(elapsed, 1e-6)
print(f"\nDone. Wrote: {OUTPUT_VIDEO}")
print(f"Frames processed: {frame_idx} | Avg speed: {fps_eff:.2f} FPS")