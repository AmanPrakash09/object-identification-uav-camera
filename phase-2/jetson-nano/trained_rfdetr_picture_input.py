from pathlib import Path
import os
import cv2
import torch
from rfdetr import RFDETRNano

SCRIPT_DIR = Path(__file__).resolve().parent          # .../phase-2/jetson-nano

print(SCRIPT_DIR)

# ---- CONFIG ----
CKPT_PATH = SCRIPT_DIR / "visdrone_rfdetr_nano_best_ema.pth"
IMAGE_PATH = SCRIPT_DIR / "example-images" / "0000006_00159_d_0000001.jpg"
OUTPUT_PATH = SCRIPT_DIR / "example-outputs" / "annotated.jpg"

THRESHOLD = 0.30

# Phase-2 class names (must match training)
PHASE2_CLASSES = {
    1: "Human",
    2: "Bicycle",
    3: "Car",
    4: "Truck",
    5: "Bus",
    6: "Other",   # keep consistent with your trained mapping
}

# ---- Checks ----
assert CKPT_PATH.exists(), f"Missing checkpoint: {CKPT_PATH}"
assert IMAGE_PATH.exists(), f"Missing image: {IMAGE_PATH}"

# ---- Device info ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Device:", device)
print("Checkpoint:", CKPT_PATH)
print("Image:", IMAGE_PATH)

# ---- Load model ----
model = RFDETRNano(pretrain_weights=str(CKPT_PATH))

# Move underlying torch module to device
model.model.model.to(device)
model.model.model.eval()

# ---- Predict ----
det = model.predict(str(IMAGE_PATH), threshold=THRESHOLD)

print(f"Detections: {len(det.xyxy)} (threshold={THRESHOLD})")

# ---- Draw boxes ----
img_bgr = cv2.imread(str(IMAGE_PATH))
if img_bgr is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

for box, score, cls in zip(det.xyxy, det.confidence, det.class_id):
    x1, y1, x2, y2 = map(int, box.tolist())
    cls = int(cls)
    label = PHASE2_CLASSES.get(cls, f"class_{cls}")
    text = f"{label} {float(score):.2f}"

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img_bgr,
        text,
        (x1, max(0, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

# ---- Save output ----
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT_PATH), img_bgr)
print("Saved annotated image to:", OUTPUT_PATH)