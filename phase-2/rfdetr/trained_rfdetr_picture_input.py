from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import torch
from rfdetr import RFDETRNano

# ------------------------------------------------------------
# Resolve paths relative to THIS script location (portable)
# ------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # .../phase-2/rfdetr
PHASE2_DIR = SCRIPT_DIR.parent                        # .../phase-2
REPO_DIR   = PHASE2_DIR.parent                        # .../object-identification-uav-camera

# ---- CONFIG ----
CKPT_PATH = SCRIPT_DIR / "models" / "visdrone_rfdetr_nano_best_ema.pth"
IMAGE_PATH = PHASE2_DIR / "visdrone-dataset" / "test-dev" / "images" / "0000006_03636_d_0000009.jpg"    # random image for input
THRESHOLD = 0.30

# Phase-2 class names (must match training)
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
assert IMAGE_PATH.exists(), f"Missing image: {IMAGE_PATH}"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("Checkpoint:", CKPT_PATH)
print("Image:", IMAGE_PATH)

# ---- Load model (RF-DETR supported path) ----
model = RFDETRNano(pretrain_weights=str(CKPT_PATH))
model.model.model.to(device)
model.model.model.eval()

# ---- Predict ----
det = model.predict(str(IMAGE_PATH), threshold=THRESHOLD)

# ---- Visualize ----
img = cv2.imread(str(IMAGE_PATH))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for box, score, cls in zip(det.xyxy, det.confidence, det.class_id):
    x1, y1, x2, y2 = map(int, box.tolist())
    cls = int(cls)
    label = PHASE2_CLASSES.get(cls, f"class_{cls}")
    text = f"{label} {float(score):.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, text, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis("off")
plt.title("RF-DETR predictions")
plt.show()

print(f"Detections: {len(det.xyxy)} (threshold={THRESHOLD})")