from pathlib import Path
import torch
from rfdetr import RFDETRNano

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent

# ---- CONFIG ----
CKPT_PATH = SCRIPT_DIR / "visdrone_rfdetr_nano_best_ema.pth"
IMAGE_PATH = SCRIPT_DIR / "example-images" / "0000006_00159_d_0000001.jpg"
OUTPUT_PATH = SCRIPT_DIR / "example-outputs" / "annotated.jpg"
THRESHOLD = 0.30

PHASE2_CLASSES = {
    1: "Human",
    2: "Bicycle",
    3: "Car",
    4: "Truck",
    5: "Bus",
    6: "Other",
}

assert CKPT_PATH.exists(), f"Missing checkpoint: {CKPT_PATH}"
assert IMAGE_PATH.exists(), f"Missing image: {IMAGE_PATH}"

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
model.model.model.to(device)
model.model.model.eval()

# ---- Predict ----
det = model.predict(str(IMAGE_PATH), threshold=THRESHOLD)
print(f"Detections: {len(det.xyxy)} (threshold={THRESHOLD})")

# ---- Draw boxes using PIL ----
img = Image.open(IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(img)

# Optional: nicer font if available, else default
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
except Exception:
    font = ImageFont.load_default()

for box, score, cls in zip(det.xyxy, det.confidence, det.class_id):
    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
    cls = int(cls)
    label = PHASE2_CLASSES.get(cls, f"class_{cls}")
    text = f"{label} {float(score):.2f}"

    # rectangle
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

    # text background for readability
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    tx, ty = x1, max(0, y1 - th - 4)
    draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=(0, 255, 0))
    draw.text((tx + 3, ty + 2), text, fill=(0, 0, 0), font=font)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
img.save(OUTPUT_PATH, quality=95)
print("Saved annotated image to:", OUTPUT_PATH)