#!/usr/bin/env python
# coding: utf-8



# ============================================================
# Cell 1: Imports, Environment Sanity Check, GCS Paths, Logging
# ============================================================

# ---- Standard library ----
import os
import sys
import json
import time
import random
import shutil
from pathlib import Path
from datetime import datetime
import re
import csv
import threading
import traceback
from types import SimpleNamespace

# ---- Numerical / scientific ----
import numpy as np

# ---- Image processing ----
import cv2

# ---- Deep learning ----
import torch
import torchvision

# ---- RF-DETR ----
from rfdetr import RFDETRNano
from rfdetr.datasets.coco import build as build_coco

# ---- Dataset / evaluation ----
from pycocotools.coco import COCO

# ---- Visualization ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Utilities ----
import psutil

# ---- Google Cloud ----
from google.cloud import storage

# ============================================================
# Reproducibility helper
# ============================================================

SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# ============================================================
# Project / GCS configuration
# ============================================================

NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR

# ---- Local staging dirs ----
LOCAL_DATA_ROOT = PROJECT_ROOT / "infrared-data"
LOCAL_MODELS_ROOT = PROJECT_ROOT / "models"
LOCAL_LOG_ROOT = PROJECT_ROOT / "logs"
LOCAL_META_ROOT = PROJECT_ROOT / "run_metadata"
LOCAL_PLOTS_ROOT = LOCAL_LOG_ROOT / "plots"

for d in [LOCAL_DATA_ROOT, LOCAL_MODELS_ROOT, LOCAL_LOG_ROOT, LOCAL_META_ROOT, LOCAL_PLOTS_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

# ---- GCS buckets / prefixes ----
GCS_DATA_BUCKET = "gs://infrared-data"
GCS_LOG_BUCKET = "gs://rfdetr-training-logs/infrared-training"
GCS_MODEL_BUCKET = "gs://rfdetr-model-versions/enhanced-training"

# ---- Local paths expected by RF-DETR COCO loader ----
SHIM_ROOT = LOCAL_DATA_ROOT
COCO_ANN_DIR = SHIM_ROOT / "annotations"
COCO_TRAIN_IMG = SHIM_ROOT / "train2017"
COCO_VAL_IMG = SHIM_ROOT / "val2017"
COCO_TEST_IMG = SHIM_ROOT / "test2017"

COCO_TRAIN_ANN = COCO_ANN_DIR / "instances_train2017.json"
COCO_VAL_ANN = COCO_ANN_DIR / "instances_val2017.json"
COCO_TEST_ANN = COCO_ANN_DIR / "instances_test2017.json"

# ---- Enhanced-training checkpoints in GCS ----
GCS_SOURCE_BEST_EMA = f"{GCS_MODEL_BUCKET}/visdrone_rfdetr_nano_best_ema.pth"
GCS_SOURCE_BEST_TOTAL = f"{GCS_MODEL_BUCKET}/visdrone_rfdetr_nano_best_total.pth"
GCS_SOURCE_LATEST = f"{GCS_MODEL_BUCKET}/visdrone_rfdetr_nano_latest_resume.pth"

LOCAL_ENHANCED_DIR = LOCAL_MODELS_ROOT / "enhanced-training"
LOCAL_ENHANCED_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_BEST_EMA_CKPT = LOCAL_ENHANCED_DIR / "visdrone_rfdetr_nano_best_ema.pth"
SOURCE_BEST_TOTAL_CKPT = LOCAL_ENHANCED_DIR / "visdrone_rfdetr_nano_best_total.pth"
SOURCE_LATEST_CKPT = LOCAL_ENHANCED_DIR / "visdrone_rfdetr_nano_latest_resume.pth"

# ---- Notebook session log ----
SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
NOTEBOOK_LOG_PATH = LOCAL_LOG_ROOT / f"notebook_session_{SESSION_TS}.log"

def log_print(*args, sep=" ", end="\n", flush=True):
    text = sep.join(str(a) for a in args) + end
    print(text, end="", flush=flush)
    with open(NOTEBOOK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text)

# ------------------------------------------------------------
# GCS helpers using google-cloud-storage only
# ------------------------------------------------------------
_gcs_client = storage.Client()

def parse_gs_uri(gs_uri: str):
    assert gs_uri.startswith("gs://"), f"Invalid GCS URI: {gs_uri}"
    path = gs_uri[len("gs://"):]
    bucket_name, _, blob_name = path.partition("/")
    return bucket_name, blob_name

def gcs_upload_file(local_path: Path, gcs_uri: str):
    local_path = Path(local_path)
    assert local_path.exists(), f"Local file does not exist: {local_path}"

    bucket_name, blob_name = parse_gs_uri(gcs_uri)
    bucket = _gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))

def gcs_upload_dir(local_dir: Path, gcs_uri: str):
    local_dir = Path(local_dir)
    assert local_dir.exists(), f"Local dir does not exist: {local_dir}"

    bucket_name, prefix = parse_gs_uri(gcs_uri)
    bucket = _gcs_client.bucket(bucket_name)

    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            blob_name = f"{prefix.rstrip('/')}/{rel}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(p))

def gcs_download_file(gcs_uri: str, local_path: Path):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    bucket_name, blob_name = parse_gs_uri(gcs_uri)
    bucket = _gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    assert blob.exists(), f"GCS object not found: {gcs_uri}"
    blob.download_to_filename(str(local_path))

def sync_gcs_prefix_to_local(gcs_uri: str, local_dir: Path):
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    bucket_name, prefix = parse_gs_uri(gcs_uri)
    bucket = _gcs_client.bucket(bucket_name)

    normalized_prefix = prefix.rstrip("/")
    if normalized_prefix:
        blobs = list(_gcs_client.list_blobs(bucket, prefix=normalized_prefix + "/"))
        prefix_base = normalized_prefix + "/"
    else:
        blobs = list(_gcs_client.list_blobs(bucket))
        prefix_base = ""

    assert blobs, f"No objects found under prefix: {gcs_uri}"

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        rel_path = blob.name[len(prefix_base):] if prefix_base else blob.name
        dst = local_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dst))

def save_current_figure(name: str):
    """
    Save current matplotlib figure locally and upload to GCS logs bucket.
    """
    safe_name = name.replace(" ", "_")
    out_path = LOCAL_PLOTS_ROOT / f"{safe_name}_{SESSION_TS}.png"
    plt.savefig(out_path, bbox_inches="tight")
    gcs_upload_file(out_path, f"{GCS_LOG_BUCKET}/plots/{out_path.name}")
    log_print(f"[PLOT SAVED] {out_path}")
    return out_path

def gcs_upload_bytes(data: bytes, gs_uri: str, content_type="application/octet-stream"):
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    bucket = _gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)

def gcs_upload_text(text: str, gs_uri: str):
    gcs_upload_bytes(text.encode("utf-8"), gs_uri, content_type="text/plain")

# ============================================================
# Environment sanity prints
# ============================================================

log_print("=== Environment Check ===")
log_print(f"Python version      : {sys.version.split()[0]}")
log_print(f"PyTorch version     : {torch.__version__}")
log_print(f"Torchvision version : {torchvision.__version__}")
log_print(f"CUDA available      : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    log_print(f"CUDA version        : {torch.version.cuda}")
    log_print(f"GPU                 : {torch.cuda.get_device_name(0)}")

log_print(f"CPU cores           : {psutil.cpu_count(logical=True)}")
log_print(f"Random seed         : {SEED}")
log_print(f"Notebook dir        : {NOTEBOOK_DIR}")
log_print(f"Local data root     : {LOCAL_DATA_ROOT}")
log_print(f"GCS data bucket     : {GCS_DATA_BUCKET}")
log_print(f"GCS log bucket      : {GCS_LOG_BUCKET}")
log_print(f"GCS model bucket    : {GCS_MODEL_BUCKET}")
log_print("=========================")

# Upload the notebook session log snapshot right away
gcs_upload_file(NOTEBOOK_LOG_PATH, f"{GCS_LOG_BUCKET}/{NOTEBOOK_LOG_PATH.name}")



# ============================================================
# Cell 2: Canonical Class Mapping (DroneVehicle -> Phase 3 Classes)
# SINGLE SOURCE OF TRUTH
# ============================================================

PHASE3_CLASSES = {
    1: "Human",
    2: "Bicycle",
    3: "Car",
    4: "Truck",
    5: "Bus",
    6: "Other",
}

NUM_CLASSES = len(PHASE3_CLASSES)
PHASE3_CLASS_NAMES = [PHASE3_CLASSES[i] for i in range(1, NUM_CLASSES + 1)]
PHASE3_CLASS_NAME_TO_ID = {name: idx for idx, name in PHASE3_CLASSES.items()}

DRONEVEHICLE_CLASSES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "van",
    4: "freight_car",
}

DRONEVEHICLE_NAME_NORMALIZATION = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "van": "van",
    "freight car": "freight_car",
    "freight_car": "freight_car",
    "freightcar": "freight_car",
}

DRONEVEHICLE_TO_PHASE3 = {
    "car": PHASE3_CLASS_NAME_TO_ID["Car"],
    "truck": PHASE3_CLASS_NAME_TO_ID["Truck"],
    "bus": PHASE3_CLASS_NAME_TO_ID["Bus"],
    "van": PHASE3_CLASS_NAME_TO_ID["Car"],
    "freight_car": PHASE3_CLASS_NAME_TO_ID["Other"],
}

COCO_CATEGORIES = [
    {
        "id": class_id,
        "name": class_name,
        "supercategory": "object",
    }
    for class_id, class_name in PHASE3_CLASSES.items()
]

DATASET_METADATA = {
    "dataset_name": "DroneVehicle",
    "num_classes": NUM_CLASSES,
    "class_id_to_name": PHASE3_CLASSES,
    "class_name_to_id": PHASE3_CLASS_NAME_TO_ID,
    "class_names_ordered": PHASE3_CLASS_NAMES,
    "coco_categories": COCO_CATEGORIES,
    "source_classes": DRONEVEHICLE_CLASSES,
    "source_to_phase3": DRONEVEHICLE_TO_PHASE3,
}

def class_id_to_name(class_id: int) -> str:
    return PHASE3_CLASSES.get(int(class_id), f"unknown_{class_id}")

def normalize_dronevehicle_class_name(raw_name: str) -> str:
    if raw_name is None:
        raise ValueError("Raw class name is None")

    name = str(raw_name).strip().lower().replace("-", " ")
    name = " ".join(name.split())

    if name not in DRONEVEHICLE_NAME_NORMALIZATION:
        raise KeyError(f"Unknown DroneVehicle class name: {raw_name}")

    return DRONEVEHICLE_NAME_NORMALIZATION[name]

def dronevehicle_name_to_phase3_id(raw_name: str) -> int:
    normalized_name = normalize_dronevehicle_class_name(raw_name)
    return DRONEVEHICLE_TO_PHASE3[normalized_name]

# ---- Sanity checks ----
assert sorted(PHASE3_CLASSES.keys()) == list(range(1, NUM_CLASSES + 1))
assert len(PHASE3_CLASS_NAMES) == NUM_CLASSES

for i, name in enumerate(PHASE3_CLASS_NAMES, start=1):
    assert PHASE3_CLASSES[i] == name

assert [c["id"] for c in COCO_CATEGORIES] == list(range(1, NUM_CLASSES + 1))
assert [c["name"] for c in COCO_CATEGORIES] == PHASE3_CLASS_NAMES

normalized_source_names = set(DRONEVEHICLE_NAME_NORMALIZATION.values())
expected_source_names = set(DRONEVEHICLE_CLASSES.values())

assert normalized_source_names == expected_source_names
assert set(DRONEVEHICLE_TO_PHASE3.keys()) == expected_source_names
assert all(mapped_id in PHASE3_CLASSES for mapped_id in DRONEVEHICLE_TO_PHASE3.values())

log_print("=== Phase 3 Class Mapping Locked ===")
log_print(f"Number of classes: {NUM_CLASSES}\n")

log_print("Phase 3 Classes:")
for k, v in PHASE3_CLASSES.items():
    log_print(f"  {k}: {v}")

log_print("\nOrdered class names:")
for i, name in enumerate(PHASE3_CLASS_NAMES, start=1):
    log_print(f"  {i}: {name}")

log_print("\nDroneVehicle Source Classes:")
for k, v in DRONEVEHICLE_CLASSES.items():
    log_print(f"  {k}: {v}")

log_print("\nDroneVehicle -> Phase 3 Mapping:")
for src_name in sorted(DRONEVEHICLE_TO_PHASE3.keys()):
    phase3_id = DRONEVEHICLE_TO_PHASE3[src_name]
    log_print(f"  {src_name:<12} -> Phase 3 {phase3_id} ({PHASE3_CLASSES[phase3_id]})")

log_print("\nCOCO Categories:")
for cat in COCO_CATEGORIES:
    log_print(f"  id={cat['id']} | name={cat['name']}")

log_print("===================================")

gcs_upload_file(NOTEBOOK_LOG_PATH, f"{GCS_LOG_BUCKET}/{NOTEBOOK_LOG_PATH.name}")



# ============================================================
# Cell 3: Sync COCO Dataset from GCS + Validate + Visual Sanity Check
# ============================================================

VALIDATE_SPLIT = "train"
NUM_SAMPLES = 4

# ------------------------------------------------------------
# Step 1: Sync dataset from Cloud Storage
# ------------------------------------------------------------
log_print("\n=== Sync COCO Dataset from GCS ===")
sync_gcs_prefix_to_local(GCS_DATA_BUCKET, SHIM_ROOT)

required_dirs = [COCO_TRAIN_IMG, COCO_VAL_IMG, COCO_TEST_IMG, COCO_ANN_DIR]
required_files = [COCO_TRAIN_ANN, COCO_VAL_ANN, COCO_TEST_ANN]

for d in required_dirs:
    assert d.exists(), f"Missing required dataset directory: {d}"

for f in required_files:
    assert f.exists(), f"Missing required annotation file: {f}"

log_print("Dataset sync complete.")
log_print(f"Shim root       : {SHIM_ROOT}")
log_print(f"Train images    : {COCO_TRAIN_IMG}")
log_print(f"Val images      : {COCO_VAL_IMG}")
log_print(f"Test images     : {COCO_TEST_IMG}")
log_print(f"Annotation dir  : {COCO_ANN_DIR}")

# ------------------------------------------------------------
# Step 2: Validate selected split
# ------------------------------------------------------------
COCO_JSON_PATH = SHIM_ROOT / "annotations" / f"instances_{VALIDATE_SPLIT}2017.json"
assert COCO_JSON_PATH.exists(), f"COCO JSON not found: {COCO_JSON_PATH}"

log_print(f"\nLoading COCO annotations from:\n{COCO_JSON_PATH}\n")

coco = COCO(str(COCO_JSON_PATH))

# ---- Validate categories ----
cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])
exported_cat_ids = [c["id"] for c in cats]
exported_cat_names = [c["name"] for c in cats]

assert exported_cat_ids == list(range(1, NUM_CLASSES + 1)), (
    f"Exported category ids mismatch.\n"
    f"Expected: {list(range(1, NUM_CLASSES + 1))}\n"
    f"Got     : {exported_cat_ids}"
)

assert exported_cat_names == PHASE3_CLASS_NAMES, (
    f"Exported category names mismatch.\n"
    f"Expected: {PHASE3_CLASS_NAMES}\n"
    f"Got     : {exported_cat_names}"
)

log_print("=== COCO Categories (Validated) ===")
for c in cats:
    log_print(f"id={c['id']} | name={c['name']}")
log_print("===================================\n")

# ---- Dataset statistics ----
num_images = len(coco.getImgIds())
num_annotations = len(coco.getAnnIds())

log_print("=== Dataset Statistics ===")
log_print(f"Images     : {num_images}")
log_print(f"Annotations: {num_annotations}")
log_print("==========================\n")

assert num_images > 0, "No images found in COCO dataset"
assert num_annotations > 0, "No annotations found in COCO dataset"

# ---- Per-class annotation counts ----
log_print("=== Per-Class Annotation Counts ===")
for c in cats:
    cat_id = c["id"]
    ann_ids = coco.getAnnIds(catIds=[cat_id])
    log_print(f"{c['name']:<8}: {len(ann_ids)}")
log_print("==================================\n")

# ---- Integrity checks ----
all_imgs = coco.loadImgs(coco.getImgIds())
all_img_ids = set(coco.getImgIds())
all_anns = coco.loadAnns(coco.getAnnIds())

invalid_category_count = 0
invalid_image_ref_count = 0
invalid_box_count = 0
invalid_box_bounds_count = 0

img_id_to_info = {img["id"]: img for img in all_imgs}

for ann in all_anns:
    if ann["category_id"] not in PHASE3_CLASSES:
        invalid_category_count += 1

    if ann["image_id"] not in all_img_ids:
        invalid_image_ref_count += 1
        continue

    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0:
        invalid_box_count += 1
        continue

    img_info = img_id_to_info[ann["image_id"]]
    img_w = img_info["width"]
    img_h = img_info["height"]

    if x < 0 or y < 0 or x + w > img_w + 1e-6 or y + h > img_h + 1e-6:
        invalid_box_bounds_count += 1

assert invalid_category_count == 0, f"Found {invalid_category_count} invalid category_ids"
assert invalid_image_ref_count == 0, f"Found {invalid_image_ref_count} invalid image references"
assert invalid_box_count == 0, f"Found {invalid_box_count} non-positive boxes"
assert invalid_box_bounds_count == 0, f"Found {invalid_box_bounds_count} out-of-bounds boxes"

log_print("=== Annotation Integrity Checks Passed ===")
log_print("All category_ids are valid")
log_print("All image references are valid")
log_print("All boxes have positive width/height")
log_print("All boxes lie within image bounds")
log_print("==========================================\n")

# ------------------------------------------------------------
# Step 3: Visual sanity check
# ------------------------------------------------------------
rng = random.Random(SEED)
sample_img_ids = rng.sample(coco.getImgIds(), min(NUM_SAMPLES, len(coco.getImgIds())))

for idx, img_id in enumerate(sample_img_ids, start=1):
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    img_path = SHIM_ROOT / VALIDATE_SPLIT / "dummy"  # placeholder so path var exists

    # Resolve actual split image directory
    if VALIDATE_SPLIT == "train":
        image_dir = COCO_TRAIN_IMG
    elif VALIDATE_SPLIT == "val":
        image_dir = COCO_VAL_IMG
    elif VALIDATE_SPLIT == "test":
        image_dir = COCO_TEST_IMG
    else:
        raise ValueError(f"Unsupported split: {VALIDATE_SPLIT}")

    img_path = image_dir / img_info["file_name"]
    assert img_path.exists(), f"Image file not found: {img_path}"

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to load image: {img_path}"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for ann in anns:
        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = map(int, [x, y, x + w, y + h])
        cls_name = class_id_to_name(ann["category_id"])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            cls_name,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    
    plt.figure(figsize=(12, 7))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{VALIDATE_SPLIT.upper()} COCO sample (image_id={img_id}, anns={len(anns)})")
    save_current_figure(f"{VALIDATE_SPLIT}_sample_{idx}_imageid_{img_id}")
    plt.close()

# ------------------------------------------------------------
# Step 4: Upload session log
# ------------------------------------------------------------
gcs_upload_file(NOTEBOOK_LOG_PATH, f"{GCS_LOG_BUCKET}/{NOTEBOOK_LOG_PATH.name}")
log_print(f"[LOG UPLOADED] {GCS_LOG_BUCKET}/{NOTEBOOK_LOG_PATH.name}")




# ============================================================
# Cell 4: Forward-Only RF-DETR Sanity Check
# Reads staged COCO dataset from local copy synced from GCS
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_print(f"\nRunning forward-only sanity check on device: {device}\n")

# ---- Verify shim paths exist ----
assert SHIM_ROOT.exists(), f"Missing shim root: {SHIM_ROOT}"
assert COCO_TRAIN_IMG.exists(), f"Missing COCO train image dir: {COCO_TRAIN_IMG}"
assert COCO_VAL_IMG.exists(), f"Missing COCO val image dir: {COCO_VAL_IMG}"
assert COCO_TRAIN_ANN.exists(), f"Missing COCO train annotation json: {COCO_TRAIN_ANN}"
assert COCO_VAL_ANN.exists(), f"Missing COCO val annotation json: {COCO_VAL_ANN}"

log_print("Using COCO shim root:", SHIM_ROOT)
log_print("Train images     :", COCO_TRAIN_IMG)
log_print("Val images       :", COCO_VAL_IMG)
log_print("Train ann        :", COCO_TRAIN_ANN)
log_print("Val ann          :", COCO_VAL_ANN)

# ---- Initialize RF-DETR Nano with canonical class count ----
model = RFDETRNano(num_classes=NUM_CLASSES)

# Access underlying torch model
torch_model = model.model.model
torch_model.to(device)
torch_model.eval()

# ---- Minimal args object required by RF-DETR COCO dataset builder ----
args = SimpleNamespace(
    dataset_file="coco",
    coco_path=str(SHIM_ROOT),
    square_resize_div_64=True,
    multi_scale=False,
    expanded_scales=False,
    do_random_resize_via_padding=False,
    patch_size=16,
    num_windows=2,
)

TARGET_IMAGE_SIZE = 512

dataset = build_coco(
    image_set="train",
    args=args,
    resolution=TARGET_IMAGE_SIZE,
)

log_print(f"\nDataset length: {len(dataset)}")
assert len(dataset) > 0, "Dataset is empty"

# ---- Fetch one sample ----
samples, targets = dataset[0]

log_print("\n=== Sample Inspection ===")
log_print(f"Sample tensor shape: {tuple(samples.shape)}")
log_print(f"Target keys        : {list(targets.keys())}")

assert "boxes" in targets, "Missing 'boxes' in targets"
assert "labels" in targets, "Missing 'labels' in targets"

num_target_boxes = len(targets["boxes"])
unique_labels = sorted(set(targets["labels"].cpu().numpy().tolist()))

log_print(f"Number of boxes    : {num_target_boxes}")
log_print(f"Unique label ids   : {unique_labels}")

assert num_target_boxes > 0, "Sample has zero target boxes"

for label_id in unique_labels:
    assert 1 <= int(label_id) <= NUM_CLASSES, (
        f"Found out-of-range label_id={label_id}; expected within 1..{NUM_CLASSES}"
    )

boxes = targets["boxes"]
assert boxes.ndim == 2 and boxes.shape[1] == 4, (
    f"Expected target boxes shape [N, 4], got {tuple(boxes.shape)}"
)

box_min = boxes.min().item()
box_max = boxes.max().item()
log_print(f"Box value range    : min={box_min:.4f}, max={box_max:.4f}")

samples = samples.unsqueeze(0).to(device)
targets = [{k: v.to(device) for k, v in targets.items()}]

with torch.no_grad():
    outputs = torch_model(samples, targets)

log_print("\nForward pass successful")
log_print("Output keys:", list(outputs.keys()))

assert "pred_logits" in outputs, "Missing 'pred_logits' in model outputs"
assert "pred_boxes" in outputs, "Missing 'pred_boxes' in model outputs"

pred_logits = outputs["pred_logits"]
pred_boxes = outputs["pred_boxes"]

log_print("\n=== Output Shape Inspection ===")
log_print(f"pred_logits shape: {tuple(pred_logits.shape)}")
log_print(f"pred_boxes shape : {tuple(pred_boxes.shape)}")

assert pred_logits.ndim == 3
assert pred_boxes.ndim == 3
assert pred_boxes.shape[-1] == 4

log_print("\n=== Class Mapping Check ===")
log_print(f"Canonical num_classes : {NUM_CLASSES}")
log_print(f"Canonical class names : {PHASE3_CLASS_NAMES}")

log_print("Forward-only sanity check passed.")

gcs_upload_file(NOTEBOOK_LOG_PATH, f"{GCS_LOG_BUCKET}/{NOTEBOOK_LOG_PATH.name}")





# ============================================================
# Cell 5: Phase 3 Training Config + GCS Checkpoint Sync + Metadata Export
# ============================================================

# ---- Target input resolution for RF-DETR Nano ----
TARGET_IMAGE_SIZE = 512

# ---- Training contract / metadata ----
DRONEVEHICLE_ORIGINAL_IMAGE_SIZE = {
    "width": 840,
    "height": 712,
}

DRONEVEHICLE_BORDER_POLICY = {
    "remove_white_border": True,
    "border_pixels": {
        "left": 100,
        "right": 100,
        "top": 100,
        "bottom": 100,
    },
    "expected_cropped_size": {
        "width": 640,
        "height": 512,
    },
    "apply_during_coco_export": True,
    "apply_consistently_to_rgb_and_ir": True,
    "adjust_annotation_coordinates_after_crop": True,
}

PREPROCESSING_POLICY = {
    "image": {
        "target_size": TARGET_IMAGE_SIZE,
        "resize_enabled": True,
        "resize_method": "resize_longest_side_then_pad",
        "preserve_aspect_ratio": True,
        "pad_value": 0,
    },
    "annotations": {
        "source_format": "coco",
        "source_box_type": "axis_aligned_coco_bbox",
        "clip_boxes_to_image": True,
        "filter_non_positive_boxes": True,
        "min_box_area_px_after_resize": 4,
    },
    "normalization": {
        "enabled": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "training_phase3": {
        "continuation_training": True,
        "initialize_from_previous_checkpoint": True,
        "source_checkpoint_family": "enhanced-training",
        "small_object_focus": True,
        "paired_modal_dataset": True,
        "planned_training_usage": "treat_rgb_and_ir_as_independent_detection_samples_after_conversion",
    }
}

# ------------------------------------------------------------
# Download source checkpoints from GCS
# ------------------------------------------------------------
log_print("\n=== Sync Enhanced-Training Checkpoints from GCS ===")
for src in [GCS_SOURCE_BEST_EMA, GCS_SOURCE_BEST_TOTAL, GCS_SOURCE_LATEST]:
    dst = LOCAL_ENHANCED_DIR / Path(parse_gs_uri(src)[1]).name
    gcs_download_file(src, dst)

log_print("Local enhanced-training dir:", LOCAL_ENHANCED_DIR)
log_print("Best EMA      :", SOURCE_BEST_EMA_CKPT)
log_print("Best total    :", SOURCE_BEST_TOTAL_CKPT)
log_print("Latest resume :", SOURCE_LATEST_CKPT)

if SOURCE_BEST_EMA_CKPT.exists():
    RESOLVED_SOURCE_CHECKPOINT = SOURCE_BEST_EMA_CKPT
elif SOURCE_LATEST_CKPT.exists():
    RESOLVED_SOURCE_CHECKPOINT = SOURCE_LATEST_CKPT
else:
    raise FileNotFoundError(
        "No enhanced-training checkpoint found after GCS sync.\n"
        f"Tried:\n- {SOURCE_BEST_EMA_CKPT}\n- {SOURCE_LATEST_CKPT}"
    )

log_print("Resolved checkpoint:", RESOLVED_SOURCE_CHECKPOINT)

# ------------------------------------------------------------
# Output dirs / GCS destinations for later container job
# ------------------------------------------------------------
OUT_ROOT = PROJECT_ROOT / "checkpoints"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "phase-3-dronevehicle-finetune"

TRAIN_CONFIG = {
    "dataset_file": "coco",
    "dataset_dir": str(SHIM_ROOT),
    "coco_path": str(SHIM_ROOT),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes": NUM_CLASSES,
    "input_size": TARGET_IMAGE_SIZE,
    "resolution": TARGET_IMAGE_SIZE,
    "batch_size": 1,
    "grad_accum_steps": 16,
    "epochs": 40,
    "lr": 1e-4,
    "amp": True,
    "use_ema": True,
    "do_benchmark": False,
    "fp16_eval": False,
    "early_stopping": True,
    "early_stopping_patience": 5,
    # Linux container recommendation: change from 0 to something > 0 later
    "num_workers": 4,
    "checkpoint_interval": 1,
    "tensorboard": True,
}

log_print("\n=== Training Config ===")
for k, v in TRAIN_CONFIG.items():
    log_print(f"{k}: {v}")

# ---- Dataset summary from staged files ----
def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

train_coco = load_json(COCO_TRAIN_ANN)
val_coco = load_json(COCO_VAL_ANN)

train_images_total = len(train_coco["images"])
val_images_total = len(val_coco["images"])
train_annotations_total = len(train_coco["annotations"])
val_annotations_total = len(val_coco["annotations"])

log_print("\n=== Dataset Summary ===")
log_print(f"Train images      : {train_images_total}")
log_print(f"Train annotations : {train_annotations_total}")
log_print(f"Val images        : {val_images_total}")
log_print(f"Val annotations   : {val_annotations_total}")
log_print(f"Shim root         : {SHIM_ROOT}")

RUN_CANONICAL_METADATA = {
    "created_at": datetime.now().isoformat(),
    "experiment_name": EXPERIMENT_NAME,
    "based_on": "enhanced-training",
    "dataset_name": "DroneVehicle",
    "dataset_modalities": ["rgb", "infrared"],
    "continuation_checkpoint_local": str(RESOLVED_SOURCE_CHECKPOINT),
    "continuation_checkpoint_gcs_prefix": GCS_MODEL_BUCKET,
    "shim_root_local": str(SHIM_ROOT),
    "shim_root_gcs": GCS_DATA_BUCKET,
    "gcs_log_prefix": GCS_LOG_BUCKET,
    "gcs_model_prefix": "gs://rfdetr-model-versions/infrared-training",
    "num_classes": NUM_CLASSES,
    "class_id_to_name": PHASE3_CLASSES,
    "class_names_ordered": PHASE3_CLASS_NAMES,
    "coco_categories": COCO_CATEGORIES,
    "target_image_size": TARGET_IMAGE_SIZE,
    "original_image_size": DRONEVEHICLE_ORIGINAL_IMAGE_SIZE,
    "border_policy": DRONEVEHICLE_BORDER_POLICY,
    "preprocessing_policy": PREPROCESSING_POLICY,
    "train_config": TRAIN_CONFIG,
    "dataset_summary": {
        "train_images_total": train_images_total,
        "train_annotations_total": train_annotations_total,
        "val_images_total": val_images_total,
        "val_annotations_total": val_annotations_total,
    },
    "notes": {
        "primary_change_1": "read COCO-formatted infrared dataset shim from Cloud Storage",
        "primary_change_2": "continue training from enhanced-training checkpoint stored in Cloud Storage",
        "primary_change_3": "train on both RGB and infrared images via unified COCO dataset layout",
        "primary_change_4": "training logs and plots should be uploaded to Cloud Storage",
        "primary_change_5": "full multi-epoch training will run inside containerized Vertex job, not notebook",
    },
}

CANONICAL_METADATA_PATH = LOCAL_META_ROOT / f"{EXPERIMENT_NAME}_canonical_training_metadata.json"

with open(CANONICAL_METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(RUN_CANONICAL_METADATA, f, indent=2)

log_print(f"\nSaved canonical metadata to:\n{CANONICAL_METADATA_PATH}")

# Upload metadata + notebook log to GCS
gcs_upload_file(CANONICAL_METADATA_PATH, f"{GCS_LOG_BUCKET}/metadata/{CANONICAL_METADATA_PATH.name}")
gcs_upload_file(NOTEBOOK_LOG_PATH, f"{GCS_LOG_BUCKET}/{NOTEBOOK_LOG_PATH.name}")

log_print("\nMetadata and notebook log uploaded to GCS.")




# ============================================================
# Cell 6: Container Logging + GCS Sync Helpers + Epoch Metrics
# ============================================================

# ------------------------------------------------------------
# GCS destination prefixes for full training runs
# ------------------------------------------------------------
GCS_TRAIN_LOG_PREFIX = "gs://rfdetr-training-logs/infrared-training"
GCS_TRAIN_MODEL_PREFIX = "gs://rfdetr-model-versions/infrared-training"

# ------------------------------------------------------------
# Local output roots
# ------------------------------------------------------------
OUT_ROOT = PROJECT_ROOT / "checkpoints"
LOG_DIR = LOCAL_LOG_ROOT
META_DIR = LOCAL_META_ROOT

OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Structured epoch metrics tracker
# ------------------------------------------------------------
class EpochMetricsTracker:
    """
    Parses stdout/stderr text lines and writes structured epoch metrics
    to JSONL and CSV so they can be plotted later.

    This is intentionally tolerant: RF-DETR / DETR-style logs can vary.
    We capture what we can whenever it appears.
    """
    def __init__(self, jsonl_path: Path, csv_path: Path):
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)

        self.current_epoch = None
        self.current_phase = None
        self.last_train_iter = None
        self.last_train_iter_total = None
        self.last_val_iter = None
        self.last_val_iter_total = None

        self.rows = []
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0

        # Common patterns seen in DETR-style logs
        self.epoch_iter_re = re.compile(r"Epoch:\s*\[(\d+)\]\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]")
        self.test_iter_re = re.compile(r"Test:\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]")
        self.metric_pair_re = re.compile(r"([A-Za-z0-9_/@\-.]+)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    def _append_row(self, row: dict):
        row = dict(row)
        row["timestamp_utc"] = datetime.utcnow().isoformat()

        # JSONL append
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        # CSV append (union-style header support)
        preferred_order = [
            "timestamp_utc",
            "epoch",
            "phase",
            "iter",
            "iter_total",
            "class_error",
            "loss",
            "loss_ce",
            "loss_bbox",
            "loss_giou",
            "lr",
            "precision",
            "recall",
            "map",
            "map50",
            "map75",
            "small_map",
            "medium_map",
            "large_map",
            "raw_line",
        ]

        existing_keys = list(row.keys())
        header = [k for k in preferred_order if k in row] + [k for k in existing_keys if k not in preferred_order]

        if not self._csv_header_written:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(row)
            self._csv_header_written = True
        else:
            # Re-open with existing header
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                current_header = next(reader)

            missing = [k for k in header if k not in current_header]
            if missing:
                # Rewrite CSV with expanded header
                new_header = current_header + missing
                existing_rows = []
                with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        existing_rows.append(r)

                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=new_header, extrasaction="ignore")
                    writer.writeheader()
                    for r in existing_rows:
                        writer.writerow(r)
                    writer.writerow(row)
            else:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=current_header, extrasaction="ignore")
                    writer.writerow(row)

        self.rows.append(row)

    def consume_line(self, line: str):
        stripped = line.strip()
        if not stripped:
            return

        # Track training iter
        m = self.epoch_iter_re.search(stripped)
        if m:
            self.current_epoch = int(m.group(1)) + 1   # make epochs 1-based in stored files
            self.current_phase = "train"
            self.last_train_iter = int(m.group(2))
            self.last_train_iter_total = int(m.group(3))

            # Save periodic train progress snapshot
            self._append_row({
                "epoch": self.current_epoch,
                "phase": "train_progress",
                "iter": self.last_train_iter,
                "iter_total": self.last_train_iter_total,
                "raw_line": stripped,
            })
            return

        # Track validation iter
        t = self.test_iter_re.search(stripped)
        if t:
            self.current_phase = "val"
            self.last_val_iter = int(t.group(1))
            self.last_val_iter_total = int(t.group(2))

            self._append_row({
                "epoch": self.current_epoch,
                "phase": "val_progress",
                "iter": self.last_val_iter,
                "iter_total": self.last_val_iter_total,
                "raw_line": stripped,
            })
            return

        # Parse metric pairs from any line that looks metric-like
        pairs = dict(self.metric_pair_re.findall(stripped))
        if pairs:
            row = {
                "epoch": self.current_epoch,
                "phase": self.current_phase or "unknown",
                "raw_line": stripped,
            }

            key_map = {
                "class_error": "class_error",
                "loss": "loss",
                "loss_ce": "loss_ce",
                "loss_bbox": "loss_bbox",
                "loss_giou": "loss_giou",
                "lr": "lr",
                "precision": "precision",
                "recall": "recall",
                "map": "map",
                "mAP": "map",
                "AP": "map",
                "AP50": "map50",
                "AP75": "map75",
                "small_AP": "small_map",
                "medium_AP": "medium_map",
                "large_AP": "large_map",
            }

            captured_any = False
            for raw_k, raw_v in pairs.items():
                k = key_map.get(raw_k, raw_k)
                try:
                    row[k] = float(raw_v)
                    captured_any = True
                except Exception:
                    pass

            if captured_any:
                if self.current_phase == "train":
                    row["iter"] = self.last_train_iter
                    row["iter_total"] = self.last_train_iter_total
                elif self.current_phase == "val":
                    row["iter"] = self.last_val_iter
                    row["iter_total"] = self.last_val_iter_total

                self._append_row(row)
            return

        # Capture final COCO summary lines if present
        if "Average Precision" in stripped or "Average Recall" in stripped:
            self._append_row({
                "epoch": self.current_epoch,
                "phase": "coco_eval_summary",
                "raw_line": stripped,
            })

# ------------------------------------------------------------
# Tee stdout/stderr to file + console + metrics parser
# ------------------------------------------------------------
class TeeStream:
    def __init__(self, *streams, tracker=None):
        self.streams = streams
        self.tracker = tracker
        self._buffer = ""

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

        if self.tracker is not None:
            self._buffer += data
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self.tracker.consume_line(line)

    def flush(self):
        for s in self.streams:
            s.flush()

        if self.tracker is not None and self._buffer.strip():
            self.tracker.consume_line(self._buffer)
            self._buffer = ""

# ------------------------------------------------------------
# Periodic uploader thread
# ------------------------------------------------------------
class PeriodicGCSUploader:
    def __init__(
        self,
        run_log_path: Path,
        output_dir: Path,
        run_name: str,
        metrics_jsonl_path: Path = None,
        metrics_csv_path: Path = None,
        interval_sec: int = 120,
    ):
        self.run_log_path = Path(run_log_path)
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.metrics_jsonl_path = Path(metrics_jsonl_path) if metrics_jsonl_path else None
        self.metrics_csv_path = Path(metrics_csv_path) if metrics_csv_path else None
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.thread = None

        self.run_log_gcs = f"{GCS_TRAIN_LOG_PREFIX}/{run_name}/{self.run_log_path.name}"
        self.run_ckpt_gcs = f"{GCS_TRAIN_MODEL_PREFIX}/{run_name}"
        self.metrics_jsonl_gcs = (
            f"{GCS_TRAIN_LOG_PREFIX}/{run_name}/{self.metrics_jsonl_path.name}"
            if self.metrics_jsonl_path else None
        )
        self.metrics_csv_gcs = (
            f"{GCS_TRAIN_LOG_PREFIX}/{run_name}/{self.metrics_csv_path.name}"
            if self.metrics_csv_path else None
        )

    def _sync_once(self):
        try:
            if self.run_log_path.exists():
                gcs_upload_file(self.run_log_path, self.run_log_gcs)

            if self.metrics_jsonl_path and self.metrics_jsonl_path.exists():
                gcs_upload_file(self.metrics_jsonl_path, self.metrics_jsonl_gcs)

            if self.metrics_csv_path and self.metrics_csv_path.exists():
                gcs_upload_file(self.metrics_csv_path, self.metrics_csv_gcs)

            if self.output_dir.exists():
                gcs_upload_dir(self.output_dir, self.run_ckpt_gcs)

        except Exception as e:
            print(f"[WARN] Periodic GCS sync failed: {e}", flush=True)

    def _loop(self):
        while not self.stop_event.is_set():
            self._sync_once()
            self.stop_event.wait(self.interval_sec)

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=30)
        self._sync_once()

print("Container logging, GCS sync, and epoch metrics tracking helpers ready.")





# ============================================================
# Cell 7: Full RF-DETR Phase 3 Training Launch (Container)
# With structured epoch metrics logging
# ============================================================

assert torch.cuda.is_available(), "CUDA GPU is required for training."
assert RESOLVED_SOURCE_CHECKPOINT.exists(), (
    f"Resolved checkpoint not found: {RESOLVED_SOURCE_CHECKPOINT}"
)

print("GPU:", torch.cuda.get_device_name(0), flush=True)

run_name = time.strftime("phase3_dronevehicle_%Y%m%d_%H%M%S")
output_dir = OUT_ROOT / run_name
output_dir.mkdir(parents=True, exist_ok=True)

log_path = LOG_DIR / f"{run_name}.log"
run_metadata_path = META_DIR / f"{run_name}_metadata.json"
plots_dir = LOG_DIR / "plots" / run_name
plots_dir.mkdir(parents=True, exist_ok=True)

# ---- New structured metrics files ----
epoch_metrics_jsonl_path = LOG_DIR / f"{run_name}_epoch_metrics.jsonl"
epoch_metrics_csv_path = LOG_DIR / f"{run_name}_epoch_metrics.csv"

FULL_TRAIN_CONFIG = dict(TRAIN_CONFIG)
FULL_TRAIN_CONFIG["output_dir"] = str(output_dir)

# ------------------------------------------------------------
# Save run metadata
# ------------------------------------------------------------
run_metadata = {
    "run_name": run_name,
    "experiment_name": "phase-3-dronevehicle-finetune",
    "based_on": "enhanced-training",
    "dataset_name": "DroneVehicle",
    "dataset_modalities": ["rgb", "infrared"],
    "log_path": str(log_path),
    "epoch_metrics_jsonl_path": str(epoch_metrics_jsonl_path),
    "epoch_metrics_csv_path": str(epoch_metrics_csv_path),
    "output_dir": str(output_dir),
    "seed": SEED,
    "source_checkpoint_local": str(RESOLVED_SOURCE_CHECKPOINT),
    "source_checkpoint_gcs_prefix": GCS_MODEL_BUCKET,
    "shim_root_local": str(SHIM_ROOT),
    "shim_root_gcs": GCS_DATA_BUCKET,
    "gcs_log_prefix": f"{GCS_TRAIN_LOG_PREFIX}/{run_name}",
    "gcs_model_prefix": f"{GCS_TRAIN_MODEL_PREFIX}/{run_name}",
    "num_classes": NUM_CLASSES,
    "class_names_ordered": PHASE3_CLASS_NAMES,
    "class_id_to_name": PHASE3_CLASSES,
    "target_image_size": TARGET_IMAGE_SIZE,
    "train_config": FULL_TRAIN_CONFIG,
    "dataset_summary": {
        "train_images_total": len(load_json(COCO_TRAIN_ANN)["images"]),
        "train_annotations_total": len(load_json(COCO_TRAIN_ANN)["annotations"]),
        "val_images_total": len(load_json(COCO_VAL_ANN)["images"]),
        "val_annotations_total": len(load_json(COCO_VAL_ANN)["annotations"]),
    },
    "notes": {
        "structured_epoch_metrics": True,
        "epoch_metrics_files": [
            str(epoch_metrics_jsonl_path),
            str(epoch_metrics_csv_path),
        ],
        "primary_change_1": "read COCO-formatted infrared dataset shim from Cloud Storage",
        "primary_change_2": "continue training from enhanced-training checkpoint stored in Cloud Storage",
        "primary_change_3": "train on both RGB and infrared images via unified COCO layout",
        "primary_change_4": "stream logs/checkpoints/epoch metrics to Cloud Storage during training",
        "primary_change_5": "containerized execution for Vertex AI",
    },
}

with open(run_metadata_path, "w", encoding="utf-8") as f:
    json.dump(run_metadata, f, indent=2)

print("Run metadata saved:", run_metadata_path, flush=True)

gcs_upload_file(
    run_metadata_path,
    f"{GCS_TRAIN_LOG_PREFIX}/{run_name}/metadata/{run_metadata_path.name}"
)

# ------------------------------------------------------------
# Enable safer CUDA allocation behavior
# ------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ------------------------------------------------------------
# Metrics tracker
# ------------------------------------------------------------
metrics_tracker = EpochMetricsTracker(
    jsonl_path=epoch_metrics_jsonl_path,
    csv_path=epoch_metrics_csv_path,
)

# ------------------------------------------------------------
# Redirect stdout/stderr to tee + parser
# ------------------------------------------------------------
orig_stdout = sys.stdout
orig_stderr = sys.stderr

log_file = open(log_path, "w", encoding="utf-8", buffering=1)
tee = TeeStream(orig_stdout, log_file, tracker=metrics_tracker)
sys.stdout = tee
sys.stderr = tee

uploader = PeriodicGCSUploader(
    run_log_path=log_path,
    output_dir=output_dir,
    run_name=run_name,
    metrics_jsonl_path=epoch_metrics_jsonl_path,
    metrics_csv_path=epoch_metrics_csv_path,
    interval_sec=120,
)
uploader.start()

test_results = None
training_exception = None

try:
    print("=== Full Training Startup ===", flush=True)
    print("Run name:", run_name, flush=True)
    print("Output dir:", output_dir, flush=True)
    print("Log path:", log_path, flush=True)
    print("Epoch metrics JSONL:", epoch_metrics_jsonl_path, flush=True)
    print("Epoch metrics CSV:", epoch_metrics_csv_path, flush=True)
    print("Metadata path:", run_metadata_path, flush=True)
    print("Checkpoint:", RESOLVED_SOURCE_CHECKPOINT, flush=True)
    print("Shim root:", SHIM_ROOT, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("NUM_CLASSES:", NUM_CLASSES, flush=True)
    print("PHASE3_CLASS_NAMES:", PHASE3_CLASS_NAMES, flush=True)
    print("TRAIN_CONFIG:", FULL_TRAIN_CONFIG, flush=True)
    print("=============================", flush=True)

    model = RFDETRNano(
        num_classes=NUM_CLASSES,
        pretrain_weights=str(RESOLVED_SOURCE_CHECKPOINT),
    )

    test_results = model.train(**FULL_TRAIN_CONFIG)

    print("\nTraining completed successfully.", flush=True)
    print("Returned test_results:", test_results, flush=True)

except Exception as e:
    training_exception = e
    print("\n[ERROR] Training failed with exception:", flush=True)
    traceback.print_exc()

finally:
    try:
        if test_results is not None:
            results_path = META_DIR / f"{run_name}_test_results.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(test_results, f, indent=2, default=str)

            gcs_upload_file(
                results_path,
                f"{GCS_TRAIN_LOG_PREFIX}/{run_name}/metadata/{results_path.name}"
            )
    except Exception as e:
        print(f"[WARN] Failed to save/upload test results: {e}", flush=True)

    # Final explicit uploads
    try:
        if epoch_metrics_jsonl_path.exists():
            gcs_upload_file(
                epoch_metrics_jsonl_path,
                f"{GCS_TRAIN_LOG_PREFIX}/{run_name}/{epoch_metrics_jsonl_path.name}"
            )
        if epoch_metrics_csv_path.exists():
            gcs_upload_file(
                epoch_metrics_csv_path,
                f"{GCS_TRAIN_LOG_PREFIX}/{run_name}/{epoch_metrics_csv_path.name}"
            )
        if log_path.exists():
            gcs_upload_file(
                log_path,
                f"{GCS_TRAIN_LOG_PREFIX}/{run_name}/{log_path.name}"
            )
    except Exception as e:
        print(f"[WARN] Failed final metrics/log upload: {e}", flush=True)

    try:
        uploader.stop()
    except Exception as e:
        print(f"[WARN] Failed to stop uploader cleanly: {e}", flush=True)

    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    log_file.close()

print("\nTraining process complete.")
print("Log file:", log_path)
print("Epoch metrics JSONL:", epoch_metrics_jsonl_path)
print("Epoch metrics CSV:", epoch_metrics_csv_path)
print("Output dir:", output_dir)
print("Run metadata:", run_metadata_path)

if training_exception is not None:
    raise RuntimeError(
        f"Phase 3 training failed. See local log: {log_path} "
        f"and GCS log prefix: {GCS_TRAIN_LOG_PREFIX}/{run_name}"
    ) from training_exception




# ============================================================
# Cell 8: Post-Run Log Parsing + Class Audit + Metrics Check
# ============================================================

assert log_path.exists(), f"Log file not found: {log_path}"
assert epoch_metrics_jsonl_path.exists(), f"Epoch metrics JSONL not found: {epoch_metrics_jsonl_path}"
assert epoch_metrics_csv_path.exists(), f"Epoch metrics CSV not found: {epoch_metrics_csv_path}"

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    log_text = f.read()

tail_lines = log_text.splitlines()[-1200:]

print("\n===== Log Audit =====")

if "Traceback" in log_text:
    print("WARNING: Training log contains a traceback / failure.")
else:
    print("No traceback detected in log.")

if (
    "num_classes=90" in log_text
    or "num_classes': 90" in log_text
    or '"num_classes": 90' in log_text
):
    print("WARNING: Found evidence of num_classes=90 in the training log.")
else:
    print("No explicit num_classes=90 string found in log.")

unexpected_names = ["Motorcycle", "motorcycle", "person"]
found_unexpected = [name for name in unexpected_names if name in log_text]

if found_unexpected:
    print("WARNING: Found unexpected class-name strings in log:", found_unexpected)
else:
    print("No obvious unexpected default COCO class-name strings found in log.")

print("\n===== COCO EVAL (from log) =====")
capture = False
found_eval = False
eval_block_lines = []

for ln in tail_lines:
    if "IoU metric: bbox" in ln:
        capture = True
        found_eval = True
    if capture:
        s = ln.rstrip()
        if s:
            print(s)
            eval_block_lines.append(s)

if not found_eval:
    print("No completed COCO evaluation block found in log.")

# ------------------------------------------------------------
# Metrics file summary
# ------------------------------------------------------------
print("\n===== Structured Epoch Metrics =====")

with open(epoch_metrics_jsonl_path, "r", encoding="utf-8") as f:
    metric_rows = [json.loads(line) for line in f if line.strip()]

print(f"JSONL rows: {len(metric_rows)}")
print(f"CSV path   : {epoch_metrics_csv_path}")

epochs_seen = sorted({
    row.get("epoch")
    for row in metric_rows
    if row.get("epoch") is not None
})

print(f"Epochs seen in structured logs: {epochs_seen}")

# Show a few rows that contain likely useful metrics
interesting_rows = []
for row in metric_rows:
    keys = set(row.keys())
    if any(k in keys for k in ["precision", "recall", "map", "map50", "loss", "loss_ce", "loss_bbox", "loss_giou"]):
        interesting_rows.append(row)

print(f"Rows containing numeric metrics: {len(interesting_rows)}")
for row in interesting_rows[:10]:
    print(row)

# ------------------------------------------------------------
# Final test results
# ------------------------------------------------------------
if test_results is not None:
    print("\n===== Key Test Results =====")
    for k in ["class_error", "loss", "loss_ce", "loss_bbox", "loss_giou", "coco_eval_bbox"]:
        if k in test_results:
            print(f"{k}: {test_results[k]}")
else:
    print("\nNo in-memory test_results object available from training.")

# ------------------------------------------------------------
# Show latest checkpoints
# ------------------------------------------------------------
ckpts = sorted(output_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)

if ckpts:
    print("\n===== Latest Checkpoints =====")
    for p in ckpts[-10:]:
        print(" -", p.name)
else:
    print("\nNo checkpoints found in output_dir.")

# ------------------------------------------------------------
# Save audit summary and upload
# ------------------------------------------------------------
audit_summary = {
    "run_name": run_name,
    "traceback_found": "Traceback" in log_text,
    "num_classes_90_found": (
        "num_classes=90" in log_text
        or "num_classes': 90" in log_text
        or '"num_classes": 90' in log_text
    ),
    "unexpected_names_found": found_unexpected,
    "checkpoints_found": [p.name for p in ckpts],
    "test_results_present": test_results is not None,
    "epoch_metrics_jsonl_path": str(epoch_metrics_jsonl_path),
    "epoch_metrics_csv_path": str(epoch_metrics_csv_path),
    "epoch_metrics_rows": len(metric_rows),
    "epochs_seen": epochs_seen,
    "interesting_metric_rows": len(interesting_rows),
}

audit_path = META_DIR / f"{run_name}_audit_summary.json"
with open(audit_path, "w", encoding="utf-8") as f:
    json.dump(audit_summary, f, indent=2)

gcs_upload_file(audit_path, f"{GCS_TRAIN_LOG_PREFIX}/{run_name}/metadata/{audit_path.name}")
gcs_upload_file(log_path, f"{GCS_TRAIN_LOG_PREFIX}/{run_name}/{log_path.name}")
gcs_upload_file(epoch_metrics_jsonl_path, f"{GCS_TRAIN_LOG_PREFIX}/{run_name}/{epoch_metrics_jsonl_path.name}")
gcs_upload_file(epoch_metrics_csv_path, f"{GCS_TRAIN_LOG_PREFIX}/{run_name}/{epoch_metrics_csv_path.name}")

print("\nAudit summary saved:", audit_path)





# ============================================================
# Cell 9: Pin Current Run Checkpoints + Upload Stable Copies
# ============================================================

assert output_dir.exists(), f"Run output_dir does not exist: {output_dir}"

MODELS_ROOT = PROJECT_ROOT / "models"
INFRARED_MODELS_DIR = MODELS_ROOT / "infrared-training"
INFRARED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Candidate checkpoints from training ----
best_ema = output_dir / "checkpoint_best_ema.pth"
best_total = output_dir / "checkpoint_best_total.pth"
latest = output_dir / "checkpoint.pth"

print("=== Checkpoint Availability ===")
print("best_ema   :", best_ema.exists(), best_ema)
print("best_total :", best_total.exists(), best_total)
print("latest     :", latest.exists(), latest)

if not best_ema.exists() and not latest.exists():
    raise FileNotFoundError(
        "No usable checkpoints found in this run output directory.\n"
        "Expected at least one of:\n"
        f"- {best_ema}\n"
        f"- {latest}"
    )

# ---- Stable destination names ----
dst_best_ema = INFRARED_MODELS_DIR / "dronevehicle_rfdetr_nano_best_ema.pth"
dst_latest = INFRARED_MODELS_DIR / "dronevehicle_rfdetr_nano_latest_resume.pth"
dst_best_total = INFRARED_MODELS_DIR / "dronevehicle_rfdetr_nano_best_total.pth"

# ---- Copy locally ----
if best_ema.exists():
    shutil.copy2(best_ema, dst_best_ema)
    print("[COPY] Best EMA ->", dst_best_ema)
else:
    print("[WARN] best_ema checkpoint missing; not copied")

if latest.exists():
    shutil.copy2(latest, dst_latest)
    print("[COPY] Latest resume ->", dst_latest)
else:
    print("[WARN] latest checkpoint missing; not copied")

if best_total.exists():
    shutil.copy2(best_total, dst_best_total)
    print("[COPY] Best total ->", dst_best_total)
else:
    print("[WARN] checkpoint_best_total.pth not found; not copied")

# ---- Save metadata for this pinned model set ----
infrared_training_metadata = {
    "source_run_name": run_name,
    "source_output_dir": str(output_dir),
    "source_log_path": str(log_path),
    "experiment_name": "phase-3-dronevehicle-finetune",
    "based_on": "enhanced-training",
    "dataset_name": "DroneVehicle",
    "dataset_modalities": ["rgb", "infrared"],
    "source_checkpoint_local": str(RESOLVED_SOURCE_CHECKPOINT),
    "source_checkpoint_gcs_prefix": GCS_MODEL_BUCKET,
    "shim_root_local": str(SHIM_ROOT),
    "shim_root_gcs": GCS_DATA_BUCKET,
    "num_classes": NUM_CLASSES,
    "class_names_ordered": PHASE3_CLASS_NAMES,
    "target_image_size": TARGET_IMAGE_SIZE,
    "train_config": FULL_TRAIN_CONFIG,
    "test_results": test_results if test_results is not None else None,
    "pinned_files_local": {
        "best_ema": str(dst_best_ema) if best_ema.exists() else None,
        "latest_resume": str(dst_latest) if latest.exists() else None,
        "best_total": str(dst_best_total) if best_total.exists() else None,
    },
    "gcs_destinations": {
        "run_prefix": f"{GCS_TRAIN_MODEL_PREFIX}/{run_name}",
        "latest_prefix": f"{GCS_TRAIN_MODEL_PREFIX}/latest",
    },
}

metadata_path = INFRARED_MODELS_DIR / "metadata.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(infrared_training_metadata, f, indent=2, default=str)

print("\nPinned checkpoints summary:")
print(" Best EMA      :", str(dst_best_ema) if best_ema.exists() else None)
print(" Latest resume :", str(dst_latest) if latest.exists() else None)
print(" Best total    :", str(dst_best_total) if best_total.exists() else None)
print(" Metadata      :", metadata_path)

# ------------------------------------------------------------
# Upload to GCS:
# - run-specific copies
# - latest stable copies
# - metadata
# ------------------------------------------------------------
if best_ema.exists():
    gcs_upload_file(best_ema, f"{GCS_TRAIN_MODEL_PREFIX}/{run_name}/{best_ema.name}")
    gcs_upload_file(dst_best_ema, f"{GCS_TRAIN_MODEL_PREFIX}/latest/{dst_best_ema.name}")

if latest.exists():
    gcs_upload_file(latest, f"{GCS_TRAIN_MODEL_PREFIX}/{run_name}/{latest.name}")
    gcs_upload_file(dst_latest, f"{GCS_TRAIN_MODEL_PREFIX}/latest/{dst_latest.name}")

if best_total.exists():
    gcs_upload_file(best_total, f"{GCS_TRAIN_MODEL_PREFIX}/{run_name}/{best_total.name}")
    gcs_upload_file(dst_best_total, f"{GCS_TRAIN_MODEL_PREFIX}/latest/{dst_best_total.name}")

gcs_upload_file(metadata_path, f"{GCS_TRAIN_MODEL_PREFIX}/{run_name}/{metadata_path.name}")
gcs_upload_file(metadata_path, f"{GCS_TRAIN_MODEL_PREFIX}/latest/{metadata_path.name}")

print("\nModel artifacts uploaded to GCS.")




# ============================================================
# Cell 10: Inspect Suspicious Log Lines
# ============================================================

assert log_path.exists(), f"Log file not found: {log_path}"

patterns = [
    "num_classes=90",
    "motorcycle",
    "Motorcycle",
    "person",
    "class_names",
    "OutOfMemoryError",
    "CUDA out of memory",
    "Traceback",
    "RuntimeError",
]

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

matches = []
for i, line in enumerate(lines, start=1):
    lower = line.lower()
    for p in patterns:
        if p.lower() in lower:
            matches.append((i, line.rstrip()))
            break

print("=== Suspicious Log Matches ===")
if not matches:
    print("No matches found.")
else:
    for line_no, text in matches[:200]:
        print(f"{line_no}: {text}")

# Save and upload suspicious matches report
suspicious_report_path = META_DIR / f"{run_name}_suspicious_log_matches.txt"
with open(suspicious_report_path, "w", encoding="utf-8") as f:
    if not matches:
        f.write("No matches found.\n")
    else:
        for line_no, text in matches[:200]:
            f.write(f"{line_no}: {text}\n")

gcs_upload_file(
    suspicious_report_path,
    f"{GCS_TRAIN_LOG_PREFIX}/{run_name}/metadata/{suspicious_report_path.name}"
)