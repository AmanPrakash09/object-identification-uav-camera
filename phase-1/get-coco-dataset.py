""" 
Download and set up the COCO 2017 dataset using FiftyOne
"""

import fiftyone as fo
import fiftyone.zoo as foz

classes = ["person", "bicycle", "car", "bus", "train", "truck", "boat", "motorcycle", "airplane"]

dataset_dir = "./coco-dataset"
fo.config.dataset_zoo_dir = dataset_dir

print("Downloading COCO 2017 datasets via FiftyOne...")
print("Classes:", classes)

# --- Validation Split ---
print("Downloading COCO 2017 Validation split...")
val_dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections", "segmentations"],
    classes=classes,
    max_samples=None
)
print("Validation split ready!")
print(val_dataset)

# --- Training Split ---
print("Downloading COCO 2017 Training split...")
train_dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections", "segmentations"],
    classes=classes,
    max_samples=None
)
print("Training split ready!")
print(train_dataset)

# Optional: visualize to verify
# print("Launching FiftyOne App for visual inspection (close to continue)...")
# session = fo.launch_app(val_dataset)
# session.wait()
