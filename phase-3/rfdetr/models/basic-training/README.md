# Basic Training

This model directory contains the baseline fine-tuned RF-DETR Nano checkpoints produced from the `rfdetr-training-rgb.ipynb` notebook.

## Summary

This training run establishes the first clean and reproducible baseline for Phase 3 object detection on the RGB VisDrone-derived dataset. It uses the canonical 6-class taxonomy for this project and fixes several consistency issues that existed in the previous phase.

This run trains for **20 epochs** and serves as the reference point for future experiments such as enhanced training, tiling, and resolution scaling.

## Phase 3 goals

The goal of this phase is to create a stable baseline model and training pipeline before attempting more advanced improvements. The focus is on:

- making the class taxonomy fully consistent across conversion, training, evaluation, and inference
- validating the dataset and COCO export pipeline end to end
- producing pinned checkpoints for future comparison
- establishing a reliable baseline before testing higher resolutions or other enhancements

## What was improved from the previous phase

Compared to the previous phase, the training pipeline was cleaned up and made much more strict and reproducible.

### 1. Canonical class mapping was fixed
A single source of truth was introduced for the project taxonomy:

- Human
- Bicycle
- Car
- Truck
- Bus
- Other

This canonical mapping is now reused consistently for:

- VisDrone-to-project label remapping
- COCO export
- dataset validation
- evaluation
- inference visualization

This was done to prevent class-name drift and mismatches such as incorrect default COCO class names appearing in final outputs.

### 2. Dataset structure validation was strengthened
The dataset layout is now explicitly validated before conversion or training. Checks include:

- split directory existence
- `images/` and `annotations/` existence
- image/annotation count matching
- filename stem matching between images and annotations

This reduces silent path and file mismatches.

### 3. COCO conversion was made more robust
The VisDrone-to-COCO conversion step was improved to:

- use the canonical class taxonomy
- parse annotation rows more defensively
- skip malformed rows safely
- clip boxes to image bounds
- track conversion statistics and skip reasons
- record per-class annotation counts

This makes the exported dataset more reliable and easier to audit.

### 4. COCO validation was added and expanded
After conversion, the generated COCO dataset is now validated to ensure:

- category IDs match the canonical project classes
- category names match the canonical project classes
- image and annotation counts are correct
- annotation references are valid
- bounding boxes are well formed

This catches issues before training begins.

### 5. RF-DETR COCO shim was verified
The COCO-style dataset structure expected by RF-DETR was rebuilt and checked more carefully. This includes:

- copying train/val images into COCO-style folders
- copying annotation JSON files into RF-DETR’s expected structure
- validating copied annotation categories after the copy step

### 6. Forward-only sanity checking was added
Before training, a forward-only debug pass was used to verify:

- the RF-DETR dataset builds correctly
- target labels are in the expected class range
- sample boxes are valid
- the model can complete a forward pass without shape or compatibility issues

### 7. Training was split into safer stages
Instead of jumping straight into a long run, the workflow now includes:

- configuration and metadata export
- a smoke-test training run
- a full training run with logging
- post-run auditing of outputs

This makes debugging easier and reduces the chance of wasting time on a broken run.

### 8. Final class reporting was verified
The final evaluation output was checked to confirm that reported class names match the project taxonomy:

- Human
- Bicycle
- Car
- Truck
- Bus
- Other

This was important because earlier training behavior showed evidence of internal RF-DETR default COCO metadata appearing in logs. In this baseline run, the final reported class map matches the intended project classes.

### 9. Checkpoint pinning was introduced
The best checkpoints from the run were copied into a stable model directory with fixed names so they can be reused for:

- inference
- future fine-tuning
- comparison against later experiments

## What is different in this phase

This phase is different from the previous phase because the focus is not just on “getting training to run,” but on making the whole training pipeline trustworthy and repeatable.

The main difference is that this phase emphasizes:

- correctness of class taxonomy
- consistent metadata across all steps
- reproducibility
- dataset auditability
- stable baseline generation

This is the first version intended to be used as a proper baseline for later experiments.

## Training setup

Baseline configuration used in this run:

- **Model:** RF-DETR Nano
- **Pretraining:** RF-DETR Nano pretrained weights used as initialization
- **Classes:** 6 project-specific classes
- **Input resolution:** 384
- **Epochs:** 20
- **Batch size:** 2
- **Gradient accumulation steps:** 8
- **Effective batch size:** 16
- **Learning rate:** 1e-4
- **AMP:** enabled
- **EMA:** enabled
- **Early stopping:** enabled
- **Workers:** 0 (Windows-safe setting)

## Purpose of this model folder

This folder stores the baseline model checkpoints for the **basic-training** stage. These checkpoints are intended to be the comparison point for later experiments such as:

- enhanced training
- higher input resolution
- tiling-based training or inference
- filtered-label experiments
- class-balance improvements

## Files expected in this folder

Typical contents include:

- `visdrone_rfdetr_nano_best_ema.pth`
- `visdrone_rfdetr_nano_latest_resume.pth`
- `visdrone_rfdetr_nano_best_total.pth` (if available)
- `metadata.json`

The path files will not exist in this folder unless the training notebook is run. They are too large to push to GitHub.

## Notes

This baseline is intentionally conservative. It keeps the original 384 resolution and moderate epoch count so that later improvements can be measured cleanly against it.

Future training folders should document only the new changes introduced beyond this baseline.