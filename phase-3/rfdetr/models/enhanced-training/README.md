# Enhanced Training

This model directory contains the enhanced RF-DETR Nano checkpoints produced from the enhanced Phase 3 training notebook.

## Summary

This training run builds directly on the **basic-training** baseline and is intended to test whether a longer training schedule and higher input resolution improve detection quality on the RGB VisDrone-derived dataset.

Compared to the baseline, this run increases the training budget and image resolution while keeping the same canonical 6-class taxonomy and the same overall training pipeline structure.

This run was configured for **40 epochs**, but it completed successfully with **exit code 0** and stopped early at approximately **epoch 16**, which is consistent with the configured early stopping behavior.

## Relationship to the baseline

This experiment is an extension of `models/basic-training`.

The basic-training run established:

- the clean canonical 6-class mapping
- the validated COCO export pipeline
- the stable RF-DETR-compatible dataset layout
- the baseline 384-resolution training setup

This enhanced run keeps those fixes and increases the training difficulty and capacity to test whether the model can benefit from more training and more image detail.

## Main changes from basic training

Relative to the basic-training baseline, the following changes were introduced.

### 1. Input resolution was increased
The training input resolution was increased from:

- **384 -> 512**

The motivation was to preserve more spatial detail, especially for smaller objects that may be harder to detect at lower resolution.

This is particularly relevant for UAV imagery because many targets occupy only a small portion of the frame.

### 2. Training duration was increased
The configured number of epochs was increased from:

- **20 -> 40**

The goal was to give the model more opportunity to improve after the baseline run, especially now that it was training at a larger resolution.

### 3. The run remained based on the same canonical taxonomy
This enhanced run still uses the same project-specific canonical classes:

- Human
- Bicycle
- Car
- Truck
- Bus
- Other

No taxonomy changes were introduced here. That is important because it keeps comparisons against the baseline fair and interpretable.

### 4. The same reproducible pipeline structure was kept
The run continued to use the stricter Phase 3 training pipeline introduced in the baseline, including:

- canonical class mapping
- validated dataset structure
- validated COCO export
- RF-DETR dataset compatibility checks
- run metadata export
- checkpoint pinning
- post-run log auditing

This means the enhanced run changes model capacity and training budget, not the underlying label system.

## Why training stopped at 16/40 epochs

Although the run was configured for 40 epochs, it stopped early and still exited successfully with:

- **exit code 0**

That means the run did **not** crash and did **not** fail with an execution error.

The most likely reason is **early stopping**.

This training configuration kept early stopping enabled, so if validation performance stopped improving for enough consecutive checks, the training process would terminate before reaching the maximum epoch count.

In practical terms, this usually means one of the following:

- validation performance plateaued
- improvements became too small to count as meaningful
- the model had already extracted most of the available gain from this dataset/setup

It does **not** necessarily mean the model became worse after epoch 16. More often, it means the validation metric was no longer improving enough to justify continuing the run.

Because the run exited cleanly, the early stop should be interpreted as a normal training outcome rather than an error.

## Possible reasons improvement plateaued

There are several reasonable explanations for why the run may have stopped at epoch 16 instead of using all 40 epochs.

### 1. The model may have saturated on this dataset/setup
RF-DETR Nano may have already captured most of the useful patterns available under the current label mapping and dataset quality.

### 2. The validation set may not support continued measurable gains
The validation split is not huge, so performance improvements at later epochs may be noisy or too small to register as significant.

### 3. Higher resolution does not always guarantee continued gains
Increasing resolution helps preserve detail, but it also makes optimization heavier. In some cases, it improves results only modestly before plateauing.

### 4. The current dataset may be the limiting factor
If label quality, class ambiguity, or object scale distribution are the main bottlenecks, then simply adding epochs may not continue to help much.

## Important note about errors

This run should be considered a **successful completed run**, not a failed one.

Why:

- the process exited with **code 0**
- checkpoints were produced
- logging completed normally
- the stop behavior is consistent with early stopping

So this run ended early by training logic, not by notebook failure.

## Training setup

Enhanced configuration used in this run:

- **Model:** RF-DETR Nano
- **Pretraining:** RF-DETR Nano pretrained weights used as initialization
- **Classes:** 6 project-specific classes
- **Input resolution:** 512
- **Configured epochs:** 40
- **Actual stop point:** about epoch 16
- **Batch size:** 1 or 2 depending on machine/VRAM-adjusted config used for the run
- **Gradient accumulation:** used to maintain an effective larger batch
- **Learning rate:** 1e-4
- **AMP:** enabled
- **EMA:** enabled
- **Early stopping:** enabled
- **Workers:** low-worker / Windows-safe configuration

## Purpose of this model folder

This folder stores the checkpoints from the **enhanced-training** stage.

These checkpoints are intended to help answer:

- does higher resolution improve small-object detection?
- does longer training improve results beyond the baseline?
- is the baseline already near saturation?
- should future effort focus on resolution, tiling, data quality, or label refinement?

## Files expected in this folder

Typical contents include:

- `visdrone_rfdetr_nano_best_ema.pth`
- `visdrone_rfdetr_nano_latest_resume.pth`
- `visdrone_rfdetr_nano_best_total.pth` (if available)
- `metadata.json`

Large checkpoint files are not expected to be committed to GitHub.

## Notes

This run is best understood as a controlled extension of the baseline rather than a completely new pipeline.

The key experiment here was:

- **more detail** through higher input resolution
- **more time** through a longer training schedule

Since the run stopped cleanly before 40 epochs, the result suggests that the current setup may already be close to a plateau, and future gains may require changes beyond simply training longer.