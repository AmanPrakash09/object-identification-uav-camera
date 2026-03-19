# Infrared Training

This model directory contains the RF-DETR Nano checkpoints produced from the infrared Phase 3 training run on the DroneVehicle RGB/infrared dataset, using the cloud-based Vertex AI training pipeline.

## Summary

This training run builds directly on the enhanced-training stage and shifts the project from RGB-only VisDrone-derived training to a vehicle-focused RGB/infrared dataset. The goal of this phase was not just to continue fine-tuning, but to adapt the detector to paired aerial RGB and thermal-style infrared imagery while keeping the same canonical project taxonomy and the same overall RF-DETR training framework.

Unlike the earlier RGB VisDrone runs, this phase uses the DroneVehicle COCO-formatted infrared-data shim stored in Cloud Storage and trains from the enhanced-training checkpoint family rather than from generic pretrained RF-DETR weights. The final run achieved a validation mAP@50 of 0.8639, validation mAP@50:95 of 0.6263, precision of 0.8244, and recall of 0.79. The corresponding held-out test metrics were very similar, with test mAP@50 of 0.8682, test mAP@50:95 of 0.6274, precision of 0.8176, and recall of 0.79.

## Relationship to enhanced training

This experiment is an extension of models/enhanced-training.

The enhanced-training run established:
- the clean canonical 6-class mapping
- the stable 512-resolution training setup
- the validated COCO-based RF-DETR data pipeline
- the stronger training schedule beyond the original basic baseline

This infrared run keeps those improvements and then changes the dataset domain, data source, training environment, and initialization point:
- it starts from the enhanced-training checkpoint
- it trains on the DroneVehicle RGB/infrared COCO shim
- it runs through the Vertex AI containerized training pipeline
- it stores logs, metrics, and checkpoints in Google Cloud Storage

So this run is best understood as a continuation and domain adaptation stage rather than a fresh baseline.

## How the dataset is different

This is the biggest change in the project so far.

### 1. The source dataset changed

The earlier runs used a VisDrone-derived RGB dataset focused on the project’s 6-class taxonomy. This run switches to DroneVehicle, which is structurally and visually different.

The DroneVehicle data is centered on aerial vehicle detection and includes both:
- RGB imagery
- infrared imagery

That means this phase moves beyond standard visible-spectrum UAV imagery and explicitly introduces low-light / thermal-like signal into training.

### 2. The label distribution is different

In practice, the infrared training run is vehicle-dominant. The evaluation outputs report only:
- Car
- Truck
- Bus
- all

rather than all 6 project classes. That matches the underlying DroneVehicle domain, where the useful positive categories in this phase are vehicle classes rather than the full earlier VisDrone-style class spread.

So while the project still keeps the same canonical taxonomy machinery, the actual signal used in this run is concentrated on vehicle detection.

### 3. Data Layout was made COCO-compatible Separately

For this phase, the dataset was prepared in advance as an RF-DETR-compatible COCO shim in `phase-3\rfdetr\training\rfdetr_training_infrared.ipynb` and uploaded to Cloud Storage. The training script `phase-3\rfdetr\training\google-cloud-vertex-ai\infrared\trainer\rfdetr_training_infrared.py` therefore no longer needed to rebuild the dataset structure from raw source annotations during the training run. Instead, it:
- synced the COCO-formatted dataset from GCS
- validated the annotations and categories
- performed a forward-only sanity check
- launched full training from the enhanced checkpoint

This made the cloud training flow cleaner and more reproducible.

### 4. RGB and infrared were treated as independent COCO samples

The training script uses the unified COCO layout under `train2017`, `val2017`, and `test2017`, with annotations under `instances_train2017.json`, `instances_val2017.json`, and `instances_test2017.json`. The full training job uses the training split for optimization and the validation split for evaluation/model selection, while the test split is kept separate for later holdout reporting.

## Main changes from enhanced training

Relative to the enhanced RGB training stage, the following changes were introduced.

### 1. Training started from the enhanced checkpoint family

Instead of initializing from generic RF-DETR Nano pretrained weights, this run initializes from the previously trained enhanced checkpoint set:
- `visdrone_rfdetr_nano_best_ema.pth`
- `visdrone_rfdetr_nano_best_total.pth`
- `visdrone_rfdetr_nano_latest_resume.pth`

The script resolves the best available continuation checkpoint and uses it as the starting point for the infrared run.

### 2. The training domain changed from RGB-only to RGB + infrared

This is the defining change of the experiment. The model is no longer being optimized only for visible-spectrum UAV imagery. It is being adapted to a dataset that includes infrared views and is much more vehicle-centric.

### 3. The run moved to Vertex AI

This phase was trained through a containerized Vertex AI workflow rather than through the original local notebook-only process. The cloud pipeline was designed to:
- download the dataset from Cloud Storage
- download the source checkpoint from Cloud Storage
- run full training in a managed GPU environment
- upload logs, epoch metrics, metadata, and checkpoints back to Cloud Storage during training

This was necessary because full training at 512 resolution had become too time-consuming locally.

### 4. Structured epoch logging was added

A major goal of this migration was not just training speed, but observability. The script writes structured per-epoch metrics to JSONL and CSV so that later plots of precision, recall, AP, and loss trends can be reproduced from saved artifacts rather than only from notebook output.

## Training outcome

This run appears to have completed 18 epochs. The metric plot shows epochs indexed from 0 through 18, and the trend is consistent with a completed run rather than an early crash.

Across training:
- **training loss** steadily decreased from just above **4.0** to about **2.67**
- **validation loss** stayed much flatter, mostly in the **3.2–3.5** range
- **AP@50** improved quickly early on, then plateaued in the mid/high **0.85** range for the base model and slightly above that for the EMA model
- **AP@50:95** climbed from about **0.51–0.54** at the start to roughly **0.62–0.63**
- **average recall** rose into the **0.73–0.74+** range, with the EMA model consistently slightly stronger than the base model

The EMA model outperformed the regular model throughout most of training, which matches the checkpointing strategy used and supports relying on the EMA checkpoint as the strongest primary artifact.

The final validation results reported in results.json are:
- Validation mAP@50:95: 0.6263
- Validation mAP@50: 0.8639
- Validation precision: 0.8244
- Validation recall: 0.79

Per class on validation:
- Car: mAP@50:95 = 0.6426, mAP@50 = 0.9343
- Truck: mAP@50:95 = 0.5110, mAP@50 = 0.7286
- Bus: mAP@50:95 = 0.7252, mAP@50 = 0.9288

The corresponding test results are:
- Test mAP@50:95: 0.6274
- Test mAP@50: 0.8682
- Test precision: 0.8176
- Test recall: 0.79

Per class on test:
- Car: mAP@50:95 = 0.6434, mAP@50 = 0.9363
- Truck: mAP@50:95 = 0.5095, mAP@50 = 0.7302
- Bus: mAP@50:95 = 0.7294, mAP@50 = 0.9380

These validation and test numbers are very close to each other, which is a good sign that the model generalized stably across the held-out splits.

## What the logs show

The training log also records the per-epoch progression. Early epochs started around:

- epoch 0 EMA mAP@50:95 of 0.5370
- epoch 1 EMA mAP@50:95 of 0.5720
- epoch 2 EMA mAP@50:95 of 0.5870
- epoch 3 EMA mAP@50:95 of 0.5930
- epoch 4 EMA mAP@50:95 of 0.5984
- epoch 5 EMA mAP@50:95 of 0.6081
- epoch 6 EMA mAP@50:95 of 0.6086

The early log trend shows rapid gains in the first several epochs, which aligns with the attached plots and supports the idea that most of the useful adaptation happened early, followed by slower improvement and plateauing.

The log also shows each epoch taking roughly 1 hour 40 minutes to 1 hour 50 minutes including evaluation. For example, the early recorded epoch_time values were around 1:41:34, 1:45:24, 1:47:43, 1:48:57, 1:49:20, 1:49:21, and 1:50:06.

## Training setup

Infrared configuration used in this run:
- Model: RF-DETR Nano
- Initialization: continuation from enhanced-training checkpoint
- Dataset: DroneVehicle COCO-formatted RGB/infrared shim
- Classes: project canonical taxonomy retained, but this run’s active positives are vehicle-focused
- Input resolution: 512
- Configured epochs: 40
- Completed epochs: about 19 plotted / 18 completed intervals in the final run artifacts
- Batch size: 1
- Gradient accumulation steps: 16
- Effective batch size: 16
- Learning rate: 1e-4
- AMP: enabled
- EMA: enabled
- Early stopping: enabled
- Workers: 4 in the cloud/container training config

These settings come directly from the Vertex-oriented training script we created for this phase.

## Purpose of this model folder

This folder stores the checkpoints from the infrared-training stage.

These checkpoints are intended to answer questions such as:
- how well does the enhanced RGB-trained model adapt to infrared and mixed-modality vehicle imagery?
- does DroneVehicle fine-tuning improve vehicle detection quality over the RGB-only VisDrone stages?
- how much of the gain comes from continuation training versus domain shift?
- should future work focus on infrared-only inference, multimodal evaluation, or further dataset-specific refinement?

## Files expected in this folder

Typical contents include:

- `dronevehicle_rfdetr_nano_best_ema.pth`
- `dronevehicle_rfdetr_nano_latest_resume.pth`
- `dronevehicle_rfdetr_nano_best_total.pth`
- `metadata.json`

Large checkpoint files are not expected to be committed to GitHub.

## Notes

This run is important because it is the first model in the project trained on the DroneVehicle infrared pipeline rather than the earlier VisDrone RGB pipeline.

The key experiment here was not just “train longer,” but:
- continue from the strongest RGB checkpoint family
- switch to a new aerial vehicle dataset
- incorporate infrared imagery
- move training into a cloud-managed Vertex AI workflow
- preserve reproducibility through structured logging, metadata export, and checkpoint pinning

Overall, this phase should be interpreted as a domain adaptation and cloud-training milestone rather than just another incremental hyperparameter tweak.