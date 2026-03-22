# Training Model on Google Cloud's Vertex AI

Training this model locally can take a very long time. In our case, a single epoch took several hours on a local GPU, which made full multi-epoch training slow and inconvenient. For that reason, we migrated training to **Google Cloud Vertex AI** so we could use stronger GPUs, run training in a managed environment, store artifacts centrally, and avoid tying up a local machine for days.

We have provided detailed steps below on how to replicate our migration. Here is the documentation we referred to: https://docs.cloud.google.com/vertex-ai/docs/training/overview

## Why We Migrated to Vertex AI

We moved training to Vertex AI for a few main reasons:

- **Faster training with stronger GPUs**
- **Managed training jobs** so long-running training can continue in the cloud without depending on a local laptop or workstation
- **Cloud Storage integration** for datasets, logs, checkpoints, and model versions
- **Containerized reproducibility** so the training environment can be rebuilt consistently
- **Artifact versioning** through Artifact Registry and model/checkpoint storage in Cloud Storage

---

## Cloud Storage Buckets

Before training, we created three Google Cloud Storage buckets. This can be done via the Google Cloud console.

### 1. `infrared-data`
This bucket stores the COCO-formatted infrared dataset shim used by RF-DETR.

#### Purpose
- Stores the dataset in the exact layout expected by the training script.
- Allows the Vertex training container to download the dataset at runtime.

#### Structure
```text
gs://infrared-data/
  annotations/
    instances_train2017.json
    instances_val2017.json
    instances_test2017.json
  train2017/
    ...
  val2017/
    ...
  test2017/
    ...
```
Instructions on how to upload the dataset shim are shown below.

### 2. `rfdetr-training-logs`

This bucket stores training logs, metadata, plots, and structured epoch metrics.

#### Purpose
- Preserves long-running training logs outside the container.
- Stores run metadata for debugging and reproducibility.
- Stores epoch metrics so training progress can be analyzed later.
- Stores plots generated during sanity checks or post-training analysis.

#### Structure
```
gs://rfdetr-training-logs/infrared-training/
  <run_name>/
    <run_name>.log
    <run_name>_epoch_metrics.jsonl
    <run_name>_epoch_metrics.csv
    metadata/
      <run_name>_metadata.json
      <run_name>_audit_summary.json
      <run_name>_test_results.json
      <run_name>_suspicious_log_matches.txt
  plots/
    ...
  metadata/
    phase-3-dronevehicle-finetune_canonical_training_metadata.json
```

### 3. `rfdetr-model-versions`

This bucket stores source checkpoints and newly trained model checkpoints.

#### Purpose

- Stores the original checkpoints used to initialize continuation training.
- Stores run-specific checkpoints from Vertex AI training jobs.
- Stores pinned “latest” model files for future inference or continued fine-tuning.

#### Structure
```
gs://rfdetr-model-versions/
  enhanced-training/
    visdrone_rfdetr_nano_best_ema.pth
    visdrone_rfdetr_nano_best_total.pth
    visdrone_rfdetr_nano_latest_resume.pth

  infrared-training/
    <run_name>/
      checkpoint.pth
      checkpoint_best_ema.pth
      checkpoint_best_total.pth
      metadata.json

    latest/
      dronevehicle_rfdetr_nano_best_ema.pth
      dronevehicle_rfdetr_nano_best_total.pth
      dronevehicle_rfdetr_nano_latest_resume.pth
      metadata.json
```
**Note:** the model checkpoints in `enhanced-training` were uploaded via the Google Cloud console. We had run the `rfdetr-training-rgb-enhanced.ipynb` file locally and had the checkpoints on our machine. Since uploading these checkpoints to the bucket doesn't take too much time, the process can be done with the Google Cloud console. To upload the large training data, refer to the section below.

## Uploading the Dataset Shim to Cloud Storage

After preparing the infrared-data-shim folder locally, we uploaded it to the infrared-data bucket using the Google Cloud CLI which should be installed on your local machine (and added to path if needed). 

#### Step 1: Authenticate
```
gcloud auth login
```

#### Step 2: Sync the folder to Cloud Storage
```
gsutil -m rsync -r ./infrared-data-shim gs://infrared-data
```
This recursively synced the local dataset folder into the bucket and can take a long time. If the process is stopped in the middle, simply rerun the command to pick up where the process left off without having to start from scratch.

Replace `./infrared-data-shim` and `gs://infrared-data` with your actual local path and bucket name if they differ.

## Creating the Vertex AI Workbench Instance

Before building the training container, we created a Vertex AI Workbench instance to prepare the training files in a cloud development environment. Go to the "Workbench" tab in Vertex AI to create a new instance.

#### Why we used Workbench

The Workbench instance was used to:

- prepare the training script in the cloud
- upload the trainer folder
- build the Docker image
- push the image to Artifact Registry
- launch and monitor Vertex AI training jobs

This made it easier to work inside the same Google Cloud environment where the training job would later run.

#### What to upload

After creating the Workbench instance, upload the entire `infrared` folder to the VM.

Example local project structure:
```
infrared/
  trainer/
    rfdetr_training_infrared.py
  Dockerfile
  requirements.txt
```
The `trainer` folder contains the Python training code that will later be copied into the Docker image.

#### Recommended Workbench setup

When creating the Workbench instance:

- choose the same region you plan to use for training if possible
- use a machine large enough to build Docker images comfortably, GPU is not required since this isn't where the model will be trained
- make sure Docker is available or can be installed
- authenticate with Google Cloud from the Workbench terminal before interacting with Cloud Storage or Artifact Registry

#### Why the `trainer` folder matters

The Docker image build later uses the `trainer` folder as part of the container filesystem. For example, the Dockerfile copies it into the image with a command like:
```
COPY trainer /app/trainer
```
Because of that, the Workbench VM needs to contain the `trainer` folder in the correct location before running `docker build`.

## Creating and Using Artifact Registry

We created an Artifact Registry repository to store the Docker image used by Vertex AI training. The repository was created using the Google Cloud console and the image URI was defined in a terminal on our workbench instance.

#### Artifact Registry repository
Example repo name:
```
infrared-training-repo
```

#### Image URI
```
IMAGE_URI=us-west1-docker.pkg.dev/$PROJECT_ID/infrared-training-repo/infrared_image:latest
```

## Building and Pushing the Training Container

After creating the repository, we built the Docker image from the repo root and pushed it to Artifact Registry. The commands were used on a terminal in our workbench instance.

#### Step 1: Configure Docker authentication for Artifact Registry
```
gcloud auth configure-docker us-west1-docker.pkg.dev
```

#### Step 2: Build the Docker image
```
docker build ./ -t $IMAGE_URI
```

#### Step 3: Push the Docker image
```
docker push $IMAGE_URI
```

This image contains:

- the training script
- Python dependencies
- RF-DETR training environment
- GCS integration for dataset download and artifact upload

## Vertex AI Training Job

Once the image was pushed, we launched a custom training job in Vertex AI using the Google Cloud console. Go to the "Training" tab in Vertex AI to create a new training pipeline.

#### Training configuration used

We used a single-worker, single-GPU training configuration.

Example configuration:

- Worker pool count: 1
- Machine type: `n1-standard-8`
- GPU type: `NVIDIA_TESLA_V100`
- GPU count: 1
- Boot disk type: SSD
- Boot disk size: 150 GB
- Provisioning model: Standard

#### Why this setup

Our script is currently written as a single-worker, single-GPU training job, so adding multiple workers or multiple GPUs would not automatically improve training unless the code is modified for distributed training.

The training pipeline:

1. Downloads the COCO dataset shim from `gs://infrared-data`
2. Downloads the enhanced-training checkpoint from `gs://rfdetr-model-versions/enhanced-training`
3. Runs dataset validation and a forward-only sanity check
4. Launches training
5. Continuously uploads logs, metrics, and checkpoints to Cloud Storage
6. Pins final model checkpoints back to `gs://rfdetr-model-versions/infrared-training`

## Quota Increase for GPU Training

When attempting to use stronger GPUs, Vertex AI may fail if the region does not have enough training quota.

In our case, we needed to request a quota increase for Vertex AI custom model training GPUs in our region.

#### Example quota request

Since we were using V100 in `us-west1`, we made a request quota for `CustomModelTrainingV100GPUsPerProjectPerRegion` in that region. This is done via the Google Cloud console.

The quota request should match the number of GPUs you actually need. Since our current script uses **1 worker and 1 GPU**, requesting a quota value of 1 was sufficient.

## Logging and Checkpoint Strategy

We designed the training pipeline so that long-running jobs store outputs outside the container.

#### Logs bucket

The training job writes:

- raw run logs
- structured epoch metrics (jsonl and csv)
- metadata files
- audit summaries
- suspicious log match reports

to:
```
gs://rfdetr-training-logs/infrared-training/
```

#### Model bucket

The training job writes:

- run-specific checkpoints
- pinned latest model files
- metadata for the latest promoted model

to:
```
gs://rfdetr-model-versions/infrared-training/
```

This allows:

- resuming training later
- comparing runs
- plotting epoch metrics later
- using the latest promoted checkpoint for inference

## Summary of Migration

The full migration from local training to Vertex AI involved:

1. Preparing the RF-DETR-compatible COCO dataset shim
2. Creating Cloud Storage buckets for:
    - dataset
    - logs
    - model versions
3. Uploading the dataset shim to Cloud Storage
4. Uploading source checkpoints to Cloud Storage
5. Containerizing the training script with Docker
6. Pushing the image to Artifact Registry
7. Creating and launching a Vertex AI custom training job
8. Selecting a single-GPU machine configuration
9. Requesting GPU quota increases where needed
10. Storing logs, metrics, and checkpoints back in Cloud Storage