# How to Get COCO Dataset

Follow these steps to download and set up the COCO 2017 dataset for Phase 1:

1. Navigate to the Phase 1 directory
```
cd object-identification-uav-camera/phase-1
```

2. Create/Activate virtual environment and install dependencies
```
python -m venv coco-env

.\coco-env\Scripts\activate        # Windows  
# or  
source coco-env/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

3. Run the script to get the dataset
```
python get_coco_dataset.py
```
This script will:
- Download COCO 2017 Train and Validation splits
- Filter for UAV-relevant classes (person, car, truck, etc.)
- Store all data under `phase-1/coco-dataset/coco-2017/`

## Model Evaluation Criteria
Try to use these metrics to benchmark the models in the notebooks.

### **Accuracy Metrics**

| Metric | Description |
|---------|-------------|
| **mAP@50** | Mean Average Precision at IoU = 0.50. Measures detection accuracy with moderate overlap tolerance. |
| **mAP@50:95** | Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95 (step of 0.05). Provides a stricter, more comprehensive accuracy measure. |
| **Precision** | Fraction of correct detections among all detections made — `TP / (TP + FP)`. |
| **Recall** | Fraction of actual objects correctly detected — `TP / (TP + FN)`. |
| **F1-Score** | Harmonic mean of Precision and Recall — `2 × (Precision × Recall) / (Precision + Recall)`. |
| **Confusion Matrix** | Visual breakdown of class-wise detection performance. |


### **Performance and Resource Metrics**


| Metric | Description |
|---------|-------------|
| **Frames Per Second (FPS)** | Number of frames processed per second during inference — higher is better. |
| **Latency (ms/frame)** | Average time to process one frame — lower is better. |
| **Model Size (MB)** | Total model file size — smaller models are more deployable on UAV hardware. |
| **Memory Usage (MB)** | Total RAM and VRAM used during inference. |
| **GPU Utilization (%)** | Percentage of GPU resources consumed during operation. |