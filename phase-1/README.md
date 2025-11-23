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