# Object Identification via UAV Camera

## File Structure
```
object-identification-uav-camera/
│
├── README.md
│
└── phase-1/
    │
    ├── coco-env/
    │   ├── Scripts/
    │   └── Lib/
    │
    ├── coco-dataset/
    │   └── coco-2017/
    │       ├── raw/
    │       ├── train/
    │       │   ├── data/
    │       │   └── labels.json
    │       ├── validation/
    │       │   ├── data/
    │       │   └── labels.json
    │       └── info.json
    │
    ├── get_coco_dataset.py
    │
    ├── opencv-evaluation/
    │   ├── venv/
    │   └── opencv_coco_benchmark.ipynb
    │
    ├── rfdetr-evaluation/
    │   ├── venv/
    │   └── rfdetr_coco_benchmark.ipynb
    │
    └── yolov12-evaluation/
        ├── venv/
        └── yolov12_coco_benchmark.ipynb
```