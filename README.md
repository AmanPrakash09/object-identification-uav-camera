# Object Identification via UAV Camera

## File Structure
```
object-identification-uav-camera/
│
├── README.md
│
phase-3/
├── .gitignore
├── README.md
├── carla_vid_data/
├── jetson-nano/
    ├── README.md
    ├── det_track_only_usb.py
    ├── det_track_still_camera_speed_estimations.py
    ├── det_track_still_camera_speed_multi_model.py
    ├── requirements.txt
    ├── trained_rfdetr_live_inference.py
    └── trained_rfdetr_live_inference_usb.py
└── rfdetr/
    ├── README.md
    ├── requirements.txt
    ├── annotations/
    ├── calibration/
    ├── coco_annotations/
    ├── exports/
    ├── models/
    │   ├── README.md
    │   ├── basic-training/
    │   │   ├── metadata.json
    │   │   └── README.md
    │   ├── enhanced-training/
    │   │   ├── metadata.json
    │   │   └── README.md
    │   ├── infrared-training/
    │   ├── infrared-training-isolated/
    │   │   └── metadata.json
    │   │   └── README.md
    │   ├── infrared-training-smoke/
    │   └── rgb-video-training/
    ├── run_metadata/
    ├── tracking-speed/
    │   ├── .ipynb_checkpoints/
    │   ├── README.md
    │   ├── rfdetr_tracking_infrared.ipynb
    │   ├── rfdetr_tracking_speed.ipynb
    │   ├── rfdetr_tracking_speed_class_optimization.ipynb
    │   ├── rfdetr_tracking_speed_improvements.ipynb
    │   ├── rfdetr_tracking_speed_infrared.ipynb
    │   └── rfdetr_tracking_speed_vector_additions.ipynb
    ├── training/
    │   ├── .ipynb_checkpoints/
    │   ├── rfdetr_training_infrared.ipynb
    │   ├── rfdetr_training_rgb.ipynb
    │   ├── rfdetr_training_rgb_enhanced.ipynb
    │   ├── google-cloud-vertex-ai/
    │   │   ├── README.md
    │   │   └── infrared/
    │   │       ├── Dockerfile
    │   │       ├── requirements.txt
    │   │       └── trainer/
    │   │           └── rfdetr_training_infrared.py
    │   └── vm-remote-training/
    │       ├── requirements.txt
    │       ├── rfdetr_training_infrared_isolated.ipynb
    │       └── rfdetr_training_rgb_video.ipynb
    │       └── README.md
    └── videos/
```