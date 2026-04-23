# Jetson Orin Nano Inference and Runtime Setup

This folder contains the Jetson-side runtime scripts used to run Phase 3 RF-DETR models on a **Jetson Orin Nano 8GB**. These scripts are designed for live inference with cameras connected directly to the Jetson and are intended for deployment-style testing rather than offline notebook training.

The setup described here assumes:

- **Jetson Orin Nano 8GB**
- **JetPack 6.2.x**
- **Ubuntu 22.04**
- **CUDA 12.6-era Jetson environment**

## Summary

These scripts take trained RF-DETR checkpoints and run them live on the Jetson using camera feeds, GPU inference, object tracking, and in some cases approximate speed estimation.

The main goals of this runtime stage are:

- run trained RGB and infrared RF-DETR models directly on-device
- verify that Jetson GPU inference is working correctly
- test USB and CSI camera pipelines
- visualize detections and tracked objects in real time
- estimate object motion/speed in selected scripts
- compare RGB and infrared live behavior on embedded hardware

This README explains:

- what each Jetson script does
- which checkpoints they expect
- how to create the Python environment correctly on Jetson
- how to avoid the slow CPU-only PyTorch issue
- how to verify GStreamer / `gi` support
- how to run the scripts

---

## Why this setup matters

Jetson is not a standard desktop Python environment.

A normal `pip install torch torchvision` often installs the wrong build on Jetson, which can leave PyTorch running in **CPU-only mode**. That causes a large performance drop and was the main reason an earlier environment was slow.

This setup avoids that problem by:

- installing system packages first
- creating the virtual environment with `--system-site-packages`
- installing **Jetson-specific PyTorch wheels**
- verifying CUDA before installing the rest of the Python requirements
- verifying that `gi` and GStreamer work before running the app

---

## Scripts in this folder

### 1. `det_track_only_usb.py`

This is the simplest live runtime script in the folder.

It:

- opens a **USB webcam**
- loads the **RGB RF-DETR checkpoint**
- runs object detection
- converts detections into `supervision` detections
- tracks objects with **ByteTrack**
- displays live tracked detections with `pygame`

This script is useful when you only want:

- detection
- tracking
- a simple RGB live demo

and do **not** need speed estimation.

### 2. `det_track_still_camera_speed_estimations.py`

This script extends the USB detection/tracking flow by adding **speed estimation**.

It:

- opens a **USB webcam**
- loads the **RGB RF-DETR checkpoint**
- runs detection + ByteTrack tracking
- estimates object speed from tracked motion over time
- uses class-specific dimension priors to estimate local meters-per-pixel scale
- smooths speed estimates over a short temporal window
- overlays object ID, class, confidence, and estimated speed on screen

This is the main USB script when you want a **single RGB model with approximate speed estimation**.

### 3. `det_track_still_camera_speed_multi_model.py`

This script is similar to the previous speed-estimation script, but it supports **switching between two checkpoints** while the app is running.

It:

- opens a **USB webcam**
- starts in **RGB mode**
- can toggle between:
  - `visdrone_rfdetr_nano_best_ema.pth`
  - `dronevehicle_rfdetr_nano_best_ema.pth`
- resets tracking state when the model is switched
- continues to run detection, tracking, and speed estimation after switching

This is useful for quick live comparison between the **RGB-trained model** and the **infrared-trained model** while keeping the same camera/display flow.

### 4. `trained_rfdetr_live_inference.py`

This script is the more integrated Jetson live inference pipeline for the onboard camera setup.

It is designed around **CSI cameras** connected to the Jetson and includes:

- camera configuration for:
  - `IMX219 RGB (CAM0)`
  - `IMX462 IR (CAM1)`
- RF-DETR live inference
- ByteTrack tracking
- speed estimation logic
- GStreamer CSI capture through `nvarguscamerasrc`
- checkpoint selection per camera stream

This script is the main Jetson live runtime when using the **Jetson camera stack** rather than a USB webcam.

### 5. `trained_rfdetr_live_inference_usb.py`

This script is similar in spirit to the previous live inference script, but it mixes:

- a **USB RGB webcam**
- a **CSI infrared camera**
- per-camera checkpoint selection

It supports:

- RGB USB input on one side
- infrared CSI input on the other side
- model loading for RGB and IR checkpoints
- live inference, tracking, and speed estimation
- a more flexible mixed-camera runtime configuration

This is useful when the physical deployment setup uses **USB for RGB** and **CSI for infrared** rather than CSI for both.

---

## Checkpoints expected by the scripts

Depending on the script, the folder should contain one or both of these model files:

- `visdrone_rfdetr_nano_best_ema.pth`
- `dronevehicle_rfdetr_nano_best_ema.pth`

In general:

- the **RGB scripts** use `visdrone_rfdetr_nano_best_ema.pth`
- the **infrared / dual-model scripts** also use `dronevehicle_rfdetr_nano_best_ema.pth`

If the checkpoint files are missing, the scripts will fail at startup.

---

## Recommended project layout

A simple project layout on the Jetson looks like this:

```text
~/projects/jetson-rfdetr/
├── requirements.txt
├── trained_rfdetr_live_inference_usb.py
├── trained_rfdetr_live_inference.py
├── det_track_only_usb.py
├── det_track_still_camera_speed_estimations.py
├── det_track_still_camera_speed_multi_model.py
├── visdrone_rfdetr_nano_best_ema.pth
├── dronevehicle_rfdetr_nano_best_ema.pth
└── venv/
```

You can keep additional scripts in the same folder if needed, but this is the minimum structure that matches the runtime assumptions in the files.

## Creating the environment on Jetson

This setup assumes a **Jetson Orin Nano on JetPack 6.2.x / Ubuntu 22.04.**

### 1. Copy the project files onto the Jetson

Create a project folder and place your runtime files there:

- your Python scripts
- your checkpoint files
- your requirements.txt

Example:
```
mkdir -p ~/projects/jetson-rfdetr
cd ~/projects/jetson-rfdetr
```

Then copy your files there using SCP, USB, Git, or any other method you prefer.

At the end, `ls` should show at least:
```
requirements.txt
trained_rfdetr_live_inference_usb.py
visdrone_rfdetr_nano_best_ema.pth
dronevehicle_rfdetr_nano_best_ema.pth
```

### 2. Check the JetPack version

Run:
```
cat /etc/nv_tegra_release
```

You want this to match the expected JetPack 6.2.x-era environment.

### 3. Install system packages first

These packages cover:

- Python venv support
- OpenBLAS needed by Jetson PyTorch
- gi / GStreamer support
- webcam and GStreamer runtime pieces
```
sudo apt update
sudo apt install -y \
    python3-pip python3-venv libopenblas-dev \
    python3-gi python3-gi-cairo gir1.2-gstreamer-1.0 \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

### 4. Create the virtual environment the right way

Because these scripts use `gi`, create the venv with system packages visible:
```
cd ~/projects/jetson-rfdetr
python3 -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

The `--system-site-packages` part is important because it lets the venv see Ubuntu-installed packages like `python3-gi`.

## Install Jetson GPU PyTorch first

This is the most important step.

Do **not** install generic PyPI CPU wheels for torch and torchvision. On Jetson, that can leave you with a CPU-only environment, which makes the scripts much slower.

For a JetPack 6.2 / CUDA 12.6 environment, install Jetson-specific wheels first:
```
pip install torch torchvision torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu12.6.8/index
```

## Verify PyTorch is using the GPU

Before installing the rest of the dependencies, confirm that PyTorch is actually using CUDA:
```
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

You want output shaped like:

- a torch version without `+cpu`
- a CUDA version string
- `True`
- the Jetson GPU name

If this prints `False` or shows a CPU-only build, stop and fix PyTorch before continuing.

## Install the remaining Python requirements

Your `requirements.txt` should not be used to install generic `torch` and `torchvision` on Jetson.

Before using it, remove lines like:
```
torch==...
torchvision==...
```

Then install the rest:
```
pip install -r requirements.txt
```

Verify gi / GStreamer support

These scripts import:
```
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
```

So test that explicitly before running the app:
```
python -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print(Gst.version_string())"
```

If that works, your `gi` / GStreamer support is correctly available inside the environment.

## Run the scripts
### USB detection + tracking only
```
python det_track_only_usb.py
```

### USB detection + tracking + speed estimation
```
python det_track_still_camera_speed_estimations.py
```
### USB detection + tracking + speed estimation + model switching
```
python det_track_still_camera_speed_multi_model.py
```
### CSI camera live inference setup
```
python trained_rfdetr_live_inference.py
```
### Mixed USB RGB + CSI IR live inference setup
```
python trained_rfdetr_live_inference_usb.py
```

## Recommended install order every time

Use this order whenever you rebuild the environment:

1. copy scripts, checkpoint files, and `requirements.txt`
2. install apt/system dependencies
3. create the venv with `--system-site-packages`
4. install Jetson-specific `torch` / `torchvision` first
5. verify CUDA-enabled PyTorch
6. install remaining requirements
7. verify `gi`
8. run the desired script

This order reduces the chance of ending up with the wrong PyTorch build.

## If the scripts are slow

The first thing to check is whether PyTorch is actually running with CUDA.

Run:
```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

If it says `False` or shows a CPU-only build, the environment is wrong.

That is the most likely reason for major slowdown.

## Notes about cameras

These scripts use GStreamer pipelines rather than basic OpenCV camera capture.

Depending on the script, the source may be:

- USB webcam through `v4l2src`
- CSI camera through `nvarguscamerasrc`

Some scripts try MJPEG first for USB and then fall back to raw frames if needed. The CSI-based scripts are intended for the Jetson camera stack and use sensor IDs to select the connected camera.

That means the exact camera wiring and device names need to match the script configuration.

## What these scripts are for in the overall project

These Jetson scripts are not training scripts. They are the **deployment/runtime** side of the project.

They exist to answer questions like:

- can the trained RGB model run live on embedded hardware?
- can the infrared model run live on embedded hardware?
- can USB and CSI camera pipelines be supported cleanly?
- can tracking be overlaid in real time?
- can approximate speed estimation be added on-device?
- can the system switch between RGB and infrared models in a live application?

So this folder represents the transition from offline notebook training to **real-time on-device inference.**

## Conclusion

The Jetson runtime setup provides a practical embedded deployment environment for the Phase 3 RF-DETR models.

It supports:

- live GPU inference on Jetson
- RGB and infrared checkpoint loading
- USB and CSI camera pipelines
- ByteTrack-based object tracking
- optional speed estimation
- live visualization on-device

The most important environment detail is making sure the Jetson uses the **correct CUDA-enabled PyTorch build** and that the venv is created with `--system-site-packages` so the `gi` / GStreamer pieces remain available.