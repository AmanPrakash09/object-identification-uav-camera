# EfficientDet Model Generation & Jupyter Setup

This folder is used to create and export an EfficientDet model (PyTorch + ONNX)
using a clean virtual environment and a Jupyter notebook.

## 1. Create and activate virtual environment

Open a terminal and run:

```bash
cd /Users/g/Documents/CPEN491/object-identification-uav-camera/phase-1/efficientdet-evaluation

python3 -m venv venv

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

# (On Windows, use: .\venv\Scripts\activate)

python -m ipykernel install --user --name efficientdet-env --display-name "EfficientDet Env"
jupyter lab
