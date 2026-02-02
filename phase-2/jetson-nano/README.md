# Create and Activate Virtual Environment to run Jypter Notebook

```
python -m venv venv

.\venv\Scripts\activate     # if you are testing this on a different machine and the OS is Windows  
# or  
source venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
```

You can now run scripts inside this virtual environment.

# Get Trained Model

Go to this part of our project's Google Drive:
```
CPEN/ELEC 491 - Capstone / Milestone 3 / trained_rfdetr
```

You will see 3 `.pth` files that were too large to push to GitHub:
```
visdrone_rfdetr_nano_best_ema.pth
visdrone_rfdetr_nano_best_regular.pth
visdrone_rfdetr_nano_latest_resume.pth
```

Copy `visdrone_rfdetr_nano_best_ema.pth` into this directory so that you can use our trained RF-DETR Nano.