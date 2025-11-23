# Create and Activate Virtual Environment to run Jypter Notebook

```
python -m venv venv

.\venv\Scripts\activate     # Windows  
# or  
source venv/bin/activate    # macOS/Linux

pip install -r requirements.txt

python -m ipykernel install --user --name=opencv-env --display-name "OpenCV Env"
jupyter lab
```

When using the notebook, selected "OpenCV Env" as the kernel.