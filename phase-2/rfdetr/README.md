# Create and Activate Virtual Environment to run Jypter Notebook

```
python -m venv venv

.\venv\Scripts\activate     # Windows  
# or  
source venv/bin/activate    # macOS/Linux

pip install -r requirements.txt

python -m ipykernel install --user --name=rfdetr-env --display-name "Phase 2 RF-DETR Env"
jupyter lab
```

When using the notebook, selected "Phase 2 RF-DETR Env" as the kernel.