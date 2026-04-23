To get the dataset, go to https://github.com/VisDrone/DroneVehicle.

Download the `Train`, `Validation`, and `Test` datasets. You will have to translate the page to English, download an installer, create an account, and then download the data.

After downloading, organize the data like this:
```
visdrone-dataset-infrared/
├── train/
│   ├── trainimg/
│   ├── trainimgr/
│   ├── trainlabel/
│   └── trainlabelr/
├── val/
│   ├── valimg/
│   ├── valimgr/
│   ├── vallabel/
│   └── vallabelr/
└── test/
    ├── testimg/
    ├── testimgr/
    ├── testlabel/
    └── testlabelr/
```

- trainimg/, valimg/, testimg/ contain the RGB images
- trainimgr/, valimgr/, testimgr/ contain the infrared images
- trainlabel/, vallabel/, testlabel/ contain the RGB XML annotations
- trainlabelr/, vallabelr/, testlabelr/ contain the infrared XML annotations