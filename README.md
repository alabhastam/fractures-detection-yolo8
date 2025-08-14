# 🦴 Bone Fracture Detection with YOLOv8

This project implements an **object detection** model using YOLOv8 to identify and localize **bone fractures** in medical images such as X-rays and MRI scans.

## 📂 Dataset
We use the **Human Bone Fractures Multi-modal Image Dataset (HBFMID)** in YOLO format with the following structure:

dataset/
│── train/
│   ├── images/
│   └── labels/
│── valid/
│   ├── images/
│   └── labels/
│── test/
│   ├── images/
│   └── labels/
└── data.yaml


- Each image file in `images/` has a corresponding `.txt` annotation file in `labels/`.
- Annotation format (normalized YOLO coordinates):
  
  class_id x_center y_center width height
 

## 🚀 Installation & Usage

### 1. Install Dependencies

pip install ultralytics


### 2. Train the Model

from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8s.pt')

# Train
model.train(
    data='path/to/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)


### 3. Validate the Model
metrics = model.val()
print(metrics)


### 4. Run Predictions

results = model.predict(
    source='path/to/test/images',
    conf=0.25
)

Outputs will be saved in the `runs/detect/predict` folder.

## 📊 Metrics
- Precision, Recall, mAP
- Class-wise metrics for each label

## 🎯 Features
- Supports multiple medical imaging types (X-ray, MRI, etc.)
- Can detect multiple fractures per image
- Real-time capable on GPU



## 📄 License
MIT License — free to use and modify
