# Laptop Object Detection Project

Single-class object detection for **Laptops** using YOLOv8 on Open Images Dataset.

## 📋 Quick Overview

This project implements a complete object detection pipeline:
1. ✅ Dataset collection (1,500 images)
2. ✅ Data cleaning and enhancement
3. ✅ Model training (original + cleaned datasets)
4. ✅ Performance evaluation and comparison

## 🚀 Getting Started

The main project code is located in `imageDataset/OIDv4_ToolKit/`.

### Quick Start

```bash
cd imageDataset/OIDv4_ToolKit

# Install dependencies
pip install -r requirements.txt
pip install ultralytics

# Train on original dataset
python train_yolov8_laptop.py

# Train on cleaned dataset  
python train_yolov8_laptop_cleaned.py

# Compare results
python scripts/compare_metrics.py
```

## 📁 Project Structure

```
AI-IP-project_laptopDectector/
├── README.md                          ← You are here
├── .gitignore
│
└── imageDataset/
    └── OIDv4_ToolKit/                ← Main project directory
        ├── README.md                 ← Detailed project docs
        ├── PROJECT_WORKFLOW.md       ← Complete workflow guide
        ├── QUICK_START.md            ← Quick start guide
        ├── train_yolov8_laptop.py    ← Training script (original)
        ├── train_yolov8_laptop_cleaned.py  ← Training script (cleaned)
        ├── laptop.yaml               ← Dataset config (original)
        ├── laptop_cleaned.yaml       ← Dataset config (cleaned)
        ├── scripts/                  ← Utility scripts
        ├── modules/                  ← OID downloader modules
        ├── results/                  ← Text results
        └── requirements.txt          ← Dependencies
```

## 📊 Results

- **Original Model**: mAP@0.5 = 0.871
- **Cleaned Model**: mAP@0.5 = 0.835
- **Full comparison**: See `imageDataset/OIDv4_ToolKit/results/metrics_comparison.txt`

## 📖 Documentation

For detailed documentation, see:
- **Project README**: `imageDataset/OIDv4_ToolKit/README.md`
- **Workflow Guide**: `imageDataset/OIDv4_ToolKit/PROJECT_WORKFLOW.md`
- **Quick Start**: `imageDataset/OIDv4_ToolKit/QUICK_START.md`

## 🔧 Requirements

- Python 3.8+
- ultralytics
- pandas
- numpy
- opencv-python
- tqdm

See `imageDataset/OIDv4_ToolKit/requirements.txt` for full list.

## 📝 Notes

- Large datasets and images are excluded from git (see `.gitignore`)
- Model weights and training outputs are stored locally in `runs/` directory
- All paths in config files use relative paths for portability

