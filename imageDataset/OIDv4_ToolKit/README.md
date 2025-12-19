# Laptop Object Detection Project

Single-class object detection for **Laptop** using YOLOv8 on Open Images Dataset.

## 📋 Quick Overview

This project implements a complete object detection pipeline:
1. ✅ Dataset collection (1,500 images)
2. ✅ Data cleaning and enhancement
3. ✅ Model training (original + cleaned datasets)
4. ✅ Performance evaluation and comparison

## 🚀 Main Workflow

See **[PROJECT_WORKFLOW.md](PROJECT_WORKFLOW.md)** for complete workflow guide.

### Essential Commands:

```bash
# Train on original dataset
python train_yolov8_laptop.py

# Train on cleaned dataset  
python train_yolov8_laptop_cleaned.py

# Compare results
python scripts/compare_metrics.py
```

## 📁 Project Structure

- **Main scripts**: `train_yolov8_laptop.py`, `train_yolov8_laptop_cleaned.py`
- **Helper scripts**: `scripts/` folder
- **Config files**: `laptop.yaml`, `laptop_cleaned.yaml`
- **Results**: `results/` folder
- **Dataset**: `OID/Dataset/` folder
- **Trained models**: `runs/` folder

## 📊 Results

- **Original Model**: mAP@0.5 = 0.871
- **Cleaned Model**: mAP@0.5 = 0.835
- **Full comparison**: See `results/metrics_comparison.txt`

## 📖 Documentation

- **Workflow Guide**: `PROJECT_WORKFLOW.md`
- **Metrics Comparison**: `results/metrics_comparison.txt`
