# Laptop Object Detection Project - Workflow Guide

## 📋 Project Overview
Single-class object detection for **Laptop** using YOLOv8 on Open Images Dataset.

## 🔄 Complete Workflow

### Step 1: Dataset Collection ✅
- **Dataset**: 1,500 Laptop images from Open Images Dataset
- **Location**: `OID/Dataset/train/Laptop/`
- **Status**: Completed

### Step 2: Data Cleaning ✅

#### 2.1 Remove Bad Data
- **Script**: `scripts/clean_laptop_dataset.py`
- **Command**: `python scripts/clean_laptop_dataset.py --enhance`
- **Results**:
  - Removed 136 images with invalid bounding boxes
  - Removed 8 too dark images
  - Removed 5 too bright images
  - Removed 0 duplicates
  - **Final**: 1,351 clean images

#### 2.2 Image Enhancement
- Applied brightness adjustment (factor: 1.1)
- Applied contrast adjustment (factor: 1.1)
- Applied sharpening filter
- **Output**: `OID/Dataset/train/Laptop_cleaned/`

#### 2.3 Validate Bounding Boxes
- **Script**: `scripts/validate_bboxes.py`
- **Samples**: `results/validation_samples/`
- **Status**: Visual validation samples generated

### Step 3: Dataset Splitting ✅

#### Original Dataset Split
- **Script**: `scripts/split_laptop_dataset.py`
- **Output**: `OID/Dataset/Laptop_splits/`
- **Split**: Train (1,050) / Val (225) / Test (225)

#### Cleaned Dataset Split
- **Script**: `scripts/split_cleaned_laptop_dataset.py`
- **Output**: `OID/Dataset/Laptop_cleaned_splits/`
- **Split**: Train (945) / Val (202) / Test (204)

### Step 4: Model Training ✅

#### 4.1 Train on Original Dataset
- **Script**: `train_yolov8_laptop.py`
- **Config**: `laptop.yaml`
- **Command**: `python train_yolov8_laptop.py`
- **Results**: `runs/yolov8n-laptop/`
- **Metrics**: Precision=0.911, Recall=0.783, mAP@0.5=0.871

#### 4.2 Train on Cleaned Dataset
- **Script**: `train_yolov8_laptop_cleaned.py`
- **Config**: `laptop_cleaned.yaml`
- **Command**: `python train_yolov8_laptop_cleaned.py`
- **Results**: `runs/yolov8n-laptop-cleaned/`
- **Metrics**: Precision=0.885, Recall=0.745, mAP@0.5=0.835

### Step 5: Model Evaluation ✅
- **Script**: `scripts/compare_metrics.py`
- **Command**: `python scripts/compare_metrics.py`
- **Output**: `results/metrics_comparison.txt`
- **Status**: Comparison report generated

## 📁 Project Structure

```
OIDv4_ToolKit/
├── train_yolov8_laptop.py          # Main: Train on original dataset
├── train_yolov8_laptop_cleaned.py # Main: Train on cleaned dataset
├── laptop.yaml                     # Config: Original dataset
├── laptop_cleaned.yaml            # Config: Cleaned dataset
│
├── scripts/                       # Helper scripts
│   ├── clean_laptop_dataset.py
│   ├── split_laptop_dataset.py
│   ├── split_cleaned_laptop_dataset.py
│   ├── make_laptop_labels.py
│   ├── validate_bboxes.py
│   └── compare_metrics.py
│
├── results/                        # Results and outputs
│   ├── metrics_comparison.txt
│   └── validation_samples/
│
├── OID/                            # Dataset
│   └── Dataset/
│       ├── train/Laptop/          # Original dataset
│       ├── train/Laptop_cleaned/   # Cleaned dataset
│       ├── Laptop_splits/         # Original splits
│       └── Laptop_cleaned_splits/  # Cleaned splits
│
└── runs/                           # Training results
    ├── yolov8n-laptop/            # Original model
    └── yolov8n-laptop-cleaned/    # Cleaned model
```

## 🚀 Quick Start (For Teacher/Demo)

### Show the Complete Workflow:

1. **Dataset Collection**
   ```bash
   ls OID/Dataset/train/Laptop/ | wc -l  # Show 1,500 images
   ```

2. **Data Cleaning**
   ```bash
   python scripts/clean_laptop_dataset.py --enhance
   # Show: Removed 149 images, kept 1,351
   ```

3. **Training Original Model**
   ```bash
   python train_yolov8_laptop.py
   # Results in: runs/yolov8n-laptop/
   ```

4. **Training Cleaned Model**
   ```bash
   python train_yolov8_laptop_cleaned.py
   # Results in: runs/yolov8n-laptop-cleaned/
   ```

5. **Compare Results**
   ```bash
   python scripts/compare_metrics.py
   cat results/metrics_comparison.txt
   ```

### Test the Model:
```bash
# Test on your own image
yolo detect predict model=runs/yolov8n-laptop/weights/best.pt source=my_images/image.png
```

## 📊 Key Results

| Metric | Original Dataset | Cleaned Dataset | Difference |
|--------|-----------------|-----------------|------------|
| Precision | 0.911 | 0.885 | -2.9% |
| Recall | 0.783 | 0.745 | -4.9% |
| mAP@0.5 | 0.871 | 0.835 | -4.1% |
| mAP@0.5:0.95 | 0.594 | 0.535 | -9.9% |

## 📝 Notes

- Both models perform well (mAP@0.5 > 0.83)
- Data cleaning removed 149 problematic images
- Slight performance decrease due to smaller dataset size
- All images enhanced with brightness/contrast/sharpening

## 🔍 Files to Show Teacher

1. **Main Workflow**: `train_yolov8_laptop.py`, `train_yolov8_laptop_cleaned.py`
2. **Results**: `results/metrics_comparison.txt`
3. **Validation**: `results/validation_samples/` (sample images with boxes)
4. **Trained Models**: `runs/yolov8n-laptop/weights/best.pt`, `runs/yolov8n-laptop-cleaned/weights/best.pt`

