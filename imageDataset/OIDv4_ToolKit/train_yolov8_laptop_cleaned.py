"""
Train YOLOv8 on cleaned Laptop dataset.

This script trains the model on the cleaned dataset and evaluates on test set.
Use this to compare performance: original dataset vs cleaned dataset.
"""

from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parent

    # Data config (points to Laptop_cleaned_splits folders)
    data_yaml = project_root / "laptop_cleaned.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")

    # Start from a small pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train on cleaned dataset
    print("=" * 60)
    print("Training YOLOv8 on CLEANED Laptop Dataset")
    print("=" * 60)
    print(f"Data config: {data_yaml}")
    print()

    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(project_root / "runs"),
        name="yolov8n-laptop-cleaned",  # Different name to distinguish from original
    )

    # Evaluate on test set
    print()
    print("=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    model.val(data=str(data_yaml), split="test", imgsz=640, batch=16)

    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model saved to: runs/yolov8n-laptop-cleaned/weights/best.pt")
    print()
    print("Compare metrics:")
    print("  - Original dataset: runs/yolov8n-laptop/")
    print("  - Cleaned dataset:  runs/yolov8n-laptop-cleaned/")


if __name__ == "__main__":
    main()

