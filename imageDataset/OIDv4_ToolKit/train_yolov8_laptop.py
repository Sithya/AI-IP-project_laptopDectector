from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parent

    # Data config (points to Laptop_splits folders)
    data_yaml = project_root / "laptop.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")

    # Start from a small pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(project_root / "runs"),
        name="yolov8n-laptop",
    )

    # Evaluate on test set
    model.val(data=str(data_yaml), split="test", imgsz=640, batch=16)


if __name__ == "__main__":
    main()


