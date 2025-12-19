import csv
from pathlib import Path


def get_laptop_label_id(class_descriptions_csv: Path) -> str:
    """Return the Open Images label id string for class name 'Laptop'."""
    with class_descriptions_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                label_id, class_name = row[0], row[1]
                if class_name.lower() == "laptop":
                    return label_id
    raise RuntimeError("Could not find 'Laptop' class in class-descriptions-boxable.csv")


def generate_yolo_labels_for_laptop(
    annotations_csv: Path,
    laptop_label_id: str,
    images_dir: Path,
) -> None:
    """
    Create YOLO-format text label files for each Laptop bounding box.

    Open Images bbox CSV columns (subset):
        ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,...

    X/Y values are in [0,1] relative coordinates.
    YOLO format per line:
        <class_id> <x_center> <y_center> <width> <height>
    """

    images_dir = images_dir.resolve()
    image_stems = {p.stem for p in images_dir.glob("*.jpg")}
    if not image_stems:
        raise RuntimeError(f"No JPG images found in {images_dir}")

    labels_dir = images_dir  # labels will be created next to images

    # Clear any existing .txt labels for a clean start
    for txt_path in labels_dir.glob("*.txt"):
        txt_path.unlink()

    num_boxes = 0
    num_images = 0

    with annotations_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        current_image = None
        current_file = None

        for row in reader:
            if row.get("LabelName") != laptop_label_id:
                continue

            image_id = row["ImageID"]
            if image_id not in image_stems:
                continue  # annotation for image we did not download

            try:
                xmin = float(row["XMin"])
                xmax = float(row["XMax"])
                ymin = float(row["YMin"])
                ymax = float(row["YMax"])
            except (TypeError, ValueError):
                continue

            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin

            if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                continue

            if current_image != image_id:
                if current_file is not None:
                    current_file.close()
                label_path = labels_dir / f"{image_id}.txt"
                current_file = label_path.open("w", encoding="utf-8")
                current_image = image_id
                num_images += 1

            # Single class (Laptop) => class id 0
            current_file.write(
                f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )
            num_boxes += 1

        if current_file is not None:
            current_file.close()

    print(f"Generated labels for {num_images} images, total {num_boxes} boxes.")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    csv_root = project_root / "OID" / "csv_folder"
    class_csv = csv_root / "class-descriptions-boxable.csv"
    ann_csv = csv_root / "train-annotations-bbox.csv"
    images_dir = project_root / "OID" / "Dataset" / "train" / "Laptop"

    if not class_csv.exists():
        raise FileNotFoundError(f"Missing CSV: {class_csv}")
    if not ann_csv.exists():
        raise FileNotFoundError(f"Missing CSV: {ann_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    laptop_id = get_laptop_label_id(class_csv)
    print(f"Laptop label id: {laptop_id}")

    generate_yolo_labels_for_laptop(ann_csv, laptop_id, images_dir)


if __name__ == "__main__":
    main()


