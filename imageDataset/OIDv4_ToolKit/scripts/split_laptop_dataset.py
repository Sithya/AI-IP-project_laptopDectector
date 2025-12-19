import os
import random
import shutil
from pathlib import Path


def split_dataset(
    source_dir: Path,
    output_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Split single-class OID dataset (images + txt labels) into train/val/test.

    Expected input structure:
        source_dir/
            *.jpg
            *.txt  (same stem as image)

    Output structure:
        output_root/
            train/
                images/
                labels/
            val/
                images/
                labels/
            test/
                images/
                labels/
    """

    assert 0 < train_ratio < 1 and 0 < val_ratio < 1, "Ratios must be in (0,1)"
    assert train_ratio + val_ratio < 1, "train_ratio + val_ratio must be < 1"

    source_dir = source_dir.resolve()
    output_root = output_root.resolve()

    image_paths = sorted(
        [p for p in source_dir.glob("*.jpg")] + [p for p in source_dir.glob("*.jpeg")]
    )
    if not image_paths:
        raise RuntimeError(f"No JPG/JPEG images found in {source_dir}")

    # Only keep images that have a matching txt label
    valid_images = []
    for img_path in image_paths:
        label_path = img_path.with_suffix(".txt")
        if label_path.exists():
            valid_images.append((img_path, label_path))

    if not valid_images:
        raise RuntimeError(f"No image/label pairs found in {source_dir}")

    random.seed(seed)
    random.shuffle(valid_images)

    n_total = len(valid_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_set = valid_images[:n_train]
    val_set = valid_images[n_train : n_train + n_val]
    test_set = valid_images[n_train + n_val :]

    print(f"Total pairs: {n_total}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    def prepare_split(split_name: str, pairs):
        img_out = output_root / split_name / "images"
        lbl_out = output_root / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_p, lbl_p in pairs:
            shutil.copy2(img_p, img_out / img_p.name)
            shutil.copy2(lbl_p, lbl_out / lbl_p.name)

    prepare_split("train", train_set)
    prepare_split("val", val_set)
    prepare_split("test", test_set)

    print(f"Split completed under: {output_root}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    src = project_root / "OID" / "Dataset" / "train" / "Laptop"
    out = project_root / "OID" / "Dataset" / "Laptop_splits"

    split_dataset(src, out)


