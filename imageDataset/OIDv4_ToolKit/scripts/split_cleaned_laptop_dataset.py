"""
Split the cleaned Laptop dataset into train/val/test.
This is the same as split_laptop_dataset.py but for the cleaned dataset.
"""

from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_laptop_dataset import split_dataset


def main():
    project_root = Path(__file__).resolve().parent.parent
    src = project_root / "OID" / "Dataset" / "train" / "Laptop_cleaned"
    out = project_root / "OID" / "Dataset" / "Laptop_cleaned_splits"

    print("=" * 60)
    print("Splitting Cleaned Laptop Dataset")
    print("=" * 60)
    print(f"Source: {src}")
    print(f"Output: {out}")
    print()

    split_dataset(src, out)


if __name__ == "__main__":
    main()

