"""
Data cleaning script for Laptop dataset.

This script performs:
1. Remove duplicate images (using perceptual hashing)
2. Remove images without label files
3. Remove images with invalid bounding boxes
4. Filter very dark/bright images
5. Optional: Apply image enhancement (brightness/contrast, sharpening)
"""

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def calculate_image_hash(image_path: Path) -> str:
    """Calculate MD5 hash of image file for duplicate detection."""
    with image_path.open("rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def calculate_brightness(image_path: Path) -> float:
    """
    Calculate average brightness of image (0-255 scale).
    Returns mean pixel value in grayscale.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def validate_bounding_boxes(label_path: Path) -> bool:
    """
    Validate that all bounding boxes in label file are valid.
    Returns True if all boxes are valid, False otherwise.
    """
    if not label_path.exists():
        return False
    
    try:
        with label_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if not lines:
            return False  # Empty label file
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                return False
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Check if coordinates are in valid range [0, 1]
                if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                    return False
                if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    return False
                
                # Check if box is within image bounds
                x_min = x_center - width / 2.0
                x_max = x_center + width / 2.0
                y_min = y_center - height / 2.0
                y_max = y_center + height / 2.0
                
                if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
                    return False
                    
            except (ValueError, IndexError):
                return False
        
        return True
        
    except Exception:
        return False


def enhance_image(image_path: Path, output_path: Path, 
                  brightness_factor: float = 1.1,
                  contrast_factor: float = 1.1,
                  sharpen: bool = True) -> bool:
    """
    Apply image enhancement: brightness, contrast, and optional sharpening.
    
    Args:
        image_path: Input image path
        output_path: Output image path
        brightness_factor: Brightness multiplier (1.0 = no change, >1.0 = brighter)
        contrast_factor: Contrast multiplier (1.0 = no change, >1.0 = more contrast)
        sharpen: Whether to apply sharpening filter
    
    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Apply brightness enhancement
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
        
        # Apply contrast enhancement
        if contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
        
        # Apply sharpening
        if sharpen:
            img = img.filter(ImageFilter.SHARPEN)
        
        # Save enhanced image
        img.save(output_path, quality=95)
        return True
        
    except Exception as e:
        print(f"  Warning: Failed to enhance {image_path.name}: {e}")
        return False


def clean_dataset(
    source_dir: Path,
    output_dir: Path,
    min_brightness: float = 30.0,
    max_brightness: float = 220.0,
    apply_enhancement: bool = False,
    brightness_factor: float = 1.1,
    contrast_factor: float = 1.1,
    sharpen: bool = True,
) -> Dict[str, int]:
    """
    Clean the dataset by removing duplicates, invalid images, and optionally enhancing.
    
    Args:
        source_dir: Source directory containing images and labels
        output_dir: Output directory for cleaned dataset
        min_brightness: Minimum acceptable brightness (0-255)
        max_brightness: Maximum acceptable brightness (0-255)
        apply_enhancement: Whether to apply image enhancement
        brightness_factor: Brightness enhancement factor
        contrast_factor: Contrast enhancement factor
        sharpen: Whether to apply sharpening
    
    Returns:
        Dictionary with cleaning statistics
    """
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in source_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    stats = {
        "total_images": len(image_files),
        "duplicates_removed": 0,
        "no_label_removed": 0,
        "invalid_bbox_removed": 0,
        "too_dark_removed": 0,
        "too_bright_removed": 0,
        "kept": 0,
        "enhanced": 0,
    }
    
    # Track hashes to detect duplicates
    seen_hashes: Dict[str, Path] = {}
    
    # Process each image
    for img_path in image_files:
        label_path = source_dir / f"{img_path.stem}.txt"
        
        # Check 1: Remove if no label file
        if not label_path.exists():
            stats["no_label_removed"] += 1
            print(f"  Removing {img_path.name}: no label file")
            continue
        
        # Check 2: Validate bounding boxes
        if not validate_bounding_boxes(label_path):
            stats["invalid_bbox_removed"] += 1
            print(f"  Removing {img_path.name}: invalid bounding boxes")
            continue
        
        # Check 3: Check for duplicates
        img_hash = calculate_image_hash(img_path)
        if img_hash in seen_hashes:
            stats["duplicates_removed"] += 1
            print(f"  Removing {img_path.name}: duplicate of {seen_hashes[img_hash].name}")
            continue
        seen_hashes[img_hash] = img_path
        
        # Check 4: Check brightness
        try:
            brightness = calculate_brightness(img_path)
            if brightness < min_brightness:
                stats["too_dark_removed"] += 1
                print(f"  Removing {img_path.name}: too dark (brightness={brightness:.1f})")
                continue
            if brightness > max_brightness:
                stats["too_bright_removed"] += 1
                print(f"  Removing {img_path.name}: too bright (brightness={brightness:.1f})")
                continue
        except Exception as e:
            print(f"  Warning: Could not check brightness for {img_path.name}: {e}")
            # Continue anyway - don't remove if we can't check
        
        # Image passed all checks - copy or enhance it
        output_img_path = output_dir / img_path.name
        output_label_path = output_dir / label_path.name
        
        if apply_enhancement:
            if enhance_image(img_path, output_img_path, brightness_factor, 
                           contrast_factor, sharpen):
                stats["enhanced"] += 1
            else:
                # Fallback to simple copy if enhancement fails
                shutil.copy2(img_path, output_img_path)
        else:
            shutil.copy2(img_path, output_img_path)
        
        # Copy label file
        shutil.copy2(label_path, output_label_path)
        stats["kept"] += 1
        
        if stats["kept"] % 100 == 0:
            print(f"  Processed {stats['kept']} images...")
    
    return stats


def main():
    """Main function to run data cleaning."""
    parser = argparse.ArgumentParser(description="Clean Laptop dataset")
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Apply brightness/contrast adjustment and sharpening",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=1.1,
        help="Brightness factor (default: 1.1, >1.0 = brighter)",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.1,
        help="Contrast factor (default: 1.1, >1.0 = more contrast)",
    )
    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Disable sharpening (only applies if --enhance is used)",
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent
    source_dir = project_root / "OID" / "Dataset" / "train" / "Laptop"
    output_dir = project_root / "OID" / "Dataset" / "train" / "Laptop_cleaned"
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    print("=" * 60)
    print("Laptop Dataset Cleaning")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print()
    
    apply_enh = args.enhance
    brightness_factor = args.brightness
    contrast_factor = args.contrast
    sharpen = not args.no_sharpen
    
    if apply_enh:
        print(f"Enhancement enabled:")
        print(f"  - Brightness factor: {brightness_factor}")
        print(f"  - Contrast factor: {contrast_factor}")
        print(f"  - Sharpening: {sharpen}")
    else:
        print("Enhancement disabled (use --enhance to enable)")
    
    print()
    print("Starting cleaning process...")
    print()
    
    # Run cleaning
    stats = clean_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        min_brightness=30.0,
        max_brightness=220.0,
        apply_enhancement=apply_enh,
        brightness_factor=brightness_factor,
        contrast_factor=contrast_factor,
        sharpen=sharpen,
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("Cleaning Summary")
    print("=" * 60)
    print(f"Total images processed: {stats['total_images']}")
    print(f"  ✓ Kept: {stats['kept']}")
    print(f"  ✗ Removed:")
    print(f"    - Duplicates: {stats['duplicates_removed']}")
    print(f"    - No label file: {stats['no_label_removed']}")
    print(f"    - Invalid bounding boxes: {stats['invalid_bbox_removed']}")
    print(f"    - Too dark: {stats['too_dark_removed']}")
    print(f"    - Too bright: {stats['too_bright_removed']}")
    if stats['enhanced'] > 0:
        print(f"  ✨ Enhanced: {stats['enhanced']}")
    print()
    print(f"Cleaned dataset saved to: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Review the cleaned dataset")
    print("  2. Run split_laptop_dataset.py on the cleaned dataset")
    print("  3. Retrain the model on cleaned data")
    print("  4. Compare performance: original vs cleaned")


if __name__ == "__main__":
    main()

