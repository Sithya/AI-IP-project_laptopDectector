"""
Visual validation script for bounding boxes.

This script helps you manually check if bounding boxes correctly label laptops.
It displays images with bounding boxes drawn so you can verify they're correct.
"""

import cv2
from pathlib import Path


def draw_bboxes_on_image(image_path: Path, label_path: Path, output_path: Path = None):
    """
    Draw bounding boxes on an image from YOLO label file.
    
    Args:
        image_path: Path to image file
        label_path: Path to YOLO label file (.txt)
        output_path: Optional path to save annotated image
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Warning: Could not read {image_path.name}")
        return None
    
    h, w = img.shape[:2]
    
    # Read label file
    if not label_path.exists():
        print(f"  Warning: No label file for {image_path.name}")
        return None
    
    boxes_drawn = 0
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert from normalized to pixel coordinates
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, "Laptop", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                boxes_drawn += 1
            except (ValueError, IndexError):
                continue
    
    if output_path:
        cv2.imwrite(str(output_path), img)
    
    return img, boxes_drawn


def validate_sample_images(dataset_dir: Path, num_samples: int = 20):
    """
    Validate a sample of images to check if bounding boxes are correct.
    
    Args:
        dataset_dir: Directory containing images and labels
        num_samples: Number of random images to check
    """
    import random
    
    # Find all image files
    image_files = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.jpeg"))
    if not image_files:
        print(f"No images found in {dataset_dir}")
        return
    
    # Randomly sample
    random.seed(42)
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    print("=" * 60)
    print(f"Validating {len(sample_images)} sample images")
    print("=" * 60)
    print("Instructions:")
    print("  - Images will be displayed with bounding boxes drawn")
    print("  - Press 'y' if boxes correctly label laptops")
    print("  - Press 'n' if boxes are incorrect")
    print("  - Press 'q' to quit")
    print()
    
    correct_count = 0
    incorrect_count = 0
    
    for img_path in sample_images:
        label_path = dataset_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        
        # Draw boxes
        result = draw_bboxes_on_image(img_path, label_path)
        if result is None:
            continue
        
        img, boxes_drawn = result
        
        if boxes_drawn == 0:
            print(f"  {img_path.name}: No boxes found")
            continue
        
        # Display image
        cv2.imshow(f"Validation: {img_path.name} ({boxes_drawn} boxes)", img)
        
        print(f"  {img_path.name}: {boxes_drawn} box(es) - Correct? (y/n/q): ", end="")
        
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord('q'):
            break
        elif key == ord('y'):
            correct_count += 1
            print("✓ Correct")
        elif key == ord('n'):
            incorrect_count += 1
            print("✗ Incorrect")
        else:
            print("Skipped")
    
    print()
    print("=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"Checked: {correct_count + incorrect_count} images")
    print(f"  ✓ Correct: {correct_count}")
    print(f"  ✗ Incorrect: {incorrect_count}")
    if correct_count + incorrect_count > 0:
        accuracy = (correct_count / (correct_count + incorrect_count)) * 100
        print(f"  Accuracy: {accuracy:.1f}%")


def save_validation_samples(dataset_dir: Path, output_dir: Path, num_samples: int = 10):
    """
    Save sample images with bounding boxes drawn for manual review.
    
    Args:
        dataset_dir: Directory containing images and labels
        output_dir: Directory to save annotated images
        num_samples: Number of random images to save
    """
    import random
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.jpeg"))
    if not image_files:
        print(f"No images found in {dataset_dir}")
        return
    
    # Randomly sample
    random.seed(42)
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"Saving {len(sample_images)} validation samples to {output_dir}")
    
    for img_path in sample_images:
        label_path = dataset_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        
        output_path = output_dir / f"{img_path.stem}_bboxes.jpg"
        draw_bboxes_on_image(img_path, label_path, output_path)
    
    print(f"Saved validation samples to: {output_dir}")


def main():
    """Main function."""
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parent
    
    # You can validate either original or cleaned dataset
    print("Which dataset to validate?")
    print("  1. Original dataset (Laptop)")
    print("  2. Cleaned dataset (Laptop_cleaned)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        dataset_dir = project_root / "OID" / "Dataset" / "train" / "Laptop"
    elif choice == "2":
        dataset_dir = project_root / "OID" / "Dataset" / "train" / "Laptop_cleaned"
    else:
        print("Invalid choice")
        return
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return
    
    print()
    print("Validation mode:")
    print("  1. Interactive (display images, press y/n)")
    print("  2. Save samples (save images with boxes for manual review)")
    mode = input("Enter choice (1 or 2): ").strip()
    
    if mode == "1":
        num_samples = int(input("Number of samples to check (default 20): ") or "20")
        validate_sample_images(dataset_dir, num_samples)
    elif mode == "2":
        output_dir = project_root / "validation_samples"
        num_samples = int(input("Number of samples to save (default 10): ") or "10")
        save_validation_samples(dataset_dir, output_dir, num_samples)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()

