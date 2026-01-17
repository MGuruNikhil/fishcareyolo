"""
Model Inference Testing Script

Tests inference on sample images and verifies output format matches
the expected Detection structure from the design document.

Usage:
    python test_inference.py [--weights PATH] [--image PATH] [--dir PATH]
"""

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image
from ultralytics import YOLO


# Disease classes matching the dataset
DISEASE_CLASSES = [
    "bacterial_infection",
    "fungal_infection",
    "healthy",
    "parasite",
    "white_tail",
]


class BoundingBox(NamedTuple):
    """Bounding box in normalized coordinates (0-1)."""

    x: float  # left edge
    y: float  # top edge
    width: float
    height: float


class Detection(NamedTuple):
    """A single disease detection from the model."""

    id: str
    disease_class: str
    confidence: float
    bounding_box: BoundingBox


def find_best_weights() -> Path | None:
    """Find the best.pt file from the most recent training run."""
    runs_dir = Path(__file__).parent / "runs" / "detect"

    if not runs_dir.exists():
        return None

    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        best_weights = run_dir / "weights" / "best.pt"
        if best_weights.exists():
            return best_weights

    return None


def convert_to_detections(results, min_confidence: float = 0.3) -> list[Detection]:
    """
    Convert YOLO results to Detection objects.

    The model outputs in center format [x_center, y_center, width, height].
    We convert to top-left format for the Detection structure.

    Args:
        results: YOLO inference results
        min_confidence: Minimum confidence threshold (default 0.3 per requirements)

    Returns:
        List of Detection objects sorted by confidence (descending)
    """
    detections = []
    detection_id = 0

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for i in range(len(boxes)):
            confidence = float(boxes.conf[i].item())

            # Filter by confidence threshold (Requirement 4.3)
            if confidence < min_confidence:
                continue

            class_id = int(boxes.cls[i].item())
            disease_class = DISEASE_CLASSES[class_id]

            # Get normalized bounding box (xyxy format -> convert to xywh)
            xyxy = boxes.xyxyn[i].tolist()
            x1, y1, x2, y2 = xyxy

            bounding_box = BoundingBox(
                x=x1,
                y=y1,
                width=x2 - x1,
                height=y2 - y1,
            )

            detection = Detection(
                id=f"det_{detection_id:03d}",
                disease_class=disease_class,
                confidence=confidence,
                bounding_box=bounding_box,
            )
            detections.append(detection)
            detection_id += 1

    # Sort by confidence descending (Requirement 3.3)
    detections.sort(key=lambda d: -d.confidence)

    return detections


def validate_detection(detection: Detection) -> list[str]:
    """
    Validate a detection against the expected structure.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    # Validate disease class
    if detection.disease_class not in DISEASE_CLASSES:
        errors.append(f"Invalid disease class: {detection.disease_class}")

    # Validate confidence
    if not (0.0 <= detection.confidence <= 1.0):
        errors.append(f"Confidence out of range: {detection.confidence}")

    # Validate bounding box
    bbox = detection.bounding_box
    if not (0.0 <= bbox.x <= 1.0):
        errors.append(f"BBox x out of range: {bbox.x}")
    if not (0.0 <= bbox.y <= 1.0):
        errors.append(f"BBox y out of range: {bbox.y}")
    if not (0.0 <= bbox.width <= 1.0):
        errors.append(f"BBox width out of range: {bbox.width}")
    if not (0.0 <= bbox.height <= 1.0):
        errors.append(f"BBox height out of range: {bbox.height}")

    # Validate bounding box doesn't exceed image bounds
    if bbox.x + bbox.width > 1.0 + 1e-6:
        errors.append(f"BBox exceeds right edge: x={bbox.x}, width={bbox.width}")
    if bbox.y + bbox.height > 1.0 + 1e-6:
        errors.append(f"BBox exceeds bottom edge: y={bbox.y}, height={bbox.height}")

    return errors


def run_inference(
    model: YOLO,
    image_path: Path,
    verbose: bool = True,
) -> list[Detection]:
    """
    Run inference on a single image.

    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        verbose: Whether to print results

    Returns:
        List of Detection objects
    """
    if verbose:
        print(f"\nProcessing: {image_path.name}")

    # Run inference
    results = model(str(image_path), verbose=False)

    # Convert to Detection objects
    detections = convert_to_detections(results)

    if verbose:
        if not detections:
            print("  No diseases detected (fish appears healthy)")
        else:
            print(f"  Found {len(detections)} detection(s):")
            for det in detections:
                print(f"    - {det.disease_class}: {det.confidence * 100:.1f}%")
                print(
                    f"      bbox: ({det.bounding_box.x:.3f}, {det.bounding_box.y:.3f}, "
                    f"{det.bounding_box.width:.3f}, {det.bounding_box.height:.3f})"
                )

    # Validate all detections
    for det in detections:
        errors = validate_detection(det)
        if errors:
            print(f"  WARNING: Invalid detection {det.id}:")
            for error in errors:
                print(f"    - {error}")

    return detections


def test_with_synthetic_image(model: YOLO, tmp_dir: Path) -> None:
    """Test inference with a synthetic image to verify pipeline works."""
    print("\n=== Testing with synthetic image ===")

    # Create a simple test image
    img_array = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    test_path = tmp_dir / "synthetic_test.jpg"
    img.save(test_path)

    detections = run_inference(model, test_path)
    print(f"  Pipeline working: {len(detections)} detection(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Test model inference on sample images"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (.pt or .tflite)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image to test",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to directory of images to test",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum confidence threshold (default: 0.3)",
    )

    args = parser.parse_args()

    # Find weights
    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = find_best_weights()
        if weights_path is None:
            print("Error: No weights file specified and no training runs found.")
            print("Please train a model first: python train.py")
            return
        print(f"Using weights: {weights_path}")

    # Load model
    print(f"Loading model from: {weights_path}")
    model = YOLO(str(weights_path))

    # Test with synthetic image first
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_with_synthetic_image(model, Path(tmp_dir))

    # Process specified image(s)
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return
        run_inference(model, image_path)

    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Directory not found: {dir_path}")
            return

        print(f"\n=== Processing images from: {dir_path} ===")
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [p for p in dir_path.iterdir() if p.suffix.lower() in image_extensions]

        if not images:
            print("No images found in directory")
            return

        all_detections = []
        for image_path in sorted(images)[:10]:  # Limit to first 10 for testing
            detections = run_inference(model, image_path)
            all_detections.extend(detections)

        print(f"\n=== Summary ===")
        print(f"Processed: {min(len(images), 10)} images")
        print(f"Total detections: {len(all_detections)}")

        # Count by class
        class_counts = {}
        for det in all_detections:
            class_counts[det.disease_class] = class_counts.get(det.disease_class, 0) + 1

        if class_counts:
            print("Detections by class:")
            for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
