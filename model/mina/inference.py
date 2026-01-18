"""
Inference logic for fish disease detection.
"""

from pathlib import Path

from ultralytics import YOLO

from mina.core.constants import (
    DISEASE_CLASSES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    IMAGE_EXTENSIONS,
)
from mina.core.types import BoundingBox, Detection


def convert_to_detections(
    results,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[Detection]:
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

            # Filter by confidence threshold
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

    # Sort by confidence descending
    detections.sort(key=lambda d: -d.confidence)

    return detections


def run_inference(
    model: YOLO,
    image_path: Path,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
    verbose: bool = True,
) -> list[Detection]:
    """
    Run inference on a single image.

    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        min_confidence: Minimum confidence threshold
        verbose: Whether to print results

    Returns:
        List of Detection objects
    """
    if verbose:
        print(f"\nProcessing: {image_path.name}")

    # Run inference
    results = model(str(image_path), verbose=False)

    # Convert to Detection objects
    detections = convert_to_detections(results, min_confidence)

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
        errors = det.validate()
        if errors:
            print(f"  WARNING: Invalid detection {det.id}:")
            for error in errors:
                print(f"    - {error}")

    return detections


def run_inference_on_directory(
    model: YOLO,
    dir_path: Path,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
    limit: int | None = 10,
    verbose: bool = True,
) -> list[Detection]:
    """
    Run inference on all images in a directory.

    Args:
        model: Loaded YOLO model
        dir_path: Path to directory containing images
        min_confidence: Minimum confidence threshold
        limit: Maximum number of images to process (None for all)
        verbose: Whether to print results

    Returns:
        List of all Detection objects from all images
    """
    images = [p for p in dir_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]

    if not images:
        print("No images found in directory")
        return []

    if verbose:
        print(f"\n=== Processing images from: {dir_path} ===")

    images_to_process = sorted(images)[:limit] if limit else sorted(images)

    all_detections = []
    for image_path in images_to_process:
        detections = run_inference(model, image_path, min_confidence, verbose)
        all_detections.extend(detections)

    if verbose:
        print("\n=== Summary ===")
        print(f"Processed: {len(images_to_process)} images")
        print(f"Total detections: {len(all_detections)}")

        # Count by class
        class_counts: dict[str, int] = {}
        for det in all_detections:
            class_counts[det.disease_class] = class_counts.get(det.disease_class, 0) + 1

        if class_counts:
            print("Detections by class:")
            for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                print(f"  {cls}: {count}")

    return all_detections
