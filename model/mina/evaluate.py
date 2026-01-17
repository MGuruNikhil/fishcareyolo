"""
Model evaluation logic for fish disease detection.
"""

from pathlib import Path

from ultralytics import YOLO

from mina.core.constants import (
    DISEASE_CLASSES,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    TEST_DATA_DIR,
)
from mina.core.dataset import create_test_yaml


def evaluate(
    weights: Path,
    test_dir: Path | None = None,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    confidence: float = 0.001,
    iou: float = DEFAULT_IOU_THRESHOLD,
) -> dict:
    """
    Evaluate model on test set.

    Args:
        weights: Path to trained model weights (.pt or .tflite)
        test_dir: Path to test data directory. Defaults to TEST_DATA_DIR.
        imgsz: Input image size
        confidence: Confidence threshold for predictions (low for mAP)
        iou: IoU threshold for NMS

    Returns:
        Dictionary containing evaluation metrics

    Raises:
        FileNotFoundError: If test directory or subdirectories not found
        ValueError: If no images found in test directory
    """
    if test_dir is None:
        test_dir = TEST_DATA_DIR

    # Validate test directory structure
    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Test labels directory not found: {labels_dir}")

    image_count = len(list(images_dir.glob("*")))
    if image_count == 0:
        raise ValueError(f"No images found in {images_dir}")

    print(f"Evaluating on {image_count} test images...")

    # Create temporary yaml for evaluation
    test_yaml = create_test_yaml(test_dir)

    try:
        # Load model
        model = YOLO(str(weights))

        # Run validation on test set
        results = model.val(
            data=str(test_yaml),
            imgsz=imgsz,
            conf=confidence,
            iou=iou,
            split="val",  # We pointed 'val' to test images in yaml
            verbose=True,
        )

        # Extract metrics
        metrics = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
            "per_class_ap50": {
                DISEASE_CLASSES[i]: results.box.ap50[i]
                for i in range(len(DISEASE_CLASSES))
                if i < len(results.box.ap50)
            },
        }
    finally:
        # Clean up temporary yaml
        test_yaml.unlink(missing_ok=True)

    return metrics


def print_evaluation_results(metrics: dict) -> None:
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 40)
    print(f"{'mAP@50':<25} {metrics['mAP50']:>15.4f}")
    print(f"{'mAP@50-95':<25} {metrics['mAP50-95']:>15.4f}")
    print(f"{'Precision':<25} {metrics['precision']:>15.4f}")
    print(f"{'Recall':<25} {metrics['recall']:>15.4f}")

    print(f"\n{'Per-Class AP@50':<25}")
    print("-" * 40)
    for cls_name, ap in metrics["per_class_ap50"].items():
        print(f"  {cls_name:<23} {ap:>15.4f}")

    print("=" * 60)
