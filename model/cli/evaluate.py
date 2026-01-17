"""
CLI for evaluating the model on the test set.

Usage:
    uv run mina-evaluate --weights PATH [--test-dir PATH] [--imgsz N]
"""

import argparse
from pathlib import Path

from mina.evaluate import evaluate, print_evaluation_results
from mina.core.constants import DEFAULT_IMAGE_SIZE, DEFAULT_IOU_THRESHOLD, TEST_DATA_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on held-out test set"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (.pt or .tflite)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=str(TEST_DATA_DIR),
        help=f"Path to test data directory (default: {TEST_DATA_DIR})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Input image size (default: {DEFAULT_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.001,
        help="Confidence threshold (default: 0.001, low for mAP calculation)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"IoU threshold for NMS (default: {DEFAULT_IOU_THRESHOLD})",
    )

    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}\n"
            "Make sure you ran 'uv run mina-download' to download the dataset."
        )

    metrics = evaluate(
        weights=weights_path,
        test_dir=test_dir,
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
    )

    print_evaluation_results(metrics)

    # Return non-zero exit code if mAP is very low
    if metrics["mAP50"] < 0.1:
        print("\nWARNING: mAP@50 is below 0.1 - model may have issues.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
