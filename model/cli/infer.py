"""
CLI for running inference on images.

Usage:
    uv run mina-infer [--weights PATH] [--image PATH] [--dir PATH] [--confidence N]
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from mina.inference import run_inference, run_inference_on_directory
from mina.core.model import load_model, find_best_weights
from mina.core.constants import DEFAULT_CONFIDENCE_THRESHOLD


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
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Minimum confidence threshold (default: {DEFAULT_CONFIDENCE_THRESHOLD})",
    )

    args = parser.parse_args()

    # Find weights
    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"Error: Weights file not found: {weights_path}")
            return 1
    else:
        weights_path = find_best_weights()
        if weights_path is None:
            print("Error: No weights file specified and no training runs found.")
            print("Please train a model first: uv run mina-train")
            return 1
        print(f"Using weights: {weights_path}")

    # Load model
    print(f"Loading model from: {weights_path}")
    model = load_model(weights_path)

    # Test with synthetic image first
    print("\n=== Testing with synthetic image ===")
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_array = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        test_path = Path(tmp_dir) / "synthetic_test.jpg"
        img.save(test_path)

        detections = run_inference(model, test_path, args.confidence)
        print(f"  Pipeline working: {len(detections)} detection(s)")

    # Process specified image
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return 1
        run_inference(model, image_path, args.confidence)

    # Process directory
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Directory not found: {dir_path}")
            return 1
        run_inference_on_directory(model, dir_path, args.confidence)

    return 0


if __name__ == "__main__":
    exit(main())
