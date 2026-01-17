"""
CLI for exporting the model to TFLite format.

Usage:
    uv run mina-export [--weights PATH] [--no-int8] [--imgsz N] [--output-dir PATH]
"""

import argparse

from mina.export import export_tflite, get_weights_or_default
from mina.core.constants import DEFAULT_IMAGE_SIZE


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to TFLite format for mobile deployment"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained weights file (.pt). Auto-detects if not provided.",
    )
    parser.add_argument(
        "--no-int8",
        action="store_true",
        help="Disable int8 quantization (not recommended for mobile)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Input image size for the exported model (default: {DEFAULT_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the TFLite model",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Include NMS in the model (may fail due to onnx2tf TopK issues)",
    )

    args = parser.parse_args()

    weights_path = get_weights_or_default(args.weights)

    export_tflite(
        weights_path=weights_path,
        int8=not args.no_int8,
        imgsz=args.imgsz,
        output_dir=args.output_dir,
        nms=args.nms,
    )


if __name__ == "__main__":
    main()
