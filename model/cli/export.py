"""
CLI for exporting the model to TFLite or TFJS format.

Usage:
    uv run mina-export [--weights PATH] [--no-int8] [--imgsz N] [--output-dir PATH]
    uv run mina-export --tfjs [--weights PATH] [--imgsz N] [--output-dir PATH] [--nms]
"""

import argparse

from mina.export import export_tflite, export_tfjs, get_weights_or_default
from mina.core.constants import DEFAULT_IMAGE_SIZE


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to TFLite or TFJS format"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained weights file (.pt). Auto-detects if not provided.",
    )
    parser.add_argument(
        "--tfjs",
        action="store_true",
        help="Export to TensorFlow.js format instead of TFLite",
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
        help="Output directory for the exported model",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Include NMS in the model (enabled by default for TFJS)",
    )

    args = parser.parse_args()

    weights_path = get_weights_or_default(args.weights)

    if args.tfjs:
        export_tfjs(
            weights_path=weights_path,
            imgsz=args.imgsz,
            output_dir=args.output_dir,
            nms=args.nms,
        )
    else:
        export_tflite(
            weights_path=weights_path,
            int8=not args.no_int8,
            imgsz=args.imgsz,
            output_dir=args.output_dir,
            nms=args.nms,
        )


if __name__ == "__main__":
    main()
