"""
TFLite Export Script for Fish Disease Detection Model

Exports a trained YOLOv8 model to TFLite format with int8 quantization
for mobile deployment.

Usage:
    python export_model.py --weights path/to/best.pt [--int8] [--imgsz SIZE]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def export_tflite(
    weights_path: str | Path,
    int8: bool = True,
    imgsz: int = 640,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Export YOLOv8 model to TFLite format.

    Args:
        weights_path: Path to the trained .pt weights file
        int8: Whether to use int8 quantization (recommended for mobile)
        imgsz: Input image size for the exported model
        output_dir: Optional output directory (defaults to same as weights)

    Returns:
        Path to the exported TFLite model
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print(f"Loading model from: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"Exporting to TFLite (int8={int8}, imgsz={imgsz})...")

    # Export to TFLite
    # Note: ultralytics handles the ONNX -> TFLite conversion internally
    export_path = model.export(
        format="tflite",
        int8=int8,
        imgsz=imgsz,
        # Additional optimizations for mobile
        simplify=True,
        nms=True,  # Include NMS in the model
    )

    export_path = Path(export_path)

    # Move to output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / export_path.name
        export_path.rename(final_path)
        export_path = final_path

    # Print model size
    size_mb = export_path.stat().st_size / (1024 * 1024)
    print(f"\nExport complete!")
    print(f"TFLite model saved to: {export_path}")
    print(f"Model size: {size_mb:.2f} MB")

    return export_path


def find_best_weights() -> Path | None:
    """Find the best.pt file from the most recent training run."""
    runs_dir = Path(__file__).parent / "runs" / "detect"

    if not runs_dir.exists():
        return None

    # Find most recent training run
    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        best_weights = run_dir / "weights" / "best.pt"
        if best_weights.exists():
            return best_weights

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to TFLite format for mobile deployment"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained weights file (.pt). If not provided, uses most recent training run.",
    )
    parser.add_argument(
        "--no-int8",
        action="store_true",
        help="Disable int8 quantization (not recommended for mobile)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for the exported model (default: 640)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the TFLite model",
    )

    args = parser.parse_args()

    # Find weights
    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = find_best_weights()
        if weights_path is None:
            print("Error: No weights file specified and no training runs found.")
            print("Please either:")
            print("  1. Train a model first: python train.py")
            print(
                "  2. Specify weights: python export_model.py --weights path/to/best.pt"
            )
            return
        print(f"Using weights from most recent training: {weights_path}")

    export_tflite(
        weights_path=weights_path,
        int8=not args.no_int8,
        imgsz=args.imgsz,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
