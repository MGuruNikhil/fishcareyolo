"""
TFLite export logic for fish disease detection model.
"""

from pathlib import Path

from ultralytics import YOLO

from mina.core.model import find_best_weights


def export_tflite(
    weights_path: str | Path,
    int8: bool = True,
    imgsz: int = 640,
    output_dir: str | Path | None = None,
    nms: bool = False,
) -> Path:
    """
    Export YOLOv8 model to TFLite format.

    Args:
        weights_path: Path to the trained .pt weights file
        int8: Whether to use int8 quantization (recommended for mobile)
        imgsz: Input image size for the exported model
        output_dir: Optional output directory (defaults to same as weights)
        nms: Whether to include NMS in the model. Default False because
             onnx2tf has issues with TopK operations used in YOLO NMS.
             NMS should be handled in the app instead.

    Returns:
        Path to the exported TFLite model

    Raises:
        FileNotFoundError: If weights file not found
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print(f"Loading model from: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"Exporting to TFLite (int8={int8}, imgsz={imgsz}, nms={nms})...")

    # Export to TFLite
    # Note: nms=False by default because onnx2tf has issues with TopK operations
    # used in YOLO NMS. NMS should be handled in the mobile app instead.
    export_path = model.export(
        format="tflite",
        int8=int8,
        imgsz=imgsz,
        simplify=True,
        nms=nms,
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
    print("\nExport complete!")
    print(f"TFLite model saved to: {export_path}")
    print(f"Model size: {size_mb:.2f} MB")

    return export_path


def get_weights_or_default(weights_path: str | Path | None) -> Path:
    """
    Get weights path, falling back to most recent training run.

    Args:
        weights_path: Explicit path to weights, or None to auto-detect

    Returns:
        Path to weights file

    Raises:
        FileNotFoundError: If no weights found
    """
    if weights_path:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")
        return path

    path = find_best_weights()
    if path is None:
        raise FileNotFoundError(
            "No weights file specified and no training runs found.\n"
            "Please either:\n"
            "  1. Train a model first: uv run mina-train\n"
            "  2. Specify weights: uv run mina-export --weights path/to/best.pt"
        )

    print(f"Using weights from most recent training: {path}")
    return path
