"""
Model loading and discovery utilities.
"""

from pathlib import Path

from ultralytics import YOLO

from mina.core.constants import RUNS_DIR


def load_model(weights_path: str | Path) -> YOLO:
    """
    Load a YOLO model from weights file.

    Args:
        weights_path: Path to model weights (.pt or .tflite)

    Returns:
        Loaded YOLO model

    Raises:
        FileNotFoundError: If weights file doesn't exist
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    return YOLO(str(weights_path))


def find_best_weights(runs_dir: Path | None = None) -> Path | None:
    """
    Find the best.pt file from the most recent training run.

    Args:
        runs_dir: Directory containing training runs. Defaults to RUNS_DIR.

    Returns:
        Path to best.pt if found, None otherwise
    """
    if runs_dir is None:
        runs_dir = RUNS_DIR

    if not runs_dir.exists():
        return None

    # Find most recent training run
    try:
        run_dirs = sorted(
            runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        )
    except (OSError, PermissionError):
        return None

    for run_dir in run_dirs:
        best_weights = run_dir / "weights" / "best.pt"
        if best_weights.exists():
            return best_weights

    return None


def find_tflite_weights(
    runs_dir: Path | None = None,
) -> tuple[Path | None, Path | None]:
    """
    Find both PyTorch and TFLite model paths from the most recent training run.

    Args:
        runs_dir: Directory containing training runs. Defaults to RUNS_DIR.

    Returns:
        Tuple of (pt_path, tflite_path). Either may be None if not found.
    """
    if runs_dir is None:
        runs_dir = RUNS_DIR

    if not runs_dir.exists():
        return None, None

    try:
        run_dirs = sorted(
            runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        )
    except (OSError, PermissionError):
        return None, None

    for run_dir in run_dirs:
        pt_path = run_dir / "weights" / "best.pt"
        tflite_path = pt_path.with_suffix(".tflite")

        # Also check alternative TFLite locations
        alt_tflite = run_dir / "weights" / "best_saved_model" / "best_float32.tflite"
        alt_tflite_int8 = run_dir / "weights" / "best_saved_model" / "best_int8.tflite"

        if pt_path.exists():
            for tf_path in [tflite_path, alt_tflite, alt_tflite_int8]:
                if tf_path.exists():
                    return pt_path, tf_path
            return pt_path, None

    return None, None
