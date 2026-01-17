"""
YOLOv8n Training Script for Fish Disease Detection

This script trains a YOLOv8 nano model on the fish disease dataset.
The nano variant is optimized for mobile deployment.

Usage:
    python train.py [--epochs EPOCHS] [--batch BATCH] [--imgsz IMGSZ]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


# Disease classes from the Roboflow dataset
DISEASE_CLASSES = [
    "bacterial_infection",
    "fungal_infection",
    "healthy",
    "parasite",
    "white_tail",
]


def get_data_yaml_path() -> Path:
    """Get the path to the data.yaml configuration file."""
    # Look for dataset in common locations
    possible_paths = [
        Path(__file__).parent / "data" / "data.yaml",
        Path(__file__).parent / "Mina-2" / "data.yaml",
        Path(__file__).parent / "mina-2" / "data.yaml",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"data.yaml not found. Looked in: {[str(p) for p in possible_paths]}\n"
        "Please run the dataset download script first: python dataset/get.py"
    )


def train(
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    name: str = "fish_disease",
    pretrained: str = "yolov8n.pt",
) -> Path:
    """
    Train YOLOv8n model on fish disease dataset.

    Args:
        epochs: Number of training epochs
        batch: Batch size for training
        imgsz: Input image size (square)
        name: Name for the training run
        pretrained: Pretrained model to start from

    Returns:
        Path to the best model weights
    """
    data_yaml = get_data_yaml_path()
    print(f"Using dataset config: {data_yaml}")

    # Load YOLOv8 nano model (optimized for mobile)
    model = YOLO(pretrained)

    # Train on fish disease data
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        # Save best model based on validation mAP
        save=True,
        save_period=-1,  # Only save best and last
        # Training optimizations
        patience=20,  # Early stopping patience
        workers=4,
        device="auto",  # Use GPU if available
        # Augmentation settings for better generalization
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    # Get path to best weights
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete!")
    print(f"Best weights saved to: {best_weights}")
    print(f"Validation mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(
        f"Validation mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}"
    )

    return best_weights


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n model for fish disease detection"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="fish_disease",
        help="Training run name (default: fish_disease)",
    )

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
    )


if __name__ == "__main__":
    main()
