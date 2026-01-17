# Mina - Fish Disease Detection Model

YOLOv8-based model for detecting diseases in fish, optimized for mobile deployment.

## Setup

```bash
cd model
uv sync
```

## Quick Start

```bash
# 1. Download the dataset
ROBOFLOW_API_KEY=your_key uv run mina-download

# 2. Train the model
uv run mina-train --epochs 100 --batch 16

# 3. Export to TFLite for mobile
uv run mina-export

# 4. Evaluate on test set
uv run mina-evaluate --weights runs/detect/fish_disease/weights/best.pt

# 5. Run inference on images
uv run mina-infer --dir data/images/val
```

## CLI Commands

### `mina-download`

Download the fish disease dataset from Roboflow.

```bash
ROBOFLOW_API_KEY=your_key uv run mina-download
```

This will:
1. Download the dataset from Roboflow
2. Organize training/validation data into `data/`
3. Save test data separately to `test_data/` (for final evaluation)
4. Create `data/data.yaml` configuration file

### `mina-train`

Train the YOLOv8n model.

```bash
uv run mina-train [--epochs N] [--batch N] [--imgsz N] [--name NAME]
```

Options:
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--imgsz`: Input image size (default: 640)
- `--name`: Training run name (default: fish_disease)

Results are saved to `runs/detect/{name}/`.

### `mina-export`

Export the trained model to TFLite format for mobile deployment.

```bash
uv run mina-export [--weights PATH] [--no-int8] [--imgsz N] [--output-dir PATH]
```

Options:
- `--weights`: Path to trained weights (.pt file). Auto-detects if not provided.
- `--no-int8`: Disable int8 quantization (not recommended for mobile)
- `--imgsz`: Input image size (default: 640)
- `--output-dir`: Output directory for the TFLite model

### `mina-evaluate`

Evaluate the model on the held-out test set.

```bash
uv run mina-evaluate --weights PATH [--test-dir PATH] [--imgsz N]
```

Options:
- `--weights`: Path to trained weights (.pt or .tflite)
- `--test-dir`: Path to test data directory (default: test_data)
- `--imgsz`: Input image size (default: 640)
- `--confidence`: Confidence threshold (default: 0.001)
- `--iou`: IoU threshold for NMS (default: 0.6)

Reports mAP@50, mAP@50-95, precision, recall, and per-class AP.

### `mina-infer`

Run inference on images.

```bash
uv run mina-infer [--weights PATH] [--image PATH] [--dir PATH] [--confidence N]
```

Options:
- `--weights`: Path to model weights (.pt or .tflite)
- `--image`: Test a single image
- `--dir`: Test all images in a directory
- `--confidence`: Minimum confidence threshold (default: 0.3)

## Testing

```bash
pytest tests/ -v
```

Tests cover:
- Training configuration validation
- TFLite export equivalence (Property 8)
- Detection structure validation (Property 5)
- Confidence filtering (Property 2)
- Detection sorting (Property 3)

## Disease Classes

| Index | Class |
|-------|-------|
| 0 | bacterial_infection |
| 1 | fungal_infection |
| 2 | healthy |
| 3 | parasite |
| 4 | white_tail |

## Project Structure

```
model/
├── mina/                      # Main package
│   ├── __init__.py
│   ├── core/                  # Shared code
│   │   ├── __init__.py
│   │   ├── constants.py       # Disease classes, paths, thresholds
│   │   ├── types.py           # Detection, BoundingBox types
│   │   ├── model.py           # Model loading utilities
│   │   └── dataset.py         # Dataset YAML generation
│   ├── train.py               # Training logic
│   ├── export.py              # TFLite export logic
│   ├── evaluate.py            # Evaluation logic
│   ├── inference.py           # Inference/detection logic
│   └── dataset.py             # Dataset download/organization
├── cli/                       # CLI entry points
│   ├── train.py
│   ├── export.py
│   ├── evaluate.py
│   ├── infer.py
│   └── download.py
├── tests/                     # Test suite
│   ├── conftest.py
│   ├── test_training.py
│   ├── test_inference.py
│   └── test_export.py
├── data/                      # Training/validation data
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   └── data.yaml
├── test_data/                 # Held-out test data
│   ├── images/
│   └── labels/
├── runs/                      # Training outputs
│   └── detect/
│       └── fish_disease/
│           └── weights/
│               ├── best.pt
│               └── best.tflite
├── pyproject.toml
└── README.md
```

## Architecture

The codebase follows a clean separation of concerns:

- **`mina/core/`**: Shared constants, types, and utilities. No code duplication.
- **`mina/*.py`**: Business logic modules (train, export, evaluate, inference, dataset)
- **`cli/*.py`**: Thin CLI wrappers with argument parsing
- **`tests/`**: Property-based and unit tests

All disease classes, thresholds, and paths are defined once in `mina/core/constants.py`.
