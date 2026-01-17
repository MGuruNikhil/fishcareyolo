# Fish Disease Detection Model Training

This module contains the YOLOv8 training pipeline for the Mina fish disease detection app.

## Setup

```bash
cd model
uv sync
```

## Dataset

Download the fish disease dataset from Roboflow:

```bash
ROBOFLOW_API_KEY=your_key python dataset/get.py
```

This will:
1. Download the dataset from Roboflow
2. Organize it into `data/images/train`, `data/images/val`, `data/labels/train`, `data/labels/val`
3. Create `data/data.yaml` configuration file

## Training

Train the YOLOv8n model:

```bash
python train.py --epochs 100 --batch 16
```

Options:
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--imgsz`: Input image size (default: 640)
- `--name`: Training run name (default: fish_disease)

Results are saved to `runs/detect/fish_disease/`.

## Export to TFLite

Export the trained model to TFLite format for mobile deployment:

```bash
python export_model.py --weights runs/detect/fish_disease/weights/best.pt
```

Options:
- `--weights`: Path to trained weights (.pt file)
- `--no-int8`: Disable int8 quantization (not recommended)
- `--imgsz`: Input image size (default: 640)

The TFLite model is saved alongside the weights.

## Testing

### Test inference on images:

```bash
python test_inference.py --weights runs/detect/fish_disease/weights/best.tflite --dir data/images/val
python test_inference.py --weights runs/detect/fish_disease/weights/best.tflite --image path/to/image.jpg
```

Options:
- `--weights`: Path to model (.pt or .tflite)
- `--image`: Test a single image
- `--dir`: Test all images in a directory
- `--confidence`: Minimum confidence threshold (default: 0.3)

### Run property tests:

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

## Output Structure

```
model/
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── runs/
│   └── detect/
│       └── fish_disease/
│           └── weights/
│               ├── best.pt
│               └── best.tflite
├── tests/
│   ├── test_export.py
│   ├── test_inference.py
│   └── test_training.py
├── train.py
├── export_model.py
└── test_inference.py
```
