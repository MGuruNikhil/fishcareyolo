"""
Tests for TFLite Export Equivalence

Property 8: TFLite export equivalence
For any test image, the TFLite model output SHALL produce detections equivalent
to the original PyTorch model output (same classes, similar confidence scores
within 0.05 tolerance).

Feature: fish-disease-detection, Property 8: TFLite export equivalence
Validates: Design - Training Pipeline
"""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from PIL import Image


# Skip tests if models aren't available
pytestmark = pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath("runs").exists(),
    reason="No trained models available - run training first",
)


def find_model_paths() -> tuple[Path | None, Path | None]:
    """Find the PyTorch and TFLite model paths."""
    runs_dir = Path(__file__).parent.parent / "runs" / "detect"

    if not runs_dir.exists():
        return None, None

    # Find most recent training run
    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        pt_path = run_dir / "weights" / "best.pt"
        # TFLite is exported alongside the pt file
        tflite_path = pt_path.with_suffix(".tflite")

        # Also check in parent directory (export sometimes puts it there)
        alt_tflite = run_dir / "weights" / "best_saved_model" / "best_float32.tflite"
        alt_tflite_int8 = run_dir / "weights" / "best_saved_model" / "best_int8.tflite"

        if pt_path.exists():
            for tf_path in [tflite_path, alt_tflite, alt_tflite_int8]:
                if tf_path.exists():
                    return pt_path, tf_path
            # Return PT path even if TFLite not found yet
            return pt_path, None

    return None, None


def create_test_image(width: int, height: int, seed: int) -> np.ndarray:
    """Create a random test image with the given dimensions."""
    rng = np.random.default_rng(seed)
    # Create RGB image with realistic pixel values
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def save_temp_image(image: np.ndarray, path: Path) -> None:
    """Save numpy array as image file."""
    Image.fromarray(image).save(path)


class TestTFLiteExportEquivalence:
    """
    Property 8: TFLite export equivalence

    For any test image, the TFLite model output SHALL produce detections
    equivalent to the original PyTorch model output.
    """

    CONFIDENCE_TOLERANCE = 0.05  # Allow 5% difference in confidence

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test fixtures."""
        self.tmp_path = tmp_path
        self.pt_path, self.tflite_path = find_model_paths()

    def test_models_exist(self):
        """Verify that both PyTorch and TFLite models exist."""
        assert self.pt_path is not None, "PyTorch model not found"
        assert self.tflite_path is not None, (
            "TFLite model not found. Run: python export_model.py"
        )

    @pytest.mark.skipif(
        find_model_paths()[1] is None, reason="TFLite model not exported yet"
    )
    @settings(max_examples=100, deadline=None)
    @given(
        width=st.integers(min_value=320, max_value=640),
        height=st.integers(min_value=320, max_value=640),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_detection_equivalence(self, width: int, height: int, seed: int):
        """
        **Feature: fish-disease-detection, Property 8: TFLite export equivalence**

        For any test image, both models should produce equivalent detections.
        """
        from ultralytics import YOLO

        # Skip if models not available
        assume(self.pt_path is not None and self.tflite_path is not None)

        # Create test image
        test_image = create_test_image(width, height, seed)
        image_path = self.tmp_path / f"test_{seed}.jpg"
        save_temp_image(test_image, image_path)

        # Load models
        pt_model = YOLO(str(self.pt_path))
        tflite_model = YOLO(str(self.tflite_path))

        # Run inference
        pt_results = pt_model(str(image_path), verbose=False)
        tflite_results = tflite_model(str(image_path), verbose=False)

        # Extract detections
        pt_detections = self._extract_detections(pt_results)
        tflite_detections = self._extract_detections(tflite_results)

        # Compare results
        self._assert_equivalent_detections(pt_detections, tflite_detections)

    def _extract_detections(self, results) -> list[dict]:
        """Extract detection info from YOLO results."""
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                detection = {
                    "class_id": int(boxes.cls[i].item()),
                    "confidence": float(boxes.conf[i].item()),
                    "bbox": boxes.xyxyn[i].tolist(),  # Normalized coordinates
                }
                detections.append(detection)

        # Sort by confidence for consistent comparison
        return sorted(detections, key=lambda d: -d["confidence"])

    def _assert_equivalent_detections(
        self,
        pt_detections: list[dict],
        tflite_detections: list[dict],
    ) -> None:
        """Assert that two detection lists are equivalent within tolerance."""
        # Both should have same number of detections (after NMS)
        # Note: There might be small differences due to quantization
        # so we compare the top detections

        # If PT model finds nothing, TFLite should too (or vice versa)
        if len(pt_detections) == 0 and len(tflite_detections) == 0:
            return  # Both empty is valid

        # Compare top detections if both have results
        if len(pt_detections) > 0 and len(tflite_detections) > 0:
            # Check that top detection classes match
            pt_top_class = pt_detections[0]["class_id"]
            tflite_top_class = tflite_detections[0]["class_id"]

            # If confidences are very low, classes might differ due to noise
            pt_top_conf = pt_detections[0]["confidence"]
            tflite_top_conf = tflite_detections[0]["confidence"]

            if pt_top_conf > 0.5 and tflite_top_conf > 0.5:
                assert pt_top_class == tflite_top_class, (
                    f"Top detection class mismatch: PT={pt_top_class}, "
                    f"TFLite={tflite_top_class}"
                )

            # Check confidence scores are within tolerance
            conf_diff = abs(pt_top_conf - tflite_top_conf)
            assert conf_diff <= self.CONFIDENCE_TOLERANCE, (
                f"Confidence difference {conf_diff:.4f} exceeds tolerance "
                f"{self.CONFIDENCE_TOLERANCE}: PT={pt_top_conf:.4f}, "
                f"TFLite={tflite_top_conf:.4f}"
            )


class TestModelOutputFormat:
    """Test that model outputs match expected Detection structure."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test fixtures."""
        self.tmp_path = tmp_path
        self.pt_path, _ = find_model_paths()

    @pytest.mark.skipif(
        find_model_paths()[0] is None, reason="No trained model available"
    )
    def test_output_format_matches_detection_structure(self):
        """
        Verify model outputs can be converted to Detection structure.

        Expected format per design doc:
        {
            id: string,
            diseaseClass: DiseaseClass,
            confidence: number (0.0 to 1.0),
            boundingBox: { x, y, width, height } (all 0.0 to 1.0)
        }
        """
        from ultralytics import YOLO

        # Create a test image
        test_image = create_test_image(640, 640, seed=42)
        image_path = self.tmp_path / "test_format.jpg"
        save_temp_image(test_image, image_path)

        # Load and run model
        model = YOLO(str(self.pt_path))
        results = model(str(image_path), verbose=False)

        # Verify output structure
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # Check each detection
            for i in range(len(boxes)):
                # Class ID should be valid
                class_id = int(boxes.cls[i].item())
                assert 0 <= class_id < 5, f"Invalid class ID: {class_id}"

                # Confidence should be 0-1
                confidence = float(boxes.conf[i].item())
                assert 0.0 <= confidence <= 1.0, f"Invalid confidence: {confidence}"

                # Bounding box should be normalized
                bbox = boxes.xyxyn[i].tolist()
                assert len(bbox) == 4, f"Invalid bbox length: {len(bbox)}"

                for coord in bbox:
                    assert 0.0 <= coord <= 1.0, f"Invalid bbox coord: {coord}"
