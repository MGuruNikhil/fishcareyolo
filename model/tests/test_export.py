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

from mina.core.model import find_tflite_weights
from mina.core.constants import RUNS_DIR


# Skip tests if models aren't available
pytestmark = pytest.mark.skipif(
    not RUNS_DIR.exists(),
    reason="No trained models available - run training first",
)


def create_test_image(width: int, height: int, seed: int) -> np.ndarray:
    """Create a random test image with the given dimensions."""
    rng = np.random.default_rng(seed)
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
        self.pt_path, self.tflite_path = find_tflite_weights()

    def test_models_exist(self):
        """Verify that both PyTorch and TFLite models exist."""
        assert self.pt_path is not None, "PyTorch model not found"
        assert self.tflite_path is not None, (
            "TFLite model not found. Run: uv run mina-export"
        )

    @pytest.mark.skipif(
        find_tflite_weights()[1] is None, reason="TFLite model not exported yet"
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
                    "bbox": boxes.xyxyn[i].tolist(),
                }
                detections.append(detection)

        return sorted(detections, key=lambda d: -d["confidence"])

    def _assert_equivalent_detections(
        self,
        pt_detections: list[dict],
        tflite_detections: list[dict],
    ) -> None:
        """Assert that two detection lists are equivalent within tolerance."""
        if len(pt_detections) == 0 and len(tflite_detections) == 0:
            return

        if len(pt_detections) > 0 and len(tflite_detections) > 0:
            pt_top_class = pt_detections[0]["class_id"]
            tflite_top_class = tflite_detections[0]["class_id"]

            pt_top_conf = pt_detections[0]["confidence"]
            tflite_top_conf = tflite_detections[0]["confidence"]

            if pt_top_conf > 0.5 and tflite_top_conf > 0.5:
                assert pt_top_class == tflite_top_class, (
                    f"Top detection class mismatch: PT={pt_top_class}, "
                    f"TFLite={tflite_top_class}"
                )

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
        self.pt_path, _ = find_tflite_weights()

    @pytest.mark.skipif(
        find_tflite_weights()[0] is None, reason="No trained model available"
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

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                assert 0 <= class_id < 5, f"Invalid class ID: {class_id}"

                confidence = float(boxes.conf[i].item())
                assert 0.0 <= confidence <= 1.0, f"Invalid confidence: {confidence}"

                bbox = boxes.xyxyn[i].tolist()
                assert len(bbox) == 4, f"Invalid bbox length: {len(bbox)}"

                for coord in bbox:
                    assert 0.0 <= coord <= 1.0, f"Invalid bbox coord: {coord}"
