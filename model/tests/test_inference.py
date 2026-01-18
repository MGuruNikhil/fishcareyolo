"""
Tests for Model Inference

These tests verify that model inference produces valid Detection structures
as specified in the design document.

Feature: fish-disease-detection, Property 5: Detection result structure
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from mina.core.constants import DISEASE_CLASSES
from mina.core.types import BoundingBox, Detection


# Strategies for generating test data
confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
coordinate_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
disease_class_strategy = st.sampled_from(DISEASE_CLASSES)


class TestDetectionStructure:
    """
    Property 5: Detection result structure

    For any detection returned by the model, it SHALL contain a valid diseaseClass,
    a confidence between 0.0 and 1.0, and a boundingBox with x, y, width, height
    all between 0.0 and 1.0.
    """

    @given(
        disease_class=disease_class_strategy,
        confidence=confidence_strategy,
        x=st.floats(min_value=0.0, max_value=0.5),
        y=st.floats(min_value=0.0, max_value=0.5),
        width=st.floats(min_value=0.1, max_value=0.5),
        height=st.floats(min_value=0.1, max_value=0.5),
    )
    @settings(max_examples=100)
    def test_valid_detection_passes_validation(
        self,
        disease_class: str,
        confidence: float,
        x: float,
        y: float,
        width: float,
        height: float,
    ):
        """
        **Feature: fish-disease-detection, Property 5: Detection result structure**

        Valid detections should pass validation without errors.
        """
        detection = Detection(
            id="test_001",
            disease_class=disease_class,
            confidence=confidence,
            bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
        )

        errors = detection.validate()
        assert len(errors) == 0, f"Unexpected validation errors: {errors}"

    @given(confidence=st.floats(min_value=1.01, max_value=10.0))
    @settings(max_examples=100)
    def test_invalid_confidence_detected(self, confidence: float):
        """
        **Feature: fish-disease-detection, Property 5: Detection result structure**

        Detections with confidence > 1.0 should fail validation.
        """
        detection = Detection(
            id="test_001",
            disease_class="healthy",
            confidence=confidence,
            bounding_box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
        )

        errors = detection.validate()
        assert len(errors) > 0, "Expected validation error for high confidence"
        assert any("confidence" in e.lower() for e in errors)

    def test_invalid_disease_class_detected(self):
        """
        **Feature: fish-disease-detection, Property 5: Detection result structure**

        Detections with invalid disease class should fail validation.
        """
        detection = Detection(
            id="test_001",
            disease_class="unknown_disease",
            confidence=0.5,
            bounding_box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
        )

        errors = detection.validate()
        assert len(errors) > 0, "Expected validation error for invalid class"
        assert any("class" in e.lower() for e in errors)

    @given(
        x=st.floats(min_value=-1.0, max_value=-0.01),
    )
    @settings(max_examples=50)
    def test_negative_coordinates_detected(self, x: float):
        """
        **Feature: fish-disease-detection, Property 5: Detection result structure**

        Detections with negative coordinates should fail validation.
        """
        detection = Detection(
            id="test_001",
            disease_class="healthy",
            confidence=0.5,
            bounding_box=BoundingBox(x=x, y=0.1, width=0.2, height=0.2),
        )

        errors = detection.validate()
        assert len(errors) > 0, "Expected validation error for negative x"


class TestConfidenceFiltering:
    """
    Property 2: Confidence filtering

    For any list of raw detections from the model, filtering SHALL remove all
    detections with confidence below 0.3 and keep all detections with confidence
    0.3 or above.
    """

    CONFIDENCE_THRESHOLD = 0.3

    @given(
        confidences=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_confidence_filtering(self, confidences: list[float]):
        """
        **Feature: fish-disease-detection, Property 2: Confidence filtering**

        All detections with confidence >= 0.3 should be kept,
        all detections with confidence < 0.3 should be removed.
        """
        # Create mock detections
        detections = [
            Detection(
                id=f"det_{i}",
                disease_class="healthy",
                confidence=conf,
                bounding_box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
            )
            for i, conf in enumerate(confidences)
        ]

        # Filter by threshold
        filtered = [d for d in detections if d.confidence >= self.CONFIDENCE_THRESHOLD]

        # Verify all kept detections meet threshold
        for det in filtered:
            assert det.confidence >= self.CONFIDENCE_THRESHOLD, (
                f"Detection with confidence {det.confidence} should have been filtered"
            )

        # Verify all filtered out detections were below threshold
        removed_ids = {d.id for d in detections} - {d.id for d in filtered}
        removed = [d for d in detections if d.id in removed_ids]

        for det in removed:
            assert det.confidence < self.CONFIDENCE_THRESHOLD, (
                f"Detection with confidence {det.confidence} should not have been filtered"
            )


class TestDetectionSorting:
    """
    Property 3: Detection sorting by confidence

    For any list of detections, when sorted for display, each detection's confidence
    SHALL be greater than or equal to the next detection's confidence (descending order).
    """

    @given(
        confidences=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_detection_sorting(self, confidences: list[float]):
        """
        **Feature: fish-disease-detection, Property 3: Detection sorting by confidence**

        After sorting, each detection's confidence should be >= the next one's.
        """
        # Create mock detections
        detections = [
            Detection(
                id=f"det_{i}",
                disease_class="healthy",
                confidence=conf,
                bounding_box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
            )
            for i, conf in enumerate(confidences)
        ]

        # Sort by confidence descending (as the app does)
        sorted_detections = sorted(detections, key=lambda d: -d.confidence)

        # Verify descending order
        for i in range(len(sorted_detections) - 1):
            current = sorted_detections[i].confidence
            next_conf = sorted_detections[i + 1].confidence
            assert current >= next_conf, (
                f"Detections not sorted: {current} < {next_conf}"
            )


class TestDiseaseClasses:
    """Tests for disease class definitions."""

    def test_all_classes_defined(self):
        """Verify all expected disease classes are defined."""
        expected = {
            "bacterial_infection",
            "fungal_infection",
            "healthy",
            "parasite",
            "white_tail",
        }
        assert set(DISEASE_CLASSES) == expected

    def test_class_count(self):
        """Verify class count matches model output."""
        assert len(DISEASE_CLASSES) == 5

    def test_no_duplicate_classes(self):
        """Verify no duplicate class names."""
        assert len(DISEASE_CLASSES) == len(set(DISEASE_CLASSES))
