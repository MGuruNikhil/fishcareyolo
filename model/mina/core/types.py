"""
Type definitions for the fish disease detection model.
"""

from typing import NamedTuple

from mina.core.constants import DISEASE_CLASSES


class BoundingBox(NamedTuple):
    """Bounding box in normalized coordinates (0-1)."""

    x: float  # left edge
    y: float  # top edge
    width: float
    height: float

    def is_valid(self) -> bool:
        """Check if bounding box coordinates are valid (within 0-1 range)."""
        return (
            0.0 <= self.x <= 1.0
            and 0.0 <= self.y <= 1.0
            and 0.0 <= self.width <= 1.0
            and 0.0 <= self.height <= 1.0
            and self.x + self.width <= 1.0 + 1e-6
            and self.y + self.height <= 1.0 + 1e-6
        )


class Detection(NamedTuple):
    """A single disease detection from the model."""

    id: str
    disease_class: str
    confidence: float
    bounding_box: BoundingBox

    def is_valid(self) -> bool:
        """Check if detection has valid values."""
        return (
            self.disease_class in DISEASE_CLASSES
            and 0.0 <= self.confidence <= 1.0
            and self.bounding_box.is_valid()
        )

    def validate(self) -> list[str]:
        """
        Validate detection against expected structure.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        if self.disease_class not in DISEASE_CLASSES:
            errors.append(f"Invalid disease class: {self.disease_class}")

        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"Confidence out of range: {self.confidence}")

        bbox = self.bounding_box
        if not (0.0 <= bbox.x <= 1.0):
            errors.append(f"BBox x out of range: {bbox.x}")
        if not (0.0 <= bbox.y <= 1.0):
            errors.append(f"BBox y out of range: {bbox.y}")
        if not (0.0 <= bbox.width <= 1.0):
            errors.append(f"BBox width out of range: {bbox.width}")
        if not (0.0 <= bbox.height <= 1.0):
            errors.append(f"BBox height out of range: {bbox.height}")

        if bbox.x + bbox.width > 1.0 + 1e-6:
            errors.append(f"BBox exceeds right edge: x={bbox.x}, width={bbox.width}")
        if bbox.y + bbox.height > 1.0 + 1e-6:
            errors.append(f"BBox exceeds bottom edge: y={bbox.y}, height={bbox.height}")

        return errors
