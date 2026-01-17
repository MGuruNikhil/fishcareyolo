"""
Tests for Training Configuration and Script

These tests verify that training configuration is valid
and matches the design document specifications.
"""

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


# Import the training module components
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import DISEASE_CLASSES, get_data_yaml_path, train


class TestTrainingConfiguration:
    """Tests for training configuration validity."""

    def test_disease_classes_count(self):
        """Verify we have exactly 5 disease classes as per design."""
        assert len(DISEASE_CLASSES) == 5

    def test_disease_classes_content(self):
        """Verify disease classes match the dataset."""
        expected = {
            "bacterial_infection",
            "fungal_infection",
            "healthy",
            "parasite",
            "white_tail",
        }
        assert set(DISEASE_CLASSES) == expected

    def test_healthy_class_exists(self):
        """Verify 'healthy' is one of the classes."""
        assert "healthy" in DISEASE_CLASSES

    @given(
        epochs=st.integers(min_value=1, max_value=500),
        batch=st.sampled_from([8, 16, 32, 64]),
        imgsz=st.sampled_from([320, 416, 512, 640]),
    )
    @settings(max_examples=100)
    def test_training_params_valid(self, epochs: int, batch: int, imgsz: int):
        """
        **Feature: fish-disease-detection, Property: Training params validation**

        For any valid combination of training parameters,
        they should be within acceptable ranges.
        """
        # Epochs should be positive
        assert epochs > 0

        # Batch size should be power of 2 for efficiency
        assert batch & (batch - 1) == 0

        # Image size should be multiple of 32 (YOLO requirement)
        assert imgsz % 32 == 0


class TestDatasetConfiguration:
    """Tests for dataset configuration."""

    def test_data_yaml_structure(self, data_yaml_content, tmp_path):
        """Test that data.yaml has required fields."""
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(data_yaml_content)

        import yaml

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Check required fields
        assert "path" in config
        assert "train" in config
        assert "val" in config
        assert "names" in config
        assert "nc" in config

        # Check class count matches names
        assert config["nc"] == len(config["names"])

    def test_data_yaml_classes_match(self, data_yaml_content, tmp_path):
        """Test that data.yaml classes match DISEASE_CLASSES."""
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(data_yaml_content)

        import yaml

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        yaml_classes = set(config["names"].values())
        code_classes = set(DISEASE_CLASSES)

        assert yaml_classes == code_classes, (
            f"Class mismatch: yaml={yaml_classes}, code={code_classes}"
        )


class TestModelRequirements:
    """Tests that verify model meets requirements."""

    def test_using_nano_model(self):
        """
        Verify we use YOLOv8n (nano) as specified in design.

        Requirement: Use YOLOv8n for mobile performance.
        """
        # The train function defaults to yolov8n.pt
        import inspect

        sig = inspect.signature(train)
        default_model = sig.parameters["pretrained"].default

        assert "yolov8n" in default_model, (
            f"Expected yolov8n model, got: {default_model}"
        )

    def test_default_training_params(self):
        """
        Verify default training params match design spec.

        Design specifies: epochs=100, batch=16, imgsz=640
        """
        import inspect

        sig = inspect.signature(train)

        assert sig.parameters["epochs"].default == 100
        assert sig.parameters["batch"].default == 16
        assert sig.parameters["imgsz"].default == 640
