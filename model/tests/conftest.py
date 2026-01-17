"""
Pytest configuration and shared fixtures for model tests.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def temp_image_dir(tmp_path):
    """Create a temporary directory with test images."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create a few test images
    for i in range(3):
        img_array = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(images_dir / f"test_{i}.jpg")

    return images_dir


@pytest.fixture
def sample_image(tmp_path):
    """Create a single sample image for testing."""
    img_array = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    img_path = tmp_path / "sample.jpg"
    img.save(img_path)

    return img_path


@pytest.fixture
def data_yaml_content():
    """Sample data.yaml content for testing."""
    return """
path: ./data
train: images/train
val: images/val

names:
  0: bacterial_infection
  1: fungal_infection
  2: healthy
  3: parasite
  4: white_tail

nc: 5
"""
