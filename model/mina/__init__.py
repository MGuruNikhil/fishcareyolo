"""
Mina - Fish Disease Detection Model

A YOLOv8-based model for detecting diseases in fish.
"""

from mina.core.constants import DISEASE_CLASSES
from mina.core.types import BoundingBox, Detection

__version__ = "0.1.0"
__all__ = ["DISEASE_CLASSES", "BoundingBox", "Detection"]
