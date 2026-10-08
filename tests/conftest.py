"""
Pytest Shared Configuration and Mock Fixtures.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.detection.detector import DetectionResult


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Returns a synthetic 3-channel BGR image frame (640x480)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection() -> DetectionResult:
    """Returns a sample DetectionResult object."""
    return DetectionResult(
        class_id=0,
        class_name="pothole",
        confidence=0.88,
        x1=100.0,
        y1=150.0,
        x2=200.0,
        y2=250.0,
        bbox_width=100.0,
        bbox_height=100.0,
        area_ratio=0.032,
        frame_number=10,
        timestamp_sec=0.333,
        severity="MEDIUM",
    )
