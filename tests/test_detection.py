"""
Unit Tests for Detection Preprocessing, Postprocessing, and Dataclass structures.
"""

import numpy as np
import pytest
from app.detection.detector import DetectionResult
from app.detection.postprocessing import draw_detection_overlay
from app.detection.preprocessing import letterbox_image, preprocess_frame


def test_letterbox_image_dimensions():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    letterboxed, ratio, (pad_w, pad_h) = letterbox_image(img, new_shape=(640, 640))

    assert letterboxed.shape == (640, 640, 3)
    assert ratio == 1.0  # 640 / 640
    assert pad_w == 0
    assert pad_h == 80  # (640 - 480) / 2


def test_preprocess_frame_empty_raises_error():
    with pytest.raises(ValueError):
        preprocess_frame(np.array([]))


def test_detection_result_serialization(sample_detection):
    data = sample_detection.to_dict()
    assert data["class_id"] == 0
    assert data["class_name"] == "pothole"
    assert data["confidence"] == 0.88
    assert data["bbox_width"] == 100.0


def test_draw_detection_overlay(sample_frame, sample_detection):
    annotated = draw_detection_overlay(
        frame=sample_frame,
        detections=[sample_detection.to_dict()],
        frame_number=1,
        timestamp_sec=0.033,
        gps_info={"latitude": 37.7749, "longitude": -122.4194},
    )
    assert annotated.shape == sample_frame.shape
    assert isinstance(annotated, np.ndarray)
