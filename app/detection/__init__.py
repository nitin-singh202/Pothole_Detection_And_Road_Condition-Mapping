"""
Computer Vision and Object Detection Sub-package.
"""

from app.detection.detector import PotholeDetector, DetectionResult
from app.detection.preprocessing import letterbox_image, preprocess_frame
from app.detection.postprocessing import draw_detection_overlay

__all__ = [
    "PotholeDetector",
    "DetectionResult",
    "letterbox_image",
    "preprocess_frame",
    "draw_detection_overlay",
]
