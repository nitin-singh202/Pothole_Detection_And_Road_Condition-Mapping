"""
Rule-Based Road Condition Severity Classifier.

Computes physical severity estimation using 2D geometric surface area footprints,
bounding box aspect ratios, and detection confidence scores.

IMPORTANT LIMITATION & DISCLAIMER:
Monocular 2D RGB video capture cannot measure physical vertical pothole depth without
active LiDAR or calibrated stereo-depth sensors. Severity levels (LOW, MEDIUM, HIGH)
reflect relative visual road surface disruption area, not structural cavity depth.
"""

from enum import Enum
from typing import Optional
from config import Config


class SeverityLevel(str, Enum):
    """Categorical severity classifications for road surface distress."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class SeverityClassifier:
    """Classifies pothole severity using deterministic bounding box geometric ratios."""

    def __init__(
        self,
        low_max_area: Optional[float] = None,
        med_max_area: Optional[float] = None,
    ):
        """
        Initializes classifier with configurable area threshold bounds.

        Args:
            low_max_area: Maximum area ratio for LOW severity. Default from Config (0.015).
            med_max_area: Maximum area ratio for MEDIUM severity. Default from Config (0.045).
        """
        self.low_max_area = low_max_area if low_max_area is not None else Config.SEVERITY_LOW_MAX_AREA
        self.med_max_area = med_max_area if med_max_area is not None else Config.SEVERITY_MED_MAX_AREA

    def classify_by_area_ratio(self, area_ratio: float, confidence: float = 1.0) -> SeverityLevel:
        """
        Classifies severity based on normalized bounding box area ratio.

        Args:
            area_ratio: (bbox_width * bbox_height) / (frame_width * frame_height).
            confidence: Model detection confidence (0.0 - 1.0).

        Returns:
            SeverityLevel: Enum instance (LOW, MEDIUM, or HIGH).
        """
        if area_ratio < 0:
            return SeverityLevel.UNKNOWN

        # Classification rule:
        # Area < low_threshold -> LOW
        # low_threshold <= Area < med_threshold -> MEDIUM
        # Area >= med_threshold -> HIGH
        if area_ratio < self.low_max_area:
            return SeverityLevel.LOW
        elif area_ratio < self.med_max_area:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.HIGH

    def classify_detection(
        self,
        bbox_width: float,
        bbox_height: float,
        frame_width: int,
        frame_height: int,
        confidence: float = 1.0,
    ) -> SeverityLevel:
        """
        Classifies severity directly from pixel bounding box and frame dimensions.

        Args:
            bbox_width: Width of bounding box in pixels.
            bbox_height: Height of bounding box in pixels.
            frame_width: Width of source video frame in pixels.
            frame_height: Height of source video frame in pixels.
            confidence: Detection confidence.

        Returns:
            SeverityLevel: LOW, MEDIUM, or HIGH.
        """
        frame_area = float(frame_width * frame_height)
        if frame_area <= 0:
            return SeverityLevel.UNKNOWN

        box_area = float(bbox_width * bbox_height)
        area_ratio = box_area / frame_area
        return self.classify_by_area_ratio(area_ratio, confidence)
