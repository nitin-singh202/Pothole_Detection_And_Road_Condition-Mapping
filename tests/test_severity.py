"""
Unit Tests for Severity Classifier Module.
"""

import pytest
from app.severity.classifier import SeverityClassifier, SeverityLevel


def test_severity_levels_by_area_ratio():
    classifier = SeverityClassifier(low_max_area=0.015, med_max_area=0.045)

    assert classifier.classify_by_area_ratio(0.005) == SeverityLevel.LOW
    assert classifier.classify_by_area_ratio(0.014) == SeverityLevel.LOW
    assert classifier.classify_by_area_ratio(0.015) == SeverityLevel.MEDIUM
    assert classifier.classify_by_area_ratio(0.030) == SeverityLevel.MEDIUM
    assert classifier.classify_by_area_ratio(0.045) == SeverityLevel.HIGH
    assert classifier.classify_by_area_ratio(0.120) == SeverityLevel.HIGH


def test_severity_classify_detection_dimensions():
    classifier = SeverityClassifier(low_max_area=0.015, med_max_area=0.045)

    # Frame: 1000x1000 = 1,000,000 px^2
    # Small box: 50x50 = 2,500 px^2 (Ratio = 0.0025 -> LOW)
    assert classifier.classify_detection(50, 50, 1000, 1000) == SeverityLevel.LOW

    # Medium box: 150x150 = 22,500 px^2 (Ratio = 0.0225 -> MEDIUM)
    assert classifier.classify_detection(150, 150, 1000, 1000) == SeverityLevel.MEDIUM

    # Large box: 300x300 = 90,000 px^2 (Ratio = 0.0900 -> HIGH)
    assert classifier.classify_detection(300, 300, 1000, 1000) == SeverityLevel.HIGH
