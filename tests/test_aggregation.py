"""
Unit Tests for Spatio-Temporal Detection Aggregator Module.
"""

from app.aggregation.aggregator import DetectionAggregator
from app.detection.detector import DetectionResult


def test_aggregation_merges_consecutive_frames():
    aggregator = DetectionAggregator(max_centroid_distance_px=50.0, max_time_gap_sec=0.5)

    # 3 consecutive detections of the same pothole
    dets = [
        DetectionResult(0, "pothole", 0.70, 100, 100, 150, 150, 50, 50, 0.01, 1, 0.033, "LOW"),
        DetectionResult(0, "pothole", 0.85, 102, 101, 152, 151, 50, 50, 0.01, 2, 0.066, "LOW"),
        DetectionResult(0, "pothole", 0.80, 105, 103, 155, 153, 50, 50, 0.01, 3, 0.099, "LOW"),
    ]

    events = aggregator.aggregate(dets, video_id="test_vid_001")
    assert len(events) == 1
    event = events[0]
    assert event.observation_count == 3
    assert event.max_confidence == 0.85
    assert event.start_frame == 1
    assert event.end_frame == 3


def test_aggregation_separates_distant_potholes():
    aggregator = DetectionAggregator(max_centroid_distance_px=50.0, max_time_gap_sec=0.5)

    # 2 detections far apart in pixels
    dets = [
        DetectionResult(0, "pothole", 0.80, 50, 50, 100, 100, 50, 50, 0.01, 1, 0.033, "LOW"),
        DetectionResult(0, "pothole", 0.90, 500, 400, 550, 450, 50, 50, 0.01, 1, 0.033, "LOW"),
    ]

    events = aggregator.aggregate(dets, video_id="test_vid_002")
    assert len(events) == 2
