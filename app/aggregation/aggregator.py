"""
Spatio-Temporal Detection Aggregation Engine.

Solves the multi-frame duplicate detection problem by grouping sequential bounding box
observations of the same physical road pothole into consolidated RoadConditionEvents.
Uses Euclidean centroid distance and temporal proximity clustering.
"""

import math
import uuid
from dataclasses import asdict, dataclass
from typing import List, Optional
from app.detection.detector import DetectionResult
from app.utils.logger import setup_logger

logger = setup_logger("aggregator")

SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "UNKNOWN": 0}


@dataclass
class RoadConditionEvent:
    """Consolidated representation of a distinct physical road defect observed across frames."""

    event_id: str
    video_id: str
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    observation_count: int
    max_confidence: float
    overall_severity: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    max_area_ratio: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class DetectionAggregator:
    """Aggregates frame-by-frame detections into unique road defect events."""

    def __init__(
        self,
        max_centroid_distance_px: float = 75.0,
        max_time_gap_sec: float = 1.0,
    ):
        """
        Args:
            max_centroid_distance_px: Max pixel distance between box centroids to be considered same pothole.
            max_time_gap_sec: Max elapsed time in seconds between detections to maintain an active track.
        """
        self.max_distance = max_centroid_distance_px
        self.max_time_gap = max_time_gap_sec

    def _centroid(self, det: DetectionResult) -> tuple:
        cx = (det.x1 + det.x2) / 2.0
        cy = (det.y1 + det.y2) / 2.0
        return cx, cy

    def _distance(self, c1: tuple, c2: tuple) -> float:
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def aggregate(
        self,
        detections: List[DetectionResult],
        video_id: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> List[RoadConditionEvent]:
        """
        Fuses a chronological list of frame detections into unique road condition events.

        Args:
            detections: List of DetectionResult objects sorted by frame_number/timestamp.
            video_id: Identifier of the source video.
            latitude: Optional global or interpolated GPS latitude.
            longitude: Optional global or interpolated GPS longitude.

        Returns:
            List[RoadConditionEvent]: Unique consolidated road defect instances.
        """
        if not detections:
            return []

        # Ensure detections are sorted chronologically
        sorted_dets = sorted(detections, key=lambda d: (d.frame_number, d.timestamp_sec))

        active_clusters: List[List[DetectionResult]] = []

        for det in sorted_dets:
            det_cx, det_cy = self._centroid(det)
            matched_cluster_idx = -1
            min_dist = float("inf")

            # Check compatibility against the latest observation in each active cluster
            for idx, cluster in enumerate(active_clusters):
                last_det = cluster[-1]
                time_diff = det.timestamp_sec - last_det.timestamp_sec

                if 0 <= time_diff <= self.max_time_gap:
                    last_cx, last_cy = self._centroid(last_det)
                    dist = self._distance((det_cx, det_cy), (last_cx, last_cy))
                    if dist <= self.max_distance and dist < min_dist:
                        min_dist = dist
                        matched_cluster_idx = idx

            if matched_cluster_idx != -1:
                active_clusters[matched_cluster_idx].append(det)
            else:
                # Start a new cluster
                active_clusters.append([det])

        # Convert clusters into consolidated RoadConditionEvents
        events: List[RoadConditionEvent] = []
        for cluster in active_clusters:
            # Find the detection with the highest confidence as the representative observation
            rep_det = max(cluster, key=lambda d: d.confidence)
            # Find the maximum severity reached during the event
            highest_sev = max(
                cluster,
                key=lambda d: SEVERITY_ORDER.get(d.severity.upper(), 0),
            ).severity.upper()

            event = RoadConditionEvent(
                event_id=str(uuid.uuid4()),
                video_id=video_id,
                start_frame=cluster[0].frame_number,
                end_frame=cluster[-1].frame_number,
                start_time_sec=cluster[0].timestamp_sec,
                end_time_sec=cluster[-1].timestamp_sec,
                observation_count=len(cluster),
                max_confidence=rep_det.confidence,
                overall_severity=highest_sev,
                bbox_x1=rep_det.x1,
                bbox_y1=rep_det.y1,
                bbox_x2=rep_det.x2,
                bbox_y2=rep_det.y2,
                max_area_ratio=max(d.area_ratio for d in cluster),
                latitude=latitude,
                longitude=longitude,
            )
            events.append(event)

        logger.info(
            f"Aggregated {len(detections)} frame detections into {len(events)} unique physical road condition events."
        )
        return events
