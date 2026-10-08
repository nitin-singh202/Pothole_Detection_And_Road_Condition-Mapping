"""
Master Video Processing and Road Condition Mapping Pipeline.

Orchestrates frame decoding, YOLOv8 inference, severity classification,
visual overlay rendering, GPS extraction, multi-frame aggregation, and relational storage.
"""

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
import cv2
import numpy as np

from app.aggregation.aggregator import DetectionAggregator, RoadConditionEvent
from app.database.connection import DatabaseConnectionPool
from app.database.repository import DetectionRepository, RoadConditionRepository, VideoRepository
from app.detection.detector import DetectionResult, PotholeDetector
from app.detection.postprocessing import draw_detection_overlay
from app.gps.extractor import GPSCoordinate, GPSExtractor
from app.severity.classifier import SeverityClassifier
from app.utils.logger import setup_logger
from config import Config

logger = setup_logger("pipeline")


@dataclass
class ProcessingResult:
    """Encapsulates the complete result of a video processing run."""

    video_id: str
    input_video_path: str
    annotated_video_path: str
    total_frames: int
    processed_frames: int
    fps: float
    duration_sec: float
    processing_time_sec: float
    effective_fps: float
    has_gps: bool
    gps_latitude: Optional[float]
    gps_longitude: Optional[float]
    total_detections: int
    unique_potholes: int
    severity_breakdown: Dict[str, int]
    detections: List[dict]
    road_conditions: List[dict]
    persisted_to_db: bool

    def to_dict(self) -> dict:
        return asdict(self)


class VideoProcessingPipeline:
    """Coordinates computer vision, GPS parsing, severity analysis, and DB persistence."""

    def __init__(
        self,
        detector: Optional[PotholeDetector] = None,
        severity_classifier: Optional[SeverityClassifier] = None,
        gps_extractor: Optional[GPSExtractor] = None,
        aggregator: Optional[DetectionAggregator] = None,
        frame_skip: Optional[int] = None,
    ):
        self.detector = detector or PotholeDetector()
        self.severity_classifier = severity_classifier or SeverityClassifier()
        self.gps_extractor = gps_extractor or GPSExtractor()
        self.aggregator = aggregator or DetectionAggregator()
        self.frame_skip = frame_skip if frame_skip is not None else Config.FRAME_SKIP

    def process_video(
        self,
        video_path: Path,
        video_id: str,
        output_video_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        save_to_database: bool = True,
    ) -> ProcessingResult:
        """
        Executes end-to-end processing on a single video file.

        Args:
            video_path: Path to the input video file.
            video_id: Unique string identifier for this video.
            output_video_dir: Destination folder for annotated video. Defaults to Config.OUTPUT_VIDEO_FOLDER.
            progress_callback: Optional callback func(progress_fraction, status_text).
            save_to_database: If True, attempts to persist results to MySQL.

        Returns:
            ProcessingResult: Comprehensive processing report.
        """
        start_time = time.time()
        logger.info(f"Initiating processing for video [{video_id}]: {video_path}")

        if not video_path.exists():
            raise FileNotFoundError(f"Input video file not found at: {video_path}")

        # 1. Extract GPS Telemetry Metadata
        if progress_callback:
            progress_callback(0.05, "Extracting GPS and container metadata...")
        gps_coord: GPSCoordinate = self.gps_extractor.extract_video_gps(video_path)

        # 2. Open Video Stream via OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"OpenCV failed to open video stream: {video_path}")
            raise ValueError(f"Cannot decode video stream from {video_path.name}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # Fallback standard FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1
        duration_sec = total_frames / fps

        # Update Video metadata in Database if connected
        db_available = DatabaseConnectionPool.check_connection() if save_to_database else False
        if db_available:
            try:
                VideoRepository.update_video_metadata(
                    video_id=video_id,
                    duration_sec=round(duration_sec, 2),
                    total_frames=total_frames,
                    fps=round(fps, 2),
                    resolution_width=width,
                    resolution_height=height,
                    has_gps=gps_coord.is_valid,
                    gps_latitude=gps_coord.latitude,
                    gps_longitude=gps_coord.longitude,
                )
            except Exception as e:
                logger.warning(f"Failed to update video metadata in DB: {e}")

        # 3. Setup VideoWriter for Annotated Output
        out_dir = output_video_dir or Config.OUTPUT_VIDEO_FOLDER
        out_dir.mkdir(parents=True, exist_ok=True)
        annotated_video_path = out_dir / f"{video_id}_annotated.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))

        frame_idx = 0
        processed_frames_count = 0
        all_detections: List[DetectionResult] = []
        cached_annotated_frame: Optional[np.ndarray] = None

        logger.info(f"Video Info: {width}x{height} @ {fps:.1f} FPS, Total Frames: {total_frames}, Frame Skip: {self.frame_skip}")

        # 4. Frame-by-Frame Processing Loop
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            timestamp_sec = frame_idx / fps

            # Frame skipping logic: Only run YOLO inference on every N-th frame
            should_run_inference = (frame_idx % self.frame_skip == 0) or (frame_idx == 1)

            if should_run_inference:
                processed_frames_count += 1
                # Run YOLO Detector
                frame_dets = self.detector.detect_frame(
                    frame=frame,
                    frame_number=frame_idx,
                    timestamp_sec=timestamp_sec,
                )

                # Classify Severity for each detection
                for det in frame_dets:
                    det.severity = self.severity_classifier.classify_by_area_ratio(
                        det.area_ratio, det.confidence
                    ).value
                    all_detections.append(det)

                # Draw bounding boxes and HUD
                annotated_frame = draw_detection_overlay(
                    frame=frame,
                    detections=[d.to_dict() for d in frame_dets],
                    frame_number=frame_idx,
                    timestamp_sec=timestamp_sec,
                    gps_info=gps_coord.to_dict(),
                )
                cached_annotated_frame = annotated_frame
            else:
                # Reuse last annotated frame HUD or write raw frame with telemetry
                annotated_frame = draw_detection_overlay(
                    frame=frame,
                    detections=[],
                    frame_number=frame_idx,
                    timestamp_sec=timestamp_sec,
                    gps_info=gps_coord.to_dict(),
                )

            writer.write(annotated_frame)

            # Update progress callback every 10 frames
            if progress_callback and frame_idx % 10 == 0:
                progress = min(0.90, 0.10 + (0.80 * (frame_idx / total_frames)))
                progress_callback(progress, f"Processing Frame {frame_idx}/{total_frames}...")

        # Release video handles
        cap.release()
        writer.release()

        # 5. Multi-Frame Spatial-Temporal Aggregation
        if progress_callback:
            progress_callback(0.92, "Aggregating multi-frame detections...")

        road_events: List[RoadConditionEvent] = self.aggregator.aggregate(
            detections=all_detections,
            video_id=video_id,
            latitude=gps_coord.latitude,
            longitude=gps_coord.longitude,
        )

        # 6. Severity Distribution Calculation
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for event in road_events:
            sev = event.overall_severity.upper()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        elapsed_time = time.time() - start_time
        effective_fps = round(frame_idx / elapsed_time, 2) if elapsed_time > 0 else 0.0

        # 7. Relational Persistence to MySQL
        persisted = False
        if db_available and save_to_database:
            try:
                if progress_callback:
                    progress_callback(0.96, "Persisting results to MySQL database...")

                # Save raw detections
                raw_dict_list = [d.to_dict() for d in all_detections]
                for d in raw_dict_list:
                    d["video_id"] = video_id
                DetectionRepository.batch_insert(raw_dict_list)

                # Save aggregated road conditions
                condition_dict_list = [e.to_dict() for e in road_events]
                RoadConditionRepository.batch_insert(condition_dict_list)

                # Update Video record completion status
                VideoRepository.complete_video_processing(
                    video_id=video_id,
                    total_detections=len(all_detections),
                    total_unique_potholes=len(road_events),
                    processing_time_sec=round(elapsed_time, 2),
                    annotated_video_path=str(annotated_video_path),
                )
                persisted = True
                logger.info("Successfully persisted detections and road conditions to MySQL.")
            except Exception as e:
                logger.error(f"Database persistence error: {e}")

        if progress_callback:
            progress_callback(1.0, "Video processing completed successfully!")

        result = ProcessingResult(
            video_id=video_id,
            input_video_path=str(video_path),
            annotated_video_path=str(annotated_video_path),
            total_frames=frame_idx,
            processed_frames=processed_frames_count,
            fps=round(fps, 2),
            duration_sec=round(duration_sec, 2),
            processing_time_sec=round(elapsed_time, 2),
            effective_fps=effective_fps,
            has_gps=gps_coord.is_valid,
            gps_latitude=gps_coord.latitude,
            gps_longitude=gps_coord.longitude,
            total_detections=len(all_detections),
            unique_potholes=len(road_events),
            severity_breakdown=severity_counts,
            detections=[d.to_dict() for d in all_detections],
            road_conditions=[e.to_dict() for e in road_events],
            persisted_to_db=persisted,
        )

        logger.info(
            f"Completed video [{video_id}]: {len(all_detections)} detections -> "
            f"{len(road_events)} unique potholes in {elapsed_time:.2f}s ({effective_fps} FPS)."
        )
        return result
