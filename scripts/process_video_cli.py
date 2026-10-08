"""
Command-Line Interface (CLI) for Standalone Video Processing.

Enables headless, batch execution of the pothole detection and mapping pipeline
from the terminal without launching web servers or GUI dashboards.
"""

import argparse
import sys
import uuid
from pathlib import Path

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.detection.detector import PotholeDetector
from app.processing.pipeline import VideoProcessingPipeline
from app.utils.logger import setup_logger
from config import Config

logger = setup_logger("process_video_cli")


def main():
    parser = argparse.ArgumentParser(description="Run Pothole Detection Pipeline on a local video file.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file (.mp4, .avi, etc.)")
    parser.add_argument("--model", type=str, default=Config.MODEL_PATH, help="Path to YOLO weights file")
    parser.add_argument("--conf", type=float, default=Config.CONFIDENCE_THRESHOLD, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--frame-skip", type=int, default=Config.FRAME_SKIP, help="Frame skipping step")
    parser.add_argument("--no-db", action="store_true", help="Skip database persistence")
    args = parser.parse_args()

    input_path = Path(args.video).resolve()
    if not input_path.exists():
        logger.error(f"Input video not found: {input_path}")
        sys.exit(1)

    video_id = str(uuid.uuid4())
    logger.info(f"Generated Job Video ID: {video_id}")

    detector = PotholeDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
    )
    pipeline = VideoProcessingPipeline(
        detector=detector,
        frame_skip=args.frame_skip,
    )

    def console_progress(progress: float, msg: str):
        pct = int(progress * 100)
        sys.stdout.write(f"\r[{pct:3d}%] {msg:<40}")
        sys.stdout.flush()

    try:
        result = pipeline.process_video(
            video_path=input_path,
            video_id=video_id,
            progress_callback=console_progress,
            save_to_database=not args.no_db,
        )

        print("\n\n" + "=" * 70)
        print("                     CLI PROCESSING SUMMARY")
        print("=" * 70)
        print(f"Video ID            : {result.video_id}")
        print(f"Total Video Frames  : {result.total_frames}")
        print(f"Processed Frames    : {result.processed_frames}")
        print(f"Video Duration      : {result.duration_sec:.2f} s")
        print(f"Processing Time     : {result.processing_time_sec:.2f} s")
        print(f"Effective Speed     : {result.effective_fps:.1f} FPS")
        print(f"GPS Available       : {result.has_gps}")
        if result.has_gps:
            print(f"GPS Coordinates     : Lat={result.gps_latitude}, Lon={result.gps_longitude}")
        print(f"Total Detections    : {result.total_detections}")
        print(f"Unique Potholes     : {result.unique_potholes}")
        print(f"Severity Breakdown  : {result.severity_breakdown}")
        print(f"Annotated Video File: {result.annotated_video_path}")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
