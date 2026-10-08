"""
Flask REST API Endpoints and Application Factory.

Exposes RESTful endpoints for video uploading, automated pipeline processing,
detection inspection, severity summaries, and geospatial coordinate queries.
"""

from pathlib import Path
from typing import Optional
from flask import Blueprint, Flask, jsonify, request, send_file
from flask_cors import CORS

from app.api.schemas import ProcessVideoRequest
from app.database.connection import DatabaseConnectionPool
from app.database.repository import (
    DetectionRepository,
    RoadConditionRepository,
    VideoRepository,
)
from app.detection.detector import PotholeDetector
from app.processing.pipeline import VideoProcessingPipeline
from app.utils.file_utils import get_unique_filename, is_allowed_file
from app.utils.logger import setup_logger
from config import Config

logger = setup_logger("api_routes")

# Create Flask Blueprint with /api/v1 prefix
api_bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")

# Lazy-loaded pipeline singleton
_pipeline_instance: Optional[VideoProcessingPipeline] = None


def get_pipeline() -> VideoProcessingPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = VideoProcessingPipeline()
    return _pipeline_instance


# ------------------------------------------------------------------------------
# 1. Health Check Endpoint
# ------------------------------------------------------------------------------
@api_bp.route("/health", methods=["GET"])
def health_check():
    """
    GET /api/v1/health
    Returns system operational status, DB connectivity, and model availability.
    """
    db_ok = DatabaseConnectionPool.check_connection()
    model_exists = Path(Config.MODEL_PATH).exists() or Path("yolov8n.pt").exists()

    status_code = 200 if (db_ok or not Config.DB_PASSWORD) else 200
    return (
        jsonify(
            {
                "status": "healthy",
                "version": "1.0.0",
                "database_connected": db_ok,
                "model_available": model_exists,
                "device": Config.DEVICE,
            }
        ),
        status_code,
    )


# ------------------------------------------------------------------------------
# 2. Video Upload Endpoint
# ------------------------------------------------------------------------------
@api_bp.route("/videos/upload", methods=["POST"])
def upload_video():
    """
    POST /api/v1/videos/upload
    Ingests and validates raw video files, storing them safely in data/raw/.
    """
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No 'video' file field present in multipart request."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not is_allowed_file(file.filename):
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Unsupported file type. Permitted: {list(Config.ALLOWED_EXTENSIONS)}",
                }
            ),
            400,
        )

    # Generate safe unique filename
    video_id, stored_filename = get_unique_filename(file.filename)
    dest_path = Config.UPLOAD_FOLDER / stored_filename

    try:
        file.save(str(dest_path))
        logger.info(f"Video uploaded successfully: {file.filename} -> {dest_path.name} [ID: {video_id}]")

        # Register video in Database if active
        if DatabaseConnectionPool.check_connection():
            VideoRepository.create_video(
                video_id=video_id,
                original_filename=file.filename,
                stored_filename=stored_filename,
                status="PENDING",
            )

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Video uploaded successfully and queued for processing.",
                    "video_id": video_id,
                    "original_filename": file.filename,
                    "stored_filename": stored_filename,
                }
            ),
            201,
        )

    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        return jsonify({"success": False, "error": f"File save error: {str(e)}"}), 500


# ------------------------------------------------------------------------------
# 3. Video Processing Endpoint
# ------------------------------------------------------------------------------
@api_bp.route("/videos/process", methods=["POST"])
def process_video():
    """
    POST /api/v1/videos/process
    Triggers end-to-end computer vision and mapping pipeline on an uploaded video.
    """
    data = request.get_json() or {}
    video_id = data.get("video_id")

    if not video_id:
        return jsonify({"success": False, "error": "'video_id' parameter is required."}), 400

    # Locate video file in data/raw/
    matched_files = list(Config.UPLOAD_FOLDER.glob(f"*{video_id.replace('-', '')[:8]}*"))
    if not matched_files:
        matched_files = list(Config.UPLOAD_FOLDER.glob(f"*{video_id}*"))

    if not matched_files:
        return jsonify({"success": False, "error": f"Video with ID '{video_id}' not found in upload directory."}), 404

    video_path = matched_files[0]

    try:
        pipeline = get_pipeline()
        result = pipeline.process_video(
            video_path=video_path,
            video_id=video_id,
            save_to_database=data.get("save_to_db", True),
        )

        return (
            jsonify(
                {
                    "success": True,
                    "video_id": result.video_id,
                    "total_frames": result.total_frames,
                    "processed_frames": result.processed_frames,
                    "duration_sec": result.duration_sec,
                    "processing_time_sec": result.processing_time_sec,
                    "effective_fps": result.effective_fps,
                    "has_gps": result.has_gps,
                    "gps_latitude": result.gps_latitude,
                    "gps_longitude": result.gps_longitude,
                    "total_detections": result.total_detections,
                    "unique_potholes": result.unique_potholes,
                    "severity_breakdown": result.severity_breakdown,
                    "annotated_video_url": f"/api/v1/videos/{video_id}/download-annotated",
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Pipeline processing failed for video [{video_id}]: {e}")
        if DatabaseConnectionPool.check_connection():
            try:
                VideoRepository.mark_video_failed(video_id, str(e))
            except Exception:
                pass
        return jsonify({"success": False, "error": f"Processing failure: {str(e)}"}), 500


# ------------------------------------------------------------------------------
# 4. Video Inspection & Download Endpoints
# ------------------------------------------------------------------------------
@api_bp.route("/videos/<video_id>", methods=["GET"])
def get_video_info(video_id: str):
    """GET /api/v1/videos/<video_id> - Fetches stored video record metadata."""
    if not DatabaseConnectionPool.check_connection():
        return jsonify({"error": "Database service unavailable."}), 503

    record = VideoRepository.get_video_by_id(video_id)
    if not record:
        return jsonify({"error": "Video not found."}), 404
    return jsonify({"success": True, "video": record}), 200


@api_bp.route("/videos/<video_id>/detections", methods=["GET"])
def get_video_detections(video_id: str):
    """GET /api/v1/videos/<video_id>/detections - Retrieves frame-level detections."""
    if not DatabaseConnectionPool.check_connection():
        return jsonify({"error": "Database service unavailable."}), 503

    dets = DetectionRepository.get_by_video_id(video_id)
    return jsonify({"success": True, "count": len(dets), "detections": dets}), 200


@api_bp.route("/videos/<video_id>/conditions", methods=["GET"])
def get_video_conditions(video_id: str):
    """GET /api/v1/videos/<video_id>/conditions - Retrieves unique aggregated road defects."""
    if not DatabaseConnectionPool.check_connection():
        return jsonify({"error": "Database service unavailable."}), 503

    conds = RoadConditionRepository.get_by_video_id(video_id)
    return jsonify({"success": True, "count": len(conds), "conditions": conds}), 200


@api_bp.route("/videos/<video_id>/download-annotated", methods=["GET"])
def download_annotated_video(video_id: str):
    """GET /api/v1/videos/<video_id>/download-annotated - Serves rendered output video file."""
    annotated_path = Config.OUTPUT_VIDEO_FOLDER / f"{video_id}_annotated.mp4"
    if not annotated_path.exists():
        return jsonify({"error": "Annotated video not found."}), 404
    return send_file(str(annotated_path), mimetype="video/mp4", as_attachment=False)


# ------------------------------------------------------------------------------
# 5. Global Road Conditions & Geospatial Endpoints
# ------------------------------------------------------------------------------
@api_bp.route("/road-conditions", methods=["GET"])
def list_road_conditions():
    """GET /api/v1/road-conditions?severity=HIGH&limit=50"""
    severity = request.args.get("severity")
    limit = int(request.args.get("limit", 100))

    if not DatabaseConnectionPool.check_connection():
        return jsonify({"error": "Database service unavailable."}), 503

    conds = RoadConditionRepository.get_all_conditions(severity=severity, limit=limit)
    return jsonify({"success": True, "count": len(conds), "conditions": conds}), 200


@api_bp.route("/road-conditions/summary", methods=["GET"])
def road_condition_summary():
    """GET /api/v1/road-conditions/summary - Returns severity breakdown totals."""
    if not DatabaseConnectionPool.check_connection():
        return jsonify({"error": "Database service unavailable."}), 503

    stats = RoadConditionRepository.get_summary_statistics()
    return jsonify({"success": True, "summary": stats}), 200


@api_bp.route("/road-conditions/map", methods=["GET"])
def get_map_points():
    """GET /api/v1/road-conditions/map - Returns all geocoded road defects for map visualization."""
    if not DatabaseConnectionPool.check_connection():
        return jsonify({"error": "Database service unavailable."}), 503

    points = RoadConditionRepository.get_geolocated_points()
    return jsonify({"success": True, "count": len(points), "points": points}), 200


# ------------------------------------------------------------------------------
# Application Factory
# ------------------------------------------------------------------------------
def create_app() -> Flask:
    """Creates and configures the Flask application instance."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = Config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH_MB * 1024 * 1024

    # Enable Cross-Origin Resource Sharing for Streamlit integration
    CORS(app)

    # Register API Blueprint
    app.register_blueprint(api_bp)

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({"success": False, "error": f"File exceeds maximum upload size of {Config.MAX_CONTENT_LENGTH_MB} MB."}), 413

    @app.errorhandler(404)
    def resource_not_found(error):
        return jsonify({"success": False, "error": "Requested API endpoint not found."}), 404

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({"success": False, "error": "Internal server error occurred."}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    logger.info(f"Starting Flask REST API Server on http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=True)
