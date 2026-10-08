"""
Database Repository Layer.

Implements the Repository Pattern for parameterized CRUD operations on videos,
raw frame-level detections, and aggregated road-condition events.
Ensures zero SQL injection vulnerabilities through 100% parameterized statements.
"""

from typing import Any, Dict, List, Optional
from app.database.connection import get_db_connection
from app.utils.logger import setup_logger

logger = setup_logger("repository")


class VideoRepository:
    """Handles persistence and queries for the 'videos' table."""

    @staticmethod
    def create_video(
        video_id: str,
        original_filename: str,
        stored_filename: str,
        status: str = "PENDING",
    ) -> bool:
        sql = """
            INSERT INTO videos (video_id, original_filename, stored_filename, status)
            VALUES (%s, %s, %s, %s)
        """
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (video_id, original_filename, stored_filename, status))
        return True

    @staticmethod
    def update_video_metadata(
        video_id: str,
        duration_sec: float,
        total_frames: int,
        fps: float,
        resolution_width: int,
        resolution_height: int,
        has_gps: bool = False,
        gps_latitude: Optional[float] = None,
        gps_longitude: Optional[float] = None,
    ) -> bool:
        sql = """
            UPDATE videos
            SET duration_sec = %s, total_frames = %s, fps = %s,
                resolution_width = %s, resolution_height = %s,
                has_gps = %s, gps_latitude = %s, gps_longitude = %s
            WHERE video_id = %s
        """
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        duration_sec,
                        total_frames,
                        fps,
                        resolution_width,
                        resolution_height,
                        has_gps,
                        gps_latitude,
                        gps_longitude,
                        video_id,
                    ),
                )
        return True

    @staticmethod
    def complete_video_processing(
        video_id: str,
        total_detections: int,
        total_unique_potholes: int,
        processing_time_sec: float,
        annotated_video_path: str,
    ) -> bool:
        sql = """
            UPDATE videos
            SET status = 'COMPLETED',
                total_detections = %s,
                total_unique_potholes = %s,
                processing_time_sec = %s,
                annotated_video_path = %s
            WHERE video_id = %s
        """
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        total_detections,
                        total_unique_potholes,
                        processing_time_sec,
                        annotated_video_path,
                        video_id,
                    ),
                )
        return True

    @staticmethod
    def mark_video_failed(video_id: str, error_message: str) -> bool:
        sql = "UPDATE videos SET status = 'FAILED', error_message = %s WHERE video_id = %s"
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (error_message, video_id))
        return True

    @staticmethod
    def get_video_by_id(video_id: str) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM videos WHERE video_id = %s"
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (video_id,))
                return cursor.fetchone()

    @staticmethod
    def list_all_videos(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM videos ORDER BY created_at DESC LIMIT %s OFFSET %s"
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (limit, offset))
                return cursor.fetchall()


class DetectionRepository:
    """Handles batch operations and queries for the 'detections' table."""

    @staticmethod
    def batch_insert(detections: List[Dict[str, Any]]) -> int:
        if not detections:
            return 0

        sql = """
            INSERT INTO detections (
                video_id, frame_number, timestamp_sec, class_id, class_name,
                confidence, severity, bbox_x1, bbox_y1, bbox_x2, bbox_y2, area_ratio
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = [
            (
                d["video_id"],
                d["frame_number"],
                d["timestamp_sec"],
                d.get("class_id", 0),
                d.get("class_name", "pothole"),
                d["confidence"],
                d["severity"],
                d["x1"],
                d["y1"],
                d["x2"],
                d["y2"],
                d["area_ratio"],
            )
            for d in detections
        ]

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, values)
        return len(values)

    @staticmethod
    def get_by_video_id(video_id: str) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM detections WHERE video_id = %s ORDER BY frame_number ASC"
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (video_id,))
                return cursor.fetchall()


class RoadConditionRepository:
    """Handles operations on aggregated unique road-condition events."""

    @staticmethod
    def batch_insert(conditions: List[Dict[str, Any]]) -> int:
        if not conditions:
            return 0

        sql = """
            INSERT INTO road_conditions (
                event_id, video_id, start_frame, end_frame, start_time_sec, end_time_sec,
                observation_count, max_confidence, overall_severity,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2, max_area_ratio,
                latitude, longitude, flag_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = [
            (
                c["event_id"],
                c["video_id"],
                c["start_frame"],
                c["end_frame"],
                c["start_time_sec"],
                c["end_time_sec"],
                c["observation_count"],
                c["max_confidence"],
                c["overall_severity"],
                c["bbox_x1"],
                c["bbox_y1"],
                c["bbox_x2"],
                c["bbox_y2"],
                c["max_area_ratio"],
                c.get("latitude"),
                c.get("longitude"),
                c.get("flag_status", "FLAGGED"),
            )
            for c in conditions
        ]

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, values)
        return len(values)

    @staticmethod
    def get_by_video_id(video_id: str) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM road_conditions WHERE video_id = %s ORDER BY start_frame ASC"
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (video_id,))
                return cursor.fetchall()

    @staticmethod
    def get_all_conditions(
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if severity:
            sql = "SELECT * FROM road_conditions WHERE overall_severity = %s ORDER BY created_at DESC LIMIT %s"
            params = (severity.upper(), limit)
        else:
            sql = "SELECT * FROM road_conditions ORDER BY created_at DESC LIMIT %s"
            params = (limit,)

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()

    @staticmethod
    def get_summary_statistics() -> Dict[str, Any]:
        """Calculates global severity breakdown and total defect counts."""
        sql_counts = """
            SELECT
                COUNT(*) as total_events,
                SUM(CASE WHEN overall_severity = 'LOW' THEN 1 ELSE 0 END) as low_count,
                SUM(CASE WHEN overall_severity = 'MEDIUM' THEN 1 ELSE 0 END) as medium_count,
                SUM(CASE WHEN overall_severity = 'HIGH' THEN 1 ELSE 0 END) as high_count,
                AVG(max_confidence) as avg_confidence
            FROM road_conditions
        """
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql_counts)
                row = cursor.fetchone() or {}
                return {
                    "total_events": int(row.get("total_events") or 0),
                    "low_count": int(row.get("low_count") or 0),
                    "medium_count": int(row.get("medium_count") or 0),
                    "high_count": int(row.get("high_count") or 0),
                    "avg_confidence": round(float(row.get("avg_confidence") or 0.0), 4),
                }

    @staticmethod
    def get_geolocated_points() -> List[Dict[str, Any]]:
        """Retrieves all road condition events that have valid GPS coordinates."""
        sql = """
            SELECT event_id, video_id, overall_severity, max_confidence,
                   observation_count, latitude, longitude, created_at
            FROM road_conditions
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY created_at DESC
        """
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                return cursor.fetchall()
