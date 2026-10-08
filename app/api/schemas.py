"""
API Request and Response Data Validation Schemas.

Uses Pydantic to enforce type safety, input validation, and clean serialization
across the Flask REST API surface.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """API health status schema."""

    model_config = {"protected_namespaces": ()}

    status: str = "healthy"
    version: str = "1.0.0"
    database_connected: bool
    model_loaded: bool


class VideoUploadResponse(BaseModel):
    """Video upload response schema."""

    success: bool
    message: str
    video_id: str
    original_filename: str
    stored_filename: str
    file_path: str


class ProcessVideoRequest(BaseModel):
    """Payload for invoking pipeline on an uploaded video."""

    video_id: str = Field(..., description="Unique ID of the uploaded video")
    confidence_threshold: Optional[float] = Field(None, ge=0.05, le=1.0)
    frame_skip: Optional[int] = Field(None, ge=1, le=30)
    save_to_db: Optional[bool] = True


class ProcessVideoResponse(BaseModel):
    """Structured response for video processing completion."""

    success: bool
    video_id: str
    total_frames: int
    duration_sec: float
    processing_time_sec: float
    effective_fps: float
    has_gps: bool
    gps_latitude: Optional[float]
    gps_longitude: Optional[float]
    total_detections: int
    unique_potholes: int
    severity_breakdown: Dict[str, int]
    annotated_video_url: str


class RoadConditionSummaryResponse(BaseModel):
    """Aggregated road defect statistics response."""

    total_events: int
    low_count: int
    medium_count: int
    high_count: int
    avg_confidence: float
