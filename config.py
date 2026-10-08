"""
Centralized Configuration Module for Pothole Detection & Road Condition Mapping System.

This module loads environment variables from a .env file and exposes a validated,
singleton Config class. All filesystem paths are resolved using standard pathlib.Path
for seamless cross-platform (Windows / Linux / macOS) compatibility.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory: Absolute path to the repository root
BASE_DIR = Path(__file__).resolve().parent

# Explicitly load .env file from root directory if it exists
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    # If .env does not exist, load default environment or system variables
    load_dotenv()


class Config:
    """Application configuration parameters with sensible defaults."""

    # --------------------------------------------------------------------------
    # Base Directories
    # --------------------------------------------------------------------------
    BASE_DIR: Path = BASE_DIR
    UPLOAD_FOLDER: Path = BASE_DIR / os.getenv("UPLOAD_FOLDER", "data/raw")
    PROCESSED_FOLDER: Path = BASE_DIR / os.getenv("PROCESSED_FOLDER", "data/processed")
    OUTPUT_VIDEO_FOLDER: Path = BASE_DIR / os.getenv("OUTPUT_VIDEO_FOLDER", "outputs/videos")
    OUTPUT_IMAGE_FOLDER: Path = BASE_DIR / os.getenv("OUTPUT_IMAGE_FOLDER", "outputs/images")
    OUTPUT_REPORT_FOLDER: Path = BASE_DIR / os.getenv("OUTPUT_REPORT_FOLDER", "outputs/reports")
    LOG_DIR: Path = BASE_DIR / os.getenv("LOG_DIR", "logs")

    # --------------------------------------------------------------------------
    # MySQL Database Settings
    # --------------------------------------------------------------------------
    DB_HOST: str = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_USER: str = os.getenv("DB_USER", "root")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_NAME: str = os.getenv("DB_NAME", "pothole_detection_db")

    # --------------------------------------------------------------------------
    # Flask REST API Settings
    # --------------------------------------------------------------------------
    FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
    FLASK_HOST: str = os.getenv("FLASK_HOST", "127.0.0.1")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-replace-in-production")
    MAX_CONTENT_LENGTH_MB: int = int(os.getenv("MAX_CONTENT_LENGTH_MB", "200"))
    ALLOWED_EXTENSIONS: set = set(
        ext.strip().lower()
        for ext in os.getenv("ALLOWED_EXTENSIONS", "mp4,avi,mov,mkv").split(",")
    )

    # --------------------------------------------------------------------------
    # Streamlit Dashboard Settings
    # --------------------------------------------------------------------------
    STREAMLIT_SERVER_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://127.0.0.1:5000/api/v1")

    # --------------------------------------------------------------------------
    # Model & Computer Vision Settings
    # --------------------------------------------------------------------------
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/pothole_yolov8n.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    DEVICE: str = os.getenv("DEVICE", "cpu")
    FRAME_SKIP: int = int(os.getenv("FRAME_SKIP", "2"))
    TARGET_INFERENCE_SIZE: int = int(os.getenv("TARGET_INFERENCE_SIZE", "640"))

    # --------------------------------------------------------------------------
    # Severity Thresholds (Area Ratio = bbox_area / frame_area)
    # --------------------------------------------------------------------------
    SEVERITY_LOW_MAX_AREA: float = float(os.getenv("SEVERITY_LOW_MAX_AREA", "0.015"))
    SEVERITY_MED_MAX_AREA: float = float(os.getenv("SEVERITY_MED_MAX_AREA", "0.045"))

    # --------------------------------------------------------------------------
    # Logging Configuration
    # --------------------------------------------------------------------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Path = LOG_DIR / "app.log"

    @classmethod
    def initialize_directories(cls) -> None:
        """Ensure all required runtime directories exist on the local filesystem."""
        directories = [
            cls.UPLOAD_FOLDER,
            cls.PROCESSED_FOLDER,
            cls.OUTPUT_VIDEO_FOLDER,
            cls.OUTPUT_IMAGE_FOLDER,
            cls.OUTPUT_REPORT_FOLDER,
            cls.LOG_DIR,
            cls.BASE_DIR / "models",
            cls.BASE_DIR / "data/sample",
        ]
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)


# Initialize directories upon import to prevent missing folder errors
Config.initialize_directories()
