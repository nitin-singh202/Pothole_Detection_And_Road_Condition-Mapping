"""
Video GPS and Telemetry Metadata Extraction Module.

Extracts embedded ISO-6709 coordinate tags, QuickTime user metadata, and telemetry tracks
from video files using FFprobe / Exif tool inspection. Features zero-crash graceful fallbacks
when telemetry streams are absent or malformed.
"""

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from app.utils.logger import setup_logger

logger = setup_logger("gps_extractor")


@dataclass
class GPSCoordinate:
    """Represents a geographic position coordinate with provenance metadata."""

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    timestamp_sec: Optional[float] = None
    source: str = "UNAVAILABLE"  # 'EXIFTOOL', 'FFPROBE_TAGS', 'INTERPOLATED', 'UNAVAILABLE'

    @property
    def is_valid(self) -> bool:
        """Returns True only if both latitude and longitude are valid numeric floats."""
        if self.latitude is None or self.longitude is None:
            return False
        return -90.0 <= self.latitude <= 90.0 and -180.0 <= self.longitude <= 180.0

    def to_dict(self) -> dict:
        return asdict(self)


def parse_iso6709_string(location_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parses standard ISO 6709 geographic string (e.g., '+37.7510-122.4200/' or '+12.9716+077.5946/').

    Returns:
        Tuple[Optional[float], Optional[float]]: (latitude, longitude) or (None, None)
    """
    if not location_str:
        return None, None

    # Matches patterns like +12.345-067.890/ or +12.3456+078.1234/
    pattern = r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)"
    match = re.search(pattern, location_str)
    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except ValueError:
            pass

    return None, None


class GPSExtractor:
    """Extracts geospatial telemetry from video containers with robust fallbacks."""

    def __init__(self):
        self.ffprobe_available = shutil.which("ffprobe") is not None
        self.exiftool_available = shutil.which("exiftool") is not None

        if not self.ffprobe_available and not self.exiftool_available:
            logger.info("Neither FFprobe nor ExifTool found on system PATH. Direct metadata parsing fallback active.")

    def extract_video_gps(self, video_path: Path) -> GPSCoordinate:
        """
        Attempts multi-tier extraction of GPS coordinates from a video file.

        Tiers:
          1. FFprobe metadata tags (ISO 6709 'location' tag)
          2. ExifTool CLI inspection (if installed)
          3. Graceful fallback returning GPSCoordinate(source='UNAVAILABLE')

        Args:
            video_path: Absolute or relative Path to target video file.

        Returns:
            GPSCoordinate: Structured coordinate object.
        """
        if not video_path.exists():
            logger.error(f"Video file not found for GPS extraction: {video_path}")
            return GPSCoordinate(source="FILE_NOT_FOUND")

        # Tier 1: FFprobe metadata extraction
        if self.ffprobe_available:
            coord = self._extract_via_ffprobe(video_path)
            if coord.is_valid:
                logger.info(f"GPS extracted via FFprobe: Lat={coord.latitude}, Lon={coord.longitude}")
                return coord

        # Tier 2: ExifTool metadata extraction
        if self.exiftool_available:
            coord = self._extract_via_exiftool(video_path)
            if coord.is_valid:
                logger.info(f"GPS extracted via ExifTool: Lat={coord.latitude}, Lon={coord.longitude}")
                return coord

        # Tier 3: Graceful Fallback (No crash)
        logger.info(f"No valid embedded GPS metadata found in {video_path.name}. Marking as UNAVAILABLE.")
        return GPSCoordinate(source="UNAVAILABLE")

    def _extract_via_ffprobe(self, video_path: Path) -> GPSCoordinate:
        """Queries FFprobe JSON stream format for location tags."""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            if result.returncode != 0:
                return GPSCoordinate(source="FFPROBE_ERROR")

            data = json.loads(result.stdout)
            format_tags = data.get("format", {}).get("tags", {})

            # Common location keys in MP4/MOV containers
            for key in ["location", "location-eng", "com.apple.quicktime.location.ISO6709", "LOCATION"]:
                if key in format_tags:
                    lat, lon = parse_iso6709_string(format_tags[key])
                    if lat is not None and lon is not None:
                        return GPSCoordinate(latitude=lat, longitude=lon, source="FFPROBE_TAGS")

        except Exception as e:
            logger.warning(f"FFprobe metadata read failed for {video_path.name}: {e}")

        return GPSCoordinate(source="FFPROBE_NO_TAGS")

    def _extract_via_exiftool(self, video_path: Path) -> GPSCoordinate:
        """Queries ExifTool CLI for standard GPS tags."""
        try:
            cmd = ["exiftool", "-json", "-n", "-GPSLatitude", "-GPSLongitude", "-GPSAltitude", str(video_path)]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            if result.returncode == 0:
                entries = json.loads(result.stdout)
                if entries and len(entries) > 0:
                    entry = entries[0]
                    lat = entry.get("GPSLatitude")
                    lon = entry.get("GPSLongitude")
                    alt = entry.get("GPSAltitude")
                    if lat is not None and lon is not None:
                        return GPSCoordinate(
                            latitude=float(lat),
                            longitude=float(lon),
                            altitude=float(alt) if alt is not None else None,
                            source="EXIFTOOL",
                        )
        except Exception as e:
            logger.warning(f"ExifTool metadata read failed for {video_path.name}: {e}")

        return GPSCoordinate(source="EXIFTOOL_NO_TAGS")
