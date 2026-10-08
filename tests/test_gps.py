"""
Unit Tests for GPS Extractor and ISO-6709 Coordinate Parser.
"""

from pathlib import Path
from app.gps.extractor import GPSCoordinate, GPSExtractor, parse_iso6709_string


def test_parse_iso6709_string_valid():
    lat, lon = parse_iso6709_string("+37.7510-122.4200/")
    assert lat == 37.7510
    assert lon == -122.4200

    lat2, lon2 = parse_iso6709_string("+12.9716+077.5946/")
    assert lat2 == 12.9716
    assert lon2 == 77.5946


def test_parse_iso6709_string_invalid():
    lat, lon = parse_iso6709_string("random_text_without_coordinates")
    assert lat is None
    assert lon is None


def test_gps_coordinate_validation():
    coord_valid = GPSCoordinate(latitude=12.5, longitude=77.5, source="TEST")
    assert coord_valid.is_valid is True

    coord_invalid = GPSCoordinate(latitude=190.0, longitude=77.5, source="TEST")
    assert coord_invalid.is_valid is False

    coord_none = GPSCoordinate(latitude=None, longitude=None, source="UNAVAILABLE")
    assert coord_none.is_valid is False


def test_gps_extractor_missing_file_graceful():
    extractor = GPSExtractor()
    coord = extractor.extract_video_gps(Path("non_existent_video_path.mp4"))
    assert coord.is_valid is False
    assert coord.source == "FILE_NOT_FOUND"
