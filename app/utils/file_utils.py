"""
File and Directory Utilities Module.

Provides robust validation, filename sanitization, and path helper functions
to protect against path traversal vulnerabilities and ensure Windows path compatibility.
"""

import os
import re
import uuid
from pathlib import Path
from typing import Set, Tuple
from config import Config


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes an untrusted filename by stripping path delimiters and special characters.

    Args:
        filename: The raw filename received from an upload or input.

    Returns:
        str: A clean, safe filename containing only alphanumeric characters, dashes, underscores, and dots.
    """
    # Extract only the base name (prevents ../ path traversal attacks)
    base_name = os.path.basename(filename)
    # Remove any character that is not alphanumeric, dot, underscore, or hyphen
    cleaned = re.sub(r"[^\w\.-]", "_", base_name)
    # Ensure it's not empty or just dots
    if not cleaned or cleaned.replace(".", "") == "":
        cleaned = f"upload_{uuid.uuid4().hex[:8]}"
    return cleaned


def is_allowed_file(filename: str, allowed_extensions: Set[str] = None) -> bool:
    """
    Verifies whether a given filename has a permitted file extension.

    Args:
        filename: Name of the file to inspect.
        allowed_extensions: Optional set of allowed lowercased extensions.
                           Defaults to Config.ALLOWED_EXTENSIONS.

    Returns:
        bool: True if extension is permitted, False otherwise.
    """
    if allowed_extensions is None:
        allowed_extensions = Config.ALLOWED_EXTENSIONS

    if "." not in filename:
        return False

    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_extensions


def get_unique_filename(original_filename: str) -> Tuple[str, str]:
    """
    Generates a collision-resistant unique filename using a UUID4 prefix while preserving
    the sanitized original base name and extension.

    Args:
        original_filename: The raw uploaded filename.

    Returns:
        Tuple[str, str]: (unique_video_id, unique_filename)
            Example: ("a1b2c3d4-e5f6-7890-1234-56789abcdef0", "a1b2c3d4_road_survey.mp4")
    """
    video_id = str(uuid.uuid4())
    sanitized = sanitize_filename(original_filename)
    short_uuid = video_id.replace("-", "")[:8]
    unique_filename = f"{short_uuid}_{sanitized}"
    return video_id, unique_filename


def ensure_dir_exists(path: Path) -> Path:
    """
    Ensures that a directory exists, creating parents if necessary.

    Args:
        path: Path object representing the target directory.

    Returns:
        Path: The same Path object for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
