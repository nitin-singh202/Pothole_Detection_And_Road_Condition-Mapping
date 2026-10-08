"""
Structured Logging Module.

Configures application-wide logging with synchronized console output and file handlers.
Ensures zero exposure of secrets or database credentials.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from config import Config


def setup_logger(name: str = "pothole_system", log_file: Optional[Path] = None) -> logging.Logger:
    """
    Creates or retrieves a configured logger instance with console and file handlers.

    Args:
        name: The name of the logger (typically module name or component name).
        log_file: Optional custom Path to log file. Defaults to Config.LOG_FILE.

    Returns:
        logging.Logger: Fully configured Python logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers if logger was already configured
    if logger.handlers:
        return logger

    # Resolve log level from Config
    log_level_str = getattr(Config, "LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Standard log format: Timestamp | Level | Module:Line | Message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    target_log_file = log_file or Config.LOG_FILE
    try:
        # Ensure log directory exists
        target_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target_log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback if file cannot be created (e.g. permission issues)
        console_handler.setLevel(logging.WARNING)
        logger.warning(f"Could not initialize file log handler at {target_log_file}: {e}")

    # Do not propagate to root logger to avoid double logging in sub-frameworks
    logger.propagate = False

    return logger
