"""
Computer Vision Post-Processing and Video Annotation Rendering.

Renders bounding boxes, severity badges, confidence tags, and telemetry headers
onto processed video frames with high-contrast text backgrounds.
"""

from typing import List, Optional
import cv2
import numpy as np

# BGR Color Palette for Severity Levels
SEVERITY_COLORS = {
    "LOW": (30, 200, 30),       # Green
    "MEDIUM": (0, 165, 255),    # Orange
    "HIGH": (30, 30, 235),      # Red
    "UNKNOWN": (200, 200, 200), # Gray
}


def draw_detection_overlay(
    frame: np.ndarray,
    detections: List[dict],
    frame_number: int,
    timestamp_sec: float,
    gps_info: Optional[dict] = None,
) -> np.ndarray:
    """
    Renders visual bounding boxes, labels, and telemetry banner onto a video frame.

    Args:
        frame: Original BGR video frame from OpenCV.
        detections: List of detection dictionaries containing bbox coordinates and severity.
        frame_number: Current frame index.
        timestamp_sec: Current video timestamp in seconds.
        gps_info: Optional dict with 'latitude', 'longitude', 'source'.

    Returns:
        np.ndarray: Annotated video frame.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # 1. Render Each Bounding Box and Severity Badge
    for det in detections:
        x1 = max(0, int(det["x1"]))
        y1 = max(0, int(det["y1"]))
        x2 = min(w, int(det["x2"]))
        y2 = min(h, int(det["y2"]))

        confidence = det.get("confidence", 0.0)
        severity = det.get("severity", "LOW").upper()
        color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["LOW"])

        # Bounding box rectangle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label tag: "POTHOLE | 87% | HIGH"
        label_text = f"POTHOLE {confidence * 100:.0f}% [{severity}]"
        font_scale = 0.5
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Draw filled background for label text above bbox (or inside if at top edge)
        text_y = max(y1, th + 8)
        cv2.rectangle(
            canvas,
            (x1, text_y - th - 6),
            (x1 + tw + 6, text_y + 2),
            color,
            cv2.FILLED,
        )
        # White text over colored badge
        cv2.putText(
            canvas,
            label_text,
            (x1 + 3, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    # 2. Render Top HUD Telemetry Banner
    banner_height = 36
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), cv2.FILLED)
    # Blend semi-transparent banner
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Left Telemetry: Time & Frame Index
    time_str = f"Time: {timestamp_sec:.2f}s | Frame: #{frame_number} | Detections: {len(detections)}"
    cv2.putText(canvas, time_str, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Right Telemetry: GPS Location or Fallback
    if gps_info and gps_info.get("latitude") is not None and gps_info.get("longitude") is not None:
        gps_str = f"GPS: {gps_info['latitude']:.5f}, {gps_info['longitude']:.5f}"
        gps_color = (100, 255, 100)  # Light Green
    else:
        gps_str = "GPS: Unavailable"
        gps_color = (160, 160, 160)  # Dim Gray

    (gw, gh), _ = cv2.getTextSize(gps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(canvas, gps_str, (w - gw - 15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, gps_color, 1, cv2.LINE_AA)

    return canvas
