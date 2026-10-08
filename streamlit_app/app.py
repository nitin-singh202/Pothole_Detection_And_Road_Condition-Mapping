"""
Streamlit Web Dashboard - Master Entrypoint & Overview Page.
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Ensure repository root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.database.connection import DatabaseConnectionPool
from app.database.repository import RoadConditionRepository, VideoRepository
from config import Config

st.set_page_config(
    page_title="Pothole Detection & Road Condition Mapping System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛣️ Pothole Detection & Road Condition Mapping System")
st.markdown(
    """
    **End-to-End Computer Vision & Geospatial Infrastructure for Automated Road Distress Assessment.**
    Processes vehicle video streams, detects potholes using **YOLOv8**, extracts **GPS** coordinates, 
    estimates surface severity, and persists structured audit logs in **MySQL**.
    """
)
st.divider()

# System Status Indicators
col1, col2, col3, col4 = st.columns(4)

with col1:
    db_status = DatabaseConnectionPool.check_connection()
    if db_status:
        st.success("🟢 MySQL Database: Connected")
    else:
        st.warning("🟡 MySQL: Disconnected (Local Mode)")

with col2:
    model_path = Path(Config.MODEL_PATH)
    if model_path.exists() or Path("yolov8n.pt").exists():
        st.success(f"🟢 YOLOv8: Ready ({Config.DEVICE.upper()})")
    else:
        st.info("🔵 YOLOv8: Using yolov8n base")

with col3:
    st.info(f"⚙️ Frame Skip: 1 in {Config.FRAME_SKIP}")

with col4:
    st.info(f"🎯 Conf Threshold: {Config.CONFIDENCE_THRESHOLD}")

st.divider()

# Global Road Distress Summary Cards
if db_status:
    try:
        stats = RoadConditionRepository.get_summary_statistics()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Flagged Defects", stats["total_events"])
        c2.metric("High Severity Alerts", stats["high_count"], delta_color="inverse")
        c3.metric("Medium Severity Defects", stats["medium_count"])
        c4.metric("Low Severity Anomalies", stats["low_count"])
    except Exception as e:
        st.info("Summary metrics will populate once videos are processed into the database.")
else:
    st.info("💡 Start the MySQL database and run the video processing pipeline to aggregate multi-video metrics.")

st.subheader("System Architecture & Workflow")
st.markdown(
    """
    ```
    VIDEO INPUT (.mp4, .avi) ──► OPENCV FRAME DECODER ──► YOLOV8 OBJECT DETECTOR
                                                                │
                                                                ▼
    STREAMLIT MAP & DASHBOARD ◄── FLASK REST API ◄── MYSQL ◄── SEVERITY + GPS + AGGREGATION
    ```
    
    ### How to Use this Dashboard:
    1. **Upload & Process (`Page 1`):** Upload road survey video files, configure inference settings, and run the real-time detection pipeline.
    2. **Detections Explorer (`Page 2`):** Inspect frame-by-frame detections, bounding box overlays, and aggregated defect logs.
    3. **Road Map (`Page 3`):** Explore geospatial Leaflet map with color-coded severity markers for GPS-tagged road hazards.
    """
)
