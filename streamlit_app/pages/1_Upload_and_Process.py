"""
Streamlit Page 1: Video Upload & Pipeline Processing Runner.
"""

import sys
import time
from pathlib import Path
import streamlit as st

# Ensure repository root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.detection.detector import PotholeDetector
from app.processing.pipeline import VideoProcessingPipeline
from app.severity.classifier import SeverityClassifier
from app.utils.file_utils import get_unique_filename, is_allowed_file
from config import Config
from streamlit_app.components.charts import render_confidence_histogram, render_severity_pie_chart

st.set_page_config(page_title="Upload & Process Video", page_icon="📹", layout="wide")

st.title("📹 Video Ingestion & Processing Pipeline")
st.markdown("Upload road survey videos, tune inference thresholds, and trigger automated pothole detection.")
st.divider()

# Left Sidebar / Top Config Controls
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("⚙️ Processing Parameters")
    conf_thresh = st.slider("Detection Confidence Threshold", 0.10, 0.90, float(Config.CONFIDENCE_THRESHOLD), 0.05)
    frame_skip = st.slider("Frame Skip Factor (Process 1 frame every N)", 1, 10, int(Config.FRAME_SKIP), 1)
    save_db = st.checkbox("Save Results to MySQL Database", value=True)

    uploaded_file = st.file_uploader(
        "Choose Road Video File",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload dashboard camera footage or roadside survey recording.",
    )

with col_right:
    if uploaded_file is not None:
        st.subheader("Preview Input Video")
        st.video(uploaded_file)

        if st.button("🚀 Run Detection Pipeline", type="primary", use_container_width=True):
            # 1. Save uploaded file to data/raw/
            video_id, stored_filename = get_unique_filename(uploaded_file.name)
            raw_save_path = Config.UPLOAD_FOLDER / stored_filename

            with open(raw_save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"Video uploaded successfully. Video ID: `{video_id}`")

            # 2. Setup Progress Bar
            progress_bar = st.progress(0, text="Initializing YOLOv8 Engine...")

            def update_progress(frac: float, text: str):
                progress_bar.progress(int(frac * 100), text=text)

            # 3. Initialize Decoupled Pipeline
            detector = PotholeDetector(confidence_threshold=conf_thresh)
            pipeline = VideoProcessingPipeline(detector=detector, frame_skip=frame_skip)

            try:
                # Run Pipeline
                result = pipeline.process_video(
                    video_path=raw_save_path,
                    video_id=video_id,
                    progress_callback=update_progress,
                    save_to_database=save_db,
                )

                st.balloons()
                st.subheader("🎉 Processing Complete!")

                # Key Metrics Cards
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Detections", result.total_detections)
                m2.metric("Unique Potholes", result.unique_potholes)
                m3.metric("Processing Time", f"{result.processing_time_sec}s")
                m4.metric("Effective FPS", f"{result.effective_fps}")
                m5.metric("GPS Attached", "Yes" if result.has_gps else "No")

                st.divider()

                # Display Annotated Output Video
                st.subheader("🎬 Annotated Output Video")
                annotated_path = Path(result.annotated_video_path)
                if annotated_path.exists():
                    st.video(str(annotated_path))
                else:
                    st.warning("Annotated video output was not found.")

                # Distribution Charts
                ch1, ch2 = st.columns(2)
                with ch1:
                    render_severity_pie_chart(result.severity_breakdown)
                with ch2:
                    render_confidence_histogram(result.detections)

                # Store active video_id in session state for explorer page
                st.session_state["last_video_id"] = video_id

            except Exception as e:
                st.error(f"❌ Processing failed with error: {e}")
                st.exception(e)
