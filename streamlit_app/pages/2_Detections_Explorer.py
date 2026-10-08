"""
Streamlit Page 2: Detections & Road Conditions Record Explorer.
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure repository root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.database.connection import DatabaseConnectionPool
from app.database.repository import DetectionRepository, RoadConditionRepository, VideoRepository

st.set_page_config(page_title="Detections Explorer", page_icon="🔍", layout="wide")

st.title("🔍 Detections & Road Conditions Explorer")
st.markdown("Inspect granular frame-level detections and aggregated road defect events.")
st.divider()

db_available = DatabaseConnectionPool.check_connection()

if not db_available:
    st.warning("⚠️ MySQL Database is offline. Start MySQL and process videos to populate persistent records.")
else:
    # Fetch list of processed videos
    videos = VideoRepository.list_all_videos(limit=50)

    if not videos:
        st.info("No video records found in the database. Upload and process a video in Page 1.")
    else:
        # Selection dropdown
        video_options = {f"{v['original_filename']} ({v['video_id'][:8]}...) - {v['status']}": v['video_id'] for v in videos}
        selected_label = st.selectbox("Select Video Record:", list(video_options.keys()))
        selected_video_id = video_options[selected_label]

        video_record = VideoRepository.get_video_by_id(selected_video_id)

        # Overview Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Status", video_record.get("status", "N/A"))
        m2.metric("Total Detections", video_record.get("total_detections", 0))
        m3.metric("Unique Potholes", video_record.get("total_unique_potholes", 0))
        m4.metric("Processing Time", f"{video_record.get('processing_time_sec', 0.0)}s")

        tab1, tab2, tab3 = st.tabs(["🛣️ Unique Road Conditions", "🎯 Raw Frame Detections", "📋 Video Metadata"])

        with tab1:
            st.subheader("Aggregated Unique Road Defects")
            conditions = RoadConditionRepository.get_by_video_id(selected_video_id)

            if conditions:
                df_cond = pd.DataFrame(conditions)
                # Severity filter
                sev_filter = st.multiselect(
                    "Filter by Severity:",
                    ["LOW", "MEDIUM", "HIGH"],
                    default=["LOW", "MEDIUM", "HIGH"],
                )
                if "overall_severity" in df_cond.columns:
                    filtered_df = df_cond[df_cond["overall_severity"].isin(sev_filter)]
                else:
                    filtered_df = df_cond

                st.dataframe(filtered_df, use_container_width=True)

                # Export to CSV
                csv_data = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Road Conditions to CSV",
                    data=csv_data,
                    file_name=f"road_conditions_{selected_video_id[:8]}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No unique road conditions aggregated for this video.")

        with tab2:
            st.subheader("Raw Frame-Level YOLO Detections")
            dets = DetectionRepository.get_by_video_id(selected_video_id)
            if dets:
                df_dets = pd.DataFrame(dets)
                st.dataframe(df_dets, use_container_width=True)
            else:
                st.info("No raw frame detections logged for this video.")

        with tab3:
            st.subheader("Video Container & Stream Metadata")
            st.json(video_record)
