"""
Streamlit Page 3: Geospatial Map View for Flagged Road Conditions.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure repository root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.database.connection import DatabaseConnectionPool
from app.database.repository import RoadConditionRepository
from streamlit_app.components.map_view import render_folium_map

st.set_page_config(page_title="Road Condition Map", page_icon="🗺️", layout="wide")

st.title("🗺️ Geospatial Road Condition & Pothole Map")
st.markdown("Interactive GIS map displaying all GPS-geocoded road defect locations and severity classifications.")
st.divider()

db_available = DatabaseConnectionPool.check_connection()

if not db_available:
    st.warning("⚠️ MySQL Database is offline. Map points will be fetched once the database is active.")
else:
    points = RoadConditionRepository.get_geolocated_points()

    # Filter Controls
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Map Controls")
        sev_filter = st.multiselect(
            "Filter by Severity:",
            ["LOW", "MEDIUM", "HIGH"],
            default=["LOW", "MEDIUM", "HIGH"],
        )
        st.info(
            "💡 **Marker Legend:**\n\n"
            "🔴 **Red:** High Severity (Large area disruption)\n\n"
            "🟠 **Orange:** Medium Severity\n\n"
            "🟢 **Green:** Low Severity (Minor anomaly)"
        )

    with col2:
        if points:
            filtered_points = [p for p in points if p.get("overall_severity", "LOW").upper() in sev_filter]
            st.metric("Visible Pothole Markers", len(filtered_points))
            render_folium_map(filtered_points, height=560)
        else:
            st.info(
                "📍 **No GPS Points in Database:** Detections are currently recorded without embedded GPS tags. "
                "Videos captured with GPS-enabled dashcams (or containing QuickTime ISO-6709 location tags) will appear automatically on this map."
            )
