"""
Folium Geospatial Map Component for Streamlit.

Renders interactive OpenStreetMap views with color-coded severity markers,
HTML telemetry popups, and cluster support.
"""

from typing import Any, Dict, List
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import st_folium

# Color mapping for Folium map markers
FOLIUM_SEVERITY_COLORS = {
    "LOW": "green",
    "MEDIUM": "orange",
    "HIGH": "red",
    "UNKNOWN": "gray",
}


def render_folium_map(points: List[Dict[str, Any]], height: int = 500) -> None:
    """
    Renders an interactive Folium map for GPS-tagged road condition events.

    Args:
        points: List of dictionaries with latitude, longitude, overall_severity, max_confidence, etc.
        height: Pixel height of map widget.
    """
    valid_points = [p for p in points if p.get("latitude") is not None and p.get("longitude") is not None]

    if not valid_points:
        st.warning(
            "📍 **No GPS Coordinates Available:** The processed videos did not contain embedded GPS telemetry tags. "
            "Visual detections and severity statistics remain fully available in the explorer."
        )
        return

    # Compute center of map as mean of valid coordinates
    avg_lat = sum(float(p["latitude"]) for p in valid_points) / len(valid_points)
    avg_lon = sum(float(p["longitude"]) for p in valid_points) / len(valid_points)

    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    marker_cluster = MarkerCluster().add_to(m)

    for p in valid_points:
        lat = float(p["latitude"])
        lon = float(p["longitude"])
        sev = p.get("overall_severity", "LOW").upper()
        conf = float(p.get("max_confidence", 0.0))
        obs = p.get("observation_count", 1)
        event_id = p.get("event_id", "N/A")[:8]

        marker_color = FOLIUM_SEVERITY_COLORS.get(sev, "blue")

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 180px;">
            <h4 style="margin: 0 0 5px 0; color: #333;">Pothole Flag #{event_id}</h4>
            <hr style="margin: 4px 0; border: 0; border-top: 1px solid #ccc;">
            <b>Severity:</b> <span style="color: {'#d9534f' if sev=='HIGH' else '#f0ad4e' if sev=='MEDIUM' else '#5cb85c'}; font-weight: bold;">{sev}</span><br>
            <b>Confidence:</b> {conf * 100:.1f}%<br>
            <b>Frames Observed:</b> {obs}<br>
            <b>Coordinates:</b> {lat:.5f}, {lon:.5f}
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Pothole [{sev}] - Conf: {conf*100:.0f}%",
            icon=folium.Icon(color=marker_color, icon="exclamation-triangle", prefix="fa"),
        ).add_to(marker_cluster)

    st_folium(m, width=None, height=height, use_container_width=True)
