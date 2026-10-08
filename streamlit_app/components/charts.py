"""
Data Visualization Charts for Streamlit Dashboard.

Renders Plotly interactive pie charts, severity breakdowns, and confidence histograms.
"""

from typing import Dict, List
import pandas as pd
import plotly.express as px
import streamlit as st


def render_severity_pie_chart(severity_counts: Dict[str, int]) -> None:
    """Renders an interactive Plotly donut chart showing severity distribution."""
    df = pd.DataFrame(
        [
            {"Severity": "LOW", "Count": severity_counts.get("LOW", 0), "Color": "#28a745"},
            {"Severity": "MEDIUM", "Count": severity_counts.get("MEDIUM", 0), "Color": "#fd7e14"},
            {"Severity": "HIGH", "Count": severity_counts.get("HIGH", 0), "Color": "#dc3545"},
        ]
    )

    total = df["Count"].sum()
    if total == 0:
        st.info("No detections available to generate severity breakdown.")
        return

    fig = px.pie(
        df,
        values="Count",
        names="Severity",
        color="Severity",
        color_discrete_map={"LOW": "#28a745", "MEDIUM": "#fd7e14", "HIGH": "#dc3545"},
        hole=0.45,
        title="Road Surface Distress Severity Breakdown",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=320)
    st.plotly_chart(fig, use_container_width=True)


def render_confidence_histogram(detections: List[dict]) -> None:
    """Renders confidence distribution histogram."""
    if not detections:
        return

    confs = [d.get("confidence", d.get("max_confidence", 0.0)) for d in detections]
    df = pd.DataFrame({"Confidence": confs})

    fig = px.histogram(
        df,
        x="Confidence",
        nbins=15,
        range_x=[0.0, 1.0],
        title="Detection Confidence Distribution",
        color_discrete_sequence=["#1f77b4"],
    )
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=300, yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
