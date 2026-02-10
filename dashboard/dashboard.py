"""
Streamlit Dashboard for Voice Emotion Analysis.
Professional, minimalistic interface with light theme and black/white/gray palette.
"""

import os
import sys
import json
import requests
import tempfile
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.emotion_service import analyze_audio
from utils.time_utils import parse_mmss_to_seconds

# ---- Page Configuration ----
st.set_page_config(
    page_title="Voice Emotion Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Custom CSS for Light Theme ----
st.markdown("""
<style>
    /* Force light theme */
    .stApp {
        background-color: #ffffff;
    }

    /* Header styling */
    .main-title {
        font-size: 32px;
        font-weight: 600;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }

    .sub-title {
        font-size: 15px;
        color: #757575;
        text-align: center;
        margin-bottom: 32px;
    }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #1a1a1a;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 8px;
        margin-bottom: 16px;
        margin-top: 24px;
    }

    /* Metric cards */
    .metric-card {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #1a1a1a;
    }

    .metric-label {
        font-size: 13px;
        color: #999999;
        margin-top: 4px;
    }

    /* Table styling */
    .dataframe {
        font-size: 14px !important;
    }

    /* Footer */
    .footer-text {
        font-size: 12px;
        color: #999999;
        text-align: center;
        margin-top: 48px;
        padding-top: 16px;
        border-top: 1px solid #e0e0e0;
    }

    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Divider */
    .divider {
        height: 1px;
        background-color: #e0e0e0;
        margin: 32px 0;
    }

    div[data-testid="stMetricValue"] {
        color: #1a1a1a;
    }

    div[data-testid="stMetricLabel"] {
        color: #757575;
    }
</style>
""", unsafe_allow_html=True)

# Monochrome color palette for charts
CHART_COLORS = ["#1a1a1a", "#555555", "#888888", "#bbbbbb", "#d9d9d9", "#f0f0f0"]
EMOTION_COLORS = {
    "neutral": "#999999",
    "happy": "#555555",
    "sad": "#bbbbbb",
    "angry": "#1a1a1a",
}

FLASK_API_URL = os.environ.get("FLASK_API_URL", "http://localhost:5000")


def main():
    """Main dashboard application."""

    # ---- Header ----
    st.markdown('<div class="main-title">Voice Emotion Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload an audio file to detect emotions with time-based tracking</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- Audio Upload ----
    st.markdown('<div class="section-header">Upload Audio</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Select an audio file",
        type=["wav", "mp3"],
        help="Supported formats: WAV, MP3. Max size: 50 MB.",
        label_visibility="collapsed",
    )

    use_flask_api = st.checkbox("Use Flask API (requires Flask server running)", value=False)

    if uploaded_file is not None:
        st.markdown(f"**Selected:** {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

        if st.button("Analyze", type="primary"):
            with st.spinner("Analyzing audio..."):
                result = _run_analysis(uploaded_file, use_flask_api)

            if result and "error" not in result:
                _display_results(result)
            elif result and "error" in result:
                st.error(result["error"])
            else:
                st.error("Analysis failed. Please try again.")

    # ---- Footer ----
    st.markdown('<div class="footer-text">Voice Emotion Analysis Dashboard</div>',
                unsafe_allow_html=True)


def _run_analysis(uploaded_file, use_flask_api: bool) -> dict:
    """
    Run emotion analysis either directly or via Flask API.

    Args:
        uploaded_file: Streamlit uploaded file object.
        use_flask_api: Whether to use the Flask API.

    Returns:
        Analysis result dictionary.
    """
    if use_flask_api:
        return _analyze_via_flask(uploaded_file)
    else:
        return _analyze_directly(uploaded_file)


def _analyze_directly(uploaded_file) -> dict:
    """Analyze audio directly using the emotion service."""
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        result = analyze_audio(tmp_path, chunk_duration=2.0)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return result

    except Exception as e:
        return {"error": str(e)}


def _analyze_via_flask(uploaded_file) -> dict:
    """Analyze audio by sending to Flask API."""
    try:
        files = {"audio": (uploaded_file.name, uploaded_file.getbuffer(), "audio/wav")}
        response = requests.post(f"{FLASK_API_URL}/upload", files=files, timeout=120)

        if response.status_code == 200:
            return response.json()
        else:
            data = response.json()
            return {"error": data.get("error", "Server error")}

    except requests.ConnectionError:
        return {"error": "Cannot connect to Flask server. Make sure it is running."}
    except Exception as e:
        return {"error": str(e)}


def _display_results(result: dict):
    """Display the complete analysis results on the dashboard."""

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- Metrics Row ----
    segments = result.get("segments", [])
    summary = result.get("summary", {})
    duration = result.get("duration", "00:00")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Duration", duration)
    with col2:
        st.metric("Segments Detected", len(segments))
    with col3:
        # Find dominant emotion
        dominant = max(summary, key=lambda k: summary[k]["percentage"]) if summary else "N/A"
        st.metric("Dominant Emotion", dominant.capitalize())

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- Emotion Timeline Table ----
    st.markdown('<div class="section-header">Emotion Timeline</div>', unsafe_allow_html=True)

    if segments:
        df = pd.DataFrame(segments)
        df.columns = ["Start Time", "End Time", "Emotion"]
        df["Emotion"] = df["Emotion"].str.capitalize()
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No emotion segments detected.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- Two Column Layout for Charts ----
    col_left, col_right = st.columns(2)

    # ---- Emotion Over Time Chart ----
    with col_left:
        st.markdown('<div class="section-header">Emotion Over Time</div>', unsafe_allow_html=True)
        _plot_emotion_timeline(segments)

    # ---- Emotion Distribution Chart ----
    with col_right:
        st.markdown('<div class="section-header">Emotion Distribution</div>', unsafe_allow_html=True)
        _plot_emotion_distribution(summary)

    # ---- Raw Segments Detail ----
    raw_segments = result.get("raw_segments", [])
    if raw_segments:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Detailed Chunk Analysis</div>', unsafe_allow_html=True)

        raw_df = pd.DataFrame(raw_segments)
        raw_df.columns = ["Start", "End", "Emotion", "Confidence"]
        raw_df["Emotion"] = raw_df["Emotion"].str.capitalize()
        raw_df["Confidence"] = raw_df["Confidence"].apply(lambda x: f"{x:.4f}")
        st.dataframe(raw_df, use_container_width=True, hide_index=True)


def _plot_emotion_timeline(segments: list):
    """
    Create an emotion over time chart using Plotly.
    X-axis: Time, Y-axis: Emotion categories.
    """
    if not segments:
        st.info("No data available for timeline chart.")
        return

    emotions = []
    times = []

    for seg in segments:
        start_sec = parse_mmss_to_seconds(seg["start"])
        end_sec = parse_mmss_to_seconds(seg["end"])
        emotion = seg["emotion"].capitalize()

        emotions.append(emotion)
        times.append(start_sec)
        emotions.append(emotion)
        times.append(end_sec)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times,
        y=emotions,
        mode="lines+markers",
        line=dict(color="#1a1a1a", width=2),
        marker=dict(color="#1a1a1a", size=6),
        hovertemplate="Time: %{x:.1f}s<br>Emotion: %{y}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Emotion",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=13, color="#1a1a1a"),
        margin=dict(l=20, r=20, t=20, b=40),
        height=350,
        xaxis=dict(
            showgrid=True,
            gridcolor="#f0f0f0",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f0f0f0",
            zeroline=False,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def _plot_emotion_distribution(summary: dict):
    """
    Create an emotion distribution bar chart using Plotly.
    Shows percentage of time spent in each emotion.
    """
    if not summary:
        st.info("No data available for distribution chart.")
        return

    emotions = [e.capitalize() for e in summary.keys()]
    percentages = [summary[e]["percentage"] for e in summary.keys()]

    # Assign gray shades
    colors = []
    for i, _ in enumerate(emotions):
        if i < len(CHART_COLORS):
            colors.append(CHART_COLORS[i])
        else:
            colors.append("#888888")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=emotions,
        y=percentages,
        marker_color=colors[:len(emotions)],
        text=[f"{p}%" for p in percentages],
        textposition="outside",
        textfont=dict(size=13, color="#1a1a1a"),
        hovertemplate="Emotion: %{x}<br>Percentage: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Emotion",
        yaxis_title="Percentage (%)",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=13, color="#1a1a1a"),
        margin=dict(l=20, r=20, t=20, b=40),
        height=350,
        yaxis=dict(
            showgrid=True,
            gridcolor="#f0f0f0",
            zeroline=False,
            range=[0, max(percentages) * 1.2] if percentages else [0, 100],
        ),
        xaxis=dict(
            showgrid=False,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
