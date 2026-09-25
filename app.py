"""Streamlit dashboard for the multi-modal deepfake detector."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from modules.audio_engine import analyze_audio
from modules.image_engine import analyze_image
from modules.live_audio_engine import process_audio_chunk
from modules.video_engine import analyze_video


st.set_page_config(
    page_title="Sentinel Media Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #091016; }
    [data-testid="stSidebar"] { background: #0d171d; border-right: 1px solid #20333a; }
    .block-container { max-width: 1320px; padding-top: 2rem; }
    h1, h2, h3 { letter-spacing: 0.02em; }
    .hero { padding: 1.4rem 1.6rem; border: 1px solid #24434a; background: linear-gradient(135deg, #10252b, #0b151b); border-radius: 8px; margin-bottom: 1.4rem; }
    .hero p { color: #9cb2b7; margin-bottom: 0; }
    .status { color: #62e6b1; font-size: 0.78rem; letter-spacing: 0.12em; font-weight: 700; }
    .result { padding: 1rem; border-left: 3px solid #62e6b1; background: #101f24; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def score_color(score: float) -> str:
    return "#ff6b6b" if score >= 50 else "#62e6b1"


def show_score(score: float, label: str) -> None:
    st.metric(label, f"{score:.1f}%")
    st.progress(min(max(score / 100.0, 0.0), 1.0), text=f"{label}: {score:.1f}%")
    st.markdown(
        f'<div class="result" style="border-left-color:{score_color(score)}">Signal assessment: <strong>{label}</strong></div>',
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="hero"><div class="status">SENTINEL // MEDIA FORENSICS CONSOLE</div><h1>Multi-Modal Deepfake Detector</h1><p>Inspect audio, imagery, video, and live voice streams with explainable signal indicators.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("System status")
    st.success("Analysis engines online")
    st.caption("Heuristic scores are triage indicators, not forensic proof.")
    st.divider()
    st.caption("Accepted inputs")
    st.caption("Audio: WAV, MP3 | Image: JPG, PNG | Video: MP4, MOV, AVI")


audio_tab, image_tab, video_tab, live_tab = st.tabs(
    ["🎙️ Audio Deepfake", "🖼️ Image Deepfake", "🎥 Video Deepfake", "📞 Live Call Interceptor"]
)

with audio_tab:
    st.subheader("Audio authenticity scan")
    audio_file = st.file_uploader("Upload a voice recording", type=["wav", "mp3"], key="audio")
    if audio_file is not None:
        try:
            suffix = Path(audio_file.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                temporary_file.write(audio_file.getvalue())
                audio_path = temporary_file.name
            result = analyze_audio(audio_path)
            left, right = st.columns([1, 2])
            with left:
                show_score(result["risk_score"], result["classification"])
                st.caption(f"Duration: {result['duration_seconds']} s | Sample rate: {result['sample_rate']} Hz")
            with right:
                figure, axis = plt.subplots(figsize=(10, 4))
                axis.imshow(result["mel_spectrogram"], origin="lower", aspect="auto", cmap="magma")
                axis.set_title("Mel-spectrogram (dB)")
                axis.set_xlabel("Time frames")
                axis.set_ylabel("Mel bands")
                figure.tight_layout()
                st.pyplot(figure, clear_figure=True)
                plt.close(figure)
        except Exception as error:
            st.error(f"Audio analysis failed: {error}")

with image_tab:
    st.subheader("Image frequency artifact scan")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image")
    if image_file is not None:
        try:
            image_bytes = image_file.getvalue()
            result = analyze_image(image_bytes)
            left, right = st.columns([1, 2])
            with left:
                show_score(result["risk_score"], result["classification"])
                st.caption(f"Dimensions: {result['width']} x {result['height']} px")
                st.caption(f"Artifact severity: {result['artifact_severity']}")
                st.caption(f"Spatial variance score: {result['spatial_variance_score']:.1f}%")
                st.caption(f"High-frequency noise score: {result['high_frequency_noise_score']:.1f}%")
                st.caption(f"Signal agreement: {result['artifact_agreement']:.1f}%")
            with right:
                st.image(image_bytes, caption="Uploaded media", use_container_width=True)
                st.caption(
                    f"FFT ratio: {result['frequency_ratio']} | "
                    f"Noise ratio: {result['high_frequency_noise_ratio']}"
                )
        except Exception as error:
            st.error(f"Image analysis failed: {error}")

with video_tab:
    st.subheader("Video frame consistency scan")
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="video")
    if video_file is not None:
        try:
            suffix = Path(video_file.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                temporary_file.write(video_file.getvalue())
                video_path = temporary_file.name
            result = analyze_video(video_path)
            left, right = st.columns([1, 2])
            with left:
                show_score(result["risk_score"], result["classification"])
            with right:
                st.metric("Frames analyzed", result["frames_analyzed"])
                st.metric("Total frames", result["total_frames"])
                st.caption(f"Average Laplacian variance: {result['average_laplacian_variance']}")
                st.caption(f"Edge variation: {result['edge_variation']}")
        except Exception as error:
            st.error(f"Video analysis failed: {error}")

with live_tab:
    st.subheader("Virtual audio stream monitor")
    st.caption("Simulation mode generates PCM chunks. A PyAudio or virtual-cable callback can pass its input directly to process_audio_chunk().")
    if "live_result" not in st.session_state:
        st.session_state.live_result = {"risk_score": 0.0, "classification": "WAITING FOR SIGNAL", "rms": 0.0, "peak": 0.0}
    if st.button("Analyze incoming chunk", type="primary"):
        time_axis = np.linspace(0, 0.25, 4_000, endpoint=False)
        chunk = (0.12 * np.sin(2 * np.pi * 220 * time_axis) + np.random.normal(0, 0.01, time_axis.size)).astype(np.float32)
        st.session_state.live_result = process_audio_chunk(chunk)
    result = st.session_state.live_result
    show_score(float(result["risk_score"]), str(result["classification"]))
    st.caption(f"RMS: {result['rms']} | Peak: {result['peak']}")
