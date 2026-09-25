import os
import queue
import time
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from audio_listener import AudioListener
from claim_extractor import extract_claim
from fact_verifier import verify_claim

load_dotenv()

# Page Setup
st.set_page_config(page_title="Fact Checker Dashboard", layout="wide")

# Neumorphic Purple Modern UI CSS
st.markdown("""
<style>
    /* Global Background Override */
    .stApp {
        background-color: #F4F5FA !important;
        color: #1E1E1E !important;
        font-family: 'Inter', sans-serif;
    }

    /* Force text visibility */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #1E1E1E !important;
    }
    
    /* Header Title */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7F00FF 0%, #E100FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    /* Top Metric Card Boxes */
    .custom-card {
        background-color: #FFFFFF !important;
        border-radius: 16px;
        padding: 12px 18px;
        box-shadow: 0px 8px 20px rgba(127, 0, 255, 0.06);
        border: 1px solid #E2E4F0;
        text-align: center;
    }

    /* Streamlit Buttons Style Fix */
    .stButton > button {
        background: linear-gradient(135deg, #7F00FF 0%, #E100FF 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 10px 20px !important;
        box-shadow: 0px 4px 12px rgba(127, 0, 255, 0.2) !important;
    }
    .stButton > button p {
        color: #FFFFFF !important;
    }

    /* Badges Style */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .badge-active { background-color: #E8F5E9 !important; color: #2E7D32 !important; }
    .badge-idle { background-color: #EDE7F6 !important; color: #512DA8 !important; }

    /* Verdict Cards */
    .verdict-card {
        border-radius: 16px;
        padding: 16px;
        margin-top: 12px;
        font-size: 0.95rem;
        font-weight: 500;
    }
    .verdict-correct {
        background-color: #E8F5E9 !important;
        border-left: 6px solid #4CAF50;
        color: #1B5E20 !important;
    }
    .verdict-incorrect {
        background-color: #FFEBEE !important;
        border-left: 6px solid #EF5350;
        color: #C62828 !important;
    }
    .verdict-unverified {
        background-color: #FFFDE7 !important;
        border-left: 6px solid #FBC02D;
        color: #F57F17 !important;
    }

    /* Input Field Styling */
    .stTextInput input {
        background-color: #FFFFFF !important;
        color: #1E1E1E !important;
        border-radius: 12px !important;
        border: 1px solid #D0D0F0 !important;
    }

    /* Speech Bubble Style */
    .chat-bubble {
        background-color: #EDE7F6 !important;
        color: #311B92 !important;
        padding: 12px 18px;
        border-radius: 16px;
        margin-bottom: 8px;
        font-size: 0.92rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Session State Setup
if "listener" not in st.session_state:
    st.session_state.listener = None
if "results" not in st.session_state:
    st.session_state.results = []
if "transcripts" not in st.session_state:
    st.session_state.transcripts = []

# Title Section
st.markdown('<div class="main-title">🛡️ Real-Time Audio Fact-Checker</div>', unsafe_allow_html=True)
st.write("Live audio fact verification running locally with high-precision reasoning.")

# Top Control Bar
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("▶️ Start Listening", use_container_width=True):
            if st.session_state.listener is None:
                st.session_state.listener = AudioListener(model_size="small")
                st.session_state.listener.start()
                st.rerun()
    with btn_col2:
        if st.button("⏹️ Stop Listening", use_container_width=True):
            if st.session_state.listener is not None:
                st.session_state.listener.stop()
                st.session_state.listener = None
                st.rerun()

is_listening = st.session_state.listener is not None and st.session_state.listener.is_running

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("**Microphone Status**")
    if is_listening:
        st.markdown('<span class="status-badge badge-active">🟢 LIVE LISTENING</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-idle">⚪ IDLE</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("**Local LLM Model**")
    st.markdown('<span class="status-badge badge-idle">⚡ qwen2.5:7b</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Manual Text Input Option for Quick Testing
manual_claim = st.text_input("💬 Manual Text Input (Type a claim to test directly without mic):", placeholder="e.g. capital of pakistan is lahore")
if st.button("Check Manual Claim"):
    if manual_claim.strip():
        with st.spinner("Verifying claim..."):
            extracted = extract_claim(manual_claim)
            
            # Safe NoneType handling
            if isinstance(extracted, str) and "NO_CLAIM" not in extracted and len(extracted) > 3:
                claim_to_verify = extracted
            else:
                claim_to_verify = manual_claim.strip()
            
            verdict = verify_claim(claim_to_verify)
            if verdict:
                st.session_state.results.insert(0, verdict)
                st.rerun()

st.divider()

# Main Dashboard Panels
left_panel, right_panel = st.columns([1.6, 1])

with left_panel:
    st.markdown("### 📊 Verified Claims")
    if not st.session_state.results:
        st.info("No claims processed yet. Click 'Start Listening' or type a claim above.")
    for res in st.session_state.results:
        v_class = "verdict-unverified"
        verdict_str = str(res).upper()
        if "CORRECT" in verdict_str and "INCORRECT" not in verdict_str:
            v_class = "verdict-correct"
        elif "INCORRECT" in verdict_str:
            v_class = "verdict-incorrect"
            
        st.markdown(f'''
        <div class="verdict-card {v_class}">
            {res}
        </div>
        ''', unsafe_allow_html=True)

with right_panel:
    st.markdown("### 🎙️ Live Transcript")
    
    if is_listening and st.session_state.listener:
        try:
            while not st.session_state.listener.transcripts.empty():
                text = st.session_state.listener.transcripts.get_nowait()
                if text:
                    st.session_state.transcripts.append(text)
                    extracted = extract_claim(text)
                    
                    if isinstance(extracted, str) and "NO_CLAIM" not in extracted and len(extracted) > 3:
                        claim_to_verify = extracted
                    else:
                        claim_to_verify = text.strip()

                    res = verify_claim(claim_to_verify)
                    if res:
                        st.session_state.results.insert(0, res)
        except queue.Empty:
            pass

    if st.session_state.transcripts:
        for t in reversed(st.session_state.transcripts[-6:]):
            st.markdown(f'<div class="chat-bubble">🗣️ {t}</div>', unsafe_allow_html=True)
    else:
        st.caption("Listening... speak clearly into your microphone.")

# Auto-rerun for real-time streaming updates
if is_listening:
    time.sleep(1)
    st.rerun()
    