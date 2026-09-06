import cv2
import io
import time
import sqlite3
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. Database Setup & Management Functions
# ---------------------------------------------------------
DB_FILE = "telemetry_logs.db"

def init_db():
    """Database and table initialization."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source_type TEXT NOT NULL,
            object_class TEXT NOT NULL,
            confidence_score REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_detections_to_db(source_type, detected_items_with_conf):
    """Batch insert detected objects into SQLite database."""
    if not detected_items_with_conf:
        return
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    records = [
        (current_time, source_type, obj_name, round(conf, 2))
        for obj_name, conf in detected_items_with_conf
    ]
    
    cursor.executemany("""
        INSERT INTO detection_logs (timestamp, source_type, object_class, confidence_score)
        VALUES (?, ?, ?, ?)
    """, records)
    
    conn.commit()
    conn.close()

def fetch_logs(limit=500):
    """Fetch recent telemetry records from database."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        f"SELECT id AS 'Log ID', timestamp AS 'Timestamp', source_type AS 'Source', object_class AS 'Detected Class', confidence_score AS 'Confidence' FROM detection_logs ORDER BY id DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df

def clear_database():
    """Clear all records from database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detection_logs")
    conn.commit()
    conn.close()

# Initialize Database on launch
init_db()

# ---------------------------------------------------------
# 2. Page Setup & Professional Design Architecture
# ---------------------------------------------------------
st.set_page_config(
    page_title="VisionMetrics Pro | Enterprise Analytics & DB",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ENTERPRISE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp { 
        background-color: #F8FAFC; 
        color: #0F172A; 
    }
    
    section[data-testid="stSidebar"] { 
        background-color: #F1F5F9 !important; 
        border-right: 1px solid #E2E8F0; 
    }
    
    section[data-testid="stSidebar"] *, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span {
        color: #334155 !important;
        font-weight: 500 !important;
    }

    .brand-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .brand-header h1 { 
        color: #0F172A !important; 
        font-size: 22px !important; 
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        margin: 0 0 4px 0 !important; 
    }
    .brand-header p { 
        color: #64748B !important; 
        font-size: 13px !important; 
        margin: 0 !important; 
    }

    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: left;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .metric-value { 
        font-size: 28px; 
        font-weight: 700; 
        color: #0F172A; 
        line-height: 1.2;
    }
    .metric-label { 
        font-size: 11px; 
        font-weight: 600; 
        color: #64748B; 
        text-transform: uppercase; 
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .inventory-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .inventory-pill {
        background-color: #F8FAFC; 
        color: #334155; 
        font-weight: 600;
        padding: 6px 12px; 
        border-radius: 6px; 
        font-size: 12px;
        border: 1px solid #CBD5E1;
        display: inline-flex;
        align-items: center;
    }
    .inventory-count {
        background-color: #2563EB; 
        color: #FFFFFF;
        border-radius: 4px; 
        padding: 2px 6px; 
        margin-left: 8px; 
        font-size: 11px;
        font-weight: 700;
    }
    
    .panel-card { 
        background-color: #FFFFFF; 
        border: 1px solid #E2E8F0; 
        border-radius: 10px; 
        padding: 20px; 
        margin-bottom: 20px; 
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.02);
    }
    .panel-title {
        font-size: 15px;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 16px;
    }
</style>
"""
st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Top Header Navigation
# ---------------------------------------------------------
st.markdown("""
<div class="brand-header">
    <h1>VisionMetrics Pro | Edge Detection Console</h1>
    <p>Real-Time Computer Vision Pipeline • Persistent SQLite Database Telemetry</p>
</div>
""", unsafe_allow_html=True)

if 'last_sound_time' not in st.session_state:
    st.session_state.last_sound_time = 0

# ---------------------------------------------------------
# 4. Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Engine Parameters")

model_type = st.sidebar.radio(
    "Inference Engine Architecture:",
    ["YOLOv8 Nano Engine (Standard COCO)", "Custom Edge Weights (.pt)"]
)

@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)

try:
    if model_type == "YOLOv8 Nano Engine (Standard COCO)":
        model = load_model("yolov8n.pt")
        st.sidebar.caption("Status: Standard Weights (COCO-80) Loaded")
    else:
        custom_weights = st.sidebar.text_input("Local File Path for Custom (.pt) Weights:", "best.pt")
        model = load_model(custom_weights)
        st.sidebar.caption(f"Status: Custom Weights ({custom_weights}) Loaded")
except Exception as e:
    st.sidebar.error(f"Engine Load Failed: {e}")
    st.stop()

confidence = st.sidebar.slider(
    "Confidence Threshold (IoU):",
    min_value=0.1, max_value=1.0, value=0.5, step=0.05
)

resolution_option = st.sidebar.selectbox(
    "Ingestion Resolution Profile:",
    ["480p Balance (Optimized FPS)", "320p High Speed (Low Latency)", "720p HD (High Precision)"]
)

enable_db_logging = st.sidebar.checkbox("💾 Enable Live Database Logging", value=True)

st.sidebar.markdown("---")
st.sidebar.success("Telemetry Status: Engine & DB Operational")

# ---------------------------------------------------------
# 5. Core Detection Function
# ---------------------------------------------------------
def run_detection(image_array, conf):
    results = model(image_array, conf=conf, imgsz=320, verbose=False)[0]
    detected_items_with_conf = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        name = model.names[cls_id]
        detected_items_with_conf.append((name, score))
        
    annotated_img = results.plot()
    return annotated_img, detected_items_with_conf

# ---------------------------------------------------------
# 6. Main Navigation Tabs
# ---------------------------------------------------------
tab_live, tab_db = st.tabs(["📡 Live Video Telemetry", "🗄️ Database Telemetry Logs"])

# ---------------------------------------------------------
# TAB 1: Live Ingestion Stream & Image File Analysis
# ---------------------------------------------------------
with tab_live:
    source_type = st.radio(
        "Select Active Input Stream Source:",
        ["Webcam Stream (Real-Time Feed)", "Static Asset Upload (JPEG/PNG)"],
        horizontal=True
    )

    if source_type == "Webcam Stream (Real-Time Feed)":
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>📹 Camera Feed Stream</div>", unsafe_allow_html=True)
            run_cam = st.checkbox("Initialize Live Feed Stream", value=False)
            frame_placeholder = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_right:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>📊 Live Metrics & Target Counts</div>", unsafe_allow_html=True)
            
            m_col1, m_col2 = st.columns(2)
            total_metric = m_col1.empty()
            fps_metric = m_col2.empty()
            
            badges_placeholder = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

        if run_cam:
            cap = cv2.VideoCapture(0)
            
            if "High Speed" in resolution_option:
                target_width, target_height = 426, 240
            elif "Balance" in resolution_option:
                target_width, target_height = 640, 360
            else:
                target_width, target_height = 1280, 720
                
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
            
            prev_time = time.time()
            
            while run_cam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Hardware Execution Error: Unable to access video interface.")
                    break
                    
                frame_resized = cv2.resize(frame, (target_width, target_height))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                curr_time = time.time()
                fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                processed_frame, detected_items = run_detection(frame_rgb, confidence)
                
                # Database Logging
                if enable_db_logging and detected_items:
                    log_detections_to_db("Webcam", detected_items)
                
                frame_placeholder.image(processed_frame, use_container_width=True)
                
                classes_only = [item[0] for item in detected_items]
                counts = {item: classes_only.count(item) for item in set(classes_only)}
                
                total_metric.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Detected Objects</div>
                    <div class="metric-value">{len(classes_only)}</div>
                </div>
                """, unsafe_allow_html=True)
                
                fps_metric.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Stream Speed</div>
                    <div class="metric-value">{fps} <span style="font-size:14px; color:#64748B;">FPS</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                badges_html = "<div class='inventory-container'>"
                for name, cnt in counts.items():
                    badges_html += f"<div class='inventory-pill'>{name.upper()} <span class='inventory-count'>{cnt}</span></div>"
                badges_html += "</div>"
                
                badges_placeholder.markdown(badges_html if counts else "<p style='color:#94A3B8; font-size:13px;'>No target objects detected in stream viewport.</p>", unsafe_allow_html=True)

            cap.release()

    else:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>📁 Image File Ingestion Interface</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select Target Asset File", type=["jpg", "png", "jpeg"])
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_np = np.array(image)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
                st.markdown("<div class='panel-title'>🖼️ Raw Input Source</div>", unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
                st.markdown("<div class='panel-title'>🎯 Inference Annotation View</div>", unsafe_allow_html=True)
                processed_frame, detected_items = run_detection(img_np, confidence)
                st.image(processed_frame, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            if enable_db_logging and detected_items:
                log_detections_to_db("Static Image", detected_items)
                st.success("✅ Detections logged to SQLite database successfully!")

# ---------------------------------------------------------
# TAB 2: Historical Database Logs & Management
# ---------------------------------------------------------
with tab_db:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>🗄️ SQLite Telemetry Logs Inspector</div>", unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns([1, 4])
    
    if col_btn1.button("🔄 Refresh Data Table"):
        st.rerun()
        
    df_logs = fetch_logs()
    
    if not df_logs.empty:
        st.dataframe(df_logs, use_container_width=True, height=400)
        
        c_exp, c_del = st.columns(2)
        
        csv_data = df_logs.to_csv(index=False).encode('utf-8')
        c_exp.download_button(
            label="📥 Export Full Database Log (.CSV)",
            data=csv_data,
            file_name=f"db_telemetry_export_{int(time.time())}.csv",
            mime="text/csv"
        )
        
        if c_del.button("⚠️ Clear All Database Records"):
            clear_database()
            st.success("Database records cleared successfully.")
            st.rerun()
    else:
        st.info("Database is currently empty. Run the live stream or process images to populate telemetry logs.")
        
    st.markdown("</div>", unsafe_allow_html=True)