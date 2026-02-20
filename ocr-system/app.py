"""
GTX 980 Ti Optimized OCR Dashboard

This application is optimized for NVIDIA GTX 980 Ti (Maxwell architecture):
- GPU-accelerated preprocessing using OpenCV CUDA operations
- Hybrid OCR: GPU detection + CPU recognition
- Leverages GTX 980 Ti's strong memory bandwidth (336.5 GB/s)

Key optimizations:
1. CUDA 11.2.2, cuDNN 8.2.1 for Maxwell compatibility
2. OpenCV CUDA for image preprocessing (4-10x speedup)
3. CPU-based text recognition (avoids Maxwell compatibility issues)
"""

import streamlit as st
import pandas as pd
import os
import time
import logging

# Import decoupled OCR system
from ocr_system import OCRSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration (env for Docker; local defaults for venv)
WATCH_FOLDER = os.environ.get("WATCH_FOLDER", "./watch_folder")
RESULTS_FILE = os.environ.get("RESULTS_FILE", "./results.csv")
POLL_INTERVAL = 15  # seconds

# Session state keys
SESSION_KEYS = [
    'processing_stats'
]

# --- UI Functions ---

def init_session_state():
    """Initialize Streamlit session state variables."""
    for key in SESSION_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None
    
    if 'processing_stats' not in st.session_state or st.session_state.processing_stats is None:
        st.session_state.processing_stats = {
            'total_processed': 0,
            'total_time': 0.0
        }

@st.cache_resource
def get_ocr_system() -> OCRSystem:
    """Create and cache the OCR system instance."""
    system = OCRSystem(
        watch_folder=WATCH_FOLDER,
        results_file=RESULTS_FILE,
        poll_interval=POLL_INTERVAL
    )
    system.initialize()
    return system

def render_sidebar(system: OCRSystem):
    """Render the sidebar with system status and controls."""
    st.sidebar.header("System Status")
    
    # GPU Status
    if system.gpu_available:
        st.sidebar.success("🟢 GPU Preprocessing: Enabled")
        # Access safely defaults if method fails or returns None
        try:
             # Preprocessor is typed Optional, but check is verified by gpu_available flag logic
             if system.preprocessor:
                gpu_info = system.preprocessor.get_gpu_info()
                st.sidebar.info(f"CUDA Devices: {gpu_info.get('device_count', 0)}")
        except Exception:
            pass
    else:
        st.sidebar.warning("🟡 GPU Preprocessing: CPU Fallback")
    
    # OCR Status
    st.sidebar.info(f"OCR Engine: Hybrid (GPU Det + CPU Rec)")
    
    # Watch folder info
    st.sidebar.divider()
    st.sidebar.header("Configuration")
    st.sidebar.info(f"📁 Monitoring: `{WATCH_FOLDER}`")
    st.sidebar.info(f"⏱️ Poll Interval: {POLL_INTERVAL}s")
    
    # Controls
    st.sidebar.divider()
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear Results"):
        if os.path.exists(RESULTS_FILE):
            try:
                os.remove(RESULTS_FILE)
                st.success("Results cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear results: {e}")

def render_main_content(data: pd.DataFrame):
    """Render the main content area."""
    if data.empty:
        st.info("📷 No results yet. Drop an image into the watch folder to begin.")
        return
    
    # Get latest entry
    latest = data.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Latest Image")
        try:
            image_path = os.path.join(WATCH_FOLDER, latest['Filename'])
            if os.path.exists(image_path):
                st.image(image_path, caption=latest['Filename'], use_column_width=True)
            else:
                st.warning(f"Image file not found: {latest['Filename']}")
        except Exception as e:
            st.warning(f"Could not display image: {e}")
    
    with col2:
        st.subheader("📝 Extracted Text")
        st.text(latest.get('Extracted Text', '') or '(no text extracted)')
        
        # Show timing info
        if 'Total Time' in latest:
            st.caption(f"⏱️ Processing time: {latest['Total Time']}")
            if 'Preprocess Time' in latest and 'OCR Time' in latest:
                st.caption(
                    f"Preprocessing: {latest['Preprocess Time']} | "
                    f"OCR: {latest['OCR Time']}"
                )
    
    # History section
    st.divider()
    st.subheader("📜 Processing History")
    
    # Show last 10 entries
    history_df = data.tail(10).iloc[::-1]
    
    # Format for display
    display_cols = ['Timestamp', 'Filename', 'Total Time']
    if all(col in data.columns for col in display_cols):
        st.dataframe(
            history_df[display_cols], 
            use_container_width=True
        )
    else:
        st.dataframe(history_df, use_container_width=True)
    
    # Expandable full text view
    with st.expander("View All Extracted Text"):
        for idx, row in history_df.iterrows():
            filename = row.get('Filename', 'Unknown')
            timestamp = row.get('Timestamp', '')
            text = row.get('Extracted Text', '')
            st.markdown(f"**{filename}** ({timestamp})")
            st.text(text)
            st.divider()

def render_performance_metrics(data: pd.DataFrame):
    """Render performance metrics and charts."""
    if data.empty or 'Total Time' not in data.columns:
        return
    
    st.divider()
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    try:
        # Parse timing data
        # Handle cases where column might not be string (if pre-parsed)
        if data['Total Time'].dtype == object:
            total_times = data['Total Time'].str.replace('s', '', regex=False).astype(float)
        else:
            total_times = data['Total Time']
            
        with col1:
            st.metric("Avg Processing Time", f"{total_times.mean():.2f}s")
        
        with col2:
            st.metric("Min Time", f"{total_times.min():.2f}s")
        
        with col3:
            st.metric("Max Time", f"{total_times.max():.2f}s")
            
        # Add a simple chart if enough data
        if len(total_times) > 1:
            st.line_chart(total_times)
            st.caption("Processing time trend")
            
    except Exception as e:
        logger.warning(f"Failed to calculate metrics: {e}")

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="GTX 980 Ti OCR Dashboard",
        page_icon="🖥️",
        layout="wide"
    )
    
    init_session_state()
    
    # Ensure paths exist for venv runs (Docker creates these via Dockerfile/volumes)
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    results_dir = os.path.dirname(RESULTS_FILE)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    st.title("🖥️ GTX 980 Ti Optimized OCR Dashboard")
    st.markdown(
        """
        GPU-accelerated preprocessing with hybrid OCR (GPU detection + CPU recognition).
        Optimized for NVIDIA GTX 980 Ti (Maxwell architecture).
        """
    )
    
    # Initialize Core System
    system = get_ocr_system()
    
    # Start background monitoring
    system.start_monitoring()
    
    # Render UI
    render_sidebar(system)
    
    # Auto-refresh logic
    auto_refresh = st.checkbox("Auto-refresh UI", value=True)
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 30, 10)
        time.sleep(0.5)
        st.empty() # Placeholder
        time.sleep(refresh_interval)
        st.rerun()
    
    # Load Data
    data = pd.DataFrame()
    if os.path.exists(RESULTS_FILE):
        try:
            # Handle empty file check
            if os.path.getsize(RESULTS_FILE) > 0:
                data = pd.read_csv(RESULTS_FILE)
            else:
                data = pd.DataFrame()
        except Exception as e:
            st.error(f"Error reading results file: {e}")
            
    # Render Content
    render_main_content(data)
    render_performance_metrics(data)

if __name__ == "__main__":
    main()