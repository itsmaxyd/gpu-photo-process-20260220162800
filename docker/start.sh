#!/bin/bash
# Startup script for GTX 980 Ti OCR System
# Runs both Streamlit dashboard and FastAPI server

set -e

echo "Starting GTX 980 Ti OCR System..."

# Create watch folder if it doesn't exist
mkdir -p /app/watch_folder

# Start FastAPI server in background
echo "Starting REST API server on port 8000..."
uvicorn api_server:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Start Streamlit dashboard in foreground
echo "Starting Streamlit dashboard on port 8501..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

# If Streamlit exits, kill the API server
kill $API_PID 2>/dev/null || true
