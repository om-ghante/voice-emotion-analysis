#!/bin/bash
# Start script for Render deployment.
# Launches Flask API in the background and Streamlit dashboard as the main process.

set -e

# Use PORT from Render, default to 10000
PORT="${PORT:-10000}"

# Start Flask API server in background on internal port 5000
echo "Starting Flask API on port 5000..."
gunicorn app:app --bind 0.0.0.0:5000 --workers 1 --timeout 120 &
FLASK_PID=$!

# Wait briefly for Flask to start
sleep 2

# Export Flask URL for Streamlit
export FLASK_API_URL="http://localhost:5000"

# Start Streamlit dashboard on the Render-assigned port
echo "Starting Streamlit dashboard on port $PORT..."
streamlit run dashboard/dashboard.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS true \
    --browser.gatherUsageStats false \
    --theme.base light \
    --theme.primaryColor "#1a1a1a" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#fafafa" \
    --theme.textColor "#1a1a1a"
