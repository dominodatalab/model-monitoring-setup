#!/bin/bash
#
# Launch Model Monitoring Dashboard
#
# Usage:
#   bash /mnt/dev/launch.sh              # Uses port 8888 (Domino production)
#   PORT=8501 bash /mnt/dev/launch.sh    # Uses port 8501 (development)
#

# Default to Domino production port 8888, but allow override
PORT=${PORT:-8888}

# Kill any existing Streamlit processes
echo "Checking for existing Streamlit processes..."
pkill -f "streamlit run" 2>/dev/null || true
sleep 1

# Change to app directory
cd /mnt/dev

# Create Streamlit config directory
mkdir -p ~/.streamlit

# Create Streamlit configuration file
cat > ~/.streamlit/config.toml <<EOF
[browser]
gatherUsageStats = false

[server]
address = "0.0.0.0"
port = $PORT
enableCORS = false
enableXsrfProtection = false
headless = true

[theme]
base = "dark"
primaryColor = "#FF6B35"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#3C3A42"
textColor = "#97A3B7"
EOF

# Display startup banner
echo ""
echo "=========================================="
echo "  Model Monitoring Dashboard"
echo "=========================================="
echo ""

# Generate and display the Streamlit URL for Domino
if [ -n "${DOMINO_RUN_HOST_PATH:-}" ]; then
    CLEAN_PATH=$(echo "$DOMINO_RUN_HOST_PATH" | sed 's|/r||g')
    STREAMLIT_URL="https://ews.domino-eval.com${CLEAN_PATH}proxy/${PORT}/"
    echo "Streamlit URL: $STREAMLIT_URL"
    echo "=========================================="
    echo ""
else
    echo "Running locally on port $PORT"
    echo "=========================================="
    echo ""
fi

# Launch Streamlit app
streamlit run app.py
