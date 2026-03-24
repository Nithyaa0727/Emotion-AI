#!/bin/bash
echo "================================================"
echo "  Emotion AI — Setup Script (Mac/Linux)"
echo "================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found."
    echo "Install from: https://python.org"
    exit 1
fi
echo "[OK] Python found: $(python3 --version)"

# Check model file
if [ ! -f "model/efficientnet_v2_s_finetuned.pth" ]; then
    echo ""
    echo "[WARNING] Model file not found!"
    echo "Place efficientnet_v2_s_finetuned.pth inside the 'model' folder."
    echo "Download: kaggle.com/code/mohammedaazam7/face-recg > Output"
    echo ""
fi

# Install dependencies
echo ""
echo "[STEP 1] Installing dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies."
    exit 1
fi
echo "[OK] Dependencies installed"

# Run app
echo ""
echo "[STEP 2] Starting Emotion AI..."
echo ""
echo "================================================"
echo "  Open your browser: http://localhost:5000"
echo "  Press Ctrl+C to stop"
echo "================================================"
echo ""
cd app && python3 app.py
