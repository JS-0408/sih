#!/usr/bin/env bash
# ======================================================================
# Yaazhi GeoAlign OS -- Linux / macOS 1-Click Launch Script
# ======================================================================

set -e

echo "======================================================================"
echo "   YAAZHI GEOALIGN OS -- GEOSPATIAL IMAGE REGISTRATION PLATFORM"
echo "======================================================================"
echo ""

# 1. Check Python
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "[ERROR] Python 3 is not installed."
    exit 1
fi

echo "[1/3] Python binary: $PYTHON_BIN ($($PYTHON_BIN --version))"

# 2. Dependencies
echo "[2/3] Verifying required Python dependencies..."
$PYTHON_BIN -m pip install -r requirements.txt flask --quiet

# 3. Open Browser & Launch Server
echo "[3/3] Launching Web Server at http://127.0.0.1:5000 ..."
echo ""
echo "======================================================================"
echo "  Dashboard URL: http://127.0.0.1:5000"
echo "======================================================================"
echo ""

# Auto-open browser depending on OS
if command -v xdg-open &>/dev/null; then
    (sleep 2 && xdg-open "http://127.0.0.1:5000") &
elif command -v open &>/dev/null; then
    (sleep 2 && open "http://127.0.0.1:5000") &
fi

$PYTHON_BIN server.py
