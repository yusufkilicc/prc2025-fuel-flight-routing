#!/usr/bin/env bash
# ============================================================
#  PRC Fuel Routing - Setup (macOS / Linux)
#  Creates a local virtual environment and installs the web
#  app's runtime dependencies. Run once; then use ./run.sh.
# ============================================================
set -e
cd "$(dirname "$0")"

PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
    echo "[ERROR] Python not found. Install Python 3.11+ and retry."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating virtual environment (.venv) ..."
    "$PY" -m venv .venv
fi

echo "Installing dependencies (this may take a couple of minutes) ..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-web.txt

echo
echo "Setup complete. Start the app with:  ./run.sh"
