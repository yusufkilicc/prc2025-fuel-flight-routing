#!/usr/bin/env bash
# ============================================================
#  PRC Fuel Routing - Run (macOS / Linux)
#  Launches the web app and opens the interface in the browser
#  (once the server is ready). Runs setup automatically first time.
# ============================================================
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "First run - setting up..."
    bash ./setup.sh
fi

echo "Starting the app at http://localhost:8600  (press Ctrl+C to stop) ..."
exec .venv/bin/python start_app.py
