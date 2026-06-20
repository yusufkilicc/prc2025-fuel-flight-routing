"""
One-command launcher: starts the web app and opens the browser **once the
server is actually ready** (avoids the "connection refused" flash on first run).

    python start_app.py
"""
import os
import sys
import time
import threading
import webbrowser
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "web"))

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8600"))
URL = f"http://localhost:{PORT}"


def _open_when_ready():
    for _ in range(120):  # up to ~60 s for a cold venv start
        try:
            urllib.request.urlopen(URL, timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    webbrowser.open(URL)


if __name__ == "__main__":
    import uvicorn
    from web_app import app  # importing does not start a second server (guarded)

    threading.Thread(target=_open_when_ready, daemon=True).start()
    print(f"Opening {URL} in your browser once ready ... (Ctrl+C to stop)")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
