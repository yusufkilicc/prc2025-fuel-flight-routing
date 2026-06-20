@echo off
REM ============================================================
REM  PRC Fuel Routing - Run (Windows)
REM  Launches the web app and opens the interface in the browser
REM  (once the server is ready). Runs setup automatically first time.
REM ============================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run - setting up...
    call setup.bat
    if errorlevel 1 exit /b 1
)

echo Starting the app at http://localhost:8600  (press Ctrl+C to stop) ...
".venv\Scripts\python.exe" start_app.py
