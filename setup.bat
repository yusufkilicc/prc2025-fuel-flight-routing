@echo off
REM ============================================================
REM  PRC Fuel Routing - Setup (Windows)
REM  Creates a local virtual environment and installs the web
REM  app's runtime dependencies. Run once; then use run.bat.
REM ============================================================
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH. Install Python 3.11+ from python.org and retry.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment ^(.venv^) ...
    python -m venv .venv
)

echo Installing dependencies ^(this may take a couple of minutes^) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements-web.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo Setup complete. Start the app with:  run.bat
echo.
