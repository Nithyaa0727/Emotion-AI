@echo off
echo ================================================
echo   Emotion AI v2 — Setup Script (Windows)
echo ================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Please install Python 3.10+ from python.org
    pause
    exit /b
)
echo [OK] Python found
python --version

REM Check pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip not found.
    pause
    exit /b
)
echo [OK] pip found

REM Check model file
echo.
if not exist "model\efficientnet_v2_s_finetuned.pth" (
    echo [WARNING] Model file not found!
    echo Please place efficientnet_v2_s_finetuned.pth inside the "model" folder.
    echo Download model^> Output
    echo.
    pause
) else (
    echo [OK] Model file found
)

REM Install dependencies
echo.
echo [STEP 1] Installing dependencies...
echo This may take a few minutes on first run...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies.
    echo Try running as Administrator or check your internet connection.
    pause
    exit /b
)
echo.
echo [OK] All dependencies installed

REM Check if app folder exists
if not exist "app\app.py" (
    echo [ERROR] app\app.py not found.
    echo Make sure you are running this from the project root folder.
    pause
    exit /b
)

REM Run app
echo.
echo [STEP 2] Starting Emotion AI...
echo.
echo ================================================
echo   Open your browser: http://localhost:5000
echo   Press Ctrl+C to stop the server
echo ================================================
echo.
cd app
python app.py

echo.
echo [INFO] Server stopped.
pause