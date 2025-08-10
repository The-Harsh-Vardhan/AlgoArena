@echo off
echo 🏟️ AlgoArena - Dynamic Tabular ML Platform
echo Starting the application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "streamlit_app\app.py" (
    echo ❌ Please run this script from the AlgoArena root directory.
    echo Make sure you can see the 'streamlit_app' folder.
    pause
    exit /b 1
)

REM Check if Streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Streamlit not found. Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies.
        echo Please run: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo ✅ Dependencies verified!
echo 🚀 Launching AlgoArena...
echo 📱 Your browser should open automatically to http://localhost:8501
echo.
echo 💡 If the browser doesn't open automatically, visit:
echo    http://localhost:8501
echo.
echo 🛑 To stop the application, press Ctrl+C in this window
echo.

REM Launch the Streamlit application
streamlit run streamlit_app\app.py

echo.
echo 👋 Thanks for using AlgoArena!
pause