@echo off
title Oreja Complete Launcher
echo 🎙️ Starting Oreja Complete Launcher...
echo.

REM Check if virtual environment exists and use it
if exist "venv\Scripts\python.exe" (
    echo ✅ Using virtual environment...
    "venv\Scripts\python.exe" launch_oreja_analytics.py
) else (
    echo ⚠️ Virtual environment not found, using system Python...
    python launch_oreja_analytics.py
)

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ❌ An error occurred. Press any key to exit...
    pause >nul
) 