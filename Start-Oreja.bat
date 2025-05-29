@echo off
title Oreja Launcher
echo Starting Oreja Conference Transcription...
echo.

REM Run the PowerShell launcher script
powershell -ExecutionPolicy Bypass -File "%~dp0Start-Oreja.ps1"

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause >nul
) 