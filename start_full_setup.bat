@echo off
title FORGE-Lite Deepfake Detection Platform
echo ======================================================================
echo           FORGE-Lite Deepfake Detection Platform Setup
echo ======================================================================
echo.
echo [*] Checking Python and Node environments...
python --version
node --version
echo.
echo [*] Launching FORGE-Lite full system (FastAPI + Unified Frontend)...
python run_app.py
pause
