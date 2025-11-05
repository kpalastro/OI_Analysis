@echo off
cd /d "%~dp0"
echo ==========================================
echo     OI Tracker Web Application
echo ==========================================
echo.
echo Working Directory: %CD%
echo.
echo Starting server...
echo.
python oi_tracker_web.py
echo.
echo.
pause

