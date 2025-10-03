@echo off
echo FrameSift Lite GUI Launcher
echo ========================

REM Change to the script directory
cd /d "%~dp0"

REM Try to launch with the configured Python
if exist "C:\Python\python.exe" (
    echo Using Python from C:\Python\python.exe
    "C:\Python\python.exe" launch_gui.py
) else (
    echo Using system Python
    python launch_gui.py
)

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Press any key to exit...
    pause >nul
)