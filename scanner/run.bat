@echo off
title HMAxEMA Stock Scanner
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Please install Python from https://python.org
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM Check if dependencies are installed
python -c "import customtkinter" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  First time setup - installing dependencies...
    echo.
    pip install -r requirements.txt
    echo.
    if errorlevel 1 (
        echo  [ERROR] Failed to install dependencies.
        echo  Please run: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo  Dependencies installed successfully!
    echo.
)

REM Launch the GUI
echo Starting HMAxEMA Stock Scanner...
python app.py

if errorlevel 1 (
    echo.
    echo  [ERROR] Scanner exited with an error.
    pause
)
