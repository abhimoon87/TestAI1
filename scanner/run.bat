@echo off
title HMAxEMA Stock Scanner — Web
cd /d "%~dp0.."

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
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  First time setup - installing dependencies...
    echo.
    pip install -r scanner/requirements.txt
    echo.
    if errorlevel 1 (
        echo  [ERROR] Failed to install dependencies.
        echo  Please run: pip install -r scanner/requirements.txt
        pause
        exit /b 1
    )
    echo  Dependencies installed successfully!
    echo.
)

REM Launch the Web App
echo Starting HMAxEMA Stock Scanner (Web)...
echo Browser will open automatically at http://localhost:5000
python -m scanner.web_app

if errorlevel 1 (
    echo.
    echo  [ERROR] Scanner exited with an error.
    pause
)
