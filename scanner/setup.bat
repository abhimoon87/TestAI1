@echo off
title HMAxEMA Stock Scanner — Setup
echo.
echo ========================================
echo   HMAxEMA Stock Scanner — First Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [1/4] Checking Python... NOT FOUND
    echo.
    echo  Python is required. Please install from:
    echo  https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: Check "Add Python to PATH" during install!
    echo.
    pause
    exit /b 1
)
echo [1/4] Checking Python... OK
python --version

REM Install pip
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo [3/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [ERROR] Failed to install some dependencies.
    echo  Try running: pip install yfinance pandas numpy customtkinter
    pause
    exit /b 1
)
echo [3/4] Dependencies installed... OK

REM Test import
echo [4/4] Testing imports...
python -c "import yfinance, pandas, numpy, customtkinter; print('All imports OK')"
if errorlevel 1 (
    echo  [ERROR] Import test failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Setup complete! Launching scanner...
echo ========================================
echo.

REM Launch
python app.py
