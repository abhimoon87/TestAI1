#!/bin/bash
# HMAxEMA Stock Scanner — macOS/Linux Launcher
cd "$(dirname "$0")/.."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "  [ERROR] Python 3 is not installed."
    echo "  Install via: brew install python3  (macOS)"
    echo "               sudo apt install python3  (Linux)"
    echo ""
    read -p "  Press Enter to exit..."
    exit 1
fi

# Check dependencies
python3 -c "import customtkinter" &> /dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "  First time setup - installing dependencies..."
    echo ""
    pip3 install -r scanner/requirements.txt
    if [ $? -ne 0 ]; then
        echo "  [ERROR] Failed to install dependencies."
        read -p "  Press Enter to exit..."
        exit 1
    fi
    echo "  Dependencies installed successfully!"
    echo ""
fi

# Launch
echo "Starting HMAxEMA Stock Scanner..."
python3 -m scanner.app
