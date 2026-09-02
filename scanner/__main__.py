"""
Entry point for running the scanner as a package.

Usage:
    python scanner              # launches GUI (default)
    python scanner --cli        # launches interactive CLI
    python scanner --gui        # launches GUI explicitly
"""
import sys


def main():
    # Initialize trace log first — captures everything from here on
    try:
        from .trace import setup_trace

        setup_trace()
    except Exception:
        pass  # never block startup on trace init

    args = sys.argv[1:]

    if "--cli" in args:
        from .run_scanner import run_scan
        run_scan()
    else:
        # Default: launch the GUI
        from .app import ScannerApp
        app = ScannerApp()
        app.mainloop()


if __name__ == "__main__":
    main()
