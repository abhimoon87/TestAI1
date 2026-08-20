"""
Entry point for running the scanner as a package.

Usage:
    python scanner              # launches GUI (default)
    python scanner --cli        # launches interactive CLI
    python scanner --gui        # launches GUI explicitly
"""
import sys


def main():
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
