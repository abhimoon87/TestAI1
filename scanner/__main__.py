"""
Entry point for running the scanner as a package.

Usage:
    python scanner              # launches GUI (default)
    python scanner --cli        # launches interactive CLI
    python scanner --gui        # launches GUI explicitly
"""
import sys


def main():
    try:
        from .trace import setup_trace
        setup_trace()
    except Exception:
        pass

    args = sys.argv[1:]

    if "--cli" in args:
        from .run_scanner import run_scan
        run_scan()
    else:
        import flet as ft
        from .app import main as app_main
        ft.run(app_main)


if __name__ == "__main__":
    main()
