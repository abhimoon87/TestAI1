"""
Entry point for running the scanner as a package.

Usage:
    python scanner              # launches GUI (default, Flet edition)
    python scanner --cli        # launches interactive CLI
    python scanner --gui        # launches GUI explicitly (Flet edition)
"""
import sys


def main():
    try:
        from .shared.trace import setup_trace
        setup_trace()
    except Exception as exc:
        print(f"Trace setup skipped: {exc}", file=sys.stderr)

    args = sys.argv[1:]

    if "--cli" in args:
        from .backend.run_scanner import run_scan
        run_scan()
    else:
        import flet as ft

        from .ui.app import main as app_main
        ft.run(app_main)


if __name__ == "__main__":
    main()
