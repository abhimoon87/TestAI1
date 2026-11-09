"""
Entry point for running the scanner as a package.

Usage:
    python scanner              # launches GUI (default, Flet edition)
    python scanner --cli        # launches interactive CLI
    python scanner --gui        # launches GUI explicitly (Flet edition)
"""

import socket
import sys

# CPU-bound workers (yfinance, scoring, VACUUM) otherwise hold the GIL for
# 5ms slices — the Flet event loop starves and the UI feels frozen. 1ms
# slices keep clicks/navigation responsive while scans burn CPU.
sys.setswitchinterval(0.001)

# Arbitrary fixed port for the single-instance lock; a clash with unrelated
# software would only falsely report "already running".
_LOCK_PORT = 47832
# Kept alive so the socket isn't GC'd (GC would close it and free the port).
_lock_sock: socket.socket | None = None


def _acquire_single_instance_lock(port: int = _LOCK_PORT) -> socket.socket:
    """Hold a localhost bind for the life of this process.

    The OS frees the socket on exit — even a crash — so there are no stale
    lock files to clean up. A second instance fails to bind and exits.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        sock.close()
        raise SystemExit(
            "HMAxEMA is already running — close the other instance first."
        ) from None
    return sock


def main():
    global _lock_sock
    _lock_sock = _acquire_single_instance_lock()

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
