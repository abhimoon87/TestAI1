"""
Centralized trace logging for the entire HMAxEMA Scanner application.

Creates a single `trace.log` that captures:
- App lifecycle (startup, theme, view switches)
- Scan pipeline (universe resolve → batch download → filter → scoring → display)
- Data fetch (yfinance, nselib, cache hits/misses)
- Scoring (filter decisions, scores)
- Provider enrichment (sentiment, fundamentals, macro)
- UI events (button clicks, searches, exports)
- Errors with full stack traces

Usage:
    from .trace import setup_trace, trace, get_trace_path

    setup_trace()  # call once at startup (idempotent)
    @trace
    def my_func(...): ...

All existing `logging.getLogger(__name__)` calls automatically flow to
`trace.log` because setup_trace() configures the root logger.

Log file: scanner/trace.log (rotating, 5 MB × 5 backups)
"""

from __future__ import annotations

import functools
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
import uuid
import psutil
from collections.abc import Callable
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

SCANNER_DIR = Path(__file__).parent
PROJECT_ROOT = SCANNER_DIR.parent.parent
APPLOG_DIR = PROJECT_ROOT / "AppLog"
DEFAULT_TRACE_FILE = APPLOG_DIR / "trace.log"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 5

# Custom TRACE level (below DEBUG) for ultra-verbose entry/exit
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, msg, args, **kwargs)


logging.Logger.trace = _trace  # type: ignore[attr-defined]

_configured = False
_trace_path: Path = DEFAULT_TRACE_FILE


def get_trace_path() -> Path:
    return _trace_path

# ── Global exception handling ─────────────────────────────────────────────

_prev_excepthook = sys.excepthook
_prev_thread_hook = getattr(threading, "excepthook", None)

# Correlation ID generator
def _generate_correlation_id():
    """Generate a unique correlation ID for request tracking."""
    return f"{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"

# Global exception handler
def _global_exception_handler(exc_type, exc_value, exc_tb):
    logger = logging.getLogger("unhandled")
    logger.critical(
        "GLOBAL_EXCEPTION: thread=%s module=%s error=%s stack=%s",
        threading.current_thread().name,
        exc_tb.tb_frame.f_code.co_filename,
        str(exc_value),
        traceback.format_exception(exc_type, exc_value, exc_tb),
        exc_info=(exc_type, exc_value, exc_tb)
    )

# Thread exception handler
def _thread_exception_handler(args):
    logger = logging.getLogger("thread")
    logger.error(
        "THREAD_EXCEPTION: thread=%s module=%s error=%s",
        args.thread.name if args.thread else "?",
        args.filename,
        str(args.exc_value),
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )

# System health monitoring
def _log_system_health():
    """Log system health metrics."""
    logger = logging.getLogger("system")
    logger.info(
        "SYSTEM_HEALTH: memory=%.1f%% cpu=%.1f%% threads=%d",
        psutil.virtual_memory().percent,
        psutil.cpu_percent(),
        threading.active_count()
    )

# Apply global exception handlers
sys.excepthook = _global_exception_handler
if hasattr(threading, "excepthook"):
    threading.excepthook = _thread_exception_handler

# ── Uncaught exception hook ────────────────────────────────────────────

def _excepthook(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logging.getLogger("unhandled").critical(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
    )
    # Chain to previous hook (e.g. pytest, Sentry)
    if _prev_excepthook is not sys.__excepthook__:
        _prev_excepthook(exc_type, exc_value, exc_tb)

# ── Thread exception hook (Python 3.8+) ────────────────────────────────

def _thread_excepthook(args):
    logging.getLogger("thread").error(
        "Thread exception: thread=%s exception=%s",
        args.thread.name if args.thread else "?",
        args.exc_value,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )
    if _prev_thread_hook is not None and _prev_thread_hook is not threading.excepthook:
        _prev_thread_hook(args)


def setup_trace(
    log_file: str | os.PathLike | None = None,
    level: int = logging.INFO,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
    console: bool = True,
    also_scan_log: bool = True,
) -> Path:
    """
    Configure root logging to file + console. Idempotent — safe to call
    multiple times (reconfigures only once unless force=True).
    Returns the trace file path.
    """
    global _configured, _trace_path
    if _configured:
        return _trace_path

    if log_file is not None:
        _trace_path = Path(log_file)
    _trace_path.parent.mkdir(parents=True, exist_ok=True)

    # Formatter — ms precision, thread, module.func:lineno
    fmt = (
        "%(asctime)s.%(msecs)03d | %(levelname)-5s | %(threadName)-12s | "
        "%(name)s.%(funcName)s:%(lineno)d | %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(min(level, TRACE_LEVEL))

    # ── Rotating file handler ─────────────────────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        str(_trace_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # ── Console handler (stderr) ───────────────────────────────────────────
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.INFO)  # console less verbose
        ch.setFormatter(formatter)
        root.addHandler(ch)

    # ── Scan log mirror (optional) — also route INFO+ to scan.log ─────────
    # scan.log is the user-visible activity log; trace.log is the detailed one.
    # We keep them separate; setup_trace does not duplicate into scan.log
    # unless also_scan_log=True and you want a single sink.
    # Currently we leave scan.log to app.py's _log() file writes.

    # ── Log startup banner ─────────────────────────────────────────────────
    logging.getLogger("trace").info(
        "Trace initialized -- file=%s level=%s maxBytes=%d backups=%d",
        _trace_path,
        logging.getLevelName(level),
        max_bytes,
        backup_count,
    )
    logging.getLogger("trace").info(
        "Python %s | Platform %s | CWD %s",
        sys.version.split()[0],
        sys.platform,
        os.getcwd(),
    )

    # Log system health after setup
    _log_system_health()

    _configured = True
    return _trace_path


def trace(
    _func: Callable | None = None,
    *,
    level: int = TRACE_LEVEL,
    log_args: bool = True,
    log_result: bool = False,
    max_arg_len: int = 300,
) -> Callable:
    """
    Decorator that logs function entry, exit (with duration), and exceptions.

    @trace
    def foo(x, y): ...

    @trace(level=logging.DEBUG, log_args=False)
    def hot_path(...): ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            if not logger.isEnabledFor(level):
                return func(*args, **kwargs)

            # Generate correlation ID for request tracking
            correlation_id = _generate_correlation_id()

            # Build arg repr (truncate, hide large DataFrames)
            def _repr(v):
                try:
                    s = repr(v)
                    # For DataFrames, show shape instead of full data
                    if "DataFrame" in type(v).__name__:
                        shape = getattr(v, "shape", "?")
                        s = f"<DataFrame shape={shape}>"
                    if len(s) > max_arg_len:
                        s = s[:max_arg_len] + "..."
                    return s
                except Exception:
                    return f"<{type(v).__name__}>"

            if log_args:
                args_r = ", ".join(_repr(a) for a in args)
                kwargs_r = ", ".join(f"{k}={_repr(v)}" for k, v in kwargs.items())
                all_r = ", ".join(filter(None, [args_r, kwargs_r]))
                logger.log(
                    level,
                    "TRACE_START: correlation_id=%s func=%s args=%s",
                    correlation_id, func.__qualname__, all_r
                )
            else:
                logger.log(
                    level,
                    "TRACE_START: correlation_id=%s func=%s",
                    correlation_id, func.__qualname__
                )

            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                dt = (time.perf_counter() - t0) * 1000
                if log_result:
                    logger.log(
                        level,
                        "TRACE_COMPLETE: correlation_id=%s func=%s result=%s elapsed_ms=%.1f",
                        correlation_id, func.__qualname__, _repr(result), dt
                    )
                else:
                    logger.log(
                        level,
                        "TRACE_COMPLETE: correlation_id=%s func=%s elapsed_ms=%.1f",
                        correlation_id, func.__qualname__, dt
                    )
                return result
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000
                logger.log(
                    logging.ERROR if level <= logging.DEBUG else level,
                    "TRACE_ERROR: correlation_id=%s func=%s error=%s elapsed_ms=%.1f",
                    correlation_id,
                    func.__qualname__,
                    str(e),
                    dt,
                    exc_info=True
                )
                raise

        return wrapper

    if _func is not None:
        return decorator(_func)
    return decorator

# Add system health logging to setup_trace
_original_setup_trace = setup_trace

def setup_trace_with_health(
    log_file: str | os.PathLike | None = None,
    level: int = logging.INFO,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
    console: bool = True,
    also_scan_log: bool = True,
) -> Path:
    """
    Enhanced setup_trace with system health monitoring.
    """
    trace_path = _original_setup_trace(
        log_file=log_file,
        level=level,
        max_bytes=max_bytes,
        backup_count=backup_count,
        console=console,
        also_scan_log=also_scan_log,
    )

    # Log system health after setup
    _log_system_health()

    return trace_path
