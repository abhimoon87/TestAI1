"""Shared timeout helpers for provider/fetcher calls that can hang."""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

TIMEOUT = object()  # Sentinel for timed-out provider calls


def call_with_timeout(fn, timeout: float):
    """Run *fn* on a daemon thread; return ``TIMEOUT`` if it exceeds *timeout*.

    Sets a socket-level deadline so third-party HTTP libs that ignore
    ``requests`` timeout kwargs cannot hang past *timeout*.
    """
    import socket as _socket

    result = [None]
    exc = [None]

    def _target():
        old = None
        try:
            old = _socket.getdefaulttimeout()
            _socket.setdefaulttimeout(timeout)
        except Exception:
            logger.debug("Failed to set socket timeout", exc_info=True)
        try:
            result[0] = fn()
        except Exception as e:
            exc[0] = e
        finally:
            try:
                if old is not None:
                    _socket.setdefaulttimeout(old)
            except Exception:
                logger.debug("Failed to restore socket timeout", exc_info=True)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        logger.debug(
            "Provider %s timed out after %.1fs", getattr(fn, "__name__", "?"), timeout
        )
        return TIMEOUT
    if exc[0] is not None:
        raise exc[0]
    return result[0]
