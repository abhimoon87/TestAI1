"""
HMAxEMA Stock Scanner — Indian Market Screener
"""

import logging
import sys

# ── Package Logger ─────────────────────────────────────────────────────────
# All modules in this package use `logger = logging.getLogger(__name__)`
# to emit messages.  The root ``scanner`` logger is configured here so
# that callers only need a single ``logging.basicConfig()`` call (or none
# at all — a default StreamHandler to stderr is attached automatically).
#
# Levels used:
#   DEBUG    — verbose diagnostic detail (off by default)
#   INFO     — normal progress / user-facing status messages
#   WARNING  — non-fatal issues (provider fallback, missing data)
#   ERROR    — operation-level failures (all providers failed, etc.)

logger = logging.getLogger("scanner")

# Attach a default handler so the package works out-of-the-box even if
# the application has not called logging.basicConfig().
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
