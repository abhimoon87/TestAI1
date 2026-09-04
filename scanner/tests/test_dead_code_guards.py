"""Regression guards for dead code removed from the package.

Every symbol in ``DEAD_SYMBOLS`` was confirmed definition-only (zero
references package-wide, tests included) immediately before its removal.
This module re-scans the whole scanner package textually and fails the
moment any of them reappears -- as a definition, an import, a test
reference, or even a stray docstring mention -- so a "re-add the helper"
cannot land silently.

If a symbol must legitimately come back, delete it from ``DEAD_SYMBOLS``
in the same change that re-adds it.
"""

import pathlib
import re

import pytest

# fmt: off
DEAD_SYMBOLS = [
    # themes.py
    "apply_theme",
    # scanner_engine.py
    "ScanProgress",
    # data_fetcher.py
    "fetch_stock_fast",
    # symbol_fetcher.py
    "_STATIC_FALLBACKS",
    "fetch_bse_static_universes",
    "MAX_BSE_VALIDATE",
    "validate_bse_symbols",
    # free_apis.py (a duplicate copy also lived in indian_market.py)
    "PincodeData",
    "fetch_pincode_data",
    "fetch_all_free_apis",
    "WorldMacroData",
    "fetch_world_macro_data",
    "TopStockPick",
    "fetch_top_stocks",
    # premium_finance.py
    "fetch_premium_finance",
    # trace.py
    "log_call",
    "tail_trace",
    # backtest.py and universes.py each carried an unused copy
    "SECTOR_COLORS",
]
# fmt: on

_PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
_THIS_FILE = pathlib.Path(__file__).resolve()

# Scan once at collection time: every *.py under scanner/ except this file.
_PACKAGE_TEXT = "\n".join(
    p.read_text(encoding="utf-8", errors="ignore")
    for p in sorted(_PKG_ROOT.rglob("*.py"))
    if p.resolve() != _THIS_FILE
)


@pytest.mark.parametrize("symbol", DEAD_SYMBOLS)
def test_dead_symbol_stays_removed(symbol):
    """A removed dead symbol must not be referenced anywhere in the package."""
    assert re.search(rf"\b{re.escape(symbol)}\b", _PACKAGE_TEXT) is None, (
        f"'{symbol}' was removed as dead code but is referenced again; "
        "drop it from DEAD_SYMBOLS only when re-adding it on purpose"
    )
