"""Static universe fallback — last-resort symbol lists must resolve.

Guards the import path (scanner.shared.universes): a wrong relative import
here only surfaces when live NSE fetches fail, which is exactly when the
fallback matters.
"""

from scanner.api.symbol_fetcher import _get_static_fallback

_KEYS = (
    "mainboard",
    "fno",
    "nifty50",
    "niftynext50",
    "midcap150",
    "smallcap250",
    "unique",
    "bse_all",
    "all_market",
)


def test_static_fallback_lists_resolve():
    for key in _KEYS:
        assert _get_static_fallback(key), f"{key} fallback empty (import broken?)"
