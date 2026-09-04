"""Shared index utilities with zero scanner-internal imports.

Breaking ``_normalize_daily_index`` out of ``data_fetcher`` into this module
eliminates the circular import between ``data_fetcher`` and ``data_providers``
(both need this function, but each imports the other at module level).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a daily OHLCV frame onto tz-naive IST trade dates (midnight).

    Daily frames arrive in two flavors: local-midnight stamps (the yfinance
    ``.NS`` path) and UTC-close stamps at 18:30 on the previous day (some
    fallback providers).  Both encode the same NSE trade day, but the stamps
    hash differently -- so cross-ticker date unions (backtest alignment,
    relative-strength date masks) double-count every day and drift by one.
    This treats any naive stamp as UTC, converts to Asia/Kolkata and truncates
    to midnight: 18:30 UTC becomes 00:00 the next day, and local midnights are
    unchanged (00:00 UTC -> 05:30 IST, same date).  Same-day collisions from
    duplicate or dual-provider rows keep the last bar.
    """
    if df is None or df.empty:
        return df
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(idx))
        except Exception:
            return df
    if idx.tz is None:
        try:
            idx = idx.tz_localize("UTC")
        except Exception:
            logger.debug("Timezone localization to UTC failed", exc_info=True)
    try:
        dates = idx.tz_convert("Asia/Kolkata").tz_localize(None).normalize()
    except Exception:
        dates = idx.normalize() if getattr(idx, "tz", None) is None \
            else idx.tz_localize(None).normalize()
    out = df.copy()
    out.index = dates
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()
