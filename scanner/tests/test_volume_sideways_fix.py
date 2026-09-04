"""Tests for the volume / sideways scoring fixes.

Covers the two behaviours requested after the MAHABANK deep-dive:
1. Volume — a move that accumulated volume over the breakout week should
   still score even when the single latest bar printed below its MAs.
2. Sideways — a strong directional move must not be labelled sideways on a
   lone Choppiness/slope trigger; genuinely flat markets still are.
"""

import numpy as np
import pandas as pd

from scanner.indicators import adx
from scanner.scoring import _compute_sideways, compute_scores


def _ohlcv(close: np.ndarray, volume: np.ndarray | None = None) -> pd.DataFrame:
    n = len(close)
    high = close * 1.01
    low = close * 0.99
    open_ = close * 0.995
    if volume is None:
        volume = np.full(n, 1_000_000.0)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_volume_scores_breakout_week_participation():
    """Last bar quiet but the prior 5 bars expanded -> volume > 0."""
    n = 130
    close = 100.0 + np.arange(n) * 0.05  # gentle drift, nothing weird
    volume = np.full(n, 1_000_000.0)
    volume[-6:-1] = 2_400_000.0   # 5 expanding bars...
    volume[-1] = 600_000.0        # ...then a quiet closing day
    df = _ohlcv(close, volume)

    result = compute_scores(df, timeframe="D", settings={})
    assert result is not None
    # The quiet last bar alone would have scored 0 under the old logic
    assert result["volume"] > 0


def test_volume_still_zero_when_participation_is_low():
    """No volume expansion anywhere -> the category stays at zero."""
    n = 130
    close = 100.0 + np.arange(n) * 0.05
    volume = np.full(n, 500_000.0)   # flat, and below nothing meaningful
    volume[-1] = 300_000.0
    df = _ohlcv(close, volume)
    result = compute_scores(df, timeframe="D", settings={})
    assert result is not None
    assert result["volume"] == 0


def _sideways_for(close: np.ndarray, volume: np.ndarray | None = None) -> bool:
    df = _ohlcv(close, volume)
    adx_val = adx(df["high"], df["low"], df["close"], 14)
    return _compute_sideways(df, adx_val, {})["is_sideways"]


def test_strong_rally_is_not_sideways():
    """A >5% one-month rally must override lone chop/slope triggers."""
    n = 130
    # ~13% rally over the final 21 bars from a flat base
    base = np.full(n - 21, 100.0)
    rally = np.linspace(100.0, 113.0, 21)
    close = np.concatenate([base, rally])
    # Trend needs ADX evidence too; this must stay trending regardless
    assert not _sideways_for(close)


def test_flat_market_still_sideways():
    """Real sideways (constant price) is still flagged sideways."""
    assert _sideways_for(np.full(130, 100.0))


def test_sideways_reasons_empty_when_strong_move_overrides():
    """When a strong move clears the flag, reasons must not lie."""
    n = 130
    close = np.concatenate([np.full(n - 21, 100.0), np.linspace(100.0, 112.0, 21)])
    df = _ohlcv(close)
    adx_val = adx(df["high"], df["low"], df["close"], 14)
    out = _compute_sideways(df, adx_val, {})
    assert not out["is_sideways"]
    assert out["reasons"] == []
