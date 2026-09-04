"""Parity tests: backtest.py and scoring.py must agree on identical fixtures.

The scanner has two implementations of the same 100-pt scoring model:

  - ``scoring.compute_scores``  — the live per-stock scorer used by the GUI /
    CLI (scores the *last* bar of the DataFrame it is given).
  - ``backtest.compute_score_at_bar`` — the per-bar scorer used by the
    backtest simulation (scores any ``bar_idx``).

After the volume-participation and direction-aware-sideways fixes were ported
into ``backtest.py`` (so live-scan and backtest agree), these tests pin that
parity: for the same OHLCV window, same NIFTY reference, and same settings,
both scorers must return the same ``total`` and the same ``is_sideways`` flag.

All data is synthetic — no network access.
"""

import numpy as np
import pandas as pd
import pytest

from scanner.backtest import DEFAULT_SETTINGS as BT_DEFAULTS
from scanner.backtest import (
    compute_score_at_bar,
    precompute_nifty,
    precompute_stock,
)
from scanner.scoring import compute_scores
from scanner.settings_store import DEFAULT_SETTINGS as GUI_DEFAULTS

# ---------------------------------------------------------------------------
# Synthetic fixtures (>= WARMUP_BARS=260 so precompute_stock accepts them)
# ---------------------------------------------------------------------------

def _ohlcv(n: int, seed: int, close_start: float = 100.0) -> pd.DataFrame:
    """Deterministic OHLCV: random walk with drift + volume bursts."""
    rng = np.random.RandomState(seed)
    drift = rng.randn(n) * 0.018 + 0.0008
    close = close_start * np.exp(np.cumsum(drift))
    spread = np.abs(rng.randn(n)) * 0.004
    high = close * (1 + spread)
    low = close * (1 - spread)
    open_ = close * (1 + rng.randn(n) * 0.002)
    volume = (rng.rand(n) * 1_200_000 + 600_000).astype(int)
    # Volume burst mid-series (breaks the flat-volume assumption)
    burst = slice(n // 2, n // 2 + 8)
    volume[burst] = volume[burst] * 4
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": volume},
        index=dates,
    )


@pytest.fixture
def index_ohlcv():
    """A NIFTY-like reference series on the same calendar (500 bars)."""
    return _ohlcv(500, seed=999, close_start=18_000.0)


@pytest.fixture
def settings():
    """Merged settings with a fast/loose-ish MA config; both scorers see it."""
    s = {**BT_DEFAULTS, **GUI_DEFAULTS}
    s.update({
        "fast_ma_type": "HMA", "fast_ma_len": 20,
        "slow_ma_type": "EMA", "slow_ma_len": 40,
        "crossover_lookback": 20,
        "vp_lookback": 50,            # keep VP window well under slice length
        "adx_threshold": 20.0,
        "chop_threshold": 61.8,
        "flat_threshold": 0.5,
        "sideways_strong_move_pct": 5.0,
        "volume_participation_len": 5,
    })
    return s


def _score_both(df: pd.DataFrame, nifty_df: pd.DataFrame, settings: dict,
                window_end: int | None = None):
    """Score the window ending at ``window_end`` (default: last bar) with both
    scorers on identical inputs and return (backtest_total, live_total,
    backtest_sideways, live_sideways)."""
    if window_end is None:
        window_end = len(df) - 1

    # Live scorer works on the window; NIFTY must be date-aligned to the same
    # window end or relative strength compares mismatched dates.
    stock_df = df.iloc[: window_end + 1].copy()
    idx_df = nifty_df[nifty_df.index <= df.index[window_end]].copy()

    # .iloc[:].copy() drops custom attributes — carry _fundamentals over so
    # the live scorer sees the same fundamentals as precompute_stock did.
    fund = getattr(df, "_fundamentals", None)
    if fund is not None:
        object.__setattr__(stock_df, "_fundamentals", fund)

    stock = precompute_stock("T", df, settings)  # precompute on full history
    assert stock is not None, "fixture too short for precompute_stock warmup"

    nifty_pre = precompute_nifty(idx_df)

    bt = compute_score_at_bar(stock, window_end, nifty_pre, settings)
    live = compute_scores(stock_df, "D", idx_df, settings)
    assert bt is not None and live is not None, "scorer returned None"

    return (
        bt["total"], live["total"],
        bool(bt["is_sideways"]), bool(live["is_sideways"]),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScorerParity:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_last_bar_total_and_sideways_match(self, seed, index_ohlcv, settings):
        """Whole-window (last bar) scoring agrees between both scorers."""
        df = _ohlcv(500, seed=seed)
        bt_t, live_t, bt_sw, live_sw = _score_both(df, index_ohlcv, settings)
        assert round(bt_t, 1) == round(live_t, 1), \
            f"total mismatch: backtest={bt_t} live={live_t}"
        assert bt_sw == live_sw, \
            f"sideways mismatch: backtest={bt_sw} live={live_sw}"

    @pytest.mark.parametrize("window_end", [320, 400, 470])
    def test_mid_history_windows_match(self, window_end, index_ohlcv, settings):
        """Scoring an earlier window (bar not at the end) also agrees."""
        df = _ohlcv(500, seed=7)
        bt_t, live_t, bt_sw, live_sw = _score_both(
            df, index_ohlcv, settings, window_end=window_end)
        assert round(bt_t, 1) == round(live_t, 1), \
            f"bar {window_end}: total mismatch: backtest={bt_t} live={live_t}"
        assert bt_sw == live_sw, \
            f"bar {window_end}: sideways mismatch: backtest={bt_sw} live={live_sw}"

    def test_fundamentals_attach_parity(self, index_ohlcv, settings):
        """When the frame carries _fundamentals, both scorers credit them."""
        df = _ohlcv(500, seed=5)
        fund = {"pe_ratio": 12.0, "eps_growth": 25.0,
                "rev_growth": 18.0, "roe": 22.0}  # -> 20/20 in both
        object.__setattr__(df, "_fundamentals", fund)
        bt_t, live_t, bt_sw, live_sw = _score_both(df, index_ohlcv, settings)
        assert round(bt_t, 1) == round(live_t, 1)
        assert bt_sw == live_sw
        # Sanity: fundamentals actually counted (without them both would be
        # lower) — proves the fixture attached and both scorers used it.
        plain_t, _, _, _ = _score_both(_ohlcv(500, seed=5), index_ohlcv, settings)
        assert round(bt_t, 1) > round(plain_t, 1)

    def test_tunable_knobs_do_not_break_parity(self, index_ohlcv, settings):
        """Non-default volume window / strong-move guard stay in sync."""
        df = _ohlcv(500, seed=11)
        for vol_len, move_pct in [(3, 5.0), (10, 5.0), (5, 3.0), (10, 8.0)]:
            s = dict(settings)
            s["volume_participation_len"] = vol_len
            s["sideways_strong_move_pct"] = move_pct
            bt_t, live_t, bt_sw, live_sw = _score_both(df, index_ohlcv, s)
            assert round(bt_t, 1) == round(live_t, 1), \
                f"vol_len={vol_len} move={move_pct}: totals differ"
            assert bt_sw == live_sw, \
                f"vol_len={vol_len} move={move_pct}: sideways differ"
