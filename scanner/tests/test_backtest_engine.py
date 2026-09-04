"""Offline regression tests for BacktestEngine calendar alignment.

The FNO 2x day-count anomaly: two cached tickers (GSPL, TATAMOTORS) carried
UTC-close stamps at 18:30 on the previous day while the other 108 tickers
shared local-midnight trade dates.  Same NSE trade days, different hashes --
so the engine's cross-ticker date union in ``run()`` (~backtest.py:889)
treated them as separate calendars and the simulation ran on ~2x the real
number of days.  The fetch layer now normalizes every daily frame onto one
tz-naive midnight IST calendar; these tests prove that invariant holds when
legacy mixed-flavor frames flow out of the disk cache through ``load_data()``
into ``run()``.
"""

import hashlib
import json
import os
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scanner import data_providers
from scanner.backtest import WARMUP_BARS, BacktestEngine


@pytest.fixture(autouse=True)
def _isolate_price_cache(tmp_path, monkeypatch):
    """Point the on-disk price cache at a temp dir (no shared .cache)."""
    monkeypatch.setattr(data_providers, "CACHE_DIR", str(tmp_path / "price_cache"))
    yield


# ── Helpers ────────────────────────────────────────────────────────────────


def _frame(days, seed):
    """OHLCV frame indexed by the given timestamps (random walk values)."""
    n = len(days)
    rng = np.random.RandomState(seed)
    close = 500 + np.cumsum(rng.randn(n) * 2)
    return pd.DataFrame({
        "open": close + rng.randn(n),
        "high": close + np.abs(rng.randn(n)) * 2,
        "low": close - np.abs(rng.randn(n)) * 2,
        "close": close,
        "volume": (rng.rand(n) * 1e6 + 5e5).astype(int),
    }, index=pd.DatetimeIndex(days))


def _write_legacy_cache_entry(ticker, period, provider, df):
    """Write a pkl entry the way pre-fix code did: raw stamps, no normalization."""
    os.makedirs(data_providers.CACHE_DIR, exist_ok=True)
    raw = f"{ticker}_{period}_{provider}_{date.today().isoformat()}"
    key = hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()
    cache_file = os.path.join(data_providers.CACHE_DIR, f"{key}.pkl")
    meta_file = os.path.join(data_providers.CACHE_DIR, f"{key}.meta")
    df.to_pickle(cache_file)
    with open(meta_file, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "rows": len(df)}, f)


def _seed_flavored_cache(n=500, start="2023-01-01"):
    """Plant two LEGACY cache entries covering the SAME trade days in two flavors.

    ``MIDNIGHT`` holds local-midnight naive stamps (the yfinance .NS path);
    ``UTCCLOSE`` holds tz-aware 18:30-UTC stamps on the previous day (the
    legacy jugaad/nselib path).  Written raw (as the pre-fix cache layer did),
    these union to ~2x n until a reader normalizes them.
    """
    trade_days = pd.bdate_range(start, periods=n)
    _write_legacy_cache_entry("MIDNIGHT", "2y", "cache", _frame(trade_days, seed=1))
    prev = trade_days - pd.Timedelta(days=1) + pd.Timedelta(hours=18, minutes=30)
    _write_legacy_cache_entry("UTCCLOSE", "2y", "cache", _frame(prev.tz_localize("UTC"), seed=2))
    return trade_days


def _load_engine(tmp_path, monkeypatch):
    """Seed the flavored cache, patch index data, load the engine offline."""
    trade_days = _seed_flavored_cache()
    nifty = _frame(trade_days, seed=3)
    monkeypatch.setattr(
        "scanner.backtest.fetch_index_data", lambda *a, **k: nifty
    )
    mock_yf = MagicMock()
    mock_yf.download.side_effect = AssertionError("yfinance must not be called")
    with patch.dict("sys.modules", {"yfinance": mock_yf}):
        engine = BacktestEngine()
        engine.load_data(["MIDNIGHT", "UTCCLOSE"], period="2y")
    return engine, trade_days


# ══════════════════════════════════════════════════════════════════════════════
# Calendar alignment
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineCalendarAlignment:
    def test_mixed_flavor_cache_heals_to_one_calendar(self, tmp_path, monkeypatch):
        """Both flavors load and land on the identical trade-date calendar."""
        engine, trade_days = _load_engine(tmp_path, monkeypatch)

        assert len(engine.stocks) == 2
        a, b = engine.stocks
        # Every stock's simulation index is the SAME calendar -- no drift.
        assert a.df.index.equals(b.df.index)
        assert a.df.index.tz is None
        assert list(a.df.index) == list(trade_days)

    def test_union_never_exceeds_shared_calendar(self, tmp_path, monkeypatch):
        """run()'s cross-ticker union equals one calendar, not ~2x it."""
        engine, trade_days = _load_engine(tmp_path, monkeypatch)

        # Mirror the union run() builds for its simulation window (~backtest.py:889).
        union = sorted(set().union(*[s.df.index for s in engine.stocks]))
        assert len(union) == len(trade_days)  # 500, not ~1000
        assert union == list(trade_days)

    def test_run_simulation_window_not_doubled(self, tmp_path, monkeypatch, capsys):
        """The engine's simulated day count matches the real single calendar."""
        engine, trade_days = _load_engine(tmp_path, monkeypatch)

        metrics = engine.run()
        out = capsys.readouterr().out

        expected_days = len(trade_days) - WARMUP_BARS
        assert f"Simulation: {expected_days} trading days" in out
        assert isinstance(metrics, dict)

    def test_raw_flavors_would_have_doubled(self, tmp_path, monkeypatch):
        """Control: un-normalized mixed frames really do inflate the union.

        This documents why the fetch layer (not run()) must own normalization:
        without it, two frames for the same 500 trade days union to ~1000.
        """
        engine, trade_days = _load_engine(tmp_path, monkeypatch)
        a, b = engine.stocks  # already normalized by the fetch layer

        # Rebuild the un-normalized flavors and confirm the union doubles.
        midnight = _frame(trade_days, seed=1)
        prev = trade_days - pd.Timedelta(days=1) + pd.Timedelta(hours=18, minutes=30)
        utc_close = _frame(prev.tz_localize("UTC"), seed=2)
        raw_union = midnight.index.union(utc_close.index)
        assert len(raw_union) == 2 * len(trade_days)
        # The same frames, once through the fetch layer, share one calendar.
        assert list(a.df.index) == list(trade_days)
        assert list(b.df.index) == list(trade_days)
