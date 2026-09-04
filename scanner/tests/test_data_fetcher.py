"""Unit tests for scanner.data_fetcher — data fetching and resampling.

All external API calls (yfinance, jugaad, nselib) are mocked so tests
run fast and offline.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scanner import data_fetcher
from scanner.data_fetcher import (
    CHUNK,
    FALLBACK_PROVIDER_TIMEOUT,
    MAX_PARALLEL_CHUNKS,
    _extend_period_for_timeframe,
    fetch_batch_yfinance,
    fetch_fundamentals,
    fetch_index_data,
    fetch_stock_data,
    resample_ohlcv,
)


@pytest.fixture(autouse=True)
def _isolate_negative_cache(tmp_path, monkeypatch):
    """Point the on-disk negative cache at a temp file and reset it per test."""
    monkeypatch.setattr(
        data_fetcher, "_NEGATIVE_CACHE_PATH",
        str(tmp_path / "dead_symbols.json"),
    )
    monkeypatch.setattr(data_fetcher, "_negative_cache", None)
    monkeypatch.setattr(
        data_fetcher, "_negative_cache_ttl_hours",
        float(data_fetcher.NEGATIVE_CACHE_TTL_HOURS),
    )
    yield
    monkeypatch.setattr(data_fetcher, "_negative_cache", None)


@pytest.fixture(autouse=True)
def _isolate_enrichment_cache(tmp_path, monkeypatch):
    """Point the on-disk enrichment cache at a temp file and reset per test."""
    monkeypatch.setattr(
        data_fetcher, "_ENRICHMENT_CACHE_PATH",
        str(tmp_path / "enrichment_cache.json"),
    )
    monkeypatch.setattr(data_fetcher, "_enrichment_cache", None)
    monkeypatch.setattr(
        data_fetcher, "ENRICHMENT_CACHE_TTL_HOURS",
        float(data_fetcher.ENRICHMENT_CACHE_TTL_HOURS),
    )
    yield
    monkeypatch.setattr(data_fetcher, "_enrichment_cache", None)


@pytest.fixture(autouse=True)
def _isolate_price_cache(tmp_path, monkeypatch):
    """Point the per-ticker price pkl cache at a temp dir (no shared .cache).

    The batch download path now consults the same disk cache as the
    per-ticker providers; without isolation tests could read real pkl files
    left by live scans.
    """
    from scanner import data_providers

    monkeypatch.setattr(data_providers, "CACHE_DIR", str(tmp_path / "price_cache"))
    yield


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_daily_ohlcv(n=200, start="2024-01-01"):
    """Create a realistic daily OHLCV DataFrame."""
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.RandomState(42)
    close = 500 + np.cumsum(rng.randn(n) * 2)
    return pd.DataFrame({
        "open": close + rng.randn(n),
        "high": close + np.abs(rng.randn(n)) * 2,
        "low": close - np.abs(rng.randn(n)) * 2,
        "close": close,
        "volume": (rng.rand(n) * 1e6 + 5e5).astype(int),
    }, index=dates)


def _make_yf_download_result(tickers, n=200, force_multi=False):
    """Simulate yfinance.download() return value with MultiIndex columns.

    By default a single ticker yields the flat DataFrame shape yfinance uses;
    pass force_multi=True to force the MultiIndex shape (e.g. to simulate a
    partial result where one requested ticker is missing).
    """
    dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.RandomState(42)
    close = 500 + np.cumsum(rng.randn(n) * 2)

    if len(tickers) == 1 and not force_multi:
        # Single ticker → flat DataFrame
        return pd.DataFrame({
            "Open": close + rng.randn(n),
            "High": close + np.abs(rng.randn(n)) * 2,
            "Low": close - np.abs(rng.randn(n)) * 2,
            "Close": close,
            "Volume": (rng.rand(n) * 1e6 + 5e5).astype(int),
        }, index=dates)

    # Multiple tickers → MultiIndex columns
    arrays = []
    for t in tickers:
        arrays.extend([(t, c) for c in ["Open", "High", "Low", "Close", "Volume"]])
    multi_cols = pd.MultiIndex.from_tuples(arrays)

    data = np.zeros((n, len(tickers) * 5))
    for i, t in enumerate(tickers):
        rng2 = np.random.RandomState(42 + i)
        c = 500 + np.cumsum(rng2.randn(n) * 2)
        data[:, i * 5] = c + rng2.randn(n)           # Open
        data[:, i * 5 + 1] = c + np.abs(rng2.randn(n)) * 2  # High
        data[:, i * 5 + 2] = c - np.abs(rng2.randn(n)) * 2  # Low
        data[:, i * 5 + 3] = c                        # Close
        data[:, i * 5 + 4] = rng2.rand(n) * 1e6 + 5e5  # Volume

    return pd.DataFrame(data, index=dates, columns=multi_cols)


# ══════════════════════════════════════════════════════════════════════════════
# resample_ohlcv
# ══════════════════════════════════════════════════════════════════════════════


class TestResampleOhlcv:
    def test_daily_no_change(self):
        """Timeframe='D' should return input unchanged."""
        df = _make_daily_ohlcv(50)
        result = resample_ohlcv(df, "D")
        pd.testing.assert_frame_equal(result, df)

    def test_weekly_reduces_bars(self):
        """Weekly resampling should reduce bar count by ~5x."""
        df = _make_daily_ohlcv(200)
        result = resample_ohlcv(df, "W")
        assert result is not None
        assert len(result) < len(df)
        assert len(result) >= 30  # ~40 weeks in 200 trading days

    def test_monthly_reduces_bars(self):
        """Monthly resampling should reduce bar count by ~20x."""
        df = _make_daily_ohlcv(500)
        result = resample_ohlcv(df, "M")
        assert result is not None
        assert len(result) < len(df)

    def test_weekly_ohlc_correctness(self):
        """Weekly open = first daily open, high = max, low = min, close = last."""
        # Create 10 daily bars in one week
        dates = pd.bdate_range("2024-01-01", periods=10)
        df = pd.DataFrame({
            "open":  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "high":  [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            "low":   [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "close": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "volume": [100] * 10,
        }, index=dates)
        result = resample_ohlcv(df, "W")
        assert result is not None
        # Should have 1-2 weekly bars
        assert len(result) <= 2

    def test_none_input(self):
        assert resample_ohlcv(None, "W") is None

    def test_empty_dataframe(self):
        result = resample_ohlcv(pd.DataFrame(), "W")
        assert result is not None and result.empty

    def test_non_datetime_index_with_date_column(self):
        """DataFrame with a 'date' column should be resampled."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=200),
            "open": np.ones(200),
            "high": np.ones(200) * 1.1,
            "low": np.ones(200) * 0.9,
            "close": np.ones(200),
            "volume": np.ones(200) * 1000,
        })
        result = resample_ohlcv(df, "W")
        assert result is not None
        assert len(result) < 200

    def test_timezone_stripped(self):
        """Timezone-aware index should be handled gracefully."""
        dates = pd.bdate_range("2024-01-01", periods=200, tz="UTC")
        df = _make_daily_ohlcv(200)
        df.index = dates
        result = resample_ohlcv(df, "W")
        assert result is not None
        assert result.index.tz is None


# ══════════════════════════════════════════════════════════════════════════════
# _extend_period_for_timeframe
# ══════════════════════════════════════════════════════════════════════════════


class TestExtendPeriod:
    def test_daily_unchanged(self):
        assert _extend_period_for_timeframe("1y", "D") == "1y"

    def test_weekly_extends(self):
        assert _extend_period_for_timeframe("1y", "W") == "2y"
        assert _extend_period_for_timeframe("6mo", "W") == "1y"

    def test_monthly_extends_more(self):
        assert _extend_period_for_timeframe("1y", "M") == "5y"
        assert _extend_period_for_timeframe("6mo", "M") == "2y"

    def test_unknown_period_defaults(self):
        assert _extend_period_for_timeframe("10y", "W") == "2y"
        assert _extend_period_for_timeframe("10y", "M") == "5y"


# ══════════════════════════════════════════════════════════════════════════════
# fetch_batch_yfinance
# ══════════════════════════════════════════════════════════════════════════════


class TestFetchBatchYfinance:
    def test_multi_ticker_batch(self):
        """Multi-ticker batch download with mocked yfinance."""
        tickers = ["RELIANCE", "TCS"]
        mock_data = _make_yf_download_result([f"{t}.NS" for t in tickers], n=200)

        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_batch_yfinance(tickers, period="1y")

        assert len(result) == 2
        assert "RELIANCE" in result
        assert "TCS" in result
        for t in tickers:
            df = result[t]
            assert list(df.columns) == ["open", "high", "low", "close", "volume"]
            assert len(df) >= 50

    def test_single_ticker_batch(self):
        """Single-ticker batch (flat DataFrame, not MultiIndex)."""
        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=200)

        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_batch_yfinance(["RELIANCE"], period="1y")

        assert len(result) == 1
        assert "RELIANCE" in result

    def test_empty_download(self):
        """Empty download result → empty dict."""
        mock_yf = MagicMock()
        mock_yf.download.return_value = pd.DataFrame()
        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = None

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
                result = fetch_batch_yfinance(["RELIANCE"], period="1y")

        assert result == {}

    def test_partial_failure(self):
        """One ticker succeeds, one fails (too few rows after dropna)."""
        dates = pd.bdate_range("2024-01-01", periods=200)
        rng = np.random.RandomState(42)
        close = 500 + np.cumsum(rng.randn(200) * 2)

        # Good ticker data (200 rows)
        good = np.column_stack([
            close + rng.randn(200),
            close + np.abs(rng.randn(200)),
            close - np.abs(rng.randn(200)),
            close,
            rng.rand(200) * 1e6,
        ])

        # Bad ticker data — same length but contains NaN so dropna leaves < 20 rows
        bad = np.full((200, 5), np.nan)
        bad[:10, :] = 100  # only 10 valid rows

        multi_cols = pd.MultiIndex.from_tuples([
            ("RELIANCE.NS", c) for c in ["Open", "High", "Low", "Close", "Volume"]
        ] + [
            ("TCS.NS", c) for c in ["Open", "High", "Low", "Close", "Volume"]
        ])
        data = np.hstack([good, bad])
        mock_data = pd.DataFrame(data, index=dates, columns=multi_cols)

        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = None

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
                result = fetch_batch_yfinance(["RELIANCE", "TCS"], period="1y")

        assert "RELIANCE" in result
        assert "TCS" not in result  # too short after dropna, and fallback returns None

    def test_import_error_returns_empty(self):
        """If yfinance is not importable, return empty dict."""
        with patch.dict("sys.modules", {"yfinance": None}):
            result = fetch_batch_yfinance(["RELIANCE"], period="1y")
        assert result == {}

    def test_weekly_timeframe_resamples(self):
        """Batch fetch with timeframe='W' should resample to weekly."""
        # Need enough daily bars so that after weekly resampling we still have >= 50
        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=500)

        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_batch_yfinance(["RELIANCE"], period="2y", timeframe="W")

        assert "RELIANCE" in result
        df = result["RELIANCE"]
        # Weekly from 500 daily bars ≈ 100 weeks, must be < 500 and >= 50
        assert len(df) < 500
        assert len(df) >= 50

    def test_ticker_suffix_mapping(self):
        """Tickers should have .NS suffix added for yfinance."""
        mock_data = _make_yf_download_result(["INFY.NS"], n=200)

        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            fetch_batch_yfinance(["INFY"], period="1y")

        # Verify .NS suffix was added
        call_args = mock_yf.download.call_args
        assert call_args[0][0] == ["INFY.NS"]


class TestBatchDownloadCache:
    """Batch downloads must serve fresh disk-cached bars and skip re-downloading."""

    def test_all_cached_skips_yfinance_entirely(self):
        """Every ticker fresh on disk -> no yfinance call, results still returned."""
        cached = {
            "RELIANCE": _make_daily_ohlcv(200),
            "TCS": _make_daily_ohlcv(200),
        }

        def fake_get(ticker, period, provider):
            return cached.get(ticker)

        mock_yf = MagicMock()
        mock_yf.download.side_effect = AssertionError("yfinance must not be called")

        with patch("scanner.data_fetcher._get_cached", side_effect=fake_get):
            with patch("scanner.data_fetcher._set_cached") as mock_set:
                with patch.dict("sys.modules", {"yfinance": mock_yf}):
                    result = fetch_batch_yfinance(["RELIANCE", "TCS"], period="1y")

        assert set(result) == {"RELIANCE", "TCS"}
        assert mock_set.call_count == 0  # nothing freshly downloaded to store

    def test_mixed_cached_and_fresh_downloads_only_missing(self):
        """Cached tickers skip the network; only the uncached one hits yfinance."""
        cached = {"RELIANCE": _make_daily_ohlcv(200)}
        mock_data = _make_yf_download_result(["TCS.NS"], n=200)

        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data
        written = {}

        def fake_set(ticker, period, provider, df):
            written[ticker] = (period, df)

        with patch("scanner.data_fetcher._get_cached", side_effect=lambda t, p, pr: cached.get(t)):
            with patch("scanner.data_fetcher._set_cached", side_effect=fake_set):
                with patch.dict("sys.modules", {"yfinance": mock_yf}):
                    result = fetch_batch_yfinance(["RELIANCE", "TCS"], period="1y")

        assert set(result) == {"RELIANCE", "TCS"}
        # Only the uncached ticker was requested from yfinance
        call_args = mock_yf.download.call_args
        assert call_args[0][0] == ["TCS.NS"]
        # The fresh result was cached as daily bars under the download period
        assert written["TCS"][0] == "1y"
        assert list(written["TCS"][1].columns) == ["open", "high", "low", "close", "volume"]
        assert len(written["TCS"][1]) == 200

    def test_short_cached_frame_is_redownloaded(self):
        """A cached frame too short for scoring must not be trusted as fresh."""
        cached = {"RELIANCE": _make_daily_ohlcv(30)}  # < 50 bars
        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=200)

        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        with patch("scanner.data_fetcher._get_cached", side_effect=lambda t, p, pr: cached.get(t)):
            with patch("scanner.data_fetcher._set_cached"):
                with patch.dict("sys.modules", {"yfinance": mock_yf}):
                    result = fetch_batch_yfinance(["RELIANCE"], period="1y")

        assert "RELIANCE" in result
        assert len(result["RELIANCE"]) == 200  # from the fresh download, not the stale cache

    def test_weekly_scan_replays_cached_daily_bars(self):
        """Cached daily bars are resampled for non-daily timeframes."""
        daily = _make_daily_ohlcv(500)

        def fake_get(ticker, period, provider):
            assert period == "5y"  # download period extended for weekly resampling
            return daily

        mock_yf = MagicMock()
        mock_yf.download.side_effect = AssertionError("yfinance must not be called")

        with patch("scanner.data_fetcher._get_cached", side_effect=fake_get):
            with patch("scanner.data_fetcher._set_cached"):
                with patch.dict("sys.modules", {"yfinance": mock_yf}):
                    result = fetch_batch_yfinance(["RELIANCE"], period="2y", timeframe="W")

        assert "RELIANCE" in result
        df = result["RELIANCE"]
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert 50 <= len(df) < 500  # weekly resample of the cached daily bars

    def test_fresh_download_is_cached_for_the_next_scan(self):
        """A cold chunk writes every accepted frame to the disk cache."""
        mock_data = _make_yf_download_result(["RELIANCE.NS", "TCS.NS"], n=200)

        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data
        written = {}

        def fake_set(ticker, period, provider, df):
            written[ticker] = df

        with patch("scanner.data_fetcher._get_cached", return_value=None):
            with patch("scanner.data_fetcher._set_cached", side_effect=fake_set):
                with patch.dict("sys.modules", {"yfinance": mock_yf}):
                    result = fetch_batch_yfinance(["RELIANCE", "TCS"], period="1y")

        assert set(result) == {"RELIANCE", "TCS"}
        assert set(written) == {"RELIANCE", "TCS"}
        for df in written.values():
            assert list(df.columns) == ["open", "high", "low", "close", "volume"]
            assert len(df) == 200


# ══════════════════════════════════════════════════════════════════════════════
# fetch_stock_data / fetch_index_data / fetch_fundamentals
# ══════════════════════════════════════════════════════════════════════════════


class TestFetchStockData:
    def test_returns_dataframe_on_success(self):
        """fetch_stock_data should return a DataFrame from the provider."""
        df = _make_daily_ohlcv(200)
        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = df
        mock_provider.fetch_fundamentals.return_value = {"pe_ratio": 20.0}

        with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
            result = fetch_stock_data("RELIANCE", period="1y")

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 50

    def test_returns_none_on_failure(self):
        """fetch_stock_data should return None when all retries fail."""
        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = None

        with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
            result = fetch_stock_data("INVALID", period="1y")

        assert result is None

    def test_attaches_fundamentals(self):
        """Fundamentals should be attached as _fundamentals attribute."""
        df = _make_daily_ohlcv(200)
        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = df
        mock_provider.fetch_fundamentals.return_value = {"pe_ratio": 15.0, "roe": 22.0}

        with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
            result = fetch_stock_data("RELIANCE", period="1y")

        assert hasattr(result, "_fundamentals")
        assert result._fundamentals["pe_ratio"] == 15.0

    def test_retries_on_exception(self):
        """Should retry on exception and eventually fail."""
        mock_provider = MagicMock()
        mock_provider.fetch_stock.side_effect = [Exception("network"), Exception("network")]

        with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
            with patch("scanner.data_fetcher.time.sleep"):  # skip sleep in tests
                result = fetch_stock_data("RELIANCE", period="1y", retries=2)

        assert result is None
        assert mock_provider.fetch_stock.call_count == 2


class TestFetchIndexData:
    def test_returns_dataframe(self):
        df = _make_daily_ohlcv(200)
        mock_provider = MagicMock()
        mock_provider.fetch_index.return_value = df

        with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
            result = fetch_index_data("^NSEI", period="1y")

        assert result is not None
        assert len(result) == 200

    def test_returns_none_on_failure(self):
        mock_provider = MagicMock()
        mock_provider.fetch_index.return_value = None

        with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
            result = fetch_index_data("^NSEI", period="1y")

        assert result is None


class TestFetchFundamentals:
    def test_returns_dict(self):
        mock_provider = MagicMock()
        mock_provider.fetch_fundamentals.return_value = {"pe_ratio": 20.0}

        with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
            result = fetch_fundamentals("RELIANCE")

        assert result == {"pe_ratio": 20.0}

    def test_returns_none_on_failure(self):
        mock_provider = MagicMock()
        mock_provider.fetch_fundamentals.return_value = None

        with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
            result = fetch_fundamentals("INVALID")

        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# Batch fallback (jugaad-data/nselib) for tickers yfinance missed
# ══════════════════════════════════════════════════════════════════════════════


class TestFetchBatchFallback:
    def test_recovers_missing_ticker(self):
        """Ticker missed by yfinance should be recovered via provider fallback."""
        # yfinance MultiIndex result that only contains RELIANCE -> TCS missing
        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=200, force_multi=True)
        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = _make_daily_ohlcv(200)

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
                result = fetch_batch_yfinance(["RELIANCE", "TCS"], period="1y")

        assert "RELIANCE" in result  # from the yfinance chunk
        assert "TCS" in result       # recovered via fallback
        assert list(result["TCS"].columns) == ["open", "high", "low", "close", "volume"]
        # Fallback must skip yfinance — it just failed at batch level
        mock_provider.fetch_stock.assert_called_once_with(
            "TCS", "1y", skip=("yfinance",),
            provider_timeout=FALLBACK_PROVIDER_TIMEOUT,
        )

    def test_no_fallback_when_batch_complete(self):
        """When yfinance returns every ticker, the fallback should not run."""
        mock_data = _make_yf_download_result(["RELIANCE.NS", "TCS.NS"], n=200)
        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        mock_provider = MagicMock()

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
                result = fetch_batch_yfinance(["RELIANCE", "TCS"], period="1y")

        assert len(result) == 2
        mock_provider.fetch_stock.assert_not_called()

    def test_missing_ticker_stays_missing_when_fallback_fails(self):
        """Ticker remains absent if the fallback providers also return nothing."""
        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=200, force_multi=True)
        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = None

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
                result = fetch_batch_yfinance(["RELIANCE", "TCS"], period="1y")

        assert "RELIANCE" in result
        assert "TCS" not in result

    def test_cancel_event_stops_fallback(self):
        """Cancel fired mid-pass aborts the fallback but keeps fetched data."""
        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=200, force_multi=True)
        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        mock_provider = MagicMock()

        def slow_fetch(*a, **kw):
            time.sleep(2.0)
            return _make_daily_ohlcv(200)

        mock_provider.fetch_stock.side_effect = slow_fetch
        cancel_event = threading.Event()

        def _fire():
            time.sleep(0.5)
            cancel_event.set()

        threading.Thread(target=_fire, daemon=True).start()

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
                result = fetch_batch_yfinance(
                    ["RELIANCE", "TCS"], period="1y", cancel_event=cancel_event
                )

        assert "RELIANCE" in result  # yfinance phase completed
        assert "TCS" not in result   # slow fallback fetch aborted by cancel

    def test_weekly_fallback_resamples(self):
        """Fallback frames should be resampled like batch frames (W timeframe)."""
        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=500, force_multi=True)
        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = _make_daily_ohlcv(500)

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
                result = fetch_batch_yfinance(["RELIANCE", "TCS"], period="1y", timeframe="W")

        assert "RELIANCE" in result
        assert "TCS" in result
        # Weekly period 1y extends to 2y for the per-ticker fallback
        mock_provider.fetch_stock.assert_called_once_with(
            "TCS", "2y", skip=("yfinance",),
            provider_timeout=FALLBACK_PROVIDER_TIMEOUT,
        )
        assert len(result["TCS"]) < 500  # resampled to weekly bars
        assert len(result["TCS"]) >= 50

    def test_fallback_skips_non_nse_symbols(self):
        """BSE-only symbols should not be sent to the NSE-only fallback."""
        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=200, force_multi=True)
        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = _make_daily_ohlcv(200)

        tickers = ["RELIANCE", "TCS", "BSEONLYXYZ"]
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
                with patch("scanner.data_fetcher.FALLBACK_FILTER_MIN_MISSING", 1):
                    with patch(
                        "scanner.data_fetcher._nse_membership_set",
                        return_value={"RELIANCE", "TCS"},
                    ):
                        result = fetch_batch_yfinance(tickers, period="1y")

        assert "RELIANCE" in result       # from the yfinance chunk
        assert "TCS" in result            # NSE member → attempted & recovered
        assert "BSEONLYXYZ" not in result  # non-NSE → never attempted
        # Only the NSE-member miss reached the fallback providers
        mock_provider.fetch_stock.assert_called_once_with(
            "TCS", "1y", skip=("yfinance",),
            provider_timeout=FALLBACK_PROVIDER_TIMEOUT,
        )

    def test_fallback_attempts_all_when_nse_list_unavailable(self):
        """If the NSE membership list can't be resolved, attempt every miss."""
        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=200, force_multi=True)
        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = _make_daily_ohlcv(200)

        tickers = ["RELIANCE", "TCS", "BSEONLYXYZ"]
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
                with patch("scanner.data_fetcher.FALLBACK_FILTER_MIN_MISSING", 1):
                    with patch("scanner.data_fetcher._nse_membership_set", return_value=None):
                        result = fetch_batch_yfinance(tickers, period="1y")

        assert "RELIANCE" in result
        assert "TCS" in result          # unfiltered → both misses attempted
        assert "BSEONLYXYZ" in result
        assert mock_provider.fetch_stock.call_count == 2

    def test_small_missing_sets_skip_nse_list_consult(self):
        """Below the threshold, no NSE list fetch should happen."""
        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=200, force_multi=True)
        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = _make_daily_ohlcv(200)

        # 1 missing ticker < FALLBACK_FILTER_MIN_MISSING (default 25)
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
                with patch("scanner.data_fetcher._nse_membership_set") as mock_membership:
                    result = fetch_batch_yfinance(["RELIANCE", "TCS"], period="1y")

        assert "TCS" in result
        mock_membership.assert_not_called()


class TestNegativeCache:
    """Dead symbols are remembered on disk so later scans skip re-attempts."""

    def test_dead_symbol_skipped_on_next_scan(self):
        """A symbol that failed the whole fallback chain once is not re-attempted."""
        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=200, force_multi=True)
        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        # Run 1: TCS + ZZZ miss yfinance AND the fallback providers -> dead
        dead_provider = MagicMock()
        dead_provider.fetch_stock.return_value = None
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=dead_provider):
                result1 = fetch_batch_yfinance(["RELIANCE", "TCS", "ZZZ"], period="1y")

        assert "TCS" not in result1
        assert "ZZZ" not in result1
        assert data_fetcher._negative_cache_contains("TCS")
        assert data_fetcher._negative_cache_contains("ZZZ")

        # Run 2: providers would succeed now, but the negative cache must
        # prevent TCS/ZZZ from being attempted at all.
        alive_provider = MagicMock()
        alive_provider.fetch_stock.return_value = _make_daily_ohlcv(200)
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=alive_provider):
                result2 = fetch_batch_yfinance(["RELIANCE", "TCS", "ZZZ"], period="1y")

        assert "RELIANCE" in result2
        assert "TCS" not in result2
        assert "ZZZ" not in result2
        alive_provider.fetch_stock.assert_not_called()

    def test_expired_entry_is_reattempted_and_cleared_on_success(self):
        """After the TTL window, a marked symbol is tried again; success clears it."""
        data_fetcher._negative_cache_update(marks=["STALE"])
        # Age the entry past the 24h TTL
        data_fetcher._negative_cache["STALE"] = time.time() - 25 * 3600

        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=200, force_multi=True)
        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = _make_daily_ohlcv(200)

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
                result = fetch_batch_yfinance(["RELIANCE", "STALE"], period="1y")

        assert "STALE" in result  # re-attempted after TTL expiry
        # Recovered symbol is no longer considered dead
        assert not data_fetcher._negative_cache_contains("STALE")

    def test_symbol_with_data_but_few_bars_is_not_marked_dead(self):
        """Short-history names (recent IPOs) are misses, not dead symbols."""
        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=200, force_multi=True)
        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = _make_daily_ohlcv(30)  # < 50 bars

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
                result = fetch_batch_yfinance(["RELIANCE", "NEWIPO"], period="1y")

        assert "NEWIPO" not in result  # still a miss for this scan...
        # ...but NOT cached as dead — next scan attempts it again
        assert not data_fetcher._negative_cache_contains("NEWIPO")

    def test_mark_survives_new_process_load_from_disk(self):
        """Marks persist to disk and are read back on a fresh (re)load."""
        data_fetcher._negative_cache_update(marks=["GONER"])
        # Simulate a new process: drop the in-memory state, force a disk load
        data_fetcher._negative_cache = None

        assert data_fetcher._negative_cache_contains("GONER")

        # An expired entry on disk is dropped on load
        import json
        with open(data_fetcher._NEGATIVE_CACHE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["ANCIENT"] = time.time() - 48 * 3600
        with open(data_fetcher._NEGATIVE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        data_fetcher._negative_cache = None

        assert data_fetcher._negative_cache_contains("GONER")
        assert not data_fetcher._negative_cache_contains("ANCIENT")

    def test_ttl_override_changes_expiry(self):
        """set_negative_cache_ttl_hours() shrinks/extends the expiry window."""
        data_fetcher._negative_cache_update(marks=["S1"])
        data_fetcher._negative_cache["S1"] = time.time() - 2 * 3600  # 2h old
        assert data_fetcher._negative_cache_contains("S1")  # default 24h TTL

        data_fetcher.set_negative_cache_ttl_hours(1)  # shrink to 1h
        assert not data_fetcher._negative_cache_contains("S1")  # now expired
        assert data_fetcher.negative_cache_ttl_hours() == 1.0

        data_fetcher.set_negative_cache_ttl_hours(None)  # back to default
        assert data_fetcher.negative_cache_ttl_hours() == data_fetcher.NEGATIVE_CACHE_TTL_HOURS

    def test_skip_count_reset_and_accumulate(self):
        """The per-scan skip counter resets and only accumulates positive skips."""
        data_fetcher.reset_negative_cache_skip_count()
        data_fetcher._record_negative_cache_skips(0)  # no-op
        assert data_fetcher.negative_cache_skip_count() == 0
        data_fetcher._record_negative_cache_skips(12)
        assert data_fetcher.negative_cache_skip_count() == 12
        data_fetcher.reset_negative_cache_skip_count()
        assert data_fetcher.negative_cache_skip_count() == 0


class TestEnrichmentCache:
    """Phase-2 provider results are cached so repeat scans skip the 5-provider fetch."""

    def test_put_then_get_round_trip(self):
        """A cached entry returns provider keys and fundamentals as stored."""
        providers = {"_sentiment_score": 0.9, "_insider_score": 5}
        data_fetcher._enrichment_cache_put("TCS", providers, {"pe_ratio": 20.0})

        entry = data_fetcher._enrichment_cache_get("TCS")
        assert entry is not None
        assert entry["providers"] == providers
        assert entry["fundamentals"] == {"pe_ratio": 20.0}

    def test_known_none_fundamentals_round_trip(self):
        """A ticker with no fundamentals is cached as None, not re-fetched."""
        data_fetcher._enrichment_cache_put("SMALL", {"_social_score": 0.2}, None)
        entry = data_fetcher._enrichment_cache_get("SMALL")
        assert entry is not None
        assert entry["fundamentals"] is None

    def test_empty_providers_are_not_cached(self):
        """All-providers-failed (transient outage) must not freeze into the cache."""
        data_fetcher._enrichment_cache_put("TCS", {}, {"pe_ratio": 20.0})
        assert data_fetcher._enrichment_cache_get("TCS") is None
        assert data_fetcher.enrichment_cache_size() == 0

    def test_expired_entry_is_evicted(self):
        """Past the TTL window the cache is re-populated from the providers."""
        data_fetcher._enrichment_cache_put("STALE", {"_social_score": 0.5}, None)
        data_fetcher._enrichment_cache["STALE"]["ts"] = time.time() - 25 * 3600

        assert data_fetcher._enrichment_cache_get("STALE") is None
        assert "STALE" not in data_fetcher._enrichment_cache

    def test_entry_survives_new_process_load_from_disk(self):
        """Entries persist to disk and reload on a fresh process."""
        data_fetcher._enrichment_cache_put("RELIANCE", {"_fii_is_buying": True}, None)
        data_fetcher._enrichment_cache = None  # simulate a new process

        entry = data_fetcher._enrichment_cache_get("RELIANCE")
        assert entry is not None
        assert entry["providers"] == {"_fii_is_buying": True}

        # An expired entry on disk is dropped on load
        import json
        with open(data_fetcher._ENRICHMENT_CACHE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["ANCIENT"] = {"ts": time.time() - 48 * 3600, "providers": {}, "fundamentals": None}
        with open(data_fetcher._ENRICHMENT_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        data_fetcher._enrichment_cache = None

        assert data_fetcher._enrichment_cache_get("RELIANCE") is not None
        assert data_fetcher._enrichment_cache_get("ANCIENT") is None

    def test_clear_wipes_memory_and_disk(self):
        """enrichment_cache_clear() empties both the in-memory map and the file."""
        data_fetcher._enrichment_cache_put("TCS", {"_social_score": 0.5}, None)
        assert data_fetcher.enrichment_cache_size() == 1

        data_fetcher.enrichment_cache_clear()
        assert data_fetcher.enrichment_cache_size() == 0
        assert data_fetcher._enrichment_cache_get("TCS") is None

    def test_hit_miss_counts_reset_and_accumulate(self):
        """Per-scan hit/miss counters reset and accumulate from worker threads."""
        data_fetcher.reset_enrichment_cache_counts()
        assert data_fetcher.enrichment_cache_hits() == 0
        assert data_fetcher.enrichment_cache_misses() == 0
        data_fetcher._record_enrichment_cache_hit()
        data_fetcher._record_enrichment_cache_miss()
        data_fetcher._record_enrichment_cache_hit()
        assert data_fetcher.enrichment_cache_hits() == 2
        assert data_fetcher.enrichment_cache_misses() == 1
        data_fetcher.reset_enrichment_cache_counts()
        assert data_fetcher.enrichment_cache_hits() == 0


class TestAbortableBatchDownload:
    """Stop must be able to interrupt yfinance chunks, not just the fallback."""

    def test_cancel_preset_skips_yfinance_entirely(self):
        """With cancel already set, no yfinance download should be scheduled."""
        mock_yf = MagicMock()
        cancel_event = threading.Event()
        cancel_event.set()
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=MagicMock()):
                result = fetch_batch_yfinance(
                    ["AAA", "BBB"], period="1y", cancel_event=cancel_event
                )
        assert result == {}
        mock_yf.download.assert_not_called()

    def test_cancel_mid_batch_returns_without_waiting_for_chunk(self):
        """Stop returns promptly even while a chunk download is still running."""
        def slow_download(*a, **kw):
            time.sleep(8)
            return _make_yf_download_result(["RELIANCE.NS"], n=200)

        mock_yf = MagicMock()
        mock_yf.download.side_effect = slow_download
        cancel_event = threading.Event()

        def _fire():
            time.sleep(0.5)
            cancel_event.set()

        threading.Thread(target=_fire, daemon=True).start()
        t0 = time.time()
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=MagicMock()):
                result = fetch_batch_yfinance(
                    ["AAA", "BBB"], period="1y", cancel_event=cancel_event
                )
        elapsed = time.time() - t0
        assert elapsed < 5  # returned while the 8s download was still running
        assert result == {}

    def test_cancel_between_batches_stops_scheduling(self):
        """Cancel fired between outer batches must not schedule further ones."""
        # 9 chunks of 200 -> two outer batches (8 + 1)
        tickers = [f"T{i}" for i in range((MAX_PARALLEL_CHUNKS + 1) * CHUNK)]
        calls = {"n": 0}
        cancel_event = threading.Event()

        def counting_download(*a, **kw):
            calls["n"] += 1
            return _make_yf_download_result(["T0.NS"], n=200, force_multi=True)

        mock_yf = MagicMock()
        mock_yf.download.side_effect = counting_download

        real_sleep = time.sleep
        fired = {"done": False}

        def sleep_then_cancel(secs):
            real_sleep(secs)
            if not fired["done"]:
                fired["done"] = True
                cancel_event.set()  # cancel lands between batch 1 and batch 2

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=MagicMock()):
                with patch("scanner.data_fetcher.time.sleep", side_effect=sleep_then_cancel):
                    result = fetch_batch_yfinance(
                        tickers, period="1y", cancel_event=cancel_event
                    )

        assert calls["n"] == MAX_PARALLEL_CHUNKS  # batch 1 only — batch 2 never scheduled
        assert "T0" in result


class TestScanStartStalePrune:
    """fetch_batch_yfinance (scan start) sweeps unreachable previous-day entries."""

    @staticmethod
    def _write_stale_entry(d, name):
        import json as _json
        import os as _os
        from datetime import datetime as _dt
        from datetime import timedelta as _td
        stale = (_dt.now() - _td(days=1)).isoformat()
        _os.makedirs(str(d), exist_ok=True)
        _make_daily_ohlcv(60).to_pickle(_os.path.join(str(d), name + ".pkl"))
        with open(_os.path.join(str(d), name + ".meta"), "w") as f:
            _json.dump({"timestamp": stale, "rows": 60}, f)

    def test_batch_fetch_prunes_stale_entries(self, tmp_path, monkeypatch):
        """A stale-day pair on disk is gone after the fetch pass runs."""
        import os as _os

        from scanner import data_providers
        monkeypatch.setattr("scanner.data_providers._last_prune_ts", 0.0)
        cache_dir = data_providers.CACHE_DIR
        self._write_stale_entry(cache_dir, "stale1")
        self._write_stale_entry(cache_dir, "stale2")
        stale_pkl = _os.path.join(cache_dir, "stale1.pkl")
        assert _os.path.exists(stale_pkl)

        mock_yf = MagicMock()
        mock_yf.download.return_value = _make_yf_download_result(["RELIANCE.NS"], n=200)
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=MagicMock()):
                result = fetch_batch_yfinance(["RELIANCE"], period="1y")

        assert "RELIANCE" in result
        assert not _os.path.exists(stale_pkl)         # stale pair pruned...
        assert not _os.path.exists(stale_pkl[:-4] + ".meta")
        fresh = [f for f in _os.listdir(cache_dir) if f.endswith(".pkl")]
        assert len(fresh) == 1                        # ...today's new entry kept


# ══════════════════════════════════════════════════════════════════════════════
# _normalize_daily_index — one trade-date calendar across stamp flavors
# ══════════════════════════════════════════════════════════════════════════════
# Regression: two cached FNO tickers (GSPL, TATAMOTORS) carried 18:30 UTC
# previous-day stamps while the other 108 shared local-midnight trade dates.
# Same NSE trade days, different hashes -> the engine's cross-ticker date
# union doubled (~1489 dates instead of ~745) and every FNO backtest ran on
# a fake calendar. Normalization must make both flavors hash identically.


def _ohlcv_like(days, seed=0):
    """OHLCV frame indexed by the given timestamps (values are irrelevant)."""
    rng = np.random.RandomState(seed)
    close = 500 + np.cumsum(rng.randn(len(days)) * 2)
    return pd.DataFrame({
        "open": close + rng.randn(len(days)),
        "high": close + np.abs(rng.randn(len(days))) * 2,
        "low": close - np.abs(rng.randn(len(days))) * 2,
        "close": close,
        "volume": (rng.rand(len(days)) * 1e6 + 5e5).astype(int),
    }, index=pd.DatetimeIndex(days))


def _trade_days(n=200, start="2024-01-01"):
    return pd.bdate_range(start, periods=n)


def _utc_close_stamps(trade_days, tz_aware):
    """The same trade days encoded as 18:30 UTC stamps on the previous day."""
    prev = trade_days - pd.Timedelta(days=1) + pd.Timedelta(hours=18, minutes=30)
    return prev.tz_localize("UTC") if tz_aware else prev


class TestNormalizeDailyIndex:
    def test_midnight_and_utc_close_flavors_converge(self):
        """Local-midnight and tz-aware UTC-close frames map to the same calendar."""
        trade_days = _trade_days(200)

        # Flavor A: local-midnight naive stamps (yfinance .NS path)
        local = _ohlcv_like(trade_days)
        # Flavor B: same trade days as 18:30 UTC stamps on the previous day
        utc_close = _ohlcv_like(_utc_close_stamps(trade_days, tz_aware=True))

        norm_local = data_fetcher._normalize_daily_index(local)
        norm_utc = data_fetcher._normalize_daily_index(utc_close)

        # Both collapse onto the exact same trade-date calendar...
        assert norm_local.index.equals(norm_utc.index)
        # ...which is the original trade days: tz-naive midnights, ascending.
        assert list(norm_local.index) == list(trade_days)
        assert norm_local.index.tz is None
        assert norm_local.index.is_monotonic_increasing
        assert list(norm_utc.index) == list(trade_days)

    def test_naive_utc_close_flavor_matches_aware_flavor(self):
        """Even tz-naive 18:30 stamps (tz dropped in the cache layer) converge."""
        trade_days = _trade_days(120)
        aware = _ohlcv_like(_utc_close_stamps(trade_days, tz_aware=True))
        naive = _ohlcv_like(_utc_close_stamps(trade_days, tz_aware=False))

        norm_aware = data_fetcher._normalize_daily_index(aware)
        norm_naive = data_fetcher._normalize_daily_index(naive)
        assert norm_aware.index.equals(norm_naive.index)
        assert list(norm_aware.index) == list(trade_days)

    def test_normalized_union_does_not_double(self):
        """The 2x union: disjoint raw calendars collapse to one after normalize."""
        trade_days = _trade_days(200)
        a = _ohlcv_like(trade_days)  # midnight flavor
        b = _ohlcv_like(_utc_close_stamps(trade_days, tz_aware=False))  # prev-18:30

        # Before normalization the two calendars are disjoint -> ~2x inflation.
        assert len(a.index.union(b.index)) == 400

        # After normalization both share one calendar -> no inflation.
        na = data_fetcher._normalize_daily_index(a)
        nb = data_fetcher._normalize_daily_index(b)
        assert na.index.equals(nb.index)
        assert len(na.index.union(nb.index)) == len(trade_days)

    def test_same_day_duplicates_keep_last_bar(self):
        """Dual-provider rows for one day dedupe to a single midnight bar."""
        trade_days = _trade_days(50)
        df = _ohlcv_like(trade_days)
        dup = pd.concat([df, df.iloc[24:34]])  # days 24-33 appear twice

        result = data_fetcher._normalize_daily_index(dup)
        assert result.index.is_unique
        assert len(result) == len(trade_days)
        assert result.index.equals(pd.DatetimeIndex(trade_days))

    def test_descending_input_is_sorted(self):
        """Provider feeds can arrive newest-first; normalize must sort ascending."""
        trade_days = _trade_days(100)
        df = _ohlcv_like(trade_days).iloc[::-1]

        result = data_fetcher._normalize_daily_index(df)
        assert result.index.is_monotonic_increasing
        assert len(result) == len(trade_days)

    def test_none_and_empty_pass_through(self):
        assert data_fetcher._normalize_daily_index(None) is None
        empty = data_fetcher._normalize_daily_index(pd.DataFrame())
        assert empty is not None and empty.empty
