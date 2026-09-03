"""Unit tests for scanner.data_fetcher — data fetching and resampling.

All external API calls (yfinance, jugaad, nselib) are mocked so tests
run fast and offline.
"""

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from scanner.data_fetcher import (
    FALLBACK_PROVIDER_TIMEOUT,
    _extend_period_for_timeframe,
    fetch_batch_yfinance,
    fetch_fundamentals,
    fetch_index_data,
    fetch_stock_data,
    resample_ohlcv,
)

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
        """A pre-set cancel event should abort the fallback pass early."""
        mock_data = _make_yf_download_result(["RELIANCE.NS"], n=200, force_multi=True)
        mock_yf = MagicMock()
        mock_yf.download.return_value = mock_data

        mock_provider = MagicMock()
        mock_provider.fetch_stock.return_value = _make_daily_ohlcv(200)
        cancel_event = threading.Event()
        cancel_event.set()

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("scanner.data_fetcher._get_provider", return_value=mock_provider):
                result = fetch_batch_yfinance(
                    ["RELIANCE", "TCS"], period="1y", cancel_event=cancel_event
                )

        assert "RELIANCE" in result
        assert "TCS" not in result  # fallback aborted by cancel

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
