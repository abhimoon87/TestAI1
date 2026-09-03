"""Unit tests for scanner.data_providers — DataProvider class and fallback logic.

All external API calls (yfinance, jugaad, nselib, finnhub, alpha_vantage)
are mocked so tests run fast and offline.
"""

import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from scanner.data_providers import (
    DataProvider,
    _cache_key,
    _fetch_fundamentals_yfinance,
    _fetch_yfinance,
    _fetch_yfinance_index,
    _get_cached,
    _set_cached,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_ohlcv(n=200):
    """Create a realistic OHLCV DataFrame."""
    dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.RandomState(42)
    close = 500 + np.cumsum(rng.randn(n) * 2)
    return pd.DataFrame({
        "open": close + rng.randn(n),
        "high": close + np.abs(rng.randn(n)) * 2,
        "low": close - np.abs(rng.randn(n)) * 2,
        "close": close,
        "volume": (rng.rand(n) * 1e6 + 5e5).astype(int),
    }, index=dates)


def _make_yf_history(n=200):
    """Simulate yfinance.Ticker.history() return value."""
    df = _make_ohlcv(n)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Cache functions
# ══════════════════════════════════════════════════════════════════════════════


class TestCacheKey:
    def test_deterministic(self):
        """Same inputs should produce same key."""
        k1 = _cache_key("RELIANCE", "1y", "cache")
        k2 = _cache_key("RELIANCE", "1y", "cache")
        assert k1 == k2

    def test_different_tickers(self):
        """Different tickers should produce different keys."""
        k1 = _cache_key("RELIANCE", "1y", "cache")
        k2 = _cache_key("TCS", "1y", "cache")
        assert k1 != k2

    def test_different_periods(self):
        k1 = _cache_key("RELIANCE", "1y", "cache")
        k2 = _cache_key("RELIANCE", "2y", "cache")
        assert k1 != k2


class TestCacheRoundTrip:
    def test_set_then_get(self, tmp_path):
        """Writing to cache and reading back should return the same data."""
        with patch("scanner.data_providers.CACHE_DIR", str(tmp_path)):
            df = _make_ohlcv(50)
            _set_cached("TEST", "1y", "test_cache", df)
            result = _get_cached("TEST", "1y", "test_cache")

        assert result is not None
        pd.testing.assert_frame_equal(result, df)

    def test_expired_cache_returns_none(self, tmp_path):
        """Cache older than TTL should return None."""
        with patch("scanner.data_providers.CACHE_DIR", str(tmp_path)):
            df = _make_ohlcv(50)
            _set_cached("TEST", "1y", "test_cache", df)

            # Manually set timestamp to 5 hours ago
            key = _cache_key("TEST", "1y", "test_cache")
            meta_file = tmp_path / f"{key}.meta"
            with open(meta_file, "w") as f:
                json.dump({
                    "timestamp": (datetime.now() - timedelta(hours=5)).isoformat(),
                    "rows": 50,
                }, f)

            result = _get_cached("TEST", "1y", "test_cache")

        assert result is None

    def test_missing_cache_returns_none(self, tmp_path):
        """Non-existent cache should return None."""
        with patch("scanner.data_providers.CACHE_DIR", str(tmp_path)):
            result = _get_cached("NONEXISTENT", "1y", "cache")
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# yfinance fetch functions (mocked)
# ══════════════════════════════════════════════════════════════════════════════


class TestFetchYfinance:
    def test_returns_normalized_dataframe(self):
        """_fetch_yfinance should return lowercase-named OHLCV DataFrame."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_yf_history(200)
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = _fetch_yfinance("RELIANCE", "1y")

        assert result is not None
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        assert len(result) >= 50

    def test_returns_none_on_empty(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = _fetch_yfinance("INVALID", "1y")

        assert result is None

    def test_adds_ns_suffix(self):
        """Ticker should have .NS suffix added."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_yf_history(200)
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            _fetch_yfinance("RELIANCE", "1y")

        mock_yf.Ticker.assert_called_once_with("RELIANCE.NS")

    def test_import_error_returns_none(self):
        with patch.dict("sys.modules", {"yfinance": None}):
            result = _fetch_yfinance("RELIANCE", "1y")
        assert result is None


class TestFetchYfinanceIndex:
    def test_returns_dataframe(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_yf_history(200)
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = _fetch_yfinance_index("^NSEI", "1y")

        assert result is not None
        assert len(result) >= 50

    def test_no_ns_suffix_for_index(self):
        """Index tickers should NOT get .NS suffix."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_yf_history(200)
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            _fetch_yfinance_index("^NSEI", "1y")

        mock_yf.Ticker.assert_called_once_with("^NSEI")


class TestFetchFundamentalsYfinance:
    def test_returns_fundamentals_dict(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "trailingPE": 20.5,
            "earningsGrowth": 0.15,
            "revenueGrowth": 0.12,
            "returnOnEquity": 0.22,
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = _fetch_fundamentals_yfinance("RELIANCE")

        assert result is not None
        assert result["pe_ratio"] == 20.5
        assert result["eps_growth"] == 15.0  # 0.15 * 100
        assert result["rev_growth"] == 12.0  # 0.12 * 100
        assert result["roe"] == 22.0  # 0.22 * 100

    def test_returns_none_on_empty_info(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = _fetch_fundamentals_yfinance("INVALID")

        assert result is None

    def test_handles_none_values(self):
        """Fields that are None should remain None."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {"trailingPE": 15.0}
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = _fetch_fundamentals_yfinance("RELIANCE")

        assert result["pe_ratio"] == 15.0
        assert result["eps_growth"] is None
        assert result["rev_growth"] is None
        assert result["roe"] is None


# ══════════════════════════════════════════════════════════════════════════════
# DataProvider class
# ══════════════════════════════════════════════════════════════════════════════


class TestDataProvider:
    def test_fetch_stock_uses_fallback(self):
        """When jugaad fails, should fall back to yfinance."""
        provider = DataProvider(use_cache=False)

        mock_jugaad = MagicMock(return_value=None)
        mock_yf_history = _make_yf_history(200)
        mock_yf_ticker = MagicMock()
        mock_yf_ticker.history.return_value = mock_yf_history
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_yf_ticker

        with patch("scanner.data_providers._fetch_jugaad", mock_jugaad):
            with patch.dict("sys.modules", {"yfinance": mock_yf}):
                result = provider.fetch_stock("RELIANCE", "1y")

        assert result is not None
        assert provider.last_provider == "yfinance"
        mock_jugaad.assert_called_once()

    def test_fetch_stock_returns_none_all_fail(self):
        """When all providers fail, should return None."""
        provider = DataProvider(use_cache=False)

        with patch("scanner.data_providers._fetch_jugaad", return_value=None):
            with patch("scanner.data_providers._fetch_yfinance", return_value=None):
                with patch("scanner.data_providers._fetch_nselib", return_value=None):
                    result = provider.fetch_stock("INVALID", "1y")

        assert result is None
        assert provider.last_provider is None
        assert provider.last_error == "All providers failed"

    def test_fetch_stock_uses_cache(self):
        """Cache hit should bypass provider chain."""
        provider = DataProvider(use_cache=True)
        df = _make_ohlcv(200)

        with patch("scanner.data_providers._get_cached", return_value=df):
            result = provider.fetch_stock("RELIANCE", "1y")

        pd.testing.assert_frame_equal(result, df)
        assert provider.last_provider == "cache"

    def test_fetch_stock_caches_result(self):
        """Successful fetch should write to cache."""
        provider = DataProvider(use_cache=True)
        _make_ohlcv(200)

        mock_yf_history = _make_yf_history(200)
        mock_yf_ticker = MagicMock()
        mock_yf_ticker.history.return_value = mock_yf_history
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_yf_ticker

        with patch("scanner.data_providers._get_cached", return_value=None):
            with patch("scanner.data_providers._set_cached") as mock_set:
                with patch("scanner.data_providers._fetch_jugaad", return_value=None):
                    with patch.dict("sys.modules", {"yfinance": mock_yf}):
                        provider.fetch_stock("RELIANCE", "1y")

        mock_set.assert_called_once()

    def test_fetch_index_fallback(self):
        """Index fetch should fall back through providers."""
        provider = DataProvider(use_cache=False)

        mock_yf_history = _make_yf_history(200)
        mock_yf_ticker = MagicMock()
        mock_yf_ticker.history.return_value = mock_yf_history
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_yf_ticker

        with patch("scanner.data_providers._fetch_jugaad_index", return_value=None):
            with patch.dict("sys.modules", {"yfinance": mock_yf}):
                result = provider.fetch_index("^NSEI", "1y")

        assert result is not None
        assert provider.last_provider == "yfinance"

    def test_fetch_fundamentals_fallback(self):
        """Fundamentals fetch should fall back through providers."""
        provider = DataProvider(use_cache=False)

        mock_yf_ticker = MagicMock()
        mock_yf_ticker.info = {"trailingPE": 20.0, "returnOnEquity": 0.22}
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_yf_ticker

        with patch("scanner.data_providers._fetch_fundamentals_finnhub", return_value=None):
            with patch("scanner.data_providers._fetch_fundamentals_alpha_vantage", return_value=None):
                with patch.dict("sys.modules", {"yfinance": mock_yf}):
                    result = provider.fetch_fundamentals("RELIANCE")

        assert result is not None
        assert result["pe_ratio"] == 20.0
        assert provider.last_provider == "yfinance"

    def test_fetch_fundamentals_all_fail(self):
        """When all fundamental providers fail, return None."""
        provider = DataProvider(use_cache=False)

        with patch("scanner.data_providers._fetch_fundamentals_finnhub", return_value=None):
            with patch("scanner.data_providers._fetch_fundamentals_alpha_vantage", return_value=None):
                with patch("scanner.data_providers._fetch_fundamentals_yfinance", return_value=None):
                    with patch("scanner.data_providers._fetch_fundamentals_nselib", return_value=None):
                        result = provider.fetch_fundamentals("INVALID")

        assert result is None

    def test_min_bars_filter(self):
        """Stocks with < 50 bars should be rejected."""
        provider = DataProvider(use_cache=False)

        short_df = _make_ohlcv(30)  # only 30 bars

        mock_yf_ticker = MagicMock()
        mock_yf_ticker.history.return_value = short_df
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_yf_ticker

        with patch("scanner.data_providers._fetch_jugaad", return_value=None):
            with patch.dict("sys.modules", {"yfinance": mock_yf}):
                result = provider.fetch_stock("SHORT", "1y")

        assert result is None

    def test_clear_cache(self, tmp_path):
        """clear_cache should remove the cache directory."""
        with patch("scanner.data_providers.CACHE_DIR", str(tmp_path)):
            # Create some cache files
            (tmp_path / "test.pkl").touch()
            (tmp_path / "test.meta").touch()

            provider = DataProvider(use_cache=True)
            provider.clear_cache()

            assert not (tmp_path / "test.pkl").exists()
            assert tmp_path.exists()  # dir should be recreated

    def test_fetch_stock_provider_timeout_falls_through(self):
        """A provider exceeding provider_timeout is skipped for the next one."""
        provider = DataProvider(use_cache=False)

        slow_jugaad = MagicMock(side_effect=lambda: time.sleep(0.5))
        mock_yf_ticker = MagicMock()
        mock_yf_ticker.history.return_value = _make_yf_history(200)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_yf_ticker

        with patch("scanner.data_providers._fetch_jugaad", slow_jugaad):
            with patch.dict("sys.modules", {"yfinance": mock_yf}):
                result = provider.fetch_stock("RELIANCE", "1y", provider_timeout=0.05)

        assert result is not None
        assert provider.last_provider == "yfinance"
        slow_jugaad.assert_called_once()

    def test_fetch_stock_all_providers_timeout_is_bounded(self):
        """When every provider hangs, the call returns quickly instead of stalling."""
        provider = DataProvider(use_cache=False)

        def _slow():
            time.sleep(0.5)
            return None

        with patch("scanner.data_providers._fetch_jugaad", side_effect=_slow):
            with patch("scanner.data_providers._fetch_yfinance", side_effect=_slow):
                with patch("scanner.data_providers._fetch_nselib", side_effect=_slow):
                    start = time.time()
                    result = provider.fetch_stock("SLOW", "1y", provider_timeout=0.03)
                    elapsed = time.time() - start

        assert result is None
        assert elapsed < 0.3  # bounded by 3 × 0.03s, not ~1.5s of sleeps

    def test_fetch_stock_skips_yfinance(self):
        """skip=('yfinance',) should exclude yfinance from the chain."""
        provider = DataProvider(use_cache=False)

        with patch("scanner.data_providers._fetch_jugaad", return_value=None):
            with patch("scanner.data_providers._fetch_nselib", return_value=None):
                with patch("scanner.data_providers._fetch_yfinance") as mock_yf:
                    result = provider.fetch_stock("RELIANCE", "1y", skip=("yfinance",))

        assert result is None
        mock_yf.assert_not_called()

    def test_fetch_stock_skips_jugaad(self):
        """skip=('jugaad',) should skip jugaad and let yfinance serve data."""
        provider = DataProvider(use_cache=False)

        mock_yf_ticker = MagicMock()
        mock_yf_ticker.history.return_value = _make_yf_history(200)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_yf_ticker

        with patch("scanner.data_providers._fetch_jugaad") as mock_jugaad:
            with patch.dict("sys.modules", {"yfinance": mock_yf}):
                result = provider.fetch_stock("RELIANCE", "1y", skip=("jugaad",))

        assert result is not None
        assert provider.last_provider == "yfinance"
        mock_jugaad.assert_not_called()
