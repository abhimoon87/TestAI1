"""Unit tests for scanner.api.data_providers — DataProvider class and fallback logic.

All external API calls (yfinance, jugaad, nselib, finnhub, alpha_vantage)
are mocked so tests run fast and offline.
"""

import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from scanner.api.data_providers import (
    DataProvider,
    _cache_key,
    _fetch_fundamentals_yfinance,
    _fetch_yfinance,
    _fetch_yfinance_index,
    _get_cached,
    _set_cached,
    cache_health,
    prune_stale_cache,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_ohlcv(n=200):
    """Create a realistic OHLCV DataFrame."""
    dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.RandomState(42)
    close = 500 + np.cumsum(rng.randn(n) * 2)
    return pd.DataFrame(
        {
            "open": close + rng.randn(n),
            "high": close + np.abs(rng.randn(n)) * 2,
            "low": close - np.abs(rng.randn(n)) * 2,
            "close": close,
            "volume": (rng.rand(n) * 1e6 + 5e5).astype(int),
        },
        index=dates,
    )


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
        with patch("scanner.api.data_providers.CACHE_DIR", str(tmp_path)):
            df = _make_ohlcv(50)
            _set_cached("TEST", "1y", "test_cache", df)
            result = _get_cached("TEST", "1y", "test_cache")

        assert result is not None
        pd.testing.assert_frame_equal(result, df)

    def test_expired_cache_returns_none(self, tmp_path):
        """Cache older than TTL should return None."""
        with patch("scanner.api.data_providers.CACHE_DIR", str(tmp_path)):
            df = _make_ohlcv(50)
            _set_cached("TEST", "1y", "test_cache", df)

            # Backdate the db row's expiry to 5 hours ago
            from scanner.shared import db

            key = _cache_key("TEST", "1y", "test_cache")
            db.get_conn().execute(
                "UPDATE price_cache SET expires = ? WHERE cache_key = ?",
                (time.time() - 5 * 3600, key),
            )

            result = _get_cached("TEST", "1y", "test_cache")

        assert result is None

    def test_missing_cache_returns_none(self, tmp_path):
        """Non-existent cache should return None."""
        with patch("scanner.api.data_providers.CACHE_DIR", str(tmp_path)):
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

        with patch("scanner.api.data_providers._fetch_jugaad", mock_jugaad):
            with patch.dict("sys.modules", {"yfinance": mock_yf}):
                result = provider.fetch_stock("RELIANCE", "1y")

        assert result is not None
        assert provider.last_provider == "yfinance"
        mock_jugaad.assert_called_once()

    def test_fetch_stock_returns_none_all_fail(self):
        """When all providers fail, should return None."""
        provider = DataProvider(use_cache=False)

        with patch("scanner.api.data_providers._fetch_jugaad", return_value=None):
            with patch("scanner.api.data_providers._fetch_yfinance", return_value=None):
                with patch(
                    "scanner.api.data_providers._fetch_nselib", return_value=None
                ):
                    result = provider.fetch_stock("INVALID", "1y")

        assert result is None
        assert provider.last_provider is None
        assert provider.last_error == "All providers failed"

    def test_fetch_stock_uses_cache(self):
        """Cache hit should bypass provider chain."""
        provider = DataProvider(use_cache=True)
        df = _make_ohlcv(200)

        with patch("scanner.api.data_providers._get_cached", return_value=df):
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

        with patch("scanner.api.data_providers._get_cached", return_value=None):
            with patch("scanner.api.data_providers._set_cached") as mock_set:
                with patch(
                    "scanner.api.data_providers._fetch_jugaad", return_value=None
                ):
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

        with patch("scanner.api.data_providers._fetch_jugaad_index", return_value=None):
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

        with patch(
            "scanner.api.data_providers._fetch_fundamentals_finnhub", return_value=None
        ):
            with patch(
                "scanner.api.data_providers._fetch_fundamentals_alpha_vantage",
                return_value=None,
            ):
                with patch.dict("sys.modules", {"yfinance": mock_yf}):
                    result = provider.fetch_fundamentals("RELIANCE")

        assert result is not None
        assert result["pe_ratio"] == 20.0
        assert provider.last_provider == "yfinance"

    def test_fetch_fundamentals_all_fail(self):
        """When all fundamental providers fail, return None."""
        provider = DataProvider(use_cache=False)

        with patch(
            "scanner.api.data_providers._fetch_fundamentals_finnhub", return_value=None
        ):
            with patch(
                "scanner.api.data_providers._fetch_fundamentals_alpha_vantage",
                return_value=None,
            ):
                with patch(
                    "scanner.api.data_providers._fetch_fundamentals_yfinance",
                    return_value=None,
                ):
                    with patch(
                        "scanner.api.data_providers._fetch_fundamentals_nselib",
                        return_value=None,
                    ):
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

        with patch("scanner.api.data_providers._fetch_jugaad", return_value=None):
            with patch.dict("sys.modules", {"yfinance": mock_yf}):
                result = provider.fetch_stock("SHORT", "1y")

        assert result is None

    def test_fetch_stock_provider_timeout_falls_through(self):
        """A provider exceeding provider_timeout is skipped for the next one."""
        provider = DataProvider(use_cache=False)

        slow_jugaad = MagicMock(side_effect=lambda: time.sleep(0.5))
        mock_yf_ticker = MagicMock()
        mock_yf_ticker.history.return_value = _make_yf_history(200)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_yf_ticker

        with patch("scanner.api.data_providers._fetch_jugaad", slow_jugaad):
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

        with patch("scanner.api.data_providers._fetch_jugaad", side_effect=_slow):
            with patch("scanner.api.data_providers._fetch_yfinance", side_effect=_slow):
                with patch(
                    "scanner.api.data_providers._fetch_nselib", side_effect=_slow
                ):
                    start = time.time()
                    result = provider.fetch_stock("SLOW", "1y", provider_timeout=0.03)
                    elapsed = time.time() - start

        assert result is None
        assert elapsed < 0.3  # bounded by 3 × 0.03s, not ~1.5s of sleeps

    def test_fetch_stock_skips_yfinance(self):
        """skip=('yfinance',) should exclude yfinance from the chain."""
        provider = DataProvider(use_cache=False)

        with patch("scanner.api.data_providers._fetch_jugaad", return_value=None):
            with patch("scanner.api.data_providers._fetch_nselib", return_value=None):
                with patch("scanner.api.data_providers._fetch_yfinance") as mock_yf:
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

        with patch("scanner.api.data_providers._fetch_jugaad") as mock_jugaad:
            with patch.dict("sys.modules", {"yfinance": mock_yf}):
                result = provider.fetch_stock("RELIANCE", "1y", skip=("jugaad",))

        assert result is not None
        assert provider.last_provider == "yfinance"
        mock_jugaad.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# prune_stale_cache — day-keyed entries from previous days are unreachable
# ══════════════════════════════════════════════════════════════════════════════


class TestPruneStaleCache:
    """Previous-day cache entries (never readable again) are swept."""

    @staticmethod
    def _write_entry(d, name, when):
        import os

        from scanner.tests.conftest import safe_to_parquet

        safe_to_parquet(
            pd.DataFrame({"close": [1.0, 2.0]}),
            os.path.join(str(d), name + ".parquet"),
            index=False,
        )
        with open(os.path.join(str(d), name + ".meta"), "w") as f:
            json.dump({"timestamp": when, "rows": 2}, f)

    def _reset(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scanner.api.data_providers.CACHE_DIR", str(tmp_path))
        monkeypatch.setattr("scanner.api.data_providers._last_prune_ts", 0.0)

    def test_removes_previous_day_entries_only(self, tmp_path, monkeypatch):
        self._reset(tmp_path, monkeypatch)
        fresh = datetime.now().isoformat()
        stale = (datetime.now() - timedelta(days=1)).isoformat()
        self._write_entry(tmp_path, "old1", stale)
        self._write_entry(tmp_path, "old2", stale)
        self._write_entry(tmp_path, "fresh", fresh)

        removed = prune_stale_cache()

        assert removed == 2
        assert not (tmp_path / "old1.parquet").exists()
        assert not (tmp_path / "old1.meta").exists()
        assert not (tmp_path / "old2.parquet").exists()
        assert (tmp_path / "fresh.parquet").exists()
        assert (tmp_path / "fresh.meta").exists()

    def test_rate_limit_skips_second_call_without_force(self, tmp_path, monkeypatch):
        self._reset(tmp_path, monkeypatch)
        self._write_entry(
            tmp_path, "old", (datetime.now() - timedelta(days=1)).isoformat()
        )
        assert prune_stale_cache() == 1

        # A second sweep within the interval is a no-op (rate-limited)...
        self._write_entry(
            tmp_path, "older", (datetime.now() - timedelta(days=2)).isoformat()
        )
        assert prune_stale_cache() == 0
        assert (tmp_path / "older.parquet").exists()

        # ...unless forced.
        assert prune_stale_cache(force=True) == 1
        assert not (tmp_path / "older.parquet").exists()

    def test_corrupt_meta_is_left_alone(self, tmp_path, monkeypatch):
        self._reset(tmp_path, monkeypatch)
        self._write_entry(tmp_path, "odd", "not-a-timestamp")
        with open(tmp_path / "odd.meta", "w") as f:
            f.write("{not valid json")

        assert prune_stale_cache() == 0
        assert (tmp_path / "odd.parquet").exists()  # conservative: untouched
        assert (tmp_path / "odd.meta").exists()

    def test_empty_dir_is_safe(self, tmp_path, monkeypatch):
        self._reset(tmp_path, monkeypatch)
        assert prune_stale_cache() == 0


class TestPriceCacheRowCap:
    """The row cap bounds price_cache growth beyond the TTL sweep.

    TTL alone only removes *expired* rows, so a full-market scan's ~2.4k
    parquet rows (~47 KB each) kept every overlapping 4h generation alive and
    the file only ever grew. The cap keeps the newest N by expiry.
    """

    @staticmethod
    def _seed(count, base_expiry, prefix="k"):
        from scanner.shared import db

        conn = db.get_conn()
        for i in range(count):
            conn.execute(
                "INSERT INTO price_cache (cache_key, payload, expires)"
                " VALUES (?, ?, ?)",
                (f"{prefix}{i:05d}", b"x", base_expiry + i),
            )
        return count

    def test_cap_trims_to_max_rows_keeping_newest(self, monkeypatch):
        from scanner.api import data_providers
        from scanner.shared import db

        monkeypatch.setattr(data_providers, "PRICE_CACHE_MAX_ROWS", 10)
        monkeypatch.setattr(data_providers, "_last_prune_ts", 0.0)
        now = time.time()
        self._seed(25, now + 3600)  # all still live

        assert prune_stale_cache(force=True) == 15

        rows = (
            db.get_conn()
            .execute("SELECT cache_key FROM price_cache ORDER BY expires DESC")
            .fetchall()
        )
        assert len(rows) == 10
        # The survivors are the 10 newest by expiry.
        assert [r["cache_key"] for r in rows] == [
            f"k{i:05d}" for i in range(24, 14, -1)
        ]

    def test_cap_is_a_noop_below_the_limit(self, monkeypatch):
        from scanner.api import data_providers
        from scanner.shared import db

        monkeypatch.setattr(data_providers, "PRICE_CACHE_MAX_ROWS", 10)
        monkeypatch.setattr(data_providers, "_last_prune_ts", 0.0)
        self._seed(4, time.time() + 3600)

        assert prune_stale_cache(force=True) == 0
        n = db.get_conn().execute("SELECT COUNT(*) FROM price_cache").fetchone()[0]
        assert n == 4

    def test_expiry_sweep_still_runs_alongside_the_cap(self, monkeypatch):
        """Both halves of the sweep apply: expiry delete, then the cap."""
        from scanner.api import data_providers
        from scanner.shared import db

        monkeypatch.setattr(data_providers, "PRICE_CACHE_MAX_ROWS", 4)
        monkeypatch.setattr(data_providers, "_last_prune_ts", 0.0)
        now = time.time()
        # 3 already expired, 6 live -> expiry takes 3, cap trims 6 down to 4.
        self._seed(3, now - 7200, prefix="old")
        self._seed(6, now + 3600, prefix="new")

        removed = prune_stale_cache(force=True)

        assert removed == 5  # 3 expired + 2 over the cap of 4
        remaining = (
            db.get_conn().execute("SELECT COUNT(*) FROM price_cache").fetchone()[0]
        )
        assert remaining == 4
        # No expired row survived the sweep.
        stale = (
            db.get_conn()
            .execute("SELECT COUNT(*) FROM price_cache WHERE expires <= ?", (now,))
            .fetchone()[0]
        )
        assert stale == 0

    def test_default_cap_covers_a_full_market_generation(self):
        """The shipped default must not truncate a single full-market scan."""
        from scanner.api.data_providers import PRICE_CACHE_MAX_ROWS

        # A full-market scan writes ~2,400 rows; the cap needs headroom so a
        # second concurrent scan is not evicted, but must stay well under the
        # ~2.4k rows per 4h generation times several overlapping windows.
        assert PRICE_CACHE_MAX_ROWS >= 2500
        assert PRICE_CACHE_MAX_ROWS <= 20000

    @staticmethod
    def _spy_vacuum(monkeypatch, db, fail=False):
        """Record (or fail) VACUUM statements via the shared connection.

        ``sqlite3.Connection.execute`` is a read-only C attribute, so the
        connection is wrapped in a thin proxy that the module keeps using.
        """
        calls = []
        conn = db.get_conn()
        real_execute = conn.execute

        def spy(sql, *a, **kw):
            if sql.strip().upper().startswith("VACUUM"):
                calls.append(sql)
                if fail:
                    raise db.sqlite3.OperationalError("database is locked")
            return real_execute(sql, *a, **kw)

        class _Proxy:
            def execute(self, sql, *a, **kw):
                return spy(sql, *a, **kw)

            def __getattr__(self, name):
                return getattr(conn, name)

        monkeypatch.setattr(db, "get_conn", lambda *a, **kw: _Proxy())
        return calls

    def test_vacuum_is_skipped_below_the_slack_gate(self, monkeypatch):
        """A prune that frees little must not rewrite the whole db file.

        VACUUM is the only thing that returns space to the OS here, but it
        rewrites every page — so it is gated on real slack.
        """
        from scanner.api import data_providers
        from scanner.shared import db

        monkeypatch.setattr(data_providers, "PRICE_CACHE_MAX_ROWS", 10)
        monkeypatch.setattr(data_providers, "_last_prune_ts", 0.0)
        # Gate far above any slack a small test db can accumulate.
        monkeypatch.setattr(data_providers, "VACUUM_SLACK_MB", 1e9)
        self._seed(20, time.time() + 3600)
        calls = self._spy_vacuum(monkeypatch, db)

        prune_stale_cache(force=True)

        assert calls == [], "VACUUM must not run below the slack gate"

    def test_vacuum_runs_when_slack_exceeds_the_gate(self, monkeypatch):
        """Once enough pages are free, the prune reclaims them."""
        from scanner.api import data_providers
        from scanner.shared import db

        monkeypatch.setattr(data_providers, "PRICE_CACHE_MAX_ROWS", 10)
        monkeypatch.setattr(data_providers, "_last_prune_ts", 0.0)
        # Zero gate so any real slack triggers the reclaim path.
        monkeypatch.setattr(data_providers, "VACUUM_SLACK_MB", 0.0)
        self._seed(400, time.time() + 3600)
        calls = self._spy_vacuum(monkeypatch, db)

        prune_stale_cache(force=True)

        assert len(calls) == 1, "expected exactly one VACUUM after a big trim"
        n = db.get_conn().execute("SELECT COUNT(*) FROM price_cache").fetchone()[0]
        assert n == 10  # trim still applied

    def test_vacuum_failure_does_not_break_the_prune(self, monkeypatch):
        """A VACUUM error (locked db, disk full) must not lose the trim."""
        from scanner.api import data_providers
        from scanner.shared import db

        monkeypatch.setattr(data_providers, "PRICE_CACHE_MAX_ROWS", 10)
        monkeypatch.setattr(data_providers, "_last_prune_ts", 0.0)
        monkeypatch.setattr(data_providers, "VACUUM_SLACK_MB", 0.0)
        self._seed(20, time.time() + 3600)
        self._spy_vacuum(monkeypatch, db, fail=True)

        # The sweep still reports its removals rather than raising.
        assert prune_stale_cache(force=True) == 10
        n = db.get_conn().execute("SELECT COUNT(*) FROM price_cache").fetchone()[0]
        assert n == 10


class TestCacheHealth:
    """cache_health reports reachable-fresh vs unreachable-stale counts."""

    def _reset(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scanner.api.data_providers.CACHE_DIR", str(tmp_path))
        monkeypatch.setattr("scanner.api.data_providers._last_prune_ts", 0.0)

    def test_counts_fresh_and_stale(self, tmp_path, monkeypatch):
        self._reset(tmp_path, monkeypatch)
        from scanner.shared import db

        now = time.time()
        for name, expires in [
            ("a", now + 3600),
            ("b", now + 3600),
            ("c", now - 3600),
        ]:
            db.get_conn().execute(
                "INSERT INTO price_cache (cache_key, payload, expires)"
                " VALUES (?, ?, ?)",
                (name, b"ignored", expires),
            )

        h = cache_health()
        assert h["price_entries"] == 3
        assert h["stale_entries"] == 1

    def test_empty_dir_and_last_prune_stamp(self, tmp_path, monkeypatch):
        self._reset(tmp_path, monkeypatch)
        assert cache_health() == {
            "price_entries": 0,
            "stale_entries": 0,
            "last_prune": "",
        }
        prune_stale_cache()  # records the sweep time
        h = cache_health()
        assert h["last_prune"] != ""  # ISO stamp of the in-process prune
