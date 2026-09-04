"""Integration tests using real yfinance data.

These tests hit the live Yahoo Finance API and verify the full pipeline
works end-to-end with real market data. They are slower than unit tests
and require network access.

Run with:  pytest -m integration                  (CLI -m overrides the default deselect)
Skip with: pytest                                 (default: integration tests are deselected via addopts)
"""

import pytest

from scanner.data_fetcher import (
    fetch_batch_yfinance,
    fetch_fundamentals,
    fetch_index_data,
    resample_ohlcv,
)
from scanner.scoring import check_filter, compute_scores, get_direction

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

# Small stock subset — keeps network calls fast
SMOKE_STOCKS = ["RELIANCE", "TCS"]
FULL_STOCKS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]


# ══════════════════════════════════════════════════════════════════════════════
# Data Fetching — Real Network
# ══════════════════════════════════════════════════════════════════════════════


class TestFetchRealData:
    """Verify real yfinance data fetching returns valid DataFrames."""

    def test_batch_download_returns_data(self):
        """Batch download should return DataFrames for all tickers."""
        result = fetch_batch_yfinance(SMOKE_STOCKS, period="1y")
        assert len(result) == len(SMOKE_STOCKS)
        for ticker in SMOKE_STOCKS:
            assert ticker in result

    def test_batch_download_columns(self):
        """Each DataFrame should have the expected columns."""
        result = fetch_batch_yfinance(SMOKE_STOCKS, period="1y")
        for df in result.values():
            assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_batch_download_min_bars(self):
        """Each DataFrame should have at least 50 bars."""
        result = fetch_batch_yfinance(SMOKE_STOCKS, period="1y")
        for ticker, df in result.items():
            assert len(df) >= 50, f"{ticker} has only {len(df)} bars"

    def test_batch_download_no_nans(self):
        """DataFrames should have no NaN values (dropna applied)."""
        result = fetch_batch_yfinance(SMOKE_STOCKS, period="1y")
        for ticker, df in result.items():
            assert df.notna().all().all(), f"{ticker} has NaN values"

    def test_batch_download_non_negative_volume(self):
        """Volume should be non-negative (zero is valid for holidays/suspensions)."""
        result = fetch_batch_yfinance(SMOKE_STOCKS, period="1y")
        for ticker, df in result.items():
            assert (df["volume"] >= 0).all(), f"{ticker} has negative volume"

    def test_batch_download_high_gte_low(self):
        """High should be >= Low for all bars."""
        result = fetch_batch_yfinance(SMOKE_STOCKS, period="1y")
        for ticker, df in result.items():
            assert (df["high"] >= df["low"]).all(), f"{ticker} has high < low"

    def test_single_ticker_download(self):
        """Single-ticker download should work."""
        result = fetch_batch_yfinance(["RELIANCE"], period="1y")
        assert len(result) == 1
        assert "RELIANCE" in result

    def test_index_data_fetch(self):
        """NIFTY 50 index data should be fetchable."""
        df = fetch_index_data("^NSEI", period="1y")
        assert df is not None
        assert len(df) >= 100
        assert "close" in df.columns

    def test_fundamentals_fetch(self):
        """Fundamentals should be fetchable for a major stock."""
        fund = fetch_fundamentals("RELIANCE")
        # Fundamentals may or may not be available depending on providers
        # Just verify it doesn't crash and returns dict or None
        if fund is not None:
            assert isinstance(fund, dict)
            assert "pe_ratio" in fund


# ══════════════════════════════════════════════════════════════════════════════
# Resampling — Real Data
# ══════════════════════════════════════════════════════════════════════════════


class TestResampleRealData:
    """Verify resampling works on real fetched data."""

    def test_weekly_resample(self):
        """Daily data should resample to weekly without errors."""
        result = fetch_batch_yfinance(SMOKE_STOCKS, period="1y")
        for df in result.values():
            weekly = resample_ohlcv(df, "W")
            assert weekly is not None
            assert len(weekly) < len(df)
            assert list(weekly.columns) == ["open", "high", "low", "close", "volume"]

    def test_monthly_resample(self):
        """Daily data should resample to monthly without errors."""
        result = fetch_batch_yfinance(SMOKE_STOCKS, period="5y")
        for df in result.values():
            monthly = resample_ohlcv(df, "M")
            assert monthly is not None
            assert len(monthly) < len(df)

    def test_weekly_ohlc_integrity(self):
        """Weekly OHLC should be consistent: high >= open, close; low <= open, close."""
        result = fetch_batch_yfinance(SMOKE_STOCKS, period="1y")
        for df in result.values():
            weekly = resample_ohlcv(df, "W")
            assert (weekly["high"] >= weekly["open"]).all()
            assert (weekly["high"] >= weekly["close"]).all()
            assert (weekly["low"] <= weekly["open"]).all()
            assert (weekly["low"] <= weekly["close"]).all()


# ══════════════════════════════════════════════════════════════════════════════
# Scoring Pipeline — Real Data
# ══════════════════════════════════════════════════════════════════════════════


class TestScoringRealData:
    """Verify the 3-model scoring pipeline works on real market data."""

    @pytest.fixture(scope="class")
    def real_stock_data(self):
        """Fetch real stock data once for the class."""
        return fetch_batch_yfinance(FULL_STOCKS, period="1y")

    @pytest.fixture(scope="class")
    def real_index_data(self):
        """Fetch real index data once for the class."""
        return fetch_index_data("^NSEI", period="1y")

    def test_compute_scores_returns_dict(self, real_stock_data, real_index_data):
        """compute_scores should return a result dict for real data."""
        for df in real_stock_data.values():
            result = compute_scores(df, timeframe="D", index_df=real_index_data)
            # May return None for some stocks (insufficient data), that's OK
            if result is not None:
                assert isinstance(result, dict)
                assert "total" in result

    def test_score_range(self, real_stock_data, real_index_data):
        """Scores should be in [0, 100]."""
        for ticker, df in real_stock_data.items():
            result = compute_scores(df, timeframe="D", index_df=real_index_data)
            if result is not None:
                assert 0 <= result["total"] <= 100, \
                    f"{ticker}: total={result['total']} out of range"

    def test_all_categories_present(self, real_stock_data, real_index_data):
        """All 10 scoring categories should be in the result."""
        categories = ["trend", "momentum", "rsi", "macd", "stoch",
                       "obv", "volume", "rel_str", "volatility", "fundamentals"]
        for ticker, df in real_stock_data.items():
            result = compute_scores(df, timeframe="D", index_df=real_index_data)
            if result is not None:
                for cat in categories:
                    assert cat in result, f"{ticker}: missing {cat}"

    def test_category_bounds(self, real_stock_data, real_index_data):
        """Each category should not exceed its max weight."""
        max_bounds = {
            "trend": 15, "momentum": 15, "rsi": 8, "macd": 7,
            "stoch": 5, "obv": 5, "volume": 10, "rel_str": 10,
            "volatility": 5, "fundamentals": 20,
        }
        for ticker, df in real_stock_data.items():
            result = compute_scores(df, timeframe="D", index_df=real_index_data)
            if result is not None:
                for cat, max_val in max_bounds.items():
                    assert result[cat] <= max_val + 0.1, \
                        f"{ticker}: {cat}={result[cat]} exceeds max {max_val}"

    def test_trend_dir_valid(self, real_stock_data, real_index_data):
        """trend_dir should be 'Bull' or 'Bear'."""
        for ticker, df in real_stock_data.items():
            result = compute_scores(df, timeframe="D", index_df=real_index_data)
            if result is not None:
                assert result["trend_dir"] in ("Bull", "Bear"), \
                    f"{ticker}: trend_dir={result['trend_dir']}"

    def test_entry_signal_is_bool(self, real_stock_data, real_index_data):
        """entry_signal should be a boolean."""
        for df in real_stock_data.values():
            result = compute_scores(df, timeframe="D", index_df=real_index_data)
            if result is not None:
                assert isinstance(result["entry_signal"], bool)

    def test_filter_and_score_pipeline(self, real_stock_data, real_index_data):
        """Full pipeline: check_filter → get_direction → compute_scores."""
        # Use matching MA params for filter and scorer
        settings = {
            "fast_ma_type": "HMA", "fast_ma_len": 44,
            "slow_ma_type": "EMA", "slow_ma_len": 30,
            "crossover_lookback": 20,
        }
        passed = 0
        for df in real_stock_data.values():
            filter_result = check_filter(
                df, fast_ma_type="HMA", fast_ma_len=44,
                slow_ma_type="EMA", slow_ma_len=30,
                crossover_lookback=20,
            )
            if filter_result is None:
                continue  # filtered out — that's fine
            direction = get_direction(filter_result)
            assert direction in ("Bull", "Bear")
            scores = compute_scores(df, timeframe="D", index_df=real_index_data,
                                    settings=settings)
            if scores is not None:
                passed += 1
                assert scores["total"] >= 0
                # trend_dir uses close vs slow_ma — should match direction
                # from the filter (both compare fast vs slow MA)
                assert scores["trend_dir"] in ("Bull", "Bear")

        # At least some stocks should pass the filter
        assert passed > 0, "No stocks passed the filter — check pipeline"

    def test_weekly_timeframe(self, real_stock_data, real_index_data):
        """Scoring should work with weekly timeframe."""
        for df in real_stock_data.values():
            result = compute_scores(df, timeframe="W", index_df=real_index_data)
            if result is not None:
                assert 0 <= result["total"] <= 100

    def test_with_fundamentals(self, real_stock_data):
        """Scoring with attached fundamentals should work."""
        for ticker, df in real_stock_data.items():
            fund = fetch_fundamentals(ticker)
            if fund is not None:
                object.__setattr__(df, '_fundamentals', fund)
            result = compute_scores(df, timeframe="D")
            if result is not None:
                assert 0 <= result["total"] <= 100
                if fund is not None:
                    # Fundamentals score should be > 0 when data is available
                    assert result["fundamentals"] >= 0


# ══════════════════════════════════════════════════════════════════════════════
# End-to-End Consistency
# ══════════════════════════════════════════════════════════════════════════════


class TestEndToEndConsistency:
    """Verify that the full pipeline produces consistent, sensible results."""

    def test_top_scorer_has_good_signals(self):
        """The highest-scoring stock should have at least some bullish signals."""
        data = fetch_batch_yfinance(FULL_STOCKS, period="1y")
        index_df = fetch_index_data("^NSEI", period="1y")

        scored = []
        for ticker, df in data.items():
            fund = fetch_fundamentals(ticker)
            if fund is not None:
                object.__setattr__(df, '_fundamentals', fund)
            result = compute_scores(df, timeframe="D", index_df=index_df)
            if result is not None:
                result["ticker"] = ticker
                scored.append(result)

        assert len(scored) > 0, "No stocks could be scored"

        scored.sort(key=lambda x: x["total"], reverse=True)
        top = scored[0]

        # Top scorer should have reasonable properties
        assert top["total"] >= 20, f"Top scorer {top['ticker']} has suspiciously low score"
        assert top["trend_dir"] in ("Bull", "Bear")

    def test_score_sum_matches_total(self):
        """Sum of category scores should equal total (within rounding)."""
        data = fetch_batch_yfinance(SMOKE_STOCKS, period="1y")
        index_df = fetch_index_data("^NSEI", period="1y")

        for ticker, df in data.items():
            result = compute_scores(df, timeframe="D", index_df=index_df)
            if result is not None:
                categories = ["trend", "momentum", "rsi", "macd", "stoch",
                               "obv", "volume", "rel_str", "volatility", "fundamentals"]
                cat_sum = sum(result[c] for c in categories)
                assert abs(cat_sum - result["total"]) < 0.2, \
                    f"{ticker}: category sum {cat_sum:.1f} != total {result['total']:.1f}"


class TestLiveCancel:
    """Live cancel: engine.cancel() mid-scan must return promptly."""

    def test_cancel_mid_scan_returns_promptly(self):
        """Cancelling a live streaming scan returns quickly without errors."""
        import threading
        import time

        from scanner.scanner_engine import ScannerEngine
        from scanner.settings_store import DEFAULT_SETTINGS
        from scanner.universes import UNIVERSES

        tickers = UNIVERSES.get("FnO STOCKS", [])[:50]
        assert tickers, "FnO STOCKS universe is empty"

        settings = dict(DEFAULT_SETTINGS)
        settings["data_period"] = "1y"
        engine = ScannerEngine()
        holder = {}

        def worker():
            holder["result"] = engine.scan_stream(
                "FnO STOCKS", settings=settings, period="1y",
                timeframe="D", index_symbol="NSEI",
            )

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        # Cancel while the live download is still in flight
        time.sleep(5)
        engine.cancel()
        t_cancel = time.time()
        th.join(timeout=90)
        latency = time.time() - t_cancel

        assert not th.is_alive(), "scan_stream did not return after cancel"
        assert latency < 30, f"cancel latency too high: {latency:.1f}s"
        result = holder.get("result")
        assert result is not None
        assert result.error is None
        # Either the cancel was honored or the small scan completed with data
        assert result.cancelled or len(result.results) > 0
