"""Unit tests for scanner.scanner_engine — trend-filter / rating interplay."""

import threading
import time
from typing import ClassVar
from unittest.mock import patch

import pandas as pd
import pytest

from scanner.api import data_fetcher
from scanner.backend import scanner_engine
from scanner.backend.scanner_engine import (
    ScannerEngine,
    _build_scan_warnings,
    _enrich_rows_in_place,
    _find_stale_members,
    _score_ticker,
    _stale_members_message,
    rating_ok_for_trend_filter,
)
from scanner.backend.settings_store import DEFAULT_SETTINGS
from scanner.tests.conftest import _frame_ending, _last2_df


@pytest.fixture(autouse=True)
def _isolate_enrichment_cache(tmp_path, monkeypatch):
    """Reset the enrichment cache per test."""
    data_fetcher._ENRICHMENT_CACHE.clear()
    yield
    data_fetcher._ENRICHMENT_CACHE.clear()


def _tiny_df():
    """Minimal non-empty frame — compute_scores is patched in these tests."""
    return pd.DataFrame({"close": [1.0, 2.0], "volume": [1, 1]})


class TestTrendFilterRating:
    def test_all_shows_every_rating(self):
        """The 'All' filter keeps even POOR-rated stocks."""
        assert rating_ok_for_trend_filter("All", "POOR")
        assert rating_ok_for_trend_filter("All", "WEAK")
        assert rating_ok_for_trend_filter("All", "EXCELLENT")

    def test_bullish_only_hides_poor(self):
        """Bullish Only drops POOR/WEAK but keeps the rest."""
        assert not rating_ok_for_trend_filter("Bullish Only", "POOR")
        assert not rating_ok_for_trend_filter("Bullish Only", "WEAK")
        assert rating_ok_for_trend_filter("Bullish Only", "EXCELLENT")
        assert rating_ok_for_trend_filter("Bullish Only", "GOOD")
        assert rating_ok_for_trend_filter("Bullish Only", "MODERATE")

    def test_bearish_only_hides_poor(self):
        """Bearish Only drops POOR/WEAK but keeps the rest."""
        assert not rating_ok_for_trend_filter("Bearish Only", "POOR")
        assert not rating_ok_for_trend_filter("Bearish Only", "WEAK")
        assert rating_ok_for_trend_filter("Bearish Only", "MODERATE")

    def test_bullish_candle_hides_poor(self):
        """Bullish + Candle is directional: POOR/WEAK drop, rest stay."""
        assert not rating_ok_for_trend_filter("Bullish + Candle", "POOR")
        assert not rating_ok_for_trend_filter("Bullish + Candle", "WEAK")
        assert rating_ok_for_trend_filter("Bullish + Candle", "GOOD")
        assert rating_ok_for_trend_filter("Bullish + Candle", "MODERATE")

    def test_unknown_rating_is_kept(self):
        """A missing rating must never cause a row to be hidden defensively."""
        assert rating_ok_for_trend_filter("Bullish Only", None)
        assert rating_ok_for_trend_filter("Bearish Only", "")

    def test_non_directional_values_keep_poor(self):
        """Any unrecognised filter value behaves like 'All' (no rating drop)."""
        assert rating_ok_for_trend_filter("Bull", "POOR")
        assert rating_ok_for_trend_filter("", "POOR")


class TestBullishCandleFilterGate:
    """_score_ticker's "Bullish + Candle" gate: Bull direction + latest-bar pattern."""

    @staticmethod
    def _score(df, trend_filter, monkeypatch):
        monkeypatch.setattr(
            scanner_engine,
            "check_filter",
            lambda *a, **k: {"ma_bullish": True, "crossover_bars_ago": 0},
        )
        monkeypatch.setattr(
            scanner_engine,
            "compute_scores",
            lambda *a, **k: {"total": 88.0, "combined_rating": "EXCELLENT"},
        )
        return _score_ticker(
            "TC1",
            df,
            settings={"min_score": 50.0},
            timeframe="D",
            index_df=df,
            trend_filter=trend_filter,
            is_large=True,
            enrich=lambda *a: {},
            use_enrichment_cache=False,
        )

    def test_passes_with_bullish_engulfing(self, monkeypatch):
        """Bull direction + engulfing last bar → scored, not filtered."""
        df = _last2_df(98.0, 108.0, 97.0, 107.0)
        out, reason = self._score(df, "Bullish + Candle", monkeypatch)
        assert out is not None
        assert reason == "Bull"

    def test_filters_missing_pattern(self, monkeypatch):
        """Bull direction but no engulfing/hammer on the last bar → filtered."""
        df = _last2_df(100.0, 110.0, 99.0, 102.0)
        out, reason = self._score(df, "Bullish + Candle", monkeypatch)
        assert (out, reason) == (None, "filtered")

    def test_filters_non_bullish_direction(self, monkeypatch):
        """Bear direction never reaches the pattern check → filtered."""
        monkeypatch.setattr(
            scanner_engine,
            "check_filter",
            lambda *a, **k: {"ma_bullish": False, "crossover_bars_ago": 0},
        )
        monkeypatch.setattr(
            scanner_engine,
            "compute_scores",
            lambda *a, **k: pytest.fail("must not score past the direction gate"),
        )
        df = _last2_df(98.0, 108.0, 97.0, 107.0)
        out, reason = _score_ticker(
            "TC2",
            df,
            settings={"min_score": 50.0},
            timeframe="D",
            index_df=df,
            trend_filter="Bullish + Candle",
            is_large=True,
            enrich=lambda *a: {},
            use_enrichment_cache=False,
        )
        assert (out, reason) == (None, "filtered")

    def test_all_filter_never_checks_pattern(self, monkeypatch):
        """trend_filter="All" ignores candlestick patterns entirely."""
        df = _last2_df(100.0, 110.0, 99.0, 102.0)
        out, reason = self._score(df, "All", monkeypatch)
        assert out is not None
        assert reason == "Bull"


class TestEnrichRowsInPlaceCache:
    """_enrich_one replays the provider cache instead of re-fetching."""

    def _run(self, rows, batch_data, enrich):
        with patch(
            "scanner.backend.scanner_engine.compute_scores",
            return_value={"total": 88.0, "combined_rating": "EXCELLENT"},
        ):
            return _enrich_rows_in_place(
                rows,
                batch_data,
                settings={},
                timeframe="D",
                index_df=None,
                enrich=enrich,
            )

    def test_cache_hit_skips_provider_fetch_and_uses_cached_fundamentals(self):
        """A cached ticker reuses providers + fundamentals — no network calls."""
        data_fetcher._enrichment_cache_put(
            "TCS",
            {"_sentiment_score": 0.9, "_insider_score": 5},
            {"pe_ratio": 20.0},
        )
        df = _tiny_df()
        rows = [{"ticker": "TCS", "total": 50.0}]

        def enrich(ticker, settings):
            raise AssertionError("provider enrich must not run on a cache hit")

        with patch("scanner.backend.scanner_engine.fetch_fundamentals") as mock_fund:
            out = self._run(rows, {"TCS": df}, enrich)

        mock_fund.assert_not_called()
        row = out[0]
        assert row["_sentiment_score"] == 0.9  # provider keys replayed
        assert row["_insider_score"] == 5
        assert df.attrs.get("_fundamentals") == {
            "pe_ratio": 20.0
        }  # cached fundamentals attached
        assert row["total"] == 88.0  # re-scored on top of cached data
        assert row["combined_rating"] == "EXCELLENT"

    def test_provider_keys_reach_re_score_settings(self):
        """Phase-2 re-score must see provider "_"-keys (institutional bonus)."""
        data_fetcher._enrichment_cache_put(
            "TCS", {"_fii_is_buying": True, "_dii_is_buying": False}, None
        )
        df = _tiny_df()
        rows = [{"ticker": "TCS", "total": 50.0}]
        seen: dict = {}

        def fake_compute(df_arg, timeframe=None, index_df=None, settings=None):
            seen.update(settings or {})
            return {"total": 88.0, "combined_rating": "EXCELLENT"}

        with patch(
            "scanner.backend.scanner_engine.compute_scores",
            side_effect=fake_compute,
        ):
            _enrich_rows_in_place(
                rows,
                {"TCS": df},
                settings={},
                timeframe="D",
                index_df=None,
                enrich=lambda *a, **k: AssertionError("cache hit, no enrich"),
            )
        assert seen["_fii_is_buying"] is True
        assert seen["_dii_is_buying"] is False

    def test_cache_miss_runs_providers_and_populates_cache(self):
        """A fresh ticker fetches from providers, then caches the result."""
        df = _tiny_df()
        rows = [{"ticker": "TCS", "total": 50.0}]
        calls = []

        def enrich(ticker, settings):
            calls.append(ticker)
            return {"_sentiment_score": 0.7, "_article_count": 3}

        with patch(
            "scanner.backend.scanner_engine.fetch_fundamentals",
            return_value={"pe_ratio": 15.0},
        ):
            out = self._run(rows, {"TCS": df}, enrich)

        assert calls == ["TCS"]
        row = out[0]
        assert row["_sentiment_score"] == 0.7
        assert row["_article_count"] == 3
        assert df.attrs.get("_fundamentals") == {"pe_ratio": 15.0}
        # The fetched result is now cached for the next scan
        entry = data_fetcher._enrichment_cache_get("TCS")
        assert entry is not None
        assert entry["providers"] == {"_sentiment_score": 0.7, "_article_count": 3}
        assert entry["fundamentals"] == {"pe_ratio": 15.0}

    def test_cached_known_none_fundamentals_skip_refetch(self):
        """fundamentals=None cached means 'no fundamentals' — don't re-ask."""
        data_fetcher._enrichment_cache_put("SMALL", {"_social_score": 0.2}, None)
        df = _tiny_df()
        rows = [{"ticker": "SMALL", "total": 50.0}]

        def enrich(ticker, settings):
            raise AssertionError("provider enrich must not run on a cache hit")

        with patch("scanner.backend.scanner_engine.fetch_fundamentals") as mock_fund:
            out = self._run(rows, {"SMALL": df}, enrich)

        mock_fund.assert_not_called()
        assert "_fundamentals" not in df.attrs  # stays fundamentals-free
        assert out[0]["total"] == 88.0

    def test_cache_miss_with_all_providers_empty_is_not_cached(self):
        """Providers returning nothing must not freeze an empty entry."""
        df = _tiny_df()
        rows = [{"ticker": "NEW", "total": 50.0}]

        def enrich(ticker, settings):
            return {}

        with patch(
            "scanner.backend.scanner_engine.fetch_fundamentals", return_value=None
        ):
            self._run(rows, {"NEW": df}, enrich)

        assert data_fetcher._enrichment_cache_get("NEW") is None


class TestEnrichRowsInPlaceCancel:
    def test_cancel_returns_promptly_without_waiting_for_workers(self):
        """Cancel mid-enrichment returns while slow provider calls still run."""
        rows = [{"ticker": f"T{i}", "total": 50.0} for i in range(20)]
        cancel = threading.Event()

        def slow_enrich(ticker, settings, gd):
            time.sleep(5)
            return {}

        def fire():
            time.sleep(0.5)
            cancel.set()

        threading.Thread(target=fire, daemon=True).start()
        t0 = time.time()
        out = _enrich_rows_in_place(
            rows,
            {},
            settings={},
            timeframe="D",
            index_df=None,
            enrich=slow_enrich,
            cancel_event=cancel,
        )
        elapsed = time.time() - t0
        assert elapsed < 4  # returned while the 5s enrich calls still ran
        assert out is rows  # same list, mutated in place

    def test_without_cancel_waits_for_all_rows(self):
        """No cancel event -> behaves like before: every row is enriched."""
        rows = [{"ticker": f"T{i}", "total": 50.0} for i in range(4)]
        seen = []

        def fast_enrich(ticker, settings):
            seen.append(ticker)
            return {}

        out = _enrich_rows_in_place(
            rows,
            {},
            settings={},
            timeframe="D",
            index_df=None,
            enrich=fast_enrich,
        )
        assert sorted(seen) == [f"T{i}" for i in range(4)]
        assert out is rows


class TestCancelPropagation:
    """A user cancel must be reported even when no data was ever produced.

    Regression: when the batch download was cancelled mid-flight it could
    stop cleanly before the engine loop body's own cancel check ever ran,
    so scan()/scan_stream() returned ``cancelled=False`` with zero results.
    """

    @staticmethod
    def _wait_until(evt, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline and not evt.is_set():
            time.sleep(0.01)
        assert evt.is_set(), "worker never reached the download stage"

    def test_scan_stream_cancel_during_download_is_reported(self):
        """Cancel lands while the stream generator is mid-download (no chunk
        ever yielded) — result must still say cancelled=True."""
        entered = threading.Event()

        def fake_stream(
            tickers,
            period="1y",
            timeframe="D",
            cancel_event=None,
            on_fallback_progress=None,
        ):
            # Real fetch_batch_yfinance_stream detects the cancel while the
            # in-flight batch is downloading and ends without yielding.
            entered.set()
            while cancel_event is not None and not cancel_event.is_set():
                time.sleep(0.01)
            return
            yield  # pragma: no cover - makes this a generator

        engine = ScannerEngine()
        holder = {}

        def worker():
            holder["result"] = engine.scan_stream(
                "NIFTY 50",
                settings=dict(DEFAULT_SETTINGS),
                period="1y",
                timeframe="D",
                index_symbol="NSEI",
            )

        with (
            patch("scanner.backend.scanner_engine.fetch_index_data", return_value=None),
            patch(
                "scanner.backend.scanner_engine.fetch_batch_yfinance_stream",
                fake_stream,
            ),
        ):
            th = threading.Thread(target=worker, daemon=True)
            th.start()
            self._wait_until(entered)
            engine.cancel()
            th.join(timeout=15)

        assert not th.is_alive(), "scan_stream did not return after cancel"
        result = holder["result"]
        assert result.error is None
        assert result.cancelled is True  # regression: used to be False
        assert result.results == []

    def test_scan_cancel_during_download_is_reported(self):
        """scan() has the same edge: cancel during the (non-stream) batch
        download leaves an empty batch dict and an empty scoring loop."""
        entered = threading.Event()

        def fake_batch(
            tickers,
            period="1y",
            timeframe="D",
            cancel_event=None,
            on_fallback_progress=None,
        ):
            entered.set()
            while cancel_event is not None and not cancel_event.is_set():
                time.sleep(0.01)
            return {}

        engine = ScannerEngine()
        holder = {}

        def worker():
            holder["result"] = engine.scan(
                "NIFTY 50",
                settings=dict(DEFAULT_SETTINGS),
                period="1y",
                timeframe="D",
                index_symbol="NSEI",
            )

        with (
            patch("scanner.backend.scanner_engine.fetch_index_data", return_value=None),
            patch("scanner.backend.scanner_engine.fetch_batch_yfinance", fake_batch),
        ):
            th = threading.Thread(target=worker, daemon=True)
            th.start()
            self._wait_until(entered)
            engine.cancel()
            th.join(timeout=15)

        assert not th.is_alive(), "scan did not return after cancel"
        result = holder["result"]
        assert result.error is None
        assert result.cancelled is True  # regression: used to be False
        assert result.results == []

    def test_scan_stream_cancel_between_chunks_keeps_partial_results(self):
        """Cancel landing after a chunk was already streamed keeps those
        results and is still reported as cancelled (wind-down path)."""
        entered = threading.Event()
        df = pd.DataFrame({"close": [1.0, 2.0], "volume": [1, 1]})

        def fake_stream(
            tickers,
            period="1y",
            timeframe="D",
            cancel_event=None,
            on_fallback_progress=None,
        ):
            yield {"TCS": df}  # first chunk streams normally
            entered.set()
            while cancel_event is not None and not cancel_event.is_set():
                time.sleep(0.01)

        engine = ScannerEngine()
        holder = {}

        def worker():
            holder["result"] = engine.scan_stream(
                "NIFTY 50",
                settings=dict(DEFAULT_SETTINGS),
                period="1y",
                timeframe="D",
                index_symbol="NSEI",
            )

        fake_score = {
            "ticker": "TCS",
            "total": 55.0,
            "trend_dir": "Bull",
            "combined_rating": "GOOD",
        }

        def fake_score_ticker(ticker, df, **kw):
            return fake_score, "Bull"

        with (
            patch("scanner.backend.scanner_engine.fetch_index_data", return_value=None),
            patch(
                "scanner.backend.scanner_engine.fetch_batch_yfinance_stream",
                fake_stream,
            ),
            patch("scanner.backend.scanner_engine._score_ticker", fake_score_ticker),
        ):
            th = threading.Thread(target=worker, daemon=True)
            th.start()
            self._wait_until(entered)
            engine.cancel()
            th.join(timeout=15)

        assert not th.is_alive(), "scan_stream did not return after cancel"
        result = holder["result"]
        assert result.error is None
        assert result.cancelled is True
        assert [r["ticker"] for r in result.results] == ["TCS"]  # kept

    def test_cancel_idempotent_and_per_scan(self):
        """cancel() is idempotent; a fresh engine starts un-cancelled."""
        engine = ScannerEngine()
        assert not engine._cancel_event.is_set()
        engine.cancel()
        engine.cancel()
        assert engine._cancel_event.is_set()


class TestScanWarnings:
    """_build_scan_warnings flags broad-but-weak filters and stays quiet on
    healthy scans (regression guard for the loose-filter warning banner)."""

    def test_broad_filter_few_entries_warns(self):
        settings = {"fast_ma_len": 20, "slow_ma_len": 40, "min_score": 50.0}
        results = [{"total": 52.0, "entry_signal": True}] + [
            {"total": 52.0, "entry_signal": False} for _ in range(44)
        ]
        warnings = _build_scan_warnings(settings, total=51, results=results, passed=10)
        assert len(warnings) == 1
        assert "45/51" in warnings[0]
        assert "only 1 have entry signals" in warnings[0]

    def test_broad_filter_zero_entries_warns_explicitly(self):
        settings = {"fast_ma_len": 20, "slow_ma_len": 40, "min_score": 50.0}
        results = [{"total": 45.0, "entry_signal": False} for _ in range(30)]
        warnings = _build_scan_warnings(settings, total=50, results=results, passed=0)
        assert len(warnings) == 1
        assert "NONE" in warnings[0]

    def test_healthy_scan_no_warning(self):
        settings = {"fast_ma_len": 40, "slow_ma_len": 50, "min_score": 50.0}
        results = [{"total": 62.0, "entry_signal": True} for _ in range(4)] + [
            {"total": 40.0, "entry_signal": False} for _ in range(6)
        ]
        assert _build_scan_warnings(settings, total=51, results=results, passed=4) == []

    def test_small_scan_never_warns(self):
        settings = {"fast_ma_len": 20, "slow_ma_len": 40, "min_score": 50.0}
        results = [{"total": 52.0, "entry_signal": False} for _ in range(20)]
        assert _build_scan_warnings(settings, total=51, results=results, passed=5) == []


# ══════════════════════════════════════════════════════════════════════════════
# Stale universe members — suspended/delisted names still fetched each scan
# ══════════════════════════════════════════════════════════════════════════════


class TestStaleMembers:
    """_find_stale_members flags only names whose data ended long ago."""

    def test_current_frames_not_stale(self):
        batch = {"RELIANCE": _frame_ending(0), "TCS": _frame_ending(2)}
        assert _find_stale_members(batch) == []

    def test_old_frames_flagged_oldest_first(self):
        batch = {
            "FRESH": _frame_ending(5),
            "GSPL": _frame_ending(120),  # halted months ago
            "TATAMETALI": _frame_ending(700),  # merged/delisted 2024
        }
        stale = _find_stale_members(batch)
        assert [t for t, _ in stale] == ["TATAMETALI", "GSPL"]  # oldest first
        assert stale[0][1] < stale[1][1]

    def test_threshold_is_configurable(self):
        batch = {"OLD": _frame_ending(60)}
        assert _find_stale_members(batch, max_age_days=30) == [
            ("OLD", _frame_ending(60).index[-1].date().isoformat())
        ]
        assert _find_stale_members(batch, max_age_days=90) == []

    def test_empty_and_malformed_are_safe(self):
        assert _find_stale_members({}) == []
        assert _find_stale_members({"BROKEN": object()}) == []

    def test_message_lists_names_and_count(self):
        msg = _stale_members_message(
            [("GSPL", "2026-05-11"), ("TATAMETALI", "2024-02-05")]
        )
        assert "2 universe member(s)" in msg
        assert "GSPL (2026-05-11)" in msg
        assert "TATAMETALI (2024-02-05)" in msg

    def test_message_truncates_long_lists(self):
        many = [(f"T{i}", "2026-01-01") for i in range(7)]
        msg = _stale_members_message(many)
        assert "+2 more" in msg

    def test_message_reflects_custom_threshold(self):
        msg = _stale_members_message([("GSPL", "2026-05-11")], max_age_days=90)
        assert "last bar > 90d old" in msg


class TestPromoterProviderExtraction:
    """india_fund branch surfaces trendlyne promoter holding as a provider key."""

    _SETTINGS: ClassVar[dict] = {}

    @pytest.fixture(autouse=True)
    def _stub_other_providers(self):
        """Providers always run now — stub the three this test doesn't patch."""
        with (
            patch(
                "scanner.api.market_sentiment.fetch_sentiment",
                return_value={
                    "sentiment_score": 0.0,
                    "article_count": 0,
                    "source": "none",
                },
            ),
            patch(
                "scanner.api.indian_market.fetch_indian_market_data",
                return_value={},
            ),
            patch("scanner.api.insider_data.fetch_insider_data", return_value={}),
        ):
            yield

    @staticmethod
    def _fund_result(promoter):
        from scanner.api.indian_fundamentals import TrendlyneFundamentals

        fund = (
            TrendlyneFundamentals(ticker="X", promoter_holding=promoter)
            if promoter is not None
            else None
        )
        return {
            "trendlyne": fund,
            "screener": None,
            "source": "trendlyne" if fund else "none",
        }

    def test_promoter_holding_extracted_and_rounded(self):
        with patch(
            "scanner.api.indian_fundamentals.fetch_indian_fundamentals",
            return_value=self._fund_result(51.798),
        ):
            out = ScannerEngine()._enrich_with_providers("X", dict(self._SETTINGS))
        assert out["_promoter_holding"] == 51.8

    def test_no_promoter_key_without_data(self):
        with patch(
            "scanner.api.indian_fundamentals.fetch_indian_fundamentals",
            return_value=self._fund_result(None),
        ):
            out = ScannerEngine()._enrich_with_providers("X", dict(self._SETTINGS))
        assert "_promoter_holding" not in out
