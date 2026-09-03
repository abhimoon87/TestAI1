"""Unit tests for scanner.scanner_engine — trend-filter / rating interplay."""

import threading
import time
from unittest.mock import patch

import pandas as pd
import pytest

import scanner.data_fetcher as data_fetcher
from scanner.scanner_engine import _enrich_rows_in_place, rating_ok_for_trend_filter


@pytest.fixture(autouse=True)
def _isolate_enrichment_cache(tmp_path, monkeypatch):
    """Point the enrichment cache at a temp file and reset it per test."""
    monkeypatch.setattr(
        data_fetcher, "_ENRICHMENT_CACHE_PATH",
        str(tmp_path / "enrichment_cache.json"),
    )
    monkeypatch.setattr(data_fetcher, "_enrichment_cache", None)
    yield
    monkeypatch.setattr(data_fetcher, "_enrichment_cache", None)


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

    def test_unknown_rating_is_kept(self):
        """A missing rating must never cause a row to be hidden defensively."""
        assert rating_ok_for_trend_filter("Bullish Only", None)
        assert rating_ok_for_trend_filter("Bearish Only", "")

    def test_non_directional_values_keep_poor(self):
        """Any unrecognised filter value behaves like 'All' (no rating drop)."""
        assert rating_ok_for_trend_filter("Bull", "POOR")
        assert rating_ok_for_trend_filter("", "POOR")


class TestEnrichRowsInPlaceCache:
    """_enrich_one replays the provider cache instead of re-fetching."""

    def _run(self, rows, batch_data, enrich):
        with patch("scanner.scanner_engine.compute_scores",
                   return_value={"total": 88.0, "combined_rating": "EXCELLENT"}):
            return _enrich_rows_in_place(
                rows, batch_data,
                settings={}, global_data=None,
                timeframe="D", index_df=None,
                enrich=enrich,
            )

    def test_cache_hit_skips_provider_fetch_and_uses_cached_fundamentals(self):
        """A cached ticker reuses providers + fundamentals — no network calls."""
        data_fetcher._enrichment_cache_put(
            "TCS", {"_sentiment_score": 0.9, "_insider_score": 5},
            {"pe_ratio": 20.0},
        )
        df = _tiny_df()
        rows = [{"ticker": "TCS", "total": 50.0}]

        def enrich(ticker, settings, gd):
            raise AssertionError("provider enrich must not run on a cache hit")

        with patch("scanner.scanner_engine.fetch_fundamentals") as mock_fund:
            out = self._run(rows, {"TCS": df}, enrich)

        mock_fund.assert_not_called()
        row = out[0]
        assert row["_sentiment_score"] == 0.9  # provider keys replayed
        assert row["_insider_score"] == 5
        assert df._fundamentals == {"pe_ratio": 20.0}  # cached fundamentals attached
        assert row["total"] == 88.0  # re-scored on top of cached data
        assert row["combined_rating"] == "EXCELLENT"

    def test_cache_miss_runs_providers_and_populates_cache(self):
        """A fresh ticker fetches from providers, then caches the result."""
        df = _tiny_df()
        rows = [{"ticker": "TCS", "total": 50.0}]
        calls = []

        def enrich(ticker, settings, gd):
            calls.append(ticker)
            return {"_sentiment_score": 0.7, "_article_count": 3}

        with patch("scanner.scanner_engine.fetch_fundamentals",
                   return_value={"pe_ratio": 15.0}):
            out = self._run(rows, {"TCS": df}, enrich)

        assert calls == ["TCS"]
        row = out[0]
        assert row["_sentiment_score"] == 0.7
        assert row["_article_count"] == 3
        assert df._fundamentals == {"pe_ratio": 15.0}
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

        def enrich(ticker, settings, gd):
            raise AssertionError("provider enrich must not run on a cache hit")

        with patch("scanner.scanner_engine.fetch_fundamentals") as mock_fund:
            out = self._run(rows, {"SMALL": df}, enrich)

        mock_fund.assert_not_called()
        assert not hasattr(df, "_fundamentals")  # stays fundamentals-free
        assert out[0]["total"] == 88.0

    def test_cache_miss_with_all_providers_empty_is_not_cached(self):
        """Providers returning nothing must not freeze an empty entry."""
        df = _tiny_df()
        rows = [{"ticker": "NEW", "total": 50.0}]

        def enrich(ticker, settings, gd):
            return {}

        with patch("scanner.scanner_engine.fetch_fundamentals", return_value=None):
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
            rows, {},
            settings={}, global_data=None,
            timeframe="D", index_df=None,
            enrich=slow_enrich, cancel_event=cancel,
        )
        elapsed = time.time() - t0
        assert elapsed < 4  # returned while the 5s enrich calls still ran
        assert out is rows  # same list, mutated in place

    def test_without_cancel_waits_for_all_rows(self):
        """No cancel event -> behaves like before: every row is enriched."""
        rows = [{"ticker": f"T{i}", "total": 50.0} for i in range(4)]
        seen = []

        def fast_enrich(ticker, settings, gd):
            seen.append(ticker)
            return {}

        out = _enrich_rows_in_place(
            rows, {},
            settings={}, global_data=None,
            timeframe="D", index_df=None,
            enrich=fast_enrich,
        )
        assert sorted(seen) == [f"T{i}" for i in range(4)]
        assert out is rows
