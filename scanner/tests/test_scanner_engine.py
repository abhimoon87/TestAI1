"""Unit tests for scanner.scanner_engine — trend-filter / rating interplay."""

import threading
import time

from scanner.scanner_engine import _enrich_rows_in_place, rating_ok_for_trend_filter


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
