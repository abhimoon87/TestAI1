"""Unit tests for scanner.scanner_engine — trend-filter / rating interplay."""

from scanner.scanner_engine import rating_ok_for_trend_filter


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
