"""Invariant regression tests for shared constants and core scoring invariants.

These tests protect against accidental changes to fundamental constants
and scoring behavior that would silently break the scanner output.
"""

from scanner.shared.constants import (
    DIRECTIONAL_TREND_FILTERS,
    POOR_RATINGS,
    RESULT_COLS,
    TREND_FILTERS,
    score_of,
)


def test_result_cols_exact_count():
    """RESULT_COLS must have exactly 23 columns (grid alignment)."""
    assert len(RESULT_COLS) == 23


def test_poor_ratings_are_tuple_of_str():
    """POOR_RATINGS must be a tuple of uppercase strings."""
    assert isinstance(POOR_RATINGS, tuple)
    assert all(isinstance(r, str) and r == r.upper() for r in POOR_RATINGS)


def test_directional_trend_filters_content():
    """DIRECTIONAL_TREND_FILTERS must be exactly the three directional options."""
    assert set(DIRECTIONAL_TREND_FILTERS) == {
        "Bullish Only",
        "Bearish Only",
        "Bullish + Candle",
    }


def test_trend_filters_superset_of_directional():
    """Every directional filter must be a valid dropdown/CLI value."""
    assert set(DIRECTIONAL_TREND_FILTERS) <= set(TREND_FILTERS)


def test_score_of_zero_for_empty():
    """score_of({}) returns 0."""
    assert score_of({}) == 0


def test_score_of_zero_for_none_total():
    """score_of({'total': None}) returns 0."""
    assert score_of({"total": None}) == 0


def test_score_of_returns_value():
    """score_of({'total': 72.5}) returns 72.5."""
    assert score_of({"total": 72.5}) == 72.5
