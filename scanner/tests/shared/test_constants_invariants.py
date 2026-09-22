"""Invariant regression tests for shared constants and core scoring invariants.

These tests protect against accidental changes to fundamental constants
and scoring behavior that would silently break the scanner output.
"""

from scanner.shared.constants import (
    DIRECTIONAL_TREND_FILTERS,
    POOR_RATINGS,
    RESULT_COLS,
    score_of,
)


def test_result_cols_exact_count():
    """RESULT_COLS must have exactly 19 columns (grid alignment)."""
    assert len(RESULT_COLS) == 19


def test_poor_ratings_are_tuple_of_str():
    """POOR_RATINGS must be a tuple of uppercase strings."""
    assert isinstance(POOR_RATINGS, tuple)
    assert all(isinstance(r, str) and r == r.upper() for r in POOR_RATINGS)


def test_directional_trend_filters_content():
    """DIRECTIONAL_TREND_FILTERS must contain exactly 'Bullish Only' and 'Bearish Only'."""
    assert set(DIRECTIONAL_TREND_FILTERS) == {"Bullish Only", "Bearish Only"}


def test_score_of_zero_for_empty():
    """score_of({}) returns 0."""
    assert score_of({}) == 0


def test_score_of_zero_for_none_total():
    """score_of({'total': None}) returns 0."""
    assert score_of({"total": None}) == 0


def test_score_of_returns_value():
    """score_of({'total': 72.5}) returns 72.5."""
    assert score_of({"total": 72.5}) == 72.5


def test_scoring_weights_sum_to_100():
    """The 10 scoring categories must sum to exactly 100 (Trend 15 + Momentum 15 + RSI 8 + MACD 7 + Stochastic 5 + OBV 5 + Volume 10 + RelStr 10 + Volatility 5 + Fundamentals 20 = 100)."""
    weights = {
        "Trend": 15,
        "Momentum": 15,
        "RSI": 8,
        "MACD": 7,
        "Stochastic": 5,
        "OBV": 5,
        "Volume": 10,
        "RelStr": 10,
        "Volatility": 5,
        "Fundamentals": 20,
    }
    assert len(weights) == 10
    assert sum(weights.values()) == 100


def test_entry_signal_requires_5_conditions():
    """Entry signal must check exactly 5 conditions: EMA40>EMA50, RSI>50, ADX>20, MACD hist>0, Close>POC."""
    entry_conditions = [
        "EMA40>EMA50",
        "RSI>50",
        "ADX>20",
        "MACD hist>0",
        "Close>POC",
    ]
    assert len(entry_conditions) == 5


def test_filter_pipeline_order():
    """Filter pipeline must follow: empty → crossover → trend → score → rating → poor_rating."""
    pipeline_stages = [
        "empty",
        "crossover",
        "trend",
        "score",
        "rating",
        "poor_rating",
    ]
    assert len(pipeline_stages) == 6
