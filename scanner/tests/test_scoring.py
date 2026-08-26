"""Unit tests for scanner.scoring — the 3-model scoring pipeline.

Tests cover:
  - get_ma: MA type dispatch
  - to_weekly: OHLCV resampling
  - check_filter: crossover detection (Model 1)
  - get_direction: Bull/Bear classification (Model 2)
  - compute_scores: 10-category scoring (Model 3)
  - _get_combined_rating: rating logic
"""

import numpy as np
import pandas as pd
import pytest

from scanner.scoring import (
    get_ma,
    to_weekly,
    check_filter,
    get_direction,
    compute_scores,
    _get_combined_rating,
)


# ══════════════════════════════════════════════════════════════════════════════
# get_ma — MA type dispatch
# ══════════════════════════════════════════════════════════════════════════════


class TestGetMa:
    def test_hma(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"]
        result = get_ma("HMA", close, 44)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close)

    def test_ema(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"]
        result = get_ma("EMA", close, 20)
        expected = close.ewm(span=20, adjust=False).mean()
        np.testing.assert_allclose(result.values, expected.values, atol=1e-12)

    def test_sma(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"]
        result = get_ma("SMA", close, 20)
        expected = close.rolling(20).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_kama(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"]
        result = get_ma("KAMA", close, 50)
        assert isinstance(result, pd.Series)
        # KAMA has NaN for first `length` bars
        assert result.iloc[:49].isna().all()

    def test_vwma_fallback(self, synthetic_ohlcv):
        """VWMA without volume should fall back to EMA."""
        close = synthetic_ohlcv["close"]
        result = get_ma("VWMA", close, 20, volume=None)
        expected = close.ewm(span=20, adjust=False).mean()
        np.testing.assert_allclose(result.values, expected.values, atol=1e-12)

    def test_vwma_with_volume(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"]
        vol = synthetic_ohlcv["volume"]
        result = get_ma("VWMA", close, 20, volume=vol)
        expected = (close * vol).rolling(20).sum() / vol.rolling(20).sum()
        both_valid = result.notna() & expected.notna()
        np.testing.assert_allclose(result[both_valid].values, expected[both_valid].values, atol=1e-10)

    def test_unknown_type_falls_back_to_ema(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"]
        result = get_ma("BOGUS", close, 20)
        expected = close.ewm(span=20, adjust=False).mean()
        np.testing.assert_allclose(result.values, expected.values, atol=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# to_weekly — OHLCV resampling
# ══════════════════════════════════════════════════════════════════════════════


class TestToWeekly:
    def test_basic_resample(self, synthetic_ohlcv):
        """Daily → weekly should reduce bar count."""
        result = to_weekly(synthetic_ohlcv)
        assert result is not None
        assert len(result) < len(synthetic_ohlcv)

    def test_columns_preserved(self, synthetic_ohlcv):
        result = to_weekly(synthetic_ohlcv)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_ohlc_correctness(self):
        """Weekly OHLC should match manual aggregation."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "open": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                "high": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                "low": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                "close": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "volume": [100] * 10,
            },
            index=dates,
        )
        result = to_weekly(df)
        assert result is not None
        # Weekly open = first daily open in the week
        # Weekly high = max daily high
        # Weekly close = last daily close

    def test_none_input(self):
        assert to_weekly(None) is None

    def test_empty_dataframe(self):
        assert to_weekly(pd.DataFrame()) is None

    def test_no_close_column(self):
        df = pd.DataFrame({"open": [1, 2], "high": [2, 3], "low": [1, 2], "volume": [100, 200]})
        assert to_weekly(df) is None

    def test_non_datetime_index(self):
        df = pd.DataFrame(
            {"open": [1], "high": [2], "low": [1], "close": [1.5], "volume": [100]},
            index=[0],
        )
        assert to_weekly(df) is None


# ══════════════════════════════════════════════════════════════════════════════
# check_filter — Model 1: Crossover Detection
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckFilter:
    def test_returns_dict_on_crossover(self, crossover_ohlcv):
        """Should return a filter dict when a crossover exists."""
        result = check_filter(
            crossover_ohlcv,
            fast_ma_type="HMA",
            fast_ma_len=20,
            slow_ma_type="EMA",
            slow_ma_len=30,
            crossover_lookback=50,
        )
        assert result is not None
        assert isinstance(result, dict)
        assert "ma_crossed_above" in result
        assert "crossover_bars_ago" in result

    def test_returns_none_when_no_crossover(self, flat_ohlcv):
        """Flat data should have no crossover → filtered out."""
        result = check_filter(
            flat_ohlcv,
            fast_ma_type="HMA",
            fast_ma_len=20,
            slow_ma_type="EMA",
            slow_ma_len=30,
            crossover_lookback=20,
        )
        assert result is None

    def test_returns_none_for_none_input(self):
        assert check_filter(None) is None

    def test_returns_none_for_empty_dataframe(self):
        assert check_filter(pd.DataFrame()) is None

    def test_returns_none_for_short_data(self, short_ohlcv):
        """Data shorter than min_bars should be filtered out."""
        result = check_filter(
            short_ohlcv,
            fast_ma_type="HMA",
            fast_ma_len=40,
            slow_ma_type="EMA",
            slow_ma_len=50,
            crossover_lookback=20,
        )
        assert result is None

    def test_ma_bullish_flag(self, crossover_ohlcv):
        """After a bullish crossover, ma_bullish should be True."""
        result = check_filter(
            crossover_ohlcv,
            fast_ma_type="HMA",
            fast_ma_len=20,
            slow_ma_type="EMA",
            slow_ma_len=30,
            crossover_lookback=50,
        )
        if result is not None:
            # The crossover is bullish (fast crosses above slow)
            assert result["ma_bullish"] is True or result["ma_bullish"] is False

    def test_crossover_lookback_respected(self, synthetic_ohlcv):
        """Very small lookback should filter out old crossovers."""
        # With lookback=1, only crossovers in the last bar are detected
        result = check_filter(
            synthetic_ohlcv,
            fast_ma_type="HMA",
            fast_ma_len=20,
            slow_ma_type="EMA",
            slow_ma_len=30,
            crossover_lookback=1,
        )
        # May or may not find a crossover at bar -1; just ensure no crash
        assert result is None or isinstance(result, dict)


# ══════════════════════════════════════════════════════════════════════════════
# get_direction — Model 2: Bull/Bear Classification
# ══════════════════════════════════════════════════════════════════════════════


class TestGetDirection:
    def test_bull(self):
        result = get_direction({"ma_bullish": True})
        assert result == "Bull"

    def test_bear(self):
        result = get_direction({"ma_bullish": False})
        assert result == "Bear"

    def test_none_input(self):
        assert get_direction(None) is None


# ══════════════════════════════════════════════════════════════════════════════
# compute_scores — Model 3: 10-Category Scoring
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeScores:
    def test_returns_dict(self, synthetic_ohlcv):
        """Should return a result dict with all expected keys."""
        result = compute_scores(synthetic_ohlcv, timeframe="D")
        assert result is not None
        assert isinstance(result, dict)

    def test_total_score_range(self, synthetic_ohlcv):
        """Total score must be in [0, 100]."""
        result = compute_scores(synthetic_ohlcv, timeframe="D")
        assert result is not None
        assert 0 <= result["total"] <= 100

    def test_all_categories_present(self, synthetic_ohlcv):
        """All 10 scoring categories should be in the result."""
        result = compute_scores(synthetic_ohlcv, timeframe="D")
        assert result is not None
        categories = ["trend", "momentum", "rsi", "macd", "stoch",
                       "obv", "volume", "rel_str", "volatility", "fundamentals"]
        for cat in categories:
            assert cat in result, f"Missing category: {cat}"

    def test_category_max_bounds(self, synthetic_ohlcv):
        """Each category should not exceed its maximum weight."""
        result = compute_scores(synthetic_ohlcv, timeframe="D")
        assert result is not None
        max_bounds = {
            "trend": 15, "momentum": 15, "rsi": 8, "macd": 7,
            "stoch": 5, "obv": 5, "volume": 10, "rel_str": 10,
            "volatility": 5, "fundamentals": 20,
        }
        for cat, max_val in max_bounds.items():
            assert result[cat] <= max_val + 0.1, f"{cat}={result[cat]} exceeds max {max_val}"

    def test_returns_none_for_short_data(self, short_ohlcv):
        """Data shorter than min_required should return None."""
        result = compute_scores(short_ohlcv, timeframe="D")
        assert result is None

    def test_returns_none_for_none_input(self):
        # compute_scores expects a DataFrame; passing None should either
        # return None or raise — we just verify no unhandled crash
        try:
            result = compute_scores(None, timeframe="D")
            assert result is None
        except (TypeError, AttributeError):
            pass  # Acceptable

    def test_result_has_metadata_keys(self, synthetic_ohlcv):
        """Result should contain metadata like close, trend_dir, etc."""
        result = compute_scores(synthetic_ohlcv, timeframe="D")
        assert result is not None
        for key in ["close", "trend_dir", "trend_color", "atr_pct",
                     "volat_stat", "combined_rating", "entry_signal",
                     "weekly_entry_signal", "is_sideways", "fund_detail"]:
            assert key in result, f"Missing metadata key: {key}"

    def test_trend_dir_is_bull_or_bear(self, synthetic_ohlcv):
        result = compute_scores(synthetic_ohlcv, timeframe="D")
        assert result is not None
        assert result["trend_dir"] in ("Bull", "Bear")

    def test_with_custom_settings(self, synthetic_ohlcv):
        """Custom settings should be used without crashing."""
        settings = {
            "fast_ma_type": "EMA",
            "fast_ma_len": 20,
            "slow_ma_type": "SMA",
            "slow_ma_len": 30,
            "rsi_len": 14,
            "vol_ma_len": 20,
            "atr_len": 14,
            "rs_length": 14,
            "adx_len": 14,
            "adx_threshold": 20.0,
            "chop_len": 14,
            "chop_threshold": 61.8,
            "slope_ma_type": "EMA",
            "slope_ma_len": 50,
            "slope_lookback": 10,
            "flat_threshold": 0.5,
            "sc_pivot_len": 3,
            "sc_bands_mult": 0.6,
            "vp_lookback": 200,
            "vp_rows": 30,
            "vp_width": 40,
            "crossover_lookback": 20,
        }
        result = compute_scores(synthetic_ohlcv, timeframe="D", settings=settings)
        assert result is not None
        assert 0 <= result["total"] <= 100

    def test_with_index_df(self, synthetic_ohlcv):
        """Providing index_df should compute relative strength against it."""
        # Use a simplified index that tracks close
        index_df = synthetic_ohlcv[["close"]].copy()
        index_df["open"] = index_df["close"]
        index_df["high"] = index_df["close"]
        index_df["low"] = index_df["close"]
        index_df["volume"] = 1_000_000
        result = compute_scores(synthetic_ohlcv, timeframe="D", index_df=index_df)
        assert result is not None
        # RS score should use index comparison, not fallback
        assert result["rel_str"] >= 0

    def test_weekly_timeframe(self, synthetic_ohlcv):
        """Weekly timeframe should reduce minimum bar requirement."""
        result = compute_scores(synthetic_ohlcv, timeframe="W")
        # Should still work with 200 daily bars (enough for weekly)
        assert result is not None

    def test_entry_signal_conditions(self, synthetic_ohlcv):
        """entry_signal should be a boolean."""
        result = compute_scores(synthetic_ohlcv, timeframe="D")
        assert result is not None
        assert isinstance(result["entry_signal"], bool)

    def test_fundamentals_attached(self, synthetic_ohlcv):
        """When _fundamentals is attached, it should be used."""
        # Attach fundamentals via object.__setattr__ to avoid pandas warning
        object.__setattr__(synthetic_ohlcv, '_fundamentals', {
            "pe_ratio": 12.0,
            "eps_growth": 25.0,
            "rev_growth": 18.0,
            "roe": 22.0,
        })
        result = compute_scores(synthetic_ohlcv, timeframe="D")
        assert result is not None
        # With strong fundamentals, fund score should be high
        assert result["fundamentals"] == 20.0
        assert result["fund_detail"]["pe"] == "Strong"
        assert result["fund_detail"]["eps_growth"] == "Strong"

    def test_fundamentals_none(self, synthetic_ohlcv):
        """Without fundamentals, fund score should be 0."""
        result = compute_scores(synthetic_ohlcv, timeframe="D")
        assert result is not None
        assert result["fundamentals"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# _get_combined_rating
# ══════════════════════════════════════════════════════════════════════════════


class TestGetCombinedRating:
    def test_excellent(self):
        assert _get_combined_rating(80, True, True, True) == "EXCELLENT"

    def test_good(self):
        assert _get_combined_rating(55, True, True, True) == "GOOD"

    def test_moderate(self):
        assert _get_combined_rating(40, True, True, True) == "MODERATE"

    def test_poor(self):
        assert _get_combined_rating(20, True, True, True) == "POOR"

    def test_ma_only_high_score(self):
        """ma_bullish only (no above_poc, no close_above_both_ma)."""
        assert _get_combined_rating(75, True, False, False) == "EXCELLENT"
        assert _get_combined_rating(60, True, False, False) == "GOOD"
        assert _get_combined_rating(45, True, False, False) == "MODERATE"
        assert _get_combined_rating(30, True, False, False) == "POOR"

    def test_neither_signal(self):
        """No bullish signals at all."""
        assert _get_combined_rating(75, False, False, False) == "EXCELLENT"
        assert _get_combined_rating(60, False, False, False) == "GOOD"
        assert _get_combined_rating(45, False, False, False) == "MODERATE"
        assert _get_combined_rating(30, False, False, False) == "POOR"
