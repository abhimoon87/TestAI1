"""Unit tests for scanner.indicators — all technical indicator functions.

These tests verify:
  - Output shape and NaN semantics match pandas rolling() conventions
  - Known mathematical properties (e.g. RSI ∈ [0,100], ATR > 0)
  - Vectorized implementations produce identical results to reference methods
  - Edge cases: constant series, very short input, NaN in input
"""

import math

import numpy as np
import pandas as pd
import pytest

from scanner.indicators import (
    _wma_vectorized,
    hull_ma,
    ema,
    sma,
    vwma,
    kama,
    rsi,
    macd,
    stochastic,
    obv,
    atr,
    adx,
    price_change,
    highest,
    lowest,
    volume_profile_poc,
)


# ══════════════════════════════════════════════════════════════════════════════
# Moving Averages
# ══════════════════════════════════════════════════════════════════════════════


class TestWmaVectorized:
    """Tests for the vectorized WMA helper used by hull_ma."""

    def test_matches_pandas_rolling(self):
        """_wma_vectorized should produce identical values to rolling().apply."""
        rng = np.random.RandomState(99)
        s = pd.Series(rng.randn(500).cumsum() + 100)

        for length in [5, 10, 22, 44]:
            new = _wma_vectorized(s, length)
            ref = s.rolling(length).apply(
                lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True
            )
            # Compare only at positions where BOTH are non-NaN
            both_valid = new.notna() & ref.notna()
            assert both_valid.sum() > 0, "No overlapping valid values"
            np.testing.assert_allclose(
                new[both_valid].values, ref[both_valid].values, atol=1e-10
            )

    def test_nan_positions(self):
        """First n-1 positions should be NaN (insufficient lookback)."""
        s = pd.Series(range(20), dtype=float)
        result = _wma_vectorized(s, 5)
        assert result.iloc[:4].isna().all(), "First n-1 should be NaN"
        assert result.iloc[4:].notna().all(), "Rest should be valid"

    def test_constant_series(self):
        """WMA of a constant series should equal that constant."""
        s = pd.Series([42.0] * 50)
        result = _wma_vectorized(s, 10)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 42.0, atol=1e-12)

    def test_length_exceeds_series(self):
        """When length > len(series), all output should be NaN."""
        s = pd.Series(range(5), dtype=float)
        result = _wma_vectorized(s, 10)
        assert result.isna().all()

    def test_output_length_matches_input(self):
        s = pd.Series(range(100), dtype=float)
        result = _wma_vectorized(s, 10)
        assert len(result) == len(s)


class TestHullMa:
    """Tests for the Hull Moving Average."""

    def test_matches_pandas_reference(self, synthetic_ohlcv):
        """hull_ma should match the rolling().apply reference within tolerance."""
        close = synthetic_ohlcv["close"]
        for length in [22, 44]:
            new = hull_ma(close, length)
            half = int(length / 2)
            sqrt_len = int(math.sqrt(length))
            wma_half = close.rolling(half).apply(
                lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True
            )
            wma_full = close.rolling(length).apply(
                lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True
            )
            diff = 2 * wma_half - wma_full
            ref = diff.rolling(sqrt_len).apply(
                lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True
            )
            both_valid = new.notna() & ref.notna()
            assert both_valid.sum() > 0
            np.testing.assert_allclose(
                new[both_valid].values, ref[both_valid].values, atol=1e-8
            )

    def test_constant_series(self):
        """HMA of constant series = that constant."""
        s = pd.Series([150.0] * 200)
        result = hull_ma(s, 44)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 150.0, atol=1e-10)

    def test追随_trend(self, trending_up_ohlcv):
        """HMA should be below close in a steady uptrend (it lags)."""
        close = trending_up_ohlcv["close"]
        hma = hull_ma(close, 44)
        # HMA lags, so it should be below close in a steady uptrend
        valid_mask = hma.notna()
        # Align: compare each valid HMA value with the close at the same index
        common_idx = hma[valid_mask].index
        assert (hma[common_idx].values < close[common_idx].values).all()

    def test_output_length(self, synthetic_ohlcv):
        s = synthetic_ohlcv["close"]
        result = hull_ma(s, 44)
        assert len(result) == len(s)


class TestEma:
    def test_matches_pandas(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"]
        result = ema(close, 20)
        expected = close.ewm(span=20, adjust=False).mean()
        np.testing.assert_allclose(result.values, expected.values, atol=1e-12)

    def test_constant_series(self):
        s = pd.Series([100.0] * 50)
        result = ema(s, 10)
        np.testing.assert_allclose(result.values, 100.0, atol=1e-12)

    def test_no_nans(self, synthetic_ohlcv):
        """EMA should have no NaN values (it uses all available data)."""
        result = ema(synthetic_ohlcv["close"], 20)
        assert result.notna().all()


class TestSma:
    def test_matches_pandas(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"]
        result = sma(close, 20)
        expected = close.rolling(20).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_nan_positions(self):
        s = pd.Series(range(20), dtype=float)
        result = sma(s, 5)
        assert result.iloc[:4].isna().all()
        assert result.iloc[4:].notna().all()


class TestVwma:
    def test_equal_volume_equals_sma(self, synthetic_ohlcv):
        """When volume is constant, VWMA should equal SMA."""
        close = synthetic_ohlcv["close"]
        vol = pd.Series(1.0, index=close.index)
        v = vwma(close, vol, 20)
        s = sma(close, 20)
        both_valid = v.notna() & s.notna()
        np.testing.assert_allclose(v[both_valid].values, s[both_valid].values, atol=1e-10)


class TestKama:
    def test_matches_reference(self):
        """kama should match the original for-loop reference."""
        rng = np.random.RandomState(77)
        s = pd.Series(rng.randn(300).cumsum() + 500)

        # Reference implementation
        length = 50
        fast_alpha = 2.0 / 3
        slow_alpha = 2.0 / 31
        ref = s.copy()
        k = np.nan
        for i in range(length, len(s)):
            if np.isnan(k):
                k = s.iloc[i]
            else:
                mom = abs(s.iloc[i] - s.iloc[i - length])
                vol = s.iloc[i - length : i + 1].diff().abs().sum()
                er = mom / vol if vol != 0 else 0
                sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
                k = k + sc * (s.iloc[i] - k)
            ref.iloc[i] = k

        new = kama(s, length)
        both_valid = new.notna() & ref.notna()
        np.testing.assert_allclose(new[both_valid].values, ref[both_valid].values, atol=1e-10)

    def test_constant_series(self):
        """KAMA of constant series should equal that constant."""
        s = pd.Series([200.0] * 200)
        result = kama(s, 50)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 200.0, atol=1e-10)

    def test_nan_positions(self):
        """First `length` positions should be NaN."""
        s = pd.Series(range(100), dtype=float)
        result = kama(s, 30)
        assert result.iloc[:29].isna().all()
        assert result.iloc[30:].notna().all()


# ══════════════════════════════════════════════════════════════════════════════
# Oscillators
# ══════════════════════════════════════════════════════════════════════════════


class TestRsi:
    def test_range(self, synthetic_ohlcv):
        """RSI must always be in [0, 100]."""
        result = rsi(synthetic_ohlcv["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_constant_series(self):
        """Constant series → RSI = 50 (or NaN due to zero loss)."""
        s = pd.Series([100.0] * 50)
        result = rsi(s, 14)
        # With zero gains AND zero losses, RSI can be NaN (0/0)
        # or 50 depending on implementation. Either is acceptable.
        valid = result.dropna()
        if len(valid) > 0:
            np.testing.assert_allclose(valid.values, 50.0, atol=1e-10)

    def test_strong_uptrend_high_rsi(self, rng):
        """Strong uptrend should push RSI toward 100."""
        prices = 100 + np.arange(50) * 2  # strictly increasing
        s = pd.Series(prices, dtype=float)
        result = rsi(s, 14)
        # After warmup, RSI should be very high
        assert result.iloc[-1] > 80

    def test_matches_pandas(self, synthetic_ohlcv):
        """Should match the standard pandas RSI implementation."""
        close = synthetic_ohlcv["close"]
        length = 14
        result = rsi(close, length)

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
        avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
        rs = avg_gain / avg_loss
        expected = 100 - (100 / (1 + rs))

        np.testing.assert_allclose(result.dropna().values, expected.dropna().values, atol=1e-10)


class TestMacd:
    def test_returns_three_series(self, synthetic_ohlcv):
        macd_line, signal, hist = macd(synthetic_ohlcv["close"])
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)

    def test_histogram_equals_difference(self, synthetic_ohlcv):
        macd_line, signal, hist = macd(synthetic_ohlcv["close"])
        np.testing.assert_allclose(hist.values, (macd_line - signal).values, atol=1e-12)

    def test_constant_series(self):
        """MACD of constant series → all zeros."""
        s = pd.Series([100.0] * 100)
        _, _, hist = macd(s)
        np.testing.assert_allclose(hist.dropna().values, 0.0, atol=1e-10)


class TestStochastic:
    def test_range(self, synthetic_ohlcv):
        """Stochastic %K should be in [0, 100]."""
        df = synthetic_ohlcv
        result = stochastic(df["high"], df["low"], df["close"])
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_constant_ohlcv(self):
        """When high == low == close, stochastic is 0/0 = NaN."""
        n = 50
        s = pd.Series(np.full(n, 100.0))
        result = stochastic(s, s, s)
        # Division by zero → NaN
        assert result.iloc[16:].isna().all() or (result.dropna() == 50).all()


class TestObv:
    def test_monotonic_on_uptrend(self, trending_up_ohlcv):
        """OBV should increase monotonically on a pure uptrend."""
        df = trending_up_ohlcv
        result = obv(df["close"], df["volume"])
        diff = result.diff()
        # After first bar, all increments should be positive
        assert (diff.iloc[1:] >= 0).all()

    def test_monotonic_on_downtrend(self, trending_down_ohlcv):
        """OBV should decrease monotonically on a pure downtrend."""
        df = trending_down_ohlcv
        result = obv(df["close"], df["volume"])
        diff = result.diff()
        assert (diff.iloc[1:] <= 0).all()

    def test_no_nans(self, synthetic_ohlcv):
        result = obv(synthetic_ohlcv["close"], synthetic_ohlcv["volume"])
        assert result.notna().all()


class TestAtr:
    def test_always_positive(self, synthetic_ohlcv):
        """ATR should always be positive."""
        df = synthetic_ohlcv
        result = atr(df["high"], df["low"], df["close"], 14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_constant_ohlcv(self):
        """Constant OHLV → ATR = 0 (no price movement)."""
        n = 50
        h = pd.Series(np.full(n, 101.0))
        l = pd.Series(np.full(n, 99.0))
        c = pd.Series(np.full(n, 100.0))
        result = atr(h, l, c, 14)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 2.0, atol=1e-10)


class TestAdx:
    def test_range(self, synthetic_ohlcv):
        """ADX should be in [0, 100]."""
        df = synthetic_ohlcv
        result = adx(df["high"], df["low"], df["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_trending_has_higher_adx(self, trending_up_ohlcv):
        """A strong trend should produce higher ADX than random data."""
        df = trending_up_ohlcv
        result = adx(df["high"], df["low"], df["close"], 14)
        valid = result.dropna()
        # Strong linear trend → ADX should be well above 20
        assert valid.iloc[-1] > 20


# ══════════════════════════════════════════════════════════════════════════════
# Derived Metrics
# ══════════════════════════════════════════════════════════════════════════════


class TestPriceChange:
    def test_known_values(self):
        s = pd.Series([100.0, 110.0, 121.0])
        result = price_change(s, 1)
        assert result.iloc[0] is np.nan or np.isnan(result.iloc[0])
        np.testing.assert_allclose(result.iloc[1], 10.0, atol=1e-10)
        np.testing.assert_allclose(result.iloc[2], 10.0, atol=1e-10)


class TestHighestLowest:
    def test_highest(self):
        s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
        result = highest(s, 3)
        assert result.iloc[2] == 3.0
        assert result.iloc[3] == 5.0
        assert result.iloc[4] == 5.0

    def test_lowest(self):
        s = pd.Series([5.0, 3.0, 4.0, 1.0, 2.0])
        result = lowest(s, 3)
        assert result.iloc[2] == 3.0
        assert result.iloc[3] == 1.0
        assert result.iloc[4] == 1.0


class TestVolumeProfilePoc:
    def test_poc_within_price_range(self, synthetic_ohlcv):
        """POC should always be between low and high of the lookback window."""
        df = synthetic_ohlcv
        poc = volume_profile_poc(df["high"], df["low"], df["close"], df["volume"], lookback=50)
        valid = poc.dropna()
        rolling_low = df["low"].rolling(50).min()
        rolling_high = df["high"].rolling(50).max()
        both_valid = valid.index.intersection(rolling_low.dropna().index)
        assert (valid[both_valid] >= rolling_low[both_valid]).all()
        assert (valid[both_valid] <= rolling_high[both_valid]).all()

    def test_short_lookback(self, synthetic_ohlcv):
        """POC with small lookback should still produce valid values."""
        df = synthetic_ohlcv
        poc = volume_profile_poc(df["high"], df["low"], df["close"], df["volume"], lookback=10)
        # At least some values should be non-NaN
        assert poc.notna().sum() > 0
