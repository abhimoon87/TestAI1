"""
Technical indicator calculations — Python/NumPy ports of the indicators
used by the HMAxEMA Pine Script scoring engine (HMA, EMA, SMA, KAMA,
VWMA, RSI, MACD, stochastic, OBV, ATR, ADX, volume-profile POC).

Where the Pine file defines a series function, the port aims to match it
(e.g. Wilder RSI/ATR smoothing and HMA via layered WMA); these are
vectorized so results may differ only in warm-up behaviour.
"""

import numpy as np
import pandas as pd

# ── Moving Averages ─────────────────────────────────────────────────────────

def _wma_vectorized(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average using vectorized numpy convolution.

    WMA(x, n) = Σ(x[i] × (i+1)) / Σ(1..n) for each rolling window.
    np.convolve(x, w, mode='full')[j] = Σ x[k]·w[j−k].
    With reversed weights [n, n−1, …, 1], element j gives
    1·x[j−n+1] + 2·x[j−n+2] + … + n·x[j] — heaviest on the newest bar,
    which matches the standard WMA definition.

    The convolution output wma_vals[i] is the WMA for the window ending at
    original series index (n-1+i). We place it at output index (n-1+i) to
    match pandas rolling() semantics, and NaN the first n-1 positions.
    """
    vals = series.values.astype(np.float64)
    n = length
    N = len(vals)
    weights = np.arange(n, 0, -1, dtype=np.float64)  # reversed: [n, n-1, ..., 1]
    norm = n * (n + 1) / 2.0

    full_conv = np.convolve(vals, weights, mode='full')
    conv_vals = full_conv[n - 1:] / norm  # N values: conv_vals[i] = WMA ending at index n-1+i

    # Build output: NaN first n-1 positions, valid from n-1 onward.
    # conv_vals[0..N-n] → output[n-1..N-1]
    # conv_vals[N-n+1..N-1] → tail (would need data past end of series) → NaN
    result = np.full(N, np.nan, dtype=np.float64)
    valid_count = N - n + 1
    if valid_count > 0:
        result[n - 1:n - 1 + valid_count] = conv_vals[:valid_count]

    return pd.Series(result, index=series.index, name=series.name)


def hull_ma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average (vectorized).

    Formula: WMA(2 × WMA(close, n/2) − WMA(close, n), √n)
    All three WMA layers use vectorized convolution — no Python per-window loops.
    """
    half = int(length / 2)
    sqrt_len = int(np.sqrt(length))

    wma_half = _wma_vectorized(series, half)
    wma_full = _wma_vectorized(series, length)
    diff = 2 * wma_half - wma_full
    return _wma_vectorized(diff, sqrt_len)


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(length).mean()


def vwma(series: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """Volume Weighted Moving Average."""
    return (series * volume).rolling(length).sum() / volume.rolling(length).sum()


def kama(series: pd.Series, length: int, fast_length: int = 2, slow_length: int = 30) -> pd.Series:
    """Kaufman's Adaptive Moving Average (vectorized).

    Uses pandas rolling operations and np.where for the adaptive smoothing,
    replacing the per-bar Python for-loop with vectorized batch computation.
    The recursive KAMA(k) = k + sc × (price − k) is handled by a single
    numba-free loop over the smoothing constant array.
    """
    vals = series.values.astype(np.float64)
    n = len(vals)
    result = np.full(n, np.nan, dtype=np.float64)

    fast_alpha = 2.0 / (fast_length + 1)
    slow_alpha = 2.0 / (slow_length + 1)

    # ── Momentum: |price[i] − price[i−length]| (vectorized) ──
    shifted = np.empty(n, dtype=np.float64)
    shifted[:length] = np.nan
    shifted[length:] = vals[:n - length]
    mom = np.abs(vals - shifted)

    # ── Volatility: Σ|diff| over rolling window of length+1 (vectorized) ──
    s = pd.Series(vals)
    volatility = s.diff().abs().rolling(length, min_periods=length).sum().values

    # ── Efficiency Ratio: mom / volatility ──
    # np.where still evaluates the division eagerly, so silence the
    # 0/0 -> nan divide warnings; those positions are discarded anyway.
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(volatility > 0, mom / volatility, 0.0)

    # ── Smoothing Constant: (er × (fast − slow) + slow)² ──
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2

    # ── Recursive EMA: k[i] = k[i−1] + sc[i] × (price[i] − k[i−1]) ──
    # First valid KAMA = first price with enough lookback
    k = np.nan
    for i in range(length, n):
        if np.isnan(k):
            k = vals[i]
        else:
            k = k + sc[i] * (vals[i] - k)
        result[i] = k

    return pd.Series(result, index=series.index, name=series.name)


# ── Oscillators ─────────────────────────────────────────────────────────────

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD (returns macd_line, signal_line, histogram)."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_length: int = 14, d_length: int = 3) -> pd.Series:
    """Stochastic %K."""
    lowest = low.rolling(k_length).min()
    highest = high.rolling(k_length).max()
    k = 100 * (close - lowest) / (highest - lowest)
    return k.rolling(d_length).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    diff = close.diff()
    direction = np.sign(diff).fillna(0)
    return (volume * direction).cumsum()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average Directional Index."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0), 0)
    minus_dm = np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0), 0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr_val = tr.ewm(alpha=1 / length, min_periods=length).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / length, min_periods=length).mean() / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / length, min_periods=length).mean() / atr_val

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / length, min_periods=length).mean()


# ── Derived Metrics ─────────────────────────────────────────────────────────

def price_change(series: pd.Series, period: int) -> pd.Series:
    """Percentage price change over N periods."""
    return ((series - series.shift(period)) / series.shift(period)) * 100


def highest(series: pd.Series, length: int) -> pd.Series:
    """Rolling highest value."""
    return series.rolling(length).max()


def lowest(series: pd.Series, length: int) -> pd.Series:
    """Rolling lowest value."""
    return series.rolling(length).min()


def volume_profile_poc(high: pd.Series, low: pd.Series, close: pd.Series,
                       volume: pd.Series, lookback: int = 55) -> pd.Series:
    """
    Volume Profile Point of Control (POC) - the price level with highest volume.

    Vectorized implementation: O(n × num_bins) instead of O(n × lookback × num_bins).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        lookback: Number of bars to look back (default 55 = ~11 weeks for daily)

    Returns:
        Series with POC values for each bar
    """
    n = len(close)
    poc_series = pd.Series(index=close.index, dtype=float)
    
    high_arr = high.values.astype(float)
    low_arr = low.values.astype(float)
    vol_arr = volume.values.astype(float)
    close_arr = close.values.astype(float)
    
    # Pre-allocate rolling min/max arrays (vectorized via pandas)
    rolling_high = pd.Series(high_arr, index=close.index).rolling(lookback).max().values
    rolling_low = pd.Series(low_arr, index=close.index).rolling(lookback).min().values
    
    for i in range(lookback - 1, n):
        price_min = rolling_low[i]
        price_max = rolling_high[i]
        
        if np.isnan(price_min) or np.isnan(price_max) or price_max == price_min:
            poc_series.iloc[i] = price_min if not np.isnan(price_min) else close_arr[i]
            continue
        
        # Adaptive bin count (matches original logic)
        num_bins = min(20, int((price_max - price_min) / (close_arr[i] * 0.001)) + 1)
        num_bins = max(num_bins, 5)
        
        bin_edges = np.linspace(price_min, price_max, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Extract lookback window as numpy arrays (no iloc overhead)
        start = i - lookback + 1
        w_high = high_arr[start:i + 1]
        w_low = low_arr[start:i + 1]
        w_vol = vol_arr[start:i + 1]
        
        # Vectorized: compute overlap of all bars × all bins at once
        # Shapes: bars=(B,), bins=(N+1,)
        overlap_low = np.maximum(w_low[:, None], bin_edges[None, :-1])
        overlap_high = np.minimum(w_high[:, None], bin_edges[None, 1:])
        overlap_len = np.maximum(overlap_high - overlap_low, 0.0)
        
        bar_range = w_high - w_low
        # Avoid division by zero: replace 0 ranges with 1 (overlap will be 0 anyway)
        safe_range = np.where(bar_range > 0, bar_range, 1.0)
        overlap_pct = overlap_len / safe_range[:, None]
        
        # Distribute volume: (B, 1) × (B, N) → sum over bars → (N,)
        bin_volumes = (w_vol[:, None] * overlap_pct).sum(axis=0)
        
        poc_series.iloc[i] = bin_centers[np.argmax(bin_volumes)]
    
    return poc_series
