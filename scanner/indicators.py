"""
Technical indicator calculations.
Mirrors the indicators used in the Pine Script scoring engine.
"""

import numpy as np
import pandas as pd


# ── Moving Averages ─────────────────────────────────────────────────────────

def hull_ma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average."""
    half = int(length / 2)
    sqrt_len = int(np.sqrt(length))
    wma_half = series.rolling(half).apply(lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True)
    wma_full = series.rolling(length).apply(lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True)
    diff = 2 * wma_half - wma_full
    return diff.rolling(sqrt_len).apply(lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True)


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
    """Kaufman's Adaptive Moving Average."""
    fast_alpha = 2.0 / (fast_length + 1)
    slow_alpha = 2.0 / (slow_length + 1)

    result = series.copy()
    k = np.nan

    for i in range(length, len(series)):
        if np.isnan(k):
            k = series.iloc[i]
        else:
            mom = abs(series.iloc[i] - series.iloc[i - length])
            volatility = series.iloc[i - length:i + 1].diff().abs().sum()
            er = mom / volatility if volatility != 0 else 0
            sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
            k = k + sc * (series.iloc[i] - k)
        result.iloc[i] = k

    return result


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
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
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
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        lookback: Number of bars to look back (default 55 = ~11 weeks for daily)
    
    Returns:
        Series with POC values for each bar
    """
    poc_series = pd.Series(index=close.index, dtype=float)
    
    for i in range(lookback - 1, len(close)):
        # Get the lookback window
        window_high = high.iloc[i - lookback + 1:i + 1]
        window_low = low.iloc[i - lookback + 1:i + 1]
        window_volume = volume.iloc[i - lookback + 1:i + 1]
        
        # Create price bins
        price_min = window_low.min()
        price_max = window_high.max()
        
        if price_max == price_min:
            poc_series.iloc[i] = price_min
            continue
        
        # Create bins (20 bins typically)
        num_bins = min(20, int((price_max - price_min) / (close.iloc[i] * 0.001)) + 1)
        num_bins = max(num_bins, 5)
        
        bin_edges = np.linspace(price_min, price_max, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate volume for each bin
        bin_volumes = np.zeros(num_bins)
        
        for j in range(len(window_high)):
            bar_high = window_high.iloc[j]
            bar_low = window_low.iloc[j]
            bar_vol = window_volume.iloc[j]
            
            # Distribute volume across bins that overlap with this bar
            for k in range(num_bins):
                bin_low = bin_edges[k]
                bin_high = bin_edges[k + 1]
                
                # Check if bar overlaps with this bin
                if bar_high >= bin_low and bar_low <= bin_high:
                    # Calculate overlap percentage
                    overlap_low = max(bar_low, bin_low)
                    overlap_high = min(bar_high, bin_high)
                    bar_range = bar_high - bar_low
                    
                    if bar_range > 0:
                        overlap_pct = (overlap_high - overlap_low) / bar_range
                        bin_volumes[k] += bar_vol * overlap_pct
        
        # Find the bin with highest volume (POC)
        poc_bin_idx = np.argmax(bin_volumes)
        poc_series.iloc[i] = bin_centers[poc_bin_idx]
    
    return poc_series
