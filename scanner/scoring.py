"""
10-category scoring engine.
Mirrors the Pine Script HMAxEMA Swing Trading System scoring logic exactly.
"""

import numpy as np
import pandas as pd
import math
from .indicators import (
    hull_ma, ema, sma, vwma, kama, rsi, macd, stochastic,
    obv, atr, adx, price_change, highest, lowest, volume_profile_poc
)


def get_ma(ma_type: str, src: pd.Series, length: int,
           volume: pd.Series = None) -> pd.Series:
    """Universal MA selector matching the Pine Script get_ma function."""
    if ma_type == "HMA":
        return hull_ma(src, length)
    elif ma_type == "EMA":
        return ema(src, length)
    elif ma_type == "SMA":
        return sma(src, length)
    elif ma_type == "KAMA":
        return kama(src, length)
    elif ma_type == "VWMA":
        if volume is not None and not volume.empty:
            return vwma(src, volume, length)
        return ema(src, length)  # fallback if no volume
    return ema(src, length)


def compute_scores(df: pd.DataFrame, index_df: pd.DataFrame = None,
                  fast_ma_type: str = "HMA", fast_ma_len: int = 40,
                  slow_ma_type: str = "EMA", slow_ma_len: int = 50,
                   rsi_len: int = 14, vol_ma_len: int = 20,
                   atr_len: int = 14, rs_length: int = 14,
                   adx_len: int = 14, adx_threshold: float = 20.0,
                   chop_len: int = 14, chop_threshold: float = 61.8,
                   slope_ma_type: str = "EMA", slope_ma_len: int = 50,
                   slope_lookback: int = 10, flat_threshold: float = 0.5,
                   sc_pivot_len: int = 3, sc_bands_mult: float = 0.6,
                   vp_lookback: int = 200, vp_rows: int = 30,
                   vp_width: int = 40, crossover_lookback: int = 4) -> dict:
    """
    Compute the 10-category score for a stock.

    Mirrors the Pine Script HMAxEMA Swing Trading System scoring logic.
    Max total = 105 pts (capped at 100), with Trend weighted at 20 pts.
    Entry signal (swing-trading strategy):
      (1) recent Fast MA crossed above Slow MA AND current close is above the crossover level
      (2) current close is above Volume Profile POC AND above the crossover level
      (3) techno-fundamental total score >= 50

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
        index_df: NIFTY index OHLCV DataFrame for relative strength (optional)
        fast_ma_type/slow_ma_type: MA types
        fast_ma_len/slow_ma_len: MA lengths
        rsi_len, vol_ma_len, atr_len: Indicator lengths
        adx_len, adx_threshold: ADX sideways filter params
        chop_len, chop_threshold: Choppiness index params
        slope_ma_type, slope_ma_len: Slope MA params
        slope_lookback, flat_threshold: Slope flatness params
        sc_pivot_len, sc_bands_mult: Step Channel params (for reference)

    Returns:
        Dictionary with all scores and metadata
    """
    # Adaptive minimum based on data frequency
    n = len(df)
    if n >= 100:  # Daily data
        min_required = max(slow_ma_len, slope_ma_len, atr_len) + 10
    elif n >= 40:  # Weekly data
        min_required = max(slow_ma_len, slope_ma_len, atr_len) + 5
    else:  # Monthly data
        min_required = max(slow_ma_len, slope_ma_len, atr_len) + 3
    if n < min(min_required, 25):  # At least 25 bars
        return None  # Not enough data

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── Core Indicators ──────────────────────────────────────────────────────
    fast_ma = get_ma(fast_ma_type, close, fast_ma_len, volume)
    slow_ma = get_ma(slow_ma_type, close, slow_ma_len, volume)
    rsi_val = rsi(close, rsi_len)
    _, _, macd_hist = macd(close)
    stoch_k = stochastic(high, low, close)
    obv_val = obv(close, volume)
    obv_ma = sma(obv_val, 20)
    vol_ma = sma(volume, vol_ma_len)
    atr_val = atr(high, low, close, atr_len)
    adx_val = adx(high, low, close, adx_len)

    # Price changes (adaptive to data frequency)
    # For daily: 21 bars ~1 month, 63 bars ~3 months
    # For weekly: 4 bars ~1 month, 13 bars ~3 months  
    # For monthly: 1 bar ~1 month, 3 bars ~3 months
    n = len(close)
    if n >= 100:  # Daily data (~250 bars/year)
        pc1m_period, pc3m_period = 21, 63
    elif n >= 40:  # Weekly data (~52 bars/year)
        pc1m_period, pc3m_period = 4, 13
    else:  # Monthly data (~12 bars/year)
        pc1m_period, pc3m_period = 1, 3
    pc1m = price_change(close, pc1m_period)
    pc3m = price_change(close, pc3m_period)

    # ── Sideways Filter (matches Pine Script exactly) ────────────────────────
    # ADX Trend Strength
    is_sideways_adx = adx_val.iloc[-1] < adx_threshold if not np.isnan(adx_val.iloc[-1]) else False

    # Choppiness Index
    atr1 = atr(high, low, close, 1)
    chop_sum = atr1.rolling(chop_len).sum()
    chop_range = high.rolling(chop_len).max() - low.rolling(chop_len).min()
    chop_safe_range = chop_range.replace(0, np.nan)
    chop_val = 100 * np.log10(chop_sum / chop_safe_range) / math.log10(chop_len)
    is_sideways_chop = chop_val.iloc[-1] > chop_threshold if not np.isnan(chop_val.iloc[-1]) else False

    # MA Slope Flatness
    selected_ma = get_ma(slope_ma_type, close, slope_ma_len, volume)
    if len(selected_ma) > slope_lookback and selected_ma.iloc[-1 - slope_lookback] != 0:
        ma_slope_pct = abs(
            (selected_ma.iloc[-1] - selected_ma.iloc[-1 - slope_lookback])
            / selected_ma.iloc[-1 - slope_lookback]
        ) * 100
    else:
        ma_slope_pct = 0.0
    is_sideways_slope = ma_slope_pct < flat_threshold

    is_sideways = is_sideways_adx or is_sideways_chop or is_sideways_slope

    # Current values (last bar)
    curr = {
        "close": close.iloc[-1],
        "fast_ma": fast_ma.iloc[-1],
        "slow_ma": slow_ma.iloc[-1],
        "rsi": rsi_val.iloc[-1],
        "macd_hist": macd_hist.iloc[-1],
        "macd_hist_prev": macd_hist.iloc[-2] if len(macd_hist) > 1 else np.nan,
        "stoch_k": stoch_k.iloc[-1],
        "obv": obv_val.iloc[-1],
        "obv_prev": obv_val.iloc[-2] if len(obv_val) > 1 else np.nan,
        "obv_ma": obv_ma.iloc[-1],
        "volume": volume.iloc[-1],
        "vol_ma": vol_ma.iloc[-1],
        "atr": atr_val.iloc[-1],
        "adx": adx_val.iloc[-1],
        "pc1m": pc1m.iloc[-1],
        "pc3m": pc3m.iloc[-1],
        "hh50": highest(high, 50).iloc[-1],
        "ll50": lowest(low, 50).iloc[-1],
        "is_sideways": is_sideways,
    }
    
    # ── Volume Profile POC (11-week) ──────────────────────────────────────
    # 11 weeks = 55 trading days for daily data
    vp_lookback_weeks = 11
    if n >= 100:  # Daily data
        vp_bars = vp_lookback_weeks * 5  # 55 bars for 11 weeks
    elif n >= 40:  # Weekly data
        vp_bars = vp_lookback_weeks  # 11 bars for 11 weeks
    else:  # Monthly data
        vp_bars = 3  # approximate
    
    vp_poc = volume_profile_poc(high, low, close, volume, lookback=vp_bars)
    curr["vp_poc"] = vp_poc.iloc[-1] if not np.isnan(vp_poc.iloc[-1]) else close.iloc[-1]
    
    # Check if price is above POC
    curr["above_poc"] = curr["close"] >= curr["vp_poc"]
    
    # Check for MA crossover (Fast MA crossed above Slow MA)
    # This is used as a FILTER, not for scoring
    curr["ma_bullish"] = curr["fast_ma"] > curr["slow_ma"]
    
    # Close above BOTH MAs — stricter entry confirmation
    curr["close_above_both_ma"] = curr["close"] > curr["fast_ma"] and curr["close"] > curr["slow_ma"]
    
    # Look for crossovers in last N bars (configurable)
    ma_crossed_above = False
    crossover_bars_ago = -1  # -1 means no crossover found
    crossover_count = 0  # Total crossovers found in lookback
    crossover_dates = []  # List of bars where crossovers happened
    crossover_level = None  # Price level where the most recent crossover occurred

    lookback = min(crossover_lookback, len(fast_ma) - 1)  # Check last N bars (or available bars)
    for i in range(1, lookback + 1):
        idx_curr = -i
        idx_prev = -i - 1

        if idx_prev < -len(fast_ma):
            break

        fast_curr = fast_ma.iloc[idx_curr]
        fast_prev = fast_ma.iloc[idx_prev]
        slow_curr = slow_ma.iloc[idx_curr]
        slow_prev = slow_ma.iloc[idx_prev]

        # Check if crossover happened at this bar
        if (not np.isnan(fast_curr) and not np.isnan(fast_prev) and
            not np.isnan(slow_curr) and not np.isnan(slow_prev)):
            if fast_curr > slow_curr and fast_prev <= slow_prev:
                crossover_count += 1
                crossover_dates.append(i)  # Store bars ago
                if not ma_crossed_above:  # First (most recent) crossover
                    ma_crossed_above = True
                    crossover_bars_ago = i
                    # Price level of the crossover (~ where Fast MA met Slow MA)
                    crossover_level = float(slow_curr)

    curr["ma_crossed_above"] = ma_crossed_above
    curr["crossover_bars_ago"] = crossover_bars_ago if ma_crossed_above else -1
    curr["crossover_count"] = crossover_count
    curr["crossover_dates"] = crossover_dates
    curr["crossover_level"] = crossover_level

    # Criterion: current close must be above the crossover price level
    curr["close_above_crossover"] = (
        crossover_level is not None and curr["close"] > crossover_level
    )

    # ── Category 1: TREND (max 20 pts) ──────────────────────────────────────
    # Priority: Crossover freshness + POC position are the key entry signals
    trend_score = 0.0
    
    # 1. MA alignment (0-5 pts)
    if curr["ma_bullish"]:
        trend_score += 5.0
    
    # 2. Above Volume Profile POC (0-5 pts) — key volume support
    if curr["above_poc"]:
        trend_score += 5.0
    
    # 3. Close above BOTH MAs (0-3 pts) — entry confirmation
    if curr["close_above_both_ma"]:
        trend_score += 3.0
    elif curr["close"] > curr["slow_ma"]:
        trend_score += 1.5  # Partial credit if only above slow MA
    
    # 4. Crossover freshness (0-4 pts) — recent crossover = stronger signal
    if curr["ma_crossed_above"]:
        bars = curr["crossover_bars_ago"]
        if bars <= 1:
            trend_score += 4.0  # Fresh crossover — strongest signal
        elif bars <= 2:
            trend_score += 3.0
        elif bars <= 3:
            trend_score += 2.0
        elif bars <= 4:
            trend_score += 1.0
        else:
            trend_score += 0.5  # Older crossover, diminishing value
    
    # 5. ADX trend strength (0-3 pts)
    if not np.isnan(curr["adx"]) and curr["adx"] > 25:
        trend_score += 3.0
    
    trend_score = min(trend_score, 20.0)

    # ── Category 2: MOMENTUM (max 15 pts) ───────────────────────────────────
    mom_score = 0.0
    if not np.isnan(curr["pc1m"]):
        if curr["pc1m"] > 0:
            mom_score += min(7.0, 7.0 * (curr["pc1m"] / 5.0))
        else:
            mom_score += max(-3.0, 7.0 * (curr["pc1m"] / 10.0))
    if not np.isnan(curr["pc3m"]):
        if curr["pc3m"] > 0:
            mom_score += min(8.0, 8.0 * (curr["pc3m"] / 10.0))
        else:
            mom_score += max(-4.0, 8.0 * (curr["pc3m"] / 20.0))
    mom_score = max(0.0, min(mom_score, 15.0))

    # ── Category 3: RSI (max 8 pts) ─────────────────────────────────────────
    rsi_score = 0.0
    if not np.isnan(curr["rsi"]):
        if 40 <= curr["rsi"] <= 70:
            rsi_score = 8.0 * (1.0 - abs(curr["rsi"] - 55.0) / 15.0)
        else:
            rsi_score = 0.0
    rsi_score = max(0.0, min(rsi_score, 8.0))

    # ── Category 4: MACD (max 7 pts) ────────────────────────────────────────
    macd_score = 0.0
    if not np.isnan(curr["macd_hist"]):
        if curr["macd_hist"] > 0:
            macd_score += 4.0
        if not np.isnan(curr["macd_hist_prev"]) and curr["macd_hist"] > curr["macd_hist_prev"]:
            macd_score += 3.0
    macd_score = min(macd_score, 7.0)

    # ── Category 5: STOCHASTIC (max 5 pts) ──────────────────────────────────
    stoch_score = 0.0
    if not np.isnan(curr["stoch_k"]):
        if 20 < curr["stoch_k"] < 80:
            stoch_score = 5.0
    stoch_score = min(stoch_score, 5.0)

    # ── Category 6: OBV (max 5 pts) ─────────────────────────────────────────
    obv_score = 0.0
    if not np.isnan(curr["obv"]) and not np.isnan(curr["obv_ma"]):
        if curr["obv"] > curr["obv_ma"]:
            obv_score += 3.0
        if not np.isnan(curr["obv_prev"]) and curr["obv"] > curr["obv_prev"]:
            obv_score += 2.0
    obv_score = min(obv_score, 5.0)

    # ── Category 7: VOLUME (max 10 pts) ─────────────────────────────────────
    vol_score = 0.0
    if not np.isnan(curr["vol_ma"]) and curr["vol_ma"] > 0:
        if curr["volume"] > curr["vol_ma"]:
            vol_score += 5.0
        if curr["volume"] > curr["vol_ma"] * 1.2:
            vol_score += 3.0
        # Liquidity: good if volume > 50-day average
        vol_t = sma(volume, 50).iloc[-1]
        if not np.isnan(vol_t) and curr["volume"] > vol_t:
            vol_score += 2.0
    vol_score = min(vol_score, 10.0)

    # ── Category 8: RELATIVE STRENGTH (max 10 pts) ──────────────────────────
    rs_score = 0.0
    if index_df is not None and len(index_df) > rs_length + 5:
        idx_close = index_df["close"]
        idx_rs = (idx_close.iloc[-1] / idx_close.iloc[-1 - rs_length] - 1) * 100
        stock_rs = (close.iloc[-1] / close.iloc[-1 - rs_length] - 1) * 100
        if stock_rs > idx_rs:
            rs_score += 5.0
        if stock_rs > 0:
            rs_score += 5.0
    else:
        # Fallback: use raw momentum as proxy
        if not np.isnan(curr["pc1m"]) and curr["pc1m"] > 0:
            rs_score += 5.0
        if not np.isnan(curr["pc3m"]) and curr["pc3m"] > 0:
            rs_score += 5.0
    rs_score = min(rs_score, 10.0)

    # ── Category 9: VOLATILITY (max 5 pts) ──────────────────────────────────
    atr_pct = (curr["atr"] / curr["close"]) * 100 if curr["close"] > 0 else 0
    volat_stat = "High" if atr_pct > 3 else ("Low" if atr_pct < 1 else "Medium")
    volat_score = 5.0 if volat_stat in ("Medium", "Low") else 0.0

    # ── Category 10: FUNDAMENTALS (max 20 pts) ──────────────────────────────
    fund_score = 0.0
    fund_detail = {}

    # Try to get fundamental data from yfinance (passed via df attrs or dict)
    fundamentals = getattr(df, '_fundamentals', None)
    if fundamentals:
        calc_pe = fundamentals.get("pe_ratio")
        eps_growth = fundamentals.get("eps_growth")
        rev_growth = fundamentals.get("rev_growth")
        roe = fundamentals.get("roe")

        # P/E: Lower is better (within reason)
        if calc_pe is not None and calc_pe > 0:
            if calc_pe < 15:
                fund_score += 5.0
                fund_detail["pe"] = "Strong"
            elif calc_pe < 25:
                fund_score += 3.0
                fund_detail["pe"] = "Fair"
            else:
                fund_detail["pe"] = "Expensive"
        else:
            fund_detail["pe"] = "N/A"

        # EPS Growth: Higher is better
        if eps_growth is not None:
            if eps_growth > 20:
                fund_score += 5.0
                fund_detail["eps_growth"] = "Strong"
            elif eps_growth > 0:
                fund_score += 3.0
                fund_detail["eps_growth"] = "Positive"
            else:
                fund_detail["eps_growth"] = "Negative"
        else:
            fund_detail["eps_growth"] = "N/A"

        # Revenue Growth: Higher is better
        if rev_growth is not None:
            if rev_growth > 15:
                fund_score += 5.0
                fund_detail["rev_growth"] = "Strong"
            elif rev_growth > 0:
                fund_score += 3.0
                fund_detail["rev_growth"] = "Positive"
            else:
                fund_detail["rev_growth"] = "Negative"
        else:
            fund_detail["rev_growth"] = "N/A"

        # ROE: Higher is better
        if roe is not None:
            if roe > 20:
                fund_score += 5.0
                fund_detail["roe"] = "Strong"
            elif roe > 10:
                fund_score += 3.0
                fund_detail["roe"] = "Fair"
            else:
                fund_detail["roe"] = "Weak"
        else:
            fund_detail["roe"] = "N/A"
    else:
        fund_detail = {"pe": "N/A", "eps_growth": "N/A", "rev_growth": "N/A", "roe": "N/A"}

    fund_score = min(fund_score, 20.0)

    # ── TOTAL SCORE ─────────────────────────────────────────────────────────
    total = (trend_score + mom_score + rsi_score + macd_score + stoch_score
             + obv_score + vol_score + rs_score + volat_score + fund_score)
    total = max(0.0, min(total, 100.0))

    # Build result
    result = {
        "total": round(total, 1),
        "trend":     round(trend_score, 1),
        "momentum":  round(mom_score, 1),
        "rsi":       round(rsi_score, 1),
        "macd":      round(macd_score, 1),
        "stoch":     round(stoch_score, 1),
        "obv":       round(obv_score, 1),
        "volume":    round(vol_score, 1),
        "rel_str":   round(rs_score, 1),
        "volatility": round(volat_score, 1),
        "fundamentals": round(fund_score, 1),
        # Key signals (priority indicators)
        "ma_bullish": curr["ma_bullish"],
        "close_above_both_ma": curr["close_above_both_ma"],
        "ma_crossed_above": curr["ma_crossed_above"],
        "crossover_bars_ago": curr["crossover_bars_ago"],
        "crossover_count": curr["crossover_count"],
        "crossover_dates": curr["crossover_dates"],
        "above_poc": curr["above_poc"],
        "vp_poc": round(curr["vp_poc"], 2),
        # Sideways filter info
        "is_sideways": is_sideways,
        "sideways_reasons": (
            (["ADX"] if is_sideways_adx else [])
            + (["Chop"] if is_sideways_chop else [])
            + (["Slope"] if is_sideways_slope else [])
        ),
        # Metadata
        "close": round(curr["close"], 2),
        "rsi_val": round(curr["rsi"], 1) if not np.isnan(curr["rsi"]) else None,
        "adx_val": round(curr["adx"], 1) if not np.isnan(curr["adx"]) else None,
        "pc1m": round(curr["pc1m"], 2) if not np.isnan(curr["pc1m"]) else None,
        "pc3m": round(curr["pc3m"], 2) if not np.isnan(curr["pc3m"]) else None,
        "atr_pct": round(atr_pct, 2),
        "volat_stat": volat_stat,
        "trend_dir": ("Bull" if curr["close"] > curr["slow_ma"] else "Bear"),
        "trend_color": "bull" if curr["close"] > curr["slow_ma"] else "bear",
        # Fundamentals detail
        "fund_detail": fund_detail,
        # Combined rating based on key signals + score
        "combined_rating": _get_combined_rating(total, curr["ma_bullish"], curr["above_poc"], curr["close_above_both_ma"]),
        # Swing-trading ENTRY signal (per strategy):
        #   (1) recent Fast MA crossed above Slow MA AND close above the crossover level
        #   (2) close above Volume Profile POC AND above the crossover level
        #   (3) techno-fundamental score >= 50
        "entry_signal": bool(
            curr["ma_crossed_above"]
            and curr["close_above_crossover"]
            and curr["above_poc"]
            and total >= 50
        ),
    }

    return result


def _get_combined_rating(total_score: float, ma_bullish: bool, above_poc: bool, close_above_both_ma: bool = False) -> str:
    """
    Generate combined rating based on key signals and score.
    
    Priority:
    1. Close above BOTH MAs + above POC = strongest signal (ideal entry)
    2. MA bullish + above POC = strong signal
    3. MA bullish only or above POC only = moderate
    4. Score-based rating as fallback
    
    Args:
        total_score: Total score (0-100)
        ma_bullish: True if Fast MA > Slow MA
        above_poc: True if price >= VP POC
        close_above_both_ma: True if close > Fast MA AND close > Slow MA
    
    Returns:
        'EXCELLENT', 'GOOD', 'MODERATE', or 'POOR'
    """
    # Strongest: Close above both MAs + above POC (ideal entry criteria)
    if close_above_both_ma and above_poc:
        if total_score >= 60:
            return "EXCELLENT"
        elif total_score >= 50:
            return "GOOD"
        elif total_score >= 35:
            return "MODERATE"
        else:
            return "POOR"
    
    # Strong: MA bullish + above POC (close may be between the MAs)
    elif ma_bullish and above_poc:
        if total_score >= 65:
            return "EXCELLENT"
        elif total_score >= 50:
            return "GOOD"
        elif total_score >= 40:
            return "MODERATE"
        else:
            return "POOR"
    
    # Moderate: MA bullish only
    elif ma_bullish:
        if total_score >= 70:
            return "EXCELLENT"
        elif total_score >= 55:
            return "GOOD"
        elif total_score >= 40:
            return "MODERATE"
        else:
            return "POOR"
    
    # Moderate: Above POC only
    elif above_poc:
        if total_score >= 70:
            return "EXCELLENT"
        elif total_score >= 55:
            return "GOOD"
        elif total_score >= 40:
            return "MODERATE"
        else:
            return "POOR"
    
    # Weak: Neither condition met - rely on score
    else:
        if total_score >= 70:
            return "EXCELLENT"
        elif total_score >= 55:
            return "GOOD"
        elif total_score >= 40:
            return "MODERATE"
        else:
            return "POOR"
