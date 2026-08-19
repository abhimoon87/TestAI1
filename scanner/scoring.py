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


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample an OHLCV DataFrame to weekly bars.

    Used to evaluate the higher-timeframe (weekly) entry condition
    independently of the analysis timeframe of the scan. If the data is
    already weekly (or cannot be resampled), it is returned as-is / None.
    """
    if df is None or df.empty or "close" not in df.columns:
        return None
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        return None
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    agg = {}
    for col in ["open", "high", "low", "close"]:
        if col in d.columns:
            agg[col] = {"open": "first", "high": "max", "low": "min", "close": "last"}[col]
    if "volume" in d.columns:
        agg["volume"] = "sum"
    if not agg:
        return None
    resampled = d.resample("W").agg(agg).dropna()
    return resampled if not resampled.empty else None


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

    Higher-timeframe (WEEKLY) buy condition (TradingView):
      wma((2 * wma(close, 22)) - wma(close, 44), 6)  crossed_above  ema(close, 50)
      i.e. the weekly HMA(44) crosses above the weekly EMA(50).

    Returns:
        Dictionary with all scores and metadata
    """
    # Adaptive minimum based on data frequency
    n = len(df)
    if n >= 100:  # Daily data
        min_required = max(slow_ma_len, slope_ma_len, atr_len) + 10
    elif n >= 40:  # Weekly data
        min_required = None
    else:  # Monthly data
        min_required = None
    if n >= 100:
        min_required = max(slow_ma_len, slope_ma_len, atr_len) + 10
    elif n >= 40:
        min_required = max(slow_ma_len, slope_ma_len, atr_len) + 5
    else:
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

    # ── Weekly HMA(44) x EMA(50) Higher-Timeframe Crossover ─────────────────
    # Matches the TradingView condition:
    #   wma((2 * wma(close, 22)) - wma(close, 44), 6)  crossed_above  ema(close, 50)
    # hull_ma(close, 44) computes exactly that WMA-of-WMA formula, so we
    # evaluate it on weekly bars regardless of the analysis timeframe.
    weekly_hma_bull = False
    weekly_hma_cross = False
    weekly_hma_cross_bars_ago = -1
    weekly_df = to_weekly(df)
    if weekly_df is not None and len(weekly_df) >= 60:
        w_close = weekly_df["close"]
        w_hma = hull_ma(w_close, 44)
        w_ema = ema(w_close, 50)
        if not np.isnan(w_hma.iloc[-1]) and not np.isnan(w_ema.iloc[-1]):
            weekly_hma_bull = w_hma.iloc[-1] > w_ema.iloc[-1]
        look = min(12, len(w_hma) - 1)  # up to 12 weekly bars (~3 months)
        for i in range(1, look + 1):
            ic, ip = -i, -i - 1
            if ip < -len(w_hma):
                break
            hc, hp = w_hma.iloc[ic], w_hma.iloc[ip]
            ec, ep = w_ema.iloc[ic], w_ema.iloc[ip]
            if not (np.isnan(hc) or np.isnan(hp) or np.isnan(ec) or np.isnan(ep)):
                if hc > ec and hp <= ep:
                    weekly_hma_cross = True
                    weekly_hma_cross_bars_ago = i
                    break

    # Price changes (adaptive to data frequency)
    if n >= 100:  # Daily data (~250 bars/year)
        pc1m_period, pc3m_period = 21, 63
    elif n >= 40:  # Weekly data (~52 bars/year)
        pc1m_period, pc3m_period = 4, 13
    else:  # Monthly data (~12 bars/year)
        pc1m_period, pc3m_period = 1, 3
    pc1m = price_change(close, pc1m_period)
    pc3m = price_change(close, pc3m_period)

    # ── Sideways Filter (matches Pine Script exactly) ────────────────────────
    is_sideways_adx = adx_val.iloc[-1] < adx_threshold if not np.isnan(adx_val.iloc[-1]) else False

    atr1 = atr(high, low, close, 1)
    chop_sum = atr1.rolling(chop_len).sum()
    chop_range = high.rolling(chop_len).max() - low.rolling(chop_len).min()
    chop_safe_range = chop_range.replace(0, np.nan)
    chop_val = 100 * np.log10(chop_sum / chop_safe_range) / math.log10(chop_len)
    is_sideways_chop = chop_val.iloc[-1] > chop_threshold if not np.isnan(chop_val.iloc[-1]) else False

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

    # ── Volume Profile POC ────────────────────────────────────────────────
    vp_bars = max(int(vp_lookback), 10)
    vp_poc = volume_profile_poc(high, low, close, volume, lookback=vp_bars)
    curr["vp_poc"] = vp_poc.iloc[-1] if not np.isnan(vp_poc.iloc[-1]) else close.iloc[-1]

    curr["above_poc"] = curr["close"] >= curr["vp_poc"]
    curr["ma_bullish"] = curr["fast_ma"] > curr["slow_ma"]
    curr["close_above_both_ma"] = curr["close"] > curr["fast_ma"] and curr["close"] > curr["slow_ma"]

    # Look for crossovers in last N bars (configurable)
    ma_crossed_above = False
    crossover_bars_ago = -1
    crossover_count = 0
    crossover_dates = []
    crossover_level = None

    lookback = min(crossover_lookback, len(fast_ma) - 1)
    for i in range(1, lookback + 1):
        idx_curr = -i
        idx_prev = -i - 1
        if idx_prev < -len(fast_ma):
            break
        fast_curr = fast_ma.iloc[idx_curr]
        fast_prev = fast_ma.iloc[idx_prev]
        slow_curr = slow_ma.iloc[idx_curr]
        slow_prev = slow_ma.iloc[idx_prev]
        if (not np.isnan(fast_curr) and not np.isnan(fast_prev) and
            not np.isnan(slow_curr) and not np.isnan(slow_prev)):
            if fast_curr > slow_curr and fast_prev <= slow_prev:
                crossover_count += 1
                crossover_dates.append(i)
                if not ma_crossed_above:
                    ma_crossed_above = True
                    crossover_bars_ago = i
                    crossover_level = float(slow_curr)

    curr["ma_crossed_above"] = ma_crossed_above
    curr["crossover_bars_ago"] = crossover_bars_ago if ma_crossed_above else -1
    curr["crossover_count"] = crossover_count
    curr["crossover_dates"] = crossover_dates
    curr["crossover_level"] = crossover_level
    curr["close_above_crossover"] = (
        crossover_level is not None and curr["close"] > crossover_level
    )

    # ── Category 1: TREND (max 20 pts) ──────────────────────────────────────
    trend_score = 0.0
    if curr["ma_bullish"]:
        trend_score += 5.0
    if curr["above_poc"]:
        trend_score += 5.0
    if curr["close_above_both_ma"]:
        trend_score += 3.0
    elif curr["close"] > curr["slow_ma"]:
        trend_score += 1.5
    if curr["ma_crossed_above"]:
        bars = curr["crossover_bars_ago"]
        if bars <= 1:
            trend_score += 4.0
        elif bars <= 2:
            trend_score += 3.0
        elif bars <= 3:
            trend_score += 2.0
        elif bars <= 4:
            trend_score += 1.0
        else:
            trend_score += 0.5
    if not np.isnan(curr["adx"]) and curr["adx"] > 25:
        trend_score += 3.0
    trend_score = min(trend_score, 20.0)

    # ── Category 2: MOMENTUM (max 15 pts) ───────────────────────────────────
    mom_score = 0.0
    if not np.isnan(curr["pc1m"]):
        if curr["pc1m"] > 0:
            mom_score += min(7.0, 7.0 * (curr["pc1m"] / 5.0))
        else:
            mom_score += max(-3.0, min(0.0, 7.0 * (curr["pc1m"] / 10.0)))
    if not np.isnan(curr["pc3m"]):
        if curr["pc3m"] > 0:
            mom_score += min(8.0, 8.0 * (curr["pc3m"] / 10.0))
        else:
            mom_score += max(-4.0, min(0.0, 8.0 * (curr["pc3m"] / 20.0)))
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

    fundamentals = getattr(df, '_fundamentals', None)
    if fundamentals:
        calc_pe = fundamentals.get("pe_ratio")
        eps_growth = fundamentals.get("eps_growth")
        rev_growth = fundamentals.get("rev_growth")
        roe = fundamentals.get("roe")

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
        # Weekly HMA(44) x EMA(50) higher-timeframe condition
        "weekly_hma_bull": weekly_hma_bull,
        "weekly_hma_cross": weekly_hma_cross,
        "weekly_hma_cross_bars_ago": weekly_hma_cross_bars_ago,
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
        "entry_signal": bool(
            curr["ma_crossed_above"]
            and curr["close_above_crossover"]
            and curr["above_poc"]
            and total >= 50
        ),
        # Weekly HMA(44) x EMA(50) buy trigger (TradingView condition).
        "weekly_entry_signal": bool(weekly_hma_cross),
    }

    return result


def _get_combined_rating(total_score: float, ma_bullish: bool, above_poc: bool, close_above_both_ma: bool = False) -> str:
    """
    Generate combined rating based on key signals and score.
    """
    if close_above_both_ma and above_poc:
        if total_score >= 60:
            return "EXCELLENT"
        elif total_score >= 50:
            return "GOOD"
        elif total_score >= 35:
            return "MODERATE"
        else:
            return "POOR"
    elif ma_bullish and above_poc:
        if total_score >= 65:
            return "EXCELLENT"
        elif total_score >= 50:
            return "GOOD"
        elif total_score >= 40:
            return "MODERATE"
        else:
            return "POOR"
    elif ma_bullish:
        if total_score >= 70:
            return "EXCELLENT"
        elif total_score >= 55:
            return "GOOD"
        elif total_score >= 40:
            return "MODERATE"
        else:
            return "POOR"
    elif above_poc:
        if total_score >= 70:
            return "EXCELLENT"
        elif total_score >= 55:
            return "GOOD"
        elif total_score >= 40:
            return "MODERATE"
        else:
            return "POOR"
    else:
        if total_score >= 70:
            return "EXCELLENT"
        elif total_score >= 55:
            return "GOOD"
        elif total_score >= 40:
            return "MODERATE"
        else:
            return "POOR"
