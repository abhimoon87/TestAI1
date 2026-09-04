"""
10-category scoring engine, derived from the HMAxEMA Swing Trading
System Pine Script (reference file: HMA_EMA_Swing_Strategy_v2.pine).

Category caps are kept identical to the Pine v2 indicator, total = 100:

    Trend 15 | Momentum 15 | RSI 8 | MACD 7 | Stochastic 5 | OBV 5 |
    Volume 10 | Rel. strength 10 | Volatility 5 | Fundamentals 20

Pine-vs-Python parity, per category:

  Exact match (same formulas / points):
    - Momentum (1M/3M gradient with negative clamp)
    - RSI (peak at 55, band 40-70)
    - MACD (positive histogram + rising histogram)
    - Stochastic (healthy 20-80 band)
    - OBV (above OBV-MA + rising OBV)
    - Volatility (ATR% bands -> Medium/Low scores)
    - Fundamentals thresholds (P/E < 15 / < 25; EPS > 20 % / > 0 %;
      revenue > 15 % / > 0 %; ROE > 20 % / > 10 %) — same points,
      different data source (provider chain instead of TradingView
      financials; category scores 0 when fundamentals are unavailable)

  Intentional extensions (weights still capped at the Pine max):
    - Trend: Pine v2 sums close-vs-slow (10), fast-vs-slow (5) and
      ADX > 25 (2). The Python scorer re-weights the MA stack and adds
      volume-profile POC participation + crossover-recentness points,
      all under the same 15-point cap.
    - Relative strength: Pine v2 compares the stock against an index
      AND NIFTY (two request.security legs). Python fetches a single
      index (settings['index_symbol']) and approximates the second
      leg with absolute positive momentum; without index data both
      legs are proxied from 1M/3M price change.
    - Volume: Pine v2 only rewards a single-bar volume print above
      the volume MA / 1.2x / 50-bar average. The Python scorer blends
      the last-bar check with a 5-bar participation average, so a
      sustained move that accumulated volume over the breakout week
      is not zeroed out by one quiet closing day (still capped at 10).
    - Sideways: the Pine sideways test is "any one of ADX / Choppiness /
      slope". The Python scorer is direction-aware: a strong directional
      move (|1M change| >= 5%) requires two independent sideways
      evidences (weak ADX plus chop or flat slope) before it is
      classified as sideways, so strong rallies are not mislabelled
      choppy by the Choppiness index alone.

  Metadata only (never changes the total):
    - weekly HMA(44) x EMA(50) higher-timeframe state
    - sideways (ADX / Choppiness / slope) flags
    - combined-rating labels and entry-signal flags

Refactored into composable helpers:
  - detect_crossover() — shared MA crossover detection (used by filter & scorer)
  - _compute_indicators() — compute all core indicators at once
  - _compute_weekly_hma() — weekly higher-timeframe HMA crossover
  - _compute_sideways() — ADX / Cholangirong / Slope sideways filter
  - _score_trend .. _score_fundamentals — per-category scoring (10 total)
  - compute_scores() — orchestrator that composes the above
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .indicators import (
    adx,
    atr,
    ema,
    highest,
    hull_ma,
    kama,
    lowest,
    macd,
    obv,
    price_change,
    rsi,
    sma,
    stochastic,
    volume_profile_poc,
    vwma,
)
from .trace import trace

# ══════════════════════════════════════════════════════════════════════════════
# MOVING AVERAGE SELECTOR
# ══════════════════════════════════════════════════════════════════════════════

def get_ma(ma_type: str, src: pd.Series, length: int,
           volume: pd.Series | None = None) -> pd.Series:
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


# ══════════════════════════════════════════════════════════════════════════════
# WEEKLY RESAMPLE
# ══════════════════════════════════════════════════════════════════════════════

def to_weekly(df: pd.DataFrame) -> pd.DataFrame | None:
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
    if "volume" in df.columns:
        agg["volume"] = "sum"
    if not agg:
        return None
    resampled = d.resample("W").agg(agg).dropna()
    return resampled if not resampled.empty else None


# ══════════════════════════════════════════════════════════════════════════════
# SHARED CROSSOVER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_crossover(fast_ma: pd.Series, slow_ma: pd.Series,
                     lookback: int = 20) -> dict:
    """
    Detect MA crossovers within the last *lookback* bars.

    Returns:
        {
            "crossed": bool,           # any bullish crossover found
            "bars_ago": int,           # bars since most recent crossover (-1 if none)
            "level": float | None,     # slow MA value at crossover point
            "count": int,              # total crossovers in lookback window
            "dates": list[int],        # bar offsets of all crossovers (1 = most recent)
        }
    """
    result = {"crossed": False, "bars_ago": -1, "level": None, "count": 0, "dates": []}

    lb = min(lookback, len(fast_ma) - 1)
    for i in range(1, lb + 1):
        ic, ip = -i, -i - 1
        if ip < -len(fast_ma):
            break
        fc, fp = fast_ma.iloc[ic], fast_ma.iloc[ip]
        sc, sp = slow_ma.iloc[ic], slow_ma.iloc[ip]
        if (not np.isnan(fc) and not np.isnan(fp) and
                not np.isnan(sc) and not np.isnan(sp)) and fc > sc and fp <= sp:
            result["count"] += 1
            result["dates"].append(i)
            if not result["crossed"]:
                result["crossed"] = True
                result["bars_ago"] = i
                result["level"] = float(sc)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: STOCK FILTER
# ══════════════════════════════════════════════════════════════════════════════

def check_filter(df: pd.DataFrame,
                 fast_ma_type: str = "HMA", fast_ma_len: int = 40,
                 slow_ma_type: str = "EMA", slow_ma_len: int = 50,
                 crossover_lookback: int = 20) -> dict | None:
    """
    Model 1 — Stock Filter.

    Checks for a recent MA crossover on the given OHLCV data:
      1. A recent MA crossover occurred (within crossover_lookback bars)

    Returns:
        dict with filter metadata if the crossover condition is met,
        or None if the stock is filtered out.
    """
    if df is None or df.empty:
        return None

    n = len(df)
    min_bars = max(fast_ma_len, slow_ma_len) + crossover_lookback + 10
    if n < min_bars:
        return None

    close = df["close"]
    volume = df["volume"]

    fast_ma = get_ma(fast_ma_type, close, fast_ma_len, volume)
    slow_ma = get_ma(slow_ma_type, close, slow_ma_len, volume)

    if np.isnan(fast_ma.iloc[-1]) or np.isnan(slow_ma.iloc[-1]):
        return None

    xo = detect_crossover(fast_ma, slow_ma, crossover_lookback)
    if not xo["crossed"]:
        return None  # No recent crossover → filtered out

    return {
        "ma_crossed_above": True,
        "crossover_level": xo["level"],
        "crossover_bars_ago": xo["bars_ago"],
        "ma_bullish": bool(fast_ma.iloc[-1] > slow_ma.iloc[-1]),
        "close": float(close.iloc[-1]),
        "fast_ma": float(fast_ma.iloc[-1]),
        "slow_ma": float(slow_ma.iloc[-1]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: BULLISH / BEARISH
# ══════════════════════════════════════════════════════════════════════════════

def get_direction(filter_result: dict | None) -> str | None:
    """
    Model 2 — Bullish / Bearish classification.

    Args:
        filter_result: Output dict from check_filter().

    Returns:
        "Bull" if Fast MA crossed above Slow MA (bullish crossover),
        "Bear" if Fast MA crossed below Slow MA (bearish crossover).
    """
    if filter_result is None:
        return None
    return "Bull" if filter_result["ma_bullish"] else "Bear"


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR COMPUTATION (shared by scoring pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_indicators(df: pd.DataFrame, settings: dict) -> dict:
    """
    Compute all core technical indicators for a stock.

    Returns a dict of Series and scalar values used downstream by scoring
    functions and the main compute_scores orchestrator.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    fast_ma_type = settings.get("fast_ma_type", "HMA")
    fast_ma_len = settings.get("fast_ma_len", 40)
    slow_ma_type = settings.get("slow_ma_type", "EMA")
    slow_ma_len = settings.get("slow_ma_len", 50)
    rsi_len = settings.get("rsi_len", 14)
    vol_ma_len = settings.get("vol_ma_len", 20)
    atr_len = settings.get("atr_len", 14)
    adx_len = settings.get("adx_len", 14)

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

    return {
        "close": close, "high": high, "low": low, "volume": volume,
        "fast_ma": fast_ma, "slow_ma": slow_ma,
        "rsi_val": rsi_val, "macd_hist": macd_hist,
        "stoch_k": stoch_k, "obv_val": obv_val, "obv_ma": obv_ma,
        "vol_ma": vol_ma, "atr_val": atr_val, "adx_val": adx_val,
    }


def _last_values(ind: dict, df: pd.DataFrame, settings: dict) -> dict:
    """
    Extract the last-bar values from computed indicators into a flat dict
    (the ``curr`` dict used by all scoring functions).
    """
    close = ind["close"]
    n = len(close)

    # Price changes (adaptive to data frequency)
    if n >= 100:       # Daily data (~250 bars/year)
        pc1m_period, pc3m_period = 21, 63
    elif n >= 40:      # Weekly data (~52 bars/year)
        pc1m_period, pc3m_period = 4, 13
    else:              # Monthly data (~12 bars/year)
        pc1m_period, pc3m_period = 1, 3

    return {
        "close": close.iloc[-1],
        "fast_ma": ind["fast_ma"].iloc[-1],
        "slow_ma": ind["slow_ma"].iloc[-1],
        "rsi": ind["rsi_val"].iloc[-1],
        "macd_hist": ind["macd_hist"].iloc[-1],
        "macd_hist_prev": ind["macd_hist"].iloc[-2] if len(ind["macd_hist"]) > 1 else np.nan,
        "stoch_k": ind["stoch_k"].iloc[-1],
        "obv": ind["obv_val"].iloc[-1],
        "obv_prev": ind["obv_val"].iloc[-2] if len(ind["obv_val"]) > 1 else np.nan,
        "obv_ma": ind["obv_ma"].iloc[-1],
        "volume": ind["volume"].iloc[-1],
        "vol_ma": ind["vol_ma"].iloc[-1],
        "atr": ind["atr_val"].iloc[-1],
        "adx": ind["adx_val"].iloc[-1],
        "pc1m": price_change(close, pc1m_period).iloc[-1],
        "pc3m": price_change(close, pc3m_period).iloc[-1],
        "hh50": highest(ind["high"], 50).iloc[-1],
        "ll50": lowest(ind["low"], 50).iloc[-1],
    }


# ══════════════════════════════════════════════════════════════════════════════
# WEEKLY HMA HIGHER-TIMEFRAME CHECK
# ══════════════════════════════════════════════════════════════════════════════

def _compute_weekly_hma(df: pd.DataFrame) -> dict:
    """
    Evaluate the weekly higher-timeframe crossover, HMA(44) x EMA(50).

    Python-only extension: the Pine v2 file resamples its configurable
    fast/slow MAs to the analysis timeframe (defaults HMA(40)/EMA(50))
    but defines no separate weekly check. This helper hard-codes the
    44/50 weekly swing-regime check on top of the daily scan.

    Returns:
        {"bull": bool, "cross": bool, "cross_bars_ago": int}
    """
    result = {"bull": False, "cross": False, "cross_bars_ago": -1}

    weekly_df = to_weekly(df)
    if weekly_df is None or len(weekly_df) < 2:
        return result

    w_close = weekly_df["close"]
    w_hma = hull_ma(w_close, 44)
    w_ema = ema(w_close, 50)

    if not np.isnan(w_hma.iloc[-1]) and not np.isnan(w_ema.iloc[-1]):
        result["bull"] = w_hma.iloc[-1] > w_ema.iloc[-1]

    xo = detect_crossover(w_hma, w_ema, lookback=12)
    if xo["crossed"]:
        result["cross"] = True
        result["cross_bars_ago"] = xo["bars_ago"]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SIDEWAYS FILTER (ADX + Cholangirong + Slope)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_sideways(df: pd.DataFrame, adx_val: pd.Series,
                      settings: dict) -> dict:
    """
    Compute sideways / choppy market detection.

    Direction-aware: a strong directional move (|1M change| >= 5%) needs
    two independent sideways evidences (weak ADX plus chop *or* a flat
    MA slope) before the stock is classified sideways; otherwise a single
    trigger (as in the Pine reference) is enough. This stops strong
    rallies from being mislabelled choppy by the Choppiness index alone.

    Returns:
        {"is_sideways": bool, "reasons": list[str]}
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    adx_threshold = settings.get("adx_threshold", 20.0)
    chop_len = settings.get("chop_len", 14)
    chop_threshold = settings.get("chop_threshold", 61.8)
    slope_ma_type = settings.get("slope_ma_type", "EMA")
    slope_ma_len = settings.get("slope_ma_len", 50)
    slope_lookback = settings.get("slope_lookback", 10)
    flat_threshold = settings.get("flat_threshold", 0.5)
    strong_move_pct = settings.get("sideways_strong_move_pct", 5.0)

    # ADX filter
    adx_last = adx_val.iloc[-1]
    is_sideways_adx = adx_last < adx_threshold if not np.isnan(adx_last) else False

    # Cholangirong filter
    atr1 = atr(high, low, close, 1)
    chop_sum = atr1.rolling(chop_len).sum()
    chop_range = high.rolling(chop_len).max() - low.rolling(chop_len).min()
    chop_safe_range = chop_range.replace(0, np.nan)
    chop_val = 100 * np.log10(chop_sum / chop_safe_range) / math.log10(chop_len)
    is_sideways_chop = chop_val.iloc[-1] > chop_threshold if not np.isnan(chop_val.iloc[-1]) else False

    # Slope filter
    selected_ma = get_ma(slope_ma_type, close, slope_ma_len, volume)
    if len(selected_ma) > slope_lookback and selected_ma.iloc[-1 - slope_lookback] != 0:
        ma_slope_pct = abs(
            (selected_ma.iloc[-1] - selected_ma.iloc[-1 - slope_lookback])
            / selected_ma.iloc[-1 - slope_lookback]
        ) * 100
    else:
        ma_slope_pct = 0.0
    is_sideways_slope = ma_slope_pct < flat_threshold

    # Direction-awareness: a strong 1M move overrides the single-trigger rule
    lookback = 21 if len(close) >= 100 else (4 if len(close) >= 40 else 1)
    pc1m = price_change(close, min(lookback, len(close) - 2)).iloc[-1]
    strong_move = not np.isnan(pc1m) and abs(pc1m) >= strong_move_pct

    if strong_move:
        is_sideways = is_sideways_adx and (is_sideways_chop or is_sideways_slope)
    else:
        is_sideways = is_sideways_adx or is_sideways_chop or is_sideways_slope

    reasons = []
    if is_sideways_adx:
        reasons.append("ADX")
    if is_sideways_chop:
        reasons.append("Chop")
    if is_sideways_slope:
        reasons.append("Slope")
    if not is_sideways and reasons and strong_move:
        # Strong move overrode the flag — keep the reasons honest
        reasons = []

    return {"is_sideways": is_sideways, "reasons": reasons}


# ══════════════════════════════════════════════════════════════════════════════
# PER-CATEGORY SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _score_trend(curr: dict) -> float:
    """Category 1: TREND (max 15 pts)."""
    s = 0.0
    if curr["ma_bullish"]:
        s += 4.0
    if curr["above_poc"]:
        s += 2.5
    if curr["close_above_both_ma"]:
        s += 2.5
    elif curr["close"] > curr["slow_ma"]:
        s += 1.0
    if curr["ma_crossed_above"]:
        bars = curr["crossover_bars_ago"]
        if bars <= 1:
            s += 3.0
        elif bars <= 2:
            s += 2.0
        elif bars <= 3:
            s += 1.0
        elif bars <= 4:
            s += 0.5
        else:
            s += 0.25
    if not np.isnan(curr["adx"]) and curr["adx"] > 25:
        s += 2.0
    return min(s, 15.0)


def _score_momentum(curr: dict) -> float:
    """Category 2: MOMENTUM (max 15 pts)."""
    s = 0.0
    if not np.isnan(curr["pc1m"]):
        if curr["pc1m"] > 0:
            s += min(7.0, 7.0 * (curr["pc1m"] / 5.0))
        else:
            s += max(-3.0, min(0.0, 7.0 * (curr["pc1m"] / 10.0)))
    if not np.isnan(curr["pc3m"]):
        if curr["pc3m"] > 0:
            s += min(8.0, 8.0 * (curr["pc3m"] / 10.0))
        else:
            s += max(-4.0, min(0.0, 8.0 * (curr["pc3m"] / 20.0)))
    return max(0.0, min(s, 15.0))


def _score_rsi(curr: dict) -> float:
    """Category 3: RSI (max 8 pts)."""
    if np.isnan(curr["rsi"]):
        return 0.0
    if 40 <= curr["rsi"] <= 70:
        return max(0.0, min(8.0 * (1.0 - abs(curr["rsi"] - 55.0) / 15.0), 8.0))
    return 0.0


def _score_macd(curr: dict) -> float:
    """Category 4: MACD (max 7 pts)."""
    s = 0.0
    if not np.isnan(curr["macd_hist"]):
        if curr["macd_hist"] > 0:
            s += 4.0
        if not np.isnan(curr["macd_hist_prev"]) and curr["macd_hist"] > curr["macd_hist_prev"]:
            s += 3.0
    return min(s, 7.0)


def _score_stochastic(curr: dict) -> float:
    """Category 5: STOCHASTIC (max 5 pts)."""
    if np.isnan(curr["stoch_k"]):
        return 0.0
    return 5.0 if 20 < curr["stoch_k"] < 80 else 0.0


def _score_obv(curr: dict) -> float:
    """Category 6: OBV (max 5 pts)."""
    s = 0.0
    if not np.isnan(curr["obv"]) and not np.isnan(curr["obv_ma"]):
        if curr["obv"] > curr["obv_ma"]:
            s += 3.0
        if not np.isnan(curr["obv_prev"]) and curr["obv"] > curr["obv_prev"]:
            s += 2.0
    return min(s, 5.0)


def _score_volume(curr: dict) -> float:
    """Category 7: VOLUME (max 10 pts).

    Blends the last-bar print with 5-bar participation: a move that built
    volume over the breakout week still scores even if the latest single
    bar closed quietly (see module docstring for the Pine deviation).
    """
    s = 0.0
    if not np.isnan(curr["vol_ma"]) and curr["vol_ma"] > 0:
        if curr["volume"] > curr["vol_ma"]:
            s += 3.0
        vol_5 = curr.get("vol_5")
        if vol_5 is not None and not np.isnan(vol_5):
            if vol_5 > curr["vol_ma"]:
                s += 4.0
            if vol_5 > curr["vol_ma"] * 1.2:
                s += 2.0
        vol_t = curr.get("vol_t_50")
        above_50 = (
            vol_t is not None and not np.isnan(vol_t)
            and (curr["volume"] > vol_t or (vol_5 is not None and not np.isnan(vol_5) and vol_5 > vol_t))
        )
        if above_50:
            s += 1.0
    return min(s, 10.0)


def _score_relative_strength(curr: dict, close: pd.Series,
                             index_df: pd.DataFrame | None,
                             rs_length: int) -> float:
    """Category 8: RELATIVE STRENGTH (max 10 pts)."""
    s = 0.0
    if index_df is not None and len(index_df) > rs_length + 5:
        idx_close = index_df["close"]
        idx_rs = (idx_close.iloc[-1] / idx_close.iloc[-1 - rs_length] - 1) * 100
        stock_rs = (close.iloc[-1] / close.iloc[-1 - rs_length] - 1) * 100
        if stock_rs > idx_rs:
            s += 5.0
        if stock_rs > 0:
            s += 5.0
    else:
        if not np.isnan(curr["pc1m"]) and curr["pc1m"] > 0:
            s += 5.0
        if not np.isnan(curr["pc3m"]) and curr["pc3m"] > 0:
            s += 5.0
    return min(s, 10.0)


def _score_volatility(curr: dict) -> tuple[float, float, str]:
    """Category 9: VOLATILITY (max 5 pts).

    Returns:
        (score, atr_pct, volatility_status)
    """
    atr_pct = (curr["atr"] / curr["close"]) * 100 if curr["close"] > 0 else 0
    volat_stat = "High" if atr_pct > 3 else ("Low" if atr_pct < 1 else "Medium")
    volat_score = 5.0 if volat_stat in ("Medium", "Low") else 0.0
    return volat_score, atr_pct, volat_stat


def _score_fundamentals(df: pd.DataFrame) -> tuple[float, dict]:
    """Category 10: FUNDAMENTALS (max 20 pts).

    Returns:
        (score, detail_dict)
    """
    fund_score = 0.0
    fund_detail = {}

    fundamentals = getattr(df, '_fundamentals', None)
    if not fundamentals:
        return 0.0, {"pe": "N/A", "eps_growth": "N/A", "rev_growth": "N/A", "roe": "N/A"}

    calc_pe = fundamentals.get("pe_ratio")
    eps_growth = fundamentals.get("eps_growth")
    rev_growth = fundamentals.get("rev_growth")
    roe = fundamentals.get("roe")

    # P/E
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

    # EPS Growth
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

    # Revenue Growth
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

    # ROE
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

    return min(fund_score, 20.0), fund_detail


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: COMBINED RATING
# ══════════════════════════════════════════════════════════════════════════════

def _get_combined_rating(total_score: float, ma_bullish: bool,
                         above_poc: bool, close_above_both_ma: bool = False) -> str:
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
    elif ma_bullish or above_poc:
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


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: SCORE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

@trace(level=5, log_args=False)
def compute_scores(df: pd.DataFrame, timeframe: str = "D",
                   index_df: pd.DataFrame | None = None,
                   settings: dict | None = None) -> dict | None:
    """
    Compute the 10-category score for a stock.

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
        timeframe: Analysis timeframe ('D' daily, 'W' weekly, 'M' monthly).
        index_df: Index DataFrame for relative strength comparison.
        settings: Dict with scoring parameters (see DEFAULT_SETTINGS in
                  settings_store.py). Falls back to defaults if not provided.

    Category caps mirror the Pine v2 indicator (Trend 15, Momentum 15, ...
    Fundamentals 20; total 100), with documented per-category extensions —
    see the module docstring for the exact Pine-vs-Python mapping.

    Returns:
        Dictionary with all scores and metadata, or None if insufficient data.
    """
    if settings is None:
        settings = {}

    # Entry-gate config (mirrors backtest.py's min_adx_entry gate)
    min_adx_entry = float(settings.get("min_adx_entry", 0.0) or 0.0)

    n = len(df)
    min_required = {"D": 100, "W": 50, "M": 25}.get(timeframe, 100)
    if n < min_required:
        return None  # Not enough data

    # ── Compute all indicators ─────────────────────────────────────────────
    ind = _compute_indicators(df, settings)
    curr = _last_values(ind, df, settings)

    # ── Volume Profile POC ─────────────────────────────────────────────────
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]
    if settings.get("_skip_vp") or settings.get("skip_volume_profile"):
        curr["vp_poc"] = close.iloc[-1]
        curr["above_poc"] = True
    else:
        vp_lookback = settings.get("vp_lookback", 200)
        vp_bars = max(int(vp_lookback), 10)
        vp_bars = min(vp_bars, len(df))
        vp_poc = volume_profile_poc(high, low, close, volume, lookback=vp_bars)
        curr["vp_poc"] = vp_poc.iloc[-1] if not np.isnan(vp_poc.iloc[-1]) else close.iloc[-1]
        curr["above_poc"] = curr["close"] >= curr["vp_poc"]
    curr["ma_bullish"] = curr["fast_ma"] > curr["slow_ma"]
    curr["close_above_both_ma"] = curr["close"] > curr["fast_ma"] and curr["close"] > curr["slow_ma"]

    # ── Volume references (used by _score_volume) ─────────────────────────
    # 50-bar average + participation average (mean of the last N bars, N is
    # volume_participation_len, default 5 — see the module docstring).
    vol_t_50 = sma(volume, 50).iloc[-1]
    curr["vol_t_50"] = vol_t_50
    vol_len = int(settings.get("volume_participation_len", 5))
    vol_len = max(1, min(vol_len, len(volume)))
    if len(volume) >= 1:
        curr["vol_5"] = float(volume.iloc[-vol_len:].mean())
    else:
        curr["vol_5"] = float("nan")

    # ── Crossover detection (shared helper) ────────────────────────────────
    crossover_lookback = settings.get("crossover_lookback", 20)
    xo = detect_crossover(ind["fast_ma"], ind["slow_ma"], crossover_lookback)
    curr["ma_crossed_above"] = xo["crossed"]
    curr["crossover_bars_ago"] = xo["bars_ago"] if xo["crossed"] else -1
    curr["crossover_count"] = xo["count"]
    curr["crossover_dates"] = xo["dates"]
    curr["crossover_level"] = xo["level"]
    curr["close_above_crossover"] = (
        xo["level"] is not None and curr["close"] > xo["level"]
    )

    # ── Weekly HMA higher-timeframe check ──────────────────────────────────
    weekly = _compute_weekly_hma(df)

    # ── Sideways filter ────────────────────────────────────────────────────
    sideways = _compute_sideways(df, ind["adx_val"], settings)
    curr["is_sideways"] = sideways["is_sideways"]

    # ── Per-category scoring ───────────────────────────────────────────────
    trend_score = _score_trend(curr)
    mom_score = _score_momentum(curr)
    rsi_score = _score_rsi(curr)
    macd_score = _score_macd(curr)
    stoch_score = _score_stochastic(curr)
    obv_score = _score_obv(curr)
    vol_score = _score_volume(curr)
    rs_score = _score_relative_strength(curr, close, index_df, settings.get("rs_length", 14))
    volat_score, atr_pct, volat_stat = _score_volatility(curr)
    fund_score, fund_detail = _score_fundamentals(df)

    # ── Total ──────────────────────────────────────────────────────────────
    total = (trend_score + mom_score + rsi_score + macd_score + stoch_score
             + obv_score + vol_score + rs_score + volat_score + fund_score)
    total = max(0.0, min(total, 100.0))

    # ── Build result ───────────────────────────────────────────────────────
    return {
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
        # Key signals
        "ma_bullish": curr["ma_bullish"],
        "close_above_both_ma": curr["close_above_both_ma"],
        "ma_crossed_above": curr["ma_crossed_above"],
        "crossover_bars_ago": curr["crossover_bars_ago"],
        "crossover_count": curr["crossover_count"],
        "crossover_dates": curr["crossover_dates"],
        "above_poc": curr["above_poc"],
        "vp_poc": round(curr["vp_poc"], 2),
        # Weekly HMA(44) x EMA(50) higher-timeframe condition
        "weekly_hma_bull": weekly["bull"],
        "weekly_hma_cross": weekly["cross"],
        "weekly_hma_cross_bars_ago": weekly["cross_bars_ago"],
        # Sideways filter info
        "is_sideways": sideways["is_sideways"],
        "sideways_reasons": sideways["reasons"],
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
        # Combined rating
        "combined_rating": _get_combined_rating(
            total, curr["ma_bullish"], curr["above_poc"], curr["close_above_both_ma"]
        ),
        # Swing-trading ENTRY signal (ADX gate mirrors the backtest engine:
        # backtest skips entries when ADX < min_adx_entry at the signal bar,
        # and NaN ADX fails the gate there just as it does here)
        "entry_signal": bool(
            curr["ma_crossed_above"]
            and curr["close_above_crossover"]
            and curr["above_poc"]
            and total >= 50
            and (min_adx_entry <= 0
                 or (not np.isnan(curr["adx"]) and curr["adx"] >= min_adx_entry))
        ),
        # Weekly HMA buy trigger
        "weekly_entry_signal": bool(weekly["cross"]),
    }
