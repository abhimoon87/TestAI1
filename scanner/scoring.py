"""
12-category scoring engine.
Mirrors the Pine Script HMAxEMA Swing Trading System scoring logic + new categories.

Scoring categories (12 total, max 120 pts):
  1. Trend (15 pts) — HMA/EMA crossover, close above MA, ADX
  2. Momentum (15 pts) — 1M/3M price change
  3. RSI (8 pts) — RSI(14) in 40-70 range
  4. MACD (7 pts) — MACD histogram positive/increasing
  5. Stochastic (5 pts) — Stoch K in 20-80 range
  6. OBV (5 pts) — OBV above SMA(20), rising
  7. Volume (10 pts) — Volume > SMA(20), >1.2x, >50-bar MA
  8. Relative Strength (10 pts) — Stock returns vs NIFTY 50
  9. Volatility (5 pts) — ATR-based (Medium/Low = pass)
  10. Fundamentals (20 pts) — P/E, EPS growth, Rev growth, ROE
  11. Sentiment (8 pts) — News sentiment from MarketAux/NewsAPI/GNews
  12. Social (5 pts) — Reddit/Twitter social momentum

Refactored into composable helpers:
  - detect_crossover() — shared MA crossover detection (used by filter & scorer)
  - _compute_indicators() — compute all core indicators at once
  - _compute_weekly_hma() — weekly higher-timeframe HMA crossover
  - _compute_sideways() — ADX / Cholangirong / Slope sideways filter
  - _score_trend .. _score_social — per-category scoring (12 total)
  - compute_scores() — orchestrator that composes the above
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import math
from .indicators import (
    hull_ma, ema, sma, vwma, kama, rsi, macd, stochastic,
    obv, atr, adx, price_change, highest, lowest, volume_profile_poc
)


# ══════════════════════════════════════════════════════════════════════════════
# MOVING AVERAGE SELECTOR
# ══════════════════════════════════════════════════════════════════════════════

def get_ma(ma_type: str, src: pd.Series, length: int,
           volume: Optional[pd.Series] = None) -> pd.Series:
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

def to_weekly(df: pd.DataFrame) -> Optional[pd.DataFrame]:
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
        d.index = d.index.tz_convert(None)
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
    if len(fast_ma) < 2:
        return result
    lb = min(lookback, len(fast_ma) - 1)
    for i in range(1, lb + 1):
        ic, ip = -i, -i - 1
        if ip < -len(fast_ma):
            break
        fc, fp = fast_ma.iloc[ic], fast_ma.iloc[ip]
        sc, sp = slow_ma.iloc[ic], slow_ma.iloc[ip]
        if (not np.isnan(fc) and not np.isnan(fp) and
                not np.isnan(sc) and not np.isnan(sp)):
            if fc > sc and fp <= sp:
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
                 crossover_lookback: int = 20) -> Optional[dict]:
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

def get_direction(filter_result: Optional[dict]) -> Optional[str]:
    """
    Model 2 — Bullish / Bearish classification.

    Args:
        filter_result: Output dict from check_filter().

    Returns:
        "Bull" if the current Fast MA is above Slow MA (confirmed by a
        recent bullish crossover in check_filter),
        "Bear" otherwise.
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
    Evaluate the weekly HMA(44) x EMA(50) higher-timeframe crossover.

    Matches the TradingView condition:
      hull_ma(close, 44) crossed_above ema(close, 50) on weekly bars.

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

    reasons = []

    # ADX filter
    adx_last = adx_val.iloc[-1]
    is_sideways_adx = adx_last < adx_threshold if not np.isnan(adx_last) else False
    if is_sideways_adx:
        reasons.append("ADX")

    # Cholangirong filter
    atr1 = atr(high, low, close, 1)
    chop_sum = atr1.rolling(chop_len).sum()
    chop_range = high.rolling(chop_len).max() - low.rolling(chop_len).min()
    chop_safe_range = chop_range.replace(0, np.nan)
    chop_val = 100 * np.log10(chop_sum / chop_safe_range) / math.log10(max(chop_len, 2))
    is_sideways_chop = chop_val.iloc[-1] > chop_threshold if not np.isnan(chop_val.iloc[-1]) else False
    if is_sideways_chop:
        reasons.append("Chop")

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
    if is_sideways_slope:
        reasons.append("Slope")

    return {"is_sideways": is_sideways_adx or is_sideways_chop or is_sideways_slope,
            "reasons": reasons}


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
    if not np.isnan(curr["pc3m"]):
        if curr["pc3m"] > 0:
            s += min(8.0, 8.0 * (curr["pc3m"] / 10.0))
    return min(s, 15.0)


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
    """Category 7: VOLUME (max 10 pts)."""
    s = 0.0
    if not np.isnan(curr["vol_ma"]) and curr["vol_ma"] > 0:
        if curr["volume"] > curr["vol_ma"]:
            s += 5.0
        if curr["volume"] > curr["vol_ma"] * 1.2:
            s += 3.0
        vol_t = curr.get("vol_t_50")
        if vol_t is not None and not np.isnan(vol_t) and curr["volume"] > vol_t:
            s += 2.0
    return min(s, 10.0)


def _score_relative_strength(curr: dict, close: pd.Series,
                             index_df: Optional[pd.DataFrame],
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
# CATEGORY 11: SENTIMENT (max 8 pts)
# ══════════════════════════════════════════════════════════════════════════════

def _score_sentiment(ticker: str, settings: dict) -> tuple[float, dict]:
    """
    Category 11: SENTIMENT (max 8 pts).
    
    Fetches news sentiment from MarketAux/NewsAPI/GNews.
    Score based on sentiment_score (-1.0 to 1.0) and article_count.
    
    Returns:
        (score, detail_dict)
    """
    sentiment_score = settings.get("_sentiment_score", 0.0)
    article_count = settings.get("_article_count", 0)
    sentiment_source = settings.get("_sentiment_source", "none")

    if sentiment_source == "none" or article_count == 0:
        return 0.0, {"sentiment": "N/A", "source": "none", "articles": 0}

    # Score based on sentiment (-1.0 to 1.0) mapped to 0-8 pts
    # Positive sentiment = higher score, negative = lower
    # Also factor in article count (more articles = more confidence)
    if sentiment_score > 0.3:
        base_score = 6.0 + min(sentiment_score * 2, 2.0)  # 6-8 pts
    elif sentiment_score > 0.1:
        base_score = 4.0 + sentiment_score * 10  # 4-6 pts
    elif sentiment_score > -0.1:
        base_score = 2.0 + (sentiment_score + 0.1) * 10  # 2-4 pts
    elif sentiment_score > -0.3:
        base_score = 1.0 + (sentiment_score + 0.3) * 5  # 1-2 pts
    else:
        base_score = max(0.0, 1.0 + sentiment_score * 3)  # 0-1 pts

    # Boost if many articles (more confidence)
    if article_count >= 10:
        base_score = min(base_score * 1.1, 8.0)

    detail = {
        "sentiment": f"{sentiment_score:+.2f}",
        "source": sentiment_source,
        "articles": article_count,
    }

    return min(base_score, 8.0), detail


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 12: SOCIAL (max 5 pts)
# ══════════════════════════════════════════════════════════════════════════════

def _score_social(ticker: str, settings: dict) -> tuple[float, dict]:
    """
    Category 12: SOCIAL (max 5 pts).
    
    Fetches social sentiment from Reddit + Twitter/X.
    Score based on social_score (-1.0 to 1.0) and mention_count.
    
    Returns:
        (score, detail_dict)
    """
    social_score = settings.get("_social_score", 0.0)
    mention_count = settings.get("_mention_count", 0)
    social_source = settings.get("_social_source", "none")

    if social_source == "none" or mention_count == 0:
        return 0.0, {"social": "N/A", "source": "none", "mentions": 0}

    # Score based on social sentiment (-1.0 to 1.0) mapped to 0-5 pts
    if social_score > 0.2:
        base_score = 3.0 + min(social_score * 5, 2.0)  # 3-5 pts
    elif social_score > 0:
        base_score = 2.0 + social_score * 10  # 2-3 pts
    elif social_score > -0.2:
        base_score = 1.0 + (social_score + 0.2) * 5  # 1-2 pts
    else:
        base_score = max(0.0, 1.0 + social_score * 3)  # 0-1 pts

    # Boost for high mention count (viral = more signal)
    if mention_count >= 20:
        base_score = min(base_score * 1.2, 5.0)

    detail = {
        "social": f"{social_score:+.2f}",
        "source": social_source,
        "mentions": mention_count,
    }

    return min(base_score, 5.0), detail


# ══════════════════════════════════════════════════════════════════════════════
# INSIDER BOOST/PENALTY (applied to Fundamentals category)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_insider_adjustment(fund_score: float, settings: dict) -> tuple[float, str]:
    """
    Adjust Fundamentals score based on insider trading data.
    
    Insider buying = +3 pts (max), insider selling = -3 pts (min).
    
    Returns:
        (adjusted_score, adjustment_detail)
    """
    insider_score = settings.get("_insider_score", 0.0)
    insider_source = settings.get("_insider_source", "none")

    if insider_source == "none":
        return fund_score, "N/A"

    # Map insider_score (-1.0 to 1.0) to adjustment (-3 to +3)
    adjustment = insider_score * 3.0

    adjusted = max(0.0, min(fund_score + adjustment, 20.0))

    if adjustment > 0:
        detail = f"Insider buying (+{adjustment:.1f})"
    elif adjustment < 0:
        detail = f"Insider selling ({adjustment:.1f})"
    else:
        detail = "Neutral insider"

    return adjusted, detail


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 13: DELIVERY QUALITY (max 3 pts)
# ══════════════════════════════════════════════════════════════════════════════

def _score_delivery_quality(ticker: str, settings: dict) -> tuple[float, dict]:
    """
    Category 13: DELIVERY QUALITY (max 3 pts).
    
    High delivery % indicates institutional/strong hands buying.
    Low delivery % indicates speculative/weak hands activity.
    
    Returns:
        (score, detail_dict)
    """
    delivery_pct = settings.get("_delivery_pct", 0.0)
    delivery_change = settings.get("_delivery_change_pct", 0.0)
    source = settings.get("_delivery_source", "none")

    if source == "none" or delivery_pct == 0:
        return 0.0, {"delivery": "N/A", "source": "none"}

    # Score based on delivery %
    if delivery_pct > 70:
        base_score = 3.0  # Very high delivery = strong conviction
    elif delivery_pct > 60:
        base_score = 2.5  # High delivery = institutional buying
    elif delivery_pct > 50:
        base_score = 1.5  # Moderate delivery
    elif delivery_pct > 40:
        base_score = 0.5  # Low delivery
    else:
        base_score = 0.0  # Very low = speculative

    # Bonus for improving delivery trend
    if delivery_change > 5:
        base_score = min(base_score + 0.5, 3.0)
    elif delivery_change < -5:
        base_score = max(base_score - 0.5, 0.0)

    detail = {
        "delivery": f"{delivery_pct:.1f}%",
        "change": f"{delivery_change:+.1f}%",
        "source": source,
    }

    return min(base_score, 3.0), detail


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 14: INSTITUTIONAL FLOW (max 3 pts)
# ══════════════════════════════════════════════════════════════════════════════

def _score_institutional_flow(ticker: str, settings: dict) -> tuple[float, dict]:
    """
    Category 14: INSTITUTIONAL FLOW (max 3 pts).
    
    FII (Foreign Institutional Investors) buying = positive signal
    DII (Domestic Institutional Investors) buying = positive signal
    
    Returns:
        (score, detail_dict)
    """
    fii_is_buying = settings.get("_fii_is_buying", None)
    dii_is_buying = settings.get("_dii_is_buying", None)
    fii_net = settings.get("_fii_net", 0.0)
    dii_net = settings.get("_dii_net", 0.0)
    source = settings.get("_institutional_source", "none")

    if source == "none":
        return 0.0, {"institutional": "N/A", "source": "none"}

    base_score = 0.0

    # FII buying signal
    if fii_is_buying is True:
        base_score += 1.5
    elif fii_is_buying is False:
        base_score -= 0.5

    # DII buying signal
    if dii_is_buying is True:
        base_score += 1.0
    elif dii_is_buying is False:
        base_score -= 0.3

    # Both buying = strong signal
    if fii_is_buying is True and dii_is_buying is True:
        base_score = min(base_score + 0.5, 3.0)

    base_score = max(0.0, min(base_score, 3.0))

    fii_status = "Buying" if fii_is_buying else ("Selling" if fii_is_buying is False else "N/A")
    dii_status = "Buying" if dii_is_buying else ("Selling" if dii_is_buying is False else "N/A")

    detail = {
        "institutional": f"FII: {fii_status}, DII: {dii_status}",
        "source": source,
    }

    return base_score, detail


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 15: 52-WEEK POSITION (max 2 pts)
# ══════════════════════════════════════════════════════════════════════════════

def _score_52week_position(ticker: str, settings: dict) -> tuple[float, dict]:
    """
    Category 15: 52-WEEK POSITION (max 2 pts).
    
    Stocks near 52-week highs show strength.
    Stocks near 52-week lows may be value opportunities or falling knives.
    
    Returns:
        (score, detail_dict)
    """
    position = settings.get("_52w_position", 50.0)  # 0 = at low, 100 = at high
    pct_from_high = settings.get("_52w_pct_from_high", 0.0)
    source = settings.get("_52w_source", "none")

    if source == "none":
        return 0.0, {"week52": "N/A", "source": "none"}

    # Score based on position in range
    if position > 80:
        base_score = 2.0  # Near 52-week high = strong momentum
    elif position > 60:
        base_score = 1.5  # Upper range = healthy trend
    elif position > 40:
        base_score = 1.0  # Middle range = neutral
    elif position > 20:
        base_score = 0.5  # Lower range = weakness
    else:
        base_score = 0.0  # Near 52-week low = avoid

    detail = {
        "week52": f"Position: {position:.0f}%",
        "from_high": f"{pct_from_high:+.1f}%",
        "source": source,
    }

    return base_score, detail


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 16: VALUATION QUALITY (max 2 pts)
# ══════════════════════════════════════════════════════════════════════════════

def _score_valuation_quality(ticker: str, settings: dict) -> tuple[float, dict]:
    """
    Category 16: VALUATION QUALITY (max 2 pts).
    
    Compares stock valuation vs industry peers.
    
    Returns:
        (score, detail_dict)
    """
    pe_relative = settings.get("_pe_relative_to_industry", None)
    is_quality = settings.get("_is_quality_stock", None)
    source = settings.get("_valuation_source", "none")

    if source == "none":
        return 0.0, {"valuation": "N/A", "source": "none"}

    base_score = 0.0

    # Cheap vs peers
    if pe_relative is not None:
        if pe_relative < 0.8:
            base_score += 1.0  # Significantly cheaper
        elif pe_relative < 1.0:
            base_score += 0.5  # Slightly cheaper
        elif pe_relative > 1.5:
            base_score -= 0.5  # Expensive

    # Quality stock (ROE > 15%)
    if is_quality is True:
        base_score += 1.0

    base_score = max(0.0, min(base_score, 2.0))

    pe_status = f"{pe_relative:.2f}x" if pe_relative else "N/A"
    quality_status = "Yes" if is_quality else ("No" if is_quality is False else "N/A")

    detail = {
        "valuation": f"PE vs Industry: {pe_status}, Quality: {quality_status}",
        "source": source,
    }

    return base_score, detail


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 17: COMMODITY EXPOSURE (max 2 pts)
# ══════════════════════════════════════════════════════════════════════════════

def _score_commodity_exposure(ticker: str, settings: dict) -> tuple[float, dict]:
    """
    Category 17: COMMODITY EXPOSURE (max 2 pts).
    
    Commodity price trends affect commodity-linked stocks.
    Rising commodity prices = bullish for commodity producers.
    
    Returns:
        (score, detail_dict)
    """
    commodity_trend = settings.get("_commodity_trend", None)  # "up", "down", "neutral"
    commodity_source = settings.get("_commodity_source", "none")

    if commodity_source == "none":
        return 0.0, {"commodity": "N/A", "source": "none"}

    base_score = 0.0

    if commodity_trend == "up":
        base_score = 2.0  # Rising commodities = bullish for producers
    elif commodity_trend == "neutral":
        base_score = 1.0  # Neutral = moderate
    elif commodity_trend == "down":
        base_score = 0.0  # Falling commodities = bearish

    detail = {
        "commodity": f"Trend: {commodity_trend or 'N/A'}",
        "source": commodity_source,
    }

    return base_score, detail


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 18: FOREX IMPACT (max 2 pts)
# ══════════════════════════════════════════════════════════════════════════════

def _score_forex_impact(ticker: str, settings: dict) -> tuple[float, dict]:
    """
    Category 18: FOREX IMPACT (max 2 pts).
    
    INR depreciation = negative for importers, positive for exporters.
    INR appreciation = positive for importers, negative for exporters.
    
    Returns:
        (score, detail_dict)
    """
    inr_change_1d = settings.get("_inr_change_1d", 0.0)  # % change
    inr_change_1w = settings.get("_inr_change_1w", 0.0)
    forex_source = settings.get("_forex_source", "none")

    if forex_source == "none":
        return 0.0, {"forex": "N/A", "source": "none"}

    base_score = 0.0

    # For Indian stocks, weaker INR is generally negative
    # (higher import costs, inflationary)
    if inr_change_1d < -0.5:
        base_score = 0.0  # Sharp INR depreciation = negative
    elif inr_change_1d < 0:
        base_score = 0.5  # Mild depreciation
    elif inr_change_1d == 0:
        base_score = 1.0  # Stable
    elif inr_change_1d < 0.5:
        base_score = 1.5  # Mild appreciation
    else:
        base_score = 2.0  # Strong INR appreciation = positive

    detail = {
        "forex": f"INR 1D: {inr_change_1d:+.2f}%, 1W: {inr_change_1w:+.2f}%",
        "source": forex_source,
    }

    return base_score, detail


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 19: ESG / SUSTAINABILITY (max 2 pts)
# ══════════════════════════════════════════════════════════════════════════════

def _score_esg(ticker: str, settings: dict) -> tuple[float, dict]:
    """
    Category 19: ESG / SUSTAINABILITY (max 2 pts).
    
    Companies with strong ESG profiles tend to outperform long-term.
    
    Returns:
        (score, detail_dict)
    """
    esg_score = settings.get("_esg_score", None)  # 0-100
    carbon_intensity = settings.get("_carbon_intensity", None)  # tons CO2/revenue
    esg_source = settings.get("_esg_source", "none")

    if esg_source == "none":
        return 0.0, {"esg": "N/A", "source": "none"}

    base_score = 0.0

    if esg_score is not None:
        if esg_score > 70:
            base_score = 2.0  # Strong ESG
        elif esg_score > 50:
            base_score = 1.5  # Moderate ESG
        elif esg_score > 30:
            base_score = 1.0  # Below average
        else:
            base_score = 0.0  # Poor ESG
    elif carbon_intensity is not None:
        # Lower carbon = better
        if carbon_intensity < 10:
            base_score = 1.5
        elif carbon_intensity < 50:
            base_score = 1.0
        else:
            base_score = 0.5

    detail = {
        "esg": f"Score: {esg_score or 'N/A'}, Carbon: {carbon_intensity or 'N/A'}",
        "source": esg_source,
    }

    return base_score, detail


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 20: SHARIAH COMPLIANCE (max 2 pts)
# ══════════════════════════════════════════════════════════════════════════════

def _score_shariah(ticker: str, settings: dict) -> tuple[float, dict]:
    """
    Category 20: SHARIAH COMPLIANCE (max 2 pts).
    
    Shariah-compliant stocks appeal to Islamic investors.
    Non-compliant stocks may have debt-based revenue concerns.
    
    Returns:
        (score, detail_dict)
    """
    is_shariah = settings.get("_is_shariah_compliant", None)
    shariah_source = settings.get("_shariah_source", "none")

    if shariah_source == "none":
        return 0.0, {"shariah": "N/A", "source": "none"}

    base_score = 0.0

    if is_shariah is True:
        base_score = 2.0  # Shariah compliant = full points
    elif is_shariah is False:
        base_score = 0.5  # Non-compliant = partial (still investable)
    else:
        base_score = 1.0  # Unknown = neutral

    detail = {
        "shariah": f"Compliant: {'Yes' if is_shariah else ('No' if is_shariah is False else 'Unknown')}",
        "source": shariah_source,
    }

    return base_score, detail


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


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: SCORE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def compute_scores(df: pd.DataFrame, timeframe: str = "D",
                   index_df: Optional[pd.DataFrame] = None,
                   settings: Optional[dict] = None) -> Optional[dict]:
    """
    Compute the 20-category score for a stock.

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
        timeframe: Analysis timeframe ('D' daily, 'W' weekly, 'M' monthly).
        index_df: Index DataFrame for relative strength comparison.
        settings: Dict with scoring parameters (see DEFAULT_SETTINGS in settings_store.py).
                  Falls back to defaults if not provided.

    Scoring categories (20 total, max 138 pts):
      1. Trend (15 pts) — HMA/EMA crossover, close above MA, ADX
      2. Momentum (15 pts) — 1M/3M price change
      3. RSI (8 pts) — RSI(14) in 40-70 range
      4. MACD (7 pts) — MACD histogram positive/increasing
      5. Stochastic (5 pts) — Stoch K in 20-80 range
      6. OBV (5 pts) — OBV above SMA(20), rising
      7. Volume (10 pts) — Volume > SMA(20), >1.2x, >50-bar MA
      8. Relative Strength (10 pts) — Stock returns vs NIFTY 50
      9. Volatility (5 pts) — ATR-based (Medium/Low = pass)
      10. Fundamentals (20 pts) — P/E, EPS growth, Rev growth, ROE
      11. Sentiment (8 pts) — News sentiment (MarketAux/NewsAPI/GNews)
      12. Social (5 pts) — Reddit/Twitter/WSB social momentum
      13. Delivery Quality (3 pts) — High delivery % = institutional buying
      14. Institutional Flow (3 pts) — FII/DII buying signals
      15. 52-Week Position (2 pts) — Position in 52-week range
      16. Valuation Quality (2 pts) — PE vs industry, quality metrics
      17. Commodity Exposure (2 pts) — Commodity price trends
      18. Forex Impact (2 pts) — INR/USD exchange rate impact
      19. ESG (2 pts) — Environmental/Sustainability scores
      20. Shariah (2 pts) — Shariah compliance screening

    Returns:
        Dictionary with all scores and metadata, or None if insufficient data.
    """
    if settings is None:
        settings = {}

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
    vp_lookback = settings.get("vp_lookback", 200)
    vp_bars = max(int(vp_lookback), 10)
    vp_bars = min(vp_bars, len(df))
    vp_poc = volume_profile_poc(high, low, close, volume, lookback=vp_bars)
    curr["vp_poc"] = vp_poc.iloc[-1] if not np.isnan(vp_poc.iloc[-1]) else close.iloc[-1]
    curr["above_poc"] = curr["close"] >= curr["vp_poc"]
    curr["ma_bullish"] = curr["fast_ma"] > curr["slow_ma"]
    curr["close_above_both_ma"] = curr["close"] > curr["fast_ma"] and curr["close"] > curr["slow_ma"]

    # ── 50-bar volume MA (used by _score_volume) ──────────────────────────
    vol_t_50 = sma(volume, 50).iloc[-1]
    curr["vol_t_50"] = vol_t_50

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

    # ── Insider adjustment to Fundamentals ─────────────────────────────────
    fund_score, insider_detail = _apply_insider_adjustment(fund_score, settings)

    # ── Sentiment scoring (Category 11) ───────────────────────────────────
    ticker = settings.get("_ticker", "")
    sentiment_score, sentiment_detail = _score_sentiment(ticker, settings)

    # ── Social scoring (Category 12) ──────────────────────────────────────
    social_score, social_detail = _score_social(ticker, settings)

    # ── Delivery Quality scoring (Category 13) ────────────────────────────
    delivery_score, delivery_detail = _score_delivery_quality(ticker, settings)

    # ── Institutional Flow scoring (Category 14) ──────────────────────────
    institutional_score, institutional_detail = _score_institutional_flow(ticker, settings)

    # ── 52-Week Position scoring (Category 15) ────────────────────────────
    week52_score, week52_detail = _score_52week_position(ticker, settings)

    # ── Valuation Quality scoring (Category 16) ────────────────────────────
    valuation_score, valuation_detail = _score_valuation_quality(ticker, settings)

    # ── Commodity Exposure scoring (Category 17) ───────────────────────────
    commodity_score, commodity_detail = _score_commodity_exposure(ticker, settings)

    # ── Forex Impact scoring (Category 18) ─────────────────────────────────
    forex_score, forex_detail = _score_forex_impact(ticker, settings)

    # ── ESG scoring (Category 19) ──────────────────────────────────────────
    esg_score, esg_detail = _score_esg(ticker, settings)

    # ── Shariah scoring (Category 20) ──────────────────────────────────────
    shariah_score, shariah_detail = _score_shariah(ticker, settings)

    # ── Total ──────────────────────────────────────────────────────────────
    raw_total = (trend_score + mom_score + rsi_score + macd_score + stoch_score
                 + obv_score + vol_score + rs_score + volat_score + fund_score
                 + sentiment_score + social_score + delivery_score
                 + institutional_score + week52_score + valuation_score
                 + commodity_score + forex_score + esg_score + shariah_score)

    # Cap at 138 (new max with 20 categories)
    total = max(0.0, min(raw_total, 138.0))

    # ── Build result ───────────────────────────────────────────────────────
    return {
        "total": round(total, 1),
        "total_raw": round(raw_total, 1),
        "max_possible": 138.0,
        "fundamentals_available": any(v != "N/A" for v in fund_detail.values()),
        # Original 10 categories
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
        # Categories 11-12
        "sentiment": round(sentiment_score, 1),
        "social":    round(social_score, 1),
        # Categories 13-16 (free API categories)
        "delivery":      round(delivery_score, 1),
        "institutional": round(institutional_score, 1),
        "week52":        round(week52_score, 1),
        "valuation":     round(valuation_score, 1),
        # Categories 17-20 (new public-apis categories)
        "commodity": round(commodity_score, 1),
        "forex":     round(forex_score, 1),
        "esg":       round(esg_score, 1),
        "shariah":   round(shariah_score, 1),
        # Detail for new categories
        "sentiment_detail": sentiment_detail,
        "social_detail": social_detail,
        "insider_detail": insider_detail,
        "delivery_detail": delivery_detail,
        "institutional_detail": institutional_detail,
        "week52_detail": week52_detail,
        "valuation_detail": valuation_detail,
        "commodity_detail": commodity_detail,
        "forex_detail": forex_detail,
        "esg_detail": esg_detail,
        "shariah_detail": shariah_detail,
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
        # Combined rating (now uses 130 max scale)
        "combined_rating": _get_combined_rating(
            total, curr["ma_bullish"], curr["above_poc"], curr["close_above_both_ma"]
        ),
        # Swing-trading ENTRY signal
        "entry_signal": bool(
            curr["ma_crossed_above"]
            and curr["close_above_crossover"]
            and curr["above_poc"]
            and total >= 50
        ),
        # Weekly HMA buy trigger
        "weekly_entry_signal": bool(weekly["cross"]),
    }
