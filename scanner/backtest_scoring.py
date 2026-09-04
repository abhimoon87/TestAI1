"""Score computation for single-bar evaluation in the backtest engine."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from .backtest_models import StockData
from .scoring import detect_crossover, score_bar

logger = logging.getLogger(__name__)


def compute_score_at_bar(stock: StockData, bar_idx: int,
                         nifty_df: pd.DataFrame | None,
                         settings: dict) -> dict | None:
    """
    Compute the 10-category score at a specific bar index.
    Returns None if data is insufficient.
    """
    df = stock.df
    if bar_idx < 0 or bar_idx >= len(df):
        return None

    close = df["close"]
    volume = df["volume"]

    # Extract all current values
    fast_ma = stock.fast_ma.iloc[bar_idx]
    slow_ma = stock.slow_ma.iloc[bar_idx]
    rsi_val = stock.rsi_val.iloc[bar_idx]
    macd_h = stock.macd_hist.iloc[bar_idx]
    macd_h_prev = stock.macd_hist.iloc[bar_idx - 1] if bar_idx > 0 else np.nan
    stoch_k = stock.stoch_k.iloc[bar_idx]
    obv = stock.obv_val.iloc[bar_idx]
    obv_prev = stock.obv_val.iloc[bar_idx - 1] if bar_idx > 0 else np.nan
    obv_ma = stock.obv_ma.iloc[bar_idx]
    vol_ma = stock.vol_ma.iloc[bar_idx]
    atr_val = stock.atr_val.iloc[bar_idx]
    adx_val = stock.adx_val.iloc[bar_idx]
    vol_50 = stock.vol_50.iloc[bar_idx]
    vp_poc = stock.vp_poc.iloc[bar_idx]

    if np.isnan(fast_ma) or np.isnan(slow_ma):
        return None

    close_val = close.iloc[bar_idx]
    vol_val = volume.iloc[bar_idx]

    # --- Crossover detection (look back from this bar) ---
    lookback = settings["crossover_lookback"]
    fast_series = stock.fast_ma.iloc[: bar_idx + 1]
    slow_series = stock.slow_ma.iloc[: bar_idx + 1]
    xo = detect_crossover(fast_series, slow_series, lookback)
    ma_crossed_above = xo["crossed"]
    crossover_level = xo["level"]
    crossover_bars_ago = xo["bars_ago"] if xo["crossed"] else -1
    close_above_crossover = (
        crossover_level is not None and close_val > crossover_level
    )

    # --- MA conditions ---
    ma_bullish = fast_ma > slow_ma
    close_above_both = close_val > fast_ma and close_val > slow_ma
    above_poc = not np.isnan(vp_poc) and close_val >= vp_poc

    # --- Sideways filter (direction-aware, mirrors scoring.py) ---
    adx_weak = not np.isnan(adx_val) and adx_val < settings["adx_threshold"]
    chop_weak = False
    atr1 = stock.atr1
    chop_len = settings["chop_len"]
    if bar_idx >= chop_len:
        chop_sum = atr1.iloc[bar_idx - chop_len + 1: bar_idx + 1].sum()
        chop_range = (
            df["high"].iloc[bar_idx - chop_len + 1: bar_idx + 1].max()
            - df["low"].iloc[bar_idx - chop_len + 1: bar_idx + 1].min()
        )
        if chop_range > 0:
            chop_val = 100 * math.log10(chop_sum / chop_range) / math.log10(chop_len)
            chop_weak = chop_val > settings["chop_threshold"]
    slope_weak = False
    slope_ma = stock.slope_ma.iloc[: bar_idx + 1]
    lb = settings["slope_lookback"]
    if len(slope_ma) > lb and not np.isnan(slope_ma.iloc[-1]) and not np.isnan(slope_ma.iloc[-1 - lb]):
        slope_pct = abs(
            (slope_ma.iloc[-1] - slope_ma.iloc[-1 - lb])
            / slope_ma.iloc[-1 - lb]
        ) * 100
        slope_weak = slope_pct < settings["flat_threshold"]

    n_bars = bar_idx + 1
    pc_lookback = 21 if n_bars >= 100 else (4 if n_bars >= 40 else 1)
    strong_move = False
    if bar_idx >= pc_lookback and close.iloc[bar_idx - pc_lookback] > 0:
        pc_tmp = (close_val / close.iloc[bar_idx - pc_lookback] - 1) * 100
        strong_move = abs(pc_tmp) >= settings.get("sideways_strong_move_pct", 5.0)

    if strong_move:
        is_sideways = adx_weak and (chop_weak or slope_weak)
    else:
        is_sideways = adx_weak or chop_weak or slope_weak

    # --- Weekly HMA higher-timeframe check ---
    weekly_hma_bull = False
    if bar_idx >= 250 and stock.w_hma is not None and stock.w_ema50 is not None:
        try:
            if not np.isnan(stock.w_hma.iloc[-1]) and not np.isnan(stock.w_ema50.iloc[-1]):
                weekly_hma_bull = stock.w_hma.iloc[-1] > stock.w_ema50.iloc[-1]
        except Exception as e:
            logger.debug("Weekly HMA/EMA gate failed at bar %d: %s", bar_idx, e)

    # --- Price change (adaptive) ---
    n = bar_idx + 1
    if n >= 100:
        pc1m_p, pc3m_p = 21, 63
    elif n >= 40:
        pc1m_p, pc3m_p = 4, 13
    else:
        pc1m_p, pc3m_p = 1, 3

    pc1m = (
        (close_val / close.iloc[bar_idx - pc1m_p] - 1) * 100
        if bar_idx >= pc1m_p and close.iloc[bar_idx - pc1m_p] > 0
        else 0.0
    )
    pc3m = (
        (close_val / close.iloc[bar_idx - pc3m_p] - 1) * 100
        if bar_idx >= pc3m_p and close.iloc[bar_idx - pc3m_p] > 0
        else 0.0
    )

    # --- Volume participation average (used by _score_volume via curr) ---
    vol_len = int(settings.get("volume_participation_len", 5))
    vol_len = max(1, vol_len)
    vol_5 = float(volume.iloc[max(0, bar_idx - vol_len + 1): bar_idx + 1].mean())

    # ================================================================
    # SCORING — delegate to scoring.score_bar() (single source of truth)
    # ================================================================
    curr = {
        "close": close_val,
        "fast_ma": fast_ma,
        "slow_ma": slow_ma,
        "rsi": rsi_val,
        "macd_hist": macd_h,
        "macd_hist_prev": macd_h_prev,
        "stoch_k": stoch_k,
        "obv": obv,
        "obv_prev": obv_prev,
        "obv_ma": obv_ma,
        "volume": vol_val,
        "vol_ma": vol_ma,
        "atr": atr_val,
        "adx": adx_val,
        "pc1m": pc1m,
        "pc3m": pc3m,
        "ma_bullish": ma_bullish,
        "above_poc": above_poc,
        "close_above_both_ma": close_above_both,
        "ma_crossed_above": ma_crossed_above,
        "crossover_bars_ago": crossover_bars_ago,
        "vol_5": vol_5,
        "vol_t_50": vol_50,
    }

    scores = score_bar(
        curr, close, bar_idx, nifty_df,
        settings.get("rs_length", 14), fund=stock.fundamentals or None,
    )

    return {
        "total": scores["total"],
        "ma_crossed_above": ma_crossed_above,
        "crossover_level": crossover_level,
        "crossover_bars_ago": crossover_bars_ago,
        "close_above_crossover": close_above_crossover,
        "above_poc": above_poc,
        "ma_bullish": ma_bullish,
        "close_above_both": close_above_both,
        "is_sideways": is_sideways,
        "weekly_hma_bull": weekly_hma_bull,
        "rsi": rsi_val,
        "adx": adx_val,
        "atr_pct": scores["atr_pct"],
        "pc1m": pc1m,
        "pc3m": pc3m,
    }
