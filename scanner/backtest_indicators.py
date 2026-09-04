"""Indicator precomputation for the backtest engine."""

from __future__ import annotations

import pandas as pd

from .backtest_models import StockData, WARMUP_BARS
from .indicators import (
    adx,
    atr,
    ema,
    hull_ma,
    macd,
    obv,
    rsi,
    sma,
    stochastic,
    volume_profile_poc,
)
from .scoring import get_ma, to_weekly


def precompute_stock(ticker: str, df: pd.DataFrame,
                     settings: dict) -> StockData | None:
    """Precompute all technical indicators for a stock's entire history."""
    if df is None or len(df) < WARMUP_BARS:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    obv_series = obv(close, volume)

    stock = StockData(
        ticker=ticker,
        df=df,
        fast_ma=get_ma(settings["fast_ma_type"], close, settings["fast_ma_len"], volume),
        slow_ma=get_ma(settings["slow_ma_type"], close, settings["slow_ma_len"], volume),
        rsi_val=rsi(close, settings["rsi_len"]),
        macd_hist=macd(close)[2],
        stoch_k=stochastic(high, low, close),
        obv_val=obv_series,
        obv_ma=sma(obv_series, 20),
        vol_ma=sma(volume, settings["vol_ma_len"]),
        vol_50=sma(volume, 50),
        atr_val=atr(high, low, close, settings["atr_len"]),
        atr1=atr(high, low, close, 1),
        adx_val=adx(high, low, close, settings["adx_len"]),
        vp_poc=volume_profile_poc(high, low, close, volume, lookback=settings["vp_lookback"]),
        slope_ma=get_ma(settings["slope_ma_type"], close, settings["slope_ma_len"], volume),
    )

    # Precompute weekly higher-timeframe indicators
    w_df = to_weekly(df)
    if w_df is not None and len(w_df) >= 60:
        w_close = w_df["close"]
        stock.weekly_df = w_df
        stock.w_hma = hull_ma(w_close, 44)
        stock.w_ema50 = ema(w_close, 50)

    # Attach fundamentals if available
    fund = df.attrs.get("_fundamentals")
    if fund:
        stock.fundamentals = fund

    return stock


def precompute_nifty(index_df: pd.DataFrame) -> pd.DataFrame | None:
    """Precompute NIFTY index indicators for relative strength scoring."""
    if index_df is None or len(index_df) < 50:
        return None
    df = index_df.copy()
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return None
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df
