"""
Backtest engine for the HMA/EMA Multi-Score Swing Trading Strategy.

Simulates the full pipeline on historical NIFTY 50 daily data:
  Entry:  HMA(44) crosses above EMA(30) + close > crossover level + close > POC + score >= 50
  Exit:   2% stop loss -> +20% target -> 2% trailing stop after target

Usage:
    python -m scanner.backtest               # default: NIFTY 50, 3y
    python -m scanner.backtest --years 5     # custom lookback
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .data_fetcher import fetch_batch_yfinance, fetch_index_data
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
from .scoring import detect_crossover, get_ma
from .universes import FNO_STOCKS, NIFTY_50, NIFTY_BROAD

logger = logging.getLogger(__name__)

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SETTINGS = {
    "fast_ma_type": "HMA",
    "fast_ma_len": 44,
    "slow_ma_type": "EMA",
    "slow_ma_len": 30,
    "rsi_len": 14,
    "vol_ma_len": 20,
    "atr_len": 14,
    "adx_len": 14,
    "adx_threshold": 20.0,
    "chop_len": 14,
    "chop_threshold": 61.8,
    "slope_ma_type": "EMA",
    "slope_ma_len": 50,
    "slope_lookback": 10,
    "flat_threshold": 0.5,
    "vp_lookback": 200,
    "crossover_lookback": 6,
    "rs_length": 14,
    "score_threshold": 50,
    "stop_loss_pct": 2.0,
    "target_pct": 20.0,
    "trail_pct": 2.0,
    "position_size_pct": 10.0,
    "initial_capital": 1_000_000,
    "max_positions": 5,
    "max_positions_per_sector": 2,
    # Sector rotation
    "sector_rotation_enabled": False,
    "sector_rotation_lookback": 8,    # last N trades to evaluate sector
    "sector_min_momentum": 0.0,       # min sector momentum to allow entry        "sector_block_threshold": -5.0,    # block sector if momentum below this (%)
    "sector_boost_weight": 0.5,       # bonus = min(momentum * this, 15 pts)
    # ATR-based stop loss
    "atr_stop_enabled": False,
    "atr_stop_multiplier": 2.0,       # stop = entry - multiplier * ATR
    # ATR-based trailing stop
    "atr_trail_enabled": False,
    "atr_trail_multiplier": 2.5,      # trail = peak - multiplier * ATR
    # Entry gates (backtest experiments; all default OFF so behavior is unchanged)
    "index_regime_filter": False,     # only enter while NIFTY close > its EMA
    "index_regime_ema_len": 50,       # regime EMA length
    "min_adx_entry": 0.0,             # require ADX >= this at the signal bar (0 = off)
    "blocked_entry_weekdays": [],     # skip entries whose entry date falls on these weekday ints (Mon=0..Fri=4)
}

WARMUP_BARS = 260  # ~1 year of daily data for indicators to stabilize

# Sector mapping for FNO/NIFTY stocks
SECTOR_MAP = {
    # Banking & Financial
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "KOTAKBANK": "Banking",
    "AXISBANK": "Banking", "INDUSINDBK": "Banking", "SBIN": "Banking",
    "PNB": "Banking", "BANKBARODA": "Banking", "FEDERALBNK": "Banking",
    "BANDHANBNK": "Banking", "AUBANK": "Banking", "IDFCFIRSTB": "Banking",
    "CANBK": "Banking", "UNIONBANK": "Banking",
    "BAJFINANCE": "Finance", "BAJAJFINSV": "Finance", "SBICARD": "Finance",
    "SBILIFE": "Finance", "HDFCLIFE": "Finance", "HDFCAMC": "Finance",
    "LICHSGFIN": "Finance", "MUTHOOTFIN": "Finance", "MANAPPURAM": "Finance",
    "MFSL": "Finance", "SHRIRAMFIN": "Finance", "PFC": "Finance",
    "RECLTD": "Finance", "CHOLAFIN": "Finance",
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "TECHM": "IT", "LTTS": "IT", "PERSISTENT": "IT", "MPHASIS": "IT",
    "COFORGE": "IT", "KPITTECH": "IT", "TATAELXSI": "IT",
    "ZENSARTECH": "IT", "BSOFT": "IT", "BIRLASOFT": "IT",
    # Pharma
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "TORNTPHARM": "Pharma", "LUPIN": "Pharma",
    "GLENMARK": "Pharma", "ALCHEMIST": "Pharma",
    # Auto
    "MARUTI": "Auto", "M&M": "Auto", "TATAMOTORS": "Auto",
    "HEROMOTOCO": "Auto", "BAJAJ-AUTO": "Auto", "EICHERMOT": "Auto",
    "ASHOKLEY": "Auto", "TVSMOTOR": "Auto", "MOTHERSON": "Auto",
    "ESCORTS": "Auto", "BALKRISIND": "Auto",
    # Metals & Mining
    "TATASTEEL": "Metals", "HINDALCO": "Metals", "JSWSTEEL": "Metals",
    "VEDL": "Metals", "SAIL": "Metals", "NMDC": "Metals",
    "NATIONALUM": "Metals", "JINDALSTEL": "Metals",
    # Oil & Gas
    "RELIANCE": "OilGas", "ONGC": "OilGas", "BPCL": "OilGas",
    "IOC": "OilGas", "GAIL": "OilGas", "PETRONET": "OilGas",
    "TATACOMM": "OilGas",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "BRITANNIA": "FMCG",
    "NESTLEIND": "FMCG", "TATACONSUM": "FMCG", "MARICO": "FMCG",
    "DABUR": "FMCG", "GODREJCP": "FMCG", "COLPAL": "FMCG",
    "UBL": "FMCG", "RADICO": "FMCG",
    # Power & Infrastructure
    "NTPC": "Power", "POWERGRID": "Power", "TATAPOWER": "Power",
    "ADANIGREEN": "Power",
    # Real Estate
    "GODREJPROP": "Realty", "OBEROIRLTY": "Realty", "PRESTIGE": "Realty",
    "PHOENIXLTD": "Realty",
    # Cement & Materials
    "ULTRACEMCO": "Cement", "GRASIM": "Cement", "AMBUJACEM": "Cement",
    "ACC": "Cement", "DALBHARAT": "Cement",
    # Chemicals
    "TATACHEM": "Chemicals", "COROMANDEL": "Chemicals",
    "PIDILITIND": "Chemicals", "SRF": "Chemicals",
    "CHAMBLFERT": "Chemicals", "GSPL": "Chemicals",
    # Consumer
    "TITAN": "Consumer", "TRENT": "Consumer", "VOLTAS": "Consumer",
    "HAVELLS": "Consumer", "POLYCAB": "Consumer",
    # Telecom
    "BHARTIARTL": "Telecom", "IDEA": "Telecom",
    # Ports & Logistics
    "ADANIPORTS": "Infra", "CONCOR": "Infra", "DELHIVERY": "Infra",
    # Defence & Industrials
    "HAL": "Defence", "BEL": "Defence", "COCHINSHIP": "Defence",
    # Miscellaneous
    "IRCTC": "Misc", "PVRINOX": "Misc", "ZOMATO": "Misc",
    "NYKAA": "Misc", "PAYTM": "Misc",
    "LALPATHLAB": "Misc", "METROPOLIS": "Misc",
    "DIXON": "Misc", "SONACOMS": "Misc",
    "CROMPTON": "Misc",
}

# Sector color coding for reports
SECTOR_COLORS = {
    "Banking": "#3b82f6", "Finance": "#8b5cf6", "IT": "#22c55e",
    "Pharma": "#ec4899", "Auto": "#f97316", "Metals": "#94a3b8",
    "OilGas": "#64748b", "FMCG": "#eab308", "Power": "#06b6d4",
    "Realty": "#a855f7", "Cement": "#78716c", "Chemicals": "#14b8a6",
    "Consumer": "#f43f5e", "Telecom": "#6366f1", "Infra": "#84cc16",
    "Defence": "#0ea5e9", "Misc": "#6b7280",
}

def get_sector(ticker: str) -> str:
    """Get the sector for a stock ticker."""
    return SECTOR_MAP.get(ticker, "Other")


class SectorTracker:
    """
    Tracks sector performance for rotation strategy.
    
    Maintains a rolling window of recent trade results per sector and
    computes momentum scores to decide which sectors to favor or avoid.
    """

    def __init__(self, lookback: int = 8, block_threshold: float = -0.05):
        self.lookback = lookback
        self.block_threshold = block_threshold
        self.trade_history: list[TradeResult] = []  # all closed trades
        self.sector_trades: dict[str, list[TradeResult]] = {}  # per-sector
        self.decisions: list[dict] = []  # log of rotation decisions

    def record_trade(self, trade: TradeResult):
        """Record a closed trade and update sector tracking."""
        self.trade_history.append(trade)
        sec = trade.sector
        if sec not in self.sector_trades:
            self.sector_trades[sec] = []
        self.sector_trades[sec].append(trade)

    def get_sector_momentum(self, sector: str) -> float:
        """
        Compute sector momentum score from recent trades.
        
        Momentum = weighted avg of recent P&L% (newer trades weighted more).
        Range: -100 (all losses) to +100 (all wins).
        """
        if sector not in self.sector_trades:
            return 0.0  # neutral for unseen sectors

        recent = self.sector_trades[sector][-self.lookback:]
        if not recent:
            return 0.0

        # Weighted average: most recent trade has weight=1, oldest has weight=1/N
        n = len(recent)
        total_weight = 0.0
        weighted_pnl = 0.0
        for i, t in enumerate(recent):
            weight = (i + 1) / n  # newer trades get higher weight
            weighted_pnl += weight * t.pnl_pct
            total_weight += weight

        return weighted_pnl / total_weight if total_weight > 0 else 0.0

    def get_all_momenta(self) -> dict[str, float]:
        """Get momentum scores for all sectors with trades."""
        result = {}
        for sector in self.sector_trades:
            result[sector] = self.get_sector_momentum(sector)
        return result

    def is_sector_blocked(self, sector: str) -> bool:
        """Check if a sector should be blocked from new entries."""
        momentum = self.get_sector_momentum(sector)
        # Only block if sector has enough history (min 5 trades)
        trade_count = len(self.sector_trades.get(sector, []))
        if trade_count < 5:
            return False  # not enough data to block
        return momentum < self.block_threshold  # threshold in % (e.g. -5.0 means -5%)

    def get_sector_rank(self, sector: str) -> int:
        """
        Get the rank of a sector (1 = best performing, 0 = unknown).
        """
        all_mom = self.get_all_momenta()
        if not all_mom or sector not in all_mom:
            return 0  # unknown
        sorted_sectors = sorted(all_mom.keys(), key=lambda s: -all_mom[s])
        return sorted_sectors.index(sector) + 1

    def is_top_sector(self, sector: str, top_n: int = 3) -> bool:
        """Check if a sector is in the top N performing sectors."""
        rank = self.get_sector_rank(sector)
        # Only consider top if sector has enough history (min 3 trades)
        trade_count = len(self.sector_trades.get(sector, []))
        if trade_count < 3:
            return False
        return 0 < rank <= top_n

    def log_decision(self, ticker: str, sector: str, action: str,
                     momentum: float, score: float):
        """Log a rotation decision for reporting."""
        self.decisions.append({
            "ticker": ticker,
            "sector": sector,
            "action": action,
            "momentum": momentum,
            "score": score,
        })

    def get_sector_summary(self) -> dict:
        """Get summary of all sector rotations for reporting."""
        summary = {}
        all_mom = self.get_all_momenta()
        for sector, mom in sorted(all_mom.items(), key=lambda x: -x[1]):
            trades = self.sector_trades[sector]
            n = len(trades)
            wins = sum(1 for t in trades if t.pnl > 0)
            summary[sector] = {
                "trades": n,
                "wins": wins,
                "win_rate": wins / n * 100 if n > 0 else 0,
                "momentum": mom,
                "blocked": self.is_sector_blocked(sector),
                "total_pnl": sum(t.pnl for t in trades),
            }
        return summary


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Position:
    ticker: str
    entry_date: datetime
    entry_price: float
    stop_loss: float
    target_price: float
    shares: int
    hit_target: bool = False
    peak_price: float = 0.0
    trail_stop: float = 0.0
    exit_date: datetime | None = None
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    days_held: int = 0
    entry_score: float = 0.0
    sector: str = "Other"
    atr_at_entry: float = 0.0  # ATR value at entry for ATR-based trail


@dataclass
class TradeResult:
    ticker: str
    entry_date: datetime
    entry_price: float
    entry_score: float
    stop_loss: float
    target_price: float
    exit_date: datetime
    exit_price: float
    exit_reason: str
    shares: int
    pnl: float
    pnl_pct: float
    days_held: int
    investment: float = 0.0
    sector: str = "Other"


@dataclass
class StockData:
    ticker: str
    df: pd.DataFrame
    fast_ma: pd.Series = None
    slow_ma: pd.Series = None
    rsi_val: pd.Series = None
    macd_hist: pd.Series = None
    stoch_k: pd.Series = None
    obv_val: pd.Series = None
    vol_ma: pd.Series = None
    atr_val: pd.Series = None
    adx_val: pd.Series = None
    vp_poc: pd.Series = None
    fundamentals: dict = field(default_factory=dict)


# ============================================================
# INDICATOR PRECOMPUTATION
# ============================================================

def precompute_stock(ticker: str, df: pd.DataFrame,
                     settings: dict) -> StockData | None:
    """Precompute all technical indicators for a stock's entire history."""
    if df is None or len(df) < WARMUP_BARS:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    stock = StockData(
        ticker=ticker,
        df=df,
        fast_ma=get_ma(settings["fast_ma_type"], close, settings["fast_ma_len"], volume),
        slow_ma=get_ma(settings["slow_ma_type"], close, settings["slow_ma_len"], volume),
        rsi_val=rsi(close, settings["rsi_len"]),
        macd_hist=macd(close)[2],
        stoch_k=stochastic(high, low, close),
        obv_val=obv(close, volume),
        vol_ma=sma(volume, settings["vol_ma_len"]),
        atr_val=atr(high, low, close, settings["atr_len"]),
        adx_val=adx(high, low, close, settings["adx_len"]),
        vp_poc=volume_profile_poc(high, low, close, volume, lookback=settings["vp_lookback"]),
    )

    # Attach fundamentals if available
    fund = getattr(df, "_fundamentals", None)
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


# ============================================================
# SCORE COMPUTATION (for single-bar evaluation)
# ============================================================

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
    obv_ma = sma(stock.obv_val, 20).iloc[bar_idx]
    vol_ma = stock.vol_ma.iloc[bar_idx]
    atr_val = stock.atr_val.iloc[bar_idx]
    adx_val = stock.adx_val.iloc[bar_idx]
    vol_50 = sma(volume, 50).iloc[bar_idx]
    vp_poc = stock.vp_poc.iloc[bar_idx]

    if np.isnan(fast_ma) or np.isnan(slow_ma):
        return None

    close_val = close.iloc[bar_idx]
    df["high"].iloc[bar_idx]
    df["low"].iloc[bar_idx]
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
    # A strong directional move (|1M change| >= 5%) needs two independent
    # sideways evidences (weak ADX + chop/flat slope); otherwise any single
    # trigger classifies the bar as sideways.
    adx_weak = not np.isnan(adx_val) and adx_val < settings["adx_threshold"]
    chop_weak = False
    atr1 = atr(df["high"], df["low"], df["close"], 1)
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
    slope_ma = get_ma(
        settings["slope_ma_type"], close.iloc[: bar_idx + 1],
        settings["slope_ma_len"], volume.iloc[: bar_idx + 1]
    )
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
    if bar_idx >= 250:
        sub_df = df.iloc[: bar_idx + 1].copy()
        sub_df.index = pd.to_datetime(sub_df.index)
        if sub_df.index.tz is not None:
            sub_df.index = sub_df.index.tz_localize(None)
        try:
            w_agg = {
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }
            w_df = sub_df.resample("W").agg(w_agg).dropna()
            if len(w_df) >= 60:
                w_close = w_df["close"]
                w_hma = hull_ma(w_close, 44)
                w_ema50 = ema(w_close, 50)
                if not np.isnan(w_hma.iloc[-1]) and not np.isnan(w_ema50.iloc[-1]):
                    weekly_hma_bull = w_hma.iloc[-1] > w_ema50.iloc[-1]
        except Exception:
            pass

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

    # ================================================================
    # SCORING — per-bar re-implementation of scoring.py's category
    # helpers (Pine-derived; see the scoring.py module docstring for the
    # Pine-vs-Python mapping). KEEP IN SYNC with scoring.py.
    # ================================================================

    # -- 1. Trend (max 15) --
    trend = 0.0
    if ma_bullish:
        trend += 4.0
    if above_poc:
        trend += 2.5
    if close_above_both:
        trend += 2.5
    elif close_val > slow_ma:
        trend += 1.0
    if ma_crossed_above and crossover_bars_ago >= 0:
        if crossover_bars_ago <= 1:
            trend += 3.0
        elif crossover_bars_ago <= 2:
            trend += 2.0
        elif crossover_bars_ago <= 3:
            trend += 1.0
        elif crossover_bars_ago <= 4:
            trend += 0.5
        else:
            trend += 0.25
    if not np.isnan(adx_val) and adx_val > 25:
        trend += 2.0
    trend = min(trend, 15.0)

    # -- 2. Momentum (max 15) --
    momentum = 0.0
    if pc1m > 0:
        momentum += min(7.0, 7.0 * (pc1m / 5.0))
    else:
        momentum += max(-3.0, min(0.0, 7.0 * (pc1m / 10.0)))
    if pc3m > 0:
        momentum += min(8.0, 8.0 * (pc3m / 10.0))
    else:
        momentum += max(-4.0, min(0.0, 8.0 * (pc3m / 20.0)))
    momentum = max(0.0, min(momentum, 15.0))

    # -- 3. RSI (max 8) --
    rsi_s = 0.0
    if not np.isnan(rsi_val) and 40 <= rsi_val <= 70:
        rsi_s = max(0.0, min(8.0 * (1.0 - abs(rsi_val - 55.0) / 15.0), 8.0))

    # -- 4. MACD (max 7) --
    macd_s = 0.0
    if not np.isnan(macd_h):
        if macd_h > 0:
            macd_s += 4.0
        if not np.isnan(macd_h_prev) and macd_h > macd_h_prev:
            macd_s += 3.0
    macd_s = min(macd_s, 7.0)

    # -- 5. Stochastic (max 5) --
    stoch_s = 5.0 if (not np.isnan(stoch_k) and 20 < stoch_k < 80) else 0.0

    # -- 6. OBV (max 5) --
    obv_s = 0.0
    if not np.isnan(obv) and not np.isnan(obv_ma):
        if obv > obv_ma:
            obv_s += 3.0
        if not np.isnan(obv_prev) and obv > obv_prev:
            obv_s += 2.0
    obv_s = min(obv_s, 5.0)

    # -- 7. Volume (max 10) -- participation-aware, mirrors scoring.py:
    # the single-bar print is blended with a 5-bar participation average so
    # a move that accumulated volume is not zeroed by one quiet close.
    vol_s = 0.0
    if not np.isnan(vol_ma) and vol_ma > 0:
        if vol_val > vol_ma:
            vol_s += 3.0
        vol_len = int(settings.get("volume_participation_len", 5))
        vol_len = max(1, vol_len)
        vol_5 = float(volume.iloc[max(0, bar_idx - vol_len + 1): bar_idx + 1].mean())
        if not np.isnan(vol_5):
            if vol_5 > vol_ma:
                vol_s += 4.0
            if vol_5 > vol_ma * 1.2:
                vol_s += 2.0
        if not np.isnan(vol_50) and (vol_val > vol_50 or (not np.isnan(vol_5) and vol_5 > vol_50)):
            vol_s += 1.0
    vol_s = min(vol_s, 10.0)

    # -- 8. Relative Strength (max 10) -- mirrors scoring.py: the STOCK's
    # rs_length-day return is compared against the INDEX's rs_length-day
    # return (both from the same window). The old code compared the stock's
    # adaptive pc1m (21-day) against a 14-day index return, which diverged
    # from the live scanner whenever the two windows disagreed.
    rs_s = 0.0
    rs_length = settings["rs_length"]
    if nifty_df is not None and len(nifty_df) > rs_length + 5:
        idx_close = nifty_df["close"]
        # Align by date: find closest NIFTY date <= stock date
        stock_date = df.index[bar_idx]
        if isinstance(stock_date, pd.Timestamp):
            # Use the NIFTY close series indexed by date
            mask = idx_close.index <= stock_date
            if mask.sum() > rs_length and bar_idx >= rs_length:
                idx_pos = mask.sum() - 1
                idx_rs = (idx_close.iloc[idx_pos] / idx_close.iloc[idx_pos - rs_length] - 1) * 100
                stock_rs = (close.iloc[bar_idx] / close.iloc[bar_idx - rs_length] - 1) * 100
                if stock_rs > idx_rs:
                    rs_s += 5.0
                if stock_rs > 0:
                    rs_s += 5.0
            else:
                # Fallback: use stock momentum alone (mirrors scoring.py)
                if not np.isnan(pc1m) and pc1m > 0:
                    rs_s += 5.0
                if not np.isnan(pc3m) and pc3m > 0:
                    rs_s += 5.0
        else:
            if not np.isnan(pc1m) and pc1m > 0:
                rs_s += 5.0
            if not np.isnan(pc3m) and pc3m > 0:
                rs_s += 5.0
    else:
        if not np.isnan(pc1m) and pc1m > 0:
            rs_s += 5.0
        if not np.isnan(pc3m) and pc3m > 0:
            rs_s += 5.0
    rs_s = min(rs_s, 10.0)

    # -- 9. Volatility (max 5) --
    atr_pct = (atr_val / close_val * 100) if close_val > 0 else 0
    volat_s = 5.0 if atr_pct <= 3 else 0.0

    # -- 10. Fundamentals (max 20) --
    fund_s = 0.0
    fund = stock.fundamentals
    if fund:
        pe = fund.get("pe_ratio")
        eps_g = fund.get("eps_growth")
        rev_g = fund.get("rev_growth")
        roe_v = fund.get("roe")
        if pe and pe > 0:
            fund_s += 5.0 if pe < 15 else (3.0 if pe < 25 else 0.0)
        if eps_g is not None:
            fund_s += 5.0 if eps_g > 20 else (3.0 if eps_g > 0 else 0.0)
        if rev_g is not None:
            fund_s += 5.0 if rev_g > 15 else (3.0 if rev_g > 0 else 0.0)
        if roe_v is not None:
            fund_s += 5.0 if roe_v > 20 else (3.0 if roe_v > 10 else 0.0)
    fund_s = min(fund_s, 20.0)

    # --- Total ---
    total = trend + momentum + rsi_s + macd_s + stoch_s + obv_s + vol_s + rs_s + volat_s + fund_s
    total = max(0.0, min(total, 100.0))

    return {
        "total": round(total, 1),
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
        "atr_pct": atr_pct,
        "pc1m": pc1m,
        "pc3m": pc3m,
    }


# ============================================================
# POSITION MANAGEMENT
# ============================================================

def update_position(pos: Position, bar: pd.Series,
                    bar_idx: int, settings: dict) -> TradeResult | None:
    """Check if a position should be exited on this bar. Returns TradeResult if closed."""
    high = bar["high"]
    low = bar["low"]
    opn = bar["open"]
    close_val = bar["close"]
    current_date = bar.name

    # --- Check stop loss first (gap down) ---
    if low <= pos.stop_loss:
        exit_price = max(pos.stop_loss, low)
        if opn < pos.stop_loss:
            exit_price = opn  # gap down: filled at open
        return _close_position(pos, exit_price, current_date, "STOP_LOSS")

    # --- Check target ---
    atr_trail = settings.get("atr_trail_enabled", False)
    atr_mult = settings.get("atr_trail_multiplier", 2.5)

    if not pos.hit_target and high >= pos.target_price:
        pos.hit_target = True
        pos.peak_price = pos.target_price
        if atr_trail and pos.atr_at_entry > 0:
            pos.trail_stop = pos.target_price - (atr_mult * pos.atr_at_entry)
        else:
            pos.trail_stop = pos.target_price * (1 - settings["trail_pct"] / 100)
        pos.stop_loss = max(pos.stop_loss, pos.trail_stop)

    # --- Update trailing stop ---
    if pos.hit_target:
        if close_val > pos.peak_price:
            pos.peak_price = close_val
            if atr_trail and pos.atr_at_entry > 0:
                pos.trail_stop = pos.peak_price - (atr_mult * pos.atr_at_entry)
            else:
                pos.trail_stop = pos.peak_price * (1 - settings["trail_pct"] / 100)
            pos.stop_loss = max(pos.stop_loss, pos.trail_stop)

        # Check trailing stop
        if low <= pos.trail_stop:
            exit_price = max(pos.trail_stop, low)
            return _close_position(pos, exit_price, current_date, "TRAILING_STOP")

    return None


def _close_position(pos: Position, exit_price: float,
                    exit_date: datetime, reason: str) -> TradeResult:
    """Close a position and return the trade result."""
    pos.exit_price = exit_price
    pos.exit_date = exit_date
    pos.exit_reason = reason
    pos.pnl = (exit_price - pos.entry_price) * pos.shares
    pos.pnl_pct = (exit_price / pos.entry_price - 1) * 100
    pos.days_held = (exit_date - pos.entry_date).days
    return TradeResult(
        ticker=pos.ticker,
        entry_date=pos.entry_date,
        entry_price=pos.entry_price,
        entry_score=pos.entry_score,
        stop_loss=pos.stop_loss if reason == "STOP_LOSS" else 0,
        target_price=pos.target_price,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason=reason,
        shares=pos.shares,
        pnl=pos.pnl,
        pnl_pct=pos.pnl_pct,
        days_held=pos.days_held,
        investment=pos.entry_price * pos.shares,
        sector=pos.sector,
    )


# ============================================================
# BACKTEST ENGINE
# ============================================================

class BacktestEngine:
    """Run the HMA/EMA multi-score swing strategy backtest."""

    def __init__(self, settings: dict | None = None):
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.stocks: list[StockData] = []
        self.nifty_df: pd.DataFrame | None = None
        self.positions: list[Position] = []
        self.trades: list[TradeResult] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.sector_tracker = SectorTracker(
            lookback=self.settings.get("sector_rotation_lookback", 8),
            block_threshold=self.settings.get("sector_block_threshold", -0.05),
        )

    def load_data(self, tickers: list[str], period: str = "5y"):
        """Fetch and precompute indicators for all tickers."""
        print()
        print("=" * 60)
        print("  BACKTEST: HMA/EMA Multi-Score Swing Strategy")
        print("=" * 60)
        print(f"  Stocks: {len(tickers)}")
        print(f"  Period: {period} (daily)")
        print(f"  Settings: HMA({self.settings['fast_ma_len']}) x "
              f"EMA({self.settings['slow_ma_len']}), "
              f"RSI({self.settings['rsi_len']}), "
              f"Crossover lookback: {self.settings['crossover_lookback']}")
        print(f"  Risk: {self.settings['stop_loss_pct']}% stop / "
              f"{self.settings['target_pct']}% target / "
              f"{self.settings['trail_pct']}% trail")
        print("=" * 60)
        print()

        # Fetch batch data
        print("  Downloading stock data...")
        all_data = fetch_batch_yfinance(tickers, period=period, timeframe="D")

        # Fetch NIFTY index
        print("  Downloading NIFTY 50 index...")
        nifty_raw = fetch_index_data("^NSEI", period=period)
        self.nifty_df = precompute_nifty(nifty_raw)

        # Precompute indicators
        print("  Computing indicators...")
        for ticker in tickers:
            df = all_data.get(ticker)
            if df is not None:
                stock = precompute_stock(ticker, df, self.settings)
                if stock is not None:
                    self.stocks.append(stock)

        print(f"  Ready: {len(self.stocks)}/{len(tickers)} stocks loaded")
        print()

    def run(self) -> dict:
        """Run the full backtest simulation."""
        if not self.stocks:
            print("  No stocks loaded. Aborting.")
            return {}

        settings = self.settings
        capital = settings["initial_capital"]
        cash = capital
        max_pos = settings["max_positions"]
        max_per_sector = settings.get("max_positions_per_sector", 999)
        score_threshold = settings["score_threshold"]
        stop_pct = settings["stop_loss_pct"]
        target_pct = settings["target_pct"]
        pos_size_pct = settings["position_size_pct"]

        # Get aligned trading dates across all stocks
        all_dates = set()
        for stock in self.stocks:
            all_dates.update(stock.df.index)
        trading_dates = sorted(all_dates)

        if len(trading_dates) <= WARMUP_BARS:
            print("  Not enough data after warmup period. Aborting.")
            return {}

        # Skip first WARMUP_BARS days
        backtest_dates = trading_dates[WARMUP_BARS:]

        # Optional calendar-window restriction (walk-forward / OOS runs)
        sim_start = settings.get("sim_start")
        sim_end = settings.get("sim_end")
        sim_start = pd.Timestamp(sim_start) if sim_start is not None else None
        sim_end = pd.Timestamp(sim_end) if sim_end is not None else None
        if sim_start is not None:
            backtest_dates = [d for d in backtest_dates if d >= sim_start]
        if sim_end is not None:
            backtest_dates = [d for d in backtest_dates if d <= sim_end]
        if not backtest_dates:
            print("  No dates in the simulation window after filters. Aborting.")
            return {}

        print(f"  Simulation: {len(backtest_dates)} trading days")
        print(f"  {backtest_dates[0].strftime('%Y-%m-%d')} to "
              f"{backtest_dates[-1].strftime('%Y-%m-%d')}")
        print(f"  Initial capital: Rs.{capital:,.0f}")
        print()

        # -- Sector rotation config --
        rotation_enabled = settings.get("sector_rotation_enabled", False)
        rotation_lookback = settings.get("sector_rotation_lookback", 8)
        settings.get("sector_boost_weight", 1.5)
        sector_block_threshold = settings.get("sector_block_threshold", -0.05)
        self.sector_tracker.lookback = rotation_lookback
        self.sector_tracker.block_threshold = sector_block_threshold

        # -- Index regime gate config --
        regime_filter = settings.get("index_regime_filter", False)
        regime_ema_len = int(settings.get("index_regime_ema_len", 50))
        regime_close = None
        regime_ema = None
        if regime_filter:
            if self.nifty_df is not None and len(self.nifty_df) >= regime_ema_len + 5:
                regime_close = self.nifty_df["close"]
                regime_ema = get_ma("EMA", regime_close, regime_ema_len)
            else:
                print("  WARNING: index_regime_filter enabled but no usable NIFTY "
                      "index loaded -- gate disabled (fail-open)")

        # -- Main simulation loop --
        signals_generated = 0
        signals_taken = 0
        signals_blocked = 0
        signals_boosted = 0

        for day in backtest_dates:
            # 1. Check exits on open positions
            closed_today = []
            for pos in self.positions:
                stock = next((s for s in self.stocks if s.ticker == pos.ticker), None)
                if stock is None or day not in stock.df.index:
                    continue
                bar_idx = stock.df.index.get_loc(day)
                bar = stock.df.iloc[bar_idx]
                result = update_position(pos, bar, bar_idx, settings)
                if result:
                    closed_today.append(result)
                    cash += result.pnl + result.investment

            for trade in closed_today:
                self.trades.append(trade)
                # Record in sector tracker for rotation
                if rotation_enabled:
                    self.sector_tracker.record_trade(trade)
                self.positions = [p for p in self.positions if p.exit_date is None]

            # 2. Check for new entries (if we have room)
            # Index regime gate: only enter while NIFTY close > its EMA
            regime_ok = True
            if regime_ema is not None:
                _ic = regime_close.asof(day)
                _ie = regime_ema.asof(day)
                regime_ok = not (pd.isna(_ic) or pd.isna(_ie)) and _ic > _ie
            if len(self.positions) < max_pos and regime_ok:
                for stock in self.stocks:
                    if len(self.positions) >= max_pos:
                        break

                    # Skip if already in this stock
                    if any(p.ticker == stock.ticker for p in self.positions):
                        continue

                    # Sector diversification check
                    stock_sector = get_sector(stock.ticker)
                    sector_count = sum(1 for p in self.positions if p.sector == stock_sector)
                    if sector_count >= max_per_sector:
                        continue

                    if day not in stock.df.index:
                        continue

                    bar_idx = stock.df.index.get_loc(day)

                    # Check if there was a crossover in the lookback window
                    lookback = settings["crossover_lookback"]
                    if bar_idx < lookback + 1:
                        continue

                    fast_slice = stock.fast_ma.iloc[bar_idx - lookback: bar_idx + 1]
                    slow_slice = stock.slow_ma.iloc[bar_idx - lookback: bar_idx + 1]
                    xo = detect_crossover(fast_slice, slow_slice, lookback)

                    if not xo["crossed"]:
                        continue

                    signals_generated += 1
                    crossover_level = xo["level"]

                    close_val = stock.df["close"].iloc[bar_idx]
                    if crossover_level is None or close_val <= crossover_level:
                        continue  # Must close above crossover level

                    # Min-ADX gate: require a real trend at the signal bar
                    min_adx = settings.get("min_adx_entry", 0.0)
                    if min_adx > 0:
                        _adx = stock.adx_val.iloc[bar_idx]
                        if np.isnan(_adx) or _adx < min_adx:
                            continue

                    # Compute full score
                    score_result = compute_score_at_bar(
                        stock, bar_idx, self.nifty_df, settings
                    )

                    if score_result is None:
                        continue

                    # Check all entry conditions
                    if not score_result["above_poc"]:
                        continue
                    if score_result["is_sideways"]:
                        continue

                    base_score = score_result["total"]

                    # -- SECTOR ROTATION --
                    adjusted_score = base_score
                    if rotation_enabled:
                        sector_momentum = self.sector_tracker.get_sector_momentum(stock_sector)
                        is_blocked = self.sector_tracker.is_sector_blocked(stock_sector)
                        is_top = self.sector_tracker.is_top_sector(stock_sector, top_n=3)

                        if is_blocked:
                            # Skip this stock entirely — losing sector
                            self.sector_tracker.log_decision(
                                stock.ticker, stock_sector, "BLOCKED",
                                sector_momentum, base_score
                            )
                            signals_blocked += 1
                            continue

                        if is_top and sector_momentum > 0:
                            # Add fixed bonus for top sectors (capped at +15 pts)
                            bonus = min(sector_momentum * 0.5, 15.0)
                            adjusted_score = base_score + bonus
                            signals_boosted += 1
                            self.sector_tracker.log_decision(
                                stock.ticker, stock_sector, "BOOSTED",
                                sector_momentum, base_score
                            )
                        else:
                            self.sector_tracker.log_decision(
                                stock.ticker, stock_sector, "NEUTRAL",
                                sector_momentum, base_score
                            )

                    if adjusted_score < score_threshold:
                        continue

                    # -- ENTRY SIGNAL CONFIRMED --

                    # Entry on next day's open
                    next_idx = bar_idx + 1
                    if next_idx >= len(stock.df):
                        continue
                    entry_date = stock.df.index[next_idx]
                    entry_price = stock.df["open"].iloc[next_idx]

                    if np.isnan(entry_price) or entry_price <= 0:
                        continue

                    # Window gate: never enter beyond the sim end (walk-forward)
                    if sim_end is not None and entry_date > sim_end:
                        continue

                    # Weekday gate: skip entries landing on blocked weekdays
                    blocked_wd = settings.get("blocked_entry_weekdays", [])
                    if blocked_wd and entry_date.weekday() in blocked_wd:
                        continue

                    signals_taken += 1

                    # Position sizing — ATR-based or fixed percentage stop
                    atr_stop = settings.get("atr_stop_enabled", False)
                    if atr_stop:
                        atr_val = stock.atr_val.iloc[bar_idx]
                        atr_mult = settings.get("atr_stop_multiplier", 2.0)
                        if not np.isnan(atr_val) and atr_val > 0:
                            stop_loss = entry_price - (atr_mult * atr_val)
                        else:
                            stop_loss = entry_price * (1 - stop_pct / 100)  # fallback
                    else:
                        stop_loss = entry_price * (1 - stop_pct / 100)
                    target_price = entry_price * (1 + target_pct / 100)
                    risk_per_share = entry_price - stop_loss

                    # Position size based on CURRENT portfolio value (compounding)
                    # Calculate current portfolio value for position sizing
                    current_portfolio = cash
                    for p in self.positions:
                        s = next((x for x in self.stocks if x.ticker == p.ticker), None)
                        if s and day in s.df.index:
                            current_portfolio += s.df.loc[day, "close"] * p.shares
                        else:
                            current_portfolio += p.entry_price * p.shares

                    max_investment = current_portfolio * pos_size_pct / 100
                    shares = int(max_investment / entry_price)
                    if shares <= 0:
                        continue

                    # Verify risk is within budget (configurable % of current portfolio)
                    max_risk_pct = settings.get("max_risk_per_trade", 0.02)
                    total_risk = risk_per_share * shares
                    if total_risk > current_portfolio * max_risk_pct:
                        max_shares = int(current_portfolio * max_risk_pct / risk_per_share)
                        shares = max(max_shares, 1)

                    investment = entry_price * shares
                    if investment > cash:
                        continue  # Not enough cash

                    # Store ATR at entry for ATR-based trailing stop
                    atr_at_entry = stock.atr_val.iloc[bar_idx] if not np.isnan(stock.atr_val.iloc[bar_idx]) else 0.0

                    pos = Position(
                        ticker=stock.ticker,
                        entry_date=entry_date,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target_price=target_price,
                        shares=shares,
                        peak_price=entry_price,
                        trail_stop=0,
                        entry_score=score_result["total"],
                        sector=stock_sector,
                        atr_at_entry=atr_at_entry,
                    )
                    self.positions.append(pos)
                    cash -= investment

            # 3. Record equity curve
            portfolio_value = cash
            for pos in self.positions:
                stock = next((s for s in self.stocks if s.ticker == pos.ticker), None)
                if stock and day in stock.df.index:
                    current_price = stock.df.loc[day, "close"]
                    portfolio_value += current_price * pos.shares
                else:
                    portfolio_value += pos.entry_price * pos.shares

            self.equity_curve.append((day, portfolio_value))

        # Close any remaining positions at the end of the simulated window
        sim_last = backtest_dates[-1] if backtest_dates else None
        for pos in self.positions:
            stock = next((s for s in self.stocks if s.ticker == pos.ticker), None)
            if stock is None:
                continue
            if sim_last is not None and sim_last in stock.df.index:
                last_close = stock.df.loc[sim_last, "close"]
                last_date = sim_last
            else:
                last_close = stock.df["close"].iloc[-1]
                last_date = stock.df.index[-1]
            result = _close_position(pos, last_close, last_date, "END_OF_DATA")
            self.trades.append(result)
        self.positions.clear()

        # -- Compute metrics --
        return self._compute_metrics(capital, signals_generated, signals_taken,
                                     signals_blocked, signals_boosted)

    def _compute_metrics(self, initial_capital: float,
                         signals_generated: int, signals_taken: int,
                         signals_blocked: int = 0, signals_boosted: int = 0) -> dict:
        """Compute comprehensive backtest performance metrics."""
        trades = self.trades
        eq = self.equity_curve

        if not trades:
            print("  No trades executed.")
            return {"total_trades": 0}

        # Basic stats
        total_trades = len(trades)
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        win_rate = len(winners) / total_trades * 100

        avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl_pct for t in losers]) if losers else 0
        avg_win_days = np.mean([t.days_held for t in winners]) if winners else 0
        avg_loss_days = np.mean([t.days_held for t in losers]) if losers else 0

        total_pnl = sum(t.pnl for t in trades)
        total_return = (total_pnl / initial_capital) * 100

        # Profit factor
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown
        peak_val = initial_capital
        max_dd = 0
        max_dd_pct = 0
        for _, val in eq:
            peak_val = max(peak_val, val)
            dd = peak_val - val
            dd_pct = dd / peak_val * 100
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd = dd

        # Daily returns for Sharpe/Sortino
        if len(eq) > 1:
            daily_vals = [v for _, v in eq]
            daily_rets = [
                (daily_vals[i] / daily_vals[i - 1] - 1)
                for i in range(1, len(daily_vals))
                if daily_vals[i - 1] > 0
            ]
            if daily_rets:
                avg_daily = np.mean(daily_rets)
                std_daily = np.std(daily_rets)
                sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
                downside = [r for r in daily_rets if r < 0]
                downside_std = np.std(downside) if downside else 0.001
                sortino = (avg_daily / downside_std) * np.sqrt(252) if downside_std > 0 else 0
            else:
                sharpe = sortino = 0
        else:
            sharpe = sortino = 0

        # Annualized return
        years = len(eq) / 252 if len(eq) > 0 else 1
        final_val = eq[-1][1] if eq else initial_capital
        annual_return = ((final_val / initial_capital) ** (1 / years) - 1) * 100

        # Max consecutive wins/losses
        max_consec_wins = max_consec_losses = consec_wins = consec_losses = 0
        for t in trades:
            if t.pnl > 0:
                consec_wins += 1
                consec_losses = 0
                max_consec_wins = max(max_consec_wins, consec_wins)
            else:
                consec_losses += 1
                consec_wins = 0
                max_consec_losses = max(max_consec_losses, consec_losses)

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        # Best/worst trades
        best_trade = max(trades, key=lambda t: t.pnl_pct)
        worst_trade = min(trades, key=lambda t: t.pnl_pct)

        # Average score of winners vs losers
        avg_winner_score = np.mean([t.entry_score for t in winners]) if winners else 0
        avg_loser_score = np.mean([t.entry_score for t in losers]) if losers else 0

        metrics = {
            "initial_capital": initial_capital,
            "final_value": final_val,
            "total_pnl": total_pnl,
            "total_return_pct": total_return,
            "annual_return_pct": annual_return,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "avg_win_days": avg_win_days,
            "avg_loss_days": avg_loss_days,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_consec_wins": max_consec_wins,
            "max_consec_losses": max_consec_losses,
            "exit_reasons": exit_reasons,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "avg_winner_score": avg_winner_score,
            "avg_loser_score": avg_loser_score,
            "signals_generated": signals_generated,
            "signals_taken": signals_taken,
            "signal_conversion": (signals_taken / signals_generated * 100
                                  if signals_generated > 0 else 0),
            "years": years,
            "signals_blocked": signals_blocked,
            "signals_boosted": signals_boosted,
            "sector_rotation_enabled": self.settings.get("sector_rotation_enabled", False),
        }

        # Per-stock breakdown
        stock_stats = {}
        for t in trades:
            if t.ticker not in stock_stats:
                stock_stats[t.ticker] = {
                    "trades": 0, "wins": 0, "total_pnl": 0, "total_pnl_pct": 0
                }
            s = stock_stats[t.ticker]
            s["trades"] += 1
            if t.pnl > 0:
                s["wins"] += 1
            s["total_pnl"] += t.pnl
            s["total_pnl_pct"] += t.pnl_pct

        for s in stock_stats.values():
            s["win_rate"] = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
            s["avg_pnl_pct"] = s["total_pnl_pct"] / s["trades"] if s["trades"] > 0 else 0

        metrics["stock_stats"] = stock_stats

        # Sector breakdown
        sector_stats = {}
        for t in trades:
            sec = t.sector
            if sec not in sector_stats:
                sector_stats[sec] = {
                    "trades": 0, "wins": 0, "total_pnl": 0, "total_pnl_pct": 0,
                    "stocks": set(),
                }
            ss = sector_stats[sec]
            ss["trades"] += 1
            if t.pnl > 0:
                ss["wins"] += 1
            ss["total_pnl"] += t.pnl
            ss["total_pnl_pct"] += t.pnl_pct
            ss["stocks"].add(t.ticker)

        for sec, ss in sector_stats.items():
            ss["win_rate"] = ss["wins"] / ss["trades"] * 100 if ss["trades"] > 0 else 0
            ss["avg_pnl_pct"] = ss["total_pnl_pct"] / ss["trades"] if ss["trades"] > 0 else 0
            ss["stock_count"] = len(ss["stocks"])
            del ss["stocks"]  # remove set for serialization

        metrics["sector_stats"] = sector_stats
        return metrics

    def print_report(self, metrics: dict):
        """Print a formatted performance report to console."""
        if not metrics or metrics.get("total_trades", 0) == 0:
            print("  No trades to report.")
            return

        m = metrics
        sep = "=" * 60

        print()
        print(sep)
        print("  BACKTEST RESULTS")
        print(sep)
        print()

        print(f"  Period: {m['years']:.1f} years")
        print(f"  Stocks tested: {len(self.stocks)}")
        print(f"  Initial capital: Rs.{m['initial_capital']:,.0f}")
        print(f"  Final value: Rs.{m['final_value']:,.0f}")
        print()

        print(f"  {'-' * 56}")
        print("  PERFORMANCE SUMMARY")
        print(f"  {'-' * 56}")
        print(f"  Total return:     {m['total_return_pct']:>+8.1f}%")
        print(f"  Annual return:    {m['annual_return_pct']:>+8.1f}%")
        print(f"  Total P&L:        Rs.{m['total_pnl']:>+12,.0f}")
        print(f"  Max drawdown:     Rs.{m['max_drawdown']:>12,.0f} ({m['max_drawdown_pct']:.1f}%)")
        print(f"  Sharpe ratio:     {m['sharpe_ratio']:>8.2f}")
        print(f"  Sortino ratio:    {m['sortino_ratio']:>8.2f}")
        print(f"  Profit factor:    {m['profit_factor']:>8.2f}")
        print()

        print(f"  {'-' * 56}")
        print("  TRADE STATISTICS")
        print(f"  {'-' * 56}")
        print(f"  Total trades:     {m['total_trades']:>8d}")
        print(f"  Win rate:         {m['win_rate']:>8.1f}%")
        print(f"  Avg win:          {m['avg_win_pct']:>+8.1f}% ({m['avg_win_days']:.0f} days)")
        print(f"  Avg loss:         {m['avg_loss_pct']:>+8.1f}% ({m['avg_loss_days']:.0f} days)")
        print(f"  Best trade:       {m['best_trade'].ticker} {m['best_trade'].pnl_pct:+.1f}%")
        print(f"  Worst trade:      {m['worst_trade'].ticker} {m['worst_trade'].pnl_pct:+.1f}%")
        print(f"  Max consec wins:  {m['max_consec_wins']:>8d}")
        print(f"  Max consec losses:{m['max_consec_losses']:>8d}")
        print()

        print(f"  {'-' * 56}")
        print("  ENTRY SIGNALS")
        print(f"  {'-' * 56}")
        print(f"  Signals generated:{m['signals_generated']:>8d}")
        print(f"  Signals taken:    {m['signals_taken']:>8d}")
        print(f"  Conversion rate:  {m['signal_conversion']:>8.1f}%")
        print(f"  Avg winner score: {m['avg_winner_score']:>8.1f}")
        print(f"  Avg loser score:  {m['avg_loser_score']:>8.1f}")
        print()

        print(f"  {'-' * 56}")
        print("  EXIT REASONS")
        print(f"  {'-' * 56}")
        for reason, count in sorted(m["exit_reasons"].items(),
                                     key=lambda x: -x[1]):
            pct = count / m["total_trades"] * 100
            print(f"  {reason:<20s} {count:>5d} ({pct:.0f}%)")
        print()

        # Sector rotation summary
        if m.get("sector_rotation_enabled"):
            tracker = self.sector_tracker
            rot_summary = tracker.get_sector_summary()
            if rot_summary:
                print(f"  {'-' * 56}")
                print("  SECTOR ROTATION")
                print(f"  {'-' * 56}")
                print(f"  Signals blocked:  {m['signals_blocked']:>8d} (losing sectors)")
                print(f"  Signals boosted:  {m['signals_boosted']:>8d} (top sectors)")
                print()
                print(f"  {'Sector':<14s} {'Mom':>6s} {'Status':>10s} {'Trades':>6s} {'Win%':>6s}")
                print(f"  {'-' * 56}")
                for sec, info in rot_summary.items():
                    mom = info["momentum"]
                    status = "BLOCKED" if info["blocked"] else ("TOP" if mom > 0 and info["trades"] >= 3 else "NEUTRAL")
                    color_start = "\033[91m" if status == "BLOCKED" else ("\033[92m" if status == "TOP" else "")
                    color_end = "\033[0m" if color_start else ""
                    print(f"  {sec:<14s} {mom:>+5.1f}% {color_start}{status:>10s}{color_end} {info['trades']:>6d} {info['win_rate']:>5.0f}%")
                print()

        # Sector breakdown
        sector_stats = m.get("sector_stats", {})
        if sector_stats:
            print(f"  {'-' * 56}")
            print("  SECTOR BREAKDOWN")
            print(f"  {'-' * 56}")
            print(f"  {'Sector':<14s} {'Trades':>6s} {'Win%':>6s} {'Total P&L':>12s} {'Stocks':>7s}")
            print(f"  {'-' * 56}")
            for sec in sorted(sector_stats, key=lambda s: -sector_stats[s]["total_pnl"]):
                ss = sector_stats[sec]
                print(f"  {sec:<14s} {ss['trades']:>6d} {ss['win_rate']:>5.0f}% "
                      f"Rs.{ss['total_pnl']:>+10,.0f} {ss['stock_count']:>6d}")
            print()

        # Per-stock breakdown
        print(f"  {'-' * 56}")
        print("  PER-STOCK BREAKDOWN")
        print(f"  {'-' * 56}")
        print(f"  {'Ticker':<12s} {'Trades':>6s} {'Win%':>6s} {'Total P&L':>12s} {'Avg%':>8s}")
        print(f"  {'-' * 56}")

        stock_stats = m["stock_stats"]
        for ticker in sorted(stock_stats, key=lambda t: -stock_stats[t]["total_pnl"]):
            s = stock_stats[ticker]
            print(f"  {ticker:<12s} {s['trades']:>6d} {s['win_rate']:>5.0f}% "
                  f"Rs.{s['total_pnl']:>+10,.0f} {s['avg_pnl_pct']:>+7.1f}%")

        # Top trades
        print()
        print(f"  {'-' * 56}")
        print("  TOP 10 TRADES")
        print(f"  {'-' * 56}")
        sorted_trades = sorted(self.trades, key=lambda t: -t.pnl_pct)[:10]
        print(f"  {'Ticker':<12s} {'Entry':>10s} {'Exit':>10s} {'P&L%':>8s} {'Days':>5s} {'Reason'}")
        print(f"  {'-' * 56}")
        for t in sorted_trades:
            print(f"  {t.ticker:<12s} Rs.{t.entry_price:>9,.0f} Rs.{t.exit_price:>9,.0f} "
                  f"{t.pnl_pct:>+7.1f}% {t.days_held:>4d}  {t.exit_reason} [{t.sector}]")

        print()
        print(sep)
        print()


# ============================================================
# HTML REPORT
# ============================================================

def _generate_trade_chart(t: TradeResult, stock_data: dict,
                          chart_w: int = 320, chart_h: int = 120) -> str:
    """
    Generate an SVG sparkline for a single trade showing:
    - Price path (close prices)
    - Entry level (blue)
    - Stop loss (red dashed)
    - Target (green dashed)
    - Trail activation (yellow marker)
    - Exit (orange marker)
    """
    ticker = t.ticker
    if ticker not in stock_data:
        return ""

    df = stock_data[ticker]
    # Get price data between entry and exit dates
    mask = (df.index >= t.entry_date) & (df.index <= t.exit_date)
    trade_df = df[mask]
    if len(trade_df) < 2:
        return ""

    closes = trade_df["close"].values
    n = len(closes)

    # Determine price range
    all_prices = list(closes) + [t.entry_price, t.exit_price]
    if t.target_price > 0:
        all_prices.append(t.target_price)
    if t.stop_loss > 0:
        all_prices.append(t.stop_loss)

    min_p = min(all_prices)
    max_p = max(all_prices)
    p_range = max_p - min_p if max_p > min_p else 1
    padding = p_range * 0.1
    min_p -= padding
    max_p += padding
    p_range = max_p - min_p

    margin_l, margin_r, margin_t, margin_b = 40, 10, 10, 20
    inner_w = chart_w - margin_l - margin_r
    inner_h = chart_h - margin_t - margin_b

    def sx(i):
        return margin_l + int(i / max(n - 1, 1) * inner_w)

    def sy(p):
        return margin_t + int((1 - (p - min_p) / p_range) * inner_h)

    # Build SVG
    svg_parts = [f'<svg width="{chart_w}" height="{chart_h}" style="background:#0f172a;border-radius:6px;font-family:monospace">']

    # Grid lines (subtle)
    for i in range(5):
        gy = margin_t + int(i / 4 * inner_h)
        gp = max_p - (i / 4 * p_range)
        svg_parts.append(f'<line x1="{margin_l}" y1="{gy}" x2="{chart_w - margin_r}" y2="{gy}" stroke="#1e293b" stroke-width="0.5"/>')
        svg_parts.append(f'<text x="{margin_l - 4}" y="{gy + 3}" fill="#475569" font-size="7" text-anchor="end">{gp:.0f}</text>')

    # Target line (green dashed)
    if t.target_price > min_p:
        ty = sy(t.target_price)
        svg_parts.append(f'<line x1="{margin_l}" y1="{ty}" x2="{chart_w - margin_r}" y2="{ty}" stroke="#22c55e" stroke-width="0.7" stroke-dasharray="3,3" opacity="0.6"/>')
        svg_parts.append(f'<text x="{chart_w - margin_r - 2}" y="{ty - 3}" fill="#22c55e" font-size="7" text-anchor="end" opacity="0.8">TGT</text>')

    # Stop loss line (red dashed)
    if t.stop_loss > min_p:
        sly = sy(t.stop_loss)
        svg_parts.append(f'<line x1="{margin_l}" y1="{sly}" x2="{chart_w - margin_r}" y2="{sly}" stroke="#ef4444" stroke-width="0.7" stroke-dasharray="3,3" opacity="0.6"/>')
        svg_parts.append(f'<text x="{chart_w - margin_r - 2}" y="{sly + 10}" fill="#ef4444" font-size="7" text-anchor="end" opacity="0.8">SL</text>')

    # Entry line (blue)
    ey = sy(t.entry_price)
    svg_parts.append(f'<line x1="{margin_l}" y1="{ey}" x2="{chart_w - margin_r}" y2="{ey}" stroke="#3b82f6" stroke-width="0.7" opacity="0.5"/>')
    svg_parts.append(f'<text x="{margin_l + 2}" y="{ey - 3}" fill="#3b82f6" font-size="7" opacity="0.8">ENTRY</text>')

    # Price line (white)
    price_points = " ".join(f"{sx(i)},{sy(c)}" for i, c in enumerate(closes))
    line_color = "#22c55e" if t.pnl > 0 else "#ef4444"
    svg_parts.append(f'<polyline points="{price_points}" fill="none" stroke="{line_color}" stroke-width="1.2"/>')

    # Find trail activation point (first bar where target was reached)
    trail_idx = -1
    for i, c in enumerate(closes):
        if c >= t.target_price:
            trail_idx = i
            break

    # Trail activation marker (yellow diamond)
    if trail_idx >= 0:
        tx, ty = sx(trail_idx), sy(closes[trail_idx])
        svg_parts.append(f'<polygon points="{tx},{ty-4} {tx+4},{ty} {tx},{ty+4} {tx-4},{ty}" fill="#eab308" stroke="#eab308" stroke-width="0.5"/>')

    # Entry marker (blue circle)
    svg_parts.append(f'<circle cx="{sx(0)}" cy="{sy(closes[0])}" r="3" fill="#3b82f6" stroke="#1e293b" stroke-width="1"/>')

    # Exit marker (orange circle)
    exit_color = "#22c55e" if t.pnl > 0 else "#ef4444"
    svg_parts.append(f'<circle cx="{sx(n-1)}" cy="{sy(closes[-1])}" r="3" fill="{exit_color}" stroke="#1e293b" stroke-width="1"/>')

    # Exit price label
    svg_parts.append(f'<text x="{sx(n-1) + 5}" y="{sy(closes[-1]) + 3}" fill="#f8fafc" font-size="7">Rs.{t.exit_price:,.0f}</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def generate_html_report(engine: BacktestEngine, metrics: dict,
                         filepath: str = "scanner/backtest_report.html"):
    """Generate a detailed HTML backtest report."""
    import html as html_mod

    m = metrics
    if m.get("total_trades", 0) == 0:
        print("  No trades to report in HTML.")
        return

    # --- Equity curve data ---
    [d.strftime("%Y-%m-%d") for d, _ in engine.equity_curve]
    [round(v, 0) for _, v in engine.equity_curve]

    # --- Build stock data dict for trade charts ---
    stock_data_for_charts = {}
    for stock in engine.stocks:
        stock_data_for_charts[stock.ticker] = stock.df

    # --- Trade log rows with charts ---
    trade_rows = ""
    for t in sorted(engine.trades, key=lambda x: x.entry_date):
        color = "#22c55e" if t.pnl > 0 else "#ef4444"
        score_badge = ("background:#22c55e" if t.entry_score >= 70
                       else "background:#eab308" if t.entry_score >= 50
                       else "background:#94a3b8")
        pnl_bg = "rgba(34,197,94,0.08)" if t.pnl > 0 else "rgba(239,68,68,0.08)"
        chart_svg = _generate_trade_chart(t, stock_data_for_charts)
        trade_rows += f"""
        <tr style="background:{pnl_bg}">
            <td style="vertical-align:top"><strong>{html_mod.escape(t.ticker)}</strong><br><span style="color:#64748b;font-size:0.75em">{html_mod.escape(t.sector)}</span></td>
            <td style="vertical-align:top">{t.entry_date.strftime('%Y-%m-%d')}<br><span style="color:#64748b;font-size:0.75em">{t.days_held}d</span></td>
            <td style="vertical-align:top">Rs.{t.entry_price:,.0f}<br>SL: Rs.{t.stop_loss:,.0f}</td>
            <td style="vertical-align:top"><span style="{score_badge};color:#fff;padding:2px 6px;border-radius:4px;font-size:0.85em">{t.entry_score:.0f}</span></td>
            <td style="vertical-align:top">Rs.{t.exit_price:,.0f}<br>TGT: Rs.{t.target_price:,.0f}</td>
            <td style="vertical-align:top;color:{color};font-weight:bold">{t.pnl_pct:+.1f}%</td>
            <td style="vertical-align:top;color:{color}">Rs.{t.pnl:+,.0f}</td>
            <td style="vertical-align:top">{html_mod.escape(t.exit_reason)}</td>
            <td style="vertical-align:middle;padding:4px">{chart_svg}</td>
        </tr>"""

    # --- Stock breakdown rows ---
    stock_rows = ""
    for ticker in sorted(m["stock_stats"], key=lambda t: -m["stock_stats"][t]["total_pnl"]):
        s = m["stock_stats"][ticker]
        color = "#22c55e" if s["total_pnl"] > 0 else "#ef4444"
        stock_rows += f"""
        <tr>
            <td><strong>{html_mod.escape(ticker)}</strong></td>
            <td>{s['trades']}</td>
            <td>{s['win_rate']:.0f}%</td>
            <td style="color:{color};font-weight:bold">Rs.{s['total_pnl']:+,.0f}</td>
            <td style="color:{color}">{s['avg_pnl_pct']:+.1f}%</td>
        </tr>"""

    # --- Exit reasons ---
    exit_rows = ""
    for reason, count in sorted(m["exit_reasons"].items(), key=lambda x: -x[1]):
        pct = count / m["total_trades"] * 100
        exit_rows += f"""
        <tr>
            <td>{html_mod.escape(reason)}</td>
            <td>{count}</td>
            <td>{pct:.0f}%</td>
        </tr>"""

    # --- Equity curve chart (inline SVG) ---
    if len(engine.equity_curve) > 1:
        eq_vals = [v for _, v in engine.equity_curve]
        min_v = min(eq_vals)
        max_v = max(eq_vals)
        v_range = max_v - min_v if max_v > min_v else 1
        n_points = len(eq_vals)
        chart_w, chart_h = 800, 250

        def scale_x(i):
            return int(i / max(n_points - 1, 1) * chart_w)

        def scale_y(v):
            return int(chart_h - ((v - min_v) / v_range * chart_h))

        points = " ".join(
            f"{scale_x(i)},{scale_y(v)}" for i, v in enumerate(eq_vals)
        )

        equity_chart_svg = f"""
        <svg width="{chart_w}" height="{chart_h}" style="width:100%;height:auto;background:#0f172a;border-radius:8px">
            <polyline points="{points}" fill="none" stroke="#22c55e" stroke-width="1.5"/>
            <polyline points="0,{scale_y(m['initial_capital'])} {scale_x(n_points-1)},{scale_y(m['initial_capital'])}"
                      fill="none" stroke="#64748b" stroke-width="0.5" stroke-dasharray="4"/>
        </svg>"""
    else:
        equity_chart_svg = "<p>No equity data</p>"

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Backtest Report - HMA/EMA Multi-Score Strategy</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }}
  h1 {{ color: #f8fafc; font-size: 1.8em; margin-bottom: 4px; }}
  h2 {{ color: #94a3b8; font-size: 1.1em; margin: 28px 0 12px; border-bottom: 1px solid #1e293b; padding-bottom: 6px; }}
  .subtitle {{ color: #64748b; font-size: 0.9em; margin-bottom: 24px; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .metric-card {{ background: #1e293b; border-radius: 12px; padding: 20px; }}
  .metric-label {{ color: #64748b; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; }}
  .metric-value {{ color: #f8fafc; font-size: 1.6em; font-weight: 700; margin-top: 4px; }}
  .metric-value.positive {{ color: #22c55e; }}
  .metric-value.negative {{ color: #ef4444; }}
  .metric-sub {{ color: #94a3b8; font-size: 0.85em; margin-top: 2px; }}
  .chart-container {{ background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
  th {{ background: #1e293b; color: #94a3b8; padding: 10px 12px; text-align: left; font-weight: 600; text-transform: uppercase; font-size: 0.8em; letter-spacing: 0.3px; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #1e293b; }}
  tr:hover td {{ background: #1e293b; }}
</style>
</head>
<body>

<h1>Backtest Report</h1>
<p class="subtitle">HMA/EMA Multi-Score Swing Strategy - NIFTY 50 - {m['years']:.1f} years</p>

<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-label">Total Return</div>
    <div class="metric-value {'positive' if m['total_return_pct'] >= 0 else 'negative'}">{m['total_return_pct']:+.1f}%</div>
    <div class="metric-sub">Annualized: {m['annual_return_pct']:+.1f}%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Total P&L</div>
    <div class="metric-value {'positive' if m['total_pnl'] >= 0 else 'negative'}">Rs.{m['total_pnl']:+,.0f}</div>
    <div class="metric-sub">Rs.{m['initial_capital']:,.0f} to Rs.{m['final_value']:,.0f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Win Rate</div>
    <div class="metric-value">{m['win_rate']:.1f}%</div>
    <div class="metric-sub">{m['total_trades']} trades total</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Profit Factor</div>
    <div class="metric-value">{m['profit_factor']:.2f}</div>
    <div class="metric-sub">Avg win: {m['avg_win_pct']:+.1f}% / Avg loss: {m['avg_loss_pct']:+.1f}%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Sharpe Ratio</div>
    <div class="metric-value">{m['sharpe_ratio']:.2f}</div>
    <div class="metric-sub">Sortino: {m['sortino_ratio']:.2f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Max Drawdown</div>
    <div class="metric-value negative">Rs.{m['max_drawdown']:,.0f}</div>
    <div class="metric-sub">{m['max_drawdown_pct']:.1f}% of peak</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Signals</div>
    <div class="metric-value">{m['signals_taken']}</div>
    <div class="metric-sub">{m['signals_generated']} generated ({m['signal_conversion']:.1f}% conversion)</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Avg Winner Score</div>
    <div class="metric-value">{m['avg_winner_score']:.0f}</div>
    <div class="metric-sub">vs losers: {m['avg_loser_score']:.0f}</div>
  </div>
</div>

<h2>Equity Curve</h2>
<div class="chart-container">
  {equity_chart_svg}
</div>

<h2>Exit Reasons</h2>
<table>
  <thead><tr><th>Reason</th><th>Count</th><th>%</th></tr></thead>
  <tbody>{exit_rows}</tbody>
</table>

<h2>Per-Stock Performance</h2>
<table>
  <thead><tr><th>Stock</th><th>Trades</th><th>Win%</th><th>Total P&L</th><th>Avg P&L%</th></tr></thead>
  <tbody>{stock_rows}</tbody>
</table>

<h2>All Trades ({len(engine.trades)} trades)</h2>
<table>
  <thead>
    <tr>
      <th>Stock</th><th>Entry</th><th>Entry Rs.</th><th>Score</th>
      <th>Exit</th><th>P&L%</th><th>P&L Rs.</th><th>Reason</th><th>Chart</th>
    </tr>
  </thead>
  <tbody>{trade_rows}</tbody>
</table>

</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"\n  HTML report saved: {filepath}")


# ============================================================
# CSV EXPORT
# ============================================================

def save_trades_csv(trades: list[TradeResult],
                    filepath: str = "scanner/backtest_trades.csv"):
    """Save all trades to a CSV file."""
    rows = []
    for t in trades:
        rows.append({
            "ticker": t.ticker,
            "sector": t.sector,
            "entry_date": t.entry_date.strftime("%Y-%m-%d"),
            "entry_price": round(t.entry_price, 2),
            "entry_score": round(t.entry_score, 1),
            "exit_date": t.exit_date.strftime("%Y-%m-%d"),
            "exit_price": round(t.exit_price, 2),
            "exit_reason": t.exit_reason,
            "shares": t.shares,
            "pnl": round(t.pnl, 2),
            "pnl_pct": round(t.pnl_pct, 2),
            "days_held": t.days_held,
            "investment": round(t.investment, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"  Trades CSV saved: {filepath}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backtest HMA/EMA Multi-Score Swing Strategy"
    )
    parser.add_argument("--years", type=float, default=3,
                        help="Years of historical data (default: 3)")
    parser.add_argument("--tickers", type=str, nargs="*",
                        help="Specific tickers to test (default: NIFTY 50)")
    parser.add_argument("--score", type=float, default=50,
                        help="Minimum score threshold (default: 50)")
    parser.add_argument("--stop", type=float, default=2.0,
                        help="Stop loss %% (default: 2.0)")
    parser.add_argument("--target", type=float, default=20.0,
                        help="Target profit %% (default: 20.0)")
    parser.add_argument("--trail", type=float, default=2.0,
                        help="Trailing stop %% (default: 2.0)")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="Initial capital in Rs. (default: 1000000)")
    parser.add_argument("--crossover-lookback", type=int, default=6,
                        help="Crossover detection lookback bars (default: 6)")
    parser.add_argument("--max-per-sector", type=int, default=2,
                        help="Max positions per sector (default: 2)")
    parser.add_argument("--atr-stop", action="store_true",
                        help="Use ATR-based stop loss instead of fixed percentage")
    parser.add_argument("--atr-mult", type=float, default=2.0,
                        help="ATR stop multiplier (default: 2.0)")
    parser.add_argument("--max-risk", type=float, default=0.02,
                        help="Max risk per trade as fraction of capital (default: 0.02 = 2%%)")
    parser.add_argument("--atr-trail", action="store_true",
                        help="Use ATR-based trailing stop instead of fixed percentage")
    parser.add_argument("--atr-trail-mult", type=float, default=2.5,
                        help="ATR trailing stop multiplier (default: 2.5)")
    parser.add_argument("--sector-rotation", action="store_true",
                        help="Enable sector rotation (avoid losers, boost winners)")
    parser.add_argument("--rotation-lookback", type=int, default=8,
                        help="Sector rotation lookback trades (default: 8)")
    parser.add_argument("--rotation-boost", type=float, default=0.5,
                        help="Score bonus factor for top sectors (default: 0.5)")
    parser.add_argument("--rotation-block", type=float, default=-5.0,
                        help="Block sector if momentum below this %% (default: -5.0)")
    parser.add_argument("--html", type=str, default="scanner/backtest_report.html",
                        help="HTML report path")
    parser.add_argument("--csv", type=str, default="scanner/backtest_trades.csv",
                        help="Trades CSV path")
    parser.add_argument("--no-html", action="store_true",
                        help="Skip HTML report generation")
    parser.add_argument("--universe", type=str, default="nifty50",
                        choices=["nifty50", "fno", "all"],
                        help="Stock universe: nifty50, fno, or all (default: nifty50)")
    args = parser.parse_args()

    # Build settings
    settings = {
        "score_threshold": args.score,
        "stop_loss_pct": args.stop,
        "target_pct": args.target,
        "trail_pct": args.trail,
        "initial_capital": args.capital,
        "max_positions_per_sector": args.max_per_sector,
        "sector_rotation_enabled": args.sector_rotation,
        "sector_rotation_lookback": args.rotation_lookback,
        "sector_boost_weight": args.rotation_boost,
        "sector_block_threshold": args.rotation_block,
        "atr_stop_enabled": args.atr_stop,
        "atr_stop_multiplier": args.atr_mult,
        "atr_trail_enabled": args.atr_trail,
        "atr_trail_multiplier": args.atr_trail_mult,
        "max_risk_per_trade": args.max_risk,
        "crossover_lookback": args.crossover_lookback,
    }

    # Select tickers
    if args.tickers:
        tickers = args.tickers
    elif args.universe == "fno":
        tickers = FNO_STOCKS
    elif args.universe == "all":
        tickers = NIFTY_BROAD
    else:
        tickers = NIFTY_50

    # Period
    period = f"{int(args.years)}y"

    # Run
    engine = BacktestEngine(settings)
    engine.load_data(tickers, period=period)
    metrics = engine.run()

    # Print report
    engine.print_report(metrics)

    # Save outputs
    if metrics.get("total_trades", 0) > 0:
        save_trades_csv(engine.trades, args.csv)
        if not args.no_html:
            generate_html_report(engine, metrics, args.html)


if __name__ == "__main__":
    main()
