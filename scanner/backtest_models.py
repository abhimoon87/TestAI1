"""Data models, constants, and sector tracking for the backtest engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .settings_store import DEFAULT_SETTINGS as _SCANNER_DEFAULTS

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION — backtest overrides on top of scanner defaults
# ============================================================

_BACKTEST_OVERRIDES = {
    "fast_ma_len": 44,
    "slow_ma_len": 30,
    "crossover_lookback": 6,
    "min_adx_entry": 0.0,
}

_BACKTEST_ONLY = {
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
    "sector_rotation_lookback": 8,
    "sector_min_momentum": 0.0,
    "sector_block_threshold": -5.0,
    "sector_boost_weight": 0.5,
    # ATR-based stop loss
    "atr_stop_enabled": False,
    "atr_stop_multiplier": 2.0,
    # ATR-based trailing stop
    "atr_trail_enabled": False,
    "atr_trail_multiplier": 2.5,
    # Entry gates (backtest experiments; all default OFF so behavior is unchanged)
    "index_regime_filter": False,
    "index_regime_ema_len": 50,
    "blocked_entry_weekdays": [],
}

DEFAULT_SETTINGS = {**_SCANNER_DEFAULTS, **_BACKTEST_OVERRIDES, **_BACKTEST_ONLY}

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


def get_sector(ticker: str) -> str:
    """Get the sector for a stock ticker."""
    return SECTOR_MAP.get(ticker, "Other")


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
    obv_ma: pd.Series = None
    vol_ma: pd.Series = None
    vol_50: pd.Series = None
    atr_val: pd.Series = None
    atr1: pd.Series = None
    adx_val: pd.Series = None
    vp_poc: pd.Series = None
    slope_ma: pd.Series = None
    weekly_df: pd.DataFrame = None
    w_hma: pd.Series = None
    w_ema50: pd.Series = None
    fundamentals: dict = field(default_factory=dict)


# ============================================================
# SECTOR TRACKING
# ============================================================

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
