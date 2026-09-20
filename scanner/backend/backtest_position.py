"""Position management for the backtest engine."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from .backtest_models import Position, TradeResult

logger = logging.getLogger(__name__)


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
        # Check trailing stop FIRST using the old peak/trail_stop
        # (low happens before close within a bar)
        if low <= pos.trail_stop:
            exit_price = max(pos.trail_stop, low)
            return _close_position(pos, exit_price, current_date, "TRAILING_STOP")

        # THEN update peak with close (information available after the bar)
        if close_val > pos.peak_price:
            pos.peak_price = close_val
            if atr_trail and pos.atr_at_entry > 0:
                pos.trail_stop = pos.peak_price - (atr_mult * pos.atr_at_entry)
            else:
                pos.trail_stop = pos.peak_price * (1 - settings["trail_pct"] / 100)
            pos.stop_loss = max(pos.stop_loss, pos.trail_stop)

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
