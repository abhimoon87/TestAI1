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
import os
import sys

import numpy as np
import pandas as pd

from .data_fetcher import fetch_batch_yfinance, fetch_index_data
from .scoring import detect_crossover, get_ma
from .universes import FNO_STOCKS, NIFTY_50, NIFTY_BROAD

# Sub-module re-exports (backward-compatible public API)
from .backtest_models import (  # noqa: E402
    DEFAULT_SETTINGS,
    SECTOR_MAP,
    WARMUP_BARS,
    Position,
    SectorTracker,
    StockData,
    TradeResult,
    get_sector,
)
from .backtest_indicators import (  # noqa: E402
    precompute_nifty,
    precompute_stock,
)
from .backtest_scoring import compute_score_at_bar  # noqa: E402
from .backtest_position import (  # noqa: E402
    _close_position,
    update_position,
)
from .backtest_report import (  # noqa: E402
    _generate_trade_chart,
    generate_html_report,
    save_trades_csv,
)

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
        self.equity_curve: list[tuple] = []
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
                        shares = max_shares
                        if shares <= 0:
                            continue

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
                # Sortino: downside deviation uses all returns but zeros out positive ones
                downside_devs = [min(r, 0) for r in daily_rets]
                downside_std = np.std(downside_devs) if downside_devs else 0.001
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
