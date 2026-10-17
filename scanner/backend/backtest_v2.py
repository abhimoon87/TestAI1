"""
Backtest V2 — engine for the HMAxEMA_VP V4 weekly-swing strategy.

Reuses the v1 engine's data/metrics/report pipeline and replaces the
strategy core with the V4 entry rules:

  Trigger:  weekly HMA(fast=18) x EMA(slow=44) cross, evaluated on each
            stock's last daily bar of a week (the weekly close — no
            look-ahead). A bullish cross arms a confirmation when ALL of:
            no open position, NIFTY above its EMA (regime), not sideways,
            score >= 50. Snapshot: level = signal-week HIGH, low =
            signal-week LOW (chandelier seed). Bearish cross cancels.

  Entry:    within the next confirm_weeks (15) weekly bars, evaluated
            once per weekly close, ALL five gates must hit on one bar:
              1. weekly candle green (close > open)
              2. close > max(HMA18, EMA44) — still above both MAs
              3. close > level (signal-week high breakout)
              4. close > VP POC (12-week volume-profile point of control)
              5. score >= 50 (10-category score)
            Filled at the next bar's open (= next week's first session).
            Expired after confirm_weeks with no hit. Long only, no adds.

  Exits:    weekly bearish cross -> close at next open (ALL modes);
            + optional risk exits per mode:
              signals   — weekly cross only (Pine default baseline)
              rbracket  — slPct% stop -> BE/roll at beActR from highest high
                          -> limit at rrRatio x R
              chandelier— hybrid init (signal-week low - structBuf*weeklyATR,
                          capped at structCapMult x chandelier distance), then
                          ratchet max(prev, highest high - atrMult*weeklyATR)
            Stop/trailing closes re-open the confirmation gate (re-armed
            with the same preconditions as a fresh cross).

Costs:    0.1% per side commission (Pine commission_value on entry and
          exit orders); slippage and tick rounding are not modeled.

Data: permanent daily OHLCV archive in scanner.db (table ohlcv_archive);
symbols missing/stale there are fetched via the provider chain and upserted,
so runs 2+ are offline and reproducible.

Usage:
    python -m scanner.backend.backtest_v2 --years 5
    python -m scanner.backend.backtest_v2 --tickers TATAMOTORS --max-positions 1
    python -m scanner.backend.backtest_v2 --universe nifty500 --exit-mode chandelier
    python -m scanner.backend.backtest_v2 --timeframe weekly --years 24
        # Pine running on a 1W chart: each bar IS a weekly bar
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..api.data_fetcher import fetch_batch_yfinance, fetch_index_data
from ..shared import db
from ..shared.indicators import atr as atr_fn
from ..shared.universes import get_universe
from .backtest import BacktestEngine
from .backtest_indicators import precompute_nifty, precompute_stock
from .backtest_models import TradeResult, get_sector
from .backtest_position import _close_position
from .backtest_scoring import compute_score_at_bar
from .scoring import to_weekly

logger = logging.getLogger(__name__)

# Pine input defaults (V4 file after the weekly-VP / weekly-ATR revision).
V2_DEFAULTS = {
    # Daily MAs feed the score's trend category only — the trigger is weekly.
    "fast_ma_len": 40,
    "slow_ma_len": 50,
    "w_fast_len": 18,  # weekly HMA (trigger fast leg)
    "w_slow_len": 44,  # weekly EMA (trigger slow leg)
    "score_threshold": 50,
    "confirm_weeks": 15,
    # close = Pine: bar closes above the cross-bar high, fill next open.
    # highbreak = enter when the cross-bar high breaks (buy-stop fill at level,
    # same bar); green + POC + score gates still validated on that bar.
    "entry_trigger": "close",
    "exit_mode": "signals",  # signals | rbracket | chandelier
    "sl_pct": 8.0,
    "rr_ratio": 3.0,
    "be_act_r": 1.0,
    "atr_mult": 2.0,  # x weekly ATR(14)
    "struct_buf": 0.3,
    "struct_cap_mult": 2.0,
    "risk_pct": 0.0,  # >0 switches to Pine risk sizing (equity x % / stop dist)
    "commission_pct": 0.1,  # Pine: 0.1% per side
    "vp_weekly_lookback": 12,
    "vp_rows": 30,
    "max_positions": 5,
    "position_size_pct": 10.0,  # Pine percent_of_equity=10
    "max_risk_per_trade": 1.0,  # risk cap disabled unless risk_pct set
}


@dataclass
class Pending:
    """Armed confirmation (Pine pendLevel/pendBar/pendLow + week index)."""

    bar_idx: int
    w_idx: int  # signal week (week id of the arming bar)
    level: float  # signal-week high — entry close must exceed it
    low: float  # signal-week low — chandelier structure-stop seed


# Spec-B gate keys in evaluation order (scan notes read from the first miss).
_GATE_ORDER = (
    "week_ended",
    "n_window",
    "regime",
    "ma_ok",
    "green",
    "level",
    "above_ma",
    "poc",
    "score",
)
_GATE_LABELS = {
    "week_ended": "weekly close",
    "n_window": "confirm window",
    "regime": "NIFTY regime",
    "ma_ok": "weekly MAs",
    "green": "green week",
    "level": "break signal high",
    "above_ma": "above weekly MAs",
    "poc": "above POC",
    "score": "score",
}


class V2Engine(BacktestEngine):
    """HMAxEMA_VP V4 strategy on the v1 simulation/metrics pipeline."""

    def __init__(self, settings: dict | None = None):
        merged = {**V2_DEFAULTS, **(settings or {})}
        # Pine slPct is the one stop input: rbracket 1R AND the risk-sizing
        # distance (qty = equity x risk% / (close x slPct/100)).
        merged["stop_loss_pct"] = merged["sl_pct"]
        if merged.get("risk_pct", 0.0) and merged["risk_pct"] > 0:
            # Risk sizing: let the engine's risk cap bind (equity x risk% /
            # stop distance per share) instead of the 10% investment slice.
            merged["position_size_pct"] = 100.0
            merged["max_risk_per_trade"] = merged["risk_pct"] / 100.0
        super().__init__(merged)
        self._pending: dict[str, Pending] = {}
        self._exit_pending: set[str] = set()
        self._seed_low: dict[str, float] = {}
        self._rolls: dict[str, np.ndarray] = {}  # week-start row indices
        self._periods: dict[str, pd.PeriodIndex] = {}  # W-period id per bar
        self._w_atr: dict[str, pd.Series] = {}  # weekly ATR(14) per ticker
        self._signals_generated = 0
        self._signals_taken = 0
        self._signals_blocked = 0
        self._signals_boosted = 0

    # ── Data (archive-backed) ────────────────────────────────────────────────

    def load_data(self, tickers: list[str], period: str = "5y"):
        """Load from the permanent ohlcv_archive; fetch + upsert only gaps."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("  BACKTEST V2: HMAxEMA_VP Swing (Pine V4 port)")
        logger.info("=" * 60)
        logger.info(f"  Stocks: {len(tickers)}")
        logger.info(f"  Period: {period} (daily)")
        logger.info(
            f"  Trigger: weekly HMA({self.settings['w_fast_len']}) x "
            f"EMA({self.settings['w_slow_len']}), "
            f"confirm {self.settings['confirm_weeks']}w, score>="
            f"{self.settings['score_threshold']}"
        )
        logger.info(
            f"  Exits: {self.settings['exit_mode']}  "
            f"commission: {self.settings.get('commission_pct', 0.0)}%"
        )
        logger.info(f"  Timeframe: {self.settings.get('timeframe', 'daily')}")
        logger.info("=" * 60)
        logger.info("")

        years = int(period.rstrip("y")) if period.endswith("y") else 3
        # yfinance only accepts fixed periods; long windows need "max".
        fetch_period = "max" if years > 10 else period
        now = pd.Timestamp.now().normalize()
        start = (now - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
        fresh_max = (now - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

        frames: dict[str, pd.DataFrame] = {}
        need: list[str] = []
        for t in tickers:
            rng = db.archive_range(t)
            if rng and rng[0] <= start and rng[1] >= fresh_max:
                frames[t] = db.archive_get(t, start=start)
            else:
                need.append(t)

        if need:
            logger.info(
                f"  Fetching {len(need)} symbols from providers "
                "(first run / stale archive)..."
            )
            fetched = (
                fetch_batch_yfinance(need, period=fetch_period, timeframe="D") or {}
            )
            for t, df in fetched.items():
                if df is not None and not df.empty:
                    try:
                        db.archive_put(t, df)
                    except Exception:
                        logger.info("archive_put failed for %s", t, exc_info=True)
                    # Clip to the requested window so cached and fresh runs
                    # simulate over the identical date range.
                    frames[t] = df.loc[df.index >= start]
            # Provider miss (delisted etc.): fall back to whatever is archived.
            for t in need:
                if t not in frames:
                    alt = db.archive_get(t, start=start)
                    if not alt.empty:
                        frames[t] = alt

        # Index for relative-strength scoring — same archive pattern.
        idx_raw = None
        rng = db.archive_range("^NSEI")
        if rng and rng[0] <= start and rng[1] >= fresh_max:
            idx_raw = db.archive_get("^NSEI", start=start)
        else:
            idx_raw = fetch_index_data("^NSEI", period=fetch_period)
            if idx_raw is not None and not idx_raw.empty:
                try:
                    db.archive_put("^NSEI", idx_raw)
                except Exception:
                    logger.info("archive_put failed for ^NSEI", exc_info=True)
                idx_raw = idx_raw.loc[idx_raw.index >= start]
        if idx_raw is None or idx_raw.empty:
            logger.info("  WARNING: no NIFTY index data — RS scoring degraded")
            self.nifty_df = None
        else:
            self.nifty_df = precompute_nifty(idx_raw)

        # Weekly mode: the whole simulation runs on chart-TF weekly bars
        # (Pine on a 1W chart — security("W") is the chart itself, so crosses,
        # confirm window and fills all count in weeks).
        if self.settings.get("timeframe") == "weekly":
            frames = {
                t: w
                for t, df in frames.items()
                if (w := to_weekly(df)) is not None and not w.empty
            }
            if idx_raw is not None and not idx_raw.empty:
                w_idx = to_weekly(idx_raw)
                self.nifty_df = precompute_nifty(w_idx) if w_idx is not None else None

        logger.info("  Computing indicators...")
        for ticker in tickers:
            df = frames.get(ticker)
            if df is None:
                continue
            stock = precompute_stock(ticker, df, self.settings)
            if stock is not None:
                self.stocks.append(stock)

        for stock in self.stocks:
            per = stock.df.index.to_period("W")
            self._rolls[stock.ticker] = (
                np.flatnonzero(np.asarray(per[1:] != per[:-1])) + 1
            )
            self._periods[stock.ticker] = per
            if stock.weekly_df is not None:
                w = stock.weekly_df
                self._w_atr[stock.ticker] = atr_fn(
                    w["high"], w["low"], w["close"], int(self.settings["atr_len"])
                )

        logger.info(f"  Ready: {len(self.stocks)}/{len(tickers)} stocks loaded")
        logger.info("")

    # ── Weekly helpers (week-end evaluation, no look-ahead) ──────────────────

    def _week_id(self, ticker: str, bar_idx: int) -> int:
        """Calendar-week index of a bar (matches weekly_df row order:
        both enumerate the W periods that actually contain bars)."""
        rolls = self._rolls[ticker]
        return int(np.searchsorted(rolls, bar_idx, side="right"))

    def _week_start(self, ticker: str, w: int) -> int:
        """First daily-bar index of week w."""
        rolls = self._rolls[ticker]
        return int(rolls[w - 1]) if w > 0 else 0

    def _week_ended(self, ticker: str, bar_idx: int) -> bool:
        """True when the bar is its week's last session — the weekly close,
        high and low are fully known then (spec evaluation point).

        The last loaded bar counts only once Friday's session is in (no more
        bars can land in this W period before the weekend); weekly-mode bars
        carry Sunday labels and always count."""
        per = self._periods.get(ticker)
        if per is None:
            return False
        if bar_idx < len(per) - 1:
            return bool(per[bar_idx] != per[bar_idx + 1])
        # Last bar: only a Friday (daily) or Sunday label (weekly) closes
        # the week — need the bar's real calendar weekday, not the
        # period anchor's (Period.weekday is always the W-end day).
        stock = self._stock_map.get(ticker)
        if stock is None:
            return False
        wd = stock.df.index[bar_idx].weekday()
        return wd in (4, 6)

    def _weekly_cross(self, stock, w_idx: int) -> tuple[bool, bool]:
        """(bullish, bearish) HMA x EMA cross at weekly bar w_idx."""
        wf, ws = stock.w_hma, stock.w_ema50
        if (
            wf is None
            or ws is None
            or w_idx < 1
            or w_idx >= len(wf)
            or w_idx >= len(ws)
        ):
            return False, False
        c_f, c_s = wf.iloc[w_idx], ws.iloc[w_idx]
        p_f, p_s = wf.iloc[w_idx - 1], ws.iloc[w_idx - 1]
        vals = (c_f, c_s, p_f, p_s)
        if any(pd.isna(v) for v in vals):
            return False, False
        bull = c_f > c_s and p_f <= p_s
        bear = c_f < c_s and p_f >= p_s
        return bull, bear

    def _weekly_atr_asof(self, stock, day) -> float | None:
        s = self._w_atr.get(stock.ticker)
        if s is None:
            return None
        idx = int(s.index.searchsorted(day, side="right") - 1)
        if idx < 0:
            return None
        return float(s.iloc[idx])

    def _vp_poc(self, stock, bar_idx: int) -> float | None:
        """Pine vpPOC: proportional volume bins over the weekly VP window."""
        lookback_w = int(self.settings["vp_weekly_lookback"])
        rows = int(self.settings["vp_rows"])
        rolls = self._rolls.get(stock.ticker)
        if rolls is None:
            return None
        n = int(np.searchsorted(rolls, bar_idx, side="right"))
        if n >= lookback_w:
            span = bar_idx - int(rolls[n - lookback_w]) + 1
        else:
            span = min(bar_idx + 1, lookback_w * 5)
        if span < 1:
            return None
        window = stock.df.iloc[bar_idx - span + 1 : bar_idx + 1]
        hh = float(window["high"].max())
        ll = float(window["low"].min())
        row_size = (hh - ll) / rows if hh > ll else 1e-9
        vol = np.zeros(rows)
        highs = window["high"].to_numpy(dtype=float)
        lows = window["low"].to_numpy(dtype=float)
        vols = np.nan_to_num(window["volume"].to_numpy(dtype=float), nan=0.0)
        for b_h, b_l, b_v in zip(highs, lows, vols, strict=False):
            if b_h == b_l:
                idx = int(round((b_h - ll) / row_size))
                vol[max(0, min(rows - 1, idx))] += b_v
            else:
                start = int(max(0, np.floor((b_l - ll) / row_size)))
                end = int(min(rows - 1, np.ceil((b_h - ll) / row_size) - 1))
                for r_i in range(start, end + 1):
                    bin_bot = ll + r_i * row_size
                    bin_top = ll + (r_i + 1) * row_size
                    overlap = max(0.0, min(b_h, bin_top) - max(b_l, bin_bot))
                    if overlap > 0:
                        vol[r_i] += b_v * (overlap / (b_h - b_l))
        poc_idx = int(np.argmax(vol))
        return ll + poc_idx * row_size + row_size / 2

    # ── Entries (V4 spec A: arming / spec B: confirmation) ───────────────────

    def _advance_pending(self, stock, bar_idx: int, day, regime_ok) -> None:
        """Arm / expire the confirmation window (spec A).

        Runs every bar. Arming happens only on the week's last session —
        the weekly close, high and low are fully known then (no look-ahead).
        Requires: no open position, regime OK, not sideways, score >=
        threshold. A bearish cross cancels the pending.
        """
        t = stock.ticker
        if stock.weekly_df is None:
            return
        pend = self._pending.get(t)
        if (
            pend is not None
            and self._week_id(t, bar_idx) - pend.w_idx > self.settings["confirm_weeks"]
        ):
            self._pending.pop(t, None)
        if not self._week_ended(t, bar_idx):
            return
        w = self._week_id(t, bar_idx)
        bull, bear = self._weekly_cross(stock, w)
        if bear:
            self._pending.pop(t, None)
        if not bull:
            return
        self._signals_generated += 1
        sr = None
        held = any(p.ticker == t for p in self.positions)
        if not held and regime_ok:
            sr = compute_score_at_bar(stock, bar_idx, self.nifty_df, self.settings)
        if (
            sr is None
            or sr["is_sideways"]
            or sr["total"] < self.settings["score_threshold"]
        ):
            self._signals_blocked += 1
            return
        wk = stock.df.iloc[self._week_start(t, w) : bar_idx + 1]
        self._pending[t] = Pending(
            bar_idx, w, float(wk["high"].max()), float(wk["low"].min())
        )

    def _entry_gates(self, stock, bar_idx: int, pend: Pending, regime_ok: bool):
        """Per-gate pass/fail for the spec-B confirmation (single source of
        truth for _entry_ready and the live scan).

        Returns (gates: dict[str, bool], score_result). The score gate — the
        expensive one — is only computed when every earlier gate passes.
        """
        settings = self.settings
        t = stock.ticker
        g = dict.fromkeys(_GATE_ORDER, False)
        g["week_ended"] = self._week_ended(t, bar_idx)
        g["regime"] = bool(regime_ok)
        if not g["week_ended"]:
            return g, None  # gates fire once per weekly bar
        w = self._week_id(t, bar_idx)
        n_w = w - pend.w_idx
        g["n_window"] = 1 <= n_w <= int(settings["confirm_weeks"])
        if not (g["n_window"] and g["regime"]):
            return g, None
        if w >= len(stock.w_hma) or w >= len(stock.w_ema50):
            return g, None
        w_fast = stock.w_hma.iloc[w]
        w_slow = stock.w_ema50.iloc[w]
        if pd.isna(w_fast) or pd.isna(w_slow):
            return g, None
        g["ma_ok"] = True
        wk = stock.df.iloc[self._week_start(t, w) : bar_idx + 1]
        wk_close = float(wk["close"].iloc[-1])
        g["green"] = wk_close > float(wk["open"].iloc[0])
        if settings["entry_trigger"] == "highbreak":
            g["level"] = float(wk["high"].max()) > pend.level
        else:
            g["level"] = wk_close > pend.level
        g["above_ma"] = wk_close > max(w_fast, w_slow)
        poc = self._vp_poc(stock, bar_idx)
        g["poc"] = poc is not None and wk_close > poc
        if not (g["green"] and g["level"] and g["above_ma"] and g["poc"]):
            return g, None
        sr = compute_score_at_bar(stock, bar_idx, self.nifty_df, settings)
        g["score"] = sr is not None and sr["total"] >= settings["score_threshold"]
        return g, (sr if g["score"] else None)

    def _entry_ready(self, stock, bar_idx: int, pend: Pending, regime_ok: bool):
        """The 5 spec-B gates — shared by _try_entries and run_status().

        Evaluated once per confirmed weekly bar (the week's last session).
        Returns the score result when every gate passes, else None.
        """
        _gates, sr = self._entry_gates(stock, bar_idx, pend, regime_ok)
        return sr

    def _regime_ok(self, day) -> bool:
        if self._regime_ema is None:
            return True
        _ic = self._regime_close.asof(day)
        _ie = self._regime_ema.asof(day)
        return not (pd.isna(_ic) or pd.isna(_ie)) and _ic > _ie

    def _try_entries(self, day):
        regime_ok = self._regime_ok(day)

        for stock in self.stocks:
            if day not in stock.df.index:
                continue
            bar_idx = int(stock.df.index.get_loc(day))
            # State maintenance always runs; capacity gates only block OPEN.
            self._advance_pending(stock, bar_idx, day, regime_ok)
            if not self._week_ended(stock.ticker, bar_idx):
                continue  # entry decision lands on the weekly close only
            held = any(p.ticker == stock.ticker for p in self.positions)
            if held or not regime_ok or len(self.positions) >= self._max_pos:
                continue
            stock_sector = get_sector(stock.ticker)
            sector_count = sum(1 for p in self.positions if p.sector == stock_sector)
            if sector_count >= self._max_per_sector:
                continue
            pend = self._pending.get(stock.ticker)
            if pend is None:
                continue
            sr = self._entry_ready(stock, bar_idx, pend, regime_ok)
            if sr is None:
                continue
            self._seed_low[stock.ticker] = pend.low
            self._open_position(stock, bar_idx, day, stock_sector, sr)
            self._pending.pop(stock.ticker, None)

    def _open_position(self, stock, bar_idx, day, stock_sector, score_result):
        """Next-open fill (base) or stop fill (highbreak) + Pine seeds."""
        entry_price = entry_date = None
        if self.settings["entry_trigger"] == "highbreak":
            pend = self._pending.get(stock.ticker)
            if pend is not None:
                # Buy-stop: _entry_ready proved the confirming week's high
                # traded above the signal-week level — fill at the FIRST
                # session that crossed it: at level, or that session's open
                # on a gap above it. (Weekly mode: the week is one bar, so
                # this is the eval bar itself.) Gates (green/MA/POC/score)
                # were validated on the completed week.
                w = self._week_id(stock.ticker, bar_idx)
                wk = stock.df.iloc[self._week_start(stock.ticker, w) : bar_idx + 1]
                for j in range(len(wk)):
                    b = wk.iloc[j]
                    if float(b["high"]) > pend.level:
                        entry_price = max(pend.level, float(b["open"]))
                        entry_date = wk.index[j]
                        break
        before = len(self.positions)
        super()._open_position(
            stock,
            bar_idx,
            day,
            stock_sector,
            score_result,
            entry_price=entry_price,
            entry_date=entry_date,
        )
        if len(self.positions) == before:
            self._seed_low.pop(stock.ticker, None)
            return
        pos = self.positions[-1]
        mode = self.settings["exit_mode"]
        entry = pos.entry_price
        if mode == "rbracket":
            r = entry * self.settings["sl_pct"] / 100.0
            pos.target_price = entry + r * self.settings["rr_ratio"]
        elif mode == "chandelier":
            atr_w = self._weekly_atr_asof(stock, day)
            if atr_w is not None and not np.isnan(atr_w) and atr_w > 0:
                mult = self.settings["atr_mult"]
                fallback = entry - mult * atr_w
                seed = self._seed_low.get(stock.ticker)
                struct = (
                    seed - self.settings["struct_buf"] * atr_w
                    if seed is not None
                    else fallback
                )
                pos.stop_loss = (
                    fallback
                    if (entry - struct)
                    > self.settings["struct_cap_mult"] * mult * atr_w
                    else struct
                )
        self._seed_low.pop(stock.ticker, None)

    # ── Live status (as of the latest bar) ───────────────────────────────────

    def run_status(self) -> list[dict]:
        """Entry status as of each stock's latest bar, post-run.

        Replays the trailing confirm window from scratch: run() pops a pending
        that triggers on the very last bar (no next bar to fill at backtest.py
        next_idx guard), so the end-state _pending misses those.
        """
        confirm = int(self.settings["confirm_weeks"])
        held_trades = [t for t in self.trades if t.exit_reason == "END_OF_DATA"]
        held = {t.ticker for t in held_trades}
        saved_pending = self._pending
        self._pending = {}

        for stock in self.stocks:
            start = max(0, len(stock.df) - (confirm * 5 + 3))
            for i in range(start, len(stock.df)):
                day_i = stock.df.index[i]
                self._advance_pending(stock, i, day_i, self._regime_ok(day_i))

        rows = []
        for stock in self.stocks:
            t = stock.ticker
            pend = self._pending.get(t)
            if pend is None:
                continue
            bar_idx = len(stock.df) - 1
            day = stock.df.index[bar_idx]
            n = self._week_id(t, bar_idx) - pend.w_idx
            regime_ok = self._regime_ok(day)
            sr = self._entry_ready(stock, bar_idx, pend, regime_ok)
            sector = get_sector(t)
            if t in held:
                state, note, score = "HELD", "held — entry blocked", None
            elif sr is not None:
                state, note = "TRIGGERED", ""
                score = sr["total"]
                if len(held) >= self._max_pos:
                    note = "slots full"
                elif (
                    sum(1 for h in held_trades if h.sector == sector)
                    >= self._max_per_sector
                ):
                    note = "sector cap"
            else:
                state, note, score = "ARMED", f"confirm {n}/{confirm}w", None
            rows.append(
                {
                    "ticker": t,
                    "state": state,
                    "cross_date": stock.df.index[pend.bar_idx],
                    "n": n,
                    "score": score,
                    "close": float(stock.df["close"].iloc[bar_idx]),
                    "sector": sector,
                    "note": note,
                }
            )
        self._pending = saved_pending
        return rows

    def scan_latest(self, expired_weeks: int = 6) -> list[dict]:
        """Fast live scan: latest bullish crosses + entry readiness (no run()).

        Replays only the trailing confirm window per stock into a fresh
        _pending (no portfolio), recording the FIRST week where every
        spec-B gate passes — the week the backtest would have entered
        (highbreak = signal-candle high broken; close = close above it).
        A Long is generated only while the cross's confirm window still
        covers the scan date (n = weeks since cross ≤ confirm at the run
        date); an earlier qualification with a stale window falls back to
        EXPIRED/hidden. Classifies each stock as TRIGGERED (a qualifying
        week exists in the still-open window; note says "all gates OK"
        when that was the latest evaluated week, else "qualified <date>"
        for an earlier, possibly retraced break), ARMED (pending live,
        failing gates named) or EXPIRED (never qualified, armed within
        `expired_weeks`, timed out or bear-cancelled). Skips the full
        simulation.
        """
        confirm = int(self.settings["confirm_weeks"])
        # run() normally builds these; the scan path skips run().
        if not getattr(self, "_stock_map", None):
            self._stock_map = {s.ticker: s for s in self.stocks}
            self._regime_close, self._regime_ema = self._index_regime_gate()
        saved_pending, saved_positions = self._pending, self.positions
        self._pending, self.positions = {}, []
        self._signals_generated = 0
        self._signals_blocked = 0
        expired: dict[str, dict] = {}
        first: dict[str, dict] = {}

        for stock in self.stocks:
            t = stock.ticker
            start = max(0, len(stock.df) - (confirm * 5 + 3))
            for i in range(start, len(stock.df)):
                day_i = stock.df.index[i]
                regime_ok = self._regime_ok(day_i)
                before = self._pending.get(t)
                self._advance_pending(stock, i, day_i, regime_ok)
                if before is not None and self._pending.get(t) is None:
                    w = self._week_id(t, i)
                    expired[t] = {
                        "cross_bar": before.bar_idx,
                        "bar_idx": i,
                        "reason": (
                            "timeout"
                            if w - before.w_idx > confirm
                            else "bearish cross"
                        ),
                    }
                # Backtest parity: the first window week where every gate
                # passes IS the entry (mirrors _try_entries' order —
                # advance, then gates at the week end; n=0 at the arm bar
                # fails n_window, so no same-bar false trigger).
                pend = self._pending.get(t)
                if (
                    pend is not None
                    and first.get(t) is None
                    and self._week_ended(t, i)
                ):
                    _g, sr = self._entry_gates(stock, i, pend, regime_ok)
                    if sr is not None:
                        first[t] = {
                            "bar_idx": i,
                            "cross_bar": pend.bar_idx,
                            "entry_date": stock.df.index[i],
                            "n": self._week_id(t, i) - pend.w_idx,
                            "score": float(sr["total"]),
                            "level": pend.level,
                        }

        def _arm_score(stock, bar_idx):
            sr = compute_score_at_bar(
                stock, bar_idx, self.nifty_df, self.settings
            )
            return float(sr["total"]) if sr else None

        rows: list[dict] = []
        for stock in self.stocks:
            t = stock.ticker
            bar_idx = len(stock.df) - 1
            # Gates evaluate at the last COMPLETED week — a mid-week run's
            # latest bar has not closed its week yet.
            eval_bar = bar_idx
            while eval_bar > 0 and not self._week_ended(t, eval_bar):
                eval_bar -= 1
            w_last = self._week_id(t, eval_bar)
            close = float(stock.df["close"].iloc[bar_idx])
            sector = get_sector(t)
            pend = self._pending.get(t)
            hit = first.get(t)
            cur_n = (
                w_last - self._week_id(t, hit["cross_bar"])
                if hit is not None
                else None
            )
            if hit is not None and cur_n <= confirm:
                # Backtest parity: the first window week where all gates
                # passed — survives a later bear/timeout and a retrace.
                # Only while the cross's confirm window still covers the
                # scan date: a stale qualification falls through to
                # EXPIRED/hidden, never a live Long.
                note = (
                    "all gates OK"
                    if hit["bar_idx"] == eval_bar
                    else f"qualified {hit['entry_date']:%Y-%m-%d}"
                )
                rows.append(
                    {
                        "ticker": t,
                        "sector": sector,
                        "state": "TRIGGERED",
                        "cross_date": stock.df.index[hit["cross_bar"]],
                        "n": cur_n,
                        "score": hit["score"],
                        "close": close,
                        "level": hit["level"],
                        "missing": [],
                        "note": note,
                    }
                )
            elif pend is not None:
                # A pass at eval_bar would have been recorded as `hit`
                # during the replay — this branch is ARMED-only.
                n = w_last - pend.w_idx
                regime_ok = self._regime_ok(stock.df.index[eval_bar])
                g, _sr = self._entry_gates(stock, eval_bar, pend, regime_ok)
                if not g["week_ended"]:
                    state, note = "ARMED", "awaiting weekly close"
                    missing = [k for k in _GATE_ORDER if not g[k]]
                elif not g["n_window"]:
                    state, note = "ARMED", f"cross bar n={n}/{confirm}"
                    missing = [k for k in _GATE_ORDER if not g[k]]
                elif not g["regime"]:
                    state, note = "ARMED", "NIFTY regime off"
                    missing = [k for k in _GATE_ORDER if not g[k]]
                else:
                    state = "ARMED"
                    missing = [k for k in _GATE_ORDER if not g[k]]
                    note = "missing: " + ", ".join(
                        _GATE_LABELS[k] for k in missing
                    )
                rows.append(
                    {
                        "ticker": t,
                        "sector": sector,
                        "state": state,
                        "cross_date": stock.df.index[pend.bar_idx],
                        "n": n,
                        "score": _arm_score(stock, pend.bar_idx),
                        "close": close,
                        "level": pend.level,
                        "missing": missing,
                        "note": note,
                    }
                )
            else:
                exp = expired.get(t)
                if exp is None:
                    continue
                if w_last - self._week_id(t, exp["bar_idx"]) > expired_weeks:
                    continue
                rows.append(
                    {
                        "ticker": t,
                        "sector": sector,
                        "state": "EXPIRED",
                        "cross_date": stock.df.index[exp["cross_bar"]],
                        "n": confirm + 1,
                        "score": _arm_score(stock, exp["cross_bar"]),
                        "close": close,
                        "level": None,
                        "missing": [],
                        "note": exp["reason"],
                    }
                )

        self._pending, self.positions = saved_pending, saved_positions
        self.scan_meta = {
            "as_of": (
                max(s.df.index[-1] for s in self.stocks)
                if self.stocks else None
            ),
            "scanned": len(self.stocks),
            "crosses_in_window": self._signals_generated,
            "expired_weeks": expired_weeks,
        }
        return rows

    # ── Exits (all modes) ────────────────────────────────────────────────────

    def _close(self, pos, price, day, reason: str):
        return _close_position(
            pos,
            float(price),
            day,
            reason,
            commission_pct=self.settings.get("commission_pct", 0.0),
        )

    def _risk_exit(self, pos, stock, bar) -> TradeResult | None:
        """R-Bracket / Chandelier intraday exit. Levels carried from the
        previous bar (orders issued on bar N fill on bar N+1 — Pine semantics)."""
        mode = self.settings["exit_mode"]
        if mode == "signals":
            return None
        high = float(bar["high"])
        low = float(bar["low"])
        opn = float(bar["open"])
        day = bar.name
        entry = pos.entry_price

        if mode == "rbracket":
            r = entry * self.settings["sl_pct"] / 100.0
            base = entry - r
            be = self.settings["be_act_r"]
            if pos.peak_price >= entry + be * r:
                stop = max(base, pos.peak_price - be * r)
            else:
                stop = base
            tp = entry + r * self.settings["rr_ratio"]
            if low <= stop:
                exit_px = opn if opn < stop else max(stop, low)
                return self._close(pos, exit_px, day, "RBRACKET_STOP")
            if high >= tp:
                exit_px = opn if opn > tp else tp  # limit: gap-up fills better
                return self._close(pos, exit_px, day, "TARGET")
            if high > pos.peak_price:
                pos.peak_price = high  # riskHi for the next bar
            return None

        # chandelier: pos.stop_loss IS the chandelier stop (seeded at fill)
        if low <= pos.stop_loss:
            exit_px = opn if opn < pos.stop_loss else max(pos.stop_loss, low)
            return self._close(pos, exit_px, day, "CHAND_STOP")
        atr_w = self._weekly_atr_asof(stock, day)
        if high > pos.peak_price:
            pos.peak_price = high
        if atr_w is not None and not np.isnan(atr_w) and atr_w > 0:
            cand = pos.peak_price - self.settings["atr_mult"] * atr_w
            if cand > pos.stop_loss:
                pos.stop_loss = cand
        return None

    def _check_exits(self, day):
        closed = []
        for pos in self.positions:
            stock = self._stock_map.get(pos.ticker)
            if stock is None or day not in stock.df.index:
                continue
            bar_idx = int(stock.df.index.get_loc(day))
            bar = stock.df.iloc[bar_idx]
            # Pending weekly-sell fills at today's open (issued yesterday).
            if pos.ticker in self._exit_pending:
                closed.append(self._close(pos, bar["open"], day, "WEEKLY_SELL"))
                continue
            # Weekly bear cross at the weekly close -> schedule next-open
            # exit; risk exits stay active on this bar (both order types live).
            if self._week_ended(pos.ticker, bar_idx):
                _, bear = self._weekly_cross(stock, self._week_id(pos.ticker, bar_idx))
                if bear:
                    self._exit_pending.add(pos.ticker)
            result = self._risk_exit(pos, stock, bar)
            if result is not None:
                self._exit_pending.discard(pos.ticker)
                closed.append(result)

        for trade in closed:
            self.trades.append(trade)
            self._cash += trade.pnl + trade.investment
            if self._rotation_enabled:
                self.sector_tracker.record_trade(trade)
            self._exit_pending.discard(trade.ticker)
        if closed:
            self.positions = [p for p in self.positions if p.exit_date is None]
            # JustClosed: non-signal closes re-open the confirmation gate,
            # re-armed with the same preconditions as a fresh cross.
            for trade in closed:
                if trade.exit_reason == "WEEKLY_SELL":
                    continue
                stock = self._stock_map.get(trade.ticker)
                if stock is None or trade.exit_date not in stock.df.index:
                    continue
                if stock.weekly_df is None:
                    continue
                t = trade.ticker
                bar_idx = int(stock.df.index.get_loc(trade.exit_date))
                # Use the last fully closed week for the MA state (a partial
                # week's weekly_df row would bake in that week's final close).
                w = self._week_id(t, bar_idx)
                if not self._week_ended(t, bar_idx):
                    w -= 1
                if w < 0 or w >= len(stock.w_hma) or w >= len(stock.w_ema50):
                    continue
                w_fast = stock.w_hma.iloc[w]
                w_slow = stock.w_ema50.iloc[w]
                if pd.isna(w_fast) or pd.isna(w_slow) or w_fast <= w_slow:
                    continue
                if not self._regime_ok(trade.exit_date):
                    continue
                sr = compute_score_at_bar(stock, bar_idx, self.nifty_df, self.settings)
                if (
                    sr is None
                    or sr["is_sideways"]
                    or sr["total"] < self.settings["score_threshold"]
                ):
                    continue
                ws_ = self._week_start(t, self._week_id(t, bar_idx))
                wk = stock.df.iloc[ws_ : bar_idx + 1]
                self._pending[t] = Pending(
                    bar_idx, w, float(wk["high"].max()), float(wk["low"].min())
                )


# ============================================================
# MAIN
# ============================================================


def print_status(engine: V2Engine, rows: list[dict]) -> None:
    """Console report: HELD / TRIGGERED / ARMED as of the latest bar."""
    s = engine.settings
    label = s.get("universe_label", "?")
    tf = s.get("timeframe", "daily")
    held = [t for t in engine.trades if t.exit_reason == "END_OF_DATA"]
    trig = [r for r in rows if r["state"] == "TRIGGERED"]
    armed = [r for r in rows if r["state"] == "ARMED"]

    asof = max((t.exit_date for t in held), default=None)
    if asof is None and engine.stocks:
        asof = max(st.df.index[-1] for st in engine.stocks)
    print("")
    print("=" * 74)
    print(f"  LONG STATUS as of {asof.date() if asof else '?'} — {label} — {tf} bars")
    print(
        f"  confirm {s['confirm_weeks']}w, score >= {s['score_threshold']}, "
        f"slots {s['max_positions']} x {s['position_size_pct']:.0f}%"
    )
    print("=" * 74)

    print(f"\n  HELD ({len(held)}) — open through today's close:")
    for t in sorted(held, key=lambda x: -x.pnl_pct):
        print(
            f"    {t.ticker:<14} entry {t.entry_date.date()} @Rs.{t.entry_price:<10,.1f}"
            f" mark Rs.{t.exit_price:<10,.1f} {t.pnl_pct:+7.1f}%  {t.sector}"
        )

    fill_note = (
        "fills at stop level (same bar)"
        if s.get("entry_trigger") == "highbreak"
        else "fills at next open"
    )
    print(
        f"\n  TRIGGERED ({len(trig)}) — all 5 gates green on the latest bar, "
        f"{fill_note}:"
    )
    for r in sorted(trig, key=lambda x: -(x["score"] or 0)):
        note = f"  [{r['note']}]" if r["note"] else ""
        print(
            f"    {r['ticker']:<14} score {r['score'] or 0:5.1f}  "
            f"cross {r['cross_date'].date()} n={r['n']}/{s['confirm_weeks']}w  "
            f"close {r['close']:,.1f}  {r['sector']}{note}"
        )

    print(f"\n  ARMED ({len(armed)}) — inside the confirm window, gates pending:")
    for r in sorted(armed, key=lambda x: -x["n"]):
        print(
            f"    {r['ticker']:<14} cross {r['cross_date'].date()} "
            f"n={r['n']}/{s['confirm_weeks']}w  close {r['close']:,.1f}  "
            f"{r['sector']}  ({r['note']})"
        )
    print("")


def print_scan(engine: V2Engine, rows: list[dict]) -> None:
    """Console report for scan_latest(): TRIGGERED / ARMED / EXPIRED."""
    s = engine.settings
    meta = getattr(engine, "scan_meta", {})
    as_of = meta.get("as_of")
    as_of_s = as_of.strftime("%Y-%m-%d") if as_of is not None else "-"
    confirm = int(s["confirm_weeks"])
    trig = [r for r in rows if r["state"] == "TRIGGERED"]
    armed = [r for r in rows if r["state"] == "ARMED"]
    expired = [r for r in rows if r["state"] == "EXPIRED"]
    fill_note = (
        f"buy-stop fill at level" if s["entry_trigger"] == "highbreak"
        else "fill next open"
    )

    print("")
    print("=" * 60)
    print(f"  SCAN — HMAxEMA_VP · as of {as_of_s} ({s.get('timeframe', 'daily')})")
    print(
        f"  {s.get('universe_label', '-')} · confirm {confirm}w · "
        f"score>={s['score_threshold']} · {s['entry_trigger']} trigger"
    )
    print("=" * 60)

    print(f"\n  ENTRY NOW — TRIGGERED ({len(trig)}), {fill_note}:")
    for r in sorted(trig, key=lambda x: -(x["score"] or 0)):
        lvl = f"  level {r['level']:,.1f}" if r["level"] is not None else ""
        note_s = f"  [{r['note']}]" if r["note"] != "all gates OK" else ""
        print(
            f"    {r['ticker']:<14} score {r['score'] or 0:5.1f}  "
            f"cross {r['cross_date'].date()} n={r['n']}/{confirm}w  "
            f"close {r['close']:,.1f}{lvl}  {r['sector']}{note_s}"
        )

    print(f"\n  ARMED ({len(armed)}) — bullish cross, waiting for the break:")
    for r in sorted(armed, key=lambda x: -x["n"]):
        sc = f"{r['score']:5.1f}" if r["score"] is not None else "  -  "
        print(
            f"    {r['ticker']:<14} score {sc}  cross {r['cross_date'].date()} "
            f"n={r['n']}/{confirm}w  close {r['close']:,.1f}  "
            f"{r['sector']}  ({r['note']})"
        )

    if expired:
        print(f"\n  RECENTLY EXPIRED ({len(expired)}):")
        for r in sorted(expired, key=lambda x: x["cross_date"], reverse=True):
            print(
                f"    {r['ticker']:<14} cross {r['cross_date'].date()}  "
                f"{r['sector']}  ({r['note']})"
            )

    print(
        f"\n  scanned {meta.get('scanned', len(engine.stocks))} stocks · "
        f"{meta.get('crosses_in_window', 0)} crosses in the confirm window"
    )
    print("")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Backtest the HMAxEMA_VP Swing strategy (Pine V4 port)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=260,
        help="Bars skipped before the sim window for indicator warmup "
        "(default: 260 = ~1y daily / ~5y weekly)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="daily",
        choices=["daily", "weekly"],
        help="Simulation bar size (default: daily; weekly = Pine on a 1W "
        "chart — signals, confirm window and fills all count in weeks)",
    )
    parser.add_argument(
        "--years", type=float, default=5, help="Years of history (default: 5)"
    )
    parser.add_argument(
        "--tickers", type=str, nargs="*", help="Specific tickers (default: universe)"
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="nifty50",
        choices=["nifty50", "nifty500", "midcap100", "fno", "all"],
        help="Stock universe (default: nifty50; nifty500 = live CNX500 list)",
    )
    parser.add_argument(
        "--exit-mode",
        type=str,
        default="signals",
        choices=["signals", "rbracket", "chandelier"],
        help="Exit mode (default: signals — Pine baseline)",
    )
    parser.add_argument(
        "--score", type=float, default=50, help="Min score for entry (default: 50)"
    )
    parser.add_argument(
        "--w-fast", type=int, default=18, help="Weekly HMA length (default: 18)"
    )
    parser.add_argument(
        "--w-slow", type=int, default=44, help="Weekly EMA length (default: 44)"
    )
    parser.add_argument(
        "--confirm-weeks", type=int, default=15, help="Confirm expiry in weekly bars"
    )
    parser.add_argument(
        "--capital", type=float, default=1_000_000, help="Initial capital (Rs.)"
    )
    parser.add_argument(
        "--max-positions", type=int, default=5, help="Max concurrent positions"
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=10.0,
        help="Position size as %% of current portfolio value per entry "
        "(default 10; use 25 with --max-positions 4 for equal quarters)",
    )
    parser.add_argument(
        "--max-per-sector", type=int, default=2, help="Max positions per sector"
    )
    parser.add_argument(
        "--sl-pct", type=float, default=8.0, help="Stop-loss %% (1R, default 8)"
    )
    parser.add_argument(
        "--rr", type=float, default=3.0, help="R-Bracket reward:risk (default 3)"
    )
    parser.add_argument(
        "--be-act-r", type=float, default=1.0, help="Breakeven activation in R"
    )
    parser.add_argument(
        "--atr-mult", type=float, default=2.0, help="Chandelier x weekly ATR"
    )
    parser.add_argument(
        "--struct-buf", type=float, default=0.3, help="Structure buffer x weekly ATR"
    )
    parser.add_argument(
        "--struct-cap",
        type=float,
        default=2.0,
        help="Structure stop cap x chandelier distance",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=0.0,
        help="%% equity risk per trade (0 = Pine default: 10%% position sizing)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.1,
        help="Commission %% per side (default 0.1, Pine parity)",
    )
    parser.add_argument(
        "--vp-weeks", type=int, default=12, help="VP lookback in weekly candles"
    )
    parser.add_argument("--vp-rows", type=int, default=30, help="VP price bins")
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="HTML report path (default: Reports/backtest_v2_report.html; "
        "with --scan: Reports/scan_v2_latest.html)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="Reports/backtest_v2_trades.csv",
        help="Trades CSV path",
    )
    parser.add_argument(
        "--no-html", action="store_true", help="Skip HTML report generation"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print live HELD/TRIGGERED/ARMED entry status as of the latest "
        "bar instead of the backtest report",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Fast live scan: latest bullish HMA x EMA crossovers + entry "
        "readiness as of the latest bar (skips the simulation); prints the "
        "scan console report and writes the scan dashboard",
    )
    parser.add_argument(
        "--entry-trigger",
        choices=["close", "highbreak"],
        default="close",
        help="close = Pine (bar closes above cross-bar high, fill next open); "
        "highbreak = enter when the high breaks (buy-stop fill at level, "
        "same bar)",
    )
    args = parser.parse_args()

    settings = {
        "timeframe": args.timeframe,
        "warmup_bars": args.warmup,
        "w_fast_len": args.w_fast,
        "w_slow_len": args.w_slow,
        "score_threshold": args.score,
        "confirm_weeks": args.confirm_weeks,
        "entry_trigger": args.entry_trigger,
        "exit_mode": args.exit_mode,
        "sl_pct": args.sl_pct,
        "rr_ratio": args.rr,
        "be_act_r": args.be_act_r,
        "atr_mult": args.atr_mult,
        "struct_buf": args.struct_buf,
        "struct_cap_mult": args.struct_cap,
        "risk_pct": args.risk_pct,
        "commission_pct": args.commission,
        "vp_weekly_lookback": args.vp_weeks,
        "vp_rows": args.vp_rows,
        "initial_capital": args.capital,
        "max_positions": args.max_positions,
        "max_positions_per_sector": args.max_per_sector,
        "position_size_pct": args.position_size,
    }

    if args.tickers:
        tickers = args.tickers
        settings["universe_label"] = f"Custom ({len(tickers)} tickers)"
    else:
        universe_name = {
            "nifty50": "NIFTY 50",
            "nifty500": "NIFTY 500",
            "midcap100": "NIFTY MIDCAP 100",
            "fno": "FnO STOCKS",
            "all": "ALL (Combined)",
        }[args.universe]
        tickers = get_universe(universe_name)
        settings["universe_label"] = universe_name

    period = f"{int(args.years)}y"
    engine = V2Engine(settings)
    engine.load_data(tickers, period=period)

    if args.scan:
        from .backtest_report import generate_scan_html

        rows = engine.scan_latest()
        print_scan(engine, rows)
        generate_scan_html(
            rows,
            engine.settings,
            getattr(engine, "scan_meta", {}),
            args.html or "Reports/scan_v2_latest.html",
        )
        return

    metrics = engine.run()

    if args.status:
        print_status(engine, engine.run_status())
        return

    engine.print_report(metrics)

    if metrics.get("total_trades", 0) > 0:
        from .backtest_report import generate_html_report, save_trades_csv

        save_trades_csv(engine.trades, args.csv)
        if not args.no_html:
            generate_html_report(engine, metrics, args.html or "Reports/backtest_v2_report.html")


if __name__ == "__main__":
    main()
