"""Walk-forward validation harness for the HMA/EMA backtest strategy.

Why this exists
---------------
Parameter/gate discoveries made on the full backtest window are in-sample:
a threshold that "wins" on the whole history may just be curve-fit to that
tape.  This module splits the simulation window into two halves and runs the
standard protocol:

    TRAIN half   -> sweep a gate/threshold, pick the best configuration
    TEST  half   -> evaluate the *chosen* configuration out-of-sample

and also reports the FULL window for reference.  Because the engine's
sim_start/sim_end settings restrict the simulated dates while all indicators
are precomputed causally on the full series, each half is an independent,
lookahead-free simulation (positions left open at a window end are closed at
that window's last bar, not the data's last bar).

Usage
-----
    python -m scanner.walkforward --stop 5 --target 15 --min-adx 20
    python -m scanner.walkforward --adx-sweep "0,15,20,25,30,35" --regime
    python -m scanner.walkforward --tickers RELIANCE AXISBANK --skip-full

The ADX sweep mode is the headline: it sweeps min_adx_entry on the TRAIN half,
prints the chosen threshold, and applies it to the TEST half so you can see
whether the gate holds out-of-sample.  All data is served from the disk cache
when warm (see scanner.cache), so repeated runs are fast and offline.

Results notes (default GUI MA set, HMA40 x EMA50, 3y NIFTY 50):
  * The ADX>=20 gate improved every window and config tested (+3..+10 pts),
    20 was the best threshold on both halves, and the effect survived
    out-of-sample -- but OOS magnitudes were much smaller than the full-window
    numbers (losses became smaller losses / break-even, not profits).
  * Stacking the index-regime gate or sector rotation on top of ADX>=20 did
    NOT help out-of-sample; rotation became inert once ADX filtered the pool.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from .backtest import (
    DEFAULT_SETTINGS,
    NIFTY_50,
    BacktestEngine,
    precompute_nifty,
    precompute_stock,
)
from .data_fetcher import fetch_batch_yfinance, fetch_index_data

# GUI default MA set (settings_store) that the risk sweeps anchor on.
DEFAULT_MA = {"fast_ma_len": 40, "slow_ma_len": 50, "crossover_lookback": 20}

_HEAD = ("{tag:<30} {n:>3} tr  {ret:>8.2f}%  WR {wr:>5.1f}%  "
         "PF {pf:>5.2f}  DD {dd:>5.1f}%")


def _load_stocks(tickers, period, ma_settings):
    """Fetch (from disk cache when possible) and precompute one StockData per ticker."""
    nifty_df = precompute_nifty(fetch_index_data("^NSEI", period=period))
    raw = fetch_batch_yfinance(tickers, period=period, timeframe="D")
    full = {**DEFAULT_SETTINGS, **ma_settings}
    stocks = []
    for t in tickers:
        df = raw.get(t)
        if df is None:
            continue
        s = precompute_stock(t, df, full)
        if s is not None:
            stocks.append(s)
    return stocks, nifty_df


def _run_window(stocks, nifty_df, window, label, cfg, verbose=True,
                base_settings=None):
    """Run one simulation over the given calendar window (None = full).

    ``base_settings`` mirrors the scanner/MA settings into the engine config so
    the simulation reproduces the settings being validated.  Returns the
    engine's metrics dict (or {} when nothing ran).
    """
    eng = BacktestEngine({**(base_settings or DEFAULT_MA), **cfg, **(window or {})})
    eng.stocks = stocks
    eng.nifty_df = nifty_df
    m = eng.run()
    if verbose and m.get("total_trades"):
        print(_HEAD.format(tag=label, n=m["total_trades"], ret=m["total_return_pct"],
                           wr=m["win_rate"], pf=m["profit_factor"],
                           dd=m["max_drawdown_pct"]), flush=True)
    return m


def _slim(m: dict) -> dict | None:
    """Scalar-only view of the metrics dict (safe to render / serialize)."""
    if not m.get("total_trades"):
        return None
    return {k: m[k] for k in ("total_trades", "total_return_pct", "win_rate",
                              "profit_factor", "max_drawdown_pct")}


def _trading_midpoint(stocks, warmup_bars):
    """Midpoint date of the engine's simulation window (for the train/test split)."""
    all_dates = sorted(set().union(*[set(s.df.index) for s in stocks]))
    sim = all_dates[warmup_bars:]
    return sim[0] + (sim[-1] - sim[0]) / 2


def run_walkforward(tickers=None, period="3y", stop=5.0, target=15.0, trail=2.0,
                    min_adx=None, regime=False, rotation=False,
                    rot_block=-5.0, no_thursday=False, adx_sweep=(),
                    split_date=None, skip_full=False, skip_train=False,
                    skip_test=False, ma_overrides=None, verbose=True):
    """Run the full/train/test protocol and return a structured result.

    ``ma_overrides`` carries the scanner settings that must be mirrored in the
    engine (MA type/lengths, crossover lookback, ADX gate, indicator lengths) so
    the simulation validates the *current* settings, not the engine defaults.
    ``min_adx`` defaults to ``ma_overrides['min_adx_entry']`` (or 20).

    Returns a dict::
        {n_stocks, split_date, full, train_by_adx, train_anchor, chosen_adx,
         test_no_gate, test_chosen, params}
    Each metric is ``_slim``-ed (scalars only) or None.  When ``verbose`` the
    same tables the CLI prints are echoed to stdout.
    """
    tickers = tickers or list(NIFTY_50)
    ma_cfg = {**DEFAULT_MA, **(ma_overrides or {})}
    if min_adx is None:
        min_adx = float(ma_cfg.get("min_adx_entry", 0.0) or 20.0)

    stocks, nifty_df = _load_stocks(tickers, period, ma_cfg)
    if not stocks:
        print("No stocks could be precomputed (need >= 260 bars each). Aborting.")
        return {"n_stocks": 0, "error": "not enough data (need >= 3y for the 260-bar warmup)"}

    from .backtest import WARMUP_BARS
    if split_date is None:
        split_date = _trading_midpoint(stocks, WARMUP_BARS).date().isoformat()
    train = {"sim_end": split_date}
    # Test starts the day AFTER the split so the boundary date is not double-counted
    test = {"sim_start": (pd.Timestamp(split_date) + pd.Timedelta(days=1)).date().isoformat()}
    if verbose:
        print(f"Universe {len(stocks)} stocks | split date {split_date}")

    def base_cfg(**extra):
        cfg = {"stop_loss_pct": stop, "target_pct": target, "trail_pct": trail,
               "min_adx_entry": min_adx}
        if regime:
            cfg["index_regime_filter"] = True
        if rotation:
            cfg.update({"sector_rotation_enabled": True, "sector_rotation_lookback": 8,
                        "sector_block_threshold": rot_block, "sector_boost_weight": 0.5})
        if no_thursday:
            cfg["blocked_entry_weekdays"] = [3]
        cfg.update(extra)
        return cfg

    out = {"n_stocks": len(stocks), "split_date": split_date,
           "params": {"stop": stop, "target": target, "trail": trail,
                      "min_adx": min_adx}}

    # ---- FULL window ----
    out["full"] = None
    if not skip_full:
        if verbose:
            print("\n=== FULL window ===")
        out["full"] = _slim(_run_window(
            stocks, nifty_df, None, f"S{stop:g}/T{target:g} + gates", base_cfg(),
            verbose=verbose, base_settings=ma_cfg))

    # ---- TRAIN sweep (pick threshold) ----
    out["train_by_adx"] = {}
    out["chosen_adx"] = min_adx
    if not skip_train:
        if verbose:
            print(f"\n=== TRAIN window (<= {split_date}) -- pick best here ===")
        train_res = {}
        adx_levels = sorted(set(list(adx_sweep) or ([0.0, min_adx] if min_adx else [0.0])))
        for a in adx_levels:
            m = _run_window(stocks, nifty_df, train, f"min_adx >= {a:g}",
                            base_cfg(min_adx_entry=a), verbose=verbose,
                            base_settings=ma_cfg)
            sm = _slim(m)
            out["train_by_adx"][a] = sm
            if sm:
                train_res[a] = sm["total_return_pct"]
        out["train_anchor"] = out["train_by_adx"].get(0.0)
        out["chosen_adx"] = chosen = max(train_res, key=train_res.get) if train_res else min_adx
        if verbose:
            if 0.0 in train_res and chosen == 0.0:
                print(f"-> TRAIN prefers NO ADX gate (best was adx>=0 at {train_res[0]:+.2f}%)")
            else:
                print(f"-> chosen min_adx on TRAIN: {chosen:g} "
                      f"({train_res.get(chosen, float('nan')):+.2f}%)")
    else:
        out["train_anchor"] = None

    # ---- TEST / out-of-sample ----
    out["test_no_gate"] = None
    out["test_chosen"] = None
    if not skip_test:
        if verbose:
            print(f"\n=== TEST window (> {split_date}) -- out-of-sample ===")
        out["test_no_gate"] = _slim(_run_window(
            stocks, nifty_df, test, "no gate", base_cfg(min_adx_entry=0.0),
            verbose=verbose, base_settings=ma_cfg))
        out["test_chosen"] = _slim(_run_window(
            stocks, nifty_df, test, f"chosen adx >= {out['chosen_adx']:g}",
            base_cfg(min_adx_entry=out["chosen_adx"]), verbose=verbose,
            base_settings=ma_cfg))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward validation of HMA/EMA backtest settings and gates")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Specific tickers (default: NIFTY 50)")
    parser.add_argument("--stop", type=float, default=5.0)
    parser.add_argument("--target", type=float, default=15.0)
    parser.add_argument("--trail", type=float, default=2.0)
    parser.add_argument("--min-adx", type=float, default=20.0)
    parser.add_argument("--adx-sweep", type=str, default="",
                        help="Comma list, e.g. '0,15,20,25,30' -- sweep on TRAIN, "
                             "apply the winner to TEST")
    parser.add_argument("--split-date", type=str, default=None,
                        help="YYYY-MM-DD train/test boundary (default: sim midpoint)")
    parser.add_argument("--regime", action="store_true")
    parser.add_argument("--rotation", action="store_true")
    parser.add_argument("--rot-block", type=float, default=-5.0)
    parser.add_argument("--no-thursday", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    sweep = [float(x) for x in args.adx_sweep.split(",") if x.strip()] if args.adx_sweep else ()

    t0 = time.time()
    run_walkforward(
        tickers=args.tickers,
        period=f"{int(args.years)}y",
        stop=args.stop, target=args.target, trail=args.trail,
        min_adx=args.min_adx, regime=args.regime, rotation=args.rotation,
        rot_block=args.rot_block, no_thursday=args.no_thursday,
        adx_sweep=sweep, split_date=args.split_date,
        skip_full=args.skip_full, skip_train=args.skip_train, skip_test=args.skip_test,
    )
    print(f"\n({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
