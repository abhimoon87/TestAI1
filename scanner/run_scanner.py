"""
HMAxEMA Stock Scanner — Main Runner
Scans Indian stocks and scores them using the same engine as the Pine Script indicator.

Usage:
    python scanner/run_scanner.py

Requirements:
    pip install yfinance pandas numpy
"""

import json
import logging
import os
import sys
import webbrowser
from datetime import datetime

from .data_fetcher import (
    fetch_batch_yfinance,
    fetch_index_data,
)
from .report import generate_html_report, save_report
from .scanner_engine import ScannerEngine, _enrich_rows_in_place, _score_ticker
from .universes import UNIVERSES

logger = logging.getLogger(__name__)

_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


def _load_settings():
    """Load settings from JSON file."""
    s = {}
    if os.path.exists(_SETTINGS_FILE):
        try:
            with open(_SETTINGS_FILE, "r") as f:
                s = json.load(f)
        except Exception as e:
            logger.debug("Failed to load settings: %s", e)
    return s


def print_banner():
    logger.info("""
╔══════════════════════════════════════════════════════════════╗
║          📊 HMAxEMA Stock Scanner — Indian Market           ║
║          Scoring engine mirrors Pine Script indicator        ║
╚══════════════════════════════════════════════════════════════╝
    """)


def select_universe() -> tuple:
    """Interactive universe selection. Returns (name, ticker_list)."""
    logger.info("━━━ Stock Universe ━━━")
    universes = list(UNIVERSES.keys())
    for i, name in enumerate(universes, 1):
        count = len(UNIVERSES[name])
        logger.info("  %2d. %-25s (%d stocks)", i, name, count)

    logger.info("  %2d. Custom (enter comma-separated tickers)", len(universes) + 1)

    while True:
        try:
            choice = input(f"\n  Select universe [1-{len(universes) + 1}]: ").strip()
            idx = int(choice) - 1
            if idx == len(universes):
                custom = input("  Enter tickers (comma-separated): ").strip()
                tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]
                if tickers:
                    return ("Custom", tickers)
                logger.warning("  ✗ No tickers provided.")
                continue
            if 0 <= idx < len(universes):
                name = universes[idx]
                return (name, UNIVERSES[name])
        except (ValueError, IndexError):
            pass
        logger.warning("  ✗ Invalid choice. Enter 1-%d.", len(universes) + 1)


def select_threshold() -> float:
    """Interactive threshold selection."""
    logger.info("\n━━━ Score Threshold ━━━")
    logger.info("  1. 70+  (EXCELLENT only)")
    logger.info("  2. 50+  (GOOD or better)  ← recommended")
    logger.info("  3. 30+  (MODERATE or better)")
    logger.info("  4. Custom value")

    while True:
        choice = input("\n  Select threshold [1-4]: ").strip()
        if choice == "1":
            return 70.0
        elif choice == "2":
            return 50.0
        elif choice == "3":
            return 30.0
        elif choice == "4":
            try:
                val = float(input("  Enter minimum score: ").strip())
                if 0 <= val <= 100:
                    return val
            except ValueError:
                pass
        logger.warning("  ✗ Invalid choice.")


def select_period() -> str:
    """Interactive period selection."""
    logger.info("\n━━━ Data Period ━━━")
    logger.info("  1. 6 months")
    logger.info("  2. 1 year   ← recommended")
    logger.info("  3. 2 years")

    while True:
        choice = input("\n  Select period [1-3]: ").strip()
        if choice == "1":
            return "6mo"
        elif choice == "2":
            return "1y"
        elif choice == "3":
            return "2y"
        logger.warning("  ✗ Invalid choice.")


def select_timeframe() -> str:
    """Interactive timeframe selection."""
    logger.info("\n━━━ Analysis Timeframe ━━━")
    logger.info("  1. Daily   (D)  ← default")
    logger.info("  2. Weekly  (W)")
    logger.info("  3. Monthly (M)")

    while True:
        choice = input("\n  Select timeframe [1-3]: ").strip()
        if choice == "1" or choice == "":
            return "D"
        elif choice == "2":
            return "W"
        elif choice == "3":
            return "M"
        logger.warning("  ✗ Invalid choice.")


def run_scan():
    """Main scan execution."""
    print_banner()

    # ── Selection ────────────────────────────────────────────────────────────
    universe_name, tickers = select_universe()
    threshold = select_threshold()
    period = select_period()
    timeframe = select_timeframe()

    # ── Load settings and override with CLI selections ──────────────────────
    settings = _load_settings()
    settings["data_period"] = period
    settings["timeframe"] = timeframe
    index_symbol = settings.get("index_symbol", "NSEI")

    tf_names = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
    logger.info("\n━━━ Configuration ━━━")
    logger.info("  Universe:   %s (%d stocks)", universe_name, len(tickers))
    logger.info("  Threshold:  %s+", threshold)
    logger.info("  Period:     %s", period)
    logger.info("  Timeframe:  %s", tf_names.get(timeframe, timeframe))
    logger.info("  FastMA:     %s%d  SlowMA: %s%d",
                settings.get("fast_ma_type", "HMA"), settings.get("fast_ma_len", 40),
                settings.get("slow_ma_type", "EMA"), settings.get("slow_ma_len", 50))
    logger.info("")

    # ── Fetch index for relative strength ──────────────────────────────────
    logger.info("━━━ Fetching %s index data ━━━", index_symbol)
    index_df = fetch_index_data(f"^{index_symbol}", period=period)
    if index_df is not None:
        logger.info("  ✓ %s loaded (%d bars)", index_symbol, len(index_df))
    else:
        logger.warning("  ⚠ %s data unavailable — RS will use proxy", index_symbol)
    logger.info("")

    # ── Fetch stock data ─────────────────────────────────────────────────────
    logger.info("━━━ Fetching stock data ━━━")
    stock_data = fetch_batch_yfinance(tickers, period=period)
    logger.info("\n  Fetched %d/%d stocks successfully.\n", len(stock_data), len(tickers))

    if not stock_data:
        logger.error("  ✗ No data fetched. Check your internet connection.")
        return

    # ── 3-Model Pipeline ──────────────────────────────────────────────────
    # Uses the same shared per-ticker scorer as ScannerEngine.scan / scan_stream
    # (scanner_engine._score_ticker) so the CLI can't drift from the GUI/engine.
    logger.info("━━━ 3-Model Pipeline ━━━")
    results = []
    filtered_out = 0
    direction_counts = {"Bull": 0, "Bear": 0}
    min_score_threshold = 50.0  # Swing trading threshold

    # Large custom universes (>500) behave like the engine: fast technical
    # pass first, then the top-200 are provider-enriched and re-scored so
    # totals/ratings include fundamentals.
    is_large = len(tickers) > 500
    if is_large:
        logger.info("  Large universe (%d stocks) — fast technical pass, enrich top 200 at end",
                    len(tickers))
        _engine = ScannerEngine()
        global_data = _engine._fetch_global_enrichment(settings)
        enrich = _engine._enrich_with_providers
    else:
        global_data = None
        enrich = lambda _t, s, _g: dict(s)  # no 5-provider enrichment on small lists

    for i, (ticker, df) in enumerate(stock_data.items(), 1):
        logger.info("  [%d/%d] %s...", i, len(stock_data), ticker)

        # Crossover filter -> direction -> fundamentals -> score -> rating gate
        scores, reason = _score_ticker(
            ticker, df,
            settings=settings, timeframe=timeframe, index_df=index_df,
            trend_filter="All",  # CLI has no directional filter
            is_large=is_large,
            global_data=global_data,
            enrich=enrich,
        )
        if scores is None:
            if reason == "filtered":
                filtered_out += 1
                logger.info("    ✖ filtered")
            else:
                logger.info("    ⚠ %s", reason)
            continue

        direction = scores["trend_dir"]
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
        results.append(scores)

        total = scores["total"]
        dir_icon = "▲" if direction == "Bull" else "▼"
        icon = "🟢" if total >= 70 else ("🟡" if total >= 50 else ("🟠" if total >= 30 else "🔴"))
        tag = "✓" if total >= min_score_threshold else "✗"
        sideways = " [CHOP]" if scores.get("is_sideways") else ""
        logger.info("    %s %s %s %.1f%s", dir_icon, tag, icon, total, sideways)

    # ── Phase 2 (large universes): enrich top 200 + re-score like the engine ─
    if is_large and results:
        results.sort(key=lambda x: x.get("total", 0) or 0, reverse=True)
        top_n = min(200, len(results))
        top = results[:top_n]
        rest = results[top_n:]
        logger.info("Enriching top %d of %d with fundamentals/sentiment...",
                    top_n, len(results))
        enriched_top = _enrich_rows_in_place(
            top, stock_data,
            settings=settings, global_data=global_data,
            timeframe=timeframe, index_df=index_df,
            enrich=enrich,
        )
        results = enriched_top + rest
        results.sort(key=lambda x: x.get("total", 0) or 0, reverse=True)
        logger.info("Top %d enrichment complete — scores re-computed with fundamentals",
                    top_n)

    if not results:
        logger.warning("\n  ✗ No stocks passed the filter.")
        return

    # ── Pipeline Summary ──────────────────────────────────────────────────
    logger.info("\n━━━ Pipeline Summary ━━━")
    logger.info("  Total stocks:  %d", len(stock_data))
    logger.info("  Filtered out:  %d (no recent crossover)", filtered_out)
    logger.info("  Passed filter: %d (%d Bull, %d Bear)",
                len(results), direction_counts.get('Bull', 0), direction_counts.get('Bear', 0))
    passed = len([r for r in results if r["total"] >= min_score_threshold])
    logger.info("  Scored %s+: %d", min_score_threshold, passed)

    # ── Generate report ──────────────────────────────────────────────────────
    logger.info("\n━━━ Generating report ━━━")
    html = generate_html_report(results, title=f"HMAxEMA Scanner — {universe_name}", threshold=threshold)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scanner_report_{timestamp}.html"
    save_report(html, filename)

    passed = len([r for r in results if r["total"] >= threshold])
    logger.info("  ✓ Report saved: %s", filename)
    logger.info("  ✓ %d/%d stocks scored %s+\n", passed, len(results), threshold)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("━━━ Top 10 Results ━━━")
    results.sort(key=lambda x: x["total"], reverse=True)
    logger.info("  %-5s %-15s %6s %-12s %8s %-8s", "Rank", "Ticker", "Score", "Rating", "1M Chg", "Trend")
    logger.info("  %s %s %s %s %s %s", "─"*5, "─"*15, "─"*6, "─"*12, "─"*8, "─"*8)

    for i, r in enumerate(results[:10], 1):
        score = r["total"]
        rating = r.get("combined_rating", "POOR")
        pc1m = f"{r.get('pc1m', 0) or 0:+.1f}%"
        logger.info("  %-5d %-15s %5.1f  %-12s %8s %-8s", i, r['ticker'], score, rating, pc1m, r['trend_dir'])

    # ── Open report ──────────────────────────────────────────────────────────
    open_report = input("\n  Open report in browser? [Y/n]: ").strip().lower()
    if open_report != "n":
        filepath = os.path.abspath(filename)
        webbrowser.open(f"file://{filepath}")

    logger.info("\n  ✓ Done!\n")


if __name__ == "__main__":
    try:
        run_scan()
    except KeyboardInterrupt:
        logger.info("\n\n  ✓ Scan cancelled.")
        sys.exit(0)
