"""
HMAxEMA Stock Scanner — Main Runner
Scans Indian stocks and scores them using the same engine as the Pine Script indicator.

Usage:
    python scanner/run_scanner.py

Requirements:
    pip install yfinance pandas numpy
"""

import os
import sys
import webbrowser
from datetime import datetime

from .universes import UNIVERSES, NIFTY_50
from .data_fetcher import fetch_stock_data, fetch_index_data, fetch_batch, fetch_fundamentals
from .scoring import compute_scores
from .report import generate_html_report, save_report


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          📊 HMAxEMA Stock Scanner — Indian Market           ║
║          Scoring engine mirrors Pine Script indicator        ║
╚══════════════════════════════════════════════════════════════╝
    """)


def select_universe() -> tuple:
    """Interactive universe selection. Returns (name, ticker_list)."""
    print("━━━ Stock Universe ━━━")
    universes = list(UNIVERSES.keys())
    for i, name in enumerate(universes, 1):
        count = len(UNIVERSES[name])
        print(f"  {i:2d}. {name:<25s} ({count} stocks)")

    print(f"  {len(universes) + 1:2d}. Custom (enter comma-separated tickers)")

    while True:
        try:
            choice = input(f"\n  Select universe [1-{len(universes) + 1}]: ").strip()
            idx = int(choice) - 1
            if idx == len(universes):
                custom = input("  Enter tickers (comma-separated): ").strip()
                tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]
                if tickers:
                    return ("Custom", tickers)
                print("  ✗ No tickers provided.")
                continue
            if 0 <= idx < len(universes):
                name = universes[idx]
                return (name, UNIVERSES[name])
        except (ValueError, IndexError):
            pass
        print(f"  ✗ Invalid choice. Enter 1-{len(universes) + 1}.")


def select_threshold() -> float:
    """Interactive threshold selection."""
    print("\n━━━ Score Threshold ━━━")
    print("  1. 70+  (EXCELLENT only)")
    print("  2. 50+  (GOOD or better)  ← recommended")
    print("  3. 30+  (MODERATE or better)")
    print("  4. Custom value")

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
        print("  ✗ Invalid choice.")


def select_period() -> str:
    """Interactive period selection."""
    print("\n━━━ Data Period ━━━")
    print("  1. 6 months")
    print("  2. 1 year   ← recommended")
    print("  3. 2 years")

    while True:
        choice = input("\n  Select period [1-3]: ").strip()
        if choice == "1":
            return "6mo"
        elif choice == "2":
            return "1y"
        elif choice == "3":
            return "2y"
        print("  ✗ Invalid choice.")


def run_scan():
    """Main scan execution."""
    print_banner()

    # ── Selection ────────────────────────────────────────────────────────────
    universe_name, tickers = select_universe()
    threshold = select_threshold()
    period = select_period()

    print(f"\n━━━ Configuration ━━━")
    print(f"  Universe:   {universe_name} ({len(tickers)} stocks)")
    print(f"  Threshold:  {threshold}+")
    print(f"  Period:     {period}")
    print()

    # ── Fetch NIFTY index for relative strength ──────────────────────────────
    print("━━━ Fetching NIFTY 50 index data ━━━")
    index_df = fetch_index_data("^NSEI", period=period)
    if index_df is not None:
        print(f"  ✓ NIFTY 50 loaded ({len(index_df)} bars)")
    else:
        print("  ⚠ NIFTY data unavailable — RS will use proxy")
    print()

    # ── Fetch stock data ─────────────────────────────────────────────────────
    print(f"━━━ Fetching stock data ━━━")
    stock_data = fetch_batch(tickers, period=period, delay=0.15)
    print(f"\n  Fetched {len(stock_data)}/{len(tickers)} stocks successfully.\n")

    if not stock_data:
        print("  ✗ No data fetched. Check your internet connection.")
        return

    # ── Score each stock ─────────────────────────────────────────────────────
    print("━━━ Computing scores ━━━")
    results = []
    for i, (ticker, df) in enumerate(stock_data.items(), 1):
        print(f"  [{i}/{len(stock_data)}] Scoring {ticker}...", end="", flush=True)
        # Attach fundamentals if not already present
        if not hasattr(df, '_fundamentals') or df._fundamentals is None:
            fund = fetch_fundamentals(ticker)
            if fund is not None:
                df._fundamentals = fund
        scores = compute_scores(df, index_df=index_df)
        if scores is not None:
            scores["ticker"] = ticker
            results.append(scores)
            total = scores["total"]
            icon = "🟢" if total >= 70 else ("🟡" if total >= 50 else ("🟠" if total >= 30 else "🔴"))
            sideways = " [CHOP]" if scores.get("is_sideways") else ""
            print(f" {icon} {total:.1f}{sideways}")
        else:
            print(" ⚠ insufficient data")

    if not results:
        print("\n  ✗ No stocks could be scored.")
        return

    # ── Generate report ──────────────────────────────────────────────────────
    print(f"\n━━━ Generating report ━━━")
    html = generate_html_report(results, title=f"HMAxEMA Scanner — {universe_name}", threshold=threshold)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scanner_report_{timestamp}.html"
    save_report(html, filename)

    passed = len([r for r in results if r["total"] >= threshold])
    print(f"  ✓ Report saved: {filename}")
    print(f"  ✓ {passed}/{len(results)} stocks scored {threshold}+\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("━━━ Top 10 Results ━━━")
    results.sort(key=lambda x: x["total"], reverse=True)
    print(f"  {'Rank':<5} {'Ticker':<15} {'Score':>6} {'Rating':<12} {'1M Chg':>8} {'Trend':<8}")
    print(f"  {'─'*5} {'─'*15} {'─'*6} {'─'*12} {'─'*8} {'─'*8}")

    for i, r in enumerate(results[:10], 1):
        score = r["total"]
        rating = "EXCELLENT" if score >= 70 else ("GOOD" if score >= 50 else ("MODERATE" if score >= 30 else "POOR"))
        pc1m = f"{r.get('pc1m', 0) or 0:+.1f}%"
        print(f"  {i:<5} {r['ticker']:<15} {score:>5.1f}  {rating:<12} {pc1m:>8} {r['trend_dir']:<8}")

    # ── Open report ──────────────────────────────────────────────────────────
    open_report = input("\n  Open report in browser? [Y/n]: ").strip().lower()
    if open_report != "n":
        filepath = os.path.abspath(filename)
        webbrowser.open(f"file://{filepath}")

    print("\n  ✓ Done!\n")


if __name__ == "__main__":
    try:
        run_scan()
    except KeyboardInterrupt:
        print("\n\n  ✓ Scan cancelled.")
        sys.exit(0)
