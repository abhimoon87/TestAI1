"""HTML report and CSV export for the backtest engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .backtest_models import TradeResult

if TYPE_CHECKING:
    from .backtest import BacktestEngine


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
