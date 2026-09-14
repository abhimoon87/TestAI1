"""Generate human-readable trade reasons from a score dict.

Called from the UI to show "Why this trade?" pointers when a ticker
is expanded.  Every reason is a plain string — no Flet dependency here
so the module is testable without a GUI.
"""

from __future__ import annotations

import numpy as np


def build_trade_reasons(row: dict, max_reasons: int = 8) -> list[str]:
    """Return a list of concise reason strings for *row*.

    Reasons are ordered by importance (entry signal first, then trend,
    momentum, volume, institutional, fundamentals, risk flags).
    At most *max_reasons* items are returned to keep the UI compact.
    """
    reasons: list[str] = []

    # ── Entry signal ────────────────────────────────────────────────────
    if row.get("entry_signal"):
        bars = row.get("crossover_bars_ago")
        if bars is not None and bars >= 0:
            reasons.append(f"Fresh HMA×EMA bullish crossover ({bars} bar{'s' if bars != 1 else ''} ago)")
        else:
            reasons.append("Bullish HMA×EMA crossover active")
    if row.get("weekly_entry_signal"):
        reasons.append("Weekly HMA confirming higher-timeframe uptrend")

    # ── Trend ───────────────────────────────────────────────────────────
    trend = row.get("trend") or 0
    if trend >= 10:
        reasons.append(f"Strong trend score {trend:.0f}/15")
    elif trend >= 6:
        reasons.append(f"Solid trend structure ({trend:.0f}/15)")

    adx = row.get("adx_val")
    if adx is not None and not np.isnan(adx):
        if adx >= 25:
            reasons.append(f"ADX {adx:.0f} — trending market")
        elif adx < 18:
            reasons.append(f"ADX {adx:.0f} — weak trend, cautious")

    if row.get("above_poc"):
        poc = row.get("vp_poc")
        if poc:
            reasons.append(f"Above Volume Profile POC at ₹{poc:,.0f}")

    if row.get("close_above_both_ma"):
        reasons.append("Price above both fast & slow MAs")

    # ── Momentum ────────────────────────────────────────────────────────
    mom = row.get("momentum") or 0
    if mom >= 10:
        reasons.append(f"Strong momentum ({mom:.0f}/15)")

    rsi_val = row.get("rsi_val")
    if rsi_val is not None and not np.isnan(rsi_val):
        if 40 <= rsi_val <= 70:
            reasons.append(f"RSI {rsi_val:.0f} — healthy range")
        elif rsi_val > 70:
            reasons.append(f"RSI {rsi_val:.0f} — overbought caution")

    macd = row.get("macd") or 0
    if macd >= 5:
        reasons.append(f"MACD histogram rising ({macd:.0f}/7)")

    # ── Volume ──────────────────────────────────────────────────────────
    vol = row.get("volume") or 0
    if vol >= 7:
        reasons.append(f"Strong volume confirmation ({vol:.0f}/10)")

    # ── Relative strength ───────────────────────────────────────────────
    rs = row.get("rel_str") or 0
    if rs >= 7:
        reasons.append(f"Outperforming index ({rs:.0f}/10 relative strength)")

    # ── Institutional / enrichment ──────────────────────────────────────
    if row.get("_fii_is_buying"):
        fii_net = row.get("_fii_net")
        if fii_net is not None:
            reasons.append(f"FII buying ₹{abs(fii_net):,.0f} Cr net")
        else:
            reasons.append("FII net buying signal")
    if row.get("_dii_is_buying"):
        dii_net = row.get("_dii_net")
        if dii_net is not None:
            reasons.append(f"DII buying ₹{abs(dii_net):,.0f} Cr net")
        else:
            reasons.append("DII net buying signal")

    delivery = row.get("_delivery_pct")
    if delivery is not None and delivery >= 55:
        reasons.append(f"High delivery volume ({delivery:.0f}%)")

    insider = row.get("_insider_score")
    if insider is not None and insider > 0.3:
        reasons.append("Positive insider activity")

    sentiment = row.get("_sentiment_score")
    if sentiment is not None and sentiment > 0.3:
        reasons.append("Positive news sentiment")

    # ── Fundamentals ────────────────────────────────────────────────────
    fund = row.get("fundamentals") or 0
    if fund >= 12:
        reasons.append(f"Strong fundamentals ({fund:.0f}/20)")
    elif fund >= 6:
        reasons.append(f"Decent fundamentals ({fund:.0f}/20)")

    pe_rel = row.get("_pe_relative_to_industry")
    if pe_rel is not None and pe_rel < 0.85:
        reasons.append("Undervalued vs industry P/E")

    if row.get("_is_quality_stock"):
        reasons.append("Quality stock (screener flagged)")

    # ── Volatility ──────────────────────────────────────────────────────
    volat = row.get("volatility") or 0
    if volat >= 4:
        reasons.append(f"Low volatility — steady move ({volat:.0f}/5)")

    # ── Risk flags (always last) ───────────────────────────────────────
    if rsi_val is not None and not np.isnan(rsi_val) and rsi_val > 75:
        reasons.append(f"Risk: RSI {rsi_val:.0f} — overbought")
    stoch = row.get("stoch")
    if stoch is not None and stoch <= 1:
        reasons.append("Risk: Stochastic exhaustion zone")
    sideways = row.get("is_sideways")
    if sideways:
        sr = row.get("sideways_reasons") or []
        reasons.append(f"Risk: Choppy/sideways ({', '.join(sr) if sr else 'low ADX'})")

    # ── Trim ────────────────────────────────────────────────────────────
    return reasons[:max_reasons]
