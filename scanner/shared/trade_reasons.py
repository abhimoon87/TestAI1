"""Generate human-readable trade reasons from a score dict.

Called from the UI to show "Why this trade?" pointers when a ticker
is expanded.  Every reason is a plain string — no Flet dependency here
so the module is testable without a GUI.
"""

from __future__ import annotations

import numpy as np


def _num(val, default: float = 0.0) -> float:
    """Coerce to float, tolerating None/str/NaN."""
    try:
        num = float(val)
    except (TypeError, ValueError):
        return default
    try:
        if np.isnan(num):
            return default
    except TypeError:
        return default
    return num


def build_trade_reasons(row: dict, max_reasons: int = 8) -> list[str]:
    """Return a list of concise reason strings for *row*.

    Reasons are ordered by importance (entry signal first, then trend,
    momentum, volume, institutional, fundamentals, risk flags).
    At most *max_reasons* items are returned to keep the UI compact.
    """
    try:
        max_reasons = int(max_reasons)
    except (TypeError, ValueError):
        max_reasons = 8
    if max_reasons < 0:
        max_reasons = 0
    if not isinstance(row, dict):
        return []
    reasons: list[str] = []

    # ── Entry signal ────────────────────────────────────────────────────
    if row.get("entry_signal"):
        bars = row.get("crossover_bars_ago")
        try:
            bars_n = int(bars) if bars is not None else None
        except (TypeError, ValueError):
            bars_n = None
        if bars_n is not None and bars_n >= 0:
            reasons.append(
                f"Fresh HMA×EMA bullish crossover ({bars_n} bar{'s' if bars_n != 1 else ''} ago)"
            )
        else:
            reasons.append("Bullish HMA×EMA crossover active")
    if row.get("weekly_entry_signal"):
        reasons.append("Weekly HMA confirming higher-timeframe uptrend")

    # ── Trend ───────────────────────────────────────────────────────────
    trend = _num(row.get("trend"))
    if trend >= 10:
        reasons.append(f"Strong trend score {trend:.0f}/15")
    elif trend >= 6:
        reasons.append(f"Solid trend structure ({trend:.0f}/15)")

    adx = row.get("adx_val")
    try:
        adx_n = float(adx) if adx is not None else None
        adx_bad = adx_n is None or bool(np.isnan(adx_n))
    except (TypeError, ValueError):
        adx_n, adx_bad = None, True
    if not adx_bad:
        if adx_n >= 25:
            reasons.append(f"ADX {adx_n:.0f} — trending market")
        elif adx_n < 18:
            reasons.append(f"ADX {adx_n:.0f} — weak trend, cautious")

    if row.get("above_poc"):
        poc = row.get("vp_poc")
        try:
            poc_n = float(poc) if poc else None
        except (TypeError, ValueError):
            poc_n = None
        if poc_n:
            reasons.append(f"Above Volume Profile POC at ₹{poc_n:,.0f}")

    if row.get("close_above_both_ma"):
        reasons.append("Price above both fast & slow MAs")

    # ── Momentum ────────────────────────────────────────────────────────
    mom = _num(row.get("momentum"))
    if mom >= 10:
        reasons.append(f"Strong momentum ({mom:.0f}/15)")

    rsi_val = row.get("rsi_val")
    try:
        rsi_n = float(rsi_val) if rsi_val is not None else None
        rsi_bad = rsi_n is None or bool(np.isnan(rsi_n))
    except (TypeError, ValueError):
        rsi_n, rsi_bad = None, True
    if not rsi_bad:
        if 40 <= rsi_n <= 70:
            reasons.append(f"RSI {rsi_n:.0f} — healthy range")
        elif 70 < rsi_n <= 75:
            reasons.append(f"RSI {rsi_n:.0f} — overbought caution")

    macd = _num(row.get("macd"))
    if macd >= 5:
        reasons.append(f"MACD histogram rising ({macd:.0f}/7)")

    # ── Volume ──────────────────────────────────────────────────────────
    vol = _num(row.get("volume"))
    if vol >= 7:
        reasons.append(f"Strong volume confirmation ({vol:.0f}/10)")

    # ── Relative strength ───────────────────────────────────────────────
    rs = _num(row.get("rel_str"))
    if rs >= 7:
        reasons.append(f"Outperforming index ({rs:.0f}/10 relative strength)")

    # ── Institutional / enrichment ──────────────────────────────────────
    if row.get("_fii_is_buying"):
        fii_net = row.get("_fii_net")
        try:
            fii_n = abs(float(fii_net)) if fii_net is not None else None
        except (TypeError, ValueError):
            fii_n = None
        if fii_n is not None:
            reasons.append(f"FII buying ₹{fii_n:,.0f} Cr net")
        else:
            reasons.append("FII net buying signal")
    if row.get("_dii_is_buying"):
        dii_net = row.get("_dii_net")
        try:
            dii_n = abs(float(dii_net)) if dii_net is not None else None
        except (TypeError, ValueError):
            dii_n = None
        if dii_n is not None:
            reasons.append(f"DII buying ₹{dii_n:,.0f} Cr net")
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
    if volat >= 5:
        reasons.append(f"Low volatility — steady move ({volat:.0f}/5)")

    # ── Risk flags (always last) ───────────────────────────────────────
    if not rsi_bad and rsi_n > 75:
        reasons.append(f"Risk: RSI {rsi_n:.0f} — overbought")
    stoch = row.get("stoch")
    if stoch is not None and stoch <= 1:
        reasons.append("Risk: Stochastic outside healthy zone (20-80)")
    sideways = row.get("is_sideways")
    if sideways:
        sr = row.get("sideways_reasons") or []
        reasons.append(f"Risk: Choppy/sideways ({', '.join(sr) if sr else 'low ADX'})")

    # ── Trim ────────────────────────────────────────────────────────────
    return reasons[:max_reasons]
