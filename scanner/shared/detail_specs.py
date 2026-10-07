"""Detail-panel specs shared by the app (Flet) and the HTML report.

Single source of truth for the chips, MA label and score-breakdown rows the
two detail renderers must keep identical: this module owns the values and
color *roles*, each renderer only maps a role to its own palette (theme dict
in the app, CSS vars in the report).
"""

from __future__ import annotations


def fmt_pct(v) -> str:
    """Signed %; promoter-level moves keep 2 decimals, flows get 1."""
    if v is None:
        return ""
    return f"{v:+.{2 if abs(v) < 10 else 1}f}%"


def ma_chip(r: dict) -> tuple[str, str]:
    """(text, color role) for the MA signal: crossover / bull / bear."""
    if r.get("ma_crossed_above"):
        ago = r.get("crossover_bars_ago", -1)
        cnt = r.get("crossover_count") or 0
        return (f"^ X{ago}({cnt})" if cnt > 1 else f"^ X{ago}"), "green"
    if r.get("ma_bullish"):
        return "^ Bull", "lime"
    return "v Bear", "red"


def spark_move(r: dict) -> float:
    """Net % move across the row's sparkline closes (for column sorting)."""
    px = r.get("px_tail") or []
    if len(px) < 2 or not px[0]:
        return 0.0
    return (px[-1] - px[0]) / px[0] * 100.0


def signal_specs(r: dict) -> list[tuple[str, str, str]]:
    """[(label, text, color role)] — the detail panel's signal chips."""
    rsi = float(r.get("rsi_val") or 0)
    adx = float(r.get("adx_val") or 0)
    macd = float(r.get("macd") or 0)
    atr = float(r.get("atr_pct") or 0)
    trend = r.get("trend_dir") or ""
    bull = trend == "Bull"
    above = bool(r.get("above_poc"))
    sideways = bool(r.get("is_sideways"))
    ma_text, ma_role = ma_chip(r)
    return [
        ("RSI", f"{rsi:.1f}", "green" if 40 <= rsi <= 70 else "orange"),
        ("ADX", f"{adx:.1f}", "green" if adx > 20 else "red"),
        ("MACD", f"{macd:.1f}", "green" if macd > 0 else "red"),
        ("ATR%", f"{atr:.2f}%", "orange" if atr > 3 else "text"),
        ("POC", "Above" if above else "Below", "green" if above else "red"),
        ("MA", ma_text, ma_role),
        (
            "Dir",
            f"^ {trend}" if bull else f"v {trend}",
            "green" if bull else "red",
        ),
        (
            "Chop",
            "Sideways" if sideways else "Trending",
            "orange" if sideways else "green",
        ),
    ]


# (label, score key, max points, theme role, report CSS color)
SCORE_CATS = (
    ("Trend", "trend", 15, "green", "var(--green)"),
    ("Momentum", "momentum", 15, "cyan", "var(--cyan)"),
    ("RSI", "rsi", 8, "blue", "var(--blue)"),
    ("MACD", "macd", 7, "macd", "#2dd4bf"),
    ("Stoch", "stoch", 5, "pink", "#a78bfa"),
    ("OBV", "obv", 5, "lime", "var(--lime)"),
    ("Volume", "volume", 10, "orange", "var(--orange)"),
    ("RelStr", "rel_str", 10, "lime", "var(--lime)"),
    ("Volatility", "volatility", 5, "yellow", "var(--yellow)"),
    ("Fundamental", "fundamentals", 20, "fund", "#ffe600"),
)
