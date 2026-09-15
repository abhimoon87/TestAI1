"""Shared Flet UI primitives for the HMAxEMA scanner GUI.

Pure control-construction helpers (no ScannerApp state) that are used by
``scanner.app`` and by the view-builder mixins in ``views_layout``,
``views_results`` and ``views_settings``.
"""


import flet as ft


def _border_all(width: float, color: str) -> ft.Border:
    """Return a uniform ``ft.Border`` with ``ft.BorderSide`` on all sides."""
    side = ft.BorderSide(width=width, color=color)
    return ft.Border(top=side, bottom=side, left=side, right=side)


def _glass_bg() -> str:
    """Translucent white fill for frosted-glass cards."""
    return ft.Colors.with_opacity(0.065, ft.Colors.WHITE)


def _glass_border() -> ft.Border:
    side = ft.BorderSide(width=1, color=ft.Colors.with_opacity(0.12, ft.Colors.WHITE))
    return ft.Border(top=side, bottom=side, left=side, right=side)


def _card_shadow() -> list[ft.BoxShadow]:
    return [ft.BoxShadow(blur_radius=24, color=ft.Colors.with_opacity(0.55, ft.Colors.BLACK))]


def _neon_glow(color: str, blur: int = 12) -> list[ft.BoxShadow]:
    return [ft.BoxShadow(blur_radius=blur, color=color)]


def _padding_only(left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> ft.Padding:
    return ft.Padding(left=left, top=top, right=right, bottom=bottom)


def _margin_only(left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> ft.Margin:
    return ft.Margin(left=left, top=top, right=right, bottom=bottom)


# Column definitions for the results grid: (header, width). The trailing
# "1M" column draws a mini sparkline of the last ~20 closes (see px_tail on
# each result row); "1M%" stays the numeric one-month change.
RESULT_COLS = [
    "#", "Ticker", "Score", "Rating", "ENTRY",
    "Price", "MA", "T/15", "M/15", "R/8",
    "V/7", "Vol/10", "RS/10", "F/20",
    "1M%", "Dir", "ADX", "Chop", "1M",
]


def _score_of(r: dict) -> float:
    """Total score of a result row, tolerant of missing/None values."""
    return r.get("total", 0) or 0


def score_color(score: float, c: dict) -> str:
    """Tier color for a numeric score (single source of truth)."""
    try:
        s = float(score or 0)
    except (TypeError, ValueError):
        s = 0.0
    if s >= 70:
        return c["green"]
    if s >= 50:
        return c["lime"]
    if s >= 30:
        return c["orange"]
    return c["red"]


def rating_color(rating: str, c: dict) -> str:
    """Text color for a combined rating (single source of truth)."""
    return {
        "EXCELLENT": c["green"],
        "GOOD": c["lime"],
        "MODERATE": c["orange"],
    }.get((rating or "POOR").upper(), c["red"])


def glass_card(content, c: dict, padding=None) -> ft.Container:
    """Frosted-glass card container shared by all views."""
    return ft.Container(
        content=content,
        bgcolor=_glass_bg(),
        border=_glass_border(),
        border_radius=14,
        shadow=_card_shadow(),
        padding=padding if padding is not None else _padding_only(
            left=14, right=14, top=10, bottom=10),
    )


def themed_dropdown(options: list, value, c: dict, width: int = 160,
                    height: int = 42, on_select=None, on_change=None) -> ft.Dropdown:
    """Theme-consistent dropdown shared by settings/backtest/layout.

    This Flet version's ``Dropdown`` only accepts ``on_select`` — an
    ``on_change`` handler is mapped onto it so callers keep working.
    """
    handler = on_select if on_select is not None else on_change
    return ft.Dropdown(
        options=[ft.dropdown.Option(v) if isinstance(v, str) else v
                 for v in options],
        value=value,
        width=width, height=height, text_size=13,
        bgcolor=c["option_bg"], color=c["text"],
        border_color=c["border"], border_width=1, border_radius=10,
        focused_border_color=c["purple"],
        on_select=handler,
    )
