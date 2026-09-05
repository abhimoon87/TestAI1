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


def _glass_bg() -> ft.Paint:
    """Translucent white fill for frosted-glass cards."""
    return ft.Colors.with_opacity(0.045, ft.Colors.WHITE)


def _glass_border() -> ft.Border:
    side = ft.BorderSide(width=1, color=ft.Colors.with_opacity(0.09, ft.Colors.WHITE))
    return ft.Border(top=side, bottom=side, left=side, right=side)


def _card_shadow() -> list[ft.BoxShadow]:
    return [ft.BoxShadow(blur_radius=18, color=ft.Colors.with_opacity(0.45, ft.Colors.BLACK))]


def _neon_glow(color: str, blur: int = 10) -> list[ft.BoxShadow]:
    return [ft.BoxShadow(blur_radius=blur, color=color)]


def _padding_only(left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> ft.Padding:
    return ft.Padding(left=left, top=top, right=right, bottom=bottom)


def _margin_only(left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> ft.Margin:
    return ft.Margin(left=left, top=top, right=right, bottom=bottom)


# Column definitions for the results grid: (header, width). The trailing
# "1M" column draws a mini sparkline of the last ~20 closes (see px_tail on
# each result row); "1M%" stays the numeric one-month change.
RESULT_COLS = [
    ("#", 30), ("Ticker", 92), ("Score", 46), ("Rating", 66), ("ENTRY", 50),
    ("Price", 72), ("MA", 58), ("T/15", 40), ("M/15", 40), ("R/8", 34),
    ("V/7", 34), ("Vol/10", 40), ("RS/10", 40), ("F/20", 40),
    ("1M%", 50), ("Dir", 50), ("ADX", 38), ("Chop", 40),
    ("1M", 56),
]


def _score_of(r: dict) -> float:
    """Total score of a result row, tolerant of missing/None values."""
    return r.get("total", 0) or 0
