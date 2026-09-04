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


# Column definitions for the results grid: (header, width).
RESULT_COLS = [
    ("#", 35), ("Ticker", 100), ("Score", 50), ("Rating", 78), ("ENTRY", 55),
    ("Price", 80), ("MA", 62), ("T/15", 40), ("M/15", 40), ("R/8", 35),
    ("V/7", 35), ("Vol/10", 42), ("RS/10", 42), ("F/20", 42),
    ("1M", 55), ("Dir", 58), ("ADX", 40), ("Chop", 42),
]


def _score_of(r: dict) -> float:
    """Total score of a result row, tolerant of missing/None values."""
    return r.get("total", 0) or 0
