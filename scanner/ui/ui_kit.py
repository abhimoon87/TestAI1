"""Shared Flet UI primitives for the HMAxEMA scanner GUI.

Pure control-construction helpers (no ScannerApp state) that are used by
``scanner.app`` and by the view-builder mixins in ``views_layout``,
``views_results`` and ``views_settings``. The command-palette matching
(``fuzzy_score`` / ``filter_actions`` / ``PaletteAction``) is deliberately
toolkit-free so it stays unit-testable without a page.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

import flet as ft

# ── Animation presets ───────────────────────────────────────────────
ANIM_FAST = ft.Animation(200, ft.AnimationCurve.EASE_OUT)
ANIM_NORMAL = ft.Animation(300, ft.AnimationCurve.EASE_OUT)
ANIM_SLOW = ft.Animation(500, ft.AnimationCurve.EASE_OUT)
ANIM_BOUNCE = ft.Animation(300, ft.AnimationCurve.BOUNCE_OUT)


def shimmer_cell(width: int = 100, height: int = 14) -> ft.Container:
    """A single shimmer skeleton block for loading placeholders."""
    return ft.Container(
        width=width,
        height=height,
        bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.WHITE),
        border_radius=4,
    )


def shimmer_row(num_cells: int = 4, row_height: int = 34) -> ft.Container:
    """One skeleton row with ``num_cells`` shimmer blocks."""
    cells = [shimmer_cell(width=w) for w in (40, 120, 60, 80)[:num_cells]]
    return ft.Container(
        content=ft.Row(
            controls=cells, spacing=12, alignment=ft.MainAxisAlignment.START
        ),
        height=row_height,
        padding=ft.Padding(left=8, right=8, top=0, bottom=0),
    )


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
    return [
        ft.BoxShadow(
            blur_radius=24, color=ft.Colors.with_opacity(0.55, ft.Colors.BLACK)
        )
    ]


def _neon_glow(color: str, blur: int = 12) -> list[ft.BoxShadow]:
    return [ft.BoxShadow(blur_radius=blur, color=color)]


def _padding_only(
    left: int = 0, top: int = 0, right: int = 0, bottom: int = 0
) -> ft.Padding:
    return ft.Padding(left=left, top=top, right=right, bottom=bottom)


def _margin_only(
    left: int = 0, top: int = 0, right: int = 0, bottom: int = 0
) -> ft.Margin:
    return ft.Margin(left=left, top=top, right=right, bottom=bottom)


def _score_of(r: dict) -> float:
    """Total score of a result row, tolerant of missing/None values.

    Backward-compat re-export — canonical definition lives in constants.py.
    """
    from ..shared.constants import score_of as _so

    return _so(r)


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
        padding=padding
        if padding is not None
        else _padding_only(left=14, right=14, top=10, bottom=10),
    )


def themed_dropdown(
    options: list,
    value,
    c: dict,
    width: int = 160,
    height: int = 42,
    on_select=None,
    on_change=None,
) -> ft.Dropdown:
    """Theme-consistent dropdown shared by settings/backtest/layout.

    This Flet version's ``Dropdown`` only accepts ``on_select`` — an
    ``on_change`` handler is mapped onto it so callers keep working.
    """
    handler = on_select if on_select is not None else on_change
    return ft.Dropdown(
        options=[ft.dropdown.Option(v) if isinstance(v, str) else v for v in options],
        value=value,
        width=width,
        height=height,
        text_size=13,
        bgcolor=c["option_bg"],
        color=c["text"],
        border_color=c["border"],
        border_width=1,
        border_radius=10,
        focused_border_color=c["purple"],
        on_select=handler,
    )


@dataclass
class PaletteAction:
    """One command-palette entry: labels plus the zero-arg callable to run."""

    action_id: str
    title: str
    hint: str = ""
    run: Callable[[], None] | None = None


def fuzzy_score(query: str, text: str) -> float | None:
    """Subsequence match score (higher is better), None when no match.

    Rewards: exact hits, prefix matches, word-boundary hits, compact
    (contiguous) matches, and shorter candidates. Case-insensitive.
    """
    q, t = query.strip().lower(), text.lower()
    if not q:
        return 0.0
    if q == t:
        return 1000.0
    pos, total_gap, boundary_hits = 0, 0, 0
    for ch in q:
        nxt = t.find(ch, pos)
        if nxt == -1:
            return None
        if nxt == 0 or t[nxt - 1] in " _-/":
            boundary_hits += 2.0
        total_gap += nxt - pos
        pos = nxt + 1
    compact = max(0.0, 10.0 - total_gap)
    return 100.0 + boundary_hits * 10.0 + compact - len(t) * 0.1


def filter_actions(query: str, actions: list[PaletteAction]) -> list[PaletteAction]:
    """Actions matching ``query`` against title or id, best first (stable)."""
    scored = []
    for i, a in enumerate(actions):
        best: float | None = None
        for cand in (a.title, a.action_id):
            s = fuzzy_score(query, cand)
            if s is not None:
                best = s if best is None else max(best, s)
        if best is not None or not query.strip():
            scored.append((best if best is not None else 0.0, -i, a))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [a for _, _, a in scored]


# Header-ish first cells ("Ticker", "Symbol", ...) are never holdings.
_WATCHLIST_HEADERS = {
    "TICKER",
    "TICKERS",
    "SYMBOL",
    "SYMBOLS",
    "SCRIP",
    "SCRIPS",
    "STOCK",
    "STOCKS",
    "NAME",
    "NAMES",
    "HOLDING",
}
_WATCHLIST_TOKEN = re.compile(r"^[A-Z0-9&.\-]{1,24}$")


def parse_watchlist_text(text: str, limit: int = 1000) -> list[str]:
    """Tickers parsed from pasted/file text for watchlist import.

    Accepts plain ticker lists (one per line) and CSV (first column wins);
    Yahoo-style ``.NS`` / ``.BO`` suffixes are stripped, headers skipped,
    duplicates dropped, order preserved, capped at ``limit``.
    """
    seen: set[str] = set()
    out: list[str] = []
    for line in (text or "").splitlines():
        cell = line.split(",")[0].split(";")[0].strip().upper()
        for suffix in (".NS", ".BO"):
            if cell.endswith(suffix):
                cell = cell[: -len(suffix)]
                break
        cell = cell.strip()
        if not cell or cell in seen or cell in _WATCHLIST_HEADERS:
            continue
        if not _WATCHLIST_TOKEN.match(cell):
            continue
        seen.add(cell)
        out.append(cell)
        if len(out) >= limit:
            break
    return out
