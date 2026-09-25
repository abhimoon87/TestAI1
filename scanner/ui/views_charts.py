"""Canvas-based chart helpers for the HMAxEMA scanner dashboard.

All functions return plain Flet controls (``ft.Container`` wrapping
``flet.canvas.Canvas``) so they can be embedded anywhere in the UI
without introducing external charting libraries.

Charts use the Aurora v3 dark theme tokens passed via the ``c`` dict.
"""

import logging

import flet as ft
from flet.canvas import Canvas, Path
from flet.controls.alignment import Alignment

from ..shared.constants import score_of as _score_of
from .ui_kit import RADIUS_LG, _border_all, _padding_only, score_color

logger = logging.getLogger(__name__)

# ── Score Distribution Histogram ────────────────────────────────────


def build_score_histogram(
    results: list[dict],
    c: dict,
    threshold: float = 50.0,
    width: int = 520,
    height: int = 140,
    on_bucket=None,
) -> ft.Container:
    """Canvas histogram of score distribution (10 buckets: 0-10 … 90-100).

    Features:
    - Coloured bars (red → orange → lime → green)
    - Vertical dashed threshold line at ``threshold``
    - Bucket count labels above each bar
    - X-axis labels (0, 10, 20, … 100)
    - ``on_bucket(i)``: optional tap handler — wraps the chart in a
      GestureDetector and reports the clicked bucket index (0-9)
    """
    buckets = [0] * 10
    for r in results:
        s = _score_of(r)
        idx = min(int(s // 10), 9)
        buckets[idx] += 1

    max_count = max(buckets) if buckets else 1
    if max_count == 0:
        max_count = 1

    pad_left, pad_right = 36, 12
    # Slim strips keep just enough room for the count/axis labels.
    pad_top = 24 if height >= 100 else 18
    pad_bottom = 22 if height >= 100 else 15
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom
    bar_w = chart_w / 10
    gap = 2

    shapes: list = []
    labels: list[ft.Text] = []

    # ── Bars (per-bar coloured fills) ───────────────────────────────
    for i, count in enumerate(buckets):
        x0 = pad_left + i * bar_w + gap / 2
        x1 = pad_left + (i + 1) * bar_w - gap / 2
        bar_h = (count / max_count) * chart_h if count > 0 else 0
        y0 = pad_top + chart_h - bar_h
        y1 = pad_top + chart_h
        score_mid = i * 10 + 5
        color = score_color(score_mid, c)

        if bar_h > 0:
            r = min(3, bar_h / 2)
            shapes.append(
                Path(
                    elements=[
                        Path.MoveTo(x0 + r, y0),
                        Path.LineTo(x1 - r, y0),
                        Path.ArcTo(x1, y0, x1, y0 + r, r),
                        Path.LineTo(x1, y1),
                        Path.LineTo(x0, y1),
                        Path.LineTo(x0, y0 + r),
                        Path.ArcTo(x0, y0, x0 + r, y0, r),
                        Path.Close(),
                    ],
                    paint=ft.Paint(color=color, style=ft.PaintingStyle.FILL),
                )
            )

        # Count label above bar
        if count > 0:
            labels.append(
                ft.Text(
                    str(count),
                    size=9,
                    weight=ft.FontWeight.BOLD,
                    color=color,
                    left=x0 + (x1 - x0) / 2 - 6,
                    top=y0 - 16,
                )
            )

        # X-axis label
        labels.append(
            ft.Text(
                str(i * 10),
                size=8,
                color=c["text_faint"],
                left=x0 + (x1 - x0) / 2 - 6,
                top=pad_top + chart_h + 4,
            )
        )

    # ── Threshold line ──────────────────────────────────────────────
    if 0 < threshold < 100:
        tx = pad_left + (threshold / 100) * chart_w
        shapes.append(
            Path(
                elements=[
                    Path.MoveTo(tx, pad_top),
                    Path.LineTo(tx, pad_top + chart_h),
                ],
                paint=ft.Paint(
                    color=c["cyan"],
                    stroke_width=1.5,
                    style=ft.PaintingStyle.STROKE,
                    stroke_dash_pattern=[4, 3],
                ),
            )
        )
        labels.append(
            ft.Text(
                f"{threshold:.0f}+",
                size=8,
                color=c["cyan"],
                left=tx + 3,
                top=pad_top + 2,
            )
        )

    canvas = Canvas(
        width=width,
        height=height,
        shapes=shapes,
    )

    # Overlay labels on top of the canvas using a Stack
    content = ft.Stack(
        controls=[canvas, *labels],
        width=width,
        height=height,
    )

    if on_bucket is not None:

        def _tap(e):
            pos = getattr(e, "local_position", None)
            x = getattr(pos, "x", None) if pos is not None else None
            if x is None or not (pad_left <= x <= pad_left + chart_w):
                return
            on_bucket(max(0, min(9, int((x - pad_left) / bar_w))))

        content = ft.GestureDetector(
            content=content,
            on_tap_down=_tap,
            tooltip="Click a bar to set min score",
        )

    return ft.Container(
        content=content,
        bgcolor=c["card"],
        border_radius=RADIUS_LG,
        border=_border_all(1, c["border"]),
        padding=_padding_only(left=4, right=4, top=4, bottom=4),
    )


# ── Price Line Chart ────────────────────────────────────────────────


def build_price_chart(
    closes: list[float],
    ma_fast: list[float] | None = None,
    ma_slow: list[float] | None = None,
    volume: list[float] | None = None,
    c: dict | None = None,
    width: int = 600,
    height: int = 200,
    title: str = "",
) -> ft.Container:
    """Canvas line chart with optional MA overlays and volume bars.

    ``closes``: list of closing prices (most recent last).
    ``ma_fast`` / ``ma_slow``: optional MA overlay lines.
    ``volume``: optional volume bars at the bottom (20% of height).
    """
    if c is None:
        c = {}
    if not closes or len(closes) < 2:
        return ft.Container(
            content=ft.Text(
                "Insufficient price data", size=12, color=c.get("text_dim", "#8c92b0")
            ),
            bgcolor=c.get("card", "#1a1b24"),
            border_radius=RADIUS_LG,
            width=width,
            height=height,
            alignment=Alignment.CENTER,
        )

    pad_l, pad_r, pad_t, pad_b = 8, 8, 24, 20
    chart_w = width - pad_l - pad_r
    vol_h = int(height * 0.18) if volume else 0
    chart_h = height - pad_t - pad_b - vol_h

    n = len(closes)
    lo, hi = min(closes), max(closes)
    span = (hi - lo) or 1.0

    def _x(i: int) -> float:
        return pad_l + (i / max(n - 1, 1)) * chart_w

    def _y(v: float) -> float:
        return pad_t + chart_h - ((v - lo) / span) * chart_h

    shapes: list = []
    labels: list[ft.Text] = []

    # ── Close line ──────────────────────────────────────────────────
    pts = [(_x(i), _y(v)) for i, v in enumerate(closes)]
    line_elements: list[Path.PathElement] = [Path.MoveTo(pts[0][0], pts[0][1])]
    for x, y in pts[1:]:
        line_elements.append(Path.LineTo(x, y))
    shapes.append(
        Path(
            elements=line_elements,
            paint=ft.Paint(
                color=c.get("cyan", "#22d3ee"),
                stroke_width=1.5,
                style=ft.PaintingStyle.STROKE,
                stroke_cap=ft.StrokeCap.ROUND,
            ),
        )
    )

    # Fill area under close line
    fill_elements = list(line_elements) + [
        Path.LineTo(pts[-1][0], pad_t + chart_h),
        Path.LineTo(pts[0][0], pad_t + chart_h),
    ]
    shapes.append(
        Path(
            elements=fill_elements,
            paint=ft.Paint(
                color=ft.Colors.with_opacity(0.08, c.get("cyan", "#22d3ee")),
                style=ft.PaintingStyle.FILL,
            ),
        )
    )

    # ── MA overlays ─────────────────────────────────────────────────
    def _draw_ma(ma: list[float], color: str, label: str):
        if not ma or len(ma) < 2:
            return
        offset = n - len(ma)
        ma_pts = [(_x(offset + i), _y(v)) for i, v in enumerate(ma) if v is not None]
        if len(ma_pts) < 2:
            return
        elems: list[Path.PathElement] = [Path.MoveTo(ma_pts[0][0], ma_pts[0][1])]
        for x, y in ma_pts[1:]:
            elems.append(Path.LineTo(x, y))
        shapes.append(
            Path(
                elements=elems,
                paint=ft.Paint(
                    color=color,
                    stroke_width=1,
                    style=ft.PaintingStyle.STROKE,
                    stroke_dash_pattern=[3, 2],
                ),
            )
        )
        # Legend dot
        labels.append(
            ft.Container(
                content=ft.Row(
                    [
                        ft.Container(width=8, height=3, bgcolor=color, border_radius=2),
                        ft.Text(label, size=8, color=color),
                    ],
                    spacing=3,
                ),
                left=pad_l + len(labels) * 70,
                top=4,
            )
        )

    _draw_ma(ma_fast, c.get("green", "#34d399"), "Fast MA")
    _draw_ma(ma_slow, c.get("pink", "#a78bfa"), "Slow MA")

    # ── Volume bars ─────────────────────────────────────────────────
    if volume and vol_h > 0:
        max_vol = max(volume) if volume else 1
        if max_vol == 0:
            max_vol = 1
        vol_y_base = pad_t + chart_h + vol_h
        vol_elements: list[Path.PathElement] = []
        for i, v in enumerate(volume):
            bx = _x(i)
            bh = (v / max_vol) * (vol_h - 4)
            by = vol_y_base - bh
            vol_elements.extend(
                [
                    Path.MoveTo(bx - 1, by),
                    Path.LineTo(bx + 1, by),
                    Path.LineTo(bx + 1, vol_y_base),
                    Path.LineTo(bx - 1, vol_y_base),
                ]
            )
        if vol_elements:
            # Colour volume bars: green if close > prev close, else red
            shapes.append(
                Path(
                    elements=vol_elements,
                    paint=ft.Paint(
                        color=ft.Colors.with_opacity(
                            0.35, c.get("text_faint", "#585d78")
                        ),
                        style=ft.PaintingStyle.FILL,
                    ),
                )
            )

    # ── Y-axis labels ───────────────────────────────────────────────
    for i in range(5):
        val = lo + span * i / 4
        y = _y(val)
        labels.append(
            ft.Text(
                f"{val:,.0f}",
                size=8,
                color=c.get("text_faint", "#585d78"),
                left=0,
                top=y - 6,
            )
        )

    canvas = Canvas(width=width, height=height, shapes=shapes)
    content = ft.Stack(controls=[canvas, *labels], width=width, height=height)

    result = ft.Container(
        content=content,
        bgcolor=c.get("card", "#1a1b24"),
        border_radius=RADIUS_LG,
        border=_border_all(1, c.get("border", "#2a2b38")),
        padding=_padding_only(left=2, right=2, top=2, bottom=2),
    )
    if title:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        title,
                        size=12,
                        weight=ft.FontWeight.BOLD,
                        color=c.get("text", "#e8eaf2"),
                    ),
                    result,
                ],
                spacing=4,
            ),
        )
    return result


# ── Score Breakdown Bar Chart ───────────────────────────────────────


def build_score_breakdown(
    row: dict, c: dict, width: int = 300, height: int = 200
) -> ft.Container:
    """Horizontal bar chart showing the 10 score categories.

    Each bar is proportionally sized to the category's max score.
    """
    categories = [
        ("Trend", row.get("trend", 0) or 0, 15, c.get("green", "#34d399")),
        ("Momentum", row.get("momentum", 0) or 0, 15, c.get("cyan", "#22d3ee")),
        ("RSI", row.get("rsi", 0) or 0, 8, c.get("blue", "#60a5fa")),
        ("MACD", row.get("macd", 0) or 0, 7, c.get("macd", "#aa88ff")),
        ("Stoch", row.get("stoch", 0) or 0, 5, c.get("pink", "#a78bfa")),
        ("OBV", row.get("obv", 0) or 0, 5, c.get("lime", "#a3e635")),
        ("Volume", row.get("volume", 0) or 0, 10, c.get("orange", "#fb923c")),
        ("RelStr", row.get("rel_str", 0) or 0, 10, c.get("lime", "#a3e635")),
        ("Volatility", row.get("volatility", 0) or 0, 5, c.get("yellow", "#facc15")),
        ("Fundamental", row.get("fundamentals", 0) or 0, 20, c.get("fund", "#ffe600")),
    ]

    bar_h = 14
    gap = 4
    pad_left = 72
    pad_right = 40
    bar_area_w = width - pad_left - pad_right

    shapes: list = []
    labels: list[ft.Text] = []

    for i, (name, score, max_score, color) in enumerate(categories):
        y = i * (bar_h + gap)
        ratio = min(score / max_score, 1.0) if max_score > 0 else 0
        bw = ratio * bar_area_w

        # Background track
        shapes.append(
            Path(
                elements=[
                    Path.MoveTo(pad_left, y),
                    Path.LineTo(pad_left + bar_area_w, y),
                    Path.LineTo(pad_left + bar_area_w, y + bar_h),
                    Path.LineTo(pad_left, y + bar_h),
                ],
                paint=ft.Paint(
                    color=ft.Colors.with_opacity(0.1, color),
                    style=ft.PaintingStyle.FILL,
                ),
            )
        )

        # Filled bar
        if bw > 0:
            shapes.append(
                Path(
                    elements=[
                        Path.MoveTo(pad_left, y),
                        Path.LineTo(pad_left + bw, y),
                        Path.LineTo(pad_left + bw, y + bar_h),
                        Path.LineTo(pad_left, y + bar_h),
                    ],
                    paint=ft.Paint(
                        color=ft.Colors.with_opacity(0.7, color),
                        style=ft.PaintingStyle.FILL,
                    ),
                )
            )

        # Category label (left)
        labels.append(
            ft.Text(
                name,
                size=9,
                color=c.get("text_dim", "#8c92b0"),
                left=0,
                top=y + 2,
            )
        )

        # Score value (right of bar)
        labels.append(
            ft.Text(
                f"{score:.0f}/{max_score}",
                size=9,
                weight=ft.FontWeight.BOLD,
                color=color,
                left=pad_left + bar_area_w + 6,
                top=y + 2,
            )
        )

    canvas = Canvas(width=width, height=height, shapes=shapes)
    content = ft.Stack(controls=[canvas, *labels], width=width, height=height)

    return ft.Container(
        content=content,
        bgcolor=c.get("card", "#1a1b24"),
        border_radius=RADIUS_LG,
        border=_border_all(1, c.get("border", "#2a2b38")),
        padding=_padding_only(left=4, right=4, top=4, bottom=4),
    )
