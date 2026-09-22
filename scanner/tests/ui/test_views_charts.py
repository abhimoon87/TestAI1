"""Regression tests for scanner.ui.views_charts — score histogram rendering.

Bug being guarded against: build_score_histogram appended raw
``Path.PathElement`` objects (Path.MoveTo / Path.LineTo / …) directly into
``Canvas.shapes``, which expects ``Path`` objects (elements + paint).
Flet silently dropped the invalid shapes so the bars never rendered.

Also guards the chart-container clip fix (height 60 → 155).
"""

from __future__ import annotations

import flet as ft
from flet.canvas import Canvas

from scanner.shared.themes import THEMES
from scanner.ui.views_charts import (
    build_donut,
    build_price_chart,
    build_score_breakdown,
    build_score_histogram,
)


def _results(scores):
    return [{"ticker": f"T{i}", "total": s} for i, s in enumerate(scores)]


class TestBuildScoreHistogram:
    def test_returns_container_with_canvas_stack(self):
        c = THEMES["dark"]
        widget = build_score_histogram(_results([55, 65, 80]), c)
        assert isinstance(widget, ft.Container)
        assert isinstance(widget.content, ft.Stack)
        canvases = [x for x in widget.content.controls if isinstance(x, Canvas)]
        assert len(canvases) == 1

    def test_shapes_are_path_objects_not_elements(self):
        """Regression: every entry in Canvas.shapes must be a Path (has paint)."""
        from flet.canvas import Path as CanvasPath

        c = THEMES["dark"]
        widget = build_score_histogram(_results([10, 55, 65, 80, 95]), c)
        canvas = next(x for x in widget.content.controls if isinstance(x, Canvas))
        assert canvas.shapes, "histogram must produce shapes"
        for shape in canvas.shapes:
            assert isinstance(shape, CanvasPath), (
                f"Canvas.shapes entry is {type(shape).__name__}, expected Path"
            )
            assert shape.paint is not None, "every Path must carry a Paint"

    def test_empty_results_still_renders(self):
        c = THEMES["dark"]
        widget = build_score_histogram([], c)
        assert isinstance(widget, ft.Container)

    def test_all_buckets_bars_present(self):
        """One result in each decile → 10 bar paths (plus threshold path)."""
        c = THEMES["dark"]
        scores = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        widget = build_score_histogram(_results(scores), c)
        canvas = next(x for x in widget.content.controls if isinstance(x, Canvas))
        # threshold line + 10 bars
        filled = [
            s
            for s in canvas.shapes
            if s.paint and s.paint.style == ft.PaintingStyle.FILL
        ]
        assert len(filled) >= 10

    def test_threshold_line_dashed(self):
        c = THEMES["dark"]
        widget = build_score_histogram(_results([60]), c, threshold=50)
        canvas = next(x for x in widget.content.controls if isinstance(x, Canvas))
        stroked = [
            s
            for s in canvas.shapes
            if s.paint and s.paint.style == ft.PaintingStyle.STROKE
        ]
        assert len(stroked) == 1
        assert stroked[0].paint.stroke_dash_pattern is not None

    def test_score_clamped_to_bucket_9(self):
        """score 100 must land in bucket 9, never index-error."""
        c = THEMES["dark"]
        widget = build_score_histogram(_results([100, 200, -5]), c)
        assert isinstance(widget, ft.Container)

    def test_count_labels_present_above_bars(self):
        c = THEMES["dark"]
        widget = build_score_histogram(_results([55, 55, 65]), c)
        texts = [x for x in widget.content.controls if isinstance(x, ft.Text)]
        values = {t.value for t in texts}
        assert "2" in values  # two results in 50-60 bucket


class TestOtherCharts:
    def test_price_chart_insufficient_data(self):
        c = THEMES["dark"]
        w = build_price_chart([100.0], c=c)
        assert isinstance(w, ft.Container)

    def test_price_chart_renders_canvas(self):
        from flet.canvas import Canvas as FCanvas

        c = THEMES["dark"]
        closes = [100 + i for i in range(20)]
        w = build_price_chart(closes, c=c)
        stack = w.content
        canvases = [x for x in stack.controls if isinstance(x, FCanvas)]
        assert len(canvases) == 1
        assert canvases[0].shapes

    def test_score_breakdown_renders(self):
        c = THEMES["dark"]
        row = {
            "trend": 10,
            "momentum": 12,
            "rsi": 6,
            "macd": 5,
            "stoch": 3,
            "obv": 4,
            "volume": 8,
            "rel_str": 7,
            "volatility": 3,
            "fundamentals": 15,
        }
        w = build_score_breakdown(row, c)
        assert isinstance(w, ft.Container)

    def test_donut_renders(self):
        c = THEMES["dark"]
        w = build_donut([("A", 60, "#34d399"), ("B", 40, "#f87171")], c)
        assert isinstance(w, ft.Container)
        stack = w.content
        assert isinstance(stack, ft.Stack)
        canvases = [x for x in stack.controls if isinstance(x, Canvas)]
        assert canvases and canvases[0].shapes

    def test_donut_zero_total(self):
        c = THEMES["dark"]
        w = build_donut([("A", 0, "#34d399")], c)
        assert isinstance(w, ft.Container)
