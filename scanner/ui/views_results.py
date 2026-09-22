"""Results rendering — ``ResultsViewMixin`` for ``scanner.app.ScannerApp``.

Builds and refreshes everything that lives in the main results area after a
scan: the paginated table (header + data rows), row-level news expansion,
the score chart, summary cards and the hero status line.

The methods are mixins: they run against the ``ScannerApp`` instance (via
the MRO) and rely on state such as ``self.all_results`` / ``self.theme_colors``
and on the controls created by ``LayoutViewMixin._build_main_area``
(``self.table_column``, ``self.chart_bars``, ...).
"""

import asyncio
import logging
import threading
import time

import flet as ft
from flet.canvas import Canvas, Path
from flet.controls.alignment import Alignment

logger = logging.getLogger(__name__)
from ..shared.constants import RESULT_COLS
from ..shared.constants import score_of as _score_of
from .ui_kit import (
    ANIM_FAST,
    ANIM_NORMAL,
    _border_all,
    _margin_only,
    _padding_only,
    rating_color,
    score_color,
    shimmer_row,
)

# Rating → theme-color-key accent used for row rails / washes.
_RATING_ACCENT = {
    "EXCELLENT": "green",
    "GOOD": "lime",
    "MODERATE": "orange",
    "POOR": "red",
}


def _spark_move(r: dict) -> float:
    """Net % move across the row's sparkline closes (for column sorting)."""
    px = r.get("px_tail") or []
    if len(px) < 2 or not px[0]:
        return 0.0
    return (px[-1] - px[0]) / px[0] * 100.0


class ResultsViewMixin:
    # ── Row control pool (performance optimisation) ─────────────────
    # Maps ticker → ft.Container (the outermost row wrapper).  When a
    # row already exists in the pool we only patch the ft.Text.value
    # properties inside it, avoiding a full control-tree rebuild.
    _row_pool: dict[str, ft.Container]
    # Maps ticker → list[ft.Text] (the 19 text cells in column order).
    _row_cells: dict[str, list[ft.Text]]

    def _display_results(self, results):
        with self._results_lock:
            self.all_results = list(results)
            self.filtered_results = [r for r in results if self._row_matches_filters(r)]
        self.current_page = 0
        self._render_current_page()
        self.page.update()

    def _render_current_page(self):
        if self.active_view != "dashboard":
            logger.debug("_render_current_page: SKIP active_view=%s", self.active_view)
            return
        c = self.theme_colors
        with self._results_lock:
            results = list(self.all_results)
            shown = list(self._visible_results())
        logger.info(
            "_render_current_page: results=%d shown=%d scanning=%s active_view=%s table_ctrls=%d",
            len(results),
            len(shown),
            self.scanning,
            self.active_view,
            len(self.table_column.controls),
        )

        if self.sort_col is not None and shown:
            try:
                key_fn = self._get_sort_key(self.sort_col)
                shown = sorted(shown, key=key_fn, reverse=self.sort_reverse)
            except Exception as ex:
                logger.info("Sort failed for col %s: %s", self.sort_col, ex)

        # ── Streaming fast path: patch rows in-place, no full rebuild ──
        pool = getattr(self, "_row_pool", None)
        cells = getattr(self, "_row_cells", None)
        is_stream = (
            pool is not None
            and cells is not None
            and pool  # non-empty pool required (empty dict after clear → full rebuild)
            and self.sort_col is None
            and not self._is_filter_active()
            and self.table_column.controls  # already has rows from a prior render
        )
        logger.info(
            "_render_current_page: pool=%s cells=%s is_stream=%s sort=%s filter_active=%s controls=%d",
            len(pool) if pool is not None else None,
            len(cells) if cells is not None else None,
            is_stream,
            self.sort_col,
            self._is_filter_active(),
            len(self.table_column.controls),
        )

        if is_stream and shown:
            threshold = self.settings.get("min_score", 50)
            page_size = self.page_size
            total_pages = max(1, (len(shown) + page_size - 1) // page_size)
            self.current_page = max(0, min(self.current_page, total_pages - 1))
            start = self.current_page * page_size
            page_shown = shown[start : start + page_size]

            # Remove scan placeholder if present (it blocks row display)
            if self.table_column.controls:
                first = self.table_column.controls[0]
                if isinstance(first, ft.Container) and not getattr(
                    first, "_pool_ticker", None
                ):
                    # Check if it's the scan placeholder (has spinner inside)
                    content = getattr(first, "content", None)
                    if isinstance(content, ft.Column):
                        self.table_column.controls.pop(0)

            # Build a set of currently displayed tickers
            displayed = set()
            for ctrl in self.table_column.controls:
                if isinstance(ctrl, ft.Container):
                    t = getattr(ctrl, "_pool_ticker", None)
                    if t:
                        displayed.add(t)

            # Update existing rows and append new ones
            for rank, r in enumerate(page_shown, start + 1):
                ticker = r.get("ticker", "?")
                if ticker in pool:
                    self._update_row(ticker, rank, c, threshold, len(shown))
                else:
                    score = _score_of(r)
                    is_above = score >= threshold
                    row_bg = (
                        c["card"]
                        if is_above
                        else (c["row_alt"] if rank % 2 else c["main_bg"])
                    )
                    row = self._create_row_controls(r, rank, c, row_bg, threshold)
                    row.opacity = 1  # no fade-in for streaming appends
                    self.table_column.controls.append(row)

            self.pagination_bar.visible = bool(shown and len(shown) > page_size)
            self.page_label.value = (
                f"Page {self.current_page + 1} / {total_pages}  ({len(shown)} stocks)"
            )

            # Update summary/hero/chart even during streaming (same as
            # the full rebuild path below — the table update is the only
            # part we skip).
            if results:
                threshold = self.settings.get("min_score", 50)
                filter_parts = []
                if self.filter_text:
                    filter_parts.append(f"'{self.filter_text}'")
                rating = self._rating_filter()
                if rating != "ALL":
                    filter_parts.append(f"rating {rating.title()}")
                suffix = (
                    f"  |  filter: {', '.join(filter_parts)} ({len(shown)})"
                    if filter_parts
                    else ""
                )
                self.result_count_label.value = f"{len(results)} scanned  |  {len([r for r in results if _score_of(r) >= threshold])} above {threshold:.0f}+{suffix}"
            else:
                self.result_count_label.value = "no scan yet"
            self._update_summary(results)
            self._update_hero_status(results)
            self._render_topicks(shown[:5])
            self._render_chart(results)
            logger.info(
                "_render_current_page STREAM DONE: table_ctrls=%d, shown=%d, results=%d",
                len(self.table_column.controls),
                len(shown),
                len(results),
            )
            return  # skip full rebuild below

        # ── Full rebuild path ─────────────────────────────────────────
        if getattr(self, "_row_pool", None):
            self._row_pool.clear()
        if getattr(self, "_row_cells", None):
            self._row_cells.clear()
        self.table_column.controls.clear()
        self.pagination_bar.visible = False

        if not shown:
            if self.scanning and not results:
                live = (getattr(self.progress_label, "value", "") or "").strip()
                headline = (
                    live
                    if live and live not in ("Ready", "Done", "Stopped")
                    else "Scanning — fetching batches…"
                )
                self.table_column.controls.append(self._make_scan_placeholder(headline))
            else:
                has_active_filter = self._is_filter_active()
                filtered_empty = bool(results) and has_active_filter
                msg = (
                    "No results match your filter."
                    if filtered_empty
                    else "No results found."
                )
                empty_body = [
                    ft.Icon(
                        ft.Icons.FILTER_ALT_OFF
                        if filtered_empty
                        else ft.Icons.SEARCH_OFF,
                        size=34,
                        color=c["text_faint"],
                    ),
                    ft.Text(
                        msg,
                        size=13,
                        color=c["text_dim"] if filtered_empty else c["red"],
                    ),
                    ft.Text(
                        "Loosen the search/rating filter to see the hidden rows."
                        if filtered_empty
                        else "Try a lower min-score threshold or a different universe.",
                        size=11,
                        color=c["text_faint"],
                    ),
                ]
                if filtered_empty:
                    empty_body.append(
                        ft.TextButton(
                            content=ft.Text("Reset filters", size=12),
                            on_click=self._reset_filters,
                            style=ft.ButtonStyle(color=c["cyan"]),
                        )
                    )
                self.table_column.controls.append(
                    ft.Container(
                        content=ft.Column(
                            empty_body,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            spacing=4,
                        ),
                        alignment=Alignment.CENTER,
                        padding=30,
                    )
                )
        else:
            threshold = self.settings.get("min_score", 50)
            page_size = self.page_size
            total_pages = max(1, (len(shown) + page_size - 1) // page_size)
            self.current_page = max(0, min(self.current_page, total_pages - 1))
            start = self.current_page * page_size
            page_shown = shown[start : start + page_size]

            header_row = self._make_header_row(c)
            self.table_column.controls.append(header_row)

            for rank, r in enumerate(page_shown, start + 1):
                try:
                    score = _score_of(r)
                    is_above = score >= threshold
                    row_bg = (
                        c["card"]
                        if is_above
                        else (c["row_alt"] if rank % 2 else c["main_bg"])
                    )
                    row = self._create_row_controls(r, rank, c, row_bg, threshold)
                    self.table_column.controls.append(row)
                except Exception:
                    logger.info(
                        "Full rebuild: _create_row_controls FAILED rank=%d ticker=%s",
                        rank,
                        r.get("ticker", "?"),
                        exc_info=True,
                    )

            self._animate_rows_in()

            self.pagination_bar.visible = bool(shown and len(shown) > page_size)
            self.page_label.value = (
                f"Page {self.current_page + 1} / {total_pages}  ({len(shown)} stocks)"
            )

        if results:
            threshold = self.settings.get("min_score", 50)
            filter_parts = []
            if self.filter_text:
                filter_parts.append(f"'{self.filter_text}'")
            rating = self._rating_filter()
            if rating != "ALL":
                filter_parts.append(f"rating {rating.title()}")
            suffix = (
                f"  |  filter: {', '.join(filter_parts)} ({len(shown)})"
                if filter_parts
                else ""
            )
            self.result_count_label.value = f"{len(results)} scanned  |  {len([r for r in results if _score_of(r) >= threshold])} above {threshold:.0f}+{suffix}"
        else:
            self.result_count_label.value = "no scan yet"

        self._update_summary(results)
        self._update_hero_status(results)
        self._render_topicks(shown[:5])
        self._render_chart(results)
        logger.info(
            "_render_current_page DONE: table_ctrls=%d, shown=%d, results=%d",
            len(self.table_column.controls),
            len(shown),
            len(results),
        )

    def _animate_rows_in(self):
        """Trigger fade-in on newly added data rows.

        Rows are built with ``opacity=0`` + ``animate_opacity=ANIM_FAST``.
        The caller's ``page.update()`` pushes them to the client at opacity 0,
        then this method sets ``opacity=1`` so the client renders the fade-in
        transition.

        Pooled rows that were already visible (streaming updates) keep
        ``opacity=1`` — only truly new rows animate.
        """
        try:
            motion = getattr(self, "_motion_reduced", None)
            if motion is not None and motion():
                return
            pool = getattr(self, "_row_pool", None) or {}
            rows = [
                ctrl
                for ctrl in self.table_column.controls
                if isinstance(ctrl, ft.Container)
                and hasattr(ctrl, "opacity")
                and ctrl.opacity == 0
            ]
            logger.info(
                "_animate_rows_in: found %d rows with opacity=0, pool=%d, total_ctrls=%d",
                len(rows),
                len(pool),
                len(self.table_column.controls),
            )
            if not rows:
                return
            for row in rows:
                # Pooled rows already visible stay at opacity 1
                ticker = getattr(row, "_pool_ticker", None)
                if ticker and ticker in pool:
                    row.opacity = 1
                else:
                    row.opacity = 1
        except Exception:
            logger.info("Row stagger animation failed", exc_info=True)

    def _make_header_row(self, c):
        headers = []
        for idx, text in enumerate(RESULT_COLS):
            is_sorted = self.sort_col == idx
            arrow = (
                " ▲"
                if is_sorted and not self.sort_reverse
                else (" ▼" if is_sorted else "")
            )
            color = c["cyan"] if not is_sorted else c["green"]
            headers.append(
                ft.Container(
                    content=ft.Text(
                        f"{text}{arrow}",
                        size=10,
                        weight=ft.FontWeight.BOLD,
                        color=color,
                        text_align=ft.TextAlign.LEFT if idx == 1 else None,
                    ),
                    expand=2 if idx == 1 else True,
                    on_click=lambda e, i=idx: self._on_sort(i),
                    ink=True,
                )
            )
        return ft.Container(
            content=ft.Row(controls=headers, spacing=2),
            bgcolor=c["card2"],
            border_radius=10,
            border=_border_all(1, c["border"]),
            height=34,
            padding=_padding_only(left=6, right=6, top=4, bottom=4),
            margin=_margin_only(bottom=6),
        )

    def _rating_accent(self, rating) -> str:
        return self.theme_colors.get(
            _RATING_ACCENT.get(rating, "red"), self.theme_colors["red"]
        )

    def _sentiment_badge(self, r: dict, c) -> ft.Container | None:
        """Compact news count pill (arrow + n) colored by sentiment score.

        Shown next to the ticker when the enrichment pass attached an
        ``_article_count`` / ``_sentiment_score``; None when there is no
        sentiment data (the common keyless case).
        """
        try:
            n = int(r.get("_article_count") or 0)
        except (TypeError, ValueError):
            n = 0
        if n <= 0:
            return None
        try:
            s = float(r.get("_sentiment_score") or 0.0)
        except (TypeError, ValueError):
            s = 0.0
        if s >= 0.05:
            arrow, fg, bg = "↑", c["green"], c["chip_good"]
            tone = "positive"
        elif s <= -0.05:
            arrow, fg, bg = "↓", c["red"], c["chip_bad"]
            tone = "negative"
        else:
            arrow, fg, bg = "·", c["text_dim"], c["chip_neutral"]
            tone = "neutral"
        return ft.Container(
            content=ft.Row(
                [
                    ft.Text(arrow, size=8, color=fg, weight=ft.FontWeight.BOLD),
                    ft.Text(str(n), size=8, weight=ft.FontWeight.BOLD, color=fg),
                ],
                spacing=1,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=bg,
            border_radius=5,
            padding=_padding_only(left=4, right=4, top=1, bottom=1),
            tooltip=f"News sentiment: {tone} · {n} articles — click ticker to read",
        )

    def _make_sparkline(self, px: list, ticker: str, c: dict) -> ft.Container:
        """Canvas-drawn smooth sparkline curve from price data."""
        up = px[-1] >= px[0]
        spark_color = c["green"] if up else c["red"]
        move = (px[-1] - px[0]) / px[0] * 100.0 if px[0] else 0.0
        W, H = 120, 24
        lo, hi = min(px), max(px)
        span = (hi - lo) or 1.0
        n = len(px)
        coords = []
        for i, v in enumerate(px):
            x = i / max(n - 1, 1) * W
            y = H - 2 - (v - lo) / span * (H - 4)
            coords.append((x, y))
        # Build smooth cubic bezier elements
        elements: list[Path.PathElement] = [
            Path.MoveTo(coords[0][0], coords[0][1]),
        ]
        for i in range(1, len(coords)):
            px0, py0 = coords[i - 1]
            px1, py1 = coords[i]
            mx = (px0 + px1) / 2
            elements.append(Path.CubicTo(mx, py0, mx, py1, px1, py1))
        # Line art stroke
        line_path = Path(
            elements=list(elements),
            paint=ft.Paint(
                color=spark_color,
                stroke_width=1.5,
                style=ft.PaintingStyle.STROKE,
                stroke_cap=ft.StrokeCap.ROUND,
            ),
        )
        # Fill area under the curve
        fill_elements = list(elements) + [
            Path.LineTo(W, H),
            Path.LineTo(0, H),
        ]
        fill_color = ft.Colors.with_opacity(0.15, spark_color)
        fill_path = Path(
            elements=fill_elements,
            paint=ft.Paint(
                color=fill_color,
                style=ft.PaintingStyle.FILL,
            ),
        )
        canvas = Canvas(
            width=W,
            height=H,
            shapes=[fill_path, line_path],
        )
        return ft.Container(
            content=canvas,
            expand=True,
            tooltip=f"{ticker}: {move:+.1f}% over the last {len(px)} closes",
        )

    def _make_data_row(self, r, rank, c, bg, threshold):
        total = _score_of(r)
        ticker = r.get("ticker", "?")
        trend_dir = r.get("trend_dir") or ""
        is_above = total >= threshold
        rating = r.get("combined_rating", "POOR")
        rating_txt_color = rating_color(rating, c)
        entry = bool(r.get("entry_signal"))
        accent = self._rating_accent(rating)
        cols = [
            (str(rank), c["text_dim"], 11, False),
            (ticker, c["green"] if is_above else c["text"], 12, True),
            (f"{total:.0f}", score_color(total, c), 14, True),
            (rating, rating_txt_color, 11, True),
            (
                "YES" if entry else "--",
                c["green"] if entry else c["text_dim"],
                11,
                True,
            ),
            (f"₹{r.get('close', 0) or 0:.0f}", c["text"], 12, True),
            (self._ma_text(r), self._ma_color(r), 11, False),
            (f"{r.get('trend', 0) or 0:.0f}", c["green"], 11, False),
            (f"{r.get('momentum', 0) or 0:.0f}", c["cyan"], 11, False),
            (f"{r.get('rsi', 0) or 0:.0f}", c["blue"], 11, False),
            (f"{r.get('macd', 0) or 0:.0f}", c.get("macd", c["blue"]), 11, False),
            (f"{r.get('volume', 0) or 0:.0f}", c["orange"], 11, False),
            (f"{r.get('rel_str', 0) or 0:.0f}", c["lime"], 11, False),
            (
                f"{r.get('fundamentals', 0) or 0:.0f}",
                c.get("fund", c["yellow"]),
                11,
                False,
            ),
            (
                f"{r.get('pc1m', 0) or 0:+.1f}%",
                c["green"] if (r.get("pc1m", 0) or 0) > 0 else c["red"],
                11,
                False,
            ),
            (
                ("^ " if trend_dir == "Bull" else "v ") + (trend_dir or "?"),
                c["green"] if trend_dir == "Bull" else c["red"],
                11,
                False,
            ),
            (f"{r.get('adx_val', 0) or 0:.0f}", c["text"], 11, False),
            (
                "Chop" if r.get("is_sideways") else "OK",
                c["orange"] if r.get("is_sideways") else c["green"],
                11,
                False,
            ),
        ]

        # Subtle chip washes for the rating and ENTRY columns — the text color
        # (asserted by tests) is untouched, only the cell background differs.
        wash_for = {}
        if entry:
            wash_for[4] = ft.Colors.with_opacity(0.14, c["green"])
        wash_for[3] = {
            "EXCELLENT": ft.Colors.with_opacity(0.12, c["green"]),
            "GOOD": ft.Colors.with_opacity(0.09, c["lime"]),
            "MODERATE": ft.Colors.with_opacity(0.12, c["orange"]),
            "POOR": ft.Colors.with_opacity(0.12, c["red"]),
        }.get(rating)

        controls = []
        ticker_cell = None
        for idx, ((text, color, size, bold), _col_name) in enumerate(
            zip(cols, RESULT_COLS)
        ):
            w = ft.Text(
                text,
                size=size,
                weight=ft.FontWeight.BOLD if bold else ft.FontWeight.NORMAL,
                color=color,
                max_lines=1,
                overflow=ft.TextOverflow.CLIP,
            )
            cell = ft.Container(content=w, expand=2 if idx == 1 else True)
            if idx == 0:
                # Rank cell carries a rating-colored accent rail so rows read
                # as green/lime/orange/red bands at a glance.
                cell.content = ft.Row(
                    controls=[
                        ft.Container(
                            width=3, height=18, border_radius=2, bgcolor=accent
                        ),
                        w,
                    ],
                    spacing=5,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                )
                cell.tooltip = rating.title()
            if idx in wash_for and wash_for[idx] is not None:
                cell.bgcolor = wash_for[idx]
                cell.border_radius = 6
            if idx == 1:
                ticker_cell = cell
                w.text_align = ft.TextAlign.LEFT
                w.max_lines = 1
                w.overflow = ft.TextOverflow.CLIP
                w.no_wrap = True
                badge = self._sentiment_badge(r, c)
                if badge is not None:
                    # Keep the ticker text as the row's first control so the
                    # news-expansion scanners can still find it by value; the
                    # text shrinks to make room for the badge.
                    w.expand = True
                    cell.content = ft.Row(
                        controls=[w, badge],
                        spacing=5,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    )
            if idx == 2:
                pass  # score column: just the value text, no bar
            controls.append(cell)

        # Trailing column: mini 1-month sparkline from the closes the engine
        # attached to each row (``px_tail``); dim dash when unavailable.
        px = [
            v
            for v in (r.get("px_tail") or [])
            if v is not None and isinstance(v, (int, float))
        ]
        if len(px) >= 2:
            spark_cell = self._make_sparkline(px, ticker, c)
        else:
            spark_cell = ft.Container(
                content=ft.Text(
                    "—", size=10, color=c["text_faint"], text_align=ft.TextAlign.CENTER
                ),
                expand=True,
            )
        controls.append(spark_cell)

        ticker_cell.on_click = lambda e, t=ticker: self._toggle_stock_news(t)
        ticker_cell.tooltip = "Click for news & sentiment"

        row = ft.Container(
            content=ft.Row(
                controls=controls,
                spacing=2,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=bg,
            border_radius=8,
            border=_border_all(1, c["border"]) if is_above else None,
            height=34,
            padding=_padding_only(left=6, right=6),
            margin=_margin_only(bottom=1),
            opacity=0,
            animate_opacity=ANIM_FAST,
        )
        row.on_hover = lambda e, base=bg: self._on_row_hover(row, base, e)
        return row

    # ── Pool-based row creation and incremental update ────────────────

    def _build_cell(self, text, color, size, bold, expand_extra=False):
        """Create a single cell: ft.Text inside ft.Container."""
        w = ft.Text(
            text,
            size=size,
            weight=ft.FontWeight.BOLD if bold else ft.FontWeight.NORMAL,
            color=color,
            max_lines=1,
            overflow=ft.TextOverflow.CLIP,
        )
        return ft.Container(content=w, expand=True), w

    def _create_row_controls(self, r, rank, c, bg, threshold):
        """Build a row's Flet controls and store text refs in the pool.

        Returns the outermost ``ft.Container`` (identical structure to
        ``_make_data_row``) and registers it in ``_row_pool`` /
        ``_row_cells`` so that ``_update_row`` can patch values in-place.
        """
        total = _score_of(r)
        ticker = r.get("ticker", "?")
        trend_dir = r.get("trend_dir") or ""
        is_above = total >= threshold
        rating = r.get("combined_rating", "POOR")
        rating_txt_color = rating_color(rating, c)
        entry = bool(r.get("entry_signal"))
        accent = self._rating_accent(rating)

        cols = [
            (str(rank), c["text_dim"], 11, False),
            (ticker, c["green"] if is_above else c["text"], 12, True),
            (f"{total:.0f}", score_color(total, c), 14, True),
            (rating, rating_txt_color, 11, True),
            (
                "YES" if entry else "--",
                c["green"] if entry else c["text_dim"],
                11,
                True,
            ),
            (f"₹{r.get('close', 0) or 0:.0f}", c["text"], 12, True),
            (self._ma_text(r), self._ma_color(r), 11, False),
            (f"{r.get('trend', 0) or 0:.0f}", c["green"], 11, False),
            (f"{r.get('momentum', 0) or 0:.0f}", c["cyan"], 11, False),
            (f"{r.get('rsi', 0) or 0:.0f}", c["blue"], 11, False),
            (f"{r.get('macd', 0) or 0:.0f}", c.get("macd", c["blue"]), 11, False),
            (f"{r.get('volume', 0) or 0:.0f}", c["orange"], 11, False),
            (f"{r.get('rel_str', 0) or 0:.0f}", c["lime"], 11, False),
            (
                f"{r.get('fundamentals', 0) or 0:.0f}",
                c.get("fund", c["yellow"]),
                11,
                False,
            ),
            (
                f"{r.get('pc1m', 0) or 0:+.1f}%",
                c["green"] if (r.get("pc1m", 0) or 0) > 0 else c["red"],
                11,
                False,
            ),
            (
                ("^ " if trend_dir == "Bull" else "v ") + (trend_dir or "?"),
                c["green"] if trend_dir == "Bull" else c["red"],
                11,
                False,
            ),
            (f"{r.get('adx_val', 0) or 0:.0f}", c["text"], 11, False),
            (
                "Chop" if r.get("is_sideways") else "OK",
                c["orange"] if r.get("is_sideways") else c["green"],
                11,
                False,
            ),
        ]

        wash_for = {}
        if entry:
            wash_for[4] = ft.Colors.with_opacity(0.14, c["green"])
        wash_for[3] = {
            "EXCELLENT": ft.Colors.with_opacity(0.12, c["green"]),
            "GOOD": ft.Colors.with_opacity(0.09, c["lime"]),
            "MODERATE": ft.Colors.with_opacity(0.12, c["orange"]),
            "POOR": ft.Colors.with_opacity(0.12, c["red"]),
        }.get(rating)

        cell_containers = []
        cell_texts = []
        for idx, (text, color, size, bold) in enumerate(cols):
            cell, txt = self._build_cell(text, color, size, bold)
            cell_texts.append(txt)

            if idx == 0:
                rail = ft.Container(width=3, height=18, border_radius=2, bgcolor=accent)
                cell.content = ft.Row(
                    controls=[rail, txt],
                    spacing=5,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                )
                cell.tooltip = rating.title()
            if idx in wash_for and wash_for[idx] is not None:
                cell.bgcolor = wash_for[idx]
                cell.border_radius = 6
            if idx == 1:
                txt.text_align = ft.TextAlign.LEFT
                txt.max_lines = 1
                txt.overflow = ft.TextOverflow.CLIP
                txt.no_wrap = True
                badge = self._sentiment_badge(r, c)
                if badge is not None:
                    txt.expand = True
                    cell.content = ft.Row(
                        controls=[txt, badge],
                        spacing=5,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    )
            cell_containers.append(cell)

        # Sparkline
        px = [
            v
            for v in (r.get("px_tail") or [])
            if v is not None and isinstance(v, (int, float))
        ]
        if len(px) >= 2:
            spark_cell = self._make_sparkline(px, ticker, c)
        else:
            spark_cell = ft.Container(
                content=ft.Text(
                    "—", size=10, color=c["text_faint"], text_align=ft.TextAlign.CENTER
                ),
                expand=True,
            )
        spark_cell.on_click = lambda e, t=ticker: self._show_stock_detail(t)
        spark_cell.tooltip = "Click for detail view"
        cell_containers.append(spark_cell)

        ticker_cell = cell_containers[1]
        ticker_cell.on_click = lambda e, t=ticker: self._toggle_stock_news(t)
        ticker_cell.tooltip = "Click for news & sentiment"

        row = ft.Container(
            content=ft.Row(
                controls=cell_containers,
                spacing=2,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=bg,
            border_radius=8,
            border=_border_all(1, c["border"]) if is_above else None,
            height=34,
            padding=_padding_only(left=6, right=6),
            margin=_margin_only(bottom=1),
            opacity=0,
            animate_opacity=ANIM_FAST,
        )
        row.on_hover = lambda e, base=bg: self._on_row_hover(row, base, e)

        # Register in pool
        if not hasattr(self, "_row_pool"):
            self._row_pool = {}
        if not hasattr(self, "_row_cells"):
            self._row_cells = {}
        self._row_pool[ticker] = row
        self._row_cells[ticker] = cell_texts
        row._pool_ticker = ticker
        return row

    def _update_row(self, ticker, rank, c, threshold, shown_len):
        """Patch an existing pooled row's text values in-place.

        Returns True if the row was found and updated, False otherwise.
        """
        row = getattr(self, "_row_pool", {}).get(ticker)
        cells = getattr(self, "_row_cells", {}).get(ticker)
        if row is None or cells is None:
            return False

        with self._results_lock:
            r = None
            for result in self.all_results:
                if result.get("ticker") == ticker:
                    r = result
                    break
        if r is None:
            return False

        total = _score_of(r)
        is_above = total >= threshold
        trend_dir = r.get("trend_dir") or ""
        rating = r.get("combined_rating", "POOR")
        entry = bool(r.get("entry_signal"))
        accent = self._rating_accent(rating)

        new_vals = [
            str(rank),
            ticker,
            f"{total:.0f}",
            rating,
            "YES" if entry else "--",
            f"₹{r.get('close', 0) or 0:.0f}",
            self._ma_text(r),
            f"{r.get('trend', 0) or 0:.0f}",
            f"{r.get('momentum', 0) or 0:.0f}",
            f"{r.get('rsi', 0) or 0:.0f}",
            f"{r.get('macd', 0) or 0:.0f}",
            f"{r.get('volume', 0) or 0:.0f}",
            f"{r.get('rel_str', 0) or 0:.0f}",
            f"{r.get('fundamentals', 0) or 0:.0f}",
            f"{r.get('pc1m', 0) or 0:+.1f}%",
            (("^ " if trend_dir == "Bull" else "v ") + (trend_dir or "?")),
            f"{r.get('adx_val', 0) or 0:.0f}",
            "Chop" if r.get("is_sideways") else "OK",
        ]

        new_colors = [
            c["text_dim"],
            c["green"] if is_above else c["text"],
            score_color(total, c),
            rating_color(rating, c),
            c["green"] if entry else c["text_dim"],
            c["text"],
            self._ma_color(r),
            c["green"],
            c["cyan"],
            c["blue"],
            c.get("macd", c["blue"]),
            c["orange"],
            c["lime"],
            c.get("fund", c["yellow"]),
            c["green"] if (r.get("pc1m", 0) or 0) > 0 else c["red"],
            c["green"] if trend_dir == "Bull" else c["red"],
            c["text"],
            c["orange"] if r.get("is_sideways") else c["green"],
        ]

        for txt, val, col in zip(cells, new_vals, new_colors):
            txt.value = val
            txt.color = col

        # Update row container styling
        row.bgcolor = (
            c["card"] if is_above else (c["row_alt"] if rank % 2 else c["main_bg"])
        )
        row.border = _border_all(1, c["border"]) if is_above else None

        # Update accent rail color (first cell's inner Row → first control)
        rank_cell = row.content.controls[0]
        rank_inner = rank_cell.content
        if isinstance(rank_inner, ft.Row) and rank_inner.controls:
            rail = rank_inner.controls[0]
            if isinstance(rail, ft.Container):
                rail.bgcolor = accent

        # Ensure the row is visible (streaming updates patch in-place)
        row.opacity = 1

        return True

    def _on_row_hover(self, container, base_bg, e):
        """Highlight the hovered result row (desktop mouse feedback)."""
        try:
            now_ms = time.monotonic() * 1000.0
            last = getattr(self, "_hover_last_ms", 0.0)
            if now_ms - last < 80:
                return
            self._hover_last_ms = now_ms
            hover_bg = self.theme_colors["row_hover"]
            is_hover = e.data == "true"
            container.bgcolor = hover_bg if is_hover else base_bg
            container.scale = ft.Scale(1.005) if is_hover else ft.Scale(1.0)
            container.animate_scale = ANIM_FAST
            container.update()
        except Exception:
            logger.info("Row hover update failed", exc_info=True)

    def _on_sort(self, col_idx):
        if self.sort_col == col_idx:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_col = col_idx
            self.sort_reverse = col_idx not in (0, 1)
        self.current_page = 0
        self._save_ui_prefs()

        def _apply():
            self._render_current_page()
            self._scroll_to_top()

        self._safe_update(_apply)

    def _get_sort_key(self, col_idx):
        rating_order = {"EXCELLENT": 4, "GOOD": 3, "MODERATE": 2, "POOR": 1, "WEAK": 0}

        def _ma_rank(r):
            if r.get("ma_crossed_above"):
                # More recent crossover (lower bars_ago) ranks higher.
                return 200 - (r.get("crossover_bars_ago") or 0)
            if r.get("ma_bullish"):
                return 1
            return 0

        sort_keys = {
            0: lambda r: _score_of(r),
            1: lambda r: r.get("ticker", ""),
            2: lambda r: _score_of(r),
            3: lambda r: rating_order.get(r.get("combined_rating", "POOR"), 0),
            4: lambda r: 1 if r.get("entry_signal") else 0,
            5: lambda r: r.get("close", 0) or 0,
            6: lambda r: _ma_rank(r),
            7: lambda r: r.get("trend", 0) or 0,
            8: lambda r: r.get("momentum", 0) or 0,
            9: lambda r: r.get("rsi", 0) or 0,
            10: lambda r: r.get("macd", 0) or 0,
            11: lambda r: r.get("volume", 0) or 0,
            12: lambda r: r.get("rel_str", 0) or 0,
            13: lambda r: r.get("fundamentals", 0) or 0,
            14: lambda r: r.get("pc1m", 0) or 0,
            15: lambda r: 1 if r.get("trend_dir") == "Bull" else 0,
            16: lambda r: r.get("adx_val", 0) or 0,
            17: lambda r: 1 if r.get("is_sideways") else 0,
            # Sparkline column sorts by the move it draws (last vs first close).
            18: lambda r: _spark_move(r),
        }
        return sort_keys.get(col_idx, lambda r: r.get("total", 0))

    def _ma_text(self, r):
        if r.get("ma_crossed_above"):
            ago = r.get("crossover_bars_ago", -1)
            cnt = r.get("crossover_count", 0)
            return f"^ X{ago}({cnt})" if cnt > 1 else f"^ X{ago}"
        elif r.get("ma_bullish"):
            return "^ Bull"
        return "v Bear"

    def _ma_color(self, r):
        c = self.theme_colors
        if r.get("ma_crossed_above"):
            return c["green"]
        elif r.get("ma_bullish"):
            return c["lime"]
        return c["red"]

    def _scroll_to_top(self):
        try:
            ms = getattr(self, "main_scroll", None)
            if ms is None:
                return
            # scroll_to() is a coroutine in this Flet version — it only takes
            # effect when awaited on the page's running loop (same class of
            # bug as TextField.focus(); see ScannerApp._focus_search).
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                task = loop.create_task(ms.scroll_to(offset=0, duration=200))
                task.add_done_callback(
                    lambda t: t.exception() if not t.cancelled() else None
                )
            else:
                # No running loop (unit tests): close immediately so the
                # un-awaited coroutine never warns at GC time.
                ms.scroll_to(offset=0, duration=200).close()
        except Exception:
            logger.info("Scroll-to-top failed", exc_info=True)

    def _change_page(self, delta):
        shown = self._visible_results()
        total_pages = max(1, (len(shown) + self.page_size - 1) // self.page_size)
        self.current_page = max(0, min(self.current_page + delta, total_pages - 1))
        self._render_current_page()
        self._scroll_to_top()
        self.page.update()

    def _on_page_size_change(self, e):
        try:
            raw = getattr(e, "control", None)
            new_val = (
                raw.value if raw is not None and getattr(raw, "value", None) else None
            ) or getattr(e, "data", None)
            self.page_size = int(new_val or self.page_size_dd.value or "100")
        except (TypeError, ValueError):
            try:
                self.page_size = int(self.page_size_dd.value or "100")
            except (TypeError, ValueError):
                self.page_size = 100
        self.page_size = max(1, min(500, self.page_size))
        logger.info(
            "_on_page_size_change: page_size=%d (event data=%r)",
            self.page_size,
            getattr(e, "data", None),
        )
        self.current_page = 0
        self._render_current_page()
        self._scroll_to_top()
        self._save_ui_prefs()
        self.page.update()

    def _load_all_pages(self):
        total = len(self._visible_results())
        self.page_size = min(500, total) if total > 0 else 500
        if str(self.page_size) in self.page_size_options:
            self.page_size_dd.value = str(self.page_size)
        self.current_page = 0
        self._render_current_page()
        self._scroll_to_top()
        self._save_ui_prefs()
        self.page.update()

    # ── Table-view persistence (sort / page size / rating filter) ────────

    def _load_ui_prefs(self):
        """Re-apply the table prefs saved by the last session (sort column,
        direction, page size and rating filter) after a fresh UI build."""
        try:
            sc = self.settings.get("ui_sort_col")
            if isinstance(sc, int) and 0 <= sc < len(RESULT_COLS):
                self.sort_col = sc
                self.sort_reverse = bool(self.settings.get("ui_sort_reverse", False))
            ps = self.settings.get("ui_page_size")
            if isinstance(ps, int) and ps > 0:
                self.page_size = min(500, ps)
                if str(self.page_size) in self.page_size_options:
                    self.page_size_dd.value = str(self.page_size)
            rf = self.settings.get("ui_rating_filter")
            dd = getattr(self, "rating_filter_dd", None)
            if dd is not None and rf in (
                "ALL",
                "EXCELLENT",
                "GOOD",
                "MODERATE",
                "POOR",
            ):
                dd.value = rf.title() if rf != "ALL" else "All"
        except Exception:
            logger.info("UI prefs restore failed", exc_info=True)

    def _save_ui_prefs(self):
        """Persist current table view to settings.json (sort, size, filter)."""
        try:
            s = self.settings
            s["ui_sort_col"] = self.sort_col
            s["ui_sort_reverse"] = self.sort_reverse
            s["ui_page_size"] = self.page_size
            s["ui_rating_filter"] = self._rating_filter()
            # Late import: scanner.app imports this mixin at module load.
            from ..backend.settings_store import save_settings

            save_settings(s)
        except Exception:
            logger.info("UI prefs save failed", exc_info=True)

    def _toggle_stock_news(self, ticker):
        ctrls = self.table_column.controls
        ticker_frames = [x for x in ctrls if getattr(x, "_news_ticker", None) == ticker]

        # Second click on an already-open panel collapses it immediately.
        if any(not getattr(x, "_news_loading", False) for x in ticker_frames):
            for x in ticker_frames:
                ctrls.remove(x)
            self.page.update()
            return

        # A fetch is already in flight for this ticker — ignore the click.
        if ticker_frames:
            return

        # The scan prefetched this row's stories — show them instantly.
        row = self._find_result_row(ticker)
        if row is not None and "_news_items" in row:
            self._show_news(ticker, row["_news_items"] or [])
            return

        # Fallback (rows the scan did not prefetch): show an immediate
        # "fetching…" placeholder; the worker replaces it once yfinance
        # responds.
        c = self.theme_colors
        try:
            from flet_spinkit import FadingCircle

            spinner = FadingCircle(color=c["cyan"], size=14)
        except ImportError:
            spinner = ft.ProgressRing(
                width=14, height=14, stroke_width=2, color=c["cyan"]
            )
        loading = ft.Container(
            content=ft.Row(
                [
                    spinner,
                    ft.Text("Loading news & sentiment…", size=11, color=c["text_dim"]),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=c["card2"],
            border_radius=12,
            padding=12,
            margin=_margin_only(bottom=4),
        )
        loading._news_ticker = ticker
        loading._news_loading = True
        self._insert_news_frame(ticker, loading)
        self.page.update()

        def _fetch_news():
            try:
                from ..backend.report import fetch_news_for_ticker

                parsed = fetch_news_for_ticker(ticker)
            except Exception:
                parsed = []
                logger.info("News fetch failed for ticker=%s", ticker, exc_info=True)
            self._safe_update(lambda: self._show_news(ticker, parsed, update=False))

        threading.Thread(target=_fetch_news, daemon=True).start()

    def _find_result_row(self, ticker: str) -> dict | None:
        """The live result-row dict for ``ticker`` (news/stats live on it)."""
        pool = (
            getattr(self, "all_results", None) or getattr(self, "results", None) or []
        )
        for r in pool:
            if r.get("ticker") == ticker:
                return r
        return None

    def _news_stats_row(self, row: dict | None) -> ft.Row | None:
        """Key-price chips (close / RSI / 1M move) for the news panel header."""
        if not row:
            return None
        c = self.theme_colors
        chips = []
        try:
            close = row.get("close")
            if close is not None:
                chips.append((f"₹{float(close):,.0f}", c["text"]))
        except (TypeError, ValueError):
            logger.debug("Failed to parse close price for news stats", exc_info=True)
        try:
            rsi = row.get("rsi")
            if rsi is not None:
                chips.append((f"RSI {float(rsi):.0f}", c["cyan"]))
        except (TypeError, ValueError):
            logger.debug("Failed to parse RSI for news stats", exc_info=True)
        try:
            move = row.get("pc1m")
            if move is not None:
                mv = float(move)
                chips.append((f"1M {mv:+.1f}%", c["green"] if mv > 0 else c["red"]))
        except (TypeError, ValueError):
            logger.debug("Failed to parse 1M move for news stats", exc_info=True)
        if not chips:
            return None
        ticker = (row.get("ticker") or "").strip().upper()
        controls = [
            ft.Container(
                content=ft.Text(label, size=10, weight=ft.FontWeight.BOLD, color=color),
                bgcolor=c["card"],
                border_radius=6,
                padding=_padding_only(left=8, right=8, top=3, bottom=3),
            )
            for label, color in chips
        ]
        if ticker:
            # Client-side copy (no server round-trip). Constructing the
            # action needs a live page context, which background flushes
            # don't have — so the button is best-effort and omitted there.
            try:
                copy_action = ft.CopyToClipboard(ticker)
            except Exception:
                logger.info(
                    "Copy action unavailable without page context", exc_info=True
                )
                copy_action = None
            if copy_action is not None:
                controls.append(
                    ft.IconButton(
                        icon=ft.Icons.CONTENT_COPY,
                        icon_size=14,
                        icon_color=c["text_dim"],
                        tooltip=f"Copy {ticker}",
                        action=copy_action,
                    )
                )
        return ft.Row(controls=controls, spacing=5)

    @staticmethod
    def _row_ticker_text(cell) -> ft.Text | None:
        """The ticker ft.Text inside a data-row cell (content may be a Row
        when a sentiment badge is present — the ticker is always first)."""
        inner = getattr(cell, "content", cell)
        if isinstance(inner, ft.Row):
            inner = inner.controls[0] if inner.controls else None
        return inner if isinstance(inner, ft.Text) else None

    def _insert_news_frame(self, ticker, frame):
        """Insert ``frame`` right under the given ticker's row."""
        frame.animate_offset = ANIM_NORMAL
        frame.animate_opacity = ANIM_NORMAL
        frame.opacity = 1
        ctrls = self.table_column.controls
        insert_at = None
        for i, ctrl in enumerate(ctrls):
            content = getattr(ctrl, "content", None)
            if isinstance(content, ft.Row):
                cells = content.controls or []
                if len(cells) > 1:
                    inner = self._row_ticker_text(cells[1])
                    if inner is not None and inner.value == ticker:
                        insert_at = i + 1
                        break
        if insert_at is None:
            ctrls.append(frame)
        else:
            ctrls.insert(insert_at, frame)

    def _show_news(self, ticker, items, update: bool = True):
        if self.active_view != "dashboard":
            return
        c = self.theme_colors
        ctrls = self.table_column.controls
        # Remove open news frames, but preserve loading placeholders for
        # other tickers (they're still fetching and will render when done).
        for ctrl in [x for x in ctrls if hasattr(x, "_news_ticker")]:
            if (
                getattr(ctrl, "_news_loading", False)
                and getattr(ctrl, "_news_ticker", None) != ticker
            ):
                continue
            try:
                ctrls.remove(ctrl)
            except ValueError:
                logger.debug(
                    "Failed to remove news frame (already removed?)", exc_info=True
                )

        news_controls = []
        stats = self._news_stats_row(self._find_result_row(ticker))
        if stats is not None:
            news_controls.append(stats)

        # ── Trade reasons ──────────────────────────────────────────────
        row = self._find_result_row(ticker)
        if row is not None:
            from ..shared.trade_reasons import build_trade_reasons

            reasons = build_trade_reasons(row)
            if reasons:
                reason_controls = []
                for r in reasons:
                    is_risk = r.startswith("Risk:")
                    icon_name = (
                        ft.Icons.WARNING_ROUNDED
                        if is_risk
                        else ft.Icons.CHECK_CIRCLE_ROUNDED
                    )
                    icon_color = c["orange"] if is_risk else c["green"]
                    reason_controls.append(
                        ft.Row(
                            [
                                ft.Icon(icon_name, size=12, color=icon_color),
                                ft.Text(r, size=10, color=c["text"], expand=True),
                            ],
                            spacing=4,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        )
                    )
                news_controls.append(
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text(
                                    "Why this trade?",
                                    size=11,
                                    weight=ft.FontWeight.BOLD,
                                    color=c["cyan"],
                                ),
                                ft.Divider(height=1, color=c["border"]),
                                *reason_controls,
                            ],
                            spacing=4,
                        ),
                        bgcolor=c["card"],
                        border_radius=10,
                        padding=10,
                        margin=_margin_only(bottom=4),
                    )
                )

        if not items:
            news_controls.append(
                ft.Text("No recent news found.", size=11, color=c["text_dim"])
            )
        else:
            good = sum(1 for i in items if i.get("sentiment") == "Good")
            bad = sum(1 for i in items if i.get("sentiment") == "Bad")
            neu = len(items) - good - bad
            news_controls.append(
                ft.Text(
                    f"{good} Good  |  {bad} Bad  |  {neu} Neutral",
                    size=11,
                    weight=ft.FontWeight.BOLD,
                    color=c["lime"],
                )
            )
            for item in items:
                sent = item.get("sentiment") or "Neutral"
                sent_colors = {
                    "Good": c["green"],
                    "Bad": c["red"],
                    "Neutral": c["text_dim"],
                }
                sent_bgs = {
                    "Good": c["chip_good"],
                    "Bad": c["chip_bad"],
                    "Neutral": c["card"],
                }
                sent_color = sent_colors.get(sent, c["text_dim"])
                sent_bg = sent_bgs.get(sent, c["card"])
                provider = item.get("provider") or item.get("publisher") or ""
                meta = (
                    f"{item.get('date', '')}  {provider}"
                    if provider
                    else item.get("date", "")
                )
                card_lines = [
                    ft.Row(
                        [
                            ft.Container(
                                content=ft.Text(
                                    sent,
                                    size=9,
                                    weight=ft.FontWeight.BOLD,
                                    color=sent_color,
                                ),
                                bgcolor=sent_bg,
                                border_radius=6,
                                padding=_padding_only(left=5, right=5, top=1, bottom=1),
                            ),
                            ft.Text(meta, size=9, color=c["text_dim"]),
                        ],
                        spacing=6,
                    ),
                    ft.Text(
                        item.get("title", ""),
                        size=11,
                        weight=ft.FontWeight.BOLD,
                        color=c["text"],
                        max_lines=2,
                    ),
                ]
                if item.get("summary"):
                    card_lines.append(
                        ft.Text(
                            item["summary"], size=10, color=c["text_dim"], max_lines=2
                        )
                    )
                news_controls.append(
                    ft.Container(
                        content=ft.Column(card_lines, spacing=3),
                        bgcolor=c["card"],
                        border_radius=10,
                        padding=10,
                        margin=_margin_only(bottom=3),
                    )
                )

        news_frame = ft.Container(
            content=ft.Column(controls=news_controls, spacing=5),
            bgcolor=c["card2"],
            border_radius=12,
            padding=10,
            margin=_margin_only(bottom=4),
        )
        news_frame._news_ticker = ticker
        self._insert_news_frame(ticker, news_frame)
        if update:
            self.page.update()

    def _make_scan_placeholder(self, headline="Scanning — fetching batches…"):
        """Animated shimmer skeleton shown in the results area during a scan."""
        c = self.theme_colors
        skeleton_rows = [shimmer_row() for _ in range(8)]
        try:
            from flet_spinkit import DoubleBounce

            spinner = DoubleBounce(color=c["green"], size=36)
        except ImportError:
            spinner = ft.ProgressRing(
                width=36, height=36, stroke_width=3, color=c["green"]
            )
        return ft.Container(
            content=ft.Column(
                [
                    spinner,
                    ft.Container(height=8),
                    ft.Text(
                        headline, size=13, weight=ft.FontWeight.BOLD, color=c["green"]
                    ),
                    ft.Text(
                        "First results appear after ~1 batch (~20s)",
                        size=11,
                        color=c["text_dim"],
                    ),
                    ft.Container(height=12),
                    *skeleton_rows,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=4,
            ),
            alignment=Alignment.CENTER,
            padding=40,
        )

    def _render_chart(self, results):
        c = self.theme_colors
        if not results:
            self.chart_card.visible = False
            return
        from .views_charts import build_score_histogram

        threshold = self.settings.get("min_score", 50)
        histogram = build_score_histogram(
            results, c, threshold=threshold, width=520, height=140
        )
        self.chart_bars.controls = [histogram]
        top = sorted(results, key=_score_of, reverse=True)[:50]
        peak = max(_score_of(r) for r in top) or 1
        self.chart_sub.value = f"top {len(top)}  ·  peak {peak:.0f}"
        self.chart_card.visible = True

    # ── Summary / Hero ──────────────────────────────────────────────────

    def _update_summary(self, results):
        if not results:
            for lbl in self.summary_cards.values():
                lbl.value = "—"
            return
        threshold = self.settings.get("min_score", 50)
        total = len(results)
        passed = 0
        score_sum = 0.0
        high = 0.0
        bull = bear = entry = 0
        for r in results:
            score = _score_of(r)
            if score >= threshold:
                passed += 1
            score_sum += score
            if score > high:
                high = score
            td = r.get("trend_dir")
            if td == "Bull":
                bull += 1
            elif td == "Bear":
                bear += 1
            if r.get("entry_signal"):
                entry += 1
        avg = score_sum / total
        try:
            from ..api import data_fetcher

            dead_skips = data_fetcher.negative_cache_skip_count()
        except Exception:
            logger.info("Failed to load negative cache skip count", exc_info=True)
            dead_skips = 0

        self.summary_cards["total"].value = str(total)
        self.summary_cards["passed"].value = str(passed)
        self.summary_cards["entry"].value = str(entry)
        self.summary_cards["avg"].value = f"{avg:.1f}"
        self.summary_cards["high"].value = f"{high:.0f}"
        self.summary_cards["bull"].value = str(bull)
        self.summary_cards["bear"].value = str(bear)
        self.summary_cards["dead_skip"].value = str(dead_skips)

    def _update_hero_status(self, results):
        if self.scanning:
            txt = "Scanning… fetching data and scoring stocks"
        elif results:
            threshold = self.settings.get("min_score", 50)
            passed = len([r for r in results if _score_of(r) >= threshold])
            entry_ct = len([r for r in results if r.get("entry_signal")])
            txt = f"{len(results)} stocks passed the crossover filter  ·  {passed} scored {threshold:.0f}+  ·  {entry_ct} ENTRY signals"
        else:
            txt = "Set your universe on the left, then RUN SCAN — HMA×EMA crossover • 10-factor score • news sentiment"
        c = self.theme_colors
        warnings = list(getattr(self, "last_warnings", []) or [])
        if warnings:
            txt += "\n\u26a0 " + "\n\u26a0 ".join(warnings[:2])
            if len(warnings) > 2:
                txt += f"\n\u26a0 +{len(warnings) - 2} more"
        self.hero_sub.value = txt
        if warnings:
            self.hero_sub.color = c.get("orange", c["text_dim"])
            self.hero_sub.size = 11
        else:
            self.hero_sub.color = c["hero_sub"]
            self.hero_sub.size = 12

    # ── Per-stock detail view ────────────────────────────────────────

    def _show_stock_detail(self, ticker: str):
        """Replace the table with a full-width detail panel for ``ticker``."""
        if self.active_view != "dashboard":
            return
        self._detail_ticker = ticker
        c = self.theme_colors

        row = self._find_result_row(ticker)
        if row is None:
            return

        # Build the detail panel
        panel = self._build_detail_panel(row, c)

        # Replace table content with the detail panel
        self.table_column.controls.clear()
        self.table_column.controls.append(panel)
        self.pagination_bar.visible = False
        self.page.update()

    def _build_detail_panel(self, row: dict, c: dict) -> ft.Container:
        """Build the full-width detail panel for a single stock."""
        from ..shared.trade_reasons import build_trade_reasons
        from .views_charts import build_price_chart, build_score_breakdown

        ticker = row.get("ticker", "?")
        total = _score_of(row)
        rating = row.get("combined_rating", "POOR")
        entry = bool(row.get("entry_signal"))
        trend_dir = row.get("trend_dir") or ""

        # ── Header with back button ──────────────────────────────────
        header = ft.Container(
            content=ft.Row(
                [
                    ft.IconButton(
                        icon=ft.Icons.ARROW_BACK_ROUNDED,
                        icon_size=18,
                        icon_color=c["cyan"],
                        tooltip="Back to results",
                        on_click=lambda _: self._back_to_results(),
                    ),
                    ft.Text(
                        ticker,
                        size=18,
                        weight=ft.FontWeight.BOLD,
                        color=c["text"],
                        expand=True,
                    ),
                    ft.Container(
                        content=ft.Text(
                            f"{total:.0f}",
                            size=20,
                            weight=ft.FontWeight.BOLD,
                            color=score_color(total, c),
                        ),
                        bgcolor=ft.Colors.with_opacity(0.15, score_color(total, c)),
                        border_radius=8,
                        padding=_padding_only(left=12, right=12, top=4, bottom=4),
                    ),
                    ft.Container(
                        content=ft.Text(
                            rating,
                            size=12,
                            weight=ft.FontWeight.BOLD,
                            color=rating_color(rating, c),
                        ),
                        bgcolor=ft.Colors.with_opacity(0.12, rating_color(rating, c)),
                        border_radius=6,
                        padding=_padding_only(left=8, right=8, top=3, bottom=3),
                    ),
                    ft.Container(
                        content=ft.Text(
                            "ENTRY" if entry else "NO ENTRY",
                            size=11,
                            weight=ft.FontWeight.BOLD,
                            color=c["green"] if entry else c["text_dim"],
                        ),
                        bgcolor=ft.Colors.with_opacity(0.12, c["green"])
                        if entry
                        else None,
                        border_radius=6,
                        padding=_padding_only(left=8, right=8, top=3, bottom=3),
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=c["card"],
            border_radius=12,
            border=_border_all(1, c["border"]),
            padding=_padding_only(left=12, right=12, top=8, bottom=8),
            margin=_margin_only(bottom=8),
        )

        # ── Price chart ──────────────────────────────────────────────
        px_tail = row.get("px_tail") or []
        price_chart = build_price_chart(
            closes=px_tail,
            c=c,
            width=600,
            height=200,
            title="Price (last 20 closes)",
        )

        # ── Score breakdown ──────────────────────────────────────────
        breakdown = build_score_breakdown(row, c, width=300, height=200)

        # ── Key signals ──────────────────────────────────────────────
        signals_data = [
            (
                "RSI",
                f"{row.get('rsi_val', 0) or 0:.1f}",
                c["green"] if 40 <= (row.get("rsi_val") or 0) <= 70 else c["orange"],
            ),
            (
                "ADX",
                f"{row.get('adx_val', 0) or 0:.1f}",
                c["green"] if (row.get("adx_val") or 0) > 20 else c["red"],
            ),
            (
                "MACD",
                f"{row.get('macd', 0) or 0:.1f}",
                c["green"] if (row.get("macd", 0) or 0) > 0 else c["red"],
            ),
            (
                "ATR%",
                f"{row.get('atr_pct', 0) or 0:.2f}%",
                c["orange"] if (row.get("atr_pct", 0) or 0) > 3 else c["text"],
            ),
            (
                "POC",
                "Above" if row.get("above_poc") else "Below",
                c["green"] if row.get("above_poc") else c["red"],
            ),
            ("MA", self._ma_text(row), self._ma_color(row)),
            (
                "Dir",
                f"^ {trend_dir}" if trend_dir == "Bull" else f"v {trend_dir}",
                c["green"] if trend_dir == "Bull" else c["red"],
            ),
            (
                "Chop",
                "Sideways" if row.get("is_sideways") else "Trending",
                c["orange"] if row.get("is_sideways") else c["green"],
            ),
        ]

        signal_items = []
        for label, value, color in signals_data:
            signal_items.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(label, size=9, color=c["text_faint"]),
                            ft.Text(
                                value, size=13, weight=ft.FontWeight.BOLD, color=color
                            ),
                        ],
                        spacing=2,
                    ),
                    bgcolor=c["card"],
                    border_radius=8,
                    border=_border_all(1, c["border"]),
                    padding=_padding_only(left=10, right=10, top=6, bottom=6),
                    width=80,
                )
            )

        signals_row = ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "Key Signals",
                        size=12,
                        weight=ft.FontWeight.BOLD,
                        color=c["cyan"],
                    ),
                    ft.Row(signal_items, spacing=6, wrap=True),
                ],
                spacing=6,
            ),
            bgcolor=c["card"],
            border_radius=12,
            border=_border_all(1, c["border"]),
            padding=12,
            margin=_margin_only(bottom=8),
        )

        # ── Trade reasons ────────────────────────────────────────────
        reasons = build_trade_reasons(row)
        reason_controls = []
        for r in reasons:
            is_risk = r.startswith("Risk:")
            icon_name = (
                ft.Icons.WARNING_ROUNDED if is_risk else ft.Icons.CHECK_CIRCLE_ROUNDED
            )
            icon_color = c["orange"] if is_risk else c["green"]
            reason_controls.append(
                ft.Row(
                    [
                        ft.Icon(icon_name, size=12, color=icon_color),
                        ft.Text(r, size=11, color=c["text"], expand=True),
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                )
            )

        reasons_panel = ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "Why this trade?",
                        size=12,
                        weight=ft.FontWeight.BOLD,
                        color=c["cyan"],
                    ),
                    ft.Divider(height=1, color=c["border"]),
                ]
                + (
                    reason_controls
                    if reason_controls
                    else [
                        ft.Text(
                            "No trade reasons available.", size=11, color=c["text_dim"]
                        ),
                    ]
                ),
                spacing=5,
            ),
            bgcolor=c["card"],
            border_radius=12,
            border=_border_all(1, c["border"]),
            padding=12,
            margin=_margin_only(bottom=8),
        )

        # ── Compose layout ───────────────────────────────────────────
        left_col = ft.Column(
            [
                price_chart,
                signals_row,
            ],
            spacing=8,
            expand=True,
        )

        right_col = ft.Column(
            [
                breakdown,
                reasons_panel,
            ],
            spacing=8,
            width=320,
        )

        return ft.Container(
            content=ft.Column(
                [
                    header,
                    ft.Row([left_col, right_col], spacing=12, expand=True),
                ],
                spacing=0,
            ),
            expand=True,
        )

    def _back_to_results(self):
        """Return from detail view to the results table."""
        self._detail_ticker = None
        self._render_current_page()
        self.page.update()
