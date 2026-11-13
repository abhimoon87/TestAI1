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
import os
import threading
import time

import flet as ft
from flet.canvas import Canvas, Path
from flet.controls.alignment import Alignment

logger = logging.getLogger(__name__)
from ..shared.constants import RESULT_COLS
from ..shared.constants import score_of as _score_of
from ..shared.detail_specs import ma_chip, spark_move
from .ui_kit import (
    ANIM_FAST,
    ANIM_NORMAL,
    OPTIONAL_COLS,
    RADIUS_LG,
    RADIUS_MD,
    RADIUS_SM,
    ROW_H,
    _border_all,
    _card_shadow,
    _margin_only,
    _padding_only,
    glow,
    grad_v,
    hidden_cols,
    rating_color,
    score_color,
    shimmer_row,
)

# Per-column (size, bold) — pairs with _row_specs() value/color order.
_ROW_STYLE = (
    (11, False),
    (12, True),
    (14, True),
    (11, True),
    (11, True),
    (12, True),
    (11, False),
    (11, False),
    (11, False),
    (11, False),
    (11, False),
    (11, False),
    (11, False),
    (11, False),
    (11, False),
    (11, False),
    (11, False),
    (11, False),
    (11, False),
)


def _row_specs(r, rank, c, threshold):
    """Shared 18-cell (text, color) pairs for create and update."""
    total = _score_of(r)
    is_above = total >= threshold
    trend_dir = r.get("trend_dir") or ""
    rating = r.get("combined_rating", "POOR")
    entry = bool(r.get("entry_signal"))
    state = r.get("state")  # Weekly V4 rows: TRIGGERED / ARMED / EXPIRED
    above = bool(r.get("above_poc"))
    both_ma = bool(r.get("close_above_both_ma"))
    rsi_val = r.get("rsi_val")
    fund_pts = r.get("fundamentals", 0) or 0
    if state == "TRIGGERED":
        entry_text, entry_color = "YES", c["green"]
    elif state == "ARMED":
        entry_text, entry_color = "ARMED", c["orange"]
    else:
        entry_text = "YES" if entry else "--"
        entry_color = c["green"] if entry else c["text_dim"]
    return [
        (str(rank), c["text_dim"]),
        (r.get("ticker", "?"), c["green"] if is_above else c["text"]),
        (f"{total:.0f}", score_color(total, c)),
        (rating, rating_color(rating, c)),
        (entry_text, entry_color),
        (f"₹{r.get('close', 0) or 0:.0f}", c["text"]),
        (None, None),  # MA cell — caller patches via _ma_text/_ma_color
        ("Above" if above else "Below", c["green"] if above else c["red"]),
        (
            "YES" if both_ma else "NO",
            c["green"] if both_ma else c["text_dim"],
        ),
        (f"{r.get('volume', 0) or 0:.0f}", c["orange"]),
        (f"{r.get('rel_str', 0) or 0:.0f}", c["lime"]),
        (
            f"{fund_pts:.0f}" if fund_pts > 0 else "N/A",
            c.get("fund", c["yellow"]) if fund_pts > 0 else c["text_dim"],
        ),
        (
            "—" if rsi_val is None else f"{rsi_val:.1f}",
            c["text_dim"]
            if rsi_val is None
            else (c["green"] if 40 <= rsi_val <= 70 else c["orange"]),
        ),
        (
            f"{r.get('pc1m', 0) or 0:+.1f}%",
            c["green"] if (r.get("pc1m", 0) or 0) > 0 else c["red"],
        ),
        (
            ("^ " if trend_dir == "Bull" else "v ") + (trend_dir or "?"),
            c["green"] if trend_dir == "Bull" else c["red"],
        ),
        (f"{r.get('adx_val', 0) or 0:.0f}", c["text"]),
        (r.get("volat_stat") or "—", c["text"]),
        (
            "Chop" if r.get("is_sideways") else "OK",
            c["orange"] if r.get("is_sideways") else c["green"],
        ),
    ]


def _entry_cell_tooltip(r) -> str | None:
    """Tooltip for the ENTRY cell: Weekly-V4 detail when the row has a
    state, else the daily gates that failed. Shared by row creation and
    in-place updates so a live re-render never loses the detail."""
    if r.get("state"):
        tip = [f"V4: {r['state']}"]
        if r.get("v4_cross_date"):
            tip.append(f"cross {r['v4_cross_date']}")
        if r.get("v4_n") is not None:
            tip.append(f"{r['v4_n']}/15 bars")
        if r.get("v4_level"):
            tip.append(f"level {r['v4_level']:.1f}")
        if r.get("v4_missing"):
            tip.append("missing: " + ", ".join(r["v4_missing"]))
        if r.get("v4_note"):
            tip.append(str(r["v4_note"]))
        return " · ".join(tip)
    miss = r.get("entry_missing") or []
    return "missing: " + ", ".join(miss) if miss else None


class ResultsViewMixin:
    # ── Row control pool (performance optimisation) ─────────────────
    # Maps ticker → ft.Container (the outermost row wrapper).  When a
    # row already exists in the pool we only patch the ft.Text.value
    # properties inside it, avoiding a full control-tree rebuild.
    _row_pool: dict[str, ft.Container]
    # Maps ticker → list[ft.Text] (the 22 text cells in column order).
    _row_cells: dict[str, list[ft.Text]]

    def _user_hidden(self) -> set[str]:
        """Columns turned off in the Columns menu (lazy-init: the test
        harness builds ScannerApp with ``__new__`` and never runs
        ``__init__``)."""
        if not hasattr(self, "user_hidden_cols"):
            self.user_hidden_cols = set()
        return self.user_hidden_cols

    def _row_density(self) -> str:
        """Active density preset name (unknown → Comfort)."""
        name = getattr(self, "row_density", "comfort")
        return name if name in ROW_H else "comfort"

    def _row_h(self) -> int:
        """Row height for the active density."""
        return ROW_H[self._row_density()]

    def _hidden_cols(self) -> frozenset:
        """Columns dropped at the current width tier plus any the user hid."""
        return hidden_cols(getattr(self, "width_tier", "wide")) | frozenset(
            self._user_hidden()
        )

    def _row_specs(self, r, rank, c, threshold):
        """Shared (text, color) pairs for the columns visible at this tier."""
        specs = _row_specs(r, rank, c, threshold)
        specs[6] = (self._ma_text(r), self._ma_color(r))
        hidden = self._hidden_cols()
        if hidden:
            specs = [p for i, p in enumerate(specs) if RESULT_COLS[i] not in hidden]
        return specs

    def _display_results(self, results):
        with self._results_lock:
            self.all_results = list(results)
            self.filtered_results = [r for r in results if self._row_matches_filters(r)]
        self.current_page = 0
        self._render_current_page()
        self.page.update()

    def _sorted_visible(self) -> list:
        """Visible results in display order (filter + active sort applied)."""
        shown = self._visible_results()
        if self.sort_col is not None and shown:
            try:
                key_fn = self._get_sort_key(self.sort_col)
                shown = sorted(shown, key=key_fn, reverse=self.sort_reverse)
            except Exception as ex:
                logger.info("Sort failed for col %s: %s", self.sort_col, ex)
        return shown

    def _render_current_page(self):
        if self.active_view != "dashboard":
            logger.debug("_render_current_page: SKIP active_view=%s", self.active_view)
            return
        c = self.theme_colors
        with self._results_lock:
            results = list(self.all_results)
        shown = self._sorted_visible()
        logger.info(
            "_render_current_page: results=%d shown=%d scanning=%s active_view=%s table_ctrls=%d",
            len(results),
            len(shown),
            self.scanning,
            self.active_view,
            len(self.table_column.controls),
        )

        # ── Streaming fast path: patch rows in-place, no full rebuild ──
        threshold = self.settings.get("min_score", 50)
        page_size = self.page_size
        total_pages = max(1, (len(shown) + page_size - 1) // page_size) if shown else 1
        self.current_page = max(0, min(self.current_page, total_pages - 1))
        start = self.current_page * page_size
        page_shown = shown[start : start + page_size] if shown else []

        pool = getattr(self, "_row_pool", None)
        cells = getattr(self, "_row_cells", None)
        logger.info(
            "_render_current_page: pool=%s cells=%s sort=%s filter_active=%s controls=%d",
            len(pool) if pool is not None else None,
            len(cells) if cells is not None else None,
            self.sort_col,
            self._is_filter_active(),
            len(self.table_column.controls),
        )

        # ── Pooled path: patch rows in place, rebuild the control list in
        # page order ── page_shown is already filtered + sorted, so stream,
        # filter and sort all reuse row controls instead of clearing the
        # pool every render (the old sort/filter gates made this path
        # unreachable and full-rebuilt ~4500 controls every 2s).
        if pool and cells and shown:
            # Pinned column header (outside the row scroll pane) — rebuilt
            # each render so sort arrows track sort_col/sort_reverse.
            holder = getattr(self, "header_holder", None)
            if holder is not None:
                holder.controls = [self._make_header_row(c)]

            new_controls = []
            for rank, r in enumerate(page_shown, start + 1):
                if self._update_row(r, rank, c, threshold):
                    row = pool[r.get("ticker", "?")]
                else:
                    score = _score_of(r)
                    is_above = score >= threshold
                    row_bg = (
                        c["card"]
                        if is_above
                        else (c["row_alt"] if rank % 2 else c["main_bg"])
                    )
                    row = self._create_row_controls(r, rank, c, row_bg, threshold)
                new_controls.append(row)
            self.table_column.controls = new_controls
            self._stagger_rows(new_controls)

            logger.info(
                "_render_current_page POOLED DONE: table_ctrls=%d, shown=%d, results=%d",
                len(new_controls),
                len(shown),
                len(results),
            )

        # ── Full rebuild path (first render / pool cleared / no rows) ──
        else:
            # Pool survives here: an empty filtered view must keep rows so
            # loosening the filter takes the pooled path again. Explicit
            # invalidators (scan start, width tier, detail open, settings)
            # clear the pool themselves.
            self.table_column.controls.clear()

            if not shown:
                holder = getattr(self, "header_holder", None)
                if holder is not None:
                    holder.controls = []
                if self.scanning and not results:
                    live = (getattr(self.progress_label, "value", "") or "").strip()
                    headline = (
                        live
                        if live and live not in ("Ready", "Done", "Stopped")
                        else "Scanning — fetching batches…"
                    )
                    self.table_column.controls.append(
                        self._make_scan_placeholder(headline)
                    )
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
                            "Loosen the search/rating/threshold filter to see the hidden rows."
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
                holder = getattr(self, "header_holder", None)
                if holder is not None:
                    holder.controls = [self._make_header_row(c)]

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
                self._stagger_rows(list(self.table_column.controls))

        # ── Shared tail: pagination + summary/hero/chart ──────────────
        show_bar = bool(shown and len(shown) > page_size)
        self.pagination_bar.visible = show_bar
        self.pagination_row.visible = show_bar
        if shown:
            self.page_label.value = (
                f"Page {self.current_page + 1} / {total_pages}  ({len(shown)} stocks)"
            )

        if results:
            filter_parts = []
            if self.filter_text:
                filter_parts.append(f"'{self.filter_text}'")
            rating = self._rating_filter()
            if rating != "ALL":
                filter_parts.append(f"rating {rating.title()}")
            thr = self._score_threshold()
            if thr > 0:
                filter_parts.append(f"score ≥ {thr:.0f}")
            if self._checkbox_on("inst_filter_cb"):
                filter_parts.append("FII/DII data")
            if self._checkbox_on("price_filter_cb"):
                filter_parts.append("close < ₹100")
            if self._checkbox_on("fund_filter_cb"):
                filter_parts.append("fund 0")
            # Shown count sits before the filter list so a narrow window clips
            # the filter names, not the number explaining the gap.
            above = len([r for r in results if _score_of(r) >= threshold])
            if filter_parts:
                self.result_count_label.value = (
                    f"{len(results)} scanned  |  {len(shown)} shown  |  "
                    f"{above} above {threshold:.0f}+  |  filter: "
                    f"{', '.join(filter_parts)}"
                )
            else:
                self.result_count_label.value = (
                    f"{len(results)} scanned  |  {above} above {threshold:.0f}+"
                )
        else:
            self.result_count_label.value = "no scan yet"
        self._update_filter_chips()

        self._update_summary(results)
        self._update_hero_status(results)
        self._render_topicks(shown[:5])
        self._render_chart(results)
        self._apply_kb_selection()
        logger.info(
            "_render_current_page DONE: table_ctrls=%d, shown=%d, results=%d",
            len(self.table_column.controls),
            len(shown),
            len(results),
        )

    def _stagger_rows(self, rows) -> None:
        """Batched fade-in for freshly rendered rows (report's rise feel).

        Flet animations have no delay field, so the stagger is three fixed
        opacity batches on daemon timers rather than 150 per-row timers.
        Skipped under reduce_motion, mid-scan (streaming re-renders would
        flicker every batch), and in pytest (timers would race
        update_calls asserts in tests).
        """
        if (
            not rows
            or self.scanning
            or self._motion_reduced()
            or "PYTEST_CURRENT_TEST" in os.environ
        ):
            return
        try:
            first = list(rows[:30])
            for row in first:
                row.opacity = 0.0
                row.animate_opacity = ANIM_NORMAL
            batch = max(1, len(first) // 3)
            for i in range(0, len(first), batch):
                chunk = first[i : i + batch]
                delay = 0.05 * (i // batch)

                def _reveal(chunk=chunk):
                    # Timer threads cannot page.update() directly — Flet
                    # drops it and rows stay at opacity=0. Marshal via
                    # _safe_update (UI thread + single push).
                    self._safe_update(
                        lambda c=chunk: [setattr(r, "opacity", 1.0) for r in c]
                    )

                # daemon: the comment above promises daemon timers — a
                # pending reveal must never hold process exit hostage.
                t = threading.Timer(delay, _reveal)
                t.daemon = True
                t.start()
        except Exception:
            logger.info("Row stagger failed", exc_info=True)

    def _make_header_row(self, c):
        headers = []
        hidden = self._hidden_cols()
        for idx, text in enumerate(RESULT_COLS):
            if text in hidden:
                continue
            is_sorted = self.sort_col == idx
            arrow = (
                " ▲"
                if is_sorted and not self.sort_reverse
                else (" ▼" if is_sorted else " ▲▼")
            )
            color = c["green"] if is_sorted else c["cyan"]
            headers.append(
                ft.Container(
                    content=ft.Text(
                        f"{text}{arrow}",
                        size=10,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.with_opacity(0.7, color)
                        if not is_sorted
                        else color,
                        text_align=ft.TextAlign.LEFT if idx in (0, 1) else None,
                        max_lines=1,
                        no_wrap=True,
                        style=ft.TextStyle(letter_spacing=0.6),
                    ),
                    # Ticker + Rating are double-width (18 chars of
                    # "EXCELLENT" clip at one share), rest flex equally.
                    expand=2 if idx in (1, 3) else True,
                    on_click=lambda e, i=idx: self._on_sort(i),
                    tooltip=f"Sort by {text}",
                    ink=True,
                )
            )
        return ft.Container(
            content=ft.Row(controls=headers, spacing=2),
            gradient=grad_v(c["grad_header"]),
            border_radius=RADIUS_MD,
            border=_border_all(1, c["border_light"]),
            height=self._row_h(),
            padding=_padding_only(left=6, right=6, top=4, bottom=4),
            margin=_margin_only(bottom=6),
            # Pinned above the scrolling grid — a soft drop shadow keeps
            # the seam readable while rows pass underneath.
            shadow=[
                ft.BoxShadow(
                    blur_radius=8,
                    offset=ft.Offset(0, 3),
                    color=ft.Colors.with_opacity(0.35, ft.Colors.BLACK),
                ),
                ft.BoxShadow(
                    blur_style=ft.BlurStyle.INNER,
                    blur_radius=4,
                    offset=ft.Offset(0, 1),
                    color=ft.Colors.with_opacity(0.08, ft.Colors.WHITE),
                ),
            ],
        )

    # ── Active-filter chips (pinned section header) ───────────────────

    def _make_filter_chip(self, text, on_clear=None, tip=""):
        c = self.theme_colors
        chip = ft.Container(
            content=ft.Text(text, size=10, color=c["text_dim"]),
            bgcolor=c["card2"],
            border_radius=RADIUS_SM,
            border=_border_all(1, c["border"]),
            padding=_padding_only(left=6, right=6, top=2, bottom=2),
            tooltip=tip or None,
        )
        if on_clear is not None:
            chip.on_click = on_clear
            chip.ink = True
        return chip

    def _update_filter_chips(self):
        row = getattr(self, "filter_chips_row", None)
        if row is None:
            return
        chips = []
        if self.filter_text:
            chips.append(
                self._make_filter_chip(
                    f"'{self.filter_text}'  ✕",
                    lambda e: self._clear_search_filter(),
                    "Clear ticker search",
                )
            )
        rating = self._rating_filter()
        if rating != "ALL":
            chips.append(
                self._make_filter_chip(
                    f"rating {rating.title()}  ✕",
                    lambda e: self._clear_rating_filter(),
                    "Show all ratings",
                )
            )
        thr = self._score_threshold()
        if thr > 0:
            chips.append(
                self._make_filter_chip(
                    f"score ≥ {thr:.0f}",
                    None,
                    f"Rows with score {thr:.0f} or higher — adjust with the "
                    "MIN SCORE slider",
                )
            )
        row.controls = chips

    def _clear_search_filter(self):
        if getattr(self, "search_entry", None) is not None:
            self.search_entry.value = ""
        self.filter_text = ""
        if self.all_results:
            self._display_results(self.all_results)

    def _clear_rating_filter(self):
        if getattr(self, "rating_filter_dd", None) is not None:
            self.rating_filter_dd.value = "All"
        if self.all_results:
            self._display_results(self.all_results)

    def _on_filter_toggled(self, _e):
        if self.all_results:
            self._display_results(self.all_results)

    # ── Grid keyboard navigation (↑/↓ select, Enter → detail) ─────────

    def _kb_active(self) -> bool:
        """Grid nav keys apply only on the dashboard, no overlay/input."""
        return (
            self.active_view == "dashboard"
            and getattr(self, "_palette_dlg", None) is None
            and not getattr(self, "_input_focused", False)
            and getattr(self, "_detail_ticker", None) is None
        )

    def _kb_move(self, delta: int):
        if not self._kb_active():
            return
        shown = self._sorted_visible()
        if not shown:
            return
        start = self.current_page * self.page_size
        page = shown[start : start + self.page_size] or shown
        cur = next(
            (
                i
                for i, r in enumerate(page)
                if r.get("ticker") == getattr(self, "_kb_ticker", None)
            ),
            -1,
        )
        if cur == -1:
            idx = 0 if delta > 0 else len(page) - 1
        else:
            idx = max(0, min(cur + delta, len(page) - 1))
        self._kb_ticker = page[idx].get("ticker")
        self._apply_kb_selection()
        self.page.update()

    def _kb_open(self):
        if not self._kb_active():
            return
        ticker = getattr(self, "_kb_ticker", None)
        if ticker and ticker in (getattr(self, "_row_pool", None) or {}):
            self._show_stock_detail(ticker)

    def _apply_kb_selection(self):
        """Re-apply the keyboard selection highlight to pooled rows."""
        c = self.theme_colors
        pool = getattr(self, "_row_pool", None) or {}
        selected = getattr(self, "_kb_ticker", None)
        for ticker, row in pool.items():
            base = getattr(row, "_base_bg", None)
            if base is None:
                continue
            if ticker == selected:
                row.bgcolor = c["row_hover"]
                row.border = _border_all(1, c["cyan"])
            else:
                row.bgcolor = base
                row.border = None

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
            border_radius=RADIUS_SM,
            padding=_padding_only(left=4, right=4, top=1, bottom=1),
            tooltip=f"News sentiment: {tone} · {n} articles — click to read",
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
                stroke_width=1.8,
                style=ft.PaintingStyle.STROKE,
                stroke_cap=ft.StrokeCap.ROUND,
            ),
        )
        # Fill area under the curve
        fill_elements = list(elements) + [
            Path.LineTo(W, H),
            Path.LineTo(0, H),
        ]
        fill_color = ft.Colors.with_opacity(0.22, spark_color)
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

    def _build_cell(self, text, color, size, bold):
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

        Returns the outermost ``ft.Container`` and registers it in
        ``_row_pool`` / ``_row_cells`` so ``_update_row`` can patch values.
        """
        ticker = r.get("ticker", "?")
        is_above = _score_of(r) >= threshold
        rating = r.get("combined_rating", "POOR")
        entry = bool(r.get("entry_signal"))
        accent = rating_color(rating, c)
        hidden = self._hidden_cols()
        # Style pairs track the same filtered column set as the specs.
        styles = [s for i, s in enumerate(_ROW_STYLE) if RESULT_COLS[i] not in hidden]
        cols = [
            (text, color, *styles[i])
            for i, (text, color) in enumerate(self._row_specs(r, rank, c, threshold))
        ]

        wash_for = {}
        if entry:
            wash_for[4] = ft.Colors.with_opacity(0.14, c["green"])
        elif r.get("state") == "ARMED":
            wash_for[4] = ft.Colors.with_opacity(0.10, c["orange"])
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
            # Match header: Ticker + Rating are double-width.
            if idx in (1, 3):
                cell.expand = 2
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
            if idx == 4:
                # Entry cell: V4 detail or failed daily gates (shared builder).
                cell.tooltip = _entry_cell_tooltip(r)
            if idx == 1:
                txt.text_align = ft.TextAlign.LEFT
                txt.max_lines = 1
                txt.overflow = ft.TextOverflow.CLIP
                txt.no_wrap = True
                badge = self._sentiment_badge(r, c)
                # Badge + client-side copy ride after the ticker; each
                # swallows its own tap (no row → detail navigation).
                tail = [
                    ctrl
                    for ctrl in (badge, self._copy_ticker_button(ticker, c))
                    if ctrl is not None
                ]
                if tail:
                    txt.expand = True
                    cell.content = ft.Row(
                        controls=[txt, *tail],
                        spacing=5,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    )
            cell_containers.append(cell)

        # Sparkline (the "1M" column) — skipped when the user hid it, so
        # the cell count stays aligned with the header.
        px = [
            v
            for v in (r.get("px_tail") or [])
            if v is not None and isinstance(v, (int, float))
        ]
        if "1M" in hidden:
            spark_cell = None
        elif len(px) >= 2:
            spark_cell = self._make_sparkline(px, ticker, c)
        else:
            spark_cell = ft.Container(
                content=ft.Text(
                    "—", size=10, color=c["text_faint"], text_align=ft.TextAlign.CENTER
                ),
                expand=True,
            )
        if spark_cell is not None:
            spark_cell.on_click = lambda e, t=ticker: self._show_stock_detail(t)
            spark_cell.tooltip = "Click for detail view"
            cell_containers.append(spark_cell)

        if badge is not None:
            # Only the sentiment badge opens the news frame — ticker text
            # has no handler, so it bubbles to the row → detail view.
            badge.on_click = lambda e, t=ticker: self._toggle_stock_news(t)

        row = ft.Container(
            content=ft.Row(
                controls=cell_containers,
                spacing=2,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=bg,
            border_radius=RADIUS_SM,
            border=_border_all(1, c["border"]) if is_above else None,
            shadow=_card_shadow() if is_above else None,
            height=self._row_h(),
            padding=_padding_only(left=6, right=6),
            margin=_margin_only(bottom=1),
        )
        row.on_hover = lambda e, base=bg: self._on_row_hover(row, base, e)
        # Whole row → detail view (badge keeps news, copy button copies,
        # sparkline keeps detail — Flet dispatches to the innermost
        # control with a handler).
        row.on_click = lambda e, t=ticker: self._show_stock_detail(t)
        row.tooltip = "Click for detail view"
        row._base_bg = bg
        row._above = is_above

        # Register in pool
        if not hasattr(self, "_row_pool"):
            self._row_pool = {}
        if not hasattr(self, "_row_cells"):
            self._row_cells = {}
        self._row_pool[ticker] = row
        self._row_cells[ticker] = cell_texts
        row._pool_ticker = ticker
        return row

    def _update_row(self, r, rank, c, threshold):
        """Patch an existing pooled row's text values in-place.

        ``r`` is the row dict handed in by the caller (page_shown), so no
        lock-guarded scan of all_results is needed per row — that scan was
        O(results) x page_size per render.

        Returns True if the pooled row was found and updated, False otherwise.
        """
        ticker = r.get("ticker", "?")
        row = getattr(self, "_row_pool", {}).get(ticker)
        cells = getattr(self, "_row_cells", {}).get(ticker)
        if row is None or cells is None:
            return False

        total = _score_of(r)
        is_above = total >= threshold
        rating = r.get("combined_rating", "POOR")
        accent = rating_color(rating, c)

        for txt, (val, col) in zip(
            cells, self._row_specs(r, rank, c, threshold), strict=True
        ):
            txt.value = val
            txt.color = col

        # Update row container styling
        row.bgcolor = (
            c["card"] if is_above else (c["row_alt"] if rank % 2 else c["main_bg"])
        )
        row.border = _border_all(1, c["border"]) if is_above else None
        row.shadow = glow(c["glow_emerald"], blur=12, spread=-2) if is_above else None
        row._above = is_above

        # Update accent rail color (first cell's inner Row → first control)
        rank_cell = row.content.controls[0]
        rank_inner = rank_cell.content
        if isinstance(rank_inner, ft.Row) and rank_inner.controls:
            rail = rank_inner.controls[0]
            if isinstance(rail, ft.Container):
                rail.bgcolor = accent

        # ENTRY cell (idx 4) wash + V4 tooltip — pooled rows outlive scans,
        # so a ticker can flip TRIGGERED/ARMED/clear between renders.
        hidden = self._hidden_cols()
        if "ENTRY" not in hidden:
            entry_idx = [col for col in RESULT_COLS if col not in hidden].index("ENTRY")
            entry_cell = row.content.controls[entry_idx]
            if bool(r.get("entry_signal")):
                entry_cell.bgcolor = ft.Colors.with_opacity(0.14, c["green"])
            elif r.get("state") == "ARMED":
                entry_cell.bgcolor = ft.Colors.with_opacity(0.10, c["orange"])
            else:
                entry_cell.bgcolor = None
            entry_cell.tooltip = _entry_cell_tooltip(r)

        return True

    def _on_row_hover(self, container, base_bg, e):
        """Highlight the hovered result row (desktop mouse feedback)."""
        try:
            now_ms = time.monotonic() * 1000.0
            last = getattr(self, "_hover_last_ms", 0.0)
            if now_ms - last < 80:
                return
            self._hover_last_ms = now_ms
            c = self.theme_colors
            hover_bg = c["row_hover"]
            is_hover = e.data == "true"
            if not is_hover and getattr(container, "_pool_ticker", None) == getattr(
                self, "_kb_ticker", None
            ):
                container.bgcolor = hover_bg  # keep keyboard selection
            else:
                container.bgcolor = hover_bg if is_hover else base_bg
            if is_hover:
                container.shadow = glow(c["glow_emerald"], blur=20, spread=0)
            else:
                container.shadow = (
                    _card_shadow()
                    if getattr(container, "_above", False)
                    else None
                )
            container.scale = ft.Scale(1.005) if is_hover else ft.Scale(1.0)
            container.animate_scale = ANIM_FAST
            container.update()
        except Exception:
            logger.info("Row hover update failed", exc_info=True)

    def _on_sort(self, col_idx):
        # Entry timestamp: the gap to the next _render_current_page log is
        # _save_ui_prefs() time; a stall before this line is event delivery.
        logger.info("_on_sort: click col=%s (prev=%s)", col_idx, self.sort_col)
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

    # ── Grid view toggles (Columns menu / Density menu) ──────────────

    def _toggle_user_col(self, name: str) -> None:
        """Show or hide an optional column from the Columns menu.

        The visible cell count changes, so pooled rows are dropped and the
        page is rebuilt instead of patched (``_update_row`` zips cells to
        specs with ``strict=True``).
        """
        logger.info(
            "_toggle_user_col: %s (hidden=%s)", name, sorted(self._user_hidden())
        )
        # Only tail columns may be toggled: the positional row specials at
        # indexes 0/1/3/4/6 shift cells in _create_row_controls if hidden,
        # and a column the window width already hides is out of the menu's
        # hands — recording it would keep it hidden after a reflow to wide.
        if name not in OPTIONAL_COLS or name in hidden_cols(
            getattr(self, "width_tier", "wide")
        ):
            return
        hidden = self._user_hidden()
        if name in hidden:
            hidden.discard(name)
        else:
            hidden.add(name)
        self._apply_grid_layout()

    def _set_density(self, name: str) -> None:
        """Switch row density preset (row height is baked into pooled rows)."""
        logger.info("_set_density: %s (was %s)", name, self._row_density())
        if name not in ROW_H or name == self._row_density():
            return
        self.row_density = name
        self._apply_grid_layout()

    def _apply_grid_layout(self) -> None:
        """Re-render after a layout toggle and persist both prefs."""
        if hasattr(self, "_row_pool"):
            self._row_pool.clear()
        self._save_ui_prefs()
        self._sync_menu_checks()
        self._safe_update(self._render_current_page)

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
            7: lambda r: 1 if r.get("above_poc") else 0,
            8: lambda r: 1 if r.get("close_above_both_ma") else 0,
            9: lambda r: r.get("volume", 0) or 0,
            10: lambda r: r.get("rel_str", 0) or 0,
            11: lambda r: r.get("fundamentals", 0) or 0,
            12: lambda r: r.get("rsi_val", 0) or 0,
            13: lambda r: r.get("pc1m", 0) or 0,
            14: lambda r: 1 if r.get("trend_dir") == "Bull" else 0,
            15: lambda r: r.get("adx_val", 0) or 0,
            16: lambda r: r.get("volat_stat") or "",
            17: lambda r: 1 if r.get("is_sideways") else 0,
            # Sparkline column sorts by the move it draws (last vs first close).
            18: lambda r: spark_move(r),
        }
        return sort_keys.get(col_idx, lambda r: r.get("total", 0))

    def _ma_text(self, r):
        return ma_chip(r)[0]

    def _ma_color(self, r):
        return self.theme_colors[ma_chip(r)[1]]

    def _scroll_to_top(self):
        try:
            targets = [
                t
                for t in (
                    getattr(self, "table_column", None),
                    getattr(self, "main_scroll", None),
                )
                if t is not None and hasattr(t, "scroll_to")
            ]
            if not targets:
                return
            # scroll_to() is a coroutine in this Flet version — it only takes
            # effect when awaited on the page's running loop (same class of
            # bug as TextField.focus(); see ScannerApp._focus_search).
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            for target in targets:
                if loop is not None:
                    task = loop.create_task(target.scroll_to(offset=0, duration=200))
                    task.add_done_callback(
                        lambda t: t.exception() if not t.cancelled() else None
                    )
                else:
                    # No running loop (unit tests): close immediately so the
                    # un-awaited coroutine never warns at GC time.
                    target.scroll_to(offset=0, duration=200).close()
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
        direction, page size, rating filter, density and hidden columns)
        after a fresh UI build."""
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
            dens = self.settings.get("ui_density")
            self.row_density = dens if dens in ROW_H else "comfort"
            # Stale or renamed column names are dropped against the menu's
            # allow-list, so a hand-edited prefs file can't desync the grid.
            saved = {
                n
                for n in str(self.settings.get("ui_hidden_cols") or "").split(",")
                if n
            }
            self.user_hidden_cols = saved & set(OPTIONAL_COLS)
        except Exception:
            logger.info("UI prefs restore failed", exc_info=True)

    def _save_ui_prefs(self):
        """Persist current table view to settings (sort, size, filter,
        density, hidden columns).

        Debounced on the page loop: column-header clicks fire a sort per
        click, and each save is a db write — coalesce bursts into one.
        Falls back to an immediate write when no loop is running (tests).
        """

        def _save():
            try:
                s = self.settings
                s["ui_sort_col"] = self.sort_col
                s["ui_sort_reverse"] = self.sort_reverse
                s["ui_page_size"] = self.page_size
                s["ui_rating_filter"] = self._rating_filter()
                s["ui_density"] = self._row_density()
                s["ui_hidden_cols"] = ",".join(sorted(self._user_hidden()))
                # Late import: scanner.app imports this mixin at module load.
                from ..backend.settings_store import save_settings

                save_settings(s)
            except Exception:
                logger.info("UI prefs save failed", exc_info=True)

        self._debounce("ui_prefs", 0.3, _save)

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
        spinner = ft.ProgressRing(width=14, height=14, stroke_width=2, color=c["cyan"])
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
            border_radius=RADIUS_LG,
            padding=12,
            margin=_margin_only(bottom=4),
        )
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

    def _copy_ticker_button(self, ticker: str, c: dict) -> ft.IconButton | None:
        """Icon button that copies ``ticker`` client-side (no server round-trip).

        Constructing the action needs a live page context, which background
        flushes don't have — so the button is best-effort and omitted there.
        """
        try:
            action = ft.CopyToClipboard(ticker)
        except Exception:
            logger.debug("Copy action unavailable without page context", exc_info=True)
            return None
        return ft.IconButton(
            icon=ft.Icons.CONTENT_COPY,
            icon_size=14,
            icon_color=c["text_dim"],
            tooltip=f"Copy {ticker}",
            action=action,
        )

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
                border_radius=RADIUS_SM,
                padding=_padding_only(left=8, right=8, top=3, bottom=3),
            )
            for label, color in chips
        ]
        if ticker:
            copy_btn = self._copy_ticker_button(ticker, c)
            if copy_btn is not None:
                controls.append(copy_btn)
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
        frame._news_ticker = ticker
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

    def _make_scan_placeholder(self, headline="Scanning — fetching batches…"):
        """Animated shimmer skeleton shown in the results area during a scan."""
        c = self.theme_colors
        skeleton_rows = [shimmer_row(row_height=self._row_h()) for _ in range(8)]
        spinner = ft.ProgressRing(width=36, height=36, stroke_width=3, color=c["green"])
        shimmer = ft.Shimmer(
            content=ft.Column(
                skeleton_rows,
                spacing=4,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            base_color=ft.Colors.with_opacity(0.04, ft.Colors.WHITE),
            highlight_color=ft.Colors.with_opacity(0.14, ft.Colors.WHITE),
            period=1500,
            loop=0,
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
                    shimmer,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=4,
            ),
            alignment=Alignment.CENTER,
            padding=40,
        )

    def _render_chart(self, results):
        c = self.theme_colors
        self.chart_holder.visible = bool(results)
        if not results:
            return
        from .views_charts import build_score_histogram
        from .views_layout import RAIL_W, RIGHT_W, SIDE_W

        threshold = self._score_threshold()
        # Right-aligned in the strip: size to the window but cap at 560px —
        # an uncapped strip-width canvas stretched 10 buckets into fat bars
        # with a void on the empty side.
        pw = int(getattr(self.page, "width", 0) or 1600)
        avail = max(360, min(pw - RAIL_W - SIDE_W - RIGHT_W - 440, 560))
        self.chart_bars.controls = [
            build_score_histogram(
                results,
                c,
                threshold=threshold,
                width=avail,
                height=76,
                on_bucket=self._set_threshold_from_bucket,
            )
        ]

    def _set_threshold_from_bucket(self, bucket: int):
        """Histogram click: the bucket floor (0,10,…,90) becomes min score."""
        slider = getattr(self, "threshold_slider", None)
        if slider is None:
            return
        slider.value = bucket * 10
        self.settings["min_score"] = float(bucket * 10)
        self._on_threshold_change(None)

    # ── Summary / Hero ──────────────────────────────────────────────────

    def _update_summary(self, results):
        if not results:
            for lbl in self.summary_cards.values():
                lbl.value = "—"
            return
        total = len(results)
        score_sum = 0.0
        high = 0.0
        bull = bear = 0
        for r in results:
            score = _score_of(r)
            score_sum += score
            if score > high:
                high = score
            td = r.get("trend_dir")
            if td == "Bull":
                bull += 1
            elif td == "Bear":
                bear += 1

        self.summary_cards["avg"].value = f"{score_sum / total:.1f}"
        self.summary_cards["high"].value = f"{high:.0f}"
        self.summary_cards["bull"].value = str(bull)
        self.summary_cards["bear"].value = str(bear)

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
