"""Results rendering — ``ResultsViewMixin`` for ``scanner.app.ScannerApp``.

Builds and refreshes everything that lives in the main results area after a
scan: the paginated table (header + data rows), row-level news expansion,
the score chart, summary cards and the hero status line.

The methods are mixins: they run against the ``ScannerApp`` instance (via
the MRO) and rely on state such as ``self.all_results`` / ``self.theme_colors``
and on the controls created by ``LayoutViewMixin._build_main_area``
(``self.table_column``, ``self.chart_bars``, ...).
"""

import threading
from datetime import datetime, timedelta

import flet as ft
from flet.controls.alignment import Alignment

from .report import _parse_date, _sentiment
from .ui_kit import (
    RESULT_COLS,
    _border_all,
    _margin_only,
    _padding_only,
    _score_of,
)


class ResultsViewMixin:
    def _display_results(self, results):
        with self._results_lock:
            self.all_results = list(results)
            self.filtered_results = [r for r in results if self._row_matches_filters(r)]
        self.current_page = 0
        self._render_current_page()

    def _render_current_page(self):
        c = self.theme_colors
        with self._results_lock:
            results = list(self.all_results)
            shown = list(self._visible_results())

        if self.sort_col is not None and shown:
            try:
                key_fn = self._get_sort_key(self.sort_col)
                shown = sorted(shown, key=key_fn, reverse=self.sort_reverse)
            except Exception:
                pass

        self.table_column.controls.clear()

        if not shown:
            if self.scanning and not results:
                live = (getattr(self.progress_label, "value", "") or "").strip()
                headline = live if live and live not in ("Ready", "Done", "Stopped") else "Scanning — fetching batches…"
                self.table_column.controls.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Icon(ft.Icons.SHOW_CHART, size=48, color=c["green"]),
                            ft.Text(headline, size=12, weight=ft.FontWeight.BOLD, color=c["green"]),
                            ft.Text("First results appear after ~1 batch (~20s)", size=11, color=c["text_dim"]),
                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
                        alignment=Alignment.CENTER,
                        padding=30,
                    )
                )
            else:
                has_active_filter = self._is_filter_active()
                msg = "No results match your filter." if results and has_active_filter else "No results found."
                empty_body = [
                    ft.Text(msg, size=13, color=c["red"] if not results else c["text_dim"]),
                ]
                if results and has_active_filter:
                    empty_body.append(
                        ft.TextButton(
                            content=ft.Text("Reset filters", size=12),
                            on_click=self._reset_filters,
                            style=ft.ButtonStyle(color=c["cyan"]),
                        )
                    )
                self.table_column.controls.append(
                    ft.Container(
                        content=ft.Column(empty_body, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
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
            page_shown = shown[start:start + page_size]

            header_row = self._make_header_row(c)
            self.table_column.controls.append(header_row)

            for rank, r in enumerate(page_shown, start + 1):
                score = _score_of(r)
                is_above = score >= threshold
                row_bg = c["card"] if is_above else (c["row_alt"] if rank % 2 else c["main_bg"])
                row = self._make_data_row(r, rank, c, row_bg, threshold)
                self.table_column.controls.append(row)

            self.pagination_bar.visible = bool(shown and len(shown) > page_size)
            self.page_label.value = f"Page {self.current_page+1} / {total_pages}  ({len(shown)} stocks)"

        if results:
            threshold = self.settings.get("min_score", 50)
            filter_parts = []
            if self.filter_text:
                filter_parts.append(f"'{self.filter_text}'")
            rating = self._rating_filter()
            if rating != "ALL":
                filter_parts.append(f"rating {rating.title()}")
            suffix = f"  |  filter: {', '.join(filter_parts)} ({len(shown)})" if filter_parts else ""
            self.result_count_label.value = f"{len(results)} scanned  |  {len([r for r in results if _score_of(r) >= threshold])} above {threshold:.0f}+{suffix}"
        else:
            self.result_count_label.value = "no scan yet"

        self._update_summary(results)
        self._update_hero_status(results)
        self._render_topicks(results[:5])
        self._render_chart(results)
        self.page.update()

    def _make_header_row(self, c):
        headers = []
        for idx, (text, width) in enumerate(RESULT_COLS):
            is_sorted = self.sort_col == idx
            arrow = " ▲" if is_sorted and not self.sort_reverse else (" ▼" if is_sorted else "")
            color = c["cyan"] if not is_sorted else c["green"]
            headers.append(
                ft.Container(
                    content=ft.Text(f"{text}{arrow}", size=9, weight=ft.FontWeight.BOLD, color=color),
                    width=width,
                    on_click=lambda e, i=idx: self._on_sort(i),
                    ink=True,
                )
            )
        return ft.Container(
            content=ft.Row(controls=headers, spacing=2),
            bgcolor=c["card2"],
            border_radius=10,
            border=_border_all(1, c["border"]),
            height=32,
            padding=_padding_only(left=4, right=4, top=4, bottom=4),
            margin=_margin_only(bottom=6),
        )

    def _make_data_row(self, r, rank, c, bg, threshold):
        total = _score_of(r)
        ticker = r.get("ticker", "?")
        trend_dir = r.get("trend_dir") or ""
        is_above = total >= threshold
        cols = [
            (str(rank), c["text_dim"], 11, False),
            (ticker, c["green"] if is_above else c["text"], 12, True),
            (f'{total:.0f}', c["green"] if total >= 70 else c["lime"] if total >= 50 else c["orange"] if total >= 30 else c["red"], 13, True),
            (r.get("combined_rating", "POOR"), {"EXCELLENT": c["green"], "GOOD": c["lime"], "MODERATE": c["orange"]}.get(r.get("combined_rating"), c["red"]), 10, True),
            ("YES" if r.get("entry_signal") else "--", c["green"] if r.get("entry_signal") else c["text_dim"], 10, True),
            (f'₹{r.get("close", 0) or 0:.0f}', c["text"], 11, True),
            (self._ma_text(r), self._ma_color(r), 10, False),
            (f'{r.get("trend", 0) or 0:.0f}', c["green"], 10, False),
            (f'{r.get("momentum", 0) or 0:.0f}', c["cyan"], 10, False),
            (f'{r.get("rsi", 0) or 0:.0f}', c["blue"], 10, False),
            (f'{r.get("macd", 0) or 0:.0f}', "#aa88ff", 10, False),
            (f'{r.get("volume", 0) or 0:.0f}', c["orange"], 10, False),
            (f'{r.get("rel_str", 0) or 0:.0f}', c["lime"], 10, False),
            (f'{r.get("fundamentals", 0) or 0:.0f}', "#ffe600", 10, False),
            (f'{r.get("pc1m", 0) or 0:+.1f}%', c["green"] if (r.get("pc1m", 0) or 0) > 0 else c["red"], 10, False),
            (("^ " if trend_dir == "Bull" else "v ") + (trend_dir or "?"), c["green"] if trend_dir == "Bull" else c["red"], 10, False),
            (f'{r.get("adx_val", 0) or 0:.0f}', c["text"], 10, False),
            ("Chop" if r.get("is_sideways") else "OK", c["orange"] if r.get("is_sideways") else c["green"], 10, False),
        ]

        controls = []
        ticker_cell = None
        for idx, ((text, color, size, bold), (_col_name, width)) in enumerate(zip(cols, RESULT_COLS)):
            w = ft.Text(
                text, size=size,
                weight=ft.FontWeight.BOLD if bold else ft.FontWeight.NORMAL,
                color=color,
            )
            cell = ft.Container(content=w, width=width)
            if idx == 1:
                ticker_cell = cell
            controls.append(cell)

        ticker_cell.on_click = lambda e, t=ticker: self._toggle_stock_news(t)
        ticker_cell.tooltip = "Click for news & sentiment"

        return ft.Container(
            content=ft.Row(controls=controls, spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=bg,
            border_radius=8,
            border=_border_all(1, c["border"]) if is_above else None,
            height=32,
            padding=_padding_only(left=4, right=4),
            margin=_margin_only(bottom=1),
        )

    def _on_sort(self, col_idx):
        if self.sort_col == col_idx:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_col = col_idx
            self.sort_reverse = col_idx not in (0, 1)
        self.current_page = 0
        self._render_current_page()
        self._scroll_to_top()

    def _get_sort_key(self, col_idx):
        rating_order = {"EXCELLENT": 4, "GOOD": 3, "MODERATE": 2, "POOR": 1, "WEAK": 0}
        def _ma_rank(r):
            if r.get("ma_crossed_above"):
                return 2
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
            if ms is not None:
                ms.scroll_to(offset=0, duration=200)
        except Exception:
            pass

    def _change_page(self, delta):
        shown = self._visible_results()
        total_pages = max(1, (len(shown) + self.page_size - 1) // self.page_size)
        self.current_page = max(0, min(self.current_page + delta, total_pages - 1))
        self._render_current_page()
        self._scroll_to_top()

    def _on_page_size_change(self, e):
        try:
            self.page_size = int(self.page_size_dd.value or "100")
        except ValueError:
            self.page_size = 100
        self.current_page = 0
        self._render_current_page()
        self._scroll_to_top()

    def _load_all_pages(self):
        total = len(self._visible_results())
        self.page_size = min(500, total) if total > 0 else 500
        if str(self.page_size) in self.page_size_options:
            self.page_size_dd.value = str(self.page_size)
        self.current_page = 0
        self._render_current_page()
        self._scroll_to_top()

    def _toggle_stock_news(self, ticker):

        def _fetch_news():
            try:
                import yfinance as yf
                news_items = []
                for suffix in (".NS", ".BO"):
                    try:
                        items = yf.Ticker(f"{ticker}{suffix}").news or []
                    except Exception:
                        items = []
                    if items:
                        news_items = items
                        break
                cutoff = datetime.now() - timedelta(days=60)
                parsed = []
                for item in news_items[:10]:
                    content = item.get("content", item)
                    title = content.get("title", "")
                    if not title:
                        continue
                    summary = content.get("summary", "")
                    pub_date = content.get("pubDate", "")
                    dt = _parse_date(pub_date) if pub_date else None
                    if dt is not None and dt < cutoff:
                        continue
                    provider = content.get("provider", {})
                    prov_name = provider.get("displayName", "") if isinstance(provider, dict) else ""
                    sentiment = _sentiment(title, summary)
                    parsed.append({
                        "title": title,
                        "summary": (summary[:150] + "...") if len(summary) > 150 else summary,
                        "date": pub_date[:10] if pub_date else "",
                        "provider": prov_name,
                        "sentiment": sentiment,
                    })
                self._safe_update(lambda: self._show_news(ticker, parsed))
            except Exception:
                self._safe_update(lambda: self._show_news(ticker, []))

        threading.Thread(target=_fetch_news, daemon=True).start()

    def _show_news(self, ticker, items):
        c = self.theme_colors
        ctrls = self.table_column.controls
        for ctrl in [x for x in ctrls if getattr(x, "_news_ticker", None) == ticker]:
            ctrls.remove(ctrl)
            self.page.update()
            return

        for ctrl in [x for x in ctrls if hasattr(x, "_news_ticker")]:
            ctrls.remove(ctrl)

        news_controls = []
        if not items:
            news_controls.append(ft.Text("No recent news found.", size=11, color=c["text_dim"]))
        else:
            good = sum(1 for i in items if i["sentiment"] == "Good")
            bad = sum(1 for i in items if i["sentiment"] == "Bad")
            neu = len(items) - good - bad
            news_controls.append(
                ft.Text(f"{good} Good  |  {bad} Bad  |  {neu} Neutral", size=11, weight=ft.FontWeight.BOLD, color=c["lime"])
            )
            for item in items:
                sent = item["sentiment"]
                sent_color = {"Good": c["green"], "Bad": c["red"], "Neutral": c["text_dim"]}[sent]
                sent_bg = {"Good": c["chip_good"], "Bad": c["chip_bad"], "Neutral": c["card"]}[sent]
                meta = f"{item['date']}  {item['provider']}" if item['provider'] else item['date']
                card_lines = [
                    ft.Row([
                        ft.Container(
                            content=ft.Text(sent, size=9, weight=ft.FontWeight.BOLD, color=sent_color),
                            bgcolor=sent_bg, border_radius=6,
                            padding=_padding_only(left=5, right=5, top=1, bottom=1),
                        ),
                        ft.Text(meta, size=9, color=c["text_dim"]),
                    ], spacing=6),
                    ft.Text(item["title"], size=11, weight=ft.FontWeight.BOLD, color=c["text"], max_lines=2),
                ]
                if item["summary"]:
                    card_lines.append(ft.Text(item["summary"], size=10, color=c["text_dim"], max_lines=2))
                news_controls.append(
                    ft.Container(
                        content=ft.Column(card_lines, spacing=2),
                        bgcolor=c["card"],
                        border_radius=8,
                        padding=8,
                        margin=_margin_only(bottom=2),
                    )
                )

        news_frame = ft.Container(
            content=ft.Column(controls=news_controls, spacing=4),
            bgcolor=c["card2"],
            border_radius=10,
            padding=8,
            margin=_margin_only(bottom=4),
        )
        news_frame._news_ticker = ticker

        insert_at = None
        for i, ctrl in enumerate(ctrls):
            content = getattr(ctrl, "content", None)
            if isinstance(content, ft.Row):
                cells = content.controls or []
                if len(cells) > 1:
                    cell = cells[1]
                    inner = getattr(cell, "content", cell)
                    if isinstance(inner, ft.Text) and inner.value == ticker:
                        insert_at = i + 1
                        break
        if insert_at is None:
            ctrls.append(news_frame)
        else:
            ctrls.insert(insert_at, news_frame)
        self.page.update()

    def _render_chart(self, results):
        c = self.theme_colors
        if not results:
            self.chart_card.visible = False
            return
        top = sorted(results, key=_score_of, reverse=True)[:50]
        peak = max(_score_of(r) for r in top) or 1
        bars = []
        for r in top:
            s = _score_of(r)
            color = (c["green"] if s >= 70 else c["lime"] if s >= 50
                     else c["orange"] if s >= 30 else c["red"])
            bars.append(
                ft.Container(
                    expand=True,
                    height=max(4, round(s / peak * 64)),
                    bgcolor=color,
                    border_radius=3,
                    tooltip=f'{r.get("ticker", "?")}  {s:.0f}',
                )
            )
        self.chart_bars.controls = bars
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
        passed = len([r for r in results if _score_of(r) >= threshold])
        avg = sum(_score_of(r) for r in results) / total if total else 0
        high = max((_score_of(r) for r in results), default=0)
        bull = len([r for r in results if r.get("trend_dir") == "Bull"])
        bear = len([r for r in results if r.get("trend_dir") == "Bear"])
        entry = len([r for r in results if r.get("entry_signal")])
        try:
            from . import data_fetcher
            dead_skips = data_fetcher.negative_cache_skip_count()
        except Exception:
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
            self.hero_sub.color = c["text_dim"]
            self.hero_sub.size = 12
