"""News and side-panel rendering — ``PanelsViewMixin`` for ``scanner.app.ScannerApp``.

Owns the row-level news frame and the article / research / Market Lens
side panels that open from it: panel lifecycle (load, frame, close),
article reading, web research and the extra peer/financials renderers.

The methods are mixins: they run against the ``ScannerApp`` instance (via
the MRO) and rely on state such as ``self.table_column`` /
``self.theme_colors`` and on the row helpers in ``ResultsViewMixin``
(``_find_result_row``, ``_insert_news_frame``, …).
"""

import logging
import threading

import flet as ft

from .ui_kit import (
    RADIUS_LG,
    RADIUS_MD,
    RADIUS_SM,
    _margin_only,
    _padding_only,
)

logger = logging.getLogger(__name__)


class PanelsViewMixin:
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
                        border_radius=RADIUS_MD,
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
                                border_radius=RADIUS_SM,
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
                if item.get("url"):
                    card_lines.append(
                        ft.Text(
                            "Read article →",
                            size=10,
                            color=c["cyan"],
                            on_tap=lambda e, t=ticker, u=item["url"]: (
                                self._read_article(t, u)
                            ),
                        )
                    )
                news_controls.append(
                    ft.Container(
                        content=ft.Column(card_lines, spacing=3),
                        bgcolor=c["card"],
                        border_radius=RADIUS_MD,
                        padding=10,
                        margin=_margin_only(bottom=3),
                    )
                )

        news_frame = ft.Container(
            content=ft.Column(controls=news_controls, spacing=5),
            bgcolor=c["card2"],
            border_radius=RADIUS_LG,
            padding=10,
            margin=_margin_only(bottom=4),
        )
        news_frame._news_ticker = ticker
        self._insert_news_frame(ticker, news_frame)
        if update:
            self.page.update()

    # ── Side-panel helpers (article / research / extra) ─────────────────────
    def _remove_panel(self, tag: str) -> None:
        ctrls = self.table_column.controls
        attr = f"_is_{tag}"
        for x in [x for x in ctrls if getattr(x, attr, False)]:
            ctrls.remove(x)

    def _close_panel(self, tag: str) -> None:
        self._remove_panel(tag)
        self.page.update()

    def _panel_loading(self, tag: str, label: str, place=None) -> None:
        """Drop any open panel for ``tag`` and show a spinner in its place."""
        self._remove_panel(tag)
        c = self.theme_colors
        loading = ft.Container(
            content=ft.Row(
                [
                    ft.ProgressRing(
                        width=14, height=14, stroke_width=2, color=c["cyan"]
                    ),
                    ft.Text(label, size=11, color=c["text_dim"]),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=c["card2"],
            border_radius=RADIUS_LG,
            padding=12,
            margin=_margin_only(bottom=4),
        )
        setattr(loading, f"_is_{tag}", True)
        (place or self.table_column.controls.append)(loading)
        self.page.update()

    def _panel_frame(
        self, tag: str, title: str, lines: list, spacing: int = 6
    ) -> ft.Container:
        """Shared card frame: title row with close button, then body lines."""
        c = self.theme_colors
        frame = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text(
                                title,
                                size=12,
                                weight=ft.FontWeight.BOLD,
                                color=c["cyan"],
                                expand=True,
                            ),
                            ft.IconButton(
                                icon=ft.Icons.CLOSE_ROUNDED,
                                icon_size=16,
                                icon_color=c["text_dim"],
                                tooltip="Close",
                                on_click=lambda e, t=tag: self._close_panel(t),
                            ),
                        ]
                    ),
                    *lines,
                ],
                spacing=spacing,
            ),
            bgcolor=c["card2"],
            border_radius=RADIUS_LG,
            padding=10,
            margin=_margin_only(bottom=4),
        )
        setattr(frame, f"_is_{tag}", True)
        return frame

    def _start_panel(self, load, done) -> None:
        """Run ``load`` off-thread; deliver its result to ``done`` on the UI thread."""

        def _worker():
            # load() does network/CLI I/O — it must NOT run on the UI
            # thread (done(load()) did, stalling every click for seconds).
            try:
                result = load()
            except Exception:
                logger.info("Panel load failed", exc_info=True)
                result = None
            self._safe_update(lambda: done(result))

        threading.Thread(target=_worker, daemon=True).start()

    def _read_article(self, ticker: str, url: str):
        """Fetch an article's full text (Jina Reader) and show it in place."""
        if not url:
            return
        from ..api.channels import fetch_article

        self._panel_loading(
            "article",
            "Fetching article…",
            lambda f: self._insert_news_frame(ticker, f),
        )
        self._start_panel(
            lambda: fetch_article(url), lambda text: self._show_article(ticker, text)
        )

    def _show_article(self, ticker: str, text: str | None):
        if self.active_view != "dashboard":
            return
        c = self.theme_colors
        self._remove_panel("article")
        if text:
            body = ft.Column(
                [ft.Text(text[:40000], size=11, color=c["text"], selectable=True)],
                scroll=ft.ScrollMode.AUTO,
                height=360,
                spacing=6,
            )
        else:
            body = ft.Text(
                "Article unavailable — Jina Reader returned nothing.",
                size=11,
                color=c["text_dim"],
            )
        self._insert_news_frame(ticker, self._panel_frame("article", "Article", [body]))
        self.page.update()

    def _research_ticker(self, ticker: str):
        """On-demand Exa web search for this stock (Agent-Reach channel)."""
        if getattr(self, "_research_loading", False):
            return
        self._research_loading = True
        from ..api.channels import exa_search

        self._panel_loading("research", "Searching the web…")
        self._start_panel(
            lambda: exa_search(f"{ticker} NSE stock news catalyst", 5),
            self._show_research,
        )

    def _show_research(self, results: list | None):
        self._research_loading = False
        if self.active_view != "dashboard":
            return
        c = self.theme_colors
        self._remove_panel("research")
        lines: list = []
        if results is None:
            lines.append(
                ft.Text(
                    "Exa search unavailable — requires Node.js (npx mcporter).",
                    size=11,
                    color=c["text_dim"],
                )
            )
        elif not results:
            lines.append(ft.Text("No results.", size=11, color=c["text_dim"]))
        else:
            lines.extend(
                ft.Column(
                    [
                        ft.Text(
                            r.get("title") or r.get("url", ""),
                            size=11,
                            weight=ft.FontWeight.BOLD,
                            color=c["text"],
                        ),
                        ft.Text(
                            r.get("snippet") or "",
                            size=10,
                            color=c["text_dim"],
                            max_lines=3,
                        ),
                        ft.Text(
                            r.get("url") or "",
                            size=9,
                            color=c["cyan"],
                            selectable=True,
                        ),
                    ],
                    spacing=2,
                )
                for r in results
            )
        self.table_column.controls.append(
            self._panel_frame("research", "Web research", lines)
        )
        self.page.update()

    def _show_extra(self, ticker: str, kind: str):
        """On-demand Market Lens panel: sector peers or quarterly financials."""
        if getattr(self, "_extra_loading", False):
            return
        self._extra_loading = True
        from ..api.market_lens import get_peers, get_quarterly

        self._panel_loading("extra", "Loading Market Lens…")
        self._start_panel(
            lambda: get_peers(ticker) if kind == "peers" else get_quarterly(ticker),
            lambda data: self._render_extra(kind, data),
        )

    @staticmethod
    def _extra_num(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _render_extra(self, kind: str, data):
        self._extra_loading = False
        if self.active_view != "dashboard":
            return
        c = self.theme_colors
        self._remove_panel("extra")
        title = "Sector peers" if kind == "peers" else "Quarterly financials"
        lines: list = []
        if not data:
            lines.append(
                ft.Text(
                    "Market Lens unavailable for this ticker.",
                    size=11,
                    color=c["text_dim"],
                )
            )
        elif kind == "peers":
            for p in data:
                chg = self._extra_num(p.get("pchange"))
                price = self._extra_num(p.get("closePrice"))
                pe = self._extra_num(p.get("peRatio"))
                chg_col = (
                    c["green"]
                    if chg is not None and chg > 0
                    else c["red"]
                    if chg is not None and chg < 0
                    else c["text_dim"]
                )
                lines.append(
                    ft.Row(
                        [
                            ft.Text(
                                str(p.get("ticker") or "?"),
                                size=11,
                                weight=ft.FontWeight.BOLD,
                                color=c["text"],
                                width=90,
                            ),
                            ft.Text(
                                str(p.get("name") or ""),
                                size=10,
                                color=c["text_dim"],
                                expand=True,
                                max_lines=1,
                            ),
                            ft.Text(
                                f"{price:,.2f}" if price else "—",
                                size=11,
                                color=c["text"],
                            ),
                            ft.Text(
                                f"{chg:+.2f}%" if chg is not None else "—",
                                size=11,
                                color=chg_col,
                            ),
                            ft.Text(
                                f"PE {pe:.1f}" if pe else "PE —",
                                size=10,
                                color=c["text_dim"],
                            ),
                        ],
                        spacing=8,
                    )
                )
        else:
            # Newest-first rows; ▲/▼ compares EPS with the prior quarter.
            header = ft.Row(
                [
                    ft.Text("Quarter", size=9, color=c["text_faint"], width=90),
                    ft.Text("Income", size=9, color=c["text_faint"]),
                    ft.Text("Net profit", size=9, color=c["text_faint"]),
                    ft.Text("EPS", size=9, color=c["text_faint"]),
                ],
                spacing=8,
            )
            lines.append(header)
            for i, q in enumerate(data):
                inc = self._extra_num(q.get("totalIncome"))
                profit = self._extra_num(q.get("netProfitLoss"))
                eps = self._extra_num(q.get("eps"))
                eps_txt = "—"
                eps_col = c["text_dim"]
                if eps is not None:
                    eps_txt = f"{eps:.2f}"
                    if i + 1 < len(data):
                        prev = self._extra_num(data[i + 1].get("eps"))
                        if prev:
                            if eps > prev:
                                eps_txt += " ▲"
                                eps_col = c["green"]
                            elif eps < prev:
                                eps_txt += " ▼"
                                eps_col = c["red"]
                lines.append(
                    ft.Row(
                        [
                            ft.Text(
                                str(q.get("date") or ""),
                                size=11,
                                color=c["text_dim"],
                                width=90,
                            ),
                            ft.Text(
                                f"{inc:,.0f}" if inc is not None else "—",
                                size=11,
                                color=c["text"],
                            ),
                            ft.Text(
                                f"{profit:,.0f}" if profit is not None else "—",
                                size=11,
                                color=c["text"],
                            ),
                            ft.Text(eps_txt, size=11, color=eps_col),
                        ],
                        spacing=8,
                    )
                )
        self.table_column.controls.append(
            self._panel_frame(
                "extra", f"{title} · figures as reported", lines, spacing=4
            )
        )
        self.page.update()

