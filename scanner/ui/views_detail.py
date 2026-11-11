"""Per-stock detail view — ``DetailViewMixin`` for ``scanner.app.ScannerApp``.

Replaces the results table with a full-width detail panel for one stock:
header with score/rating/entry chips, price chart, score breakdown, key
signals, trade reasons and the institutional-positioning tiles, plus the
background loads that feed them.

The methods are mixins: they run against the ``ScannerApp`` instance (via
the MRO) and rely on state such as ``self.table_column`` /
``self.theme_colors``, on ``PanelsViewMixin`` (research/extra panels) and
on the row helpers in ``ResultsViewMixin`` (``_find_result_row``, …).
"""

import logging
import threading

import flet as ft
from flet.controls.alignment import Alignment

from ..shared.constants import score_of as _score_of
from ..shared.detail_specs import fmt_cr, fmt_pct, signal_specs
from .ui_kit import (
    FS_2XL,
    FS_BASE,
    FS_LG,
    FS_MD,
    FS_XS,
    RADIUS_LG,
    RADIUS_SM,
    _border_all,
    _card_shadow,
    _margin_only,
    _padding_only,
    rating_color,
    score_color,
)

logger = logging.getLogger(__name__)


class DetailViewMixin:
    # ── Per-stock detail view ────────────────────────────────────────

    def _show_stock_detail(self, ticker: str):
        """Replace the table with a full-width detail panel for ``ticker``."""
        if self.active_view != "dashboard":
            return
        prev = getattr(self, "_detail_ticker", None)
        self._detail_ticker = ticker
        if prev != ticker:
            # Fresh detail open → allow one background reload attempt.
            # Per-ticker slots: a slow earlier thread finishing for another
            # ticker must not clobber this ticker's attempt/done state.
            self._inst_attempted.discard(ticker)
            self._inst_done.discard(ticker)
        c = self.theme_colors

        row = self._find_result_row(ticker)
        if row is None:
            return

        # Build the detail panel
        panel = self._build_detail_panel(row, c)

        # Replace table content with the detail panel
        self.table_column.controls.clear()
        self.table_column.controls.append(panel)
        # Clear the row pool so _back_to_results takes the full-rebuild path
        # (streaming would only patch pooled rows never re-added to the pane).
        if hasattr(self, "_row_pool"):
            self._row_pool.clear()
        if hasattr(self, "_row_cells"):
            self._row_cells.clear()
        holder = getattr(self, "header_holder", None)
        if holder is not None:
            holder.controls = []
        self.pagination_bar.visible = False
        self.page.update()

    # ── Institutional positioning (detail panel) ──────────────────────────

    def _inst_tile(self, label, value, total, recent, c, state=None):
        """One compact tile: label → [arrow] value → total/recent sub-line."""
        arrow, arrow_c = "", c["text_dim"]
        if state == "loading":
            sub, sub_c = "loading…", c["text_dim"]
        elif state == "missing":
            sub, sub_c = "n/a", c["text_dim"]
        elif total is None:
            sub, sub_c = "no history", c["text_dim"]
        elif total == 0:
            sub, sub_c = "No change", c["text_dim"]
        else:
            up = total > 0
            arrow, arrow_c = ("▲" if up else "▼"), (c["green"] if up else c["red"])
            sub = f"total {fmt_pct(total)}"
            if recent is not None:
                sub += f" · recent {fmt_pct(recent)}"
            sub_c = arrow_c
        val_row = ft.Row(
            ([ft.Text(arrow, size=10, color=arrow_c)] if arrow else [])
            + [ft.Text(value, size=13, weight=ft.FontWeight.BOLD, color=c["text"])],
            spacing=3,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(label, size=9, color=c["text_faint"]),
                    val_row,
                    ft.Text(sub, size=9, color=sub_c),
                ],
                spacing=2,
            ),
            bgcolor=c["card2"],
            border_radius=RADIUS_SM,
            border=_border_all(1, c["border"]),
            width=118,
            padding=_padding_only(left=10, right=10, top=6, bottom=6),
        )

    def _promoter_tile_data(self, ticker: str, pct):
        """{pct, total, recent} via today's snapshot + our history; None if no pct."""
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            return None
        if not pct:
            return None
        try:
            from ..api.indian_fundamentals import (
                promoter_position_change,
                record_promoter_snapshot,
            )

            series = record_promoter_snapshot(ticker, pct)
            total, recent = promoter_position_change(series)
            return {"pct": pct, "total": total, "recent": recent}
        except Exception:
            logger.debug("Promoter snapshot failed for %s", ticker, exc_info=True)
            return {"pct": pct, "total": None, "recent": None}

    def _start_institutional_load(self, ticker: str) -> None:
        """Fetch cash-flow history + shareholding once per open."""
        if ticker in self._inst_attempted:
            return
        self._inst_attempted.add(ticker)

        def _run():
            try:
                from ..api.indian_market import fetch_fii_dii_history

                fetch_fii_dii_history()
            except Exception:
                logger.info("Market institutional fetch failed", exc_info=True)
            try:
                row = self._find_result_row(ticker)
                if row is not None:
                    shp = row.get("_shareholding")
                    if shp is None:
                        from ..api.indian_fundamentals import (
                            fetch_shareholding_pattern,
                        )

                        shp = fetch_shareholding_pattern(ticker)
                        if shp:
                            row["_shareholding"] = shp
                    has_promo = bool((shp or {}).get("series", {}).get("promoters"))
                    if not row.get("_promoter_holding") and not has_promo:
                        from ..api.indian_fundamentals import (
                            fetch_trendlyne_fundamentals,
                        )

                        fund = fetch_trendlyne_fundamentals(ticker)
                        if fund and fund.promoter_holding:
                            row["_promoter_holding"] = round(fund.promoter_holding, 2)
            except Exception:
                logger.info("Shareholding fetch failed for %s", ticker, exc_info=True)
            # Always settle the tiles (data → real values, failure → n/a) and
            # re-render once; the attempted guard stops any retry loop.
            self._inst_done.add(ticker)

            def _re():
                if getattr(self, "_detail_ticker", None) == ticker:
                    self._show_stock_detail(ticker)

            try:
                self._safe_update(_re)
            except Exception:
                logger.debug("Institutional re-render skipped", exc_info=True)

        t = threading.Thread(target=_run, daemon=True)
        self._inst_thread = t
        t.start()

    def _institutional_card(self, ticker: str, c: dict) -> ft.Container:
        """Shareholding % and promoter % as per-stock tiles."""
        flow = None
        try:
            from ..api.indian_market import fetch_fii_dii_history, flow_window_pcts

            flow = fetch_fii_dii_history(cache_only=True)
        except Exception:
            # a later raise keeps the reads that already succeeded above
            logger.debug("Institutional cache read failed", exc_info=True)
        row = self._find_result_row(ticker) or {}
        shp = row.get("_shareholding")
        pct = row.get("_promoter_holding")
        if not flow or shp is None:
            self._start_institutional_load(ticker)
        done = ticker in self._inst_done
        state = "missing" if done else "loading"
        pending = "—" if done else "…"

        series = (shp or {}).get("series", {})

        def _flow_tile(label, key):
            if flow:
                total, recent = flow_window_pcts(flow, key)
                return self._inst_tile(
                    label, fmt_cr(flow[0][key]), total, recent, c
                )
            return self._inst_tile(label, pending, None, None, c, state=state)

        def _pct_tile(label, key, snap_pct=None, flow_key=None, marker_key=None):
            e = series.get(key)
            if e:
                return self._inst_tile(
                    label, f"{e['latest']:.1f}%", e["total"], e["recent"], c
                )
            snap = self._promoter_tile_data(ticker, snap_pct) if snap_pct else None
            if snap:
                return self._inst_tile(
                    label, f"{snap['pct']:.1f}%", snap["total"], snap["recent"], c
                )
            if flow_key and flow:
                return _flow_tile(label, flow_key)
            net = row.get(marker_key) if marker_key else None
            if net is not None:
                # scan-time NSE market net — the same markers the FII/DII
                # filter keys off, so a filtered row always renders a tile
                return self._inst_tile(label, fmt_cr(net), None, None, c)
            return self._inst_tile(label, pending, None, None, c, state=state)

        tiles = [
            _pct_tile(
                "FII", "foreign_institutions", flow_key="fii", marker_key="_fii_net"
            ),
            _pct_tile(
                "DII", "domestic_institutions", flow_key="dii", marker_key="_dii_net"
            ),
            _pct_tile("Promoters", "promoters", snap_pct=pct),
        ]

        # FPI flow and participant OI are published market-wide only — same
        # numbers every stock — so they stay out of this per-stock card.
        hint = "Shareholding"
        if shp and shp.get("quarter"):
            hint += f" · {shp['quarter']}"
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "Institutional Positioning",
                        size=FS_MD,
                        weight=ft.FontWeight.BOLD,
                        color=c["cyan"],
                    ),
                    ft.Row(tiles, spacing=6, wrap=True),
                    ft.Text(hint, size=FS_XS, color=c["text_faint"]),
                ],
                spacing=6,
            ),
            bgcolor=c["card"],
            border_radius=RADIUS_LG,
            border=_border_all(1, c["border"]),
            shadow=_card_shadow(),
            padding=12,
            margin=_margin_only(bottom=8),
        )

    def _build_detail_panel(self, row: dict, c: dict) -> ft.Container:
        """Build the full-width detail panel for a single stock."""
        from ..shared.trade_reasons import build_trade_reasons
        from .views_charts import build_price_chart, build_score_breakdown

        ticker = row.get("ticker", "?")
        total = _score_of(row)
        rating = row.get("combined_rating", "POOR")
        entry = bool(row.get("entry_signal"))
        ml_sector = str(row.get("_ml_sector") or "").strip()

        # ── Header with back button ──────────────────────────────────
        title_col = ft.Column(
            [
                ft.Text(
                    ticker,
                    size=FS_2XL,
                    weight=ft.FontWeight.BOLD,
                    color=c["text"],
                ),
                *(
                    [ft.Text(ml_sector, size=FS_XS, color=c["text_faint"])]
                    if ml_sector
                    else []
                ),
            ],
            expand=True,
            spacing=0,
        )
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
                    title_col,
                    ft.Container(
                        content=ft.Text(
                            f"{total:.0f}",
                            size=20,
                            weight=ft.FontWeight.BOLD,
                            color=score_color(total, c),
                        ),
                        bgcolor=ft.Colors.with_opacity(0.15, score_color(total, c)),
                        border_radius=RADIUS_SM,
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
                        border_radius=RADIUS_SM,
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
                        border_radius=RADIUS_SM,
                        padding=_padding_only(left=8, right=8, top=3, bottom=3),
                    ),
                    ft.IconButton(
                        icon=ft.Icons.SEARCH_ROUNDED,
                        icon_size=16,
                        icon_color=c["cyan"],
                        tooltip="Web research (Exa)",
                        on_click=lambda _, t=ticker: self._research_ticker(t),
                    ),
                    ft.IconButton(
                        icon=ft.Icons.GROUP_ROUNDED,
                        icon_size=16,
                        icon_color=c["cyan"],
                        tooltip="Sector peers (Market Lens)",
                        on_click=lambda _, t=ticker: self._show_extra(t, "peers"),
                    ),
                    ft.IconButton(
                        icon=ft.Icons.SHOW_CHART_ROUNDED,
                        icon_size=16,
                        icon_color=c["cyan"],
                        tooltip="Quarterly financials (Market Lens)",
                        on_click=lambda _, t=ticker: self._show_extra(t, "quarterly"),
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            # Subtle left-to-right lift — the one gradient inside detail.
            gradient=ft.LinearGradient(
                begin=Alignment.CENTER_LEFT,
                end=Alignment.CENTER_RIGHT,
                colors=[c["card"], c["card2"]],
            ),
            border_radius=RADIUS_LG,
            border=_border_all(1, c["border"]),
            shadow=_card_shadow(),
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
            (label, value, c[role]) for label, value, role in signal_specs(row)
        ]

        signal_items = []
        for label, value, color in signals_data:
            signal_items.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(label, size=FS_XS, color=c["text_faint"]),
                            ft.Text(
                                value,
                                size=FS_LG,
                                weight=ft.FontWeight.BOLD,
                                color=color,
                            ),
                        ],
                        spacing=2,
                    ),
                    bgcolor=c["card"],
                    border_radius=RADIUS_SM,
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
                        size=FS_MD,
                        weight=ft.FontWeight.BOLD,
                        color=c["cyan"],
                    ),
                    ft.Row(signal_items, spacing=6, wrap=True),
                ],
                spacing=6,
            ),
            bgcolor=c["card"],
            border_radius=RADIUS_LG,
            border=_border_all(1, c["border"]),
            shadow=_card_shadow(),
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
                        ft.Text(r, size=FS_BASE, color=c["text"], expand=True),
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
                        size=FS_MD,
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
                            "No trade reasons available.",
                            size=FS_BASE,
                            color=c["text_dim"],
                        ),
                    ]
                ),
                spacing=5,
            ),
            bgcolor=c["card"],
            border_radius=RADIUS_LG,
            border=_border_all(1, c["border"]),
            shadow=_card_shadow(),
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
                    # Own full-width section: independent of "Why this trade?"
                    # in the right column above.
                    self._institutional_card(ticker, c),
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
