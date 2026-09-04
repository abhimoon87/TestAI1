"""Dashboard shell builders — ``LayoutViewMixin`` for ``scanner.app.ScannerApp``.

Builds the four static panes of the app window (rail, sidebar, main area,
right panel) plus the small widgets they share (styled dropdowns, cache
cards, summary cards, top-pick cards).

The methods are mixins: they run against the ``ScannerApp`` instance (via
the MRO) and rely only on instance state such as ``self.theme_colors`` and
on the control attributes they create (``self.universe_dd`` etc.), which
the rest of the app reads after ``_build_ui()`` has run.
"""

import flet as ft
from flet.controls.alignment import Alignment

from .ui_kit import (
    _border_all,
    _card_shadow,
    _glass_bg,
    _glass_border,
    _margin_only,
    _neon_glow,
    _padding_only,
    _score_of,
)
from .universes import UNIVERSES


class LayoutViewMixin:
    def _build_rail(self) -> ft.Container:
        # Every rail item is a 52px unit: [4px pill slot, 4px gap, 44px icon],
        # centered as one block — so all icons share a single vertical axis.
        c = self.theme_colors

        def rail_icon(kind, accent, on_click):
            if kind == "home":
                icon = ft.Icons.HOME
            elif kind == "gear":
                icon = ft.Icons.SETTINGS
            elif kind == "play":
                icon = ft.Icons.PLAY_ARROW
            elif kind == "chart":
                icon = ft.Icons.INSIGHTS
            elif kind == "sun":
                icon = ft.Icons.LIGHT_MODE
            elif kind == "moon":
                icon = ft.Icons.DARK_MODE
            else:
                icon = ft.Icons.CIRCLE
            return ft.Container(
                content=ft.Icon(icon, color=accent, size=22),
                width=44, height=44,
                border_radius=22,
                bgcolor=c["card"],
                alignment=Alignment.CENTER,
                on_click=lambda e: on_click(),
                ink=True,
            )

        def rail_unit(icon, slot):
            return ft.Row(
                controls=[slot, icon],
                spacing=4,
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )

        def pill_slot():
            return ft.Container(
                width=4, height=26, border_radius=2,
                bgcolor=c["neon"], shadow=_neon_glow(c["neon"], blur=8),
            )

        def empty_slot():
            return ft.Container(width=4)

        self._rail_pills = {}
        home_pill = pill_slot()
        home_pill.visible = self.active_view == "dashboard"
        self._rail_pills["dashboard"] = home_pill
        backtest_pill = pill_slot()
        backtest_pill.visible = self.active_view == "backtest"
        self._rail_pills["backtest"] = backtest_pill

        logo = ft.Container(
            content=ft.Text("ABHI", color="white", size=11, weight=ft.FontWeight.BOLD),
            width=44, height=44, border_radius=22,
            bgcolor=c["purple"],
            alignment=Alignment.CENTER,
            shadow=_neon_glow(c["purple"], blur=16),
        )

        theme_icon_kind = "sun" if self.current_theme == "dark" else "moon"
        theme_btn = rail_icon(theme_icon_kind, c["orange"], self._switch_theme)

        rail_items = ft.Column(
            controls=[
                rail_unit(rail_icon("home", c["neon"], lambda: self._show_view("dashboard")), home_pill),
                rail_unit(rail_icon("gear", c["purple"], self._show_settings), empty_slot()),
                rail_unit(rail_icon("chart", c["cyan"], lambda: self._show_view("backtest")), backtest_pill),
                ft.Container(height=10),
                rail_unit(rail_icon("play", c["green"], self._on_action_click), empty_slot()),
            ],
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        return ft.Container(
            content=ft.Column(
                controls=[
                    rail_unit(logo, empty_slot()),
                    ft.Container(height=18),
                    rail_items,
                    ft.Container(expand=True),
                    rail_unit(theme_btn, empty_slot()),
                    ft.Container(height=12),
                ],
                spacing=0,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            width=64,
            bgcolor=c["rail_bg"],
            padding=_padding_only(top=14),
        )

    def _styled_dropdown(self, options, value, on_select=None) -> ft.Dropdown:
        """Full-width modern dropdown matching the app theme."""
        c = self.theme_colors
        return ft.Dropdown(
            options=[ft.dropdown.Option(v) for v in options],
            value=value,
            expand=True, height=46, text_size=13,
            bgcolor=c["option_bg"], color=c["text"],
            border_color=c["border"], border_width=1,
            border_radius=10,
            focused_border_color=c["purple"],
            content_padding=_padding_only(left=12, right=8, top=8, bottom=8),
            menu_height=260,
            on_select=on_select,
        )

    def _cache_card(self, title, status_lbl, clear_btn) -> ft.Container:
        """Labeled cache row: title + clear action on top, status below."""
        c = self.theme_colors
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text(title, size=11, weight=ft.FontWeight.BOLD, color=c["text"]),
                    ft.Container(expand=True),
                    clear_btn,
                ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                status_lbl,
            ], spacing=2),
            bgcolor=_glass_bg(),
            border=_glass_border(),
            border_radius=10,
            shadow=_card_shadow(),
            padding=_padding_only(left=10, right=4, top=6, bottom=6),
        )

    def _build_sidebar(self) -> ft.Container:
        c = self.theme_colors

        section = lambda t: ft.Container(
            content=ft.Text(t.upper(), size=9, weight=ft.FontWeight.BOLD, color=c["text_faint"]),
            padding=_padding_only(left=16, top=14, bottom=4),
        )
        field = lambda ctrl: ft.Container(ctrl, padding=_padding_only(left=16, right=16))

        self.universe_dd = self._styled_dropdown(
            list(UNIVERSES.keys()), "NIFTY 50",
            on_select=self._on_universe_change,
        )
        self.universe_count_label = ft.Text(
            f"{len(UNIVERSES['NIFTY 50'])} stocks", size=10, color=c["text_dim"]
        )

        self.timeframe_dd = self._styled_dropdown(["Daily", "Weekly", "Monthly"], "Daily")
        self.period_dd = self._styled_dropdown(["6 Months", "1 Year", "2 Years"], "1 Year")
        self.trend_filter_dd = self._styled_dropdown(["All", "Bullish Only", "Bearish Only"], "All")
        self.rating_filter_dd = self._styled_dropdown(
            ["All", "Excellent", "Good", "Moderate", "Poor"], "All",
            on_select=self._on_rating_change,
        )
        self.threshold_slider = ft.Slider(
            min=0, max=100, value=50, divisions=20, expand=True,
            active_color=c["purple"], inactive_color=c["progress_bg"],
            on_change=self._on_threshold_change,
        )
        self.threshold_label = ft.Text("50+", size=13, weight=ft.FontWeight.BOLD, color=c["pink"])
        threshold_chip = ft.Container(
            content=self.threshold_label,
            bgcolor=c["card2"],
            border=_border_all(1, c["border"]),
            border_radius=8,
            padding=_padding_only(left=10, right=10, top=2, bottom=2),
        )

        self.action_btn_label = ft.Text("▶  RUN SCAN", size=14, weight=ft.FontWeight.BOLD)
        self.action_btn = ft.ElevatedButton(
            content=self.action_btn_label, expand=True, height=46,
            bgcolor=c["green"], color="#052e16",
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12)),
            on_click=self._on_action_click,
        )
        self.progress_bar = ft.ProgressBar(
            height=8, color=c["progress_fg"],
            bgcolor=c["progress_bg"], value=0,
            border_radius=5,
        )
        self.progress_label = ft.Text("Ready", size=10, color=c["text_dim"])

        self.cache_status_lbl = ft.Text("Dead-symbol cache: empty", size=10, color=c["text_dim"])
        self.cache_clear_btn = ft.TextButton(
            content=ft.Text("Clear", size=11), on_click=self._clear_negative_cache,
            style=ft.ButtonStyle(color=c["red"]),
        )
        self.enrich_cache_status_lbl = ft.Text("Enrichment cache: empty", size=10, color=c["text_dim"])
        self.enrich_cache_clear_btn = ft.TextButton(
            content=ft.Text("Clear", size=11), on_click=self._clear_enrichment_cache,
            style=ft.ButtonStyle(color=c["red"]),
        )
        self.price_cache_status_lbl = ft.Text("Price cache: —", size=10, color=c["text_dim"])
        self.price_cache_prune_btn = ft.TextButton(
            content=ft.Text("Prune", size=11), on_click=self._prune_price_cache,
            style=ft.ButtonStyle(color=c["orange"]),
        )

        controls = ft.Column(
            controls=[
                ft.Container(
                    content=ft.Column([
                        ft.Text("Scanner", size=20, weight=ft.FontWeight.BOLD, color=c["text"]),
                        ft.Text("Indian Market Screener", size=11, color=c["text_dim"]),
                    ], spacing=2),
                    padding=_padding_only(left=16, top=16, bottom=6),
                ),
                ft.Divider(height=1, color=c["border"]),
                section("Stock Universe"),
                field(self.universe_dd),
                ft.Container(self.universe_count_label, padding=_padding_only(left=18, top=4)),
                section("Timeframe"),
                field(self.timeframe_dd),
                section("Data Period"),
                field(self.period_dd),
                section("Trend Filter"),
                field(self.trend_filter_dd),
                section("Rating Filter"),
                field(self.rating_filter_dd),
                section("Min Score Threshold"),
                ft.Container(
                    content=ft.Row([
                        self.threshold_slider,
                        threshold_chip,
                    ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=_padding_only(left=10, right=16),
                ),
                ft.Container(expand=True),
            ],
            spacing=0,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

        bottom = ft.Column(
            controls=[
                self.action_btn,
                self.progress_bar,
                self.progress_label,
                self._cache_card("Dead symbols", self.cache_status_lbl, self.cache_clear_btn),
                self._cache_card("Enrichment", self.enrich_cache_status_lbl, self.enrich_cache_clear_btn),
                self._cache_card("Price data", self.price_cache_status_lbl, self.price_cache_prune_btn),
            ],
            spacing=8,
        )

        return ft.Container(
            content=ft.Column(
                controls=[controls, ft.Container(content=bottom, padding=14)],
                spacing=0,
                expand=True,
            ),
            width=248,
            bgcolor=c["side_bg"],
        )

    def _build_main_area(self) -> ft.Container:
        c = self.theme_colors

        self.search_entry = ft.TextField(
            hint_text="🔍  Filter by ticker…",
            width=240, height=34, text_size=12,
            bgcolor=c["card"], color=c["text"],
            border_color=c["border"], border_width=1,
            border_radius=17,
            content_padding=_padding_only(left=12, top=4, bottom=4),
            on_change=self._on_search_change,
        )

        self.html_btn = ft.IconButton(
            icon=ft.Icons.SAVE_ALT, icon_color=c["cyan"], icon_size=18,
            tooltip="Export HTML report",
            on_click=lambda e: self._export_html(),
            disabled=True,
        )
        self.csv_btn = ft.IconButton(
            icon=ft.Icons.TABLE_CHART, icon_color=c["blue"], icon_size=18,
            tooltip="Export CSV",
            on_click=lambda e: self._export_csv(),
            disabled=True,
        )
        self.clear_btn = ft.IconButton(
            icon=ft.Icons.CLOSE, icon_color=c["red"], icon_size=18,
            tooltip="Clear results",
            on_click=lambda e: self._clear_results(),
            disabled=True,
        )

        topbar = ft.Container(
            content=ft.Row(
                controls=[
                    self.search_entry,
                    ft.Container(expand=True),
                    self.html_btn, self.csv_btn, self.clear_btn,
                ],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            height=56,
            bgcolor=c["panel_bg"],
            padding=_padding_only(left=16, right=14),
        )

        self.hero_text = ft.Text(
            "Find Your Next Swing Trade",
            size=21, weight=ft.FontWeight.BOLD, color="white",
        )
        self.hero_sub = ft.Text(
            "Set your universe on the left, then RUN SCAN — HMA×EMA crossover • 10-factor score • news sentiment",
            size=11, color="#d8ffe8",
        )
        hero = ft.Container(
            content=ft.Column(
                controls=[self.hero_text, ft.Container(height=16), self.hero_sub],
                spacing=0,
            ),
            gradient=ft.LinearGradient(
                begin=Alignment.CENTER_LEFT, end=Alignment.CENTER_RIGHT,
                colors=c["hero_grad"],
            ),
            border_radius=14,
            shadow=_card_shadow(),
            height=104,
            padding=_padding_only(left=26, top=24),
            margin=_margin_only(bottom=8),
        )

        self.summary_cards = {}
        self.summary_row = self._build_summary_row()

        self.chart_title = ft.Text("Score curve", size=12, weight=ft.FontWeight.BOLD, color=c["text"])
        self.chart_sub = ft.Text("", size=10, color=c["text_dim"])
        self.chart_bars = ft.Row(spacing=2, vertical_alignment=ft.CrossAxisAlignment.END)
        self.chart_card = ft.Container(
            content=ft.Column([
                ft.Row([
                    self.chart_title,
                    ft.Container(expand=True),
                    self.chart_sub,
                ]),
                ft.Container(content=self.chart_bars, height=64),
            ], spacing=6),
            bgcolor=_glass_bg(),
            border=_glass_border(),
            border_radius=14,
            shadow=_card_shadow(),
            padding=_padding_only(left=14, right=14, top=10, bottom=10),
            margin=_margin_only(top=8),
            visible=False,
        )

        self.result_count_label = ft.Text("no scan yet", size=11, color=c["text_dim"])
        section_header = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Text("Scan Results", size=16, weight=ft.FontWeight.BOLD, color=c["text"]),
                    ft.Container(expand=True),
                    self.result_count_label,
                ],
            ),
            padding=_padding_only(left=6, right=6, top=8, bottom=4),
        )

        self.table_column = ft.Column(spacing=0, expand=True)

        self.page_prev_btn = ft.TextButton(content=ft.Text("◀ Prev", size=12), on_click=lambda e: self._change_page(-1))
        self.page_label = ft.Text("Page 1 / 1", size=11, color=c["text_dim"])
        self.page_next_btn = ft.TextButton(content=ft.Text("Next ▶", size=12), on_click=lambda e: self._change_page(1))
        self.page_size_options = ["50", "100", "200", "500"]
        self.page_size_dd = ft.Dropdown(
            options=[ft.dropdown.Option(v) for v in self.page_size_options],
            value="100", width=110, height=40, text_size=12,
            bgcolor=c["card"], color=c["text"],
            border_color=c["border"], border_width=1,
            border_radius=8,
            focused_border_color=c["purple"],
            content_padding=_padding_only(left=12, right=8, top=8, bottom=8),
            on_select=self._on_page_size_change,
        )
        self.load_all_btn = ft.TextButton(
            content=ft.Text("Load All", size=12), on_click=lambda e: self._load_all_pages(),
            style=ft.ButtonStyle(color=c["text_dim"]),
        )
        self.pagination_row = ft.Row(
            controls=[
                self.page_prev_btn, self.page_label, self.page_next_btn,
                ft.Container(width=8),
                ft.Text("Rows:", size=11, color=c["text_dim"]),
                self.page_size_dd,
                ft.Container(expand=True),
                self.load_all_btn,
            ],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            visible=False,
        )
        self.pagination_bar = ft.Container(
            content=self.pagination_row,
            bgcolor=_glass_bg(),
            border=_glass_border(),
            border_radius=10,
            shadow=_card_shadow(),
            padding=_padding_only(left=10, right=10, top=6, bottom=6),
            margin=_margin_only(left=6, right=6, top=8, bottom=12),
            visible=False,
        )

        self.empty_label = ft.Container(
            content=ft.Text(
                "\nNo results yet — hit ▶ RUN SCAN\n",
                size=13, color=c["text_dim"], text_align=ft.TextAlign.CENTER,
            ),
            alignment=Alignment.CENTER,
            padding=40,
        )
        self.table_column.controls.append(self.empty_label)

        self.main_scroll = ft.Column(
            controls=[
                hero,
                self.summary_row,
                self.chart_card,
                section_header,
                self.table_column,
            ],
            spacing=0,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

        self.dashboard_content = ft.Column(
            controls=[
                topbar,
                ft.Container(content=self.main_scroll, expand=True, padding=_padding_only(left=6, right=6, top=6)),
                self.pagination_bar,
            ],
            spacing=0,
            expand=True,
        )
        self.main_area_box = ft.Container(
            content=self.dashboard_content,
            expand=True,
            bgcolor=c["main_bg"],
            gradient=ft.RadialGradient(
                center=Alignment(x=-1.0, y=-1.0),
                colors=["#1b2f4d", c["main_bg"]],
            ),
        )
        return self.main_area_box

    def _build_summary_row(self) -> ft.Row:
        c = self.theme_colors
        stats = [
            ("TOTAL", "total", c["cyan"], "◈"),
            ("PASSED", "passed", c["green"], "✓"),
            ("ENTRY", "entry", c["pink"], "★"),
            ("AVG", "avg", c["lime"], "⌀"),
            ("HIGH", "high", c["green"], "▲"),
            ("BULL", "bull", c["green"], "↗"),
            ("BEAR", "bear", c["red"], "↘"),
            ("DEAD-SKIP", "dead_skip", c["orange"], "∅"),
        ]
        cards = []
        for label, key, color, icon in stats:
            val_label = ft.Text("—", size=22, weight=ft.FontWeight.BOLD, color=color)
            self.summary_cards[key] = val_label
            card = ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Container(height=2, bgcolor=color, border_radius=1),
                        ft.Row([
                            ft.Text(icon, size=10, color=color),
                            ft.Text(label, size=8, weight=ft.FontWeight.BOLD, color=c["text_faint"]),
                        ], spacing=4),
                        val_label,
                    ],
                    spacing=2,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                width=122,
                bgcolor=_glass_bg(),
                border_radius=14,
                border=_glass_border(),
                shadow=_card_shadow(),
                padding=_padding_only(top=8, bottom=10, left=6, right=6),
            )
            cards.append(card)
        return ft.Row(controls=cards, spacing=8, scroll=ft.ScrollMode.AUTO)

    def _build_right_panel(self) -> ft.Container:
        c = self.theme_colors

        avatar = ft.Container(
            content=ft.Text("ABHI", size=13, weight=ft.FontWeight.BOLD, color="#8dffc4"),
            width=52, height=52, border_radius=26,
            bgcolor="#12331f",
            border=_border_all(2, c["cyan"]),
            alignment=Alignment.CENTER,
        )

        self.status_label = ft.Text("Status: Ready", size=10, color=c["text_dim"])

        self.topicks_column = ft.Column(spacing=0)
        self._render_topicks([])

        self.log_column = ft.Column(
            spacing=2, scroll=ft.ScrollMode.AUTO, auto_scroll=True, expand=True,
        )
        self.log_view = ft.Container(
            content=self.log_column,
            expand=True,
            bgcolor=c["panel_bg"],
            border=_border_all(1, c["border"]),
            border_radius=12,
            padding=8,
        )
        self.log_clear_btn = ft.TextButton(
            content=ft.Text("Clear", size=10), on_click=lambda e: self._clear_log(),
            style=ft.ButtonStyle(color=c["text_dim"]),
        )

        panel_content = ft.Column(
            controls=[
                ft.Container(
                    content=ft.Row([
                        avatar,
                        ft.Column([
                            ft.Text("HMAxEMA Scanner", size=14, weight=ft.FontWeight.BOLD, color=c["text"]),
                            ft.Text("@indian_markets", size=10, color=c["text_dim"]),
                        ], spacing=2),
                    ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=_padding_only(left=16, right=16, top=14, bottom=6),
                ),
                ft.Container(self.status_label, padding=_padding_only(left=18, top=2)),
                ft.Container(
                    content=ft.Row([
                        ft.Text("Top Picks", size=12, weight=ft.FontWeight.BOLD, color=c["text"]),
                        ft.Text("top 5", size=10, color=c["text_dim"]),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=_padding_only(left=16, right=16, top=10, bottom=4),
                ),
                ft.Container(
                    content=self.topicks_column,
                    padding=_padding_only(left=12, right=12),
                ),
                ft.Container(
                    content=ft.Row([
                        ft.Text("Recent Activity", size=12, weight=ft.FontWeight.BOLD, color=c["text"]),
                        ft.Container(expand=True),
                        ft.Text("live log", size=10, color=c["text_dim"]),
                        self.log_clear_btn,
                    ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=_padding_only(left=16, right=8, top=10, bottom=4),
                ),
                ft.Container(
                    content=self.log_view,
                    expand=True,
                    padding=_padding_only(left=10, right=10, bottom=12),
                ),
            ],
            spacing=0,
            expand=True,
        )

        return ft.Container(
            content=panel_content,
            width=300,
            bgcolor=c["side_bg"],
        )

    def _render_topicks(self, top):
        c = self.theme_colors
        self.topicks_column.controls.clear()
        if not top:
            if self.scanning:
                msg, color = "Scoring batches — leaders appear here…", c["green"]
            else:
                msg, color = "Run a scan to see leaders", c["text_dim"]
            self.topicks_column.controls.append(
                ft.Container(
                    content=ft.Text(msg, size=10, color=color),
                    padding=6,
                )
            )
            return
        for i, r in enumerate(top[:5], 1):
            score = _score_of(r)
            color = (c["green"] if score >= 70 else c["lime"] if score >= 50 else c["orange"] if score >= 30 else c["red"])
            card = ft.Container(
                content=ft.Row(
                    controls=[
                        ft.Container(
                            content=ft.Text(str(i), size=11, weight=ft.FontWeight.BOLD, color=color),
                            width=26, height=26, border_radius=13,
                            bgcolor=c["card2"],
                            alignment=Alignment.CENTER,
                        ),
                        ft.Column(
                            controls=[
                                ft.Text(r.get("ticker", "?"), size=12, weight=ft.FontWeight.BOLD, color=c["text"]),
                                ft.Text(f"₹{r.get('close', 0) or 0:,.0f}  ·  {r.get('trend_dir', '')}", size=9, color=c["text_dim"]),
                            ],
                            spacing=0,
                            expand=True,
                        ),
                        ft.Text(f"{score:.0f}", size=15, weight=ft.FontWeight.BOLD, color=color),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                bgcolor=_glass_bg(),
                border_radius=12,
                border=_glass_border(),
                shadow=_card_shadow(),
                padding=_padding_only(left=10, right=12, top=6, bottom=6),
                margin=_margin_only(bottom=3),
            )
            self.topicks_column.controls.append(card)
