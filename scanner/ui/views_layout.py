"""Dashboard shell builders — ``LayoutViewMixin`` for ``scanner.app.ScannerApp``.

Builds the four static panes of the app window (rail, sidebar, main area,
right panel) plus the small widgets they share (styled dropdowns, cache
cards, summary cards, top-pick cards).

The methods are mixins: they run against the ``ScannerApp`` instance (via
the MRO) and rely only on instance state such as ``self.theme_colors`` and
on the control attributes they create (``self.universe_dd`` etc.), which
the rest of the app reads after ``_build_ui()`` has run.
"""

import logging
import threading

import flet as ft
from flet.controls.alignment import Alignment

from ..shared.constants import score_of as _score_of
from ..shared.universes import UNIVERSES
from .ui_kit import (
    ANIM_FAST,
    ANIM_NORMAL,
    RADIUS_LG,
    RADIUS_MD,
    RADIUS_SM,
    RADIUS_XL,
    _border_all,
    _card_shadow,
    _glass_bg,
    _glass_border,
    _margin_only,
    _neon_glow,
    _padding_only,
    score_color,
)

logger = logging.getLogger(__name__)

# Fixed pane widths — single source of truth for the main-row geometry
# (must match _build_rail / _build_sidebar / _build_right_panel).
RAIL_W = 64
SIDE_W = 248
RIGHT_W = 300
# Hero geometry: market box width and its left margin.
MARKET_BOX_W = 190


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
            else:
                icon = ft.Icons.CIRCLE
            return ft.Container(
                content=ft.Icon(icon, color=accent, size=22),
                width=44,
                height=44,
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
                width=4,
                height=26,
                border_radius=2,
                bgcolor=c["neon"],
                shadow=_neon_glow(c["neon"], blur=8),
                opacity=1.0,
                animate_opacity=ANIM_NORMAL,
            )

        def empty_slot():
            return ft.Container(
                width=4,
                opacity=0.0,
                animate_opacity=ANIM_NORMAL,
                bgcolor=c["neon"],
                shadow=_neon_glow(c["neon"], blur=8),
            )

        self._rail_pills = {}
        home_pill = pill_slot() if self.active_view == "dashboard" else empty_slot()
        settings_pill = pill_slot() if self.active_view == "settings" else empty_slot()
        self._rail_pills["dashboard"] = home_pill
        self._rail_pills["settings"] = settings_pill

        logo = ft.Container(
            content=ft.Text("ABHI", color="white", size=11, weight=ft.FontWeight.BOLD),
            width=44,
            height=44,
            border_radius=22,
            bgcolor=c["purple"],
            alignment=Alignment.CENTER,
            shadow=_neon_glow(c["purple"], blur=16),
        )

        rail_items = ft.Column(
            controls=[
                rail_unit(
                    rail_icon("home", c["neon"], lambda: self._show_view("dashboard")),
                    home_pill,
                ),
                rail_unit(
                    rail_icon("gear", c["purple"], self._show_settings), settings_pill
                ),
                ft.Container(height=10),
                rail_unit(
                    rail_icon("play", c["green"], self._on_action_click), empty_slot()
                ),
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
                    ft.Container(height=12),
                ],
                spacing=0,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            width=RAIL_W,
            bgcolor=c["rail_bg"],
            padding=_padding_only(top=14),
        )

    def _styled_dropdown(self, options, value, on_select=None) -> ft.Dropdown:
        """Full-width modern dropdown matching the app theme."""
        from .ui_kit import themed_dropdown

        dd = themed_dropdown(options, value, self.theme_colors, on_select=on_select)
        dd.expand = True
        dd.height = 46
        dd.content_padding = _padding_only(left=12, right=8, top=8, bottom=8)
        dd.menu_height = 260
        return dd

    def _cache_card(self, title, status_lbl, clear_btn) -> ft.Container:
        """Labeled cache row: title + clear action on top, status below."""
        c = self.theme_colors
        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text(
                                title,
                                size=11,
                                weight=ft.FontWeight.BOLD,
                                color=c["text"],
                            ),
                            ft.Container(expand=True),
                            clear_btn,
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    status_lbl,
                ],
                spacing=3,
            ),
            bgcolor=_glass_bg(),
            border=_glass_border(),
            border_radius=RADIUS_LG,
            shadow=_card_shadow(),
            padding=_padding_only(left=10, right=4, top=7, bottom=7),
        )

    def _build_sidebar(self) -> ft.Container:
        c = self.theme_colors

        def section(t):
            return ft.Container(
                content=ft.Text(
                    t.upper(), size=9, weight=ft.FontWeight.BOLD, color=c["text_faint"]
                ),
                padding=_padding_only(left=16, top=14, bottom=4),
            )

        def field(ctrl):
            return ft.Container(ctrl, padding=_padding_only(left=16, right=16))

        # Watchlist file picker (Flet 1.0 client action — no server round-trip
        # to open the dialog). Services register on page.services, NOT
        # page.overlay (overlay is for renderable controls — a Service there
        # makes the client show "Unknown control"). Best-effort: the headless
        # test harness has no services list.
        self.file_picker = ft.FilePicker(on_result=self._on_watchlist_picked)
        try:
            self.page.services.append(self.file_picker)
        except Exception:
            logger.info(
                "FilePicker service registration failed (headless?)", exc_info=True
            )

        self.universe_dd = self._styled_dropdown(
            list(UNIVERSES.keys()),
            "NIFTY 50",
            on_select=self._on_universe_change,
        )
        self.universe_count_label = ft.Text(
            f"{len(UNIVERSES['NIFTY 50'])} stocks", size=10, color=c["text_dim"]
        )

        self.timeframe_dd = self._styled_dropdown(
            ["Daily", "Weekly", "Monthly"], "Daily"
        )
        self.period_dd = self._styled_dropdown(
            ["6 Months", "1 Year", "2 Years"], "1 Year"
        )
        self.trend_filter_dd = self._styled_dropdown(
            ["All", "Bullish Only", "Bearish Only"], "All"
        )
        self.rating_filter_dd = self._styled_dropdown(
            ["All", "Excellent", "Good", "Moderate", "Poor"],
            "All",
            on_select=self._on_rating_change,
        )
        self.threshold_slider = ft.Slider(
            min=0,
            max=100,
            value=50,
            divisions=20,
            expand=True,
            active_color=c["purple"],
            inactive_color=c["progress_bg"],
            on_change=self._on_threshold_change,
        )
        self.threshold_label = ft.Text(
            "50+", size=13, weight=ft.FontWeight.BOLD, color=c["pink"]
        )
        threshold_chip = ft.Container(
            content=self.threshold_label,
            bgcolor=c["card2"],
            border=_border_all(1, c["border"]),
            border_radius=RADIUS_SM,
            padding=_padding_only(left=10, right=10, top=2, bottom=2),
        )

        self.action_btn_label = ft.Text(
            "▶  RUN SCAN", size=14, weight=ft.FontWeight.BOLD
        )
        self.action_btn = ft.Button(
            content=self.action_btn_label,
            expand=True,
            height=48,
            bgcolor=c["green"],
            color=c["on_accent"],
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=RADIUS_LG)),
            on_click=self._on_action_click,
        )
        self.progress_bar = ft.ProgressBar(
            height=6,
            color=c["progress_fg"],
            bgcolor=c["progress_bg"],
            value=0,
            border_radius=RADIUS_SM,
        )
        self.progress_label = ft.Text("Ready", size=10, color=c["text_dim"])

        self.cache_status_lbl = ft.Text(
            "Dead-symbol cache: empty", size=10, color=c["text_dim"]
        )
        self.cache_clear_btn = ft.TextButton(
            content=ft.Text("Clear", size=11),
            on_click=self._clear_negative_cache,
            style=ft.ButtonStyle(color=c["red"]),
        )
        self.enrich_cache_status_lbl = ft.Text(
            "Enrichment cache: empty", size=10, color=c["text_dim"]
        )
        self.enrich_cache_clear_btn = ft.TextButton(
            content=ft.Text("Clear", size=11),
            on_click=self._clear_enrichment_cache,
            style=ft.ButtonStyle(color=c["red"]),
        )
        self.price_cache_status_lbl = ft.Text(
            "Price cache: —", size=10, color=c["text_dim"]
        )
        self.price_cache_prune_btn = ft.TextButton(
            content=ft.Text("Prune", size=11),
            on_click=self._prune_price_cache,
            style=ft.ButtonStyle(color=c["orange"]),
        )

        self.universe_count_container = ft.Container(
            self.universe_count_label,
            padding=_padding_only(left=18, top=4),
            visible=False,
        )
        self.watchlist_import_container = ft.Container(
            content=ft.TextButton(
                content=ft.Text("Import watchlist…", size=11),
                tooltip="Import tickers from a CSV/TXT file",
                style=ft.ButtonStyle(color=c["cyan"]),
                action=ft.PickFiles(
                    self.file_picker,
                    dialog_title="Import watchlist",
                    file_type=ft.FilePickerFileType.CUSTOM,
                    allowed_extensions=["csv", "txt"],
                ),
            ),
            padding=_padding_only(left=16, right=16),
            visible=False,
        )

        controls = ft.Column(
            controls=[
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                "Scanner",
                                size=20,
                                weight=ft.FontWeight.BOLD,
                                color=c["text"],
                            ),
                            ft.Text(
                                "Indian Market Screener", size=11, color=c["text_dim"]
                            ),
                        ],
                        spacing=2,
                    ),
                    padding=_padding_only(left=16, top=16, bottom=6),
                ),
                ft.Divider(height=1, color=c["border"]),
                section("Stock Universe"),
                field(self.universe_dd),
                self.universe_count_container,
                self.watchlist_import_container,
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
                    content=ft.Row(
                        [
                            self.threshold_slider,
                            threshold_chip,
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
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
                self._cache_card(
                    "Dead symbols", self.cache_status_lbl, self.cache_clear_btn
                ),
                self._cache_card(
                    "Enrichment",
                    self.enrich_cache_status_lbl,
                    self.enrich_cache_clear_btn,
                ),
                self._cache_card(
                    "Price data",
                    self.price_cache_status_lbl,
                    self.price_cache_prune_btn,
                ),
            ],
            spacing=8,
        )

        return ft.Container(
            content=ft.Column(
                controls=[controls, ft.Container(content=bottom, padding=14)],
                spacing=0,
                expand=True,
            ),
            width=SIDE_W,
            bgcolor=c["side_bg"],
        )

    def _build_main_area(self) -> ft.Container:
        c = self.theme_colors
        self.topbar_title = ft.Text(
            "Scanner", size=13, weight=ft.FontWeight.BOLD, color=c["text"]
        )
        self.search_entry = ft.TextField(
            hint_text="Filter by ticker…",
            width=260,
            height=36,
            text_size=12,
            bgcolor=c["card"],
            color=c["text"],
            border_color=c["border"],
            border_width=1,
            border_radius=18,
            prefix_icon=ft.Icons.SEARCH,
            suffix=ft.IconButton(
                icon=ft.Icons.CLOSE,
                icon_size=14,
                icon_color=c["text_dim"],
                tooltip="Clear filter",
                width=28,
                height=28,
                on_click=lambda e: self._clear_search_filter(),
            ),
            content_padding=_padding_only(left=10, top=4, bottom=4),
            on_change=self._on_search_change,
            on_focus=lambda e: setattr(self, "_input_focused", True),
            on_blur=lambda e: setattr(self, "_input_focused", False),
        )

        self.html_btn = ft.IconButton(
            icon=ft.Icons.SAVE_ALT,
            icon_color=c["cyan"],
            icon_size=18,
            tooltip="Export HTML report",
            on_click=lambda e: self._export_html(),
            disabled=True,
        )
        self.csv_btn = ft.IconButton(
            icon=ft.Icons.TABLE_CHART,
            icon_color=c["blue"],
            icon_size=18,
            tooltip="Export CSV",
            on_click=lambda e: self._export_csv(),
            disabled=True,
        )
        self.clear_btn = ft.IconButton(
            icon=ft.Icons.CLOSE,
            icon_color=c["red"],
            icon_size=18,
            tooltip="Clear results",
            on_click=lambda e: self._clear_results(),
            disabled=True,
        )

        self.topbar = ft.Container(
            content=ft.Row(
                controls=[
                    ft.IconButton(
                        icon=ft.Icons.MENU_ROUNDED,
                        icon_color=c["text_dim"],
                        icon_size=18,
                        tooltip="Toggle sidebar",
                        on_click=self._toggle_sidebar,
                    ),
                    self.topbar_title,
                    ft.Container(width=12),
                    self.search_entry,
                    ft.Container(expand=True),
                    self.html_btn,
                    self.csv_btn,
                    self.clear_btn,
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
            size=21,
            weight=ft.FontWeight.BOLD,
            color=c["hero_title"],
        )
        self.hero_sub = ft.Text(
            "Set your universe on the left, then RUN SCAN — HMA×EMA crossover • 10-factor score • news sentiment",
            size=11,
            color=c["hero_sub"],
        )
        # Live market readout (NIFTY level / day change) — hidden until data
        # arrives from the provider chain (see _render_market/_warm_market).
        self.market_label = ft.Text(
            "NIFTY 50",
            size=9,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.with_opacity(0.65, c["hero_sub"]),
        )
        self.market_value = ft.Text(
            "—",
            size=23,
            weight=ft.FontWeight.BOLD,
            color=c["hero_title"],
            selectable=True,
        )
        self.market_change = ft.Text("", size=10, color=c["hero_sub"], selectable=True)
        self.market_box = ft.Container(
            content=ft.Column(
                controls=[self.market_label, self.market_value, self.market_change],
                spacing=0,
                horizontal_alignment=ft.CrossAxisAlignment.END,
            ),
            width=MARKET_BOX_W,
            border_radius=RADIUS_LG,
            border=_border_all(1, c["border_light"]),
            bgcolor=c["card2"],
            padding=_padding_only(left=16, right=16, top=8, bottom=8),
            visible=False,
        )
        # Loading placeholder in the same slot — shown until the first market
        # snapshot lands (or the fetch fails, see _dismiss_market_loading).
        self.market_loading = ft.Container(
            content=ft.ProgressRing(
                width=16, height=16, stroke_width=2, color=c["hero_sub"]
            ),
            width=MARKET_BOX_W,
            height=56,
            alignment=Alignment.CENTER,
            tooltip="Fetching live market data…",
        )
        self.market_slot = ft.Stack(
            controls=[self.market_box, self.market_loading],
            width=MARKET_BOX_W,
        )
        # Secondary index readouts (BANK NIFTY / SENSEX / NIFTY IT) — a slim
        # ticker strip at the bottom of the hero; hidden until data arrives.
        self.market_strip = ft.Row(spacing=6, scroll=ft.ScrollMode.AUTO, visible=False)
        self.hero_text_col = ft.Column(
            controls=[self.hero_text, ft.Container(height=16), self.hero_sub],
            spacing=0,
        )
        # The hero Row keeps the gradient pill (text + strip) and the
        # market_slot as siblings — the NIFTY box sits outside the gradient,
        # on the page background. Slot width is fixed so layout geometry is
        # stable even when the box is hidden — no expand/flex needed.
        self.hero = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(
                        content=ft.Column(
                            controls=[
                                self.hero_text_col,
                                ft.Container(height=10),
                                self.market_strip,
                            ],
                            spacing=0,
                        ),
                        gradient=ft.LinearGradient(
                            begin=Alignment.CENTER_LEFT,
                            end=Alignment.CENTER_RIGHT,
                            colors=c["hero_grad"],
                        ),
                        border_radius=RADIUS_XL,
                        shadow=_card_shadow(),
                        padding=_padding_only(left=28, right=20, top=18, bottom=14),
                        expand=True,
                    ),
                    ft.Container(width=16),
                    self.market_slot,
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            margin=_margin_only(bottom=10),
        )

        self.summary_cards = {}
        self.insight_stats = self._build_insight_stats()

        self.chart_sub = ft.Text("tap a bar = min score", size=9, color=c["text_faint"])
        self.chart_bars = ft.Row(
            spacing=2,
            vertical_alignment=ft.CrossAxisAlignment.END,
            alignment=ft.MainAxisAlignment.CENTER,
        )
        self.chart_holder = ft.Container(
            content=self.chart_bars, height=76, visible=False
        )
        # Slim insight strip: unique scan stats (left) + clickable score
        # histogram (right) — replaces the old 8-card row + full chart card
        # (~300px) with ~90px, giving the grid five more visible rows.
        self.insight_strip = ft.Container(
            content=ft.Row(
                [
                    self.insight_stats,
                    ft.Container(expand=True),
                    self.chart_sub,
                    ft.Container(width=8),
                    self.chart_holder,
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=8,
            ),
            bgcolor=_glass_bg(),
            border=_glass_border(),
            border_radius=RADIUS_LG,
            shadow=_card_shadow(),
            padding=_padding_only(left=16, right=16, top=8, bottom=8),
        )

        self.result_count_label = ft.Text("no scan yet", size=11, color=c["text_dim"])
        self.scan_inline_label = ft.Text(
            "", size=11, weight=ft.FontWeight.BOLD, color=c["green"], visible=False
        )
        self.filter_chips_row = ft.Row(
            [], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER
        )
        self.section_header = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Text(
                        "Scan Results",
                        size=16,
                        weight=ft.FontWeight.BOLD,
                        color=c["text"],
                    ),
                    ft.Container(width=10),
                    self.filter_chips_row,
                    ft.Container(expand=True),
                    self.scan_inline_label,
                    ft.Container(width=10),
                    self.result_count_label,
                ],
            ),
            padding=_padding_only(left=8, right=8, top=10, bottom=6),
        )

        self.table_column = ft.Column(spacing=0, expand=True, scroll=ft.ScrollMode.AUTO)
        # Column header lives OUTSIDE the row scroll pane (Flet has no sticky
        # headers) so labels stay pinned above the grid while rows scroll.
        self.header_holder = ft.Column(spacing=0)

        self.page_prev_btn = ft.TextButton(
            content=ft.Text("◀ Prev", size=12), on_click=lambda e: self._change_page(-1)
        )
        self.page_label = ft.Text("Page 1 / 1", size=11, color=c["text_dim"])
        self.page_next_btn = ft.TextButton(
            content=ft.Text("Next ▶", size=12), on_click=lambda e: self._change_page(1)
        )
        self.page_size_options = ["50", "100", "200", "500"]
        self.page_size_dd = ft.Dropdown(
            options=[ft.dropdown.Option(v) for v in self.page_size_options],
            value="100",
            width=110,
            height=40,
            text_size=12,
            bgcolor=c["card"],
            color=c["text"],
            border_color=c["border"],
            border_width=1,
            border_radius=RADIUS_SM,
            focused_border_color=c["purple"],
            content_padding=_padding_only(left=12, right=8, top=8, bottom=8),
            on_select=self._on_page_size_change,
        )
        self.load_all_btn = ft.TextButton(
            content=ft.Text("Load All", size=12),
            on_click=lambda e: self._load_all_pages(),
            style=ft.ButtonStyle(color=c["text_dim"]),
        )
        self.pagination_row = ft.Row(
            controls=[
                self.page_prev_btn,
                self.page_label,
                self.page_next_btn,
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
            border_radius=RADIUS_LG,
            shadow=_card_shadow(),
            padding=_padding_only(left=12, right=12, top=7, bottom=7),
            margin=_margin_only(left=6, right=6, top=10, bottom=14),
            visible=False,
        )

        self.empty_label = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Icon(ft.Icons.SEARCH_OFF, size=48, color=c["text_faint"]),
                    ft.Text(
                        "No results yet",
                        size=15,
                        weight=ft.FontWeight.BOLD,
                        color=c["text"],
                    ),
                    ft.Text("or press Ctrl+R", size=11, color=c["text_dim"]),
                    ft.Button(
                        content=ft.Text("RUN SCAN", size=12, weight=ft.FontWeight.BOLD),
                        on_click=lambda e: self._on_action_click(e),
                        style=ft.ButtonStyle(bgcolor=c["green"], color=c["main_bg"]),
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=8,
            ),
            alignment=Alignment.CENTER,
            padding=48,
        )
        self.table_column.controls.append(self.empty_label)

        # Split scroll regions: overview (hero/cards/chart) takes its natural
        # height; the pinned section+column headers sit above the grid, which
        # scrolls independently in the remaining space.
        # ponytail: split panes instead of sticky — one scrollport per region
        self.main_scroll = ft.Column(
            controls=[
                self.hero,
                self.insight_strip,
            ],
            spacing=0,
        )
        self.results_head = ft.Column(
            controls=[self.section_header, self.header_holder],
            spacing=0,
        )

        self.dashboard_content = ft.Column(
            controls=[
                self.topbar,
                ft.Container(
                    content=self.main_scroll,
                    padding=_padding_only(left=6, right=6, top=6),
                ),
                ft.Container(
                    content=self.results_head,
                    padding=_padding_only(left=6, right=6),
                ),
                ft.Container(
                    content=self.table_column,
                    expand=True,
                    padding=_padding_only(left=6, right=6),
                ),
                self.pagination_bar,
            ],
            spacing=0,
            expand=True,
        )

        # Dashboard stays permanently mounted and visible. Settings is an
        # opaque overlay on top — switching never touches dashboard layout.
        # Both panes are positioned fill (left/top/right/bottom=0): Flet's
        # expand only works under Column/Row/View/Page, and StackFit.EXPAND
        # alone left the dashboard Column collapsed to topbar height.
        # ponytail: overlay not routes — split panes only if both must show.
        self.dashboard_view = ft.Container(
            content=self.dashboard_content,
            left=0,
            top=0,
            right=0,
            bottom=0,
            visible=True,
            opacity=1.0,
        )

        settings_body = self._build_settings_view()
        self.settings_view = ft.Container(
            content=settings_body,
            left=0,
            top=0,
            right=0,
            bottom=0,
            bgcolor=c["main_bg"],
            visible=getattr(self, "active_view", "dashboard") == "settings",
            opacity=1.0,
            animate_opacity=ANIM_NORMAL,
        )

        self.main_area_box = ft.Stack(
            controls=[
                ft.Container(
                    left=0,
                    top=0,
                    right=0,
                    bottom=0,
                    bgcolor=c["main_bg"],
                    gradient=ft.RadialGradient(
                        center=Alignment(x=-1.0, y=-1.0),
                        colors=[c["bg_radial"], c["main_bg"]],
                    ),
                ),
                self.dashboard_view,
                self.settings_view,
            ],
            expand=True,
        )
        if getattr(self, "_last_market", None) is not None:
            self._render_market(self._last_market)
        return self.main_area_box

    def _build_insight_stats(self) -> ft.Row:
        """Inline stat pills for the insight strip (unique metrics only).

        TOTAL/PASSED/ENTRY live in the hero line + section header; DEAD-SKIP
        is a scan diagnostic that the log already prints — neither earns
        permanent strip space.
        """
        c = self.theme_colors
        stats = [
            ("AVG", "avg", c["lime"]),
            ("HIGH", "high", c["green"]),
            ("BULL", "bull", c["green"]),
            ("BEAR", "bear", c["red"]),
        ]
        pills = []
        for label, key, color in stats:
            val_label = ft.Text(
                "—",
                size=14,
                weight=ft.FontWeight.BOLD,
                color=color,
                animate_opacity=ANIM_FAST,
            )
            self.summary_cards[key] = val_label
            pills.append(
                ft.Row(
                    [
                        ft.Text(
                            label,
                            size=8,
                            weight=ft.FontWeight.BOLD,
                            color=c["text_faint"],
                        ),
                        val_label,
                    ],
                    spacing=5,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                )
            )
        return ft.Row(
            pills, spacing=16, vertical_alignment=ft.CrossAxisAlignment.CENTER
        )

    def _on_card_hover(self, e, card):
        """Scale up + intensify shadow on hover for summary/top-pick cards."""
        try:
            if e.data == "true":
                card.scale = ft.Scale(1.04)
                card.opacity = 0.92
                card.shadow = [
                    ft.BoxShadow(
                        blur_radius=32,
                        color=ft.Colors.with_opacity(0.65, ft.Colors.BLACK),
                        spread_radius=2,
                    )
                ]
            else:
                card.scale = ft.Scale(1.0)
                card.opacity = 1.0
                card.shadow = _card_shadow()
            card.update()
        except Exception:
            logger.info("Card hover animation update failed", exc_info=True)

    def _build_right_panel(self) -> ft.Container:
        c = self.theme_colors

        avatar = ft.Container(
            content=ft.Text(
                "ABHI", size=13, weight=ft.FontWeight.BOLD, color=c["avatar_text"]
            ),
            width=54,
            height=54,
            border_radius=27,
            bgcolor=c["avatar_bg"],
            border=_border_all(2, c["avatar_border"]),
            alignment=Alignment.CENTER,
        )

        self.status_label = ft.Text("Status: Ready", size=10, color=c["text_dim"])

        self.topicks_column = ft.Column(spacing=0)
        self._render_topicks([])

        self.log_column = ft.Column(
            spacing=2,
            scroll=ft.ScrollMode.AUTO,
            auto_scroll=True,
            expand=True,
        )
        self.log_view = ft.Container(
            content=self.log_column,
            expand=True,
            bgcolor=c["panel_bg"],
            border=_border_all(1, c["border"]),
            border_radius=RADIUS_LG,
            padding=10,
        )
        self.log_clear_btn = ft.TextButton(
            content=ft.Text("Clear", size=10),
            on_click=lambda e: self._clear_log(),
            style=ft.ButtonStyle(color=c["text_dim"]),
        )

        panel_content = ft.Column(
            controls=[
                ft.Container(
                    content=ft.Row(
                        [
                            avatar,
                            ft.Column(
                                [
                                    ft.Text(
                                        "HMAxEMA Scanner",
                                        size=14,
                                        weight=ft.FontWeight.BOLD,
                                        color=c["text"],
                                    ),
                                    ft.Text(
                                        "@indian_markets", size=10, color=c["text_dim"]
                                    ),
                                ],
                                spacing=2,
                            ),
                        ],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    padding=_padding_only(left=16, right=16, top=14, bottom=6),
                ),
                ft.Container(self.status_label, padding=_padding_only(left=18, top=2)),
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Text(
                                "Top Picks",
                                size=12,
                                weight=ft.FontWeight.BOLD,
                                color=c["text"],
                            ),
                            ft.Text("top 5", size=10, color=c["text_dim"]),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    padding=_padding_only(left=16, right=16, top=10, bottom=4),
                ),
                ft.Container(
                    content=self.topicks_column,
                    padding=_padding_only(left=12, right=12),
                ),
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Text(
                                "Recent Activity",
                                size=12,
                                weight=ft.FontWeight.BOLD,
                                color=c["text"],
                            ),
                            ft.Container(expand=True),
                            ft.Text("live log", size=10, color=c["text_dim"]),
                            self.log_clear_btn,
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
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
            width=RIGHT_W,
            bgcolor=c["side_bg"],
        )

    def _make_score_gauge(
        self, score: float, color: str, size: int = 30
    ) -> ft.Container:
        """Compact circular score display using a progress ring + centered text."""
        return ft.Container(
            content=ft.Stack(
                controls=[
                    ft.ProgressRing(
                        width=size,
                        height=size,
                        stroke_width=3,
                        value=score / 100.0,
                        color=color,
                        bgcolor=ft.Colors.with_opacity(0.12, ft.Colors.WHITE),
                    ),
                    ft.Container(
                        content=ft.Text(
                            str(int(score)),
                            size=9,
                            weight=ft.FontWeight.BOLD,
                            color=color,
                        ),
                        alignment=Alignment.CENTER,
                    ),
                ],
                width=size,
                height=size,
            ),
            width=size,
            height=size,
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
            color = score_color(score, c)
            card = ft.Container(
                content=ft.Row(
                    controls=[
                        ft.Container(
                            content=ft.Text(
                                str(i), size=11, weight=ft.FontWeight.BOLD, color=color
                            ),
                            width=28,
                            height=28,
                            border_radius=RADIUS_LG,
                            bgcolor=c["card2"],
                            alignment=Alignment.CENTER,
                        ),
                        ft.Column(
                            controls=[
                                ft.Text(
                                    r.get("ticker", "?"),
                                    size=12,
                                    weight=ft.FontWeight.BOLD,
                                    color=c["text"],
                                ),
                                ft.Text(
                                    f"₹{r.get('close', 0) or 0:,.0f}  ·  {r.get('trend_dir', '')}",
                                    size=9,
                                    color=c["text_dim"],
                                ),
                            ],
                            spacing=1,
                            expand=True,
                        ),
                        self._make_score_gauge(score, color),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                bgcolor=_glass_bg(),
                border_radius=RADIUS_LG,
                border=_glass_border(),
                shadow=_card_shadow(),
                padding=_padding_only(left=10, right=12, top=7, bottom=7),
                margin=_margin_only(bottom=4),
                animate_scale=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
                animate_opacity=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
            )
            card.on_hover = lambda e, _card=card: self._on_card_hover(e, _card)
            self.topicks_column.controls.append(card)

    # ── Market readout (hero card) ────────────────────────────────────────
    # The hero shows the latest NIFTY 50 level / day change plus a strip of
    # secondary indices, fetched through the same provider chain the scan
    # engine uses (4 h disk cache, so this never duplicates a scan's download
    # and quietly no-ops offline). Extra indices are each capped at 8 s so an
    # offline launch cannot stall the daemon warm-up for long.
    _EXTRA_INDICES = (
        ("^NSEBANK", "BANK NIFTY"),
        ("^BSESN", "SENSEX"),
        ("^CNXIT", "NIFTY IT"),
    )
    _INDEX_FETCH_TIMEOUT = 8.0

    @staticmethod
    def _quote_from_df(df) -> dict | None:
        """{level, change, pct} from an OHLCV frame's close column."""
        try:
            if df is None or "close" not in df.columns:
                return None
            closes = df["close"].dropna()
            if len(closes) < 2:
                return None
            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2])
            if not last or not prev:
                return None
            return {
                "level": last,
                "change": last - prev,
                "pct": (last - prev) / prev * 100.0,
            }
        except Exception:
            logger.info("Quote extraction from dataframe failed", exc_info=True)
            return None

    @staticmethod
    def _fetch_index_bounded(symbol: str, period: str = "1y", timeout: float = 8.0):
        """fetch_index_data capped at ``timeout`` s (daemon thread join)."""
        box: dict = {}

        def _run():
            try:
                from ..api.data_fetcher import fetch_index_data

                box["v"] = fetch_index_data(symbol, period=period)
            except Exception:
                logger.info("Bounded index fetch failed", exc_info=True)
                box["v"] = None

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return None
        return box.get("v")

    def _market_snapshot(self) -> dict | None:
        """NIFTY quote + secondary-index quotes (worker thread only)."""
        info = self._quote_from_df(self._fetch_index_bounded("^NSEI"))
        quotes = []
        for symbol, label in self._EXTRA_INDICES:
            q = self._quote_from_df(self._fetch_index_bounded(symbol))
            if q is not None:
                quotes.append({"label": label, **q})
        out = dict(info or {})
        out["quotes"] = quotes
        return out if "level" in out else None

    def _market_chip(self, q: dict) -> ft.Container:
        """One compact index readout chip for the hero ticker strip."""
        c = self.theme_colors
        up = q["change"] >= 0
        move_color = c["green"] if up else c["red"]
        return ft.Container(
            content=ft.Row(
                [
                    ft.Text(
                        q["label"],
                        size=9,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.with_opacity(0.85, c["hero_sub"]),
                    ),
                    ft.Text(
                        f"{q['level']:,.1f}",
                        size=13,
                        weight=ft.FontWeight.BOLD,
                        color=c.get("hero_value", c["hero_title"]),
                    ),
                    ft.Text(
                        f"{'▲' if up else '▼'} {q['pct']:+.2f}%",
                        size=10,
                        weight=ft.FontWeight.BOLD,
                        color=move_color,
                    ),
                ],
                spacing=7,
                vertical_alignment=ft.CrossAxisAlignment.BASELINE,
            ),
            bgcolor=c["chip_neutral"],
            border_radius=RADIUS_MD,
            border=_border_all(1, c["border_light"]),
            padding=_padding_only(left=14, right=14, top=7, bottom=7),
            tooltip=f"{q['label']} · day change",
        )

    def _render_market(self, info: dict | None):
        """Apply a market snapshot to the hero readout + strip (UI thread)."""
        box = getattr(self, "market_box", None)
        self._last_market = info or None
        if box is None:
            return
        if not info:
            box.visible = False
            if getattr(self, "market_strip", None) is not None:
                self.market_strip.visible = False
            self._dismiss_market_loading()
            return
        c = self.theme_colors
        up = info["change"] >= 0
        self.market_value.value = f"{info['level']:,.1f}"
        self.market_change.value = (
            f"{'▲' if up else '▼'} {abs(info['change']):,.1f} ({info['pct']:+.2f}%)"
        )
        self.market_change.color = c["green"] if up else c["red"]
        self.market_value.color = c["green"] if up else c["red"]
        self._dismiss_market_loading()
        box.visible = True
        strip = getattr(self, "market_strip", None)
        if strip is not None:
            chips = [self._market_chip(q) for q in info.get("quotes", [])]
            strip.controls = chips
            strip.visible = bool(chips)

    def _dismiss_market_loading(self):
        """Hide the hero loading placeholder once (UI thread)."""
        loading = getattr(self, "market_loading", None)
        if loading is not None:
            loading.visible = False

    def _warm_market(self):
        """Refresh the hero readout from disk cache / providers (daemon)."""
        info = self._market_snapshot()
        if info is not None:
            self._safe_update(lambda: self._render_market(info))
        else:
            # No data (offline / all providers failed) — stop the spinner
            # rather than letting it spin for the whole session.
            self._safe_update(self._dismiss_market_loading)
