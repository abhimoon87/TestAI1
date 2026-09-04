"""
HMAxEMA Stock Scanner — GUI Application (Flet Edition)

Modern dark desktop app for scanning Indian stocks.
Layout: [icon rail] [nav sidebar + scan controls] [main: hero, stats, results] [profile panel]

Usage:
    python -m scanner
    python scanner/app.py
"""

import json
import logging
import os
import threading
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Any, Dict, List

import flet as ft
from flet.controls.alignment import Alignment


def _border_all(width: float, color: str) -> ft.Border:
    """Return a uniform ``ft.Border`` with ``ft.BorderSide`` on all sides."""
    side = ft.BorderSide(width=width, color=color)
    return ft.Border(top=side, bottom=side, left=side, right=side)


def _glass_bg() -> ft.Paint:
    """Translucent white fill for frosted-glass cards."""
    return ft.Colors.with_opacity(0.045, ft.Colors.WHITE)


def _glass_border() -> ft.Border:
    side = ft.BorderSide(width=1, color=ft.Colors.with_opacity(0.09, ft.Colors.WHITE))
    return ft.Border(top=side, bottom=side, left=side, right=side)


def _card_shadow() -> List[ft.BoxShadow]:
    return [ft.BoxShadow(blur_radius=18, color=ft.Colors.with_opacity(0.45, ft.Colors.BLACK))]


def _neon_glow(color: str, blur: int = 10) -> List[ft.BoxShadow]:
    return [ft.BoxShadow(blur_radius=blur, color=color)]


def _padding_only(left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> ft.Padding:
    return ft.Padding(left=left, top=top, right=right, bottom=bottom)


def _margin_only(left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> ft.Margin:
    return ft.Margin(left=left, top=top, right=right, bottom=bottom)

from .trace import setup_trace

try:
    setup_trace()
except Exception:
    pass

logger = logging.getLogger(__name__)
logger.info("app module loaded -- trace active at %s", Path(__file__).parent / "trace.log")

from .report import _parse_date, _sentiment, generate_html_report, save_report
from .themes import THEMES
from .universes import UNIVERSES

SCANNER_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(SCANNER_DIR, "settings.json")
LOG_FILE = os.path.join(SCANNER_DIR, "scan.log")
LOG_ROTATE_HOURS = 12
LOG_MAX_LINES = 500

DEFAULT_SETTINGS = {
    "fast_ma_type": "HMA", "fast_ma_len": 40,
    "slow_ma_type": "EMA", "slow_ma_len": 50,
    "rsi_len": 14, "rs_length": 14, "vol_ma_len": 20, "atr_len": 14,
    "index_symbol": "NSEI",
    "vp_lookback": 200, "vp_rows": 30, "vp_width": 40,
    "adx_len": 14, "adx_threshold": 20.0,
    "chop_len": 14, "chop_threshold": 61.8,
    "slope_ma_type": "KAMA", "slope_ma_len": 50, "slope_lookback": 10, "flat_threshold": 0.5,
    "sc_pivot_len": 3, "sc_bands_mult": 0.6, "crossover_lookback": 20,
    "min_score": 50.0, "data_period": "1y", "timeframe": "D", "trend_filter": "All",
    "negative_cache_ttl_hours": 24, "theme": "dark",
}


def load_settings() -> dict:
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings.update(json.load(f))
        except Exception as e:
            logger.debug("Failed to load settings: %s", e)
    return settings


def save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.debug("Failed to save settings: %s", e)


RESULT_COLS = [
    ("#", 35), ("Ticker", 100), ("Score", 50), ("Rating", 78), ("ENTRY", 55),
    ("Price", 80), ("MA", 62), ("T/15", 40), ("M/15", 40), ("R/8", 35),
    ("V/7", 35), ("Vol/10", 42), ("RS/10", 42), ("F/20", 42),
    ("1M", 55), ("Dir", 58), ("ADX", 40), ("Chop", 42),
]


def _score_of(r: dict) -> float:
    """Total score of a result row, tolerant of missing/None values."""
    return r.get("total", 0) or 0


class ScannerApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.settings = load_settings()
        self._apply_cache_settings()
        self.results = []
        self.all_results = []
        self.filtered_results = []
        self._results_lock = threading.Lock()
        self.scanning = False
        self.filter_text = ""
        self.active_view = "dashboard"
        self.page_size = 100
        self.current_page = 0
        self.sort_col = None
        self.sort_reverse = False

        theme_name = self.settings.get("theme", "dark")
        if theme_name not in THEMES:
            theme_name = "dark"
            self.settings["theme"] = "dark"
        self.current_theme = theme_name
        self.theme_colors = THEMES[theme_name]

        self._build_ui()
        self._load_settings_to_ui()
        self._refresh_neg_cache_ui()
        self._refresh_enrich_cache_ui()
        self._log("Scanner ready — pick a universe and hit RUN SCAN")
        self._rotate_log()

        def _warm_symbols():
            try:
                from .symbol_fetcher import _load_disk_cache
                from .universes import get_universe
                _load_disk_cache()
                get_universe("FULL MARKET (NSE+BSE ~5,900)")
            except Exception:
                pass
        threading.Thread(target=_warm_symbols, daemon=True).start()

    def _apply_cache_settings(self):
        try:
            from . import data_fetcher
            data_fetcher.set_negative_cache_ttl_hours(
                self.settings.get("negative_cache_ttl_hours", 24)
            )
        except Exception:
            pass

    def _build_ui(self):
        c = self.theme_colors
        self.page.bgcolor = c["main_bg"]
        self.page.padding = 0
        self.page.window.width = 1600
        self.page.window.height = 900
        self.page.window.min_width = 1280
        self.page.window.min_height = 800
        self.page.title = "HMAxEMA Stock Scanner — Indian Market"

        theme_mode = "dark" if self.current_theme == "dark" else "light"
        self.page.theme_mode = theme_mode

        self.page.controls.clear()

        main_row = ft.Row(
            controls=[
                self._build_rail(),
                self._build_sidebar(),
                self._build_main_area(),
                self._build_right_panel(),
            ],
            spacing=0,
            expand=True,
        )
        self.page.add(main_row)

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

        logo = ft.Container(
            content=ft.Text("H", color="white", size=17, weight=ft.FontWeight.BOLD),
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
            content=ft.Text("H", size=20, weight=ft.FontWeight.BOLD, color="#8dffc4"),
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

    # ── View switching ──────────────────────────────────────────────────

    def _show_view(self, name):
        self.active_view = name
        for vname, pill in self._rail_pills.items():
            pill.visible = (vname == name)
        if name == "dashboard":
            box = getattr(self, "main_area_box", None)
            dash = getattr(self, "dashboard_content", None)
            if box is not None and dash is not None:
                box.content = dash
        self.page.update()

    def _show_settings(self, e=None):
        self.active_view = "settings"
        for pill in self._rail_pills.values():
            pill.visible = False
        self.main_area_box.content = self._build_settings_view()
        self.page.update()

    def _on_search_change(self, e):
        self.filter_text = self.search_entry.value.strip().upper() if self.search_entry.value else ""
        if self.all_results:
            self._display_results(self.all_results)

    def _rating_filter(self) -> str:
        return str(self.rating_filter_dd.value or "ALL").upper()

    def _is_filter_active(self) -> bool:
        return bool(self.filter_text) or self._rating_filter() != "ALL"

    def _visible_results(self) -> list:
        """Results after search/rating filters.

        Returns the filtered list as-is (possibly empty) when a filter is
        active, so 'no match' is distinguishable from 'no filter'.
        """
        if self._is_filter_active():
            return self.filtered_results
        return self.all_results

    def _row_matches_filters(self, r: dict) -> bool:
        if self.filter_text and self.filter_text not in r.get("ticker", "").upper():
            return False
        rating = self._rating_filter()
        if rating != "ALL":
            combined = (r.get("combined_rating") or "POOR").upper()
            if combined != rating:
                return False
        return True

    def _on_rating_change(self, e):
        if self.all_results:
            self._display_results(self.all_results)

    def _on_universe_change(self, e):
        choice = self.universe_dd.value or "NIFTY 50"
        try:
            base = len(UNIVERSES.get(choice, []))
        except Exception:
            base = 0
        if "NSE ALL" in choice:
            base = 2567 if base == 0 else base
        elif "BSE ALL" in choice:
            base = 4500 if base == 0 else base
        elif "FULL MARKET" in choice:
            base = 5900 if base == 0 else base
        label = f"{base} stocks"
        if base > 1000:
            label += " (~5-10 min)"
        if "FULL MARKET" in choice:
            label += " — full 5,900"
        self.universe_count_label.value = label + " ..."
        self.page.update()

        def _bg():
            try:
                from .universes import get_universe
                live = get_universe(choice)
                cnt = len(live)
                if cnt and cnt != base:
                    lbl = f"{cnt} stocks"
                    if cnt > 1000:
                        lbl += " (~5-10 min)"
                    if "FULL MARKET" in choice:
                        lbl += " — full 5,900"
                    self.universe_count_label.value = lbl
                    self.page.update()
            except Exception:
                pass
        threading.Thread(target=_bg, daemon=True).start()

    def _on_threshold_change(self, e):
        val = self.threshold_slider.value
        self.threshold_label.value = f"{int(val)}+"
        self.page.update()

    def _load_settings_to_ui(self):
        try:
            min_score = float(self.settings.get("min_score", 50))
        except (ValueError, TypeError):
            min_score = 50.0
        self.threshold_slider.value = min_score
        self.threshold_label.value = f"{int(min_score)}+"
        saved_universe = self.settings.get("universe", "NIFTY 50")
        self.universe_dd.value = saved_universe if saved_universe in UNIVERSES else "NIFTY 50"
        period_map = {"6mo": "6 Months", "1y": "1 Year", "2y": "2 Years"}
        self.period_dd.value = period_map.get(self.settings.get("data_period", "1y"), "1 Year")
        tf_map = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
        self.timeframe_dd.value = tf_map.get(self.settings.get("timeframe", "D"), "Daily")
        self.trend_filter_dd.value = self.settings.get("trend_filter", "All")

    def _collect_settings(self) -> dict:
        s = dict(self.settings)
        # min_score: tolerate empty/invalid slider values; fall back to default
        try:
            s["min_score"] = float(self.threshold_slider.value)
        except (ValueError, TypeError):
            s["min_score"] = float(self.settings.get("min_score", 50.0))
        # period dropdown: map display name → engine key; default to "1y"
        period_map: Dict[str, str] = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
        s["data_period"] = period_map.get(self.period_dd.value or "1 Year", "1y")
        # timeframe dropdown: map display name → engine key; default to "D"
        tf_map: Dict[str, str] = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        s["timeframe"] = tf_map.get(self.timeframe_dd.value or "Daily", "D")
        # trend_filter dropdown
        s["trend_filter"] = self.trend_filter_dd.value or "All"
        # universe dropdown
        s["universe"] = self.universe_dd.value or "NIFTY 50"
        return s

    # ── Scanning ───────────────────────────────────────────────────────

    def _on_action_click(self, e=None):
        if self.scanning:
            self._stop_scan()
        else:
            self._start_scan()

    def _start_scan(self, e=None):
        if self.scanning:
            return

        self.settings = self._collect_settings()
        save_settings(self.settings)
        self._apply_cache_settings()

        self.scanning = True
        self._scan_cancelled = False
        self._stop_requested = False
        c = self.theme_colors
        self.action_btn_label.value = "■  STOP"
        self.action_btn.bgcolor = c["red"]
        self.progress_bar.value = 0
        self.progress_label.value = "Starting…"
        self.status_label.value = "Status: Starting…"
        self.html_btn.disabled = True
        self.csv_btn.disabled = True
        self.clear_btn.disabled = True
        self.results = []
        self.all_results = []
        self.filtered_results = []

        self.table_column.controls.clear()
        self.table_column.controls.append(
            ft.Container(
                content=ft.Column([
                    ft.Icon(ft.Icons.SHOW_CHART, size=48, color=c["green"]),
                    ft.Text("Scanning — fetching batches…", size=12, weight=ft.FontWeight.BOLD, color=c["green"]),
                    ft.Text("First results appear after ~1 batch (~20s)", size=11, color=c["text_dim"]),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
                alignment=Alignment.CENTER,
                padding=30,
            )
        )
        self.page.update()

        threading.Thread(target=self._run_scan, daemon=True).start()

    def _stop_scan(self, e=None):
        if not self.scanning:
            return
        self._stop_requested = True
        engine = getattr(self, "_scan_engine", None)
        self.action_btn.disabled = True
        self.action_btn_label.value = "◷  STOPPING…"
        self.page.update()
        if engine is not None:
            engine.cancel()
            self._log("Stop requested — finishing the current batch, then stopping...")

    def _run_scan(self):
        try:
            from .scanner_engine import ScannerEngine

            universe_name = self.universe_dd.value or "NIFTY 50"
            settings = dict(self.settings)

            engine = ScannerEngine()
            self._scan_engine = engine
            if getattr(self, "_stop_requested", False):
                engine.cancel()
            engine.set_progress_callback(
                lambda p, m: self._safe_update(lambda: self._set_progress(p, m))
            )
            engine.set_log_callback(lambda m: self._safe_update(lambda: self._log(m)))

            def _on_batch(batch):
                self._on_stream_batch(batch)

            result = engine.scan_stream(
                universe=universe_name,
                settings=settings,
                period=settings.get("data_period", "1y"),
                timeframe=settings.get("timeframe", "D"),
                trend_filter=settings.get("trend_filter", "All"),
                index_symbol=settings.get("index_symbol", "NSEI"),
                on_batch=_on_batch,
            )

            self._scan_cancelled = result.cancelled

            def _final_sync():
                self.results = result.results
                self.all_results = list(result.results)
                self.filtered_results = [r for r in result.results if self._row_matches_filters(r)]
                self._render_current_page()
                if result.cancelled:
                    self._log(f"Scan stopped — showing {len(result.results)} partial results.")
                if result.error:
                    self._log(f"Scan finished with error: {result.error}")

            self._safe_update(_final_sync)

        except Exception as e:
            self._safe_update(lambda: self._log(f"\nERROR: {e!s}"))
        finally:
            self._safe_update(self._scan_complete)

    def _on_stream_batch(self, batch):
        if not batch:
            return
        with self._results_lock:
            existing = {r.get("ticker"): idx for idx, r in enumerate(self.all_results)}
            filtered_idx = {r.get("ticker"): idx for idx, r in enumerate(self.filtered_results)}
            for r in batch:
                t = r.get("ticker")
                if t in existing:
                    self.all_results[existing[t]] = r
                else:
                    self.all_results.append(r)
                    existing[t] = len(self.all_results) - 1
                if self._row_matches_filters(r):
                    if t in filtered_idx:
                        self.filtered_results[filtered_idx[t]] = r
                    else:
                        self.filtered_results.append(r)
                        filtered_idx[t] = len(self.filtered_results) - 1
                else:
                    if t in filtered_idx:
                        self.filtered_results.pop(filtered_idx[t])
                        filtered_idx = {fr.get("ticker"): i for i, fr in enumerate(self.filtered_results)}
            self.results = list(self.all_results)
        self._safe_update(lambda: self._render_current_page())

    def _scan_complete(self):
        self.scanning = False
        c = self.theme_colors
        self.action_btn.disabled = False
        self.action_btn_label.value = "▶  RUN SCAN"
        self.action_btn.bgcolor = c["green"]
        self.progress_label.value = "Stopped" if self._scan_cancelled else "Done"
        self.status_label.value = "Status: Stopped" if self._scan_cancelled else "Status: Done"
        if not self._scan_cancelled:
            self.progress_bar.value = 1.0
        self._refresh_neg_cache_ui()
        self._refresh_enrich_cache_ui()
        if self.results:
            self.html_btn.disabled = False
            self.csv_btn.disabled = False
            self.clear_btn.disabled = False
        self._update_hero_status(self.results)
        self.page.update()

    def _safe_update(self, fn: Callable[[], None]) -> None:
        """Run a UI mutation from any thread, then push it to the page.

        The provided callable ``fn`` is executed first; any exception is
        logged at debug level.  After ``fn()`` returns, ``page.update()``
        is called to flush the changes to the UI.

        This method is thread-safe and may be called from the scanner
        worker thread as well as from callback handlers (log, progress,
        stream-batch, completion, error).
        """
        try:
            fn()
        except Exception:  # pragma: no cover
            logger.debug("UI update callback failed", exc_info=True)
        try:
            self.page.update()
        except Exception:  # pragma: no cover
            pass

    # ── Results rendering ───────────────────────────────────────────────

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
            except Exception as e:
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
        self.hero_sub.value = txt

    # ── Cache management ───────────────────────────────────────────────

    def _refresh_neg_cache_ui(self):
        if not hasattr(self, "cache_status_lbl"):
            return
        try:
            from . import data_fetcher
            n = len(data_fetcher._negative_cache_load())
            ttl_h = data_fetcher.negative_cache_ttl_hours()
        except Exception:
            n, ttl_h = 0, 24
        self.cache_status_lbl.value = f"Dead-symbol cache: {n} (auto-resets ~{ttl_h}h)" if n else "Dead-symbol cache: empty"
        self.cache_clear_btn.visible = bool(n)

    def _clear_negative_cache(self, e=None):
        try:
            from . import data_fetcher
            data_fetcher._negative_cache_update(
                clears=list(data_fetcher._negative_cache_load().keys())
            )
            self._log("Cleared dead-symbol cache — fallback will re-attempt all symbols")
        except Exception as ex:
            self._log(f"Could not clear dead-symbol cache: {ex}")
        self._refresh_neg_cache_ui()
        self.page.update()

    def _refresh_enrich_cache_ui(self):
        if not hasattr(self, "enrich_cache_status_lbl"):
            return
        try:
            from . import data_fetcher
            n = data_fetcher.enrichment_cache_size()
            ttl_h = data_fetcher.ENRICHMENT_CACHE_TTL_HOURS
        except Exception:
            n, ttl_h = 0, 24
        self.enrich_cache_status_lbl.value = f"Enrichment cache: {n} (auto-resets ~{ttl_h}h)" if n else "Enrichment cache: empty"
        self.enrich_cache_clear_btn.visible = bool(n)

    def _clear_enrichment_cache(self, e=None):
        try:
            from . import data_fetcher
            data_fetcher.enrichment_cache_clear()
            self._log("Cleared enrichment cache — next scan will re-fetch phase-2 data")
        except Exception as ex:
            self._log(f"Could not clear enrichment cache: {ex}")
        self._refresh_enrich_cache_ui()
        self.page.update()

    # ── Export ──────────────────────────────────────────────────────────

    def _export_html(self, e=None):
        if not self.results:
            return
        threshold = self.settings.get("min_score", 50)
        tf_names = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
        tf_label = tf_names.get(self.settings.get("timeframe", "D"), "Daily")
        results_snapshot = list(self.results)
        universe_name = self.universe_dd.value or "NIFTY 50"
        safe_title = f"HMAxEMA Scanner — {universe_name} — {tf_label}"

        def _bg():
            try:
                self._log("Fetching news sentiment for exported stocks...")
                html = generate_html_report(results_snapshot, title=safe_title, threshold=threshold, fetch_news=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scanner_report_{timestamp}.html"
                filepath = os.path.join(SCANNER_DIR, filename)
                save_report(html, filepath)
                self._safe_update(lambda: self._log(f"HTML report saved: {filename}"))
                webbrowser.open(f"file://{os.path.abspath(filepath)}")
            except Exception as ex:
                logger.exception("HTML export failed: %s", ex)
                self._safe_update(lambda: self._log(f"HTML export failed: {ex}"))

        threading.Thread(target=_bg, daemon=True).start()

    def _export_csv(self, e=None):
        if not self.results:
            return
        import csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scanner_results_{timestamp}.csv"
        filepath = os.path.join(SCANNER_DIR, filename)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", "Ticker", "Score", "Rating", "Price", "Trend", "Momentum",
                             "RSI", "MACD", "Stoch", "OBV", "Volume", "RelStrength", "Volatility",
                             "Fundamentals", "Direction", "RSI_Val", "ADX", "Sideways",
                             "1M_Change", "3M_Change"])
            for i, r in enumerate(self.results, 1):
                sideways_reasons = ", ".join(r.get("sideways_reasons", []))
                writer.writerow([
                    i, r.get("ticker", ""), r.get("total", 0) or 0, r.get("combined_rating", "POOR"),
                    r.get("close"), r.get("trend"), r.get("momentum"), r.get("rsi"), r.get("macd"),
                    r.get("stoch"), r.get("obv"), r.get("volume"), r.get("rel_str"), r.get("volatility"),
                    r.get("fundamentals", 0), r.get("trend_dir", ""), r.get("rsi_val"), r.get("adx_val"),
                    ("Yes" + (f" ({sideways_reasons})" if sideways_reasons else "")) if r.get("is_sideways") else "No",
                    r.get("pc1m"), r.get("pc3m"),
                ])
        self._log(f"CSV saved: {filename}")
        self.page.update()

    # ── Utilities ───────────────────────────────────────────────────────

    def _clear_results(self, e=None):
        self.results = []
        self.all_results = []
        self.filtered_results = []
        self.table_column.controls.clear()
        self.chart_card.visible = False
        self.empty_label.visible = True
        self.table_column.controls.append(self.empty_label)
        self.result_count_label.value = "no scan yet"
        self._update_summary([])
        self._update_hero_status([])
        self._render_topicks([])
        self.progress_bar.value = 0
        self.progress_label.value = "Ready"
        self.status_label.value = "Status: Ready"
        self.html_btn.disabled = True
        self.csv_btn.disabled = True
        self.clear_btn.disabled = True
        self._log("Results cleared")
        self.page.update()

    def _switch_theme(self, e=None, to=None):
        new_theme = to or ("light" if self.current_theme == "dark" else "dark")
        if new_theme not in THEMES:
            new_theme = "dark"
        self.current_theme = new_theme
        self.theme_colors = THEMES[new_theme]
        self.settings["theme"] = new_theme
        save_settings(self.settings)
        had_results = bool(self.results)
        saved_log = self._get_log_lines()
        self._build_ui()
        self._load_settings_to_ui()
        self._set_log_lines(saved_log)
        if had_results:
            self._display_results(self.results)
        self._log(f"Theme switched to {new_theme}")
        self.page.update()

    # ── Settings page (indicator parameters used by the engine) ─────

    _MA_TYPES = ["HMA", "EMA", "SMA", "KAMA", "VWMA"]

    # (key, label, kind, min, max) — kind in {"int", "float", "ma_type", "theme"}.
    # Bounds are intentionally lenient: they only reject values the engine
    # itself cannot tolerate, so a stored setting can never brick Save.
    _SETTINGS_SPEC = [
        ("Signal — HMA×EMA crossover", [
            ("fast_ma_type", "Fast MA type", "ma_type", None, None),
            ("fast_ma_len", "Fast MA length", "int", 2, 500),
            ("slow_ma_type", "Slow MA type", "ma_type", None, None),
            ("slow_ma_len", "Slow MA length", "int", 2, 500),
            ("crossover_lookback", "Crossover lookback (bars)", "int", 1, 200),
        ]),
        ("Momentum & volume", [
            ("rsi_len", "RSI length", "int", 2, 200),
            ("vol_ma_len", "Volume MA length", "int", 2, 300),
            ("rs_length", "Relative-strength length", "int", 2, 200),
        ]),
        ("Trend strength & slope", [
            ("adx_len", "ADX length", "int", 2, 200),
            ("adx_threshold", "ADX threshold", "float", 0, 100),
            ("slope_ma_type", "Slope MA type", "ma_type", None, None),
            ("slope_ma_len", "Slope MA length", "int", 2, 500),
            ("slope_lookback", "Slope lookback (bars)", "int", 1, 200),
            ("flat_threshold", "Flat-slope threshold", "float", 0, 100),
        ]),
        ("Range / chop & ATR", [
            ("atr_len", "ATR length", "int", 2, 200),
            ("chop_len", "Chop length", "int", 2, 200),
            ("chop_threshold", "Chop threshold", "float", 0, 100),
        ]),
        ("Volume profile", [
            ("vp_lookback", "VP lookback (bars)", "int", 2, 1000),
            ("vp_rows", "VP rows", "int", 2, 200),
        ]),
        ("Output, cache & theme", [
            ("min_score", "Min score", "float", 0, 100),
            ("negative_cache_ttl_hours", "Dead-cache TTL (hours)", "float", 0, 1440),
            ("theme", "Theme", "theme", None, None),
        ]),
    ]

    def _settings_input(self, key, label, kind, lo, hi):
        c = self.theme_colors
        if kind == "ma_type":
            opts = list(self._MA_TYPES)
            cur = str(self.settings.get(key, opts[0]))
            ctrl = ft.Dropdown(
                options=[ft.dropdown.Option(v) for v in opts],
                value=cur if cur in opts else opts[0],
                width=150, height=40, text_size=13,
                bgcolor=c["option_bg"], color=c["text"],
                border_color=c["border"], border_width=1, border_radius=8,
                focused_border_color=c["purple"],
            )
        elif kind == "theme":
            opts = ["dark", "light"]
            cur = str(self.settings.get(key, "dark"))
            ctrl = ft.Dropdown(
                options=[ft.dropdown.Option(v) for v in opts],
                value=cur if cur in opts else "dark",
                width=150, height=40, text_size=13,
                bgcolor=c["option_bg"], color=c["text"],
                border_color=c["border"], border_width=1, border_radius=8,
                focused_border_color=c["purple"],
            )
        else:
            ctrl = ft.TextField(
                value=str(self.settings.get(key, "")), width=150, height=40, text_size=13,
                bgcolor=c["card"], color=c["text"],
                border_color=c["border"], border_width=1, border_radius=8,
                content_padding=_padding_only(left=10, right=8, top=6, bottom=6),
            )
        ctrl._settings_key = key
        ctrl._settings_kind = kind
        ctrl._settings_lo = lo
        ctrl._settings_hi = hi
        self._settings_inputs[key] = ctrl
        return ft.Row([
            ft.Text(label, size=12, color=c["text"], expand=True),
            ctrl,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def _build_settings_view(self):
        c = self.theme_colors
        self._settings_inputs = {}
        cards = []
        for title, fields in self._SETTINGS_SPEC:
            rows = [self._settings_input(key, label, kind, lo, hi)
                    for key, label, kind, lo, hi in fields]
            cards.append(
                ft.Container(
                    content=ft.Column([
                        ft.Text(title, size=12, weight=ft.FontWeight.BOLD, color=c["cyan"]),
                        ft.Divider(height=1, color=c["border"]),
                        *rows,
                    ], spacing=8),
                    bgcolor=_glass_bg(),
                    border=_glass_border(),
                    border_radius=14,
                    shadow=_card_shadow(),
                    padding=14,
                )
            )
        self._settings_error = ft.Text("", size=11, color=c["red"])
        header = ft.Container(
            content=ft.Column([
                ft.Text("⚙  Scanner Settings", size=18, weight=ft.FontWeight.BOLD, color=c["text"]),
                ft.Text("Indicator parameters consumed by the HMA×EMA engine — applied on Save.",
                        size=11, color=c["text_dim"]),
            ], spacing=2),
            padding=_padding_only(left=4, top=12, bottom=4),
        )
        body = ft.Column(
            controls=[*cards, self._settings_error],
            spacing=10, scroll=ft.ScrollMode.AUTO, expand=True,
        )
        footer = ft.Row([
            ft.Container(expand=True),
            ft.TextButton(content=ft.Text("Cancel", size=13),
                          on_click=lambda e: self._show_view("dashboard")),
            ft.ElevatedButton(content=ft.Text("Save Settings", size=13),
                               bgcolor=c["green"], color="#052e16",
                               on_click=lambda e: self._save_settings_page()),
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)
        return ft.Column(
            controls=[
                header, body,
                ft.Container(content=footer, padding=_padding_only(top=8, bottom=14, right=6)),
            ],
            spacing=0, expand=True,
        )

    def _save_settings_page(self, e=None):
        bad = []
        for key, ctrl in self._settings_inputs.items():
            kind = ctrl._settings_kind
            if kind in ("ma_type", "theme"):
                val = ctrl.value
                ok_opts = self._MA_TYPES if kind == "ma_type" else ["dark", "light"]
                if val not in ok_opts:
                    bad.append(key)
                    continue
            else:
                raw = (ctrl.value or "").strip()
                try:
                    val = int(float(raw)) if kind == "int" else float(raw)
                    lo, hi = ctrl._settings_lo, ctrl._settings_hi
                    if (lo is not None and val < lo) or (hi is not None and val > hi):
                        raise ValueError(key)
                except (ValueError, TypeError):
                    bad.append(key)
                    continue
            self.settings[key] = val
        if bad:
            self._settings_error.value = f"Invalid value: {', '.join(bad)}"
            self.page.update()
            return
        new_theme = self.settings.get("theme", self.current_theme)
        save_settings(self.settings)
        self._apply_cache_settings()
        self._log("Settings saved")
        if new_theme != self.current_theme:
            self._switch_theme(to=new_theme)
        else:
            self._load_settings_to_ui()
            self._show_view("dashboard")

    def _clear_log(self, e=None):
        try:
            if getattr(self, "log_column", None) is not None:
                self.log_column.controls.clear()
                self.page.update()
        except Exception:
            pass

    def _get_log_lines(self) -> list:
        try:
            col = getattr(self, "log_column", None)
            if col is None:
                return []
            return [t.value for t in col.controls if isinstance(t, ft.Text)]
        except Exception:
            return []

    def _set_log_lines(self, lines):
        try:
            col = getattr(self, "log_column", None)
            if col is None:
                return
            c = self.theme_colors
            col.controls.clear()
            for line in lines[-LOG_MAX_LINES:]:
                col.controls.append(self._make_log_line(line, c))
        except Exception:
            pass

    def _reset_filters(self, e=None):
        self.filter_text = ""
        if getattr(self, "search_entry", None) is not None:
            self.search_entry.value = ""
        if getattr(self, "rating_filter_dd", None) is not None:
            self.rating_filter_dd.value = "All"
        if self.all_results:
            self._display_results(self.all_results)
        else:
            self._render_current_page()
        self._scroll_to_top()
        self.page.update()

    def _make_log_line(self, text, c):
        return ft.Text(
            text, size=10, color=c["text_dim"], selectable=True,
            font_family="Consolas",
        )

    def _log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
        try:
            col = getattr(self, "log_column", None)
            if col is not None:
                col.controls.append(
                    self._make_log_line(line.rstrip("\n"), self.theme_colors)
                )
                del col.controls[:-LOG_MAX_LINES]
        except Exception:
            pass

    def _set_progress(self, value, text=""):
        self.progress_bar.value = value
        if text:
            self.progress_label.value = text
            self.status_label.value = f"Status: {text}"

    def _rotate_log(self):
        try:
            if os.path.exists(LOG_FILE):
                age_hours = (datetime.now().timestamp() - os.path.getmtime(LOG_FILE)) / 3600
                if age_hours >= LOG_ROTATE_HOURS:
                    open(LOG_FILE, "w").close()
        except Exception:
            pass


def main(page: ft.Page):
    app = ScannerApp(page)


if __name__ == "__main__":
    ft.run(main)
