"""
HMAxEMA Stock Scanner — GUI Application (v2 "Aurora" UI)
Modern dark desktop app for scanning Indian stocks.

Layout (inspired by community-app dashboards):
    [icon rail] [nav sidebar + scan controls] [main: hero, stats, results] [profile panel]

Usage:
    python scanner/app.py

Or double-click run.bat (Windows) / run.sh (macOS/Linux)
"""

import logging
import os
import sys
import threading
import tkinter as tk
import webbrowser
from datetime import datetime

import customtkinter as ctk

logger = logging.getLogger(__name__)

from .universes import UNIVERSES
from .data_fetcher import fetch_index_data, fetch_batch_yfinance, fetch_fundamentals
from .scoring import compute_scores, check_filter, get_direction
from .report import generate_html_report, save_report, _sentiment
from .settings_store import (
    DEFAULT_SETTINGS,
    SCANNER_DIR,
    SETTINGS_FILE,
    load_settings,
    save_settings,
)
from .themes import THEMES, apply_theme
from .widgets import AvatarRing, GradientCanvas, ToolTip

LOG_FILE = os.path.join(SCANNER_DIR, "scan.log")
LOG_ROTATE_HOURS = 12  # Overwrite log file after 12 hours


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

RESULT_COLS = [
    ("#", 35), ("Ticker", 100), ("Score", 50), ("Rating", 78), ("ENTRY", 55),
    ("Price", 80), ("MA", 62), ("T/15", 40), ("M/15", 40), ("R/8", 35),
    ("V/7", 35), ("Vol/10", 42), ("RS/10", 42), ("F/20", 42),
    ("1M", 55), ("Dir", 58), ("ADX", 40), ("Chop", 42),
]


class ScannerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("HMAxEMA Stock Scanner — Indian Market")
        self.geometry("1600x900")
        self.minsize(1280, 800)
        try:
            self.state("zoomed")  # Start maximized
        except Exception:
            pass

        self.settings = load_settings()
        self.results = []
        self.scanning = False
        self._cancel_scan = False
        self.filter_text = ""
        self.active_view = "dashboard"
        self.sort_col = None
        self.sort_dir = "asc"

        # Apply saved theme
        theme_name = self.settings.get("theme", "dark")
        ctk.set_appearance_mode(THEMES[theme_name]["ctk_mode"])
        self.current_theme = theme_name
        self.theme_colors = THEMES[theme_name]

        self._build_ui()
        self._load_settings_to_ui()
        self._rotate_log()

    # ════════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        c = self.theme_colors
        self.configure(fg_color=c["root_bg"])
        for w in self.winfo_children():
            w.destroy()
        self.filter_text = ""  # search entry is recreated empty

        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_rail()
        self._build_sidebar()
        self._build_main_area()
        self._build_right_panel()

    # ── Column 0: Icon rail ──────────────────────────────────────────────

    def _rail_icon(self, parent, kind, command, accent):
        """Circular rail button with a hand-drawn vector icon (font-independent)."""
        c = self.theme_colors
        cv = tk.Canvas(parent, width=40, height=40, highlightthickness=0, bd=0,
                       bg=c["rail_bg"], cursor="hand2")
        circle = cv.create_oval(3, 3, 37, 37, fill=c["card"], outline="")

        if kind == "home":
            cv.create_polygon(20, 9, 8.5, 19, 12, 19, 12, 30, 28, 30, 28, 19, 31.5, 19,
                              fill=accent, outline="")
            cv.create_rectangle(17, 23, 23, 30, fill=c["card"], outline="")
        elif kind == "gear":
            cx = cy = 20
            import math
            for k in range(8):
                a = math.radians(k * 45)
                x1, y1 = cx + 7.5 * math.cos(a), cy + 7.5 * math.sin(a)
                x2, y2 = cx + 13 * math.cos(a), cy + 13 * math.sin(a)
                cv.create_line(x1, y1, x2, y2, fill=accent, width=3.5)
            cv.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, outline=accent, width=3)
        elif kind == "play":
            cv.create_polygon(15.5, 11.5, 29.5, 20, 15.5, 28.5, fill=accent, outline="")
        elif kind == "moon":
            cv.create_oval(12, 10, 28, 28, fill=accent, outline="")
            cv.create_oval(18, 8, 33, 25, fill=c["card"], outline="")
        elif kind == "sun":
            cx = cy = 20
            import math
            for k in range(8):
                a = math.radians(k * 45)
                cv.create_line(cx + 8.5 * math.cos(a), cy + 8.5 * math.sin(a),
                               cx + 12.5 * math.cos(a), cy + 12.5 * math.sin(a),
                               fill=accent, width=2.5)
            cv.create_oval(cx - 6.5, cy - 6.5, cx + 6.5, cy + 6.5,
                           fill=accent, outline="")

        def _enter(_e):
            cv.itemconfigure(circle, fill=c["card2"])

        def _leave(_e):
            cv.itemconfigure(circle, fill=c["card"])

        cv.bind("<Enter>", _enter)
        cv.bind("<Leave>", _leave)
        cv.bind("<Button-1>", lambda e: command())
        return cv

    def _build_rail(self):
        c = self.theme_colors
        rail = ctk.CTkFrame(self, width=64, corner_radius=0, fg_color=c["rail_bg"])
        rail.grid(row=0, column=0, sticky="nsw")
        rail.grid_propagate(False)
        rail.grid_rowconfigure(10, weight=1)

        # Logo
        logo = tk.Canvas(rail, width=44, height=44, highlightthickness=0, bd=0,
                         bg=c["rail_bg"])
        logo.grid(row=0, column=0, pady=(14, 16))
        lr = 21
        logo.create_oval(2, 2, 2 + lr * 2, 2 + lr * 2, fill=c["purple"], outline="")
        logo.create_text(2 + lr, 2 + lr, text="H", fill="white",
                         font=("Segoe UI", 17, "bold"))

        # Rail icon buttons with active pill
        self.rail_pills = {}
        self.rail_buttons = {}

        def add_rail_item(row, kind, view_or_cmd, accent):
            holder = ctk.CTkFrame(rail, fg_color="transparent", height=46, width=64)
            holder.grid(row=row, column=0, pady=4, padx=0, sticky="ew")
            holder.grid_propagate(False)
            holder.grid_columnconfigure(1, weight=1)
            pill = ctk.CTkFrame(holder, width=4, height=26, corner_radius=2,
                                fg_color=c["pink"])
            pill.grid(row=0, column=0, padx=(6, 4), pady=10, sticky="n")
            pill.grid_remove()

            def _cmd():
                if isinstance(view_or_cmd, str):
                    self._show_view(view_or_cmd)
                else:
                    view_or_cmd()

            icon = self._rail_icon(holder, kind, _cmd, accent)
            icon.grid(row=0, column=1, padx=(0, 10))
            return pill, icon

        p1, b1 = add_rail_item(1, "home", "dashboard", c["cyan"])
        p2, b2 = add_rail_item(2, "gear", "settings", c["purple"])
        _, b3 = add_rail_item(3, "play", self._start_scan, c["green"])
        self.rail_pills = {"dashboard": p1, "settings": p2}
        self.rail_buttons = {"dashboard": b1, "settings": b2}

        # Theme toggle pinned near bottom
        theme_kind = "sun" if self.current_theme == "dark" else "moon"
        self.theme_btn = self._rail_icon(rail, theme_kind, self._switch_theme, c["orange"])
        self.theme_btn.grid(row=11, column=0, pady=(0, 10), sticky="n")

    # ── Column 1: Nav sidebar + scan controls ────────────────────────────

    def _nav_item(self, parent, icon, label, view):
        c = self.theme_colors
        active = (view == self.active_view)
        item = ctk.CTkButton(
            parent, text=f"   {icon}   {label}",
            anchor="w", height=40, corner_radius=12,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold" if active else "normal"),
            fg_color=c["nav_active"] if active else "transparent",
            hover_color=c["card"] if not active else c["nav_active"],
            text_color=c["text"] if active else c["text_dim"],
            command=lambda: self._show_view(view))
        item.pack(fill="x", padx=10, pady=3)
        return item

    def _build_sidebar(self):
        c = self.theme_colors
        side = ctk.CTkFrame(self, width=210, corner_radius=0, fg_color=c["side_bg"])
        side.grid(row=0, column=1, sticky="nsw")
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)
        for r in range(4):
            side.grid_rowconfigure(r, weight=0)
        side.grid_rowconfigure(3, weight=1)   # controls scroll
        side.grid_rowconfigure(4, weight=0)   # run + progress pinned bottom

        # Header
        hdr = ctk.CTkFrame(side, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 6))
        ctk.CTkLabel(hdr, text="Scanner",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=c["text"]).pack(anchor="w")
        ctk.CTkLabel(hdr, text="Indian Market Screener",
                     font=ctk.CTkFont(size=11),
                     text_color=c["text_dim"]).pack(anchor="w")

        # Nav items
        nav = ctk.CTkFrame(side, fg_color="transparent")
        nav.grid(row=1, column=0, sticky="ew", pady=(8, 4))
        self.nav_items = {
            "dashboard": self._nav_item(nav, "\U0001F3E0", "Dashboard", "dashboard"),
            "settings": self._nav_item(nav, "\u2699", "Settings", "settings"),
        }

        div = ctk.CTkFrame(side, height=1, fg_color=c["border"])
        div.grid(row=2, column=0, sticky="ew", padx=14, pady=(6, 0))

        # Scan controls (scrollable)
        ctrl = ctk.CTkScrollableFrame(side, fg_color="transparent")
        ctrl.grid(row=3, column=0, sticky="nsew")

        def section_label(text):
            ctk.CTkLabel(ctrl, text=text.upper(),
                         font=ctk.CTkFont(size=10, weight="bold"),
                         text_color=c["cyan"]).pack(padx=14, pady=(12, 2), anchor="w")

        # Universe
        section_label("Stock Universe")
        self.universe_var = ctk.StringVar(value="NIFTY 50")
        self.universe_menu = ctk.CTkOptionMenu(
            ctrl, variable=self.universe_var, values=list(UNIVERSES.keys()),
            command=self._on_universe_change,
            width=180, height=32, corner_radius=10,
            fg_color=c["option_bg"], button_color=c["option_btn"],
            text_color=c["text"],
            button_hover_color=c["purple"], dropdown_fg_color=c["option_drop"],
            dropdown_hover_color=c["nav_active"])
        self.universe_menu.pack(padx=14, fill="x")
        self.universe_count_label = ctk.CTkLabel(
            ctrl, text=f"{len(UNIVERSES['NIFTY 50'])} stocks",
            font=ctk.CTkFont(size=10), text_color=c["text_dim"])
        self.universe_count_label.pack(padx=16, anchor="w", pady=(2, 0))

        # Timeframe
        section_label("Timeframe")
        self.timeframe_var = ctk.StringVar(value="Daily")
        ctk.CTkOptionMenu(
            ctrl, variable=self.timeframe_var,
            values=["Daily", "Weekly", "Monthly"],
            width=180, height=32, corner_radius=10,
            fg_color=c["option_bg"], button_color=c["option_btn"],
            text_color=c["text"],
            button_hover_color=c["purple"], dropdown_fg_color=c["option_drop"],
            dropdown_hover_color=c["nav_active"]).pack(padx=14, fill="x")

        # Data Period
        section_label("Data Period")
        self.period_var = ctk.StringVar(value="1 Year")
        ctk.CTkOptionMenu(
            ctrl, variable=self.period_var,
            values=["6 Months", "1 Year", "2 Years"],
            width=180, height=32, corner_radius=10,
            fg_color=c["option_bg"], button_color=c["option_btn"],
            text_color=c["text"],
            button_hover_color=c["purple"], dropdown_fg_color=c["option_drop"],
            dropdown_hover_color=c["nav_active"]).pack(padx=14, fill="x")

        # Trend Filter
        section_label("Trend Filter")
        self.trend_filter_var = ctk.StringVar(value="All")
        ctk.CTkOptionMenu(
            ctrl, variable=self.trend_filter_var,
            values=["All", "Bullish Only", "Bearish Only"],
            width=180, height=32, corner_radius=10,
            fg_color=c["option_bg"], button_color=c["option_btn"],
            text_color=c["text"],
            button_hover_color=c["purple"], dropdown_fg_color=c["option_drop"],
            dropdown_hover_color=c["nav_active"]).pack(padx=14, fill="x")

        # Score Threshold
        section_label("Min Score Threshold")
        tf_frame = ctk.CTkFrame(ctrl, fg_color="transparent")
        tf_frame.pack(padx=14, pady=(2, 8), fill="x")
        self.threshold_var = ctk.DoubleVar(value=50.0)
        self.threshold_slider = ctk.CTkSlider(
            tf_frame, from_=0, to=100, variable=self.threshold_var,
            number_of_steps=20, command=self._on_threshold_change,
            width=120, height=16, button_color=c["purple"],
            progress_color=c["purple"])
        self.threshold_slider.pack(side="left")
        self.threshold_label = ctk.CTkLabel(
            tf_frame, text="50", width=34,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=c["pink"])
        self.threshold_label.pack(side="right")

        # Pinned bottom: Run + progress
        bottom = ctk.CTkFrame(side, fg_color="transparent")
        bottom.grid(row=4, column=0, sticky="sew", padx=12, pady=(6, 14))

        self.run_btn = ctk.CTkButton(
            bottom, text="\u25b6   RUN SCAN",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=c["purple"], hover_color=c["purple_hover"],
            text_color="#04220f", height=44, corner_radius=14,
            command=self._start_scan)
        self.run_btn.pack(fill="x", pady=(0, 8))

        self.progress = ctk.CTkProgressBar(
            bottom, height=6, corner_radius=3,
            fg_color=c["progress_bg"], progress_color=c["progress_fg"])
        self.progress.pack(fill="x")
        self.progress.set(0)

        self.progress_label = ctk.CTkLabel(
            bottom, text="Ready", font=ctk.CTkFont(size=10),
            text_color=c["text_dim"], anchor="w")
        self.progress_label.pack(fill="x", pady=(2, 0))

    # ── Column 2: Main area (topbar + views) ─────────────────────────────

    def _build_main_area(self):
        c = self.theme_colors
        main = ctk.CTkFrame(self, fg_color=c["main_bg"], corner_radius=0)
        main.grid(row=0, column=2, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)
        self._main = main

        self._build_topbar(main)

        # View container
        self.view_host = ctk.CTkFrame(main, fg_color="transparent")
        self.view_host.grid(row=1, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
        self.view_host.grid_rowconfigure(0, weight=1)
        self.view_host.grid_columnconfigure(0, weight=1)

        self.views = {}
        self._build_dashboard_view()
        self._build_settings_view()
        self._show_view(self.active_view)

    def _build_topbar(self, parent):
        c = self.theme_colors
        bar = ctk.CTkFrame(parent, fg_color=c["panel_bg"], corner_radius=0, height=56)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(1, weight=1)

        # Search pill (left-center)
        search_holder = ctk.CTkFrame(bar, fg_color="transparent")
        search_holder.grid(row=0, column=0, sticky="w", padx=16)
        self.search_entry = ctk.CTkEntry(
            search_holder, placeholder_text="\u2315  Filter by ticker\u2026",
            width=240, height=34, corner_radius=17,
            fg_color=c["card"], border_color=c["border"], border_width=1,
            text_color=c["text"],
            font=ctk.CTkFont(family="Segoe UI", size=12))
        self.search_entry.pack(side="left")
        self.search_entry.bind("<KeyRelease>", self._on_search_change)

        # Right action pills
        actions = ctk.CTkFrame(bar, fg_color="transparent")
        actions.grid(row=0, column=2, sticky="e", padx=14)

        def action_btn(parent_, text, cmd, color, state="normal"):
            return ctk.CTkButton(
                parent_, text=text, width=36, height=34, corner_radius=17,
                fg_color=c["card"], hover_color=c["card2"],
                text_color=color, font=ctk.CTkFont(family="Segoe UI", size=14),
                command=cmd, state=state)

        self.html_btn = action_btn(actions, "\u2913", self._export_html, c["cyan"], "disabled")
        self.html_btn.pack(side="left", padx=3)
        ToolTip(self.html_btn, "Export HTML report")

        self.csv_btn = action_btn(actions, "\u2912", self._export_csv, c["blue"], "disabled")
        self.csv_btn.pack(side="left", padx=3)
        ToolTip(self.csv_btn, "Export CSV")

        self.clear_btn = action_btn(actions, "\u2715", self._clear_results, c["red"], "disabled")
        self.clear_btn.pack(side="left", padx=3)
        ToolTip(self.clear_btn, "Clear results")

    # ── Dashboard view ───────────────────────────────────────────────────

    def _build_dashboard_view(self):
        c = self.theme_colors
        view = ctk.CTkScrollableFrame(self.view_host, fg_color="transparent")
        view.grid(row=0, column=0, sticky="nsew")
        view.grid_columnconfigure(0, weight=1)
        self.views["dashboard"] = view
        self.table_scroll = view  # table lives inside this scrollable view

        # Hero banner (compact)
        hero = GradientCanvas(view, height=104, highlightthickness=0, bd=0)
        hero.grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 6))
        hero.set_gradient(c["hero_grad"], horizontal=True)
        hero.bind("<Configure>", lambda e: hero._paint())
        hero.create_text(26, 34, anchor="w", fill="#ffffff",
                         font=("Segoe UI", 21, "bold"),
                         text="Find Your Next Swing Trade")
        self.hero_sub = hero.create_text(26, 66, anchor="w", fill="#d8ffe8",
                                          font=("Segoe UI", 11),
                                          text="Set your universe on the left, then RUN SCAN \u2014 HMA\u00d7EMA crossover \u2022 10-factor score \u2022 news sentiment")
        self.hero_canvas = hero
        self.hero_title_id = None

        # Stats row
        stats_wrap = ctk.CTkFrame(view, fg_color="transparent")
        stats_wrap.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 8))
        self.summary_inner = stats_wrap
        self.summary_cards = {}
        self._build_summary_panel(stats_wrap)

        # Section header: Results
        sec = ctk.CTkFrame(view, fg_color="transparent")
        sec.grid(row=2, column=0, sticky="ew", padx=6)
        ctk.CTkLabel(sec, text="Scan Results",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=c["text"]).pack(side="left")
        self.result_count_label = ctk.CTkLabel(
            sec, text="no scan yet",
            font=ctk.CTkFont(size=11), text_color=c["text_dim"])
        self.result_count_label.pack(side="right")

        # Table (header bar + rows live here; rebuilt on every display/filter change)
        self.table_frame = ctk.CTkFrame(view, fg_color="transparent")
        self.table_frame.grid(row=3, column=0, sticky="ew", padx=6, pady=(6, 20))
        self.table_frame.grid_columnconfigure(0, weight=1)

        self._render_table_header()

        # Empty-state hint
        self.empty_label = ctk.CTkLabel(
            self.table_frame,
            text="\nNo results yet \u2014 hit \u25b6 RUN SCAN\n",
            font=ctk.CTkFont(size=13), text_color=c["text_dim"])
        self.empty_label.pack(pady=30, anchor="center")

    _SORT_KEYS = {
        1:  lambda r: r.get("ticker", ""),
        2:  lambda r: r.get("total", 0),
        3:  lambda r: r.get("combined_rating", "POOR"),
        5:  lambda r: r.get("close", 0),
        7:  lambda r: r.get("trend", 0),
        8:  lambda r: r.get("momentum", 0),
        9:  lambda r: r.get("rsi", 0),
        10: lambda r: r.get("macd", 0),
        11: lambda r: r.get("volume", 0),
        12: lambda r: r.get("rel_str", 0),
        13: lambda r: r.get("fundamentals", 0),
        14: lambda r: r.get("pc1m", 0) or 0,
        15: lambda r: r.get("trend_dir", ""),
        16: lambda r: r.get("adx_val", 0) or 0,
    }

    def _on_header_click(self, col_idx):
        if col_idx not in self._SORT_KEYS:
            return
        if self.sort_col == col_idx:
            self.sort_dir = "desc" if self.sort_dir == "asc" else "asc"
        else:
            self.sort_col = col_idx
            self.sort_dir = "desc" if col_idx in (2, 7, 8, 9, 10, 11, 12, 13, 14, 16) else "asc"
        key_fn = self._SORT_KEYS[col_idx]
        self.results.sort(key=key_fn, reverse=(self.sort_dir == "desc"))
        self._display_results(self.results)

    def _render_table_header(self):
        """Render the compact column-header bar as the first row of table_frame."""
        c = self.theme_colors
        hdr = ctk.CTkFrame(self.table_frame, fg_color=c["card"], corner_radius=8,
                           height=28)
        hdr.pack(fill="x", pady=(0, 4))
        hdr.pack_propagate(False)
        hdr.grid_propagate(False)
        for col_idx, (text, width) in enumerate(RESULT_COLS):
            if self.sort_col == col_idx:
                arrow = " \u25b2" if self.sort_dir == "asc" else " \u25bc"
            else:
                arrow = ""
            lbl = ctk.CTkLabel(hdr, text=text + arrow, width=width, anchor="w",
                               font=ctk.CTkFont(size=10, weight="bold"),
                               text_color=c["cyan"], cursor="hand2")
            lbl.pack(side="left", padx=1)
            if col_idx in self._SORT_KEYS:
                lbl.bind("<Button-1>", lambda e, ci=col_idx: self._on_header_click(ci))

    def _build_summary_panel(self, parent):
        """Build the summary statistic cards (values updated after a scan)."""
        c = self.theme_colors
        stats = [
            ("TOTAL", "total", c["cyan"]),
            ("PASSED", "passed", c["green"]),
            ("ENTRY", "entry", c["pink"]),
            ("AVG", "avg", c["lime"]),
            ("HIGH", "high", c["green"]),
            ("BULL", "bull", c["green"]),
            ("BEAR", "bear", c["red"]),
        ]
        self.summary_cards = {}
        for label, key, color in stats:
            card = ctk.CTkFrame(parent, fg_color=c["card"], corner_radius=12,
                                border_width=1, border_color=c["border"])
            card.pack(side="left", fill="both", expand=True, padx=4, pady=2)
            val_label = ctk.CTkLabel(card, text="\u2014",
                                     font=ctk.CTkFont(size=20, weight="bold"),
                                     text_color=color)
            val_label.pack(pady=(8, 0))
            ctk.CTkLabel(card, text=label,
                         font=ctk.CTkFont(size=9, weight="bold"),
                         text_color=c["text_dim"]).pack(pady=(0, 8))
            self.summary_cards[key] = val_label

    # ── Settings view ────────────────────────────────────────────────────

    def _build_settings_view(self):
        c = self.theme_colors
        view = ctk.CTkScrollableFrame(self.view_host, fg_color="transparent")
        view.grid(row=0, column=0, sticky="nsew")
        view.grid_remove()
        self.views["settings"] = view

        # Full-width content wrapper (pack-based → fills scrollable viewport)
        wrap = ctk.CTkFrame(view, fg_color="transparent")
        wrap.pack(fill="x", padx=24, pady=(10, 28))

        # Header
        hdr = ctk.CTkFrame(wrap, fg_color="transparent")
        hdr.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(hdr, text="Advanced Settings",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=c["text"]).pack(side="left")
        ctk.CTkLabel(hdr, text="indicator parameters \u00b7 saved to settings.json on scan",
                     font=ctk.CTkFont(size=11),
                     text_color=c["text_dim"]).pack(side="right")

        # Section cards
        self.setting_widgets = {}
        self._build_settings_panel(wrap)

        # Actions row
        btns = ctk.CTkFrame(wrap, fg_color="transparent")
        btns.pack(fill="x", pady=(4, 0))
        ctk.CTkButton(btns, text="Clear Cache", width=140, height=38,
                      corner_radius=12, fg_color=c["card"],
                      hover_color=c["card2"], border_width=1,
                      border_color=c["border"], text_color=c["text"],
                      command=self._clear_cache).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btns, text="Reset Settings", width=140, height=38,
                      corner_radius=12, fg_color=c["card"],
                      hover_color=c["card2"], border_width=1,
                      border_color=c["border"], text_color=c["orange"],
                      command=self._reset_settings).pack(side="left", padx=10)

    def _build_settings_panel(self, parent):
        """Build the advanced settings form."""
        sections = [
            ("Moving Averages (Crossover)", [
                ("Fast MA Type", "fast_ma_type", "option", ["HMA", "EMA", "SMA", "KAMA", "VWMA"]),
                ("Fast MA Length", "fast_ma_len", "int", (5, 100)),
                ("Slow MA Type", "slow_ma_type", "option", ["HMA", "EMA", "SMA", "KAMA", "VWMA"]),
                ("Slow MA Length", "slow_ma_len", "int", (10, 200)),
                ("Crossover Lookback", "crossover_lookback", "int", (1, 100)),
            ]),
            ("Technical Analysis", [
                ("RSI Length", "rsi_len", "int", (5, 50)),
                ("ATR Length", "atr_len", "int", (5, 50)),
                ("Volume MA Length", "vol_ma_len", "int", (5, 100)),
            ]),
            ("Relative Strength", [
                ("RS Length", "rs_length", "int", (5, 50)),
                ("Index Symbol", "index_symbol", "text", None),
            ]),
            ("Volume Profile", [
                ("VP Lookback (Candles)", "vp_lookback", "int", (10, 500)),
                ("VP Rows/Bins", "vp_rows", "int", (5, 100)),
                ("VP Width (Bars)", "vp_width", "int", (10, 100)),
            ]),
            ("Sideways Filter", [
                ("ADX Length", "adx_len", "int", (5, 50)),
                ("ADX Threshold", "adx_threshold", "float", (5.0, 50.0)),
                ("Chop Length", "chop_len", "int", (5, 50)),
                ("Chop Threshold", "chop_threshold", "float", (30.0, 90.0)),
                ("Slope MA Type", "slope_ma_type", "option", ["HMA", "EMA", "SMA", "KAMA", "VWMA"]),
                ("Slope MA Length", "slope_ma_len", "int", (10, 200)),
                ("Slope Lookback", "slope_lookback", "int", (3, 50)),
                ("Flat Threshold %", "flat_threshold", "float", (0.1, 5.0)),
            ]),
            ("Step Channel", [
                ("Pivot Length", "sc_pivot_len", "int", (1, 20)),
                ("Bands Multiplier", "sc_bands_mult", "float", (0.1, 3.0)),
            ]),
        ]

        c = self.theme_colors
        cols = 3
        for idx, (section_title, fields) in enumerate(sections):
            # Section card
            card = ctk.CTkFrame(parent, fg_color=c["card"], corner_radius=14,
                                border_width=1, border_color=c["border"])
            card.pack(fill="x", pady=(0, 14))

            # Section header strip
            head = ctk.CTkFrame(card, fg_color=c["card2"], corner_radius=14, height=38)
            head.pack(fill="x")
            head.pack_propagate(False)
            ctk.CTkLabel(head, text=f"  \u25b8  {section_title.upper()}",
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=c["green"]).pack(side="left", padx=10)
            ctk.CTkLabel(head, text=f"{len(fields)} params   ",
                         font=ctk.CTkFont(size=10),
                         text_color=c["text_dim"]).pack(side="right")

            # Fields grid
            body = ctk.CTkFrame(card, fg_color="transparent")
            body.pack(fill="x", padx=18, pady=(10, 14))
            for col in range(cols):
                body.grid_columnconfigure(col, weight=1, uniform=f"sec{idx}")

            for fi, (label, key, field_type, constraints) in enumerate(fields):
                cell = ctk.CTkFrame(body, fg_color="transparent", height=34)
                cell.grid(row=fi // cols, column=fi % cols, sticky="ew",
                          padx=(0, 22), pady=4)
                cell.pack_propagate(False)
                cell.grid_columnconfigure(0, weight=1)

                ctk.CTkLabel(cell, text=label,
                             font=ctk.CTkFont(size=12),
                             text_color=c["text"], anchor="w").grid(row=0, column=0, sticky="ew")

                if field_type == "option":
                    var = ctk.StringVar(value=str(self.settings.get(key, "")))
                    widget = ctk.CTkOptionMenu(
                        cell, variable=var, values=constraints,
                        width=120, height=30, corner_radius=8,
                        fg_color=c["option_bg"], button_color=c["option_btn"],
            text_color=c["text"],
                        button_hover_color=c["purple"],
                        dropdown_fg_color=c["option_drop"],
                        dropdown_hover_color=c["nav_active"],
                        font=ctk.CTkFont(size=11))
                    widget.grid(row=0, column=1)
                elif field_type == "int":
                    var = ctk.StringVar(value=str(self.settings.get(key, 0)))
                    widget = ctk.CTkEntry(
                        cell, textvariable=var, width=90, height=30, corner_radius=8,
                        fg_color=c["entry_bg"], border_color=c["entry_border"],
                        text_color=c["text"],
                        font=ctk.CTkFont(size=11))
                    widget.grid(row=0, column=1)
                elif field_type == "float":
                    var = ctk.StringVar(value=str(self.settings.get(key, 0.0)))
                    widget = ctk.CTkEntry(
                        cell, textvariable=var, width=90, height=30, corner_radius=8,
                        fg_color=c["entry_bg"], border_color=c["entry_border"],
                        text_color=c["text"],
                        font=ctk.CTkFont(size=11))
                    widget.grid(row=0, column=1)
                elif field_type == "text":
                    var = ctk.StringVar(value=str(self.settings.get(key, "")))
                    widget = ctk.CTkEntry(
                        cell, textvariable=var, width=130, height=30, corner_radius=8,
                        fg_color=c["entry_bg"], border_color=c["entry_border"],
                        text_color=c["text"],
                        font=ctk.CTkFont(size=10))
                    widget.grid(row=0, column=1)

                self.setting_widgets[key] = (var, field_type, constraints)

    # ── Column 3: Profile panel ──────────────────────────────────────────

    def _build_right_panel(self):
        c = self.theme_colors
        panel = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color=c["side_bg"])
        panel.grid(row=0, column=3, sticky="nse")
        panel.grid_propagate(False)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(3, weight=0)
        panel.grid_rowconfigure(5, weight=1)
        self._right_panel = panel

        # Profile block
        prof = ctk.CTkFrame(panel, fg_color="transparent")
        prof.grid(row=0, column=0, pady=(18, 4))
        AvatarRing(prof, size=86, letter="H", bg=c["side_bg"]).pack()
        ctk.CTkLabel(prof, text="HMAxEMA Scanner",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=c["text"]).pack(pady=(8, 0))
        ctk.CTkLabel(prof, text="@indian_markets",
                     font=ctk.CTkFont(size=10),
                     text_color=c["text_dim"]).pack()

        # Progress mirror
        pmirror = ctk.CTkFrame(panel, fg_color="transparent")
        pmirror.grid(row=1, column=0, sticky="ew", padx=18, pady=(10, 0))
        self.status_label = ctk.CTkLabel(pmirror, text="Status: Ready",
                                         font=ctk.CTkFont(size=10),
                                         text_color=c["text_dim"], anchor="w")
        self.status_label.pack(fill="x")

        # Top Picks
        tp_hdr = ctk.CTkFrame(panel, fg_color="transparent")
        tp_hdr.grid(row=2, column=0, sticky="ew", padx=16, pady=(14, 4))
        ctk.CTkLabel(tp_hdr, text="Top Picks",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=c["text"]).pack(side="left")
        ctk.CTkLabel(tp_hdr, text="top 5",
                     font=ctk.CTkFont(size=10),
                     text_color=c["text_dim"]).pack(side="right")
        self.topicks_frame = ctk.CTkFrame(panel, fg_color="transparent")
        self.topicks_frame.grid(row=3, column=0, sticky="ew", padx=10)
        self._render_topicks([])

        # Recent Activity (log feed)
        ra_hdr = ctk.CTkFrame(panel, fg_color="transparent")
        ra_hdr.grid(row=4, column=0, sticky="ew", padx=16, pady=(12, 4))
        ctk.CTkLabel(ra_hdr, text="Recent Activity",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=c["text"]).pack(side="left")
        ctk.CTkLabel(ra_hdr, text="live log",
                     font=ctk.CTkFont(size=10),
                     text_color=c["text_dim"]).pack(side="right")

        self.log_text = ctk.CTkTextbox(
            panel, fg_color=c["panel_bg"], corner_radius=12,
            text_color=c["text_dim"],
            font=ctk.CTkFont(family="Consolas", size=10),
            border_width=1, border_color=c["border"],
            state="normal")
        self.log_text.grid(row=5, column=0, sticky="nsew", padx=10, pady=(0, 12))
        self.log_text.bind("<Control-a>", lambda e: self.log_text.tag_add("sel", "1.0", "end"))
        self.log_text.bind("<Control-c>", lambda e: self._copy_selection())

    def _render_topicks(self, top):
        c = self.theme_colors
        for widget in self.topicks_frame.winfo_children():
            widget.destroy()
        if not top:
            ctk.CTkLabel(self.topicks_frame, text="Run a scan to see leaders",
                         font=ctk.CTkFont(size=10),
                         text_color=c["text_dim"]).pack(pady=6)
            return
        threshold = self.settings.get("min_score", 50)
        for i, r in enumerate(top[:5], 1):
            score = r["total"]
            color = (c["green"] if score >= 70 else
                     c["lime"] if score >= 50 else
                     c["orange"] if score >= 30 else c["red"])
            card = ctk.CTkFrame(self.topicks_frame, fg_color=c["card"],
                                corner_radius=12, height=44,
                                border_width=1, border_color=c["border"])
            card.pack(fill="x", pady=3)
            card.pack_propagate(False)
            badge = ctk.CTkLabel(card, text=str(i), width=26, height=26,
                                 corner_radius=13, fg_color=c["card2"],
                                 text_color=color,
                                 font=ctk.CTkFont(size=11, weight="bold"))
            badge.pack(side="left", padx=(10, 6), pady=9)
            info = ctk.CTkFrame(card, fg_color="transparent")
            info.pack(side="left", fill="y", expand=True)
            ctk.CTkLabel(info, text=r["ticker"],
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=c["text"], anchor="w").pack(anchor="w", pady=(8, 0))
            ctk.CTkLabel(info, text=f"\u20b9{r.get('close', 0):,.0f}  \u00b7  {r['trend_dir']}",
                         font=ctk.CTkFont(size=9),
                         text_color=c["text_dim"], anchor="w").pack(anchor="w")
            sc_lbl = ctk.CTkLabel(card, text=f"{score:.0f}",
                                  font=ctk.CTkFont(size=15, weight="bold"),
                                  text_color=color)
            sc_lbl.pack(side="right", padx=12)
            if score >= threshold:
                ctk.CTkLabel(card, text="\u25cf", width=10,
                             text_color=c["green"],
                             font=ctk.CTkFont(size=9)).pack(side="right")

    # ── View switching / search ──────────────────────────────────────────

    def _show_view(self, name: str):
        self.active_view = name
        for vname, frame in self.views.items():
            if vname == name:
                frame.grid()
            else:
                frame.grid_remove()
        # Refresh nav + rail highlights
        c = self.theme_colors
        for vname, item in getattr(self, "nav_items", {}).items():
            active = (vname == name)
            item.configure(
                font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold" if active else "normal"),
                fg_color=c["nav_active"] if active else "transparent",
                hover_color=c["nav_active"] if active else c["card"],
                text_color=c["text"] if active else c["text_dim"])
        for vname, pill in getattr(self, "rail_pills", {}).items():
            if pill is not None:
                (pill.grid if vname == name else pill.grid_remove)()

    def _on_search_change(self, event=None):
        if event and getattr(event, "keysym", "") in ("Up", "Down", "Return"):
            return
        self.filter_text = self.search_entry.get().strip().upper()
        self._display_results(self.results)

    # ── Settings Management ──────────────────────────────────────────────

    def _load_settings_to_ui(self):
        """Load saved settings into UI widgets."""
        self.threshold_slider.set(self.settings.get("min_score", 50))
        self.threshold_label.configure(text=str(int(self.settings.get("min_score", 50))))

        period_map = {"6mo": "6 Months", "1y": "1 Year", "2y": "2 Years"}
        self.period_var.set(period_map.get(self.settings.get("data_period", "1y"), "1 Year"))

        tf_map = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
        self.timeframe_var.set(tf_map.get(self.settings.get("timeframe", "D"), "Daily"))

        self.trend_filter_var.set(self.settings.get("trend_filter", "All"))

        # Advanced widgets
        if hasattr(self, "setting_widgets"):
            for key, (var, field_type, constraints) in self.setting_widgets.items():
                var.set(str(self.settings.get(key, "")))

    def _collect_settings(self) -> dict:
        """Read all settings from UI widgets."""
        s = {}
        for key, (var, field_type, constraints) in self.setting_widgets.items():
            val = var.get()
            if field_type == "int":
                try:
                    s[key] = int(val)
                except ValueError:
                    s[key] = self.settings.get(key, 0)
            elif field_type == "float":
                try:
                    s[key] = float(val)
                except ValueError:
                    s[key] = self.settings.get(key, 0.0)
            else:
                s[key] = val

        try:
            s["min_score"] = float(self.threshold_var.get())
        except ValueError:
            s["min_score"] = 50.0

        period_map = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
        s["data_period"] = period_map.get(self.period_var.get(), "1y")

        tf_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        s["timeframe"] = tf_map.get(self.timeframe_var.get(), "D")

        s["trend_filter"] = self.trend_filter_var.get()

        return s

    def _reset_settings(self):
        """Restore default settings and refresh UI."""
        self.settings.update(DEFAULT_SETTINGS.copy())
        save_settings(self.settings)
        self._load_settings_to_ui()
        self._log("Settings restored to defaults")

    def _on_universe_change(self, choice):
        count = len(UNIVERSES.get(choice, []))
        self.universe_count_label.configure(text=f"{count} stocks")

    def _on_threshold_change(self, val):
        self.threshold_label.configure(text=str(int(float(val))))

    # ════════════════════════════════════════════════════════════════════════
    # SCANNING
    # ════════════════════════════════════════════════════════════════════════

    def _start_scan(self):
        if self.scanning:
            # Toggle to cancel
            self._cancel_scan = True
            c = self.theme_colors
            self.run_btn.configure(text="\u23f9   CANCELLING\u2026",
                                   state="disabled", fg_color=c["card2"])
            return

        self._cancel_scan = False
        self.settings = self._collect_settings()
        save_settings(self.settings)

        self.scanning = True
        c = self.theme_colors
        self.run_btn.configure(state="disabled", text="\u23f3   SCANNING\u2026",
                               fg_color=c["card2"])
        self.html_btn.configure(state="disabled")
        self.csv_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        self.results = []
        self.sort_col = None
        self.sort_dir = "asc"

        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self._render_table_header()
        scanning_lbl = ctk.CTkLabel(
            self.table_frame, text="\nScanning\u2026\n",
            font=ctk.CTkFont(size=13), text_color=self.theme_colors["text_dim"])
        scanning_lbl.pack(pady=30, anchor="center")

        thread = threading.Thread(target=self._run_scan, daemon=True)
        thread.start()

    def _run_scan(self):
        """Run the scan in a background thread."""
        try:
            universe_name = self.universe_var.get()
            tickers = UNIVERSES.get(universe_name, [])
            settings = self.settings
            period = settings.get("data_period", "1y")
            timeframe = settings.get("timeframe", "D")
            trend_filter = settings.get("trend_filter", "All")

            tf_names = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
            self._log("\n" + "=" * 50)
            self._log(f"START SCAN | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._log("=" * 50)
            self._log(f"Starting scan: {universe_name} ({len(tickers)} stocks)")
            self._log(f"Timeframe: {tf_names.get(timeframe, timeframe)} | Period: {period} | Filter: {trend_filter}")
            self._log(f"FastMA={settings['fast_ma_type']}{settings['fast_ma_len']} "
                       f"SlowMA={settings['slow_ma_type']}{settings['slow_ma_len']} "
                       f"RSI={settings['rsi_len']} Threshold={settings['min_score']}")

            # Fetch NIFTY index
            self._set_progress(0, "Fetching NIFTY 50 index...")
            index_symbol = settings.get("index_symbol", "NSEI")
            index_df = fetch_index_data(f"^{index_symbol}", period=period)
            if index_df is not None:
                self._log(f"{index_symbol} index loaded ({len(index_df)} bars)")
            else:
                self._log(f"Warning: {index_symbol} index unavailable, using proxy for RS")

            # FAST: Batch download all stocks at once via yfinance
            self._set_progress(0.05, f"Batch downloading {len(tickers)} stocks...")
            self._log(f"Batch downloading {len(tickers)} stocks via yfinance...")
            batch_data = fetch_batch_yfinance(tickers, period=period, timeframe=timeframe)
            self._log(f"Batch download complete: {len(batch_data)}/{len(tickers)} stocks fetched")

            # ── 3-Model Pipeline ────────────────────────────────────────────
            results = []
            total = len(batch_data)
            filtered_out = 0
            direction_counts = {"Bull": 0, "Bear": 0}

            for i, (ticker, df) in enumerate(batch_data.items(), 1):
                if self._cancel_scan:
                    self._log("\n\u23f9  Scan cancelled by user")
                    break
                progress = 0.1 + (i / total * 0.9) if total > 0 else 0.5
                self._set_progress(progress, f"[{i}/{total}] {ticker}")

                try:
                    if df is None or df.empty:
                        continue

                    # ── MODEL 1: Stock Filter ────────────────────────────────
                    filter_result = check_filter(
                        df,
                        fast_ma_type=settings["fast_ma_type"],
                        fast_ma_len=settings["fast_ma_len"],
                        slow_ma_type=settings["slow_ma_type"],
                        slow_ma_len=settings["slow_ma_len"],
                        crossover_lookback=settings["crossover_lookback"],
                    )
                    if filter_result is None:
                        filtered_out += 1
                        continue

                    # ── MODEL 2: Bullish / Bearish ──────────────────────────
                    direction = get_direction(filter_result)

                    if trend_filter == "Bullish Only" and direction != "Bull":
                        filtered_out += 1
                        continue
                    elif trend_filter == "Bearish Only" and direction != "Bear":
                        filtered_out += 1
                        continue

                    direction_counts[direction] = direction_counts.get(direction, 0) + 1

                    # ── MODEL 3: Techno-Fundamental Scoring ─────────────────
                    if not hasattr(df, '_fundamentals') or df._fundamentals is None:
                        try:
                            fund = fetch_fundamentals(ticker)
                            if fund is not None:
                                df._fundamentals = fund
                        except Exception as e:
                            logger.debug("Fundamentals fetch failed for %s: %s", ticker, e)

                    scores = compute_scores(
                        df, timeframe=timeframe, index_df=index_df,
                        settings=settings,
                    )
                    if scores is None:
                        continue

                    scores["ticker"] = ticker
                    scores["trend_dir"] = direction  # Override with pipeline direction
                    scores["trend_color"] = direction.lower()
                    results.append(scores)

                    if len(results) % 10 == 0 or len(results) <= 5:
                        score_val = scores["total"]
                        tag = "\u2713" if score_val >= settings["min_score"] else "\u2717"
                        self._log(f"  {tag} {ticker}: {score_val:.1f}/100 ({direction})")

                except Exception as e:
                    logger.debug("Skipping %s in batch scan: %s", ticker, e)

            # Sort and store
            results.sort(key=lambda x: x["total"], reverse=True)

            passed = len([r for r in results if r["total"] >= settings["min_score"]])
            self._log("\n\u2501" * 25 + " Scan Complete ")
            self._log(f"  Total stocks:  {len(tickers)}")
            self._log(f"  Filtered out:  {filtered_out} (no recent crossover)")
            self._log(f"  Passed filter: {len(results)} ({direction_counts.get('Bull', 0)} Bull, {direction_counts.get('Bear', 0)} Bear)")
            self._log(f"  Scored {settings['min_score']}+: {passed}")

            def _apply_results(r=results):
                self.results = r
                self._display_results(r)
            self.after(0, _apply_results)

        except Exception as e:
            self._log(f"\nERROR: {str(e)}")
        finally:
            self.after(0, self._scan_complete)

    def _display_results(self, results):
        """Display filtered results in the glassy table (main thread only)."""
        tc = self.theme_colors

        # Apply ticker filter
        if self.filter_text:
            shown = [r for r in results if self.filter_text in r["ticker"].upper()]
        else:
            shown = list(results)

        # Clear table and re-render header bar
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self._render_table_header()

        if not shown:
            msg = ("No results match your filter." if results and self.filter_text
                   else "No results found.")
            ctk.CTkLabel(self.table_frame, text=msg,
                         text_color=tc["red"] if not results else tc["text_dim"],
                         font=ctk.CTkFont(size=13)).pack(pady=30, anchor="center")
        else:
            threshold = self.settings.get("min_score", 50)

            def _make_cols():
                return [
                    ("#",           35, lambda r, i: (str(i), tc["text_dim"], 11, False)),
                    ("Ticker",     100, lambda r, i: (r["ticker"],
                        tc["green"] if r["total"] >= threshold else tc["text"], 12, True)),
                    ("Score",       50, lambda r, i: (f'{r["total"]:.0f}',
                        tc["green"] if r["total"] >= 70 else (tc["lime"] if r["total"] >= 50 else (tc["orange"] if r["total"] >= 30 else tc["red"])), 13, True)),
                    ("Rating",      78, lambda r, i: (r.get("combined_rating", "POOR"),
                        {"EXCELLENT": tc["green"], "GOOD": tc["lime"], "MODERATE": tc["orange"]}.get(r.get("combined_rating"), tc["red"]), 10, True)),
                    ("ENTRY",       55, lambda r, i: ("YES" if r.get("entry_signal") else "--",
                        tc["green"] if r.get("entry_signal") else tc["text_dim"], 10, True)),
                    ("Price",       80, lambda r, i: (f'\u20b9{r.get("close", 0):.0f}', tc["text"], 11, True)),
                    ("MA",          62, lambda r, i: (self._ma_text(r), self._ma_color(r), 10, False)),
                    ("T/15",        40, lambda r, i: (f'{r.get("trend", 0):.0f}', tc["green"], 10, False)),
                    ("M/15",        40, lambda r, i: (f'{r.get("momentum", 0):.0f}', tc["cyan"], 10, False)),
                    ("R/8",         35, lambda r, i: (f'{r.get("rsi", 0):.0f}', tc["blue"], 10, False)),
                    ("V/7",         35, lambda r, i: (f'{r.get("macd", 0):.0f}', "#aa88ff", 10, False)),
                    ("Vol/10",      42, lambda r, i: (f'{r.get("volume", 0):.0f}', tc["orange"], 10, False)),
                    ("RS/10",       42, lambda r, i: (f'{r.get("rel_str", 0):.0f}', tc["lime"], 10, False)),
                    ("F/20",        42, lambda r, i: (f'{r.get("fundamentals", 0):.0f}', "#ffe600", 10, False)),
                    ("1M",          55, lambda r, i: (f'{r.get("pc1m", 0) or 0:+.1f}%',
                        tc["green"] if (r.get("pc1m", 0) or 0) > 0 else tc["red"], 10, False)),
                    ("Dir",         58, lambda r, i: (("^ " if r["trend_dir"] == "Bull" else "v ") + r["trend_dir"],
                        tc["green"] if r["trend_dir"] == "Bull" else tc["red"], 10, False)),
                    ("ADX",         40, lambda r, i: (f'{r.get("adx_val", 0) or 0:.0f}', tc["text"], 10, False)),
                    ("Chop",        42, lambda r, i: ("Chop" if r.get("is_sideways") else "OK",
                        tc["orange"] if r.get("is_sideways") else tc["green"], 10, False)),
                ]

            cols = _make_cols()

            for rank, r in enumerate(shown, 1):
                score = r["total"]
                is_above = score >= threshold
                row_bg = tc["card"] if is_above else (tc["row_alt"] if rank % 2 else tc["main_bg"])

                row = ctk.CTkFrame(self.table_frame, fg_color=row_bg, height=32,
                                   corner_radius=8,
                                   border_width=1 if is_above else 0,
                                   border_color=tc["border"])
                row.pack(fill="x", padx=0, pady=1)
                row.pack_propagate(False)

                for i, (_, width, extract) in enumerate(cols):
                    text, color, fsize, bold = extract(r, rank)
                    lbl = ctk.CTkLabel(row, text=text, width=width, anchor="w",
                                       font=ctk.CTkFont(size=fsize, weight="bold" if bold else "normal"),
                                       text_color=color)
                    lbl.pack(side="left", padx=1)
                    if i == 1:  # Ticker clickable → inline news
                        lbl.configure(cursor="hand2")
                        lbl.bind("<Button-1>",
                                 lambda e, t=r["ticker"], rf=row: self._toggle_stock_news(t, rf, rank))

        # Header meta
        if results:
            threshold = self.settings.get("min_score", 50)
            suffix = f"  |  filter: '{self.filter_text}' ({len(shown)})" if self.filter_text else ""
            self.result_count_label.configure(
                text=f"{len(results)} scanned  |  {len([r for r in results if r['total'] >= threshold])} above {threshold:.0f}+{suffix}")
        else:
            self.result_count_label.configure(text="no scan yet")

        # Summary + hero + top picks
        self._update_summary(results)
        self._update_hero_status(results)
        self._render_topicks(results[:5])

    def _update_hero_status(self, results):
        """Refresh the hero subtitle with live scan status."""
        if not hasattr(self, "hero_canvas"):
            return
        if self.scanning:
            txt = "Scanning\u2026 fetching data and scoring stocks"
        elif results:
            threshold = self.settings.get("min_score", 50)
            passed = len([r for r in results if r["total"] >= threshold])
            entry_ct = len([r for r in results if r.get("entry_signal")])
            txt = (f"{len(results)} stocks passed the crossover filter  \u00b7  "
                   f"{passed} scored {threshold:.0f}+  \u00b7  {entry_ct} ENTRY signals")
        else:
            txt = ("Set your universe on the left, then RUN SCAN \u2014 "
                   "HMA\u00d7EMA crossover \u2022 10-factor score \u2022 news sentiment")
        self.hero_canvas.itemconfigure(self.hero_sub, text=txt)

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

    def _toggle_stock_news(self, ticker: str, row_frame, rank):
        """Expand/collapse news inline below the stock row."""
        for child in row_frame.master.winfo_children():
            if getattr(child, '_news_ticker', None) == ticker:
                child.destroy()
                return

        for child in row_frame.master.winfo_children():
            if getattr(child, '_news_ticker', None):
                child.destroy()

        c = self.theme_colors
        news_frame = ctk.CTkFrame(row_frame.master, fg_color=c["card2"], corner_radius=10)
        news_frame._news_ticker = ticker
        news_frame.pack(after=row_frame, fill="x", padx=8, pady=(0, 2))

        loading = ctk.CTkLabel(news_frame, text="Loading news...",
                               font=ctk.CTkFont(size=11),
                               text_color=c["text_dim"])
        loading.pack(pady=10)

        def _parse_date(date_str):
            try:
                from datetime import datetime
                if not date_str:
                    return None
                clean = date_str.rstrip("Z").strip()[:19]
                for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(clean, fmt).date()
                    except ValueError:
                        continue
                return None
            except Exception as e:
                logger.debug("Date parse failed for '%s': %s", date_str, e)
                return None

        def _fetch():
            try:
                from datetime import date, timedelta
                import yfinance as yf
                stock = yf.Ticker(f"{ticker}.NS")
                news_items = stock.news or []
                cutoff = date.today() - timedelta(days=60)
                parsed = []
                for item in news_items:
                    content = item.get("content", item)
                    title = content.get("title", "")
                    if not title:
                        continue
                    summary = content.get("summary", "")
                    pub_date = content.get("pubDate", "") or content.get("displayTime", "")
                    pub_dt = _parse_date(pub_date)
                    if pub_dt and pub_dt < cutoff:
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
                self.after(0, lambda: _show(parsed[:10]))
            except Exception as e:
                self.after(0, lambda: _show_error(str(e)))

        def _show(items):
            if not news_frame.winfo_exists():
                return
            loading.destroy()
            if not items:
                ctk.CTkLabel(news_frame, text="No recent news found.",
                             font=ctk.CTkFont(size=11),
                             text_color=c["text_dim"]).pack(pady=8)
                return
            good = sum(1 for i in items if i["sentiment"] == "Good")
            bad = sum(1 for i in items if i["sentiment"] == "Bad")
            neu = len(items) - good - bad
            summary_text = f"{good} Good  |  {bad} Bad  |  {neu} Neutral"
            ctk.CTkLabel(news_frame, text=summary_text,
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=c["lime"]).pack(pady=(4, 6))

            for item in items:
                sent = item["sentiment"]
                sent_color = {"Good": c["green"], "Bad": c["red"], "Neutral": c["text_dim"]}[sent]
                sent_bg = {"Good": c["chip_good"], "Bad": c["chip_bad"], "Neutral": c["card"]}[sent]

                card = ctk.CTkFrame(news_frame, fg_color=c["card"], corner_radius=8)
                card.pack(fill="x", padx=6, pady=2)

                top = ctk.CTkFrame(card, fg_color="transparent")
                top.pack(fill="x", padx=8, pady=(4, 0))

                badge = ctk.CTkFrame(top, fg_color=sent_bg, corner_radius=6, height=18)
                badge.pack(side="left")
                badge.pack_propagate(False)
                ctk.CTkLabel(badge, text=sent, font=ctk.CTkFont(size=9, weight="bold"),
                             text_color=sent_color).pack(padx=5, pady=1)

                meta = f"{item['date']}  {item['provider']}" if item['provider'] else item['date']
                ctk.CTkLabel(top, text=meta,
                             font=ctk.CTkFont(size=9),
                             text_color=c["text_dim"]).pack(side="left", padx=6)

                ctk.CTkLabel(card, text=item["title"],
                             font=ctk.CTkFont(size=11, weight="bold"),
                             text_color=c["text"], wraplength=900,
                             anchor="w", justify="left").pack(anchor="w", padx=8, pady=(1, 0))

                if item["summary"]:
                    ctk.CTkLabel(card, text=item["summary"],
                                 font=ctk.CTkFont(size=10),
                                 text_color=c["text_dim"], wraplength=900,
                                 anchor="w", justify="left").pack(anchor="w", padx=8, pady=(1, 4))

        def _show_error(msg):
            if news_frame.winfo_exists():
                loading.configure(text=f"Error: {msg}", text_color=c["red"])

        threading.Thread(target=_fetch, daemon=True).start()

    def _scan_complete(self):
        """Re-enable UI after scan finishes."""
        self.scanning = False
        c = self.theme_colors
        self.run_btn.configure(state="normal", text="\u25b6   RUN SCAN",
                               fg_color=c["purple"])
        self.progress_label.configure(text="Done")
        self.status_label.configure(text="Status: Done")
        if self.results:
            self.html_btn.configure(state="normal")
            self.csv_btn.configure(state="normal")
            self.clear_btn.configure(state="normal")
        self._update_hero_status(self.results)

    # ════════════════════════════════════════════════════════════════════════
    # SUMMARY STATS
    # ════════════════════════════════════════════════════════════════════════

    def _update_summary(self, results):
        """Update the summary stat cards from scan results."""
        if not results:
            return
        threshold = self.settings.get("min_score", 50)
        total = len(results)
        passed = len([r for r in results if r["total"] >= threshold])
        avg = sum(r["total"] for r in results) / total if total else 0
        high = max(r["total"] for r in results) if results else 0
        bull = len([r for r in results if r.get("trend_dir") == "Bull"])
        bear = len([r for r in results if r.get("trend_dir") == "Bear"])
        entry = len([r for r in results if r.get("entry_signal")])
        self.summary_cards["total"].configure(text=str(total))
        self.summary_cards["passed"].configure(text=str(passed))
        self.summary_cards["entry"].configure(text=str(entry))
        self.summary_cards["avg"].configure(text=f"{avg:.1f}")
        self.summary_cards["high"].configure(text=f"{high:.0f}")
        self.summary_cards["bull"].configure(text=str(bull))
        self.summary_cards["bear"].configure(text=str(bear))

    def _reset_summary(self):
        """Reset the summary cards to their empty state."""
        for key in ("total", "passed", "entry", "avg", "high", "bull", "bear"):
            if hasattr(self, "summary_cards") and key in self.summary_cards:
                self.summary_cards[key].configure(text="\u2014")

    # ════════════════════════════════════════════════════════════════════════
    # EXPORT
    # ════════════════════════════════════════════════════════════════════════

    def _export_html(self):
        if not self.results:
            return
        threshold = self.settings.get("min_score", 50)
        tf_names = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
        tf_label = tf_names.get(self.settings.get("timeframe", "D"), "Daily")
        self._log("Fetching news sentiment for exported stocks...")
        c = self.theme_colors
        self.html_btn.configure(state="disabled", text="\u23f3 Exporting\u2026")

        def _do_export():
            try:
                html = generate_html_report(
                    self.results,
                    title=f"HMAxEMA Scanner — {self.universe_var.get()} — {tf_label}",
                    threshold=threshold,
                    fetch_news=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scanner_report_{timestamp}.html"
                filepath = os.path.join(SCANNER_DIR, filename)
                save_report(html, filepath)
                self.after(0, lambda: self._log(f"HTML report saved: {filename}"))
                self.after(0, lambda: webbrowser.open(f"file://{os.path.abspath(filepath)}"))
            except Exception as e:
                self.after(0, lambda: self._log(f"HTML export failed: {e}"))
            finally:
                self.after(0, lambda: self.html_btn.configure(
                    state="normal", text="\U0001f4c4  HTML"))

        threading.Thread(target=_do_export, daemon=True).start()

    def _export_csv(self):
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
                    i, r["ticker"], r["total"],
                    r.get("combined_rating", "POOR"),
                    r.get("close"), r["trend"], r["momentum"], r["rsi"], r["macd"],
                    r["stoch"], r["obv"], r["volume"], r["rel_str"], r["volatility"],
                    r.get("fundamentals", 0),
                    r["trend_dir"], r.get("rsi_val"), r.get("adx_val"),
                    "Yes" if r.get("is_sideways") else "No" + (f" ({sideways_reasons})" if sideways_reasons else ""),
                    r.get("pc1m"), r.get("pc3m")
                ])

        self._log(f"CSV saved: {filename}")
        os.startfile(filepath) if sys.platform == "win32" else os.system(f"open '{filepath}'")

    # ════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ════════════════════════════════════════════════════════════════════════

    def _clear_cache(self):
        """Clear the disk cache for all data providers."""
        try:
            from .data_providers import DataProvider
            provider = DataProvider()
            provider.clear_cache()
            self._log("Cache cleared successfully")
        except Exception as e:
            self._log(f"Failed to clear cache: {e}")

    def _switch_theme(self):
        """Toggle between dark and light themes (full rebuild for consistency)."""
        new_theme = "light" if self.current_theme == "dark" else "dark"
        self.current_theme = new_theme
        self.theme_colors = THEMES[new_theme]
        self.settings["theme"] = new_theme
        save_settings(self.settings)
        ctk.set_appearance_mode(THEMES[new_theme]["ctk_mode"])

        had_results = bool(self.results)
        self._build_ui()
        self._load_settings_to_ui()
        if had_results:
            self._display_results(self.results)
        self._log(f"Theme switched to {new_theme}")

    def _clear_results(self):
        """Clear the results table and reset state."""
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self._render_table_header()
        c = self.theme_colors
        self.results = []
        self.sort_col = None
        self.sort_dir = "asc"
        self.empty_label = ctk.CTkLabel(
            self.table_frame, text="\nNo results yet \u2014 hit \u25b6 RUN SCAN\n",
            font=ctk.CTkFont(size=13), text_color=c["text_dim"])
        self.empty_label.pack(pady=30, anchor="center")

        self.result_count_label.configure(text="no scan yet")
        self._reset_summary()
        self._render_topicks([])
        self._update_hero_status([])
        self.progress.set(0)
        self.progress_label.configure(text="Ready")
        self.status_label.configure(text="Status: Ready")
        self.html_btn.configure(state="disabled")
        self.csv_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        self._log("Results cleared")

    def _copy_selection(self):
        """Copy selected text from log to clipboard."""
        try:
            selection = self.log_text.get("sel.first", "sel.last")
            if selection:
                self.clipboard_clear()
                self.clipboard_append(selection)
        except Exception as e:
            logger.debug("Clipboard copy failed: %s", e)

    def _rotate_log(self):
        """Overwrite log file if it's older than LOG_ROTATE_HOURS."""
        try:
            if os.path.exists(LOG_FILE):
                age_hours = (datetime.now().timestamp() - os.path.getmtime(LOG_FILE)) / 3600
                if age_hours >= LOG_ROTATE_HOURS:
                    open(LOG_FILE, "w").close()  # Truncate
        except Exception as e:
            logger.debug("Log rotation failed: %s", e)

    def _log(self, msg: str):
        """Thread-safe log to the activity textbox and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"

        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.debug("Failed to write to log file: %s", e)

        def _append():
            try:
                self.log_text.insert("end", line)
                self.log_text.see("end")
            except Exception:
                pass  # Widget may be rebuilding during theme switch

        self.after(0, _append)

    def _set_progress(self, value: float, text: str = ""):
        """Thread-safe progress update."""
        def _update():
            self.progress.set(value)
            if text:
                self.progress_label.configure(text=text)
                self.status_label.configure(text=f"Status: {text}")
        self.after(0, _update)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = ScannerApp()
    app.mainloop()
