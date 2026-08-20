"""
HMAxEMA Stock Scanner — GUI Application
Full-featured desktop app for scanning Indian stocks.

Usage:
    python scanner/app.py

Or double-click run.bat (Windows) / run.sh (macOS/Linux)
"""

import json
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime

import customtkinter as ctk

from .universes import UNIVERSES
from .data_fetcher import fetch_stock_data, fetch_index_data, fetch_stock_fast, fetch_batch_yfinance, fetch_fundamentals
from .scoring import compute_scores, check_filter, get_direction
from .report import generate_html_report, save_report

# ── Theme ────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

SCANNER_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(SCANNER_DIR, "settings.json")
LOG_FILE = os.path.join(SCANNER_DIR, "scan.log")
LOG_ROTATE_HOURS = 12  # Overwrite log file after 12 hours

# ── Sentiment keywords for news analysis ────────────────────────────────────
SENTIMENT_GOOD = frozenset([
    "profit", "growth", "record", "gain", "surge", "rally",
    "strong", "beat", "bullish", "outperform", "order", "deal",
    "win", "partnership", "expansion", "dividend", "upgrade",
    "buy", "positive", "surpass", "boost", "rise", "jump",
    "high", "best", "stronger", "approval", "contract",
])
SENTIMENT_BAD = frozenset([
    "loss", "decline", "fall", "drop", "crash", "bearish",
    "underperform", "miss", "downgrade", "sell", "debt", "default",
    "fraud", "investigation", "lawsuit", "warning", "negative",
    "slump", "cut", "risk", "ban", "penalty", "fines",
    "slowdown", "weak", "weaker", "worst", "lower",
])

# ── Default Settings (mirrors Pine Script indicator) ─────────────────────────
DEFAULT_SETTINGS = {
    # Moving Averages
    "fast_ma_type": "HMA",
    "fast_ma_len": 40,
    "slow_ma_type": "EMA",
    "slow_ma_len": 50,
    # Technical Analysis
    "rsi_len": 14,
    "rs_length": 14,
    "vol_ma_len": 20,
    "atr_len": 14,
    # Relative Strength
    "index_symbol": "NSEI",
    # Volume Profile
    "vp_lookback": 200,
    "vp_rows": 30,
    "vp_width": 40,
    # Sideways Filter
    "adx_len": 14,
    "adx_threshold": 20.0,
    "chop_len": 14,
    "chop_threshold": 61.8,
    "slope_ma_type": "KAMA",
    "slope_ma_len": 50,
    "slope_lookback": 10,
    "flat_threshold": 0.5,
    # Step Channel
    "sc_pivot_len": 3,
    "sc_bands_mult": 0.6,
    # MA Crossover
    "crossover_lookback": 20,
    # Scanner
    "min_score": 50.0,
    "data_period": "1y",
    "timeframe": "D",
    "trend_filter": "All",
}


def load_settings() -> dict:
    """Load settings from JSON file, falling back to defaults."""
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            settings.update(saved)
        except Exception:
            pass
    return settings


def save_settings(settings: dict):
    """Save settings to JSON file."""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

RESULT_COLS = [
    ("#", 35), ("Ticker", 100), ("Score", 50), ("Rating", 75), ("ENTRY", 55),
    ("Price", 80), ("MA", 60), ("T/15", 40), ("M/15", 40), ("R/8", 35),
    ("V/7", 35), ("Vol/10", 40), ("RS/10", 40), ("F/20", 40),
    ("1M", 55), ("Dir", 55), ("ADX", 40), ("Chop", 40),
]


class ScannerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("HMAxEMA Stock Scanner — Indian Market")
        self.geometry("1920x1080")
        self.minsize(1200, 800)
        self.state("zoomed")  # Start maximized

        self.settings = load_settings()
        self.results = []
        self.scanning = False

        self._build_ui()
        self._load_settings_to_ui()
        self._rotate_log()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_area()

    def _build_sidebar(self):
        """Left sidebar: universe, settings, run button."""
        sidebar = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color="#0d1f14")
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        # ── Logo / Title ─────────────────────────────────────────────────────
        ctk.CTkLabel(sidebar, text="HMAxEMA Scanner",
                      font=ctk.CTkFont(size=20, weight="bold"),
                      text_color="#00ff88").pack(pady=(20, 2), padx=20, anchor="w")
        ctk.CTkLabel(sidebar, text="Indian Market Stock Screener",
                      font=ctk.CTkFont(size=12),
                      text_color="#6a8a6a").pack(padx=20, anchor="w")

        ctk.CTkFrame(sidebar, height=1, fg_color="#1a4a2a").pack(fill="x", padx=15, pady=12)

        # ── Universe Selector ────────────────────────────────────────────────
        ctk.CTkLabel(sidebar, text="STOCK UNIVERSE",
                      font=ctk.CTkFont(size=11, weight="bold"),
                      text_color="#00ddcc").pack(padx=20, anchor="w")

        self.universe_var = ctk.StringVar(value="NIFTY 50")
        universe_names = list(UNIVERSES.keys())
        self.universe_menu = ctk.CTkOptionMenu(
            sidebar, variable=self.universe_var,
            values=universe_names,
            command=self._on_universe_change,
            width=280, height=32,
            fg_color="#153520", button_color="#1a4a2a",
            dropdown_fg_color="#0f2a1a"
        )
        self.universe_menu.pack(padx=20, pady=(4, 8))

        self.universe_count_label = ctk.CTkLabel(
            sidebar, text=f"{len(UNIVERSES['NIFTY 50'])} stocks",
            font=ctk.CTkFont(size=11), text_color="#6a8a6a")
        self.universe_count_label.pack(padx=20, anchor="w")

        # ── Timeframe ──────────────────────────────────────────────────────
        ctk.CTkLabel(sidebar, text="TIMEFRAME",
                      font=ctk.CTkFont(size=11, weight="bold"),
                      text_color="#00ddcc").pack(padx=20, pady=(12, 0), anchor="w")

        self.timeframe_var = ctk.StringVar(value="Daily")
        timeframe_menu = ctk.CTkOptionMenu(
            sidebar, variable=self.timeframe_var,
            values=["Daily", "Weekly", "Monthly"],
            width=280, height=32,
            fg_color="#153520", button_color="#1a4a2a",
            dropdown_fg_color="#0f2a1a"
        )
        timeframe_menu.pack(padx=20, pady=(4, 8))

        # ── Data Period ──────────────────────────────────────────────────────
        ctk.CTkLabel(sidebar, text="DATA PERIOD",
                      font=ctk.CTkFont(size=11, weight="bold"),
                      text_color="#00ddcc").pack(padx=20, pady=(8, 0), anchor="w")

        self.period_var = ctk.StringVar(value="1 Year")
        period_menu = ctk.CTkOptionMenu(
            sidebar, variable=self.period_var,
            values=["6 Months", "1 Year", "2 Years"],
            width=280, height=32,
            fg_color="#153520", button_color="#1a4a2a",
            dropdown_fg_color="#0f2a1a"
        )
        period_menu.pack(padx=20, pady=(4, 8))

        # ── Trend Filter ─────────────────────────────────────────────────────
        ctk.CTkLabel(sidebar, text="TREND FILTER",
                      font=ctk.CTkFont(size=11, weight="bold"),
                      text_color="#00ddcc").pack(padx=20, pady=(8, 0), anchor="w")

        self.trend_filter_var = ctk.StringVar(value="All")
        trend_menu = ctk.CTkOptionMenu(
            sidebar, variable=self.trend_filter_var,
            values=["All", "Bullish Only", "Bearish Only"],
            width=280, height=32,
            fg_color="#153520", button_color="#1a4a2a",
            dropdown_fg_color="#0f2a1a"
        )
        trend_menu.pack(padx=20, pady=(4, 8))

        # ── Score Threshold ──────────────────────────────────────────────────
        ctk.CTkLabel(sidebar, text="MIN SCORE THRESHOLD",
                      font=ctk.CTkFont(size=11, weight="bold"),
                      text_color="#00ddcc").pack(padx=20, pady=(8, 0), anchor="w")

        self.threshold_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        self.threshold_frame.pack(padx=20, pady=(4, 8), fill="x")

        self.threshold_var = ctk.DoubleVar(value=50.0)
        self.threshold_slider = ctk.CTkSlider(
            self.threshold_frame, from_=0, to=100,
            variable=self.threshold_var, number_of_steps=20,
            command=self._on_threshold_change,
            width=200, height=16,
            button_color="#00ff88", progress_color="#00aa55"
        )
        self.threshold_slider.pack(side="left")
        self.threshold_label = ctk.CTkLabel(
            self.threshold_frame, text="50",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#aaff00", width=40)
        self.threshold_label.pack(side="right")

        # ── Settings Section (Collapsible) ───────────────────────────────────
        ctk.CTkFrame(sidebar, height=1, fg_color="#1a4a2a").pack(fill="x", padx=15, pady=10)

        self.settings_toggle = ctk.CTkButton(
            sidebar, text="  SETTINGS  (Click to expand)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00ddcc", fg_color="transparent",
            hover_color="#0f2a1a", anchor="w",
            command=self._toggle_settings, height=28)
        self.settings_toggle.pack(padx=20, fill="x")

        self.settings_frame = ctk.CTkScrollableFrame(
            sidebar, fg_color="transparent", height=0)
        self.settings_frame.pack(padx=10, fill="x")

        self._build_settings_panel(self.settings_frame)

        # ── Spacer ───────────────────────────────────────────────────────────
        ctk.CTkFrame(sidebar, fg_color="transparent").pack(fill="both", expand=True)

        # ── Run Button ───────────────────────────────────────────────────────
        self.run_btn = ctk.CTkButton(
            sidebar, text="RUN SCAN",
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#00aa55", hover_color="#00cc66",
            text_color="#000000", height=48,
            corner_radius=8,
            command=self._start_scan)
        self.run_btn.pack(padx=20, pady=(0, 8), fill="x")

        # ── Clear Cache Button ───────────────────────────────────────────────
        self.cache_btn = ctk.CTkButton(
            sidebar, text="Clear Cache",
            font=ctk.CTkFont(size=11),
            fg_color="#1a4a2a", hover_color="#2a6a3a",
            text_color="#6a8a6a", height=24,
            command=self._clear_cache)
        self.cache_btn.pack(padx=20, pady=(0, 4))

        # ── Progress Bar ─────────────────────────────────────────────────────
        self.progress = ctk.CTkProgressBar(sidebar, width=280, height=8,
                                            fg_color="#153520", progress_color="#00ff88")
        self.progress.pack(padx=20, pady=(0, 4))
        self.progress.set(0)

        self.progress_label = ctk.CTkLabel(
            sidebar, text="Ready",
            font=ctk.CTkFont(size=11), text_color="#6a8a6a")
        self.progress_label.pack(padx=20, pady=(0, 15))

    def _build_settings_panel(self, parent):
        """Build the settings form inside the scrollable frame."""
        self.setting_widgets = {}

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

        for section_title, fields in sections:
            ctk.CTkLabel(parent, text=section_title.upper(),
                          font=ctk.CTkFont(size=10, weight="bold"),
                          text_color="#6a8a6a").pack(padx=8, pady=(8, 2), anchor="w")

            for label, key, field_type, constraints in fields:
                row = ctk.CTkFrame(parent, fg_color="transparent", height=28)
                row.pack(fill="x", padx=8, pady=1)
                row.pack_propagate(False)

                ctk.CTkLabel(row, text=label,
                              font=ctk.CTkFont(size=11),
                              text_color="#c8d8c0", width=130, anchor="w").pack(side="left")

                if field_type == "option":
                    var = ctk.StringVar(value=str(self.settings.get(key, "")))
                    widget = ctk.CTkOptionMenu(
                        row, variable=var, values=constraints,
                        width=120, height=24,
                        fg_color="#153520", button_color="#1a4a2a",
                        dropdown_fg_color="#0f2a1a",
                        font=ctk.CTkFont(size=11))
                    widget.pack(side="right")
                elif field_type == "int":
                    var = ctk.StringVar(value=str(self.settings.get(key, 0)))
                    widget = ctk.CTkEntry(
                        row, textvariable=var, width=80, height=24,
                        fg_color="#153520", border_color="#1a4a2a",
                        font=ctk.CTkFont(size=11))
                    widget.pack(side="right")
                elif field_type == "float":
                    var = ctk.StringVar(value=str(self.settings.get(key, 0.0)))
                    widget = ctk.CTkEntry(
                        row, textvariable=var, width=80, height=24,
                        fg_color="#153520", border_color="#1a4a2a",
                        font=ctk.CTkFont(size=11))
                    widget.pack(side="right")
                elif field_type == "text":
                    var = ctk.StringVar(value=str(self.settings.get(key, "")))
                    widget = ctk.CTkEntry(
                        row, textvariable=var, width=120, height=24,
                        fg_color="#153520", border_color="#1a4a2a",
                        font=ctk.CTkFont(size=10))
                    widget.pack(side="right")

                self.setting_widgets[key] = (var, field_type, constraints)

    def _build_main_area(self):
        """Right main area: results table, summary stats, and log."""
        main = ctk.CTkFrame(self, fg_color="#0a1a10", corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=0)  # Header - fixed height
        main.grid_rowconfigure(1, weight=0)  # Summary stats - fixed height
        main.grid_rowconfigure(2, weight=1)  # Table - takes ALL extra space
        main.grid_rowconfigure(3, weight=0)  # Log - fixed height at bottom

        # ── Header bar ───────────────────────────────────────────────────────
        header = ctk.CTkFrame(main, fg_color="#0f2a1a", height=44)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(header, text="RESULTS",
                      font=ctk.CTkFont(size=14, weight="bold"),
                      text_color="#00ff88").pack(padx=15, side="left")

        self.result_count_label = ctk.CTkLabel(
            header, text="0 stocks scanned",
            font=ctk.CTkFont(size=12), text_color="#6a8a6a")
        self.result_count_label.pack(side="left", padx=10)

        # Export buttons
        self.html_btn = ctk.CTkButton(
            header, text="Export HTML", width=100, height=30,
            fg_color="#1a4a2a", hover_color="#2a6a3a",
            font=ctk.CTkFont(size=11),
            command=self._export_html, state="disabled")
        self.html_btn.pack(side="right", padx=(0, 10))

        self.csv_btn = ctk.CTkButton(
            header, text="Export CSV", width=100, height=30,
            fg_color="#1a4a2a", hover_color="#2a6a3a",
            font=ctk.CTkFont(size=11),
            command=self._export_csv, state="disabled")
        self.csv_btn.pack(side="right", padx=(0, 5))
        
        # Clear Results button
        self.clear_btn = ctk.CTkButton(
            header, text="Clear Results", width=100, height=30,
            fg_color="#3a2a0a", hover_color="#4a3a1a",
            font=ctk.CTkFont(size=11),
            command=self._clear_results, state="disabled")
        self.clear_btn.pack(side="right", padx=(0, 5))

        # ── Summary Stats Panel ─────────────────────────────────────────────
        self.summary_frame = ctk.CTkFrame(main, fg_color="#0f2a1a", corner_radius=0)
        self.summary_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(5, 0))
        self.summary_frame.grid_columnconfigure(0, weight=1)

        # Placeholder - will be populated by _update_summary
        self.summary_inner = ctk.CTkFrame(self.summary_frame, fg_color="transparent")
        self.summary_inner.pack(fill="x", padx=10, pady=8)

        self.summary_widgets = {}
        self._build_summary_panel(self.summary_inner)

        # ── Results Table: sticky header + scrollable body ──────────────────
        results_container = ctk.CTkFrame(main, fg_color="transparent", corner_radius=0)
        results_container.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        results_container.grid_columnconfigure(0, weight=1)
        results_container.grid_rowconfigure(0, weight=1)  # table takes all space

        # Sticky header — stays visible while the list scrolls
        # Scrolling stock list (header is created inside _display_results)
        self.table_frame = ctk.CTkScrollableFrame(results_container, fg_color="#0a1a10",
                                                   corner_radius=0)
        self.table_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # ── Log area (FIXED at bottom, NEVER expands) ──────────────────────
        log_frame = ctk.CTkFrame(main, fg_color="#061208", height=150)
        log_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        log_frame.grid_propagate(False)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        
        log_header = ctk.CTkFrame(log_frame, fg_color="#0a1a0a", height=24)
        log_header.grid(row=0, column=0, sticky="ew")
        log_header.grid_propagate(False)
        
        ctk.CTkLabel(log_header, text="  LOG",
                      font=ctk.CTkFont(size=10, weight="bold"),
                      text_color="#4a7a4a").pack(side="left", padx=5)
        
        self.log_text = ctk.CTkTextbox(
            log_frame, fg_color="#061208",
            text_color="#4a7a4a",
            font=ctk.CTkFont(family="Consolas", size=10),
            state="normal")
        self.log_text.bind("<Control-a>", lambda e: self.log_text.tag_add("sel", "1.0", "end"))
        self.log_text.bind("<Control-c>", lambda e: self._copy_selection())
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=2, pady=(0, 2))

    # ── Summary Stats Panel ──────────────────────────────────────────────────

    def _build_summary_panel(self, parent):
        """Build the summary statistics cards (values updated after a scan)."""
        stats = [
            ("TOTAL", "total", "#00ddcc"),
            ("PASSED", "passed", "#00ff88"),
            ("ENTRY", "entry", "#00ff88"),
            ("AVG", "avg", "#aaff00"),
            ("HIGH", "high", "#00ff88"),
            ("BULL", "bull", "#00ff88"),
            ("BEAR", "bear", "#ff4444"),
        ]
        self.summary_cards = {}
        for label, key, color in stats:
            card = ctk.CTkFrame(parent, fg_color="#0a1a10", corner_radius=6)
            card.pack(side="left", fill="both", expand=True, padx=4)
            val_label = ctk.CTkLabel(card, text="—",
                                    font=ctk.CTkFont(size=20, weight="bold"),
                                    text_color=color)
            val_label.pack(pady=(6, 0))
            ctk.CTkLabel(card, text=label,
                         font=ctk.CTkFont(size=9),
                         text_color="#6a8a6a").pack(pady=(0, 6))
            self.summary_cards[key] = val_label

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
                self.summary_cards[key].configure(text="—")

    # ── Settings Management ──────────────────────────────────────────────────

    def _toggle_settings(self):
        """Expand/collapse settings panel."""
        if self.settings_frame.winfo_height() > 10:
            self.settings_frame.configure(height=0)
            self.settings_toggle.configure(text="  SETTINGS  (Click to expand)")
        else:
            self.settings_frame.configure(height=400)
            self.settings_toggle.configure(text="  SETTINGS  (Click to collapse)")

    def _load_settings_to_ui(self):
        """Load saved settings into UI widgets."""
        self.threshold_slider.set(self.settings.get("min_score", 50))
        self.threshold_label.configure(text=str(int(self.settings.get("min_score", 50))))

        # Period mapping
        period_map = {"6mo": "6 Months", "1y": "1 Year", "2y": "2 Years"}
        self.period_var.set(period_map.get(self.settings.get("data_period", "1y"), "1 Year"))

        # Timeframe
        tf_map = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
        self.timeframe_var.set(tf_map.get(self.settings.get("timeframe", "D"), "Daily"))

        # Trend filter
        self.trend_filter_var.set(self.settings.get("trend_filter", "All"))

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

        # Score threshold
        try:
            s["min_score"] = float(self.threshold_var.get())
        except ValueError:
            s["min_score"] = 50.0

        # Data period
        period_map = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
        s["data_period"] = period_map.get(self.period_var.get(), "1y")

        # Timeframe
        tf_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        s["timeframe"] = tf_map.get(self.timeframe_var.get(), "D")

        # Trend filter
        s["trend_filter"] = self.trend_filter_var.get()

        return s

    def _on_universe_change(self, choice):
        count = len(UNIVERSES.get(choice, []))
        self.universe_count_label.configure(text=f"{count} stocks")

    def _on_threshold_change(self, val):
        self.threshold_label.configure(text=str(int(float(val))))

    # ── Scanning ─────────────────────────────────────────────────────────────

    def _start_scan(self):
        if self.scanning:
            return

        self.settings = self._collect_settings()
        save_settings(self.settings)

        self.scanning = True
        self.run_btn.configure(state="disabled", text="SCANNING...", fg_color="#333333")
        self.html_btn.configure(state="disabled")
        self.csv_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        self.results = []

        # Clear previous results
        for widget in self.table_frame.winfo_children():
            widget.destroy()

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
            self._log(f"\n{'='*50}")
            self._log(f"START SCAN | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._log(f"{'='*50}")
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
                progress = 0.1 + (i / total * 0.9) if total > 0 else 0.5
                self._set_progress(progress, f"[{i}/{total}] {ticker}")

                try:
                    if df is None or df.empty:
                        continue

                    # ── MODEL 1: Stock Filter ────────────────────────────────
                    # Check for recent MA crossover. Skip if none found.
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

                    # Apply trend filter (direction-based)
                    if trend_filter == "Bullish Only" and direction != "Bull":
                        filtered_out += 1
                        continue
                    elif trend_filter == "Bearish Only" and direction != "Bear":
                        filtered_out += 1
                        continue

                    direction_counts[direction] = direction_counts.get(direction, 0) + 1

                    # ── MODEL 3: Techno-Fundamental Scoring ─────────────────
                    # Fetch fundamentals only for stocks that passed the filter
                    if not hasattr(df, '_fundamentals') or df._fundamentals is None:
                        try:
                            fund = fetch_fundamentals(ticker)
                            if fund is not None:
                                df._fundamentals = fund
                        except Exception:
                            pass

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
                    pass  # Silently skip errors in batch mode

            # Sort and store
            results.sort(key=lambda x: x["total"], reverse=True)
            self.results = results

            # Update UI
            passed = len([r for r in results if r["total"] >= settings["min_score"]])
            self._log(f"\n\u2501\u2501\u2501 Scan Complete \u2501\u2501\u2501")
            self._log(f"  Total stocks:  {len(tickers)}")
            self._log(f"  Filtered out:  {filtered_out} (no recent crossover)")
            self._log(f"  Passed filter: {len(results)} ({direction_counts.get('Bull', 0)} Bull, {direction_counts.get('Bear', 0)} Bear)")
            self._log(f"  Scored {settings['min_score']}+: {passed}")
            self._log(f"{'='*50}")
            self._log(f"END SCAN")
            self._log(f"{'='*50}")

            self.after(0, lambda: self._display_results(results))

        except Exception as e:
            self._log(f"\nERROR: {str(e)}")
        finally:
            self.after(0, self._scan_complete)

    def _display_results(self, results):
        """Display results in the table (must be called from main thread)."""
        # Clear table
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        if not results:
            ctk.CTkLabel(self.table_frame, text="No results found.",
                          text_color="#ff4444",
                          font=ctk.CTkFont(size=14)).pack(pady=40)
            return

        threshold = self.settings.get("min_score", 50)

        # Column definitions: (header_text, width, extract_func)
        def _make_cols():
            return [
                ("#",           35, lambda r, i: (str(i), "#c8d8c0", 12, True)),
                ("Ticker",     100, lambda r, i: (r["ticker"], "#00ff88" if r["total"] >= threshold else "#c8d8c0", 12, True)),
                ("Score",       50, lambda r, i: (f'{r["total"]:.0f}',
                    "#00ff88" if r["total"] >= 70 else ("#aaff00" if r["total"] >= 50 else ("#ffaa00" if r["total"] >= 30 else "#ff4444")), 13, True)),
                ("Rating",      75, lambda r, i: (r.get("combined_rating", "POOR"),
                    {"EXCELLENT": "#00ff88", "GOOD": "#aaff00", "MODERATE": "#ffaa00"}.get(r.get("combined_rating"), "#ff4444"), 11, False)),
                ("ENTRY",       55, lambda r, i: ("YES" if r.get("entry_signal") else "--",
                    "#00ff88" if r.get("entry_signal") else "#ff4444", 11, False)),
                ("Price",       80, lambda r, i: (f'\u20b9{r.get("close", 0):.0f}', "#c8d8c0", 12, True)),
                ("MA",          60, lambda r, i: (self._ma_text(r), self._ma_color(r), 11, False)),
                ("T/15",        40, lambda r, i: (f'{r.get("trend", 0):.0f}', "#00ff88", 11, False)),
                ("M/15",        40, lambda r, i: (f'{r.get("momentum", 0):.0f}', "#00ddcc", 11, False)),
                ("R/8",         35, lambda r, i: (f'{r.get("rsi", 0):.0f}', "#00aaff", 11, False)),
                ("V/7",         35, lambda r, i: (f'{r.get("macd", 0):.0f}', "#aa88ff", 11, False)),
                ("Vol/10",      40, lambda r, i: (f'{r.get("volume", 0):.0f}', "#ffaa00", 11, False)),
                ("RS/10",       40, lambda r, i: (f'{r.get("rel_str", 0):.0f}', "#aaff00", 11, False)),
                ("F/20",        40, lambda r, i: (f'{r.get("fundamentals", 0):.0f}', "#ffe600", 11, False)),
                ("1M",          55, lambda r, i: (f'{r.get("pc1m", 0) or 0:+.1f}%',
                    "#00ff88" if (r.get("pc1m", 0) or 0) > 0 else "#ff4444", 11, False)),
                ("Dir",         55, lambda r, i: (("^ " if r["trend_dir"] == "Bull" else "v ") + r["trend_dir"],
                    "#00ff88" if r["trend_dir"] == "Bull" else "#ff4444", 11, False)),
                ("ADX",         40, lambda r, i: (f'{r.get("adx_val", 0) or 0:.0f}', "#c8d8c0", 11, False)),
                ("Chop",        40, lambda r, i: ("Chop" if r.get("is_sideways") else "OK",
                    "#ffaa00" if r.get("is_sideways") else "#00ff88", 11, False)),
            ]

        cols = _make_cols()

        # Header row
        hdr = ctk.CTkFrame(self.table_frame, fg_color="#153520", height=32)
        hdr.pack(fill="x", padx=2, pady=(2, 0))
        hdr.pack_propagate(False)
        for text, width, _ in cols:
            ctk.CTkLabel(hdr, text=text, width=width, anchor="w",
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color="#00ddcc").pack(side="left", padx=1)

        # Data rows
        for rank, r in enumerate(results, 1):
            score = r["total"]
            is_above = score >= threshold
            row_bg = "#0f2a1a" if is_above else "#0a1a10"

            row = ctk.CTkFrame(self.table_frame, fg_color=row_bg, height=30)
            row.pack(fill="x", padx=2, pady=1)
            row.pack_propagate(False)

            for i, (_, width, extract) in enumerate(cols):
                text, color, fsize, bold = extract(r, rank)
                lbl = ctk.CTkLabel(row, text=text, width=width, anchor="w",
                             font=ctk.CTkFont(size=fsize, weight="bold" if bold else "normal"),
                             text_color=color)
                lbl.pack(side="left", padx=1)
                # Make ticker clickable
                if i == 1:  # Ticker column
                    lbl.configure(cursor="hand2")
                    lbl.bind("<Button-1>", lambda e, t=r["ticker"], rf=row: self._toggle_stock_news(t, rf, rank))

        # Update header
        self.result_count_label.configure(
            text=f"{len(results)} stocks scanned  |  {len([r for r in results if r['total'] >= threshold])} above {threshold}+")

        # Update summary stat cards
        self._update_summary(results)

    def _ma_text(self, r):
        if r.get("ma_crossed_above"):
            ago = r.get("crossover_bars_ago", -1)
            cnt = r.get("crossover_count", 0)
            return f"^ X{ago}({cnt})" if cnt > 1 else f"^ X{ago}"
        elif r.get("ma_bullish"):
            return "^ Bull"
        return "v Bear"

    def _ma_color(self, r):
        if r.get("ma_crossed_above"):
            return "#00ff88"
        elif r.get("ma_bullish"):
            return "#aaff00"
        return "#ff4444"

    def _toggle_stock_news(self, ticker: str, row_frame, rank):
        """Expand/collapse news inline below the stock row."""
        # Check if news frame already exists for this ticker
        for child in row_frame.master.winfo_children():
            if getattr(child, '_news_ticker', None) == ticker:
                # Collapse: remove the news frame
                child.destroy()
                return

        # Collapse any other open news
        for child in row_frame.master.winfo_children():
            if getattr(child, '_news_ticker', None):
                child.destroy()

        # Create news frame below the row
        news_frame = ctk.CTkFrame(row_frame.master, fg_color="#0a1a15", corner_radius=4)
        news_frame._news_ticker = ticker
        news_frame.pack(after=row_frame, fill="x", padx=8, pady=(0, 2))

        # Loading
        loading = ctk.CTkLabel(news_frame, text="Loading news...",
                               font=ctk.CTkFont(size=11),
                               text_color="#6a8a6a")
        loading.pack(pady=10)

        def _sentiment(title, summary=""):
            """Simple keyword-based sentiment: Good / Bad / Neutral."""
            words = set((title + " " + summary).lower().split())
            g = len(words & SENTIMENT_GOOD)
            b = len(words & SENTIMENT_BAD)
            if g > b:
                return "Good"
            elif b > g:
                return "Bad"
            return "Neutral"

        def _parse_date(date_str):
            """Parse ISO date string to date object, return None on failure."""
            try:
                from datetime import datetime
                if not date_str:
                    return None
                # Strip trailing Z and truncate to 19 chars (YYYY-MM-DDTHH:MM:SS)
                clean = date_str.rstrip("Z").strip()[:19]
                for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(clean, fmt).date()
                    except ValueError:
                        continue
                return None
            except Exception:
                return None

        def _fetch():
            try:
                from datetime import date, timedelta
                import yfinance as yf
                stock = yf.Ticker(f"{ticker}.NS")
                news_items = stock.news or []
                cutoff = date.today() - timedelta(days=60)  # 2 months
                parsed = []
                for item in news_items:
                    content = item.get("content", item)
                    title = content.get("title", "")
                    if not title:
                        continue
                    summary = content.get("summary", "")
                    pub_date = content.get("pubDate", "") or content.get("displayTime", "")
                    # Filter: only last 2 months
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
                             text_color="#ff4444").pack(pady=8)
                return
            # Sentiment summary
            good = sum(1 for i in items if i["sentiment"] == "Good")
            bad = sum(1 for i in items if i["sentiment"] == "Bad")
            neu = len(items) - good - bad
            summary_text = f"{good} Good  |  {bad} Bad  |  {neu} Neutral"
            ctk.CTkLabel(news_frame, text=summary_text,
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color="#aaff00").pack(pady=(4, 6))

            for item in items:
                sent = item["sentiment"]
                sent_color = {"Good": "#00ff88", "Bad": "#ff4444", "Neutral": "#6a8a6a"}[sent]
                sent_bg = {"Good": "#0a3a1a", "Bad": "#3a0a0a", "Neutral": "#1a1a1a"}[sent]

                card = ctk.CTkFrame(news_frame, fg_color="#0f2a1a", corner_radius=4)
                card.pack(fill="x", padx=6, pady=2)

                # Top row: Good/Bad badge + date + provider
                top = ctk.CTkFrame(card, fg_color="transparent")
                top.pack(fill="x", padx=8, pady=(4, 0))

                badge = ctk.CTkFrame(top, fg_color=sent_bg, corner_radius=3, height=18)
                badge.pack(side="left")
                badge.pack_propagate(False)
                ctk.CTkLabel(badge, text=sent, font=ctk.CTkFont(size=9, weight="bold"),
                             text_color=sent_color).pack(padx=4, pady=1)

                meta = f"{item['date']}  {item['provider']}" if item['provider'] else item['date']
                ctk.CTkLabel(top, text=meta,
                             font=ctk.CTkFont(size=9),
                             text_color="#6a8a6a").pack(side="left", padx=6)

                # Title
                ctk.CTkLabel(card, text=item["title"],
                             font=ctk.CTkFont(size=11, weight="bold"),
                             text_color="#c8d8c0", wraplength=900,
                             anchor="w", justify="left").pack(anchor="w", padx=8, pady=(1, 0))

                # Summary
                if item["summary"]:
                    ctk.CTkLabel(card, text=item["summary"],
                                 font=ctk.CTkFont(size=10),
                                 text_color="#8aaa8a", wraplength=900,
                                 anchor="w", justify="left").pack(anchor="w", padx=8, pady=(1, 4))

        def _show_error(msg):
            if news_frame.winfo_exists():
                loading.configure(text=f"Error: {msg}", text_color="#ff4444")

        threading.Thread(target=_fetch, daemon=True).start()

    def _scan_complete(self):
        """Re-enable UI after scan finishes."""
        self.scanning = False
        self.run_btn.configure(state="normal", text="RUN SCAN", fg_color="#00aa55")
        self.progress_label.configure(text="Done")
        if self.results:
            self.html_btn.configure(state="normal")
            self.csv_btn.configure(state="normal")
            self.clear_btn.configure(state="normal")

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_html(self):
        if not self.results:
            return
        threshold = self.settings.get("min_score", 50)
        tf_names = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
        tf_label = tf_names.get(self.settings.get("timeframe", "D"), "Daily")
        html = generate_html_report(
            self.results,
            title=f"HMAxEMA Scanner — {self.universe_var.get()} — {tf_label}",
            threshold=threshold)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scanner_report_{timestamp}.html"
        filepath = os.path.join(SCANNER_DIR, filename)
        save_report(html, filepath)
        self._log(f"HTML report saved: {filename}")
        webbrowser.open(f"file://{os.path.abspath(filepath)}")

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
                    "EXCELLENT" if r["total"] >= 70 else ("GOOD" if r["total"] >= 50 else ("MODERATE" if r["total"] >= 30 else "POOR")),
                    r.get("close"), r["trend"], r["momentum"], r["rsi"], r["macd"],
                    r["stoch"], r["obv"], r["volume"], r["rel_str"], r["volatility"],
                    r.get("fundamentals", 0),
                    r["trend_dir"], r.get("rsi_val"), r.get("adx_val"),
                    "Yes" if r.get("is_sideways") else "No" + (f" ({sideways_reasons})" if sideways_reasons else ""),
                    r.get("pc1m"), r.get("pc3m")
                ])

        self._log(f"CSV saved: {filename}")
        os.startfile(filepath) if sys.platform == "win32" else os.system(f"open '{filepath}'")

    # ── Utilities ────────────────────────────────────────────────────────────

    def _clear_cache(self):
        """Clear the disk cache for all data providers."""
        try:
            from .data_providers import DataProvider
            provider = DataProvider()
            provider.clear_cache()
            self._log("Cache cleared successfully")
        except Exception as e:
            self._log(f"Failed to clear cache: {e}")
    
    def _clear_results(self):
        """Clear the results table and reset state."""
        # Clear the table
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        # Reset results
        self.results = []
        
        # Update labels
        self.result_count_label.configure(text="0 stocks scanned")
        
        # Reset summary cards
        self._reset_summary()
        
        # Reset progress
        self.progress.set(0)
        self.progress_label.configure(text="Ready")
        
        # Disable buttons
        self.html_btn.configure(state="disabled")
        self.csv_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        
        # Log
        self._log("Results cleared")

    def _copy_selection(self):
        """Copy selected text from log to clipboard."""
        try:
            selection = self.log_text.get("sel.first", "sel.last")
            if selection:
                self.clipboard_clear()
                self.clipboard_append(selection)
        except Exception:
            pass

    def _rotate_log(self):
        """Overwrite log file if it's older than LOG_ROTATE_HOURS."""
        try:
            if os.path.exists(LOG_FILE):
                age_hours = (datetime.now().timestamp() - os.path.getmtime(LOG_FILE)) / 3600
                if age_hours >= LOG_ROTATE_HOURS:
                    open(LOG_FILE, "w").close()  # Truncate
        except Exception:
            pass

    def _log(self, msg: str):
        """Thread-safe log to the log textbox and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"

        # Write to file
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

        def _append():
            self.log_text.insert("end", line)
            self.log_text.see("end")

        self.after(0, _append)

    def _set_progress(self, value: float, text: str = ""):
        """Thread-safe progress update."""
        def _update():
            self.progress.set(value)
            if text:
                self.progress_label.configure(text=text)
        self.after(0, _update)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = ScannerApp()
    app.mainloop()
