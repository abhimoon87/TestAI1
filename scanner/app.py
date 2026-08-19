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
from .data_fetcher import fetch_stock_data, fetch_index_data, fetch_stock_fast, fetch_batch_yfinance
from .scoring import compute_scores
from .report import generate_html_report, save_report

# ── Theme ────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

SCANNER_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(SCANNER_DIR, "settings.json")

# ── Default Settings (mirrors Pine Script indicator) ─────────────────────────
DEFAULT_SETTINGS = {
    # Moving Averages
    "fast_ma_type": "HMA",
    "fast_ma_len": 20,
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
    "slope_ma_type": "EMA",
    "slope_ma_len": 50,
    "slope_lookback": 10,
    "flat_threshold": 0.5,
    # Step Channel
    "sc_pivot_len": 3,
    "sc_bands_mult": 0.6,
    # MA Crossover
    "crossover_lookback": 4,
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

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_sidebar()
        self._build_main_area()

    def _build_sidebar(self):
        """Left sidebar: universe, settings, run button."""
        sidebar = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color="#0d1f14")
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
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
            values=["All", "Bullish Only", "Bearish Only", "MA + POC Only", "Crossover Only", "Entry Signals Only"],
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

        # ── Results Table (ScrolledFrame - takes remaining space) ────────────
        self.table_frame = ctk.CTkScrollableFrame(
            main, fg_color="#0a1a10")
        self.table_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.table_frame.grid_columnconfigure(0, weight=1)
        self.table_frame.grid_rowconfigure(0, weight=1)

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
            self._log(f"Starting scan: {universe_name} ({len(tickers)} stocks)")
            self._log(f"Timeframe: {tf_names.get(timeframe, timeframe)} | Period: {period} | Filter: {trend_filter}")
            self._log(f"FastMA={settings['fast_ma_type']}{settings['fast_ma_len']} "
                       f"SlowMA={settings['slow_ma_type']}{settings['slow_ma_len']} "
                       f"RSI={settings['rsi_len']} Threshold={settings['min_score']}")

            # Fetch NIFTY index
            self._set_progress(0, "Fetching NIFTY 50 index...")
            index_df = fetch_index_data("^NSEI", period=period)
            if index_df is not None:
                self._log(f"NIFTY 50 index loaded ({len(index_df)} bars)")
            else:
                self._log("Warning: NIFTY index unavailable, using proxy for RS")

            # FAST: Batch download all stocks at once via yfinance
            self._set_progress(0.05, f"Batch downloading {len(tickers)} stocks...")
            self._log(f"Batch downloading {len(tickers)} stocks via yfinance...")
            batch_data = fetch_batch_yfinance(tickers, period=period)
            self._log(f"Batch download complete: {len(batch_data)}/{len(tickers)} stocks fetched")

            # Score stocks from batch data
            results = []
            total = len(batch_data)
            scored = 0

            for i, (ticker, df) in enumerate(batch_data.items(), 1):
                progress = 0.1 + (i / total * 0.9) if total > 0 else 0.5
                self._set_progress(progress, f"[{i}/{total}] Scoring {ticker}")

                try:
                    if df is not None and not df.empty:
                        scores = compute_scores(
                            df, index_df=index_df,
                            fast_ma_type=settings["fast_ma_type"],
                            fast_ma_len=settings["fast_ma_len"],
                            slow_ma_type=settings["slow_ma_type"],
                            slow_ma_len=settings["slow_ma_len"],
                            rsi_len=settings["rsi_len"],
                            vol_ma_len=settings["vol_ma_len"],
                            atr_len=settings["atr_len"],
                            rs_length=settings["rs_length"],
                            adx_len=settings["adx_len"],
                            adx_threshold=settings["adx_threshold"],
                            chop_len=settings["chop_len"],
                            chop_threshold=settings["chop_threshold"],
                            slope_ma_type=settings["slope_ma_type"],
                            slope_ma_len=settings["slope_ma_len"],
                            slope_lookback=settings["slope_lookback"],
                            flat_threshold=settings["flat_threshold"],
                            sc_pivot_len=settings["sc_pivot_len"],
                            sc_bands_mult=settings["sc_bands_mult"],
                            vp_lookback=settings["vp_lookback"],
                            vp_rows=settings["vp_rows"],
                            vp_width=settings["vp_width"],
                            crossover_lookback=settings["crossover_lookback"],
                        )
                        if scores is not None:
                            scores["ticker"] = ticker

                            # Apply trend filter
                            if trend_filter == "Bullish Only" and scores["trend_dir"] != "Bull":
                                continue
                            elif trend_filter == "Bearish Only" and scores["trend_dir"] != "Bear":
                                continue
                            elif trend_filter == "MA + POC Only":
                                # Only show stocks with MA bullish alignment AND above POC
                                if not (scores.get("ma_bullish", False) and scores.get("above_poc", False)):
                                    continue
                            elif trend_filter == "Crossover Only":
                                # Only show stocks that had a crossover in the lookback period
                                if not scores.get("ma_crossed_above", False):
                                    continue
                            elif trend_filter == "Entry Signals Only":
                                # Only show stocks that meet the full swing-entry strategy:
                                # (1) Fast MA crossed above Slow MA AND close above crossover level
                                # (2) close above Volume Profile POC AND above crossover level
                                # (3) techno-fundamental score >= 50
                                if not scores.get("entry_signal", False):
                                    continue

                            results.append(scores)
                            scored += 1
                            if scored % 10 == 0 or scored <= 5:
                                self._log(f"  {ticker}: {scores['total']:.1f}/100 ({scores['trend_dir']})")
                except Exception as e:
                    pass  # Silently skip errors in batch mode

            # Sort and store
            results.sort(key=lambda x: x["total"], reverse=True)
            self.results = results

            # Update UI
            passed = len([r for r in results if r["total"] >= settings["min_score"]])
            filtered_out = len(tickers) - len(results) if trend_filter != "All" else 0
            if trend_filter == "MA + POC Only":
                filter_msg = f", {filtered_out} filtered (require MA bullish + above POC)" if filtered_out > 0 else ""
            else:
                filter_msg = f", {filtered_out} filtered by {trend_filter}" if filtered_out > 0 else ""
            self._log(f"\nScan complete: {len(results)} stocks scored, {passed} above {settings['min_score']} threshold{filter_msg}")

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

        # Header row
        header_row = ctk.CTkFrame(self.table_frame, fg_color="#153520", height=32)
        header_row.pack(fill="x", padx=2, pady=(2, 0))
        header_row.pack_propagate(False)

        headers = [("Rank", 40), ("Ticker", 100), ("Score", 60), ("Rating", 90), ("ENTRY", 60),
                   ("Price", 80), ("MA", 50), ("POC", 50), ("Trend/15", 70), ("Mom/15", 70), ("RSI/8", 55),
                   ("MACD/7", 60), ("Vol/10", 60), ("RS/10", 55), ("Fund/20", 65),
                   ("1M Chg", 70), ("Direction", 75), ("ADX", 50), ("Sideways", 55)]

        for text, width in headers:
            ctk.CTkLabel(header_row, text=text, width=width,
                          font=ctk.CTkFont(size=10, weight="bold"),
                          text_color="#00ddcc").pack(side="left", padx=1)

        # Data rows
        for rank, r in enumerate(results, 1):
            score = r["total"]
            is_above = score >= threshold

            row_bg = "#0f2a1a" if is_above else "#0a1a10"
            row = ctk.CTkFrame(self.table_frame, fg_color=row_bg, height=30)
            row.pack(fill="x", padx=2, pady=1)
            row.pack_propagate(False)

            # Rank
            rank_color = "#00ff88" if score >= 70 else ("#aaff00" if score >= 50 else ("#ffaa00" if score >= 30 else "#ff4444"))
            ctk.CTkLabel(row, text=str(rank), width=40,
                          font=ctk.CTkFont(size=11, weight="bold"),
                          text_color=rank_color).pack(side="left", padx=1)

            # Ticker
            ctk.CTkLabel(row, text=r["ticker"], width=100,
                          font=ctk.CTkFont(size=11, weight="bold"),
                          text_color="#00ff88" if is_above else "#c8d8c0",
                          anchor="w").pack(side="left", padx=1)

            # Score
            score_color = "#00ff88" if score >= 70 else ("#aaff00" if score >= 50 else ("#ffaa00" if score >= 30 else "#ff4444"))
            ctk.CTkLabel(row, text=f"{score:.1f}", width=60,
                          font=ctk.CTkFont(size=12, weight="bold"),
                          text_color=score_color).pack(side="left", padx=1)

            # Rating badge (using combined rating)
            rating = r.get('combined_rating', 'POOR')
            if rating == "EXCELLENT":
                badge_color = "#0a3a1a"
                badge_text = "#00ff88"
            elif rating == "GOOD":
                badge_color = "#1a3a0a"
                badge_text = "#aaff00"
            elif rating == "MODERATE":
                badge_color = "#3a2a0a"
                badge_text = "#ffaa00"
            else:  # POOR
                badge_color = "#3a0a0a"
                badge_text = "#ff4444"
            badge_frame = ctk.CTkFrame(row, fg_color=badge_color, corner_radius=3, height=20)
            badge_frame.pack(side="left", padx=2)
            ctk.CTkLabel(badge_frame, text=rating, width=85,
                          font=ctk.CTkFont(size=9, weight="bold"),
                          text_color=badge_text).pack(padx=4, pady=2)

            # ENTRY signal badge (full swing-entry strategy met?)
            entry_signal = r.get("entry_signal", False)
            entry_badge = ctk.CTkFrame(row, fg_color="#0a3a1a" if entry_signal else "#3a0a0a",
                                       corner_radius=3, height=20)
            entry_badge.pack(side="left", padx=2)
            ctk.CTkLabel(entry_badge, text="ENTRY" if entry_signal else "—", width=55,
                          font=ctk.CTkFont(size=9, weight="bold"),
                          text_color="#00ff88" if entry_signal else "#ff4444").pack(padx=4, pady=2)

            # Price
            ctk.CTkLabel(row, text=f"\u20b9{r.get('close', 0):.1f}", width=80,
                          font=ctk.CTkFont(size=11),
                          text_color="#c8d8c0").pack(side="left", padx=1)

            # MA Signal
            ma_bullish = r.get('ma_bullish', False)
            ma_crossed = r.get('ma_crossed_above', False)
            crossover_ago = r.get('crossover_bars_ago', -1)
            crossover_count = r.get('crossover_count', 0)
            if ma_crossed:
                if crossover_count > 1:
                    ma_text = f"^ X{crossover_ago}({crossover_count})"
                else:
                    ma_text = f"^ X{crossover_ago}"
                ma_color = "#00ff88"
            elif ma_bullish:
                ma_text = "^ Bull"
                ma_color = "#aaff00"
            else:
                ma_text = "v Bear"
                ma_color = "#ff4444"
            ctk.CTkLabel(row, text=ma_text, width=70,
                          font=ctk.CTkFont(size=10),
                          text_color=ma_color).pack(side="left", padx=1)

            # POC Signal
            above_poc = r.get('above_poc', False)
            vp_poc = r.get('vp_poc', 0)
            if above_poc:
                poc_text = f"Above"
                poc_color = "#00ff88"
            else:
                poc_text = f"Below"
                poc_color = "#ff4444"
            ctk.CTkLabel(row, text=poc_text, width=50,
                          font=ctk.CTkFont(size=10),
                          text_color=poc_color).pack(side="left", padx=1)

            # Category scores (with mini bar)
            for cat, max_val, color in [
                ("trend", 15, "#00ff88"), ("momentum", 15, "#00ddcc"),
                ("rsi", 8, "#00aaff"), ("macd", 7, "#aa88ff"),
                ("volume", 10, "#ffaa00"), ("rel_str", 10, "#aaff00"),
                ("fundamentals", 20, "#ffe600")
            ]:
                val = r.get(cat, 0)
                bar_width = int((val / max_val) * 50) if max_val > 0 else 0

                cat_frame = ctk.CTkFrame(row, fg_color="transparent", width=70, height=20)
                cat_frame.pack(side="left", padx=1)
                cat_frame.pack_propagate(False)

                # Mini bar background
                ctk.CTkFrame(cat_frame, fg_color="#153520",
                              width=50, height=6, corner_radius=2).place(x=0, y=7)
                # Mini bar fill
                if bar_width > 0:
                    ctk.CTkFrame(cat_frame, fg_color=color,
                                  width=max(bar_width, 2), height=6, corner_radius=2).place(x=0, y=7)
                # Value label
                ctk.CTkLabel(cat_frame, text=f"{val:.0f}", width=20,
                              font=ctk.CTkFont(size=9),
                              text_color="#6a8a6a").place(x=52, y=1)

            # 1M Change
            pc1m = r.get("pc1m", 0) or 0
            pc1m_color = "#00ff88" if pc1m > 0 else "#ff4444"
            ctk.CTkLabel(row, text=f"{pc1m:+.1f}%", width=70,
                          font=ctk.CTkFont(size=11),
                          text_color=pc1m_color).pack(side="left", padx=1)

            # Direction
            dir_color = "#00ff88" if r["trend_dir"] == "Bull" else "#ff4444"
            dir_icon = "^" if r["trend_dir"] == "Bull" else "v"
            ctk.CTkLabel(row, text=f"{dir_icon} {r['trend_dir']}", width=75,
                          font=ctk.CTkFont(size=11),
                          text_color=dir_color).pack(side="left", padx=1)

            # ADX
            adx_val = r.get("adx_val")
            adx_text = f"{adx_val:.0f}" if adx_val is not None else "---"
            ctk.CTkLabel(row, text=adx_text, width=50,
                          font=ctk.CTkFont(size=11),
                          text_color="#c8d8c0").pack(side="left", padx=1)

            # Sideways indicator
            is_sideways = r.get("is_sideways", False)
            sideways_text = "\u26a0 Chop" if is_sideways else "\u2713"
            sideways_color = "#ffaa00" if is_sideways else "#00ff88"
            ctk.CTkLabel(row, text=sideways_text, width=55,
                          font=ctk.CTkFont(size=10),
                          text_color=sideways_color).pack(side="left", padx=1)

        # Update header
        self.result_count_label.configure(
            text=f"{len(results)} stocks scanned  |  {len([r for r in results if r['total'] >= threshold])} above {threshold}+")

        # Update summary stat cards
        self.after(0, lambda: self._update_summary(results))

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
            from data_providers import DataProvider
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

    def _log(self, msg: str):
        """Thread-safe log to the log textbox."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"

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
