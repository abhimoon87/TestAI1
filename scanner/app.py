"""
HMAxEMA Stock Scanner — GUI Application (v2 "Aurora" UI)
Modern dark desktop app for scanning Indian stocks.

Layout (inspired by community-app dashboards):
    [icon rail] [nav sidebar + scan controls] [main: hero, stats, results] [profile panel]

Usage:
    python scanner/app.py

Or double-click run.bat (Windows) / run.sh (macOS/Linux)
"""

import json
import logging
import math
import os
import sys
import threading
import tkinter as tk
import webbrowser
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
from PIL import Image

from .trace import setup_trace

try:
    setup_trace()
except Exception:
    pass

logger = logging.getLogger(__name__)
logger.info("app module loaded -- trace active at %s", __import__("pathlib").Path(__file__).parent / "trace.log")

from .report import (
    _sentiment,
    generate_html_report,
    save_report,
)
from .themes import THEMES
from .universes import UNIVERSES

SCANNER_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(SCANNER_DIR, "settings.json")
LOG_FILE = os.path.join(SCANNER_DIR, "scan.log")
LOG_ROTATE_HOURS = 12  # Overwrite log file after 12 hours


def apply_theme(app, theme_name: str):
    """Apply a theme to the entire application."""
    ctk.set_appearance_mode(THEMES[theme_name]["ctk_mode"])
    app.current_theme = theme_name
    app.theme_colors = THEMES[theme_name]
    app.settings["theme"] = theme_name
    save_settings(app.settings)


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
    # Dead-symbol cache
    "negative_cache_ttl_hours": 24,
    # UI
    "theme": "dark",
}


def load_settings() -> dict:
    """Load settings from JSON file, falling back to defaults."""
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            settings.update(saved)
        except Exception as e:
            logger.debug("Failed to load settings: %s", e)
    return settings


def save_settings(settings: dict):
    """Save settings to JSON file."""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.debug("Failed to save settings: %s", e)


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class GradientCanvas(tk.Canvas):
    """Tk canvas that paints a smooth horizontal gradient behind its children."""

    def set_gradient(self, hex_colors, horizontal: bool = True):
        self._colors = [_hex_to_rgb(c) for c in hex_colors]
        self._horizontal = horizontal
        self._paint()

    def _paint(self):
        if not hasattr(self, "_colors") or len(self._colors) < 2:
            return
        self.delete("grad")
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        # Subtle vignette overlay for depth
        steps = max((w if self._horizontal else h) // 2, 1)
        n = len(self._colors) - 1
        for i in range(steps):
            t = i / max(steps - 1, 1) * n
            seg = min(int(t), n - 1)
            f = t - seg
            c1, c2 = self._colors[seg], self._colors[seg + 1]
            rgb = tuple(int(c1[j] + (c2[j] - c1[j]) * f) for j in range(3))
            col = "#{:02x}{:02x}{:02x}".format(*rgb)
            if self._horizontal:
                self.create_line(i * 2, 0, i * 2, h, width=2, fill=col, tags="grad")
            else:
                self.create_line(0, i * 2, w, i * 2, width=3, fill=col, tags="grad")
        self.tag_lower("grad")


class AvatarRing(tk.Canvas):
    """Decorative multi-color ring around a circle label — like the mock's avatar."""

    def __init__(self, master, size=84, letter="H", bg="#0c1e13",
                 ring_colors=("#00ddcc", "#00ff88", "#aaff00"), **kw):
        super().__init__(master, width=size, height=size, bg=bg,
                         highlightthickness=0, bd=0, **kw)
        self._size = size
        self._letter = letter
        self._ring = ring_colors
        self.bind("<Configure>", lambda e: self._draw())

    def _draw(self):
        self.delete("all")
        s = min(self.winfo_width(), self.winfo_height()) or self._size
        cx = cy = s / 2
        r_out = s / 2 - 3
        for k, col in enumerate(self._ring):
            start = 15 + k * 110
            extent = 200 + k * 35
            self.create_arc(cx - r_out, cy - r_out, cx + r_out, cy + r_out,
                            start=start, extent=extent, style="arc",
                            outline=col, width=2)
        r_in = s / 2 - 12
        self.create_oval(cx - r_in, cy - r_in, cx + r_in, cy + r_in,
                         fill="#12331f", outline="#1a4a2a", width=1)
        self.create_text(cx, cy, text=self._letter,
                         font=("Segoe UI", int(s / 3.4), "bold"), fill="#8dffc4")


class RunningBull(ctk.CTkFrame):
    """Running bull — displays the provided bull gif/png centered in grid."""

    def __init__(self, master, width=220, height=110, bg="#0f271c", **kw):
        super().__init__(master, width=width, height=height, fg_color=bg, **kw)
        self._bull_w = width
        self._bull_h = height
        self._frame = 0
        self._running = False
        self._after_id = None
        self._bg = bg
        # Load bull image (gif/png) from assets
        self._ctk_img = None
        self._label = None
        try:
            # Try gif first, then png
            candidates = [
                Path(__file__).parent / "assets" / "bull.gif",
                Path(__file__).parent / "assets" / "bull.png",
                Path(__file__).parent / "assets" / "bull_small.png",
            ]
            img_path = next((p for p in candidates if p.exists()), None)
            if img_path is not None:
                pil = Image.open(img_path).convert("RGBA")
                # Fit inside 220x110, keep aspect
                pil.thumbnail((200, 90), Image.LANCZOS)
                # For CTkImage, keep reference
                self._ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
                self._label = ctk.CTkLabel(self, image=self._ctk_img, text="", fg_color="transparent")
                self._label.place(relx=0.5, rely=0.5, anchor="center")
            else:
                # Fallback text if image missing
                self._label = ctk.CTkLabel(self, text="\U0001F402", font=ctk.CTkFont(size=48), text_color="#2E86DE", fg_color="transparent")
                self._label.place(relx=0.5, rely=0.5, anchor="center")
        except Exception:
            # Fallback to text bull on any load error
            try:
                self._label = ctk.CTkLabel(self, text="\U0001F402", font=ctk.CTkFont(size=48), text_color="#2E86DE", fg_color="transparent")
                self._label.place(relx=0.5, rely=0.5, anchor="center")
            except Exception:
                pass
        # Ground line under bull
        try:
            self._ground = tk.Canvas(self, width=width, height=14, bg=bg, highlightthickness=0, bd=0)
            self._ground.place(relx=0.5, rely=0.92, anchor="center")
            self._ground.create_line(18, 7, width - 18, 7, fill="#1e4a2f", width=2)
        except Exception:
            self._ground = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._tick()

    def stop(self):
        self._running = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self._frame = 0

    def _tick(self):
        if not self._running or not self.winfo_exists():
            return
        self._frame = (self._frame + 1) % 20
        # Bobbing animation + subtle dust
        try:
            bob = math.sin(self._frame * 0.314) * 3.0
            if self._label is not None and self._label.winfo_exists():
                self._label.place_configure(rely=0.5 + bob * 0.008)
            if getattr(self, "_ground", None) is not None and self._ground.winfo_exists():
                self._ground.delete("dust")
                dust_phase = self._frame % 4
                for i, dx in enumerate([0, 28]):
                    alpha = 0.35 if (dust_phase + i) % 2 == 0 else 0.15
                    col = "#2a5a3a" if alpha > 0.3 else "#1e3a2a"
                    self._ground.create_oval(28 + dx, 2, 38 + dx, 8, fill=col, outline="", tags="dust")
        except Exception:
            pass
        if self._running:
            self._after_id = self.after(80, self._tick)


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
        self._apply_cache_settings()
        self.results = []
        self.scanning = False
        self.filter_text = ""
        self.active_view = "dashboard"
        # Pagination / virtualization for 5,900 rows
        self.page_size = 100
        self.current_page = 0
        self.all_results: list = []  # full unfiltered
        self.filtered_results: list = []  # after search filter
        # Sorting state for scan results grid
        self.sort_col: int | None = None
        self.sort_reverse: bool = False
        self._pending_stream_after: str | None = None

        # Apply saved theme
        theme_name = self.settings.get("theme", "dark")
        ctk.set_appearance_mode(THEMES[theme_name]["ctk_mode"])
        self.current_theme = theme_name
        self.theme_colors = THEMES[theme_name]

        self._build_ui()
        self._load_settings_to_ui()
        self._rotate_log()
        # Pre-warm symbol disk cache in background (avoids 5-40s block on first scan)
        def _warm_symbols():
            try:
                from .symbol_fetcher import _load_disk_cache
                from .universes import get_universe
                _load_disk_cache()
                get_universe("FULL MARKET (NSE+BSE ~5,900)")
            except Exception:
                pass
        try:
            self.after(800, lambda: threading.Thread(target=_warm_symbols, daemon=True).start())
        except Exception:
            pass

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
        _, _b3 = add_rail_item(3, "play", self._start_scan, c["green"])
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
            anchor="w", height=38, corner_radius=10,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold" if active else "normal"),
            fg_color=c["nav_active"] if active else "transparent",
            hover_color=c["card_hover"] if not active else c["nav_active"],
            text_color=c["text"] if active else c["text_dim"],
            border_width=1 if active else 0,
            border_color=c["border"],
            command=lambda: self._show_view(view))
        item.pack(fill="x", padx=10, pady=2)
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
                         font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
                         text_color=c["text_faint"]).pack(padx=14, pady=(14, 3), anchor="w")

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

        # Rating Filter
        section_label("Rating Filter")
        self.rating_filter_var = ctk.StringVar(value="All")
        ctk.CTkOptionMenu(
            ctrl, variable=self.rating_filter_var,
            values=["All", "Excellent", "Good", "Moderate", "Poor"],
            command=self._on_rating_change,
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
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=c["purple"], hover_color=c["purple_hover"],
            text_color="#052e16", height=42, corner_radius=12,
            border_width=1, border_color=c["border"],
            command=self._start_scan)
        self.run_btn.pack(fill="x", pady=(0, 6))

        # Stop — enabled only while a scan is running; calls engine.cancel()
        self.stop_btn = ctk.CTkButton(
            bottom, text="\u23f9   STOP",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=c["red"], hover_color=c.get("red_hover", c["pink"]),
            text_color="#2a0505", height=36, corner_radius=12,
            border_width=1, border_color=c["border"],
            state="disabled",
            command=self._stop_scan)
        self.stop_btn.pack(fill="x", pady=(0, 8))

        self.progress = ctk.CTkProgressBar(
            bottom, height=6, corner_radius=3,
            fg_color=c["progress_bg"], progress_color=c["progress_fg"])
        self.progress.pack(fill="x")
        self.progress.set(0)

        self.progress_label = ctk.CTkLabel(
            bottom, text="Ready", font=ctk.CTkFont(size=10),
            text_color=c["text_dim"], anchor="w")
        self.progress_label.pack(fill="x", pady=(2, 0))

        # Dead-symbol (negative) cache — status + manual clear
        self.cache_row = ctk.CTkFrame(bottom, fg_color="transparent")
        self.cache_status_lbl = ctk.CTkLabel(
            self.cache_row, text="", font=ctk.CTkFont(size=10),
            text_color=c["text_dim"], anchor="w")
        self.cache_status_lbl.pack(side="left", fill="x", expand=True)
        self.cache_clear_btn = ctk.CTkButton(
            self.cache_row, text="Clear", width=48, height=22,
            font=ctk.CTkFont(size=10),
            fg_color=c["option_btn"], hover_color=c["red_hover"],
            text_color=c["text"], corner_radius=8,
            command=self._clear_negative_cache)
        self.cache_row.pack(fill="x", pady=(4, 0))
        self._refresh_neg_cache_ui()

        # Enrichment cache — status + manual clear (phase-2 provider results)
        self.enrich_cache_row = ctk.CTkFrame(bottom, fg_color="transparent")
        self.enrich_cache_status_lbl = ctk.CTkLabel(
            self.enrich_cache_row, text="", font=ctk.CTkFont(size=10),
            text_color=c["text_dim"], anchor="w")
        self.enrich_cache_status_lbl.pack(side="left", fill="x", expand=True)
        self.enrich_cache_clear_btn = ctk.CTkButton(
            self.enrich_cache_row, text="Clear", width=48, height=22,
            font=ctk.CTkFont(size=10),
            fg_color=c["option_btn"], hover_color=c["red_hover"],
            text_color=c["text"], corner_radius=8,
            command=self._clear_enrichment_cache)
        self.enrich_cache_row.pack(fill="x", pady=(2, 0))
        self._refresh_enrich_cache_ui()

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
        self.table_frame.grid(row=3, column=0, sticky="ew", padx=6, pady=(6, 6))
        self.table_frame.grid_columnconfigure(0, weight=1)

        self._render_table_header()

        # Pagination bar — virtualizes 5,900 rows into 100-row pages
        self.pagination_frame = ctk.CTkFrame(view, fg_color="transparent")
        self.pagination_frame.grid(row=4, column=0, sticky="ew", padx=6, pady=(0, 12))
        self.pagination_frame.grid_remove()  # hidden until results
        pag = self.pagination_frame
        self.page_prev_btn = ctk.CTkButton(pag, text="◀ Prev", width=80, height=28, corner_radius=8,
                                           fg_color=c["card"], hover_color=c["card_hover"], border_width=1, border_color=c["border"],
                                           text_color=c["text"], command=lambda: self._change_page(-1))
        self.page_prev_btn.pack(side="left", padx=2)
        self.page_label = ctk.CTkLabel(pag, text="Page 1 / 1", font=ctk.CTkFont(size=11), text_color=c["text_dim"])
        self.page_label.pack(side="left", padx=10)
        self.page_next_btn = ctk.CTkButton(pag, text="Next ▶", width=80, height=28, corner_radius=8,
                                           fg_color=c["card"], hover_color=c["card_hover"], border_width=1, border_color=c["border"],
                                           text_color=c["text"], command=lambda: self._change_page(1))
        self.page_next_btn.pack(side="left", padx=2)
        # Page size selector
        ctk.CTkLabel(pag, text="Rows:", font=ctk.CTkFont(size=10), text_color=c["text_dim"]).pack(side="left", padx=(16, 4))
        self.page_size_var = ctk.StringVar(value="100")
        ctk.CTkOptionMenu(pag, variable=self.page_size_var, values=["50", "100", "200", "500"],
                          width=80, height=28, corner_radius=8,
                          fg_color=c["card"], button_color=c["card2"], text_color=c["text"],
                          command=self._on_page_size_change).pack(side="left")
        ctk.CTkButton(pag, text="Load All (no pagination)", width=140, height=28, corner_radius=8,
                      fg_color="transparent", hover_color=c["card"], border_width=1, border_color=c["border"],
                      text_color=c["text_dim"], command=self._load_all_pages).pack(side="right")

        # Empty-state hint
        self.empty_label = ctk.CTkLabel(
            self.table_frame,
            text="\nNo results yet \u2014 hit \u25b6 RUN SCAN\n",
            font=ctk.CTkFont(size=13), text_color=c["text_dim"])
        self.empty_label.pack(pady=30, anchor="center")

    def _get_sort_key(self, col_idx: int):
        """Return a key function for sorting by column index."""
        # Rating order (higher = better)
        rating_order = {"EXCELLENT": 4, "GOOD": 3, "MODERATE": 2, "POOR": 1, "WEAK": 0}

        def _ma_rank(r):
            # Crossed above > Bullish > Bearish
            if r.get("ma_crossed_above"):
                return 2
            if r.get("ma_bullish"):
                return 1
            return 0

        sort_keys = {
            0: lambda r: r.get("total", 0),  # # — same as Score
            1: lambda r: r.get("ticker", ""),
            2: lambda r: r.get("total", 0),
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

    def _on_sort(self, col_idx: int):
        """Handle header click — toggle sort column/direction."""
        # Determine new sort state
        if self.sort_col == col_idx:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_col = col_idx
            # Default direction: Ticker & # ascending, others descending (higher is better)
            if col_idx in (0, 1):
                self.sort_reverse = False
            else:
                self.sort_reverse = True
        self.current_page = 0
        self._render_current_page()

    def _render_table_header(self):
        """Render the compact column-header bar as the first row of table_frame."""
        c = self.theme_colors
        hdr = ctk.CTkFrame(self.table_frame, fg_color=c["card2"], corner_radius=10,
                           height=32, border_width=1, border_color=c["border"])
        hdr.pack(fill="x", pady=(0, 6))
        hdr.pack_propagate(False)
        hdr.grid_propagate(False)
        for idx, (text, width) in enumerate(RESULT_COLS):
            is_sorted = (self.sort_col == idx)
            arrow = " \u25b2" if is_sorted and not self.sort_reverse else (" \u25bc" if is_sorted else "")
            label_text = f"{text}{arrow}"
            # Use label with hand cursor and click binding for sorting
            lbl = ctk.CTkLabel(hdr, text=label_text, width=width, anchor="w",
                               font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
                               text_color=c["cyan"] if not is_sorted else c["green"])
            lbl.pack(side="left", padx=2, pady=2)
            # Make header clickable for sorting (all columns sortable)
            lbl.configure(cursor="hand2")
            lbl.bind("<Button-1>", lambda e, i=idx: self._on_sort(i))
            # Also bind the same on the parent frame area for larger click target
            # Tooltip hint
            ToolTip(lbl, f"Sort by {text}{' (desc)' if is_sorted and self.sort_reverse else ' (asc)' if is_sorted else ''}")

    def _build_summary_panel(self, parent):
        """Build the summary statistic cards (values updated after a scan)."""
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
        self.summary_cards = {}
        for label, key, color, icon in stats:
            card = ctk.CTkFrame(parent, fg_color=c["card"], corner_radius=14,
                                border_width=1, border_color=c["border"])
            card.pack(side="left", fill="both", expand=True, padx=5, pady=2)
            # subtle top accent line
            accent = ctk.CTkFrame(card, height=2, fg_color=color, corner_radius=1)
            accent.pack(fill="x", padx=10, pady=(8, 0))
            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=8, pady=(4, 0))
            ctk.CTkLabel(top, text=icon, font=ctk.CTkFont(size=10),
                         text_color=color).pack(side="left")
            ctk.CTkLabel(top, text=label, font=ctk.CTkFont(family="Segoe UI", size=8, weight="bold"),
                         text_color=c["text_faint"]).pack(side="left", padx=4)
            val_label = ctk.CTkLabel(card, text="\u2014",
                                     font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
                                     text_color=color)
            val_label.pack(pady=(2, 10))
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
            ("Data / Cache", [
                ("Dead-Symbol Cache TTL (hours)", "negative_cache_ttl_hours", "int", (1, 168)),
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
        # Re-filter with pagination
        if hasattr(self, 'all_results'):
            self._display_results(self.all_results)
        else:
            self._display_results(self.results)

    def _rating_filter(self) -> str:
        """Current rating dropdown selection, uppercased ('ALL' = no filter)."""
        if hasattr(self, "rating_filter_var"):
            return str(self.rating_filter_var.get()).upper()
        return "ALL"

    def _row_matches_filters(self, r: dict) -> bool:
        """Whether a result row passes the search-text AND rating filters."""
        if self.filter_text and self.filter_text not in r.get("ticker", "").upper():
            return False
        rating = self._rating_filter()
        if rating != "ALL":
            combined = (r.get("combined_rating") or "POOR").upper()
            if combined != rating:
                return False
        return True

    def _on_rating_change(self, choice: str = ""):
        """Rating dropdown changed — re-filter the grid in place (no re-scan)."""
        if hasattr(self, "all_results"):
            self._display_results(self.all_results)

    # ── Settings Management ──────────────────────────────────────────────

    def _apply_cache_settings(self):
        """Push the persisted dead-symbol cache TTL into data_fetcher."""
        try:
            from . import data_fetcher
            data_fetcher.set_negative_cache_ttl_hours(
                self.settings.get("negative_cache_ttl_hours", 24)
            )
        except Exception as e:
            logger.debug("Could not apply cache TTL setting: %s", e)
        self._refresh_neg_cache_ui()

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
        # Show placeholder immediately, fetch live count in background for 5,900
        try:
            base = len(UNIVERSES.get(choice, []))
        except Exception:
            base = 0
        # Quick estimate for live universes
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
        self.universe_count_label.configure(text=label + " ...")
        # Background fetch to get accurate live count without blocking UI
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
                    self.after(0, lambda: self.universe_count_label.configure(text=lbl))
            except Exception:
                pass
        import threading
        threading.Thread(target=_bg, daemon=True).start()

    def _on_threshold_change(self, val):
        self.threshold_label.configure(text=str(int(float(val))))

    # ════════════════════════════════════════════════════════════════════════
    # SCANNING
    # ════════════════════════════════════════════════════════════════════════

    def _start_scan(self):
        if self.scanning:
            return

        self.settings = self._collect_settings()
        save_settings(self.settings)
        self._apply_cache_settings()

        self.scanning = True
        self._scan_cancelled = False
        c = self.theme_colors
        self.run_btn.configure(state="disabled", text="\u23f3   SCANNING\u2026",
                               fg_color=c["card2"])
        self.stop_btn.configure(state="normal", text="\u23f9   STOP")
        self.html_btn.configure(state="disabled")
        self.csv_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        self.results = []

        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self._render_table_header()
        # Centered running bull animation — visible until first batch arrives
        c = self.theme_colors
        self._bull_frame = ctk.CTkFrame(self.table_frame, fg_color="transparent")
        self._bull_frame.pack(pady=28, anchor="center", fill="x")
        bull_bg = c.get("main_bg", c.get("panel_bg", "#0f271c"))
        self._bull_anim = RunningBull(self._bull_frame, width=220, height=108, bg=bull_bg)
        self._bull_anim.pack(anchor="center")
        self._bull_anim.start()
        ctk.CTkLabel(self._bull_frame, text="Scanning — fetching batches…",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=c["green"]).pack(pady=(6, 0), anchor="center")
        ctk.CTkLabel(self._bull_frame, text="First results appear after ~1 batch (~20s)",
                     font=ctk.CTkFont(size=11),
                     text_color=c["text_dim"]).pack(anchor="center")

        thread = threading.Thread(target=self._run_scan, daemon=True)
        thread.start()

    def _stop_scan(self):
        """Cancel the running scan — the engine stops at the next safe point
        (chunk boundary / batch end) and partial results stay on the grid."""
        if not self.scanning:
            return
        engine = getattr(self, "_scan_engine", None)
        self.stop_btn.configure(state="disabled", text="\u23f9   STOPPING\u2026")
        if engine is not None:
            engine.cancel()
            self._log("Stop requested — finishing the current batch, then stopping...")
        else:
            self._log("Stop requested...")

    # ── Negative (dead-symbol) cache ──────────────────────────────────────

    def _refresh_neg_cache_ui(self):
        """Update the dead-symbol cache status label to match the on-disk cache."""
        if not hasattr(self, "cache_status_lbl"):
            return  # sidebar not built yet (early startup call)
        try:
            from . import data_fetcher
            n = len(data_fetcher._negative_cache_load())
            ttl_h = data_fetcher.negative_cache_ttl_hours()
        except Exception:
            n, ttl_h = 0, 24
        if n:
            self.cache_status_lbl.configure(
                text=f"Dead-symbol cache: {n} (auto-resets ~{ttl_h}h)"
            )
            self.cache_clear_btn.pack(side="right", padx=(6, 0))
        else:
            self.cache_status_lbl.configure(text="Dead-symbol cache: empty")
            self.cache_clear_btn.pack_forget()

    def _clear_negative_cache(self):
        """Forget every marked-dead symbol — fallback will re-attempt them all."""
        try:
            from . import data_fetcher
            data_fetcher._negative_cache_update(
                clears=list(data_fetcher._negative_cache_load().keys())
            )
            self._log("Cleared dead-symbol cache — fallback will re-attempt all symbols")
        except Exception as e:
            self._log(f"Could not clear dead-symbol cache: {e}")
        self._refresh_neg_cache_ui()

    # ── Enrichment cache ─────────────────────────────────────────────────

    def _refresh_enrich_cache_ui(self):
        """Update the enrichment cache status label to match the on-disk cache."""
        if not hasattr(self, "enrich_cache_status_lbl"):
            return  # sidebar not built yet (early startup call)
        try:
            from . import data_fetcher
            n = data_fetcher.enrichment_cache_size()
            ttl_h = data_fetcher.ENRICHMENT_CACHE_TTL_HOURS
        except Exception:
            n, ttl_h = 0, 24
        if n:
            self.enrich_cache_status_lbl.configure(
                text=f"Enrichment cache: {n} (auto-resets ~{ttl_h}h)"
            )
            self.enrich_cache_clear_btn.pack(side="right", padx=(6, 0))
        else:
            self.enrich_cache_status_lbl.configure(text="Enrichment cache: empty")
            self.enrich_cache_clear_btn.pack_forget()

    def _clear_enrichment_cache(self):
        """Wipe cached phase-2 provider results — next scan re-fetches them."""
        try:
            from . import data_fetcher
            data_fetcher.enrichment_cache_clear()
            self._log("Cleared enrichment cache — next scan will re-fetch phase-2 data")
        except Exception as e:
            self._log(f"Could not clear enrichment cache: {e}")
        self._refresh_enrich_cache_ui()

    def _on_stream_batch(self, batch: list):
        """Incremental grid update — debounced + dict-indexed (O(n))."""
        if not batch:
            return

        # Coalesce rapid batches: cancel pending render, schedule newest
        if self._pending_stream_after is not None:
            try:
                self.after_cancel(self._pending_stream_after)
            except Exception:
                pass

        def _do():
            self._pending_stream_after = None
            # Merge into master lists (replace existing ticker if already present)
            existing = {r.get("ticker"): idx for idx, r in enumerate(self.all_results)}
            filtered_idx = {r.get("ticker"): idx for idx, r in enumerate(self.filtered_results)}
            added = 0
            updated = 0
            for r in batch:
                t = r.get("ticker")
                if t in existing:
                    self.all_results[existing[t]] = r
                    updated += 1
                else:
                    self.all_results.append(r)
                    existing[t] = len(self.all_results) - 1
                    added += 1
                # Filtered mirror (search text AND rating dropdown)
                if self._row_matches_filters(r):
                    if t in filtered_idx:
                        self.filtered_results[filtered_idx[t]] = r
                    else:
                        self.filtered_results.append(r)
                        filtered_idx[t] = len(self.filtered_results) - 1
                else:
                    if t in filtered_idx:
                        self.filtered_results.pop(filtered_idx[t])
                        # Rebuild index for subsequent items in the same batch
                        filtered_idx = {fr.get("ticker"): i for i, fr in enumerate(self.filtered_results)}
            self.results = list(self.all_results)
            try:
                self._render_current_page()
            except Exception as e:
                logger.debug("Stream render failed: %s", e)
            self._log(f"Grid updated: +{added} new, ~{updated} enriched (total {len(self.all_results)})")

        # Debounce 120ms — coalesces progress+render floods during streaming
        self._pending_stream_after = self.after(120, _do)

    def _run_scan(self):
        """Run the scan in a background thread via ScannerEngine (streaming)."""
        try:
            from .scanner_engine import ScannerEngine

            universe_name = self.universe_var.get()
            settings = self.settings

            # Reset incremental state before streaming
            self.all_results = []
            self.filtered_results = []
            self.results = []
            self.current_page = 0
            self.after(0, self._render_current_page)

            engine = ScannerEngine()
            self._scan_engine = engine  # keep ref for cancel if needed
            engine.set_progress_callback(
                lambda p, m: self.after(0, lambda: self._set_progress(p, m))
            )
            engine.set_log_callback(lambda m: self.after(0, lambda: self._log(m)))

            # Streaming: each batch (~200-1000 tickers) appears on grid immediately
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

            # Final sync — ensure sorted/filtered state matches full result (main-thread safe)
            def _final_sync():
                self.results = result.results
                self.all_results = list(result.results)
                self.filtered_results = [r for r in result.results if self._row_matches_filters(r)]
                self._render_current_page()
                if result.cancelled:
                    self._log(f"Scan stopped — showing {len(result.results)} partial results.")
                if result.error:
                    self._log(f"Scan finished with error: {result.error}")

            self.after(0, _final_sync)

        except Exception as e:
            self._log(f"\nERROR: {e!s}")
        finally:
            self.after(0, self._scan_complete)

    def _display_results(self, results):
        """Display filtered results with pagination (100/page) for 5,900 rows."""
        # Store full results for pagination / filtering
        self.all_results = list(results)
        self.filtered_results = [r for r in results if self._row_matches_filters(r)]
        self.current_page = 0
        self._render_current_page()

    def _render_current_page(self):
        """Render only the current page (virtualized) — keeps 5,900 rows snappy."""
        tc = self.theme_colors
        results = getattr(self, 'all_results', [])
        shown = getattr(self, 'filtered_results', results)
        # Apply column sorting if active (sort filtered view before pagination)
        if self.sort_col is not None and shown:
            try:
                key_fn = self._get_sort_key(self.sort_col)
                shown = sorted(shown, key=key_fn, reverse=self.sort_reverse)
            except Exception as e:
                logger.debug("Sort failed for col %s: %s", self.sort_col, e)
        # Clear table and re-render header bar (stop bull animation first)
        try:
            if getattr(self, "_bull_anim", None) is not None:
                self._bull_anim.stop()
        except Exception:
            pass
        self._bull_anim = None
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self._render_table_header()

        if not shown:
            # While scanning with no data yet — show centered running bull animation
            if getattr(self, "scanning", False) and not results:
                bull_bg = tc.get("main_bg", tc.get("panel_bg", "#0f271c"))
                bull_frame = ctk.CTkFrame(self.table_frame, fg_color="transparent")
                bull_frame.pack(pady=30, anchor="center", fill="x")
                # Stop previous bull if any
                try:
                    if getattr(self, "_bull_anim", None) is not None:
                        self._bull_anim.stop()
                except Exception:
                    pass
                self._bull_anim = RunningBull(bull_frame, width=220, height=108, bg=bull_bg)
                self._bull_anim.pack(anchor="center")
                self._bull_anim.start()
                ctk.CTkLabel(bull_frame, text="Scanning \u2014 fetching batches...",
                             font=ctk.CTkFont(size=12, weight="bold"),
                             text_color=tc["green"]).pack(pady=(6, 0), anchor="center")
                ctk.CTkLabel(bull_frame, text="First results appear after ~1 batch (~20s)",
                             font=ctk.CTkFont(size=11),
                             text_color=tc["text_dim"]).pack(anchor="center")
            else:
                has_active_filter = bool(self.filter_text) or self._rating_filter() != "ALL"
                msg = ("No results match your filter." if results and has_active_filter
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

            # Pagination — slice to current page
            page_size = int(getattr(self, 'page_size', 100))
            total_pages = max(1, (len(shown) + page_size - 1) // page_size)
            if getattr(self, 'current_page', 0) >= total_pages:
                self.current_page = total_pages - 1
            self.current_page = max(self.current_page, 0)
            start = self.current_page * page_size
            page_shown = shown[start:start + page_size]
            # Update pagination bar
            try:
                if hasattr(self, 'pagination_frame'):
                    if shown and len(shown) > page_size:
                        self.pagination_frame.grid()
                        self.page_label.configure(text=f"Page {self.current_page+1} / {total_pages}  ({len(shown)} stocks)")
                        self.page_prev_btn.configure(state="normal" if self.current_page > 0 else "disabled")
                        self.page_next_btn.configure(state="normal" if self.current_page < total_pages - 1 else "disabled")
                    else:
                        self.pagination_frame.grid_remove()
            except Exception:
                pass

            for rank, r in enumerate(page_shown, start + 1):
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
                                 lambda e, t=r["ticker"], rf=row, rk=rank: self._toggle_stock_news(t, rf, rk))

        # Header meta — use filtered count but indicate pagination
        if results:
            threshold = self.settings.get("min_score", 50)
            filter_parts = []
            if self.filter_text:
                filter_parts.append(f"'{self.filter_text}'")
            rating = self._rating_filter()
            if rating != "ALL":
                filter_parts.append(f"rating {rating.title()}")
            suffix = f"  |  filter: {', '.join(filter_parts)} ({len(shown)})" if filter_parts else ""
            if len(shown) > page_size:
                suffix += f"  |  showing {len(page_shown)} of {len(shown)} (page {self.current_page+1}/{total_pages})"
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

    def _change_page(self, delta: int):
        """Paginate; delta -1 / +1 or 0 to refresh."""
        shown = getattr(self, 'filtered_results', getattr(self, 'all_results', []))
        page_size = int(getattr(self, 'page_size', 100))
        total_pages = max(1, (len(shown) + page_size - 1) // page_size)
        new_page = getattr(self, 'current_page', 0) + delta
        new_page = max(0, min(new_page, total_pages - 1))
        self.current_page = new_page
        self._render_current_page()

    def _on_page_size_change(self, value: str):
        try:
            self.page_size = int(value)
        except Exception:
            self.page_size = 100
        self.current_page = 0
        self._render_current_page()

    def _load_all_pages(self):
        """Render all results — capped at 500 rows to prevent UI freeze."""
        MAX_RENDERED_ROWS = 500
        total = len(getattr(self, 'filtered_results', []))
        self.page_size = min(MAX_RENDERED_ROWS, total) if total > 0 else MAX_RENDERED_ROWS
        self.current_page = 0
        self._render_current_page()

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
                self.after(0, lambda err=str(e): _show_error(err))

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
        # Stop running bull animation if still present
        try:
            if getattr(self, "_bull_anim", None) is not None:
                self._bull_anim.stop()
                self._bull_anim = None
        except Exception:
            pass
        c = self.theme_colors
        self.run_btn.configure(state="normal", text="\u25b6   RUN SCAN",
                               fg_color=c["purple"])
        cancelled = getattr(self, "_scan_cancelled", False)
        self.stop_btn.configure(state="disabled", text="\u23f9   STOP")
        self.progress_label.configure(text="Stopped" if cancelled else "Done")
        self.status_label.configure(text="Status: Stopped" if cancelled else "Status: Done")
        self._refresh_neg_cache_ui()  # scans may have marked new symbols dead
        self._refresh_enrich_cache_ui()  # phase-2 results may now be cached
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
        # Dead symbols skipped via the negative cache during this scan
        try:
            from . import data_fetcher
            dead_skips = data_fetcher.negative_cache_skip_count()
        except Exception:
            dead_skips = 0
        self.summary_cards["dead_skip"].configure(text=str(dead_skips))

    def _reset_summary(self):
        """Reset the summary cards to their empty state."""
        for key in ("total", "passed", "entry", "avg", "high", "bull", "bear", "dead_skip"):
            if hasattr(self, "summary_cards") and key in self.summary_cards:
                self.summary_cards[key].configure(text="\u2014")

    # ════════════════════════════════════════════════════════════════════════
    # EXPORT
    # ════════════════════════════════════════════════════════════════════════

    def _export_html(self):
        if not self.results:
            return
        # Prevent re-entry while generating
        try:
            self.html_btn.configure(state="disabled", text="\u23f3")
        except Exception:
            pass
        threshold = self.settings.get("min_score", 50)
        tf_names = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
        tf_label = tf_names.get(self.settings.get("timeframe", "D"), "Daily")
        # Snapshot results/universe on UI thread to avoid race
        results_snapshot = list(self.results)
        universe_name = self.universe_var.get()
        safe_threshold = threshold
        safe_title = f"HMAxEMA Scanner — {universe_name} — {tf_label}"

        def _bg():
            try:
                self._log("Fetching news sentiment for exported stocks...")
                html = generate_html_report(
                    results_snapshot,
                    title=safe_title,
                    threshold=safe_threshold,
                    fetch_news=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scanner_report_{timestamp}.html"
                filepath = os.path.join(SCANNER_DIR, filename)
                save_report(html, filepath)
                self.after(0, lambda: self._log(f"HTML report saved: {filename}"))
                self.after(0, lambda: webbrowser.open(f"file://{os.path.abspath(filepath)}"))
            except Exception as e:
                logger.exception("HTML export failed: %s", e)
                err_msg = str(e)
                self.after(0, lambda m=err_msg: self._log(f"HTML export failed: {m}"))
            finally:
                self.after(0, lambda: self.html_btn.configure(state="normal", text="\u2913"))

        import threading as _th
        _th.Thread(target=_bg, daemon=True).start()

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
        # B605 fix: avoid shell injection; validate inside scanner dir
        try:
            import pathlib
            import subprocess

            safe_path = pathlib.Path(filepath).resolve()
            if sys.platform == "win32":
                os.startfile(str(safe_path))  # nosec B606
            else:
                subprocess.Popen(["open", str(safe_path)])  # nosec B606
        except Exception:
            pass

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
        try:
            if getattr(self, "_bull_anim", None) is not None:
                self._bull_anim.stop()
                self._bull_anim = None
        except Exception:
            pass
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self._render_table_header()
        c = self.theme_colors
        self.results = []
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


class ToolTip:
    """Minimal tooltip for icon buttons."""

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.attributes("-topmost", True)
        lbl = tk.Label(self.tip, text=self.text, bg="#0f2a1a", fg="#c8d8c0",
                       font=("Segoe UI", 9), padx=8, pady=4)
        lbl.pack()
        self.tip.wm_geometry(f"+{x}+{y}")

    def _hide(self, event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = ScannerApp()
    app.mainloop()
