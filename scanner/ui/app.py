"""
HMAxEMA Stock Scanner — GUI Application (Flet Edition)

Modern dark desktop app for scanning Indian stocks.
Layout: [icon rail] [nav sidebar + scan controls] [main: hero, stats, results] [profile panel]

Usage:
    python -m scanner
    python scanner/app.py
"""

import asyncio
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from datetime import datetime

import flet as ft

from ..shared.constants import LOG_MAX_LINES, has_fii_dii, score_of
from ..shared.trace import setup_trace

try:
    setup_trace()
except Exception:
    logging.getLogger(__name__).debug("Trace setup failed", exc_info=True)

logger = logging.getLogger(__name__)

from ..backend.settings_store import (  # noqa: F401 — re-exported for tests
    DEFAULT_SETTINGS,
    STRATEGY_LABELS,
    load_settings,
)
from ..shared.themes import THEMES
from ..shared.universes import UNIVERSES
from .ui_kit import (
    RADIUS_LG,
    RADIUS_MD,
    TIER_COMPACT,
    TIER_WIDE,
    PaletteAction,
    filter_actions,
    parse_watchlist_text,
    width_tier,
)
from .views_audit import AuditViewMixin
from .views_cache import CacheViewMixin
from .views_detail import DetailViewMixin
from .views_export import ExportViewMixin
from .views_layout import SIDE_W, LayoutViewMixin
from .views_panels import PanelsViewMixin
from .views_results import ResultsViewMixin
from .views_scan import ScanOrchestrationMixin
from .views_settings import SettingsViewMixin

SCANNER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(SCANNER_DIR))
APPLOG_DIR = os.path.join(_PROJECT_ROOT, "AppLog")
REPORTS_DIR = os.path.join(_PROJECT_ROOT, "Reports")
LOG_FILE = os.path.join(APPLOG_DIR, "scan.log")

# The UI lock guards control mutation + page.update() in _safe_update.
# It MUST be re-entrant: _safe_update runs the wrapped fn while holding it,
# and wrapped fns re-enter _safe_update on the same thread (e.g.
# _scan_complete -> _toast -> _safe_update). On the page's UI thread that
# nested call runs inline, so a plain Lock self-deadlocks and freezes the app
# right after every scan. Kept as a module-level factory so the regression
# test in tests/test_app_threading.py pins the re-entrant behaviour.
_UI_LOCK_FACTORY = threading.RLock

_log_lock = threading.Lock()
_log_q: queue.Queue = queue.Queue()
_log_writer_running = False


def _log_writer():
    """Single background writer for scan.log — file I/O never blocks a caller."""
    while True:
        line = _log_q.get()
        try:
            os.makedirs(APPLOG_DIR, exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            logger.info("Failed to write to log file", exc_info=True)


class ScannerApp(
    ScanOrchestrationMixin,
    LayoutViewMixin,
    ResultsViewMixin,
    PanelsViewMixin,
    DetailViewMixin,
    SettingsViewMixin,
    CacheViewMixin,
    AuditViewMixin,
    ExportViewMixin,
):
    def __init__(self, page: ft.Page):
        self.page = page
        self.settings = load_settings()
        self._apply_cache_settings()
        self.results = []
        self.all_results = []
        self.filtered_results = []
        self._row_pool = {}
        self._row_cells = {}
        # Detail-panel background load: per-ticker attempt/done slots
        self._inst_attempted = set()
        self._inst_done = set()
        # Must stay re-entrant: _render_current_page holds this lock while
        # _visible_results (and _display_results before it) re-acquires it.
        # Downgrading to threading.Lock deadlocks the results grid.
        self._results_lock = threading.RLock()
        # Lock ordering contract (checked 2026-09, keep it this way):
        # _ui_lock -> _results_lock only. _safe_update takes _ui_lock and the
        # wrapped fn may take _results_lock (scan callbacks, _final_sync);
        # nothing takes _results_lock and then touches _ui_lock/page (all UI
        # pushes happen after the results lock is released). _scan_lock and
        # _state_lock are leaf locks — short sections, never nested.
        #
        # Must stay re-entrant: _safe_update runs the wrapped fn *while holding
        # _ui_lock*, and several wrapped fns re-enter _safe_update on the same
        # thread — _scan_complete calls _toast, and _toast calls _safe_update
        # to present the SnackBar. On the page's UI thread _run_on_ui_thread
        # executes that nested call inline, so a plain Lock self-deadlocks
        # there, freezing the whole app right after every scan (the engine's
        # "<- ScannerEngine.scan_stream ok" line is the last thing ever
        # logged). An RLock degrades that nested acquisition to a no-op.
        self._ui_lock = _UI_LOCK_FACTORY()
        self._scan_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._hover_last_ms = 0.0
        self.scanning = False
        self._scan_epoch = 0
        self.filter_text = ""
        self.last_warnings: list[str] = []
        self.active_view = "dashboard"
        self.page_size = 100
        self.current_page = 0
        self.sort_col = None
        self.sort_reverse = False

        self.current_theme = "dark"
        self.theme_colors = THEMES["dark"]

        # Responsive layout state — tier drives pane visibility + which
        # table columns render (see ui_kit.width_tier / hidden_cols).
        self.width_tier = "wide"
        self._pending_tier = None
        self._right_pinned = False
        self._auto_side = False
        # Grid view toggles: row density preset + columns the user turned
        # off in the Columns menu (both persisted as ui_* prefs).
        self.row_density = "comfort"
        self.user_hidden_cols: set[str] = set()

        # Initial window geometry — set exactly once here. View switches and
        # theme changes rebuild controls but must never touch the window, so a
        # maximized window stays maximized (no resize/re-layout shock).
        self.page.window.width = 1600
        self.page.window.height = 900
        self.page.window.min_width = 1280
        self.page.window.min_height = 800

        self._build_ui()
        # Global shortcuts (palette, run/stop, views, export, theme).
        self.page.on_keyboard_event = self._on_keyboard_event
        # Responsive reflow — fires only on tier changes, never per-pixel.
        self.page.on_resize = self._on_page_resize
        self._load_settings_to_ui()
        self._load_ui_prefs()
        # Prefs load after the UI is built, so re-stamp the menu checks.
        self._sync_menu_checks()
        self._log("Scanner ready — pick a universe and hit RUN SCAN")

        def _warm_symbols():
            try:
                from ..api.symbol_fetcher import _load_disk_cache
                from ..shared.universes import get_universe

                _load_disk_cache()
                get_universe("FULL MARKET (NSE+BSE ~5,900)")
            except Exception:
                logger.info("Symbol warm-up failed", exc_info=True)

        def _startup_load():
            """Saved results + cache status off the main thread — first paint
            must not wait on a disk/db read or a 100-row rebuild."""
            try:
                from ..backend.settings_store import load_results

                rows = load_results()
            except Exception:
                logger.info("Failed to load saved results", exc_info=True)
                rows = None
            if rows:
                self._safe_update(lambda: self._restore_saved_results(rows))
            self._refresh_cache_ui_bg()

        threading.Thread(target=_warm_symbols, daemon=True).start()
        threading.Thread(target=self._warm_market, daemon=True).start()
        threading.Thread(target=_startup_load, daemon=True).start()

        # News prefetcher — runs in background after each scan completes
        from ..backend.news_prefetch import NewsPrefetcher

        self._news_prefetcher = NewsPrefetcher(
            get_results=lambda: self.all_results,
            results_lock=self._results_lock,
            get_epoch=lambda: self._scan_epoch,
            get_cancelled=lambda: self._scan_cancelled,
            on_update=lambda: self._safe_update(self._flush_news_badges),
            on_log=lambda m: self._safe_update(lambda: self._log(m)),
        )

    def _apply_cache_settings(self):
        try:
            from ..api import cache_manager

            cache_manager.set_negative_ttl(
                self.settings.get("negative_cache_ttl_hours", 24)
            )
        except Exception:
            logger.info("Failed to apply cache TTL settings", exc_info=True)

    def _restore_saved_results(self, rows: list | None = None):
        """Repopulate the grid from last_results.json after a relaunch.

        ``rows`` may be pre-loaded by a caller that did the disk read on a
        worker thread (the startup path).
        """
        try:
            if rows is None:
                from ..backend.settings_store import load_results

                rows = load_results()
            if not rows:
                return
            self.results = rows
            self._display_results(rows)
            self.html_btn.disabled = False
            self.csv_btn.disabled = False
            self.clear_btn.disabled = False
            self._log(f"Restored {len(rows)} results from last scan")
        except Exception:
            logger.info("Failed to restore saved results", exc_info=True)

    def _build_ui(self):
        c = self.theme_colors
        self.page.bgcolor = c["main_bg"]
        self.page.padding = 0
        self.page.title = "HMAxEMA Stock Scanner — Indian Market"

        theme_mode = "dark"
        self.page.theme_mode = theme_mode

        self.page.controls.clear()

        self.sidebar = self._build_sidebar()
        # Rebuilds (theme switch) must preserve the collapse state — a fresh
        # Container defaults to expanded. Hide via width, not visible (see
        # the ponytail note in _apply_width_tier: visible blanks the main area).
        if getattr(self, "_sidebar_collapsed", False):
            self.sidebar.width = 0
        self.right_panel = self._build_right_panel()
        # Rebuilds must re-derive auto-hide state for the current tier.
        self.right_panel.visible = self.width_tier != TIER_COMPACT or self._right_pinned
        self.main_row = ft.Row(
            controls=[
                self._build_rail(),
                self.sidebar,
                self._build_main_area(),
                self.right_panel,
            ],
            spacing=0,
            expand=True,
        )
        self.page.add(self.main_row)

    # ── View switching ──────────────────────────────────────────────────

    def _restore_main_area(self):
        """Return to dashboard — settings overlay off, hero/grid reload independently.

        Dashboard is never hidden or re-laid-out (it sits under a Stack
        overlay). Only the settings overlay is dismissed, then hero and
        grid are re-applied in separate try/except blocks so one failure
        cannot blank the other.

        Row pool is cleared so ``_render_current_page`` full-rebuilds —
        the streaming fast-path can leave a blank grid after Settings.
        """
        rows = self.all_results or self.results
        self.active_view = "dashboard"
        if hasattr(self, "_row_pool"):
            self._row_pool.clear()
        if hasattr(self, "_row_cells"):
            self._row_cells.clear()
        # Overlay only — do not touch dashboard_view.visible/opacity/layout.
        settings = getattr(self, "settings_view", None)
        if settings is not None:
            settings.visible = False
        # No page.update() here — _display_results below pushes once with
        # the overlay dismissal + hero/grid in a single send.
        self._load_settings_to_ui()

        # Hero independent of grid.
        try:
            if getattr(self, "_last_market", None) is not None:
                self._render_market(self._last_market)
        except Exception:
            logger.info("Hero reload failed", exc_info=True)

        # Grid independent of hero (always render, even empty).
        try:
            self._display_results(rows)
        except Exception:
            logger.info("Grid reload failed", exc_info=True)

        engine = getattr(self, "_scan_engine", None)
        if engine is not None and self.scanning:
            # Same throttled registrations as scan start — raw callbacks
            # here cost one full page.update() per ticker/line (203ms each).
            engine.set_progress_callback(
                lambda p, m: self._throttle(
                    "progress",
                    0.5,
                    lambda: self._safe_update(lambda: self._set_progress(p, m)),
                )
            )
            engine.set_log_callback(self._engine_log)

    def _motion_reduced(self) -> bool:
        """User preference: suppress decorative transitions/animations."""
        try:
            return bool(self.settings.get("reduce_motion", False))
        except Exception:
            return False

    def _fade_view_in(self, view):
        """Soft fade-in after a view switch.

        Sets opacity 0, pushes that frame, then animates to 1 — the view must
        carry ``animate_opacity`` (set in ``_build_main_area``). Falls back to
        a plain single update when reduce_motion is on or the view predates
        the animation wiring (partial builds in tests).
        """
        try:
            if self._motion_reduced() or getattr(view, "animate_opacity", None) is None:
                self.page.update()
                return
            view.opacity = 0.0
            self.page.update()
            view.opacity = 1.0
            self.page.update()
        except Exception:
            logger.info("View fade failed", exc_info=True)

    def _show_view(self, name):
        old = getattr(self, "active_view", None)
        self.active_view = name
        logger.debug("view_switch: from=%s to=%s", old, name)
        for vname, pill in self._rail_pills.items():
            if vname == name:
                pill.opacity = 1.0
            else:
                pill.opacity = 0.0
        lbl = getattr(self, "topbar_title", None)
        if lbl is not None:
            lbl.value = "Scanner" if name == "dashboard" else name.title()
        if name == "dashboard":
            # Overlay off + independent hero/grid reload; dashboard never re-laid-out.
            self._restore_main_area()
        else:
            self.page.update()

    def _show_settings(self, e=None):
        self.active_view = "settings"
        for vname, pill in self._rail_pills.items():
            pill.opacity = 1.0 if vname == "settings" else 0.0
        lbl = getattr(self, "topbar_title", None)
        if lbl is not None:
            lbl.value = "Settings"
        # Strategy lives in the sidebar too — re-sync this page's dropdown
        # from the saved setting before showing (the page is built once).
        ctrl = getattr(self, "_settings_inputs", {}).get("strategy")
        if ctrl is not None:
            ctrl.value = STRATEGY_LABELS.get(
                str(self.settings.get("strategy", "daily_screener")),
                "Daily Screener",
            )
        # Overlay on — dashboard stays visible and laid out underneath.
        self.settings_view.visible = True
        self._fade_view_in(self.settings_view)

    # ── Modern chrome: shortcuts, palette, toasts ──────────────────────

    def _toggle_right_panel(self, e=None):
        """Pin/unpin the profile panel (compact tiers auto-hide it)."""
        self._right_pinned = not getattr(self, "_right_pinned", False)
        rp = getattr(self, "right_panel", None)
        if rp is not None:
            rp.visible = self._right_pinned or self.width_tier != TIER_COMPACT
        try:
            self.page.update()
        except Exception:
            logger.info("Right panel toggle failed", exc_info=True)

    def _on_page_resize(self, e=None):
        """Adopt a new width tier on resize — only on tier *changes*.

        During a streaming scan the reflow is queued instead: forcing a
        pool clear + full rebuild under a live stream would repaint all
        rows on the UI thread mid-scan. The queue drains at scan teardown.
        """
        t0 = time.monotonic()
        try:
            tier = width_tier(getattr(self.page, "width", None))
            if tier == getattr(self, "width_tier", TIER_WIDE):
                # Same tier: skip the reflow, but still drop any queued tier
                # — a resize that returned to where it started means the
                # pending entry is stale, and applying it at teardown would
                # reflow the app into a layout the window no longer matches.
                self._pending_tier = None
                return
            if self.scanning:
                self._pending_tier = tier
                return
            self._apply_width_tier(tier)
        except Exception:
            logger.info("Resize handling failed", exc_info=True)
        ms = (time.monotonic() - t0) * 1000.0
        if ms > 200.0:
            logger.info("SLOW _on_page_resize: %.0fms tier=%s", ms, self.width_tier)

    def _apply_width_tier(self, tier, render=True):
        """Pane + column reflow for a new width tier."""
        self.width_tier = tier
        self._sync_menu_checks()  # tier decides which columns are visible
        compact = tier == TIER_COMPACT

        rp = getattr(self, "right_panel", None)
        if rp is not None:
            rp.visible = not compact or getattr(self, "_right_pinned", False)

        # Sidebar auto-collapses into compact; widen restores it unless
        # the user had collapsed it manually before (or during) compact.
        # ponytail: hide via width, not visible — flipping visible on this
        # sidebar blanks the main area client-side (Flet 1.0 repaint bug;
        # visible=False on right_panel is fine), width=0 renders identically.
        box = getattr(self, "sidebar", None)
        if compact and not getattr(self, "_sidebar_collapsed", False):
            self._sidebar_collapsed = True
            self._auto_side = True
            if box is not None:
                box.width = 0
        elif not compact and getattr(self, "_auto_side", False):
            self._auto_side = False
            self._sidebar_collapsed = False
            if box is not None:
                box.width = SIDE_W

        if not render:
            return
        # Column set changed → regenerate header and full-rebuild rows.
        holder = getattr(self, "header_holder", None)
        if holder is not None:
            holder.controls = []
        if getattr(self, "_row_pool", None):
            self._row_pool.clear()
        if getattr(self, "_row_cells", None):
            self._row_cells.clear()
        if self.active_view == "dashboard":
            self._render_current_page()
        try:
            self.page.update()
        except Exception:
            logger.info("Width-tier update failed", exc_info=True)

    def _apply_pending_width_tier(self):
        """Drain a resize deferred during a streaming scan (no re-render —
        the caller re-renders right after with the new tier in effect)."""
        tier = getattr(self, "_pending_tier", None)
        if tier is None:
            return
        self._pending_tier = None
        if tier != getattr(self, "width_tier", TIER_WIDE):
            self._apply_width_tier(tier, render=False)

    def _toast(self, msg, kind="info"):
        """Non-modal feedback via a SnackBar dialog (thread-safe).

        Flet 1.0 removed ``page.snack_bar``/``page.open`` — overlays now go
        through ``page.show_dialog`` (the same call also presents AlertDialog).
        """

        def _show():
            c = self.theme_colors
            if kind == "success":
                bg, fg, accent = c["green"], c["on_accent"], c["on_accent"]
            elif kind == "error":
                bg, fg, accent = c["red"], "#ffffff", "#ffffff"
            else:
                bg, fg, accent = c["card2"], c["text"], c["cyan"]
            bar = ft.SnackBar(
                content=ft.Row(
                    [
                        ft.Container(
                            width=4, height=26, border_radius=2, bgcolor=accent
                        ),
                        ft.Text(str(msg), color=fg, size=13, expand=True),
                    ],
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                bgcolor=bg,
                shape=ft.RoundedRectangleBorder(radius=RADIUS_MD),
                open=True,
            )
            self.page.show_dialog(bar)
            # No page.update() here — _toast wraps this in _safe_update,
            # whose wrapper pushes the page once (this inner push doubled
            # every toast to two full-page sends).

        try:
            self._safe_update(_show)
        except Exception:
            logger.info("Toast failed", exc_info=True)

    def _on_keyboard_event(self, e):
        """Global shortcuts: palette, run/stop, views, export, save."""
        try:
            key = getattr(e, "key", "") or ""
            ctrl = bool(getattr(e, "ctrl", False))
            k = key.upper() if len(key) == 1 else key
            if ctrl and k == "K":
                self._open_palette()
            elif ctrl and k == "R":
                self._on_action_click()
            elif ctrl and k == "1":
                self._show_view("dashboard")
            elif ctrl and k == "2":
                self._show_settings()
            elif ctrl and k == "E":
                self._export_html()
            elif ctrl and k == "S":
                self._maybe_save_settings()
            elif not ctrl and k == "/":
                self._focus_search()
            elif not ctrl and k in ("ArrowDown", "Down"):
                self._kb_move(1)
            elif not ctrl and k in ("ArrowUp", "Up"):
                self._kb_move(-1)
            elif not ctrl and k == "Enter":
                self._kb_open()
            elif not ctrl and k == "Escape":
                self._kb_escape()
        except Exception:
            logger.info("Keyboard shortcut failed", exc_info=True)

    def _kb_escape(self):
        """Esc: leave the detail panel, else leave settings."""
        try:
            if getattr(self, "_detail_ticker", None) is not None:
                # _back_to_results pushes the page itself.
                self._back_to_results()
            elif self.active_view == "settings":
                self._show_view("dashboard")
        except Exception:
            logger.info("Escape handling failed", exc_info=True)

    def _focus_search(self):
        if self.active_view == "dashboard":
            try:
                field = self.search_entry
            except AttributeError:
                return
            try:
                # TextField.focus() is a coroutine in this Flet version — it
                # only takes effect when awaited on the page's running loop.
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            try:
                if loop is not None:
                    loop.create_task(field.focus())
                else:
                    # No running loop (unit tests): close immediately so the
                    # un-awaited coroutine never warns at GC time.
                    field.focus().close()
                self.page.update()
            except Exception:
                logger.info("_focus_search failed", exc_info=True)

    def _maybe_save_settings(self):
        if self.active_view == "settings":
            self._save_settings_page()

    def _palette_actions(self):
        run_label = "Stop scan" if self.scanning else "Run scan"
        return [
            PaletteAction(
                "go-dashboard",
                "Go to Dashboard",
                "Ctrl+1",
                lambda: self._show_view("dashboard"),
            ),
            PaletteAction(
                "go-settings", "Go to Settings", "Ctrl+2", lambda: self._show_settings()
            ),
            PaletteAction(
                "run-stop", run_label, "Ctrl+R", lambda: self._on_action_click()
            ),
            PaletteAction(
                "export-html",
                "Export HTML report",
                "Ctrl+E",
                lambda: self._export_html(),
            ),
            PaletteAction("export-csv", "Export CSV", "", lambda: self._export_csv()),
            PaletteAction(
                "clear-results", "Clear results", "", lambda: self._clear_results()
            ),
            PaletteAction(
                "open-scan-history",
                "Open scan from history…",
                "",
                lambda: self._open_scan_history(),
            ),
            PaletteAction(
                "run-backtest", "Run backtest", "", lambda: self._run_backtest()
            ),
            PaletteAction(
                "toggle-right-panel",
                "Toggle profile panel",
                "",
                lambda: self._toggle_right_panel(),
            ),
            PaletteAction(
                "focus-search", "Focus ticker search", "/", lambda: self._focus_search()
            ),
            PaletteAction(
                "save-settings",
                "Save settings",
                "Ctrl+S",
                lambda: self._maybe_save_settings(),
            ),
            PaletteAction(
                "audit", "Check stale members", "", lambda: self._run_stale_audit()
            ),
            PaletteAction(
                "import-watchlist",
                "Import watchlist from file",
                "",
                lambda: self._open_watchlist_picker(),
            ),
            PaletteAction(
                "prune-cache",
                "Prune price cache",
                "",
                lambda: self._prune_price_cache(),
            ),
        ]

    def _open_palette(self):
        """Command palette dialog (Ctrl+K): fuzzy filter, Enter runs."""
        c = self.theme_colors
        query = ft.TextField(
            hint_text="Type a command…  (Enter runs the top hit)",
            autofocus=True,
            text_size=13,
            bgcolor=c["card"],
            color=c["text"],
            border_color=c["border"],
            border_width=1,
            border_radius=RADIUS_MD,
            focused_border_color=c["green"],
            focused_border_width=1,
        )
        results_col = ft.Column(spacing=2, scroll=ft.ScrollMode.AUTO, height=300)
        state = {"actions": self._palette_actions(), "shown": []}

        def _render_list():
            q = query.value or ""
            state["shown"] = filter_actions(q, state["actions"])[:8]
            results_col.controls.clear()
            for a in state["shown"]:
                results_col.controls.append(
                    ft.TextButton(
                        content=ft.Row(
                            [
                                ft.Text(a.title, size=13, color=c["text"]),
                                ft.Container(expand=True),
                                ft.Text(a.hint, size=11, color=c["text_dim"]),
                            ]
                        ),
                        on_click=lambda _, act=a: self._run_palette_action(act),
                    )
                )
            self.page.update()

        query.on_change = lambda _: _render_list()
        query.on_submit = lambda _: (
            self._run_palette_action(state["shown"][0]) if state["shown"] else None
        )
        self._palette_query = query
        self._palette_state = state
        _render_list()
        self._palette_dlg = ft.AlertDialog(
            modal=True,
            bgcolor=c["panel_bg"],
            shape=ft.RoundedRectangleBorder(radius=RADIUS_LG),
            title=ft.Text("Command palette  (Ctrl+K)", size=14),
            content=ft.Container(
                content=ft.Column([query, results_col], spacing=8), width=480
            ),
        )
        self._safe_update(lambda: self.page.show_dialog(self._palette_dlg))

    def _run_palette_action(self, act):
        try:
            if getattr(self, "_palette_dlg", None) is not None:
                self._safe_update(lambda: self.page.pop_dialog())
                self._palette_dlg = None
        except Exception:
            logger.info("Palette close failed", exc_info=True)
        try:
            if act is not None and act.run is not None:
                act.run()
        except Exception:
            logger.info("Palette action failed", exc_info=True)

    def _on_search_change(self, _e):
        self.filter_text = (
            self.search_entry.value.strip().upper() if self.search_entry.value else ""
        )
        if self.all_results:
            # Debounced: a keystroke burst must refilter+rebuild once, not per key.
            self._debounce(
                "search", 0.2, lambda: self._display_results(self.all_results)
            )

    def _rating_filter(self) -> str:
        return str(self.rating_filter_dd.value or "ALL").upper()

    def _score_threshold(self) -> float:
        """Live grid threshold from the slider (settings as fallback)."""
        try:
            return float(self.threshold_slider.value)
        except (AttributeError, TypeError, ValueError):
            try:
                return float(self.settings.get("min_score", 50))
            except (TypeError, ValueError):
                return 50.0

    def _checkbox_on(self, name: str) -> bool:
        cb = getattr(self, name, None)
        return bool(cb is not None and cb.value)

    def _is_filter_active(self) -> bool:
        return (
            bool(self.filter_text)
            or self._rating_filter() != "ALL"
            or self._score_threshold() > 0
            or self._checkbox_on("inst_filter_cb")
            or self._checkbox_on("price_filter_cb")
            or self._checkbox_on("fund_filter_cb")
        )

    def _visible_results(self) -> list:
        """Results after search/rating/threshold filters.

        Returns a snapshot copy under ``_results_lock`` so concurrent
        streaming mutations cannot corrupt the caller's iteration.
        """
        with self._results_lock:
            if self._is_filter_active():
                return list(self.filtered_results)
            return list(self.all_results)

    def _row_matches_filters(self, r: dict) -> bool:
        if self.filter_text and self.filter_text not in r.get("ticker", "").upper():
            return False
        rating = self._rating_filter()
        if rating != "ALL":
            combined = (r.get("combined_rating") or "POOR").upper()
            if combined != rating:
                return False
        if score_of(r) < self._score_threshold():
            return False
        if self._checkbox_on("inst_filter_cb") and not has_fii_dii(r):
            return False
        if (
            self._checkbox_on("price_filter_cb")
            and 0 < (self._extra_num(r.get("close")) or 0) < 100
        ):  # unpriced rows (0/None) stay visible
            return False
        if self._checkbox_on("fund_filter_cb") and not (
            self._extra_num(r.get("fundamentals")) or 0
        ):
            return False  # 0/None shows as "0" in the grid — hide it
        return True

    def _on_rating_change(self, _e):
        if self.all_results:
            self._display_results(self.all_results)
        self._save_ui_prefs()

    def _on_universe_change(self, _e):
        choice = self.universe_dd.value or "NIFTY 50"
        try:
            base = len(UNIVERSES.get(choice, []))
        except Exception:
            logger.info("Universe length lookup failed", exc_info=True)
            base = 0
        if base == 0:
            from ..shared.constants import UNIVERSE_DEFAULT_SIZE, UNIVERSE_SIZES

            base = UNIVERSE_SIZES.get(
                next(
                    (k for k in UNIVERSE_SIZES if k in choice),
                    "",
                ),
                UNIVERSE_DEFAULT_SIZE,
            )
        label = f"{base} stocks"
        if base > 1000:
            label += " (~5-10 min)"
        if "FULL MARKET" in choice:
            label += " — full 5,900"
        self.universe_count_label.value = label + " ..."
        is_watchlist = "WATCHLIST" in choice.upper()
        self.universe_count_container.visible = is_watchlist
        self.watchlist_import_container.visible = is_watchlist
        self.page.update()

        def _bg():
            try:
                from ..shared.universes import get_universe

                live = get_universe(choice)
                cnt = len(live)
                if cnt and cnt != base:
                    lbl = f"{cnt} stocks"
                    if cnt > 1000:
                        lbl += " (~5-10 min)"
                    if "FULL MARKET" in choice:
                        lbl += " — full 5,900"
                    self._safe_update(
                        lambda l=lbl: setattr(self.universe_count_label, "value", l)
                    )
            except Exception:
                logger.info("Universe count background update failed", exc_info=True)

        threading.Thread(target=_bg, daemon=True).start()

    def _open_watchlist_picker(self):
        """Open the file picker imperatively (palette path)."""
        picker = getattr(self, "file_picker", None)
        if picker is None:
            return
        try:
            picker.pick_files(
                dialog_title="Import watchlist",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["csv", "txt"],
            )
        except Exception:
            logger.info("Watchlist picker failed", exc_info=True)

    def _on_watchlist_picked(self, e):
        """Import a CSV/TXT watchlist file as a scannable universe."""
        try:
            files = list(getattr(e, "files", None) or [])
            if not files:
                return
            path = getattr(files[0], "path", None)
            if not path:
                self._toast("Picked file has no local path", "error")
                return
            if os.path.getsize(path) > 1_000_000:  # 1 MB limit
                self._toast(
                    "File too large (>1 MB) — expected a small watchlist", "error"
                )
                return
            with open(path, encoding="utf-8", errors="replace") as fh:
                tickers = parse_watchlist_text(fh.read())
            if not tickers:
                self._toast("No tickers found in that file", "error")
                return
            name = f"WATCHLIST ({len(tickers)})"
            UNIVERSES[name] = tickers
            # Replace any previous watchlist entry so the dropdown never
            # accumulates stale ones.
            self.universe_dd.options = [
                o
                for o in self.universe_dd.options
                if not str(getattr(o, "key", o) or "").startswith("WATCHLIST (")
            ]
            self.universe_dd.options.append(ft.dropdown.Option(name))
            self.universe_dd.value = name
            self._on_universe_change(None)
            self._toast(f"Imported {len(tickers)} tickers — ready to scan", "success")
            self.page.update()
        except Exception as ex:
            logger.info("Watchlist import failed", exc_info=True)
            try:
                self._toast(f"Watchlist import failed: {ex}", "error")
            except Exception:
                logger.info("Toast display failed after watchlist error", exc_info=True)

    def _on_threshold_change(self, _e):
        val = self.threshold_slider.value
        self.threshold_label.value = f"{int(val)}+"
        # Label repaint now (cheap); the full refilter+rebuild is debounced so
        # a slider drag doesn't sort/render the grid on every tick. The old
        # code also called page.update() after _display_results — one update
        # per event is plenty.
        self.page.update()
        if self.all_results:
            self._debounce(
                "threshold", 0.15, lambda: self._display_results(self.all_results)
            )

    def _load_settings_to_ui(self):
        try:
            min_score = float(self.settings.get("min_score", 50))
        except (ValueError, TypeError):
            min_score = 50.0
        self.threshold_slider.value = min_score
        self.threshold_label.value = f"{int(min_score)}+"
        saved_universe = self.settings.get("universe", "NIFTY 50")
        self.universe_dd.value = (
            saved_universe if saved_universe in UNIVERSES else "NIFTY 50"
        )
        period_map = {"6mo": "6 Months", "1y": "1 Year", "2y": "2 Years"}
        self.period_dd.value = period_map.get(
            self.settings.get("data_period", "1y"), "1 Year"
        )
        tf_map = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
        self.timeframe_dd.value = tf_map.get(
            self.settings.get("timeframe", "D"), "Daily"
        )
        self.trend_filter_dd.value = self.settings.get("trend_filter", "All")
        strat_key = str(self.settings.get("strategy", "daily_screener"))
        self.strategy_dd.value = STRATEGY_LABELS.get(strat_key, "Daily Screener")

    def _collect_settings(self) -> dict:
        s = dict(self.settings)
        # min_score: tolerate empty/invalid slider values; fall back to default
        try:
            s["min_score"] = float(self.threshold_slider.value)
        except (ValueError, TypeError):
            s["min_score"] = float(self.settings.get("min_score", 50.0))
        # period dropdown: map display name → engine key; default to "1y"
        period_map: dict[str, str] = {
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
        }
        s["data_period"] = period_map.get(self.period_dd.value or "1 Year", "1y")
        # timeframe dropdown: map display name → engine key; default to "D"
        tf_map: dict[str, str] = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        s["timeframe"] = tf_map.get(self.timeframe_dd.value or "Daily", "D")
        # trend_filter dropdown
        s["trend_filter"] = self.trend_filter_dd.value or "All"
        # strategy dropdown: map display label → engine key
        strat_label = self.strategy_dd.value or "Daily Screener"
        rev = {v: k for k, v in STRATEGY_LABELS.items()}
        s["strategy"] = rev.get(strat_label, "daily_screener")
        # universe dropdown
        s["universe"] = self.universe_dd.value or "NIFTY 50"
        return s

    def _safe_update(self, fn: Callable[[], None]) -> None:
        """Run a UI mutation, then push it to the page — on Flet's UI thread.

        ``fn`` mutates controls (or reads state to build them), then
        ``page.update()`` flushes the changes to the client.  Flet only
        patches the client reliably when controls are touched on the thread
        that runs the page's asyncio event loop; mutating from the scanner
        worker thread races with that loop and page updates are silently
        dropped — the results grid then stays on its placeholder until the
        next UI event (e.g. clicking the window) forces a redraw.

        This method may be called from any thread; the work is marshalled
        onto the page's event loop when needed (see ``_run_on_ui_thread``).
        Exceptions inside ``fn`` or ``page.update()`` are logged/ignored,
        never raised.
        """

        def _apply():
            t0 = time.monotonic()
            with self._ui_lock:
                try:
                    fn()
                except Exception:  # pragma: no cover
                    logger.info(
                        "UI update callback failed: fn=%s",
                        getattr(fn, "__name__", fn),
                        exc_info=True,
                    )
            fn_ms = (time.monotonic() - t0) * 1000.0
            # page.update() MUST run outside _ui_lock — a slow send
            # (large control tree) would hold the lock and block all
            # subsequent _safe_update calls (including _scan_complete),
            # freezing the UI at "Finalizing scan…".
            t1 = time.monotonic()
            try:
                self.page.update()
            except Exception:  # pragma: no cover
                logger.info("page.update() failed in _safe_update", exc_info=True)
            upd_ms = (time.monotonic() - t1) * 1000.0
            if fn_ms > 200.0 or upd_ms > 200.0:
                # UI stall evidence: a slow callback or a slow full-page
                # send delays every queued click (sort, hover, resize).
                logger.info(
                    "SLOW _safe_update: fn=%s fn=%.0fms update=%.0fms",
                    getattr(fn, "__name__", fn),
                    fn_ms,
                    upd_ms,
                )

        self._run_on_ui_thread(_apply)

    def _run_on_ui_thread(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the thread that owns the page's asyncio event loop.

        When called from a worker thread the callable is scheduled with
        ``loop.call_soon_threadsafe`` (Flet must not have controls mutated
        from arbitrary threads); on the loop thread itself, or when no live
        page loop exists (e.g. unit tests with fake pages), it runs inline.
        """
        loop = self._page_event_loop()
        if loop is not None:
            thread_id = getattr(loop, "_thread_id", None)
            if thread_id is None or thread_id != threading.get_ident():
                try:
                    loop.call_soon_threadsafe(self._with_page_context(fn))
                    return
                except Exception:  # pragma: no cover — loop closed mid-call
                    logger.info(
                        "Event loop closed mid-call in _run_on_ui_thread", exc_info=True
                    )
        fn()

    def _with_page_context(self, fn: Callable[[], None]) -> Callable[[], None]:
        """Wrap ``fn`` so Flet client actions see the page context.

        Flet 1.0 resolves per-page services (clipboard, file picker, …)
        through a context variable that ``call_soon_threadsafe`` does not
        propagate; without this, constructing e.g. ``CopyToClipboard`` in a
        marshalled update raises ``RuntimeError``. Falls back to the bare
        callable if the (private) context API is unavailable.
        """
        try:
            from flet.controls.context import _context_page
        except Exception:
            logger.info("flet.controls.context import failed", exc_info=True)
            return fn

        def _run():
            token = _context_page.set(self.page)
            try:
                fn()
            finally:
                _context_page.reset(token)

        return _run

    def _page_event_loop(self):
        """Return the page's running asyncio loop, or None when unavailable."""
        try:
            conn = getattr(getattr(self.page, "session", None), "connection", None)
            loop = getattr(conn, "loop", None)
        except Exception:  # pragma: no cover
            logger.info("Failed to get page event loop", exc_info=True)
            return None
        if loop is None or not loop.is_running():
            return None
        return loop

    def _debounce(self, key: str, delay: float, fn: Callable[[], None]) -> None:
        """Run ``fn`` once on the page loop after ``delay`` seconds.

        Rapid events (search keystrokes, slider ticks) coalesce: a new call
        under the same ``key`` cancels the pending one. Without a live page
        loop (unit tests) ``fn`` runs inline so callers stay synchronous.
        """
        timers = self.__dict__.setdefault("_debounce_timers", {})
        loop = self._page_event_loop()
        if loop is None:
            fn()
            return

        def _arm():
            old = timers.pop(key, None)
            if old is not None:
                old.cancel()
            timers[key] = loop.call_later(delay, lambda: (timers.pop(key, None), fn()))

        try:
            loop.call_soon_threadsafe(_arm)
        except Exception:  # pragma: no cover — loop closed mid-call
            fn()

    def _throttle(self, key: str, interval: float, fn: Callable[[], None]) -> None:
        """Leading-edge throttle with one trailing flush of the newest ``fn``.

        The first call (or any call after ``interval``) runs ``fn`` now;
        calls inside the window coalesce into a single trailing run of the
        newest ``fn`` when the window expires. Without a live page loop
        (unit tests) ``fn`` runs inline so callers stay synchronous.
        Rate-limits per-ticker engine pushes (progress, log lines) that
        would otherwise queue hundreds of full ``page.update()`` sends.
        """
        st = self.__dict__.setdefault("_throttle_state", {})
        loop = self._page_event_loop()
        if loop is None:
            fn()
            return

        def _decide():
            now = time.monotonic()
            rec = st.get(key)
            if rec is None or now - rec[0] >= interval:
                st[key] = [now, None, None]
                fn()
                return
            rec[1] = fn  # newest pending wins
            if rec[2] is None:

                def _fire():
                    rec[2] = None
                    pending, rec[1] = rec[1], None
                    rec[0] = time.monotonic()
                    if pending is not None:
                        pending()

                rec[2] = loop.call_later(interval, _fire)

        try:
            loop.call_soon_threadsafe(_decide)
        except Exception:  # pragma: no cover — loop closed mid-call
            fn()

    # ── Results rendering ───────────────────────────────────────────────

    # ── Scan history ────────────────────────────────────────────────────

    def _open_scan_history(self, e=None):
        """Reopen an older scan from db history (palette: "Open scan…")."""
        from ..backend.settings_store import list_scans
        from .ui_kit import themed_dropdown

        scans = list_scans()
        if not scans:
            self._toast("No saved scans yet — run a scan first", "error")
            return
        c = self.theme_colors
        opts = [
            ft.dropdown.Option(
                key=str(s["id"]),
                text=f"{s['created_at'].replace('T', ' ')} — {s['rows']} rows",
            )
            for s in scans
        ]
        dd = themed_dropdown(opts, opts[0].key, c, width=360)
        dlg = ft.AlertDialog(
            modal=True,
            bgcolor=c["panel_bg"],
            shape=ft.RoundedRectangleBorder(radius=RADIUS_LG),
            actions_alignment=ft.MainAxisAlignment.END,
            title=ft.Text("Open scan from history"),
            content=dd,
            actions=[
                ft.TextButton(
                    content=ft.Text("Cancel"),
                    on_click=lambda _: self._close_dialog(dlg),
                ),
                ft.Button(
                    content=ft.Text("Open"),
                    bgcolor=c["green"],
                    color=c["on_accent"],
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=RADIUS_MD)
                    ),
                    on_click=lambda _: self._open_chosen_scan(dlg, dd.value),
                ),
            ],
        )
        self._safe_update(lambda: self.page.show_dialog(dlg))

    def _open_chosen_scan(self, dlg, scan_id):
        """Load the dialog's selected scan into the results grid."""
        self._close_dialog(dlg)
        if not scan_id:
            return
        from ..backend.settings_store import load_results

        rows = load_results(scan_id=int(scan_id))
        if not rows:
            self._toast("That scan has no rows", "error")
            return
        self._show_view("dashboard")
        self._restore_saved_results(rows)

    # ── Backtest ────────────────────────────────────────────────────────

    def _run_backtest(self, e=None):
        """Run the backtest CLI in a subprocess and open its HTML report.

        Subprocess, not in-process: ``backtest.main()`` parses ``sys.argv``.
        No cancel in v1 — the daemon thread just runs to completion.
        """
        lock = getattr(self, "_state_lock", None)
        if lock is not None:
            with lock:
                if getattr(self, "backtest_running", False):
                    return
                self.backtest_running = True
        elif getattr(self, "backtest_running", False):
            return
        else:
            self.backtest_running = True

        html_path = os.path.join(REPORTS_DIR, "backtest_report.html")
        csv_path = os.path.join(REPORTS_DIR, "backtest_trades.csv")
        os.makedirs(REPORTS_DIR, exist_ok=True)
        # Existence must mean *this* run's report: 0-trade runs write none.
        try:
            os.remove(html_path)
        except OSError:
            pass

        def _bg():
            kind, msg = "error", "Backtest failed"
            try:
                self._engine_log("Backtest started (nifty50, app settings) …")
                # Unbuffered UTF-8 so streamed lines arrive promptly and ₹ survives.
                env = {
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUNBUFFERED": "1",
                }
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "scanner.backend.backtest",
                        "--html",
                        html_path,
                        "--csv",
                        csv_path,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=_PROJECT_ROOT,
                    env=env,
                )
                for line in proc.stdout:
                    self._engine_log(line.rstrip())
                rc = proc.wait()
                if rc != 0:
                    msg = f"Backtest failed (exit {rc}) — see log"
                elif not os.path.exists(html_path):
                    kind = "info"
                    msg = "Backtest finished with no trades — no report written"
                else:
                    kind = "info"
                    msg = "Backtest done — report opened in browser"
                    webbrowser.open(f"file://{os.path.abspath(html_path)}")
            except Exception as ex:
                msg = f"Backtest failed: {ex}"
            finally:
                lock = getattr(self, "_state_lock", None)
                if lock is not None:
                    with lock:
                        self.backtest_running = False
                else:
                    self.backtest_running = False

                def _finish():
                    self._log(msg)
                    self._toast(msg, kind)

                self._safe_update(_finish)

        threading.Thread(target=_bg, daemon=True).start()

    def _close_dialog(self, dlg):
        self._safe_update(lambda: self.page.pop_dialog())

    # ── Utilities ───────────────────────────────────────────────────────

    def _persist_results_bg(self, rows: list) -> None:
        """save_results is a heavy db+file write — never on the UI thread."""

        def _bg():
            try:
                from ..backend.settings_store import save_results

                save_results(rows)
            except Exception:
                logger.info("Failed to persist results", exc_info=True)

        t = threading.Thread(target=_bg, daemon=True)
        # Joinable handle so tests (and post-mortems) can wait for the write.
        self._persist_thread = t
        t.start()

    def _clear_results(self, e=None):
        with self._results_lock:
            self.results = []
            self.all_results = []
            self.filtered_results = []
        # Clear the row control pool
        if hasattr(self, "_row_pool"):
            self._row_pool.clear()
        if hasattr(self, "_row_cells"):
            self._row_cells.clear()
        for attr in (
            "table_column",
            "chart_holder",
            "empty_label",
            "result_count_label",
            "progress_bar",
            "progress_label",
            "status_label",
            "html_btn",
            "csv_btn",
            "clear_btn",
        ):
            if getattr(self, attr, None) is None:
                return
        self.table_column.controls.clear()
        self.chart_holder.visible = False
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
        self._persist_results_bg([])
        self.page.update()

    def _clear_log(self, e=None):
        try:
            if getattr(self, "log_column", None) is not None:
                self.log_column.controls.clear()
                self.page.update()
        except Exception:
            logger.info("_clear_log failed", exc_info=True)

    def _reset_filters(self, e=None):
        self.filter_text = ""
        if getattr(self, "search_entry", None) is not None:
            self.search_entry.value = ""
        if getattr(self, "rating_filter_dd", None) is not None:
            self.rating_filter_dd.value = "All"
        if self.all_results:
            # _display_results pushes the page itself — a second update
            # here doubled every reset to two full-page sends.
            self._display_results(self.all_results)
        else:
            self._render_current_page()
            self.page.update()
        self._scroll_to_top()

    def _make_log_line(self, text, c):
        # "[HH:MM:SS] message" → dim timestamp + severity-tinted message
        ts, msg = "", text
        if text.startswith("[") and "] " in text[:12]:
            ts, msg = text.split("] ", 1)
            ts += "] "
        upper = msg.upper()
        if "ERROR" in upper or "FAIL" in upper:
            color = c["red"]
        elif "WARN" in upper:
            color = c["orange"]
        else:
            color = c["text_dim"]
        base = {"size": 10, "font_family": "Consolas"}
        return ft.Text(
            spans=[
                ft.TextSpan(ts, ft.TextStyle(**base, color=c["text_faint"])),
                ft.TextSpan(msg, ft.TextStyle(**base, color=color)),
            ],
            selectable=True,
            font_family="Consolas",
        )

    def _log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        # File append goes to a single writer thread — when _log is called on
        # the UI thread (engine messages route through _safe_update) a sync
        # write would stall the loop on every progress tick.
        global _log_writer_running
        with _log_lock:
            if not _log_writer_running:
                _log_writer_running = True
                threading.Thread(target=_log_writer, daemon=True).start()
            _log_q.put(line)

        def _append_to_panel():
            try:
                col = getattr(self, "log_column", None)
                if col is not None:
                    col.controls.append(
                        self._make_log_line(line.rstrip("\n"), self.theme_colors)
                    )
                    del col.controls[:-LOG_MAX_LINES]
            except Exception:
                logger.info("Failed to append to UI log panel", exc_info=True)

        # Control mutation must happen on the page's event-loop thread.
        self._run_on_ui_thread(_append_to_panel)


def main(page: ft.Page):
    # The instance stays alive via the bound handlers _build_ui registers on
    # page controls — no local reference needed.
    ScannerApp(page)


if __name__ == "__main__":
    ft.run(main)
