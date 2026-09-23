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
import threading
import webbrowser
from collections.abc import Callable
from datetime import datetime

import flet as ft

from ..shared.constants import LOG_MAX_LINES, LOG_ROTATE_HOURS
from ..shared.trace import setup_trace

try:
    setup_trace()
except Exception:
    logging.getLogger(__name__).debug("Trace setup failed", exc_info=True)

logger = logging.getLogger(__name__)

from ..backend.settings_store import (  # noqa: F401 — re-exported for tests
    DEFAULT_SETTINGS,
    load_settings,
    save_settings,
)
from ..shared.themes import THEMES
from ..shared.universes import UNIVERSES
from .views_layout import LayoutViewMixin
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


class ScannerApp(
    ScanOrchestrationMixin, LayoutViewMixin, ResultsViewMixin, SettingsViewMixin
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
        self._load_settings_to_ui()
        self._load_ui_prefs()
        self._restore_saved_results()
        self._refresh_neg_cache_ui()
        self._refresh_enrich_cache_ui()
        self._refresh_price_cache_ui()
        self._log("Scanner ready — pick a universe and hit RUN SCAN")
        self._rotate_log()

        def _warm_symbols():
            try:
                from ..api.symbol_fetcher import _load_disk_cache
                from ..shared.universes import get_universe

                _load_disk_cache()
                get_universe("FULL MARKET (NSE+BSE ~5,900)")
            except Exception:
                logger.info("Symbol warm-up failed", exc_info=True)

        threading.Thread(target=_warm_symbols, daemon=True).start()
        threading.Thread(target=self._warm_market, daemon=True).start()

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

    def _restore_saved_results(self):
        """Repopulate the grid from last_results.json after a relaunch."""
        try:
            from ..backend.settings_store import load_results

            rows = load_results()
            if not rows:
                return
            with self._results_lock:
                self.results = rows
                self.all_results = list(rows)
                self.filtered_results = [
                    r for r in rows if self._row_matches_filters(r)
                ]
            self.current_page = 0
            self._render_current_page()
            self.html_btn.disabled = False
            self.csv_btn.disabled = False
            self.clear_btn.disabled = False
            self._log(f"Restored {len(rows)} results from last scan")
            self.page.update()
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
        # Container defaults to visible, which would desync from the flag.
        self.sidebar.visible = not getattr(self, "_sidebar_collapsed", False)
        self.main_row = ft.Row(
            controls=[
                self._build_rail(),
                self.sidebar,
                self._build_main_area(),
                self._build_right_panel(),
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
        saved_log = self._get_log_lines()
        # Overlay only — do not touch dashboard_view.visible/opacity/layout.
        settings = getattr(self, "settings_view", None)
        if settings is not None:
            settings.visible = False
        self.page.update()
        self._load_settings_to_ui()
        self._set_log_lines(saved_log)

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
            engine.set_progress_callback(
                lambda p, m: self._safe_update(lambda: self._set_progress(p, m))
            )
            engine.set_log_callback(
                lambda msg: self._safe_update(lambda: self._log(msg))
            )

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
        if name == "dashboard":
            # Overlay off + independent hero/grid reload; dashboard never re-laid-out.
            self._restore_main_area()
        else:
            self.page.update()

    def _show_settings(self, e=None):
        self.active_view = "settings"
        for vname, pill in self._rail_pills.items():
            pill.opacity = 1.0 if vname == "settings" else 0.0
        # Overlay on — dashboard stays visible and laid out underneath.
        self.settings_view.visible = True
        self._fade_view_in(self.settings_view)

    # ── Modern chrome: sidebar collapse, shortcuts, palette, toasts ──

    def _toggle_sidebar(self, e=None):
        """Collapse/expand the sidebar in place."""
        self._sidebar_collapsed = not getattr(self, "_sidebar_collapsed", False)
        box = getattr(self, "sidebar", None)
        if box is not None:
            box.visible = not self._sidebar_collapsed
        try:
            self.page.update()
        except Exception:
            logger.info("Sidebar toggle page.update failed", exc_info=True)

    def _toast(self, msg, kind="info"):
        """Non-modal feedback via a SnackBar dialog (thread-safe).

        Flet 1.0 removed ``page.snack_bar``/``page.open`` — overlays now go
        through ``page.show_dialog`` (the same call also presents AlertDialog).
        """

        def _show():
            c = self.theme_colors
            if kind == "success":
                bg, fg = c["green"], c["on_accent"]
            elif kind == "error":
                bg, fg = c["red"], "#ffffff"
            else:
                bg, fg = c["card2"], c["text"]
            bar = ft.SnackBar(
                content=ft.Text(str(msg), color=fg, size=13),
                bgcolor=bg,
                open=True,
            )
            self.page.show_dialog(bar)
            self.page.update()

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
        except Exception:
            logger.info("Keyboard shortcut failed", exc_info=True)

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
        from .ui_kit import PaletteAction

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
                "toggle-sidebar", "Toggle sidebar", "", lambda: self._toggle_sidebar()
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
        from .ui_kit import filter_actions

        c = self.theme_colors
        query = ft.TextField(
            hint_text="Type a command…  (Enter runs the top hit)",
            autofocus=True,
            text_size=13,
            bgcolor=c["card"],
            color=c["text"],
            border_color=c["border"],
            border_width=1,
            border_radius=10,
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
        self._palette_results = results_col
        self._palette_state = state
        _render_list()
        self._palette_dlg = ft.AlertDialog(
            modal=True,
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
            self._display_results(self.all_results)

    def _rating_filter(self) -> str:
        return str(self.rating_filter_dd.value or "ALL").upper()

    def _is_filter_active(self) -> bool:
        return bool(self.filter_text) or self._rating_filter() != "ALL"

    def _visible_results(self) -> list:
        """Results after search/rating filters.

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
        from .ui_kit import parse_watchlist_text

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
        self.page.update()

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
            with self._ui_lock:
                try:
                    fn()
                except Exception:  # pragma: no cover
                    logger.info(
                        "UI update callback failed: fn=%s",
                        getattr(fn, "__name__", fn),
                        exc_info=True,
                    )
            # page.update() MUST run outside _ui_lock — a slow send
            # (large control tree) would hold the lock and block all
            # subsequent _safe_update calls (including _scan_complete),
            # freezing the UI at "Finalizing scan…".
            try:
                self.page.update()
            except Exception:  # pragma: no cover
                logger.info("page.update() failed in _safe_update", exc_info=True)

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

    # ── Results rendering ───────────────────────────────────────────────

    # ── Cache management ───────────────────────────────────────────────

    def _refresh_neg_cache_ui(self):
        if not hasattr(self, "cache_status_lbl"):
            return
        try:
            from ..api import cache_manager

            n = len(cache_manager.negative_load())
            ttl_h = cache_manager.negative_ttl_hours()
        except Exception:
            logger.info("Failed to load negative cache info", exc_info=True)
            n, ttl_h = 0, 24
        self.cache_status_lbl.value = (
            f"Dead-symbol cache: {n} (auto-resets ~{ttl_h}h)"
            if n
            else "Dead-symbol cache: empty"
        )
        self.cache_clear_btn.visible = bool(n)

    def _clear_negative_cache(self, e=None):
        try:
            from ..api import cache_manager

            cache_manager.negative_update(
                clears=list(cache_manager.negative_load().keys())
            )
            self._log(
                "Cleared dead-symbol cache — fallback will re-attempt all symbols"
            )
            self._toast("Dead-symbol cache cleared", "success")
        except Exception as ex:
            self._log(f"Could not clear dead-symbol cache: {ex}")
            self._toast(f"Could not clear dead-symbol cache: {ex}", "error")
        self._refresh_neg_cache_ui()
        self.page.update()

    def _refresh_enrich_cache_ui(self):
        if not hasattr(self, "enrich_cache_status_lbl"):
            return
        try:
            from ..api import cache_manager

            n = cache_manager.enrichment_size()
            ttl_h = cache_manager.ENRICHMENT_CACHE_TTL_HOURS
        except Exception:
            logger.info("Failed to load enrichment cache info", exc_info=True)
            n, ttl_h = 0, 24
        self.enrich_cache_status_lbl.value = (
            f"Enrichment cache: {n} (auto-resets ~{ttl_h}h)"
            if n
            else "Enrichment cache: empty"
        )
        self.enrich_cache_clear_btn.visible = bool(n)

    def _clear_enrichment_cache(self, e=None):
        try:
            from ..api import cache_manager

            cache_manager.enrichment_clear()
            self._log("Cleared enrichment cache — next scan will re-fetch phase-2 data")
            self._toast("Enrichment cache cleared", "success")
        except Exception as ex:
            self._log(f"Could not clear enrichment cache: {ex}")
            self._toast(f"Could not clear enrichment cache: {ex}", "error")
        self._refresh_enrich_cache_ui()
        self.page.update()

    def _refresh_price_cache_ui(self):
        if not hasattr(self, "price_cache_status_lbl"):
            return
        try:
            from ..api import cache_manager

            h = cache_manager.cache_health()
            n, stale = h["price_entries"], h["stale_entries"]
        except Exception:
            logger.info("Failed to load price cache health", exc_info=True)
            n, stale = 0, 0
        if not n:
            self.price_cache_status_lbl.value = "Price cache: empty"
        elif stale:
            self.price_cache_status_lbl.value = (
                f"Price cache: {n} ({stale} stale — auto-prunes on next scan)"
            )
        else:
            self.price_cache_status_lbl.value = f"Price cache: {n} (clean)"
        self.price_cache_prune_btn.visible = bool(stale)

    def _prune_price_cache(self, e=None):
        try:
            from ..api import cache_manager

            removed = cache_manager.prune_stale_cache(force=True)
            self._log(
                f"Pruned {removed} stale price-cache entrie(s) (previous trading days)"
                if removed
                else "Price cache clean — nothing to prune"
            )
            self._toast(f"Pruned {removed} stale price-cache entries", "success")
        except Exception as ex:
            self._log(f"Could not prune price cache: {ex}")
            self._toast(f"Could not prune price cache: {ex}", "error")
        self._refresh_price_cache_ui()
        self.page.update()

    def _audit_report(self, universe: str):
        """Sync core of the stale-member audit button (threaded in the GUI)."""
        from ..backend.audit_stale_members import audit_stale_members, format_report
        from ..shared.universes import UNIVERSES

        tickers = list(UNIVERSES.get(universe, []))
        res = audit_stale_members(tickers)
        return res, format_report(res)

    def _audit_fixable(self, res: dict) -> bool:
        """True when the audit result has anything apply_fixes could change."""
        return bool(
            res.get("rename_suggestions")
            or res.get("unannotated_stale")
            or res.get("annotated_fresh")
        )

    def _audit_verdict(self, res: dict) -> str:
        """One-line audit summary shown in the log + status label."""
        verdict = (
            f"{len(res['unannotated_stale'])} new candidate(s), "
            f"{len(res['annotated_stale'])} still annotated, "
            f"{len(res['missing'])} missing"
        )
        renames = res.get("rename_suggestions", {})
        if renames:
            verdict += f", {len(renames)} rename(s) suggested"
        if self._audit_fixable(res):
            verdict += " — fixes ready"
        return verdict

    def _run_stale_audit(self, e=None):
        lock = getattr(self, "_state_lock", None)
        if lock is not None:
            with lock:
                if getattr(self, "stale_audit_running", False):
                    return
                self.stale_audit_running = True
        elif getattr(self, "stale_audit_running", False):
            return
        else:
            self.stale_audit_running = True
        universe = None
        dd = getattr(self, "universe_dd", None)
        if dd is not None and getattr(dd, "value", None):
            universe = str(dd.value)
        universe = universe or "ALL (Combined)"

        def _bg():
            try:
                self._log(f"Stale-member audit: {universe} …")
                res, report = self._audit_report(universe)
                verdict = self._audit_verdict(res)
                lock = getattr(self, "_state_lock", None)
                if lock is not None:
                    with lock:
                        self._last_audit_res = res
                else:
                    self._last_audit_res = res
                self._log("\n" + report)
                self._safe_update(lambda: self._log(f"Audit done — {verdict}"))
                self._safe_update(
                    lambda: self._toast(f"Audit done — {verdict}", "info")
                )
                if hasattr(self, "stale_audit_lbl"):
                    self._safe_update(
                        lambda: setattr(
                            self.stale_audit_lbl, "value", f"Audit done — {verdict}"
                        )
                    )
                if hasattr(self, "stale_fix_btn"):
                    self._safe_update(
                        lambda: setattr(
                            self.stale_fix_btn, "disabled", not self._audit_fixable(res)
                        )
                    )
            except Exception as ex:
                msg = f"Stale-member audit failed: {ex}"
                self._safe_update(lambda: self._log(msg))
                # Never leave a stale (previously fixable) result actionable
                # after a failed audit — force a fresh audit before applying.
                lock = getattr(self, "_state_lock", None)
                if lock is not None:
                    with lock:
                        self._last_audit_res = None
                else:
                    self._last_audit_res = None
                if hasattr(self, "stale_fix_btn"):
                    self._safe_update(
                        lambda: setattr(self.stale_fix_btn, "disabled", True)
                    )
            finally:
                lock = getattr(self, "_state_lock", None)
                if lock is not None:
                    with lock:
                        self.stale_audit_running = False
                else:
                    self.stale_audit_running = False
                self._safe_update(lambda: self.page.update())

        threading.Thread(target=_bg, daemon=True).start()

    def _apply_fixes_core(self, res: dict):
        """Sync core of the Apply fixes button (threaded in the GUI)."""
        from ..backend.audit_stale_members import apply_fixes, format_fix_summary

        summary = apply_fixes(res)
        return summary, format_fix_summary(summary)

    def _apply_stale_fixes(self, e=None):
        """Ask for confirmation, then apply audit fixes to universes.py.

        The dialog lists exactly what will change (renames / annotation adds /
        removals); its Apply button runs the threaded fix, which keeps a .bak
        and re-audits afterwards to confirm the file is clean.
        """
        if getattr(self, "stale_fix_running", False):
            return
        lock = getattr(self, "_state_lock", None)
        if lock is not None:
            with lock:
                res = getattr(self, "_last_audit_res", None)
        else:
            res = getattr(self, "_last_audit_res", None)
        if not res or not self._audit_fixable(res):
            return
        parts = []
        renames = len(res.get("rename_suggestions", {}))
        if renames:
            parts.append(f"{renames} rename(s)")
        adds = len(res.get("unannotated_stale", []))
        if adds:
            parts.append(f"{adds} annotation add(s)")
        removes = len(res.get("annotated_fresh", []))
        if removes:
            parts.append(f"{removes} annotation removal(s)")
        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Apply audit fixes?"),
            content=ft.Text(
                f"This edits scanner/shared/universes.py ({', '.join(parts)}). "
                "A .bak backup is kept and the edit is syntax-checked. "
                "The app re-audits afterwards; restart it for scans to "
                "pick up the new universe lists.",
                size=13,
            ),
            actions=[
                ft.TextButton(
                    content=ft.Text("Cancel"),
                    on_click=lambda _: self._close_dialog(dlg),
                ),
                ft.Button(
                    content=ft.Text("Apply fixes"),
                    on_click=lambda _: self._run_apply_fixes(dlg),
                ),
            ],
        )
        self._safe_update(lambda: self.page.show_dialog(dlg))

    def _close_dialog(self, dlg):
        self._safe_update(lambda: self.page.pop_dialog())

    def _reload_universes(self):
        """Re-read universes.py after a fix so a re-audit sees the edits."""
        try:
            import importlib

            import scanner.shared.universes as univ

            importlib.reload(univ)
        except Exception as ex:
            self._log(f"Could not reload universes after fix: {ex}")

    def _run_apply_fixes(self, dlg=None):
        """Threaded fix runner: apply_fixes, log the summary, re-audit."""
        if dlg is not None:
            self._close_dialog(dlg)
        res = getattr(self, "_last_audit_res", None)
        if not res or getattr(self, "stale_fix_running", False):
            return
        self.stale_fix_running = True

        def _bg():
            try:
                self._log("Applying audit fixes to scanner/shared/universes.py …")
                summary, text = self._apply_fixes_core(res)
                self._log("\n" + text)
                if summary.get("changed"):
                    verdict = (
                        f"{len(summary['renamed'])} rename(s), "
                        f"{len(summary['annotated_added'])} annotation(s) added, "
                        f"{len(summary['annotated_removed'])} removed — "
                        "universes.py updated (backup kept; restart the app "
                        "for scans to pick up the new symbols)"
                    )
                else:
                    verdict = "Nothing to apply (annotation list and symbols current)"
                self._safe_update(lambda: self._log(f"Fix done — {verdict}"))
                if hasattr(self, "stale_audit_lbl"):
                    self._safe_update(
                        lambda: setattr(
                            self.stale_audit_lbl, "value", f"Fix done — {verdict}"
                        )
                    )
                if hasattr(self, "stale_fix_btn"):
                    self._safe_update(
                        lambda: setattr(self.stale_fix_btn, "disabled", True)
                    )
                # Confirm the file is clean: reload the edited module and
                # re-run the same audit (cache-warm, fast).
                if summary.get("changed"):
                    self._reload_universes()
                    self._safe_update(
                        lambda: self._log(
                            "Re-auditing to confirm universes.py is clean …"
                        )
                    )
                    self._run_stale_audit()
            except Exception as ex:
                msg = f"Applying audit fixes failed: {ex}"
                self._safe_update(lambda: self._log(msg))
            finally:
                # Clear only the result we applied — the re-audit started above
                # may already have stored a newer result that must survive.
                # The check-and-clear is atomic under the lock; without it a
                # racing audit could store a fresh result between our read and
                # write and we would silently discard it.
                lock = getattr(self, "_state_lock", None)
                if lock is not None:
                    with lock:
                        if self._last_audit_res is res:
                            self._last_audit_res = None
                elif self._last_audit_res is res:
                    self._last_audit_res = None
                self.stale_fix_running = False
                self._safe_update(lambda: self.page.update())

        threading.Thread(target=_bg, daemon=True).start()

    # ── Export ──────────────────────────────────────────────────────────

    def _export_html(self, e=None):
        if not self.results:
            return
        from ..backend.report import generate_html_report, save_report

        threshold = self.settings.get("min_score", 50)
        tf_names = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
        tf_label = tf_names.get(self.settings.get("timeframe", "D"), "Daily")
        results_snapshot = list(self.results)
        universe_name = self.universe_dd.value or "NIFTY 50"
        safe_title = f"HMAxEMA Scanner — {universe_name} — {tf_label}"

        def _bg():
            try:
                os.makedirs(REPORTS_DIR, exist_ok=True)
                self._log("Fetching news sentiment for exported stocks...")
                html = generate_html_report(
                    results_snapshot,
                    title=safe_title,
                    threshold=threshold,
                    fetch_news=True,
                )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scanner_report_{timestamp}.html"
                filepath = os.path.join(REPORTS_DIR, filename)
                save_report(html, filepath)
                self._safe_update(lambda: self._log(f"HTML report saved: {filename}"))
                self._safe_update(
                    lambda: self._toast(f"Report saved: {filename}", "success")
                )
                webbrowser.open(f"file://{os.path.abspath(filepath)}")
            except Exception as ex:
                logger.exception("HTML export failed")  # traceback carries ex
                msg = f"HTML export failed: {ex}"
                # Bind first — ``ex`` is cleared on except-block exit.
                self._safe_update(lambda: self._log(msg))
                self._safe_update(lambda: self._toast(msg, "error"))

        threading.Thread(target=_bg, daemon=True).start()

    def _export_csv(self, e=None):
        if not self.results:
            return
        import csv

        with self._results_lock:
            results_snapshot = list(self.results)

        def _bg():
            try:
                os.makedirs(REPORTS_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scanner_results_{timestamp}.csv"
                filepath = os.path.join(REPORTS_DIR, filename)
                with open(filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "Rank",
                            "Ticker",
                            "Score",
                            "Rating",
                            "Price",
                            "MA_Signal",
                            "POC",
                            "Both_MA",
                            "Trend",
                            "Momentum",
                            "RSI",
                            "MACD",
                            "Volume",
                            "RelStrength",
                            "Fundamentals",
                            "Direction",
                            "RSI_Val",
                            "ADX",
                            "1M_Change",
                            "Volatility_Stat",
                            "Sideways",
                        ]
                    )
                    for i, r in enumerate(results_snapshot, 1):
                        sideways_reasons = ", ".join(r.get("sideways_reasons", []))
                        ma_signal = "Bull" if r.get("ma_bullish") else "Bear"
                        if r.get("ma_crossed_above"):
                            bars = r.get("crossover_bars_ago")
                            ma_signal += f" (x{bars}b)" if bars is not None else " (x)"
                        poc = (
                            f"Above ({r.get('vp_poc', '—')})"
                            if r.get("above_poc")
                            else f"Below ({r.get('vp_poc', '—')})"
                        )
                        both_ma = "Yes" if r.get("close_above_both_ma") else "No"
                        writer.writerow(
                            [
                                i,
                                r.get("ticker", ""),
                                r.get("total", 0) or 0,
                                r.get("combined_rating", "POOR"),
                                r.get("close"),
                                ma_signal,
                                poc,
                                both_ma,
                                r.get("trend"),
                                r.get("momentum"),
                                r.get("rsi"),
                                r.get("macd"),
                                r.get("volume"),
                                r.get("rel_str"),
                                r.get("fundamentals", 0),
                                r.get("trend_dir", ""),
                                r.get("rsi_val"),
                                r.get("adx_val"),
                                r.get("pc1m"),
                                r.get("volat_stat", ""),
                                (
                                    "Yes"
                                    + (
                                        f" ({sideways_reasons})"
                                        if sideways_reasons
                                        else ""
                                    )
                                )
                                if r.get("is_sideways")
                                else "No",
                            ]
                        )
                self._safe_update(lambda: self._log(f"CSV saved: {filename}"))
                self._safe_update(
                    lambda: self._toast(f"CSV saved: {filename}", "success")
                )
            except Exception as exc:
                err = str(exc)
                self._safe_update(lambda: self._log(f"CSV export failed: {err}"))
                self._safe_update(
                    lambda: self._toast(f"CSV export failed: {err}", "error")
                )

        threading.Thread(target=_bg, daemon=True).start()

    # ── Utilities ───────────────────────────────────────────────────────

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
            "chart_card",
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
        try:
            from ..backend.settings_store import save_results

            save_results([])
        except Exception:
            logger.info("Failed to clear persisted results", exc_info=True)
        self.page.update()

    def _clear_log(self, e=None):
        try:
            if getattr(self, "log_column", None) is not None:
                self.log_column.controls.clear()
                self.page.update()
        except Exception:
            logger.info("_clear_log failed", exc_info=True)

    def _get_log_lines(self) -> list:
        try:
            col = getattr(self, "log_column", None)
            if col is None:
                return []
            return [t.value for t in col.controls if isinstance(t, ft.Text)]
        except Exception:
            logger.info("_get_log_lines failed", exc_info=True)
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
            logger.info("_set_log_lines failed", exc_info=True)

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
            text,
            size=10,
            color=c["text_dim"],
            selectable=True,
            font_family="Consolas",
        )

    def _log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        try:
            os.makedirs(APPLOG_DIR, exist_ok=True)
            with _log_lock:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            logger.info("Failed to write to log file", exc_info=True)

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

    def _rotate_log(self):
        try:
            if os.path.exists(LOG_FILE):
                age_hours = (
                    datetime.now().timestamp() - os.path.getmtime(LOG_FILE)
                ) / 3600
                if age_hours >= LOG_ROTATE_HOURS:
                    with _log_lock:
                        # open("w") already truncates — no explicit truncate needed.
                        # Truncation stays inside the lock so a concurrent _log
                        # can neither interleave a write nor reopen a stale handle.
                        with open(LOG_FILE, "w"):
                            pass
        except Exception:
            logger.info("Log rotation failed", exc_info=True)


def main(page: ft.Page):
    # The instance stays alive via the bound handlers _build_ui registers on
    # page controls — no local reference needed.
    ScannerApp(page)


if __name__ == "__main__":
    ft.run(main)
