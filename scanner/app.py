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
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import flet as ft
from flet.controls.alignment import Alignment

from .trace import setup_trace

try:
    setup_trace()
except Exception:
    pass

logger = logging.getLogger(__name__)
logger.info("app module loaded -- trace active at %s", Path(__file__).parent / "trace.log")

from .report import generate_html_report, save_report
from .themes import THEMES
from .universes import UNIVERSES
from .views_backtest import BacktestViewMixin
from .views_layout import LayoutViewMixin
from .views_results import ResultsViewMixin
from .views_settings import SettingsViewMixin

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
    "slope_ma_type": "EMA", "slope_ma_len": 50, "slope_lookback": 10, "flat_threshold": 0.5,
    "sideways_strong_move_pct": 5.0, "volume_participation_len": 5,
    "min_adx_entry": 20.0,
    "sc_pivot_len": 3, "sc_bands_mult": 0.6, "crossover_lookback": 20,
    "min_score": 50.0, "data_period": "1y", "timeframe": "D", "trend_filter": "All",
    "negative_cache_ttl_hours": 24, "theme": "dark",
    "stale_member_max_age_days": 45.0,
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


class ScannerApp(LayoutViewMixin, ResultsViewMixin, SettingsViewMixin, BacktestViewMixin):
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
        self.last_warnings: list[str] = []
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
        self._refresh_price_cache_ui()
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
        elif name == "backtest":
            box = getattr(self, "main_area_box", None)
            if box is not None:
                box.content = self._build_backtest_view()
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
        period_map: dict[str, str] = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
        s["data_period"] = period_map.get(self.period_dd.value or "1 Year", "1y")
        # timeframe dropdown: map display name → engine key; default to "D"
        tf_map: dict[str, str] = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
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
                self.last_warnings = list(getattr(result, "warnings", []) or [])
                self._render_current_page()
                if result.cancelled:
                    self._log(f"Scan stopped — showing {len(result.results)} partial results.")
                if result.error:
                    self._log(f"Scan finished with error: {result.error}")

            self._safe_update(_final_sync)

        except Exception as e:
            msg = f"\nERROR: {e!s}"
            # Bind the message first: ``e`` is cleared when the except block
            # exits, so a closure referencing it would NameError if deferred.
            self._safe_update(lambda: self._log(msg))
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
        self._refresh_price_cache_ui()
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

    def _refresh_price_cache_ui(self):
        if not hasattr(self, "price_cache_status_lbl"):
            return
        try:
            from .data_providers import cache_health
            h = cache_health()
            n, stale = h["price_entries"], h["stale_entries"]
        except Exception:
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
            from .data_providers import prune_stale_cache
            removed = prune_stale_cache(force=True)
            self._log(
                f"Pruned {removed} stale price-cache entrie(s) "
                "(previous trading days)" if removed else "Price cache clean — nothing to prune"
            )
        except Exception as ex:
            self._log(f"Could not prune price cache: {ex}")
        self._refresh_price_cache_ui()
        self.page.update()

    def _audit_report(self, universe: str):
        """Sync core of the stale-member audit button (threaded in the GUI)."""
        from .audit_stale_members import audit_stale_members, format_report
        from .universes import UNIVERSES
        tickers = list(UNIVERSES.get(universe, []))
        res = audit_stale_members(tickers)
        return res, format_report(res)

    def _audit_fixable(self, res: dict) -> bool:
        """True when the audit result has anything apply_fixes could change."""
        return bool(res.get("rename_suggestions")
                    or res.get("unannotated_stale")
                    or res.get("annotated_fresh"))

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
        if getattr(self, "stale_audit_running", False):
            return
        universe = None
        dd = getattr(self, "universe_dd", None)
        if dd is not None and getattr(dd, "value", None):
            universe = str(dd.value)
        universe = universe or "ALL (Combined)"
        self.stale_audit_running = True

        def _bg():
            try:
                self._log(f"Stale-member audit: {universe} …")
                res, report = self._audit_report(universe)
                verdict = self._audit_verdict(res)
                self._last_audit_res = res
                self._log("\n" + report)
                self._safe_update(lambda: self._log(f"Audit done — {verdict}"))
                if hasattr(self, "stale_audit_lbl"):
                    self._safe_update(
                        lambda: setattr(self.stale_audit_lbl, "value",
                                        f"Audit done — {verdict}")
                    )
                if hasattr(self, "stale_fix_btn"):
                    self._safe_update(
                        lambda: setattr(self.stale_fix_btn, "disabled",
                                        not self._audit_fixable(res))
                    )
            except Exception as ex:
                msg = f"Stale-member audit failed: {ex}"
                self._safe_update(lambda: self._log(msg))
                # Never leave a stale (previously fixable) result actionable
                # after a failed audit — force a fresh audit before applying.
                self._last_audit_res = None
                if hasattr(self, "stale_fix_btn"):
                    self._safe_update(
                        lambda: setattr(self.stale_fix_btn, "disabled", True)
                    )
            finally:
                self.stale_audit_running = False
                self._safe_update(lambda: self.page.update())

        threading.Thread(target=_bg, daemon=True).start()

    def _apply_fixes_core(self, res: dict):
        """Sync core of the Apply fixes button (threaded in the GUI)."""
        from .audit_stale_members import apply_fixes, format_fix_summary

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
                f"This edits scanner/universes.py ({', '.join(parts)}). "
                "A .bak backup is kept and the edit is syntax-checked. "
                "The app re-audits afterwards; restart it for scans to "
                "pick up the new universe lists.",
                size=13,
            ),
            actions=[
                ft.TextButton(content=ft.Text("Cancel"),
                              on_click=lambda _: self._close_dialog(dlg)),
                ft.ElevatedButton(content=ft.Text("Apply fixes"),
                                  on_click=lambda _: self._run_apply_fixes(dlg)),
            ],
        )
        self._safe_update(lambda: self.page.open(dlg))

    def _close_dialog(self, dlg):
        self._safe_update(lambda: self.page.close(dlg))

    def _reload_universes(self):
        """Re-read universes.py after a fix so a re-audit sees the edits."""
        try:
            import importlib

            import scanner.universes as univ

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
                self._log("Applying audit fixes to scanner/universes.py …")
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
                        lambda: setattr(self.stale_audit_lbl, "value",
                                        f"Fix done — {verdict}")
                    )
                if hasattr(self, "stale_fix_btn"):
                    self._safe_update(
                        lambda: setattr(self.stale_fix_btn, "disabled", True)
                    )
                # Confirm the file is clean: reload the edited module and
                # re-run the same audit (cache-warm, fast).
                if summary.get("changed"):
                    self._reload_universes()
                    self._safe_update(lambda: self._log(
                        "Re-auditing to confirm universes.py is clean …"))
                    self._run_stale_audit()
            except Exception as ex:
                msg = f"Applying audit fixes failed: {ex}"
                self._safe_update(lambda: self._log(msg))
            finally:
                # Clear only the result we applied — the re-audit started above
                # may already have stored a newer result that must survive.
                if self._last_audit_res is res:
                    self._last_audit_res = None
                self.stale_fix_running = False
                self._safe_update(lambda: self.page.update())

        threading.Thread(target=_bg, daemon=True).start()

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
                msg = f"HTML export failed: {ex}"
                # Bind first — ``ex`` is cleared on except-block exit.
                self._safe_update(lambda: self._log(msg))

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
    # The instance stays alive via the bound handlers _build_ui registers on
    # page controls — no local reference needed.
    ScannerApp(page)


if __name__ == "__main__":
    ft.run(main)
