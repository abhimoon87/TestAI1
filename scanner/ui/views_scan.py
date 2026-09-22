"""Scan orchestration — ``ScanOrchestrationMixin`` for ``scanner.app.ScannerApp``.

Owns the start/stop/run/stream/complete lifecycle for a scan: settings
collection, engine callbacks, streaming batch merge, and the final UI sync.

The methods are mixins: they run against the ``ScannerApp`` instance (via
the MRO) and rely on state/controls created elsewhere (``self._scan_lock``,
``self.action_btn``, ``self.table_column``, …).
"""

import logging
import threading

from ..backend.settings_store import save_settings
from ..shared.constants import NEWS_PREFETCH_TOP

logger = logging.getLogger(__name__)


class ScanOrchestrationMixin:
    def _on_action_click(self, e=None):
        if self.scanning:
            self._stop_scan()
        else:
            self._start_scan()

    def _start_scan(self, e=None):
        with self._scan_lock:
            if self.scanning:
                return
            self.scanning = True

        try:
            self.settings = self._collect_settings()
            save_settings(self.settings)
            self._apply_cache_settings()
        except Exception as exc:
            with self._scan_lock:
                self.scanning = False
            msg = f"Scan setup failed: {exc}"
            toast_msg = f"Settings save failed: {exc}"
            self._safe_update(lambda: self._log(msg))
            self._safe_update(lambda: self._toast(toast_msg, "error"))
            return

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
        with self._results_lock:
            self.results = []
            self.all_results = []
            self.filtered_results = []
        # Reset the row control pool so streaming builds fresh controls
        if hasattr(self, "_row_pool"):
            self._row_pool.clear()
        if hasattr(self, "_row_cells"):
            self._row_cells.clear()

        if self.active_view == "dashboard":
            self.table_column.controls.clear()
            self.table_column.controls.append(self._make_scan_placeholder())
        self.page.update()

        threading.Thread(target=self._run_scan, daemon=True).start()

    def _stop_scan(self, e=None):
        if not self.scanning:
            return
        self._stop_requested = True
        with self._scan_lock:
            engine = self._scan_engine
        self.action_btn.disabled = True
        self.action_btn_label.value = "◷  STOPPING…"
        self.page.update()
        if engine is not None:
            engine.cancel()
            self._log("Stop requested — finishing the current batch, then stopping...")

    def _run_scan(self):
        # Bump the scan epoch so background passes from an older scan (e.g.
        # a news prefetch) can detect they no longer own the results.
        with self._scan_lock:
            self._scan_epoch += 1
        try:
            from ..backend.scanner_engine import ScannerEngine

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
                # If result.results is empty but we have streaming results,
                # preserve the streaming results (they survived via _on_stream_batch)
                final_results = result.results
                if not final_results and self.all_results:
                    final_results = self.all_results
                logger.info(
                    "_final_sync: result.results=%d self.all_results=%d final_results=%d",
                    len(result.results),
                    len(self.all_results),
                    len(final_results),
                )
                with self._results_lock:
                    self.results = final_results
                    self.all_results = list(final_results)
                    self.filtered_results = [
                        r for r in final_results if self._row_matches_filters(r)
                    ]
                self.last_warnings = list(getattr(result, "warnings", []) or [])
                # NOTE: _render_current_page() is NOT called here —
                # _scan_complete (queued right after) does its own render
                # plus the final status update.  Calling it here would
                # double-render and, on large result sets, the second
                # page.update() from _scan_complete could queue behind
                # a slow first update, leaving the UI stuck at
                # "Finalizing scan…".
                if result.cancelled:
                    self._log(
                        f"Scan stopped — showing {len(final_results)} partial results."
                    )
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
        import time as _time

        now = _time.monotonic()
        # Debounce: re-render at most once every 2s to avoid thrashing
        # the Flet grid on rapid batch arrivals.
        last_render = getattr(self, "_last_stream_render", 0.0)
        should_render = (now - last_render) >= 2.0
        with self._results_lock:
            existing = {r.get("ticker"): idx for idx, r in enumerate(self.all_results)}
            filtered_idx = {
                r.get("ticker"): idx for idx, r in enumerate(self.filtered_results)
            }
            filtered_removed: list[str] = []
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
                elif t in filtered_idx:
                    # Defer removal: popping mid-loop would shift the positions
                    # the remaining filtered_idx entries point at, removing
                    # the wrong rows — or raising IndexError, which the engine
                    # swallows and the grid then never updates.
                    filtered_removed.append(t)
            if filtered_removed:
                removed = set(filtered_removed)
                self.filtered_results = [
                    fr
                    for fr in self.filtered_results
                    if fr.get("ticker") not in removed
                ]
            self.results = self.all_results
        if should_render:
            self._last_stream_render = now
            self._safe_update(lambda: self._render_current_page())

    def _scan_complete(self):
        with self._scan_lock:
            self.scanning = False
        c = self.theme_colors
        self.action_btn.disabled = False
        self.action_btn_label.value = "▶  RUN SCAN"
        self.action_btn.bgcolor = c["green"]
        self.progress_label.value = "Stopped" if self._scan_cancelled else "Done"
        self.status_label.value = (
            "Status: Stopped" if self._scan_cancelled else "Status: Done"
        )
        if not self._scan_cancelled:
            self.progress_bar.value = 1.0
        self._refresh_neg_cache_ui()
        self._refresh_enrich_cache_ui()
        self._refresh_price_cache_ui()
        if self.results:
            self.html_btn.disabled = False
            self.csv_btn.disabled = False
            self.clear_btn.disabled = False
        self._toast(
            f"Scan {'stopped' if self._scan_cancelled else 'complete'} — "
            f"{len(self.results)} results",
            "info" if self._scan_cancelled else "success",
        )
        # Re-render the full page now that scanning=False, so the hero
        # subtitle shows the final count and the table reflects the
        # definitive result set (not the last streaming batch snapshot).
        # Clear the row pool to force a full rebuild (not the streaming
        # fast-path which skips summary/hero/chart updates).
        logger.info(
            "_scan_complete: results=%d all_results=%d active_view=%s",
            len(self.results),
            len(self.all_results),
            self.active_view,
        )
        if hasattr(self, "_row_pool"):
            self._row_pool.clear()
        if hasattr(self, "_row_cells"):
            self._row_cells.clear()
        try:
            self._render_current_page()
        except Exception:
            logger.info("_scan_complete: _render_current_page FAILED", exc_info=True)
        # The scan just re-fetched NIFTY through the provider chain — refresh
        # the hero readout from that cache (fast, mostly disk reads).
        threading.Thread(target=self._warm_market, daemon=True).start()
        # Prefetch the top-scored rows' news in the background so clicking a
        # row opens its stories instantly (no per-click yfinance round-trip).
        if self.results and not self._scan_cancelled:
            self._news_prefetcher.start(NEWS_PREFETCH_TOP)

    def _flush_news_badges(self):
        """Repaint the visible table so prefetched news badges appear."""
        try:
            with self._scan_lock:
                still_scanning = self.scanning
            if not still_scanning:
                self._render_current_page()
        except Exception:
            logger.info("_flush_news_badges failed", exc_info=True)

    def _set_progress(self, value, text=""):
        self.progress_bar.value = value
        if text:
            self.progress_label.value = text
            self.status_label.value = f"Status: {text}"
