"""Report export — ``ExportViewMixin`` for ``scanner.app.ScannerApp``.

Owns the HTML/CSV export actions: snapshotting the result grid, writing
timestamped files under ``Reports/`` and opening the HTML report.

The methods are mixins: they run against the ``ScannerApp`` instance (via
the MRO) and rely on state/controls created elsewhere (``self._log``,
``self._toast``, ``self._safe_update``, …).
"""

import logging
import os
import threading
import webbrowser
from datetime import datetime

from ..shared.constants import score_of

logger = logging.getLogger(__name__)

# Mirrors scanner.ui.app.REPORTS_DIR — app.py keeps its own copy (backtest).
SCANNER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(SCANNER_DIR))
REPORTS_DIR = os.path.join(_PROJECT_ROOT, "Reports")


class ExportViewMixin:
    # ── Export ──────────────────────────────────────────────────────────

    def _export_html(self, e=None):
        if not self.results:
            return
        from ..backend.report import generate_html_report, save_report

        threshold = self.settings.get("min_score", 50)
        tf_names = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
        tf_label = tf_names.get(self.settings.get("timeframe", "D"), "Daily")
        # Export the full scan: the report re-applies the grid filter state
        # client-side, so its checkboxes/chips stay interactive like the app.
        # Counts/status track the full scan; news prefetch stays scoped to
        # what the grid shows so export time doesn't grow with the universe.
        all_src = self.all_results or self.results
        results_snapshot = list(all_src)
        news_targets = (
            self._visible_results()
            if getattr(self, "all_results", None)
            else list(self.results)
        )
        universe_name = self.universe_dd.value or "NIFTY 50"
        safe_title = "HMAxEMA Scanner"
        filters = {
            "search": self.filter_text or "",
            "rating": self._rating_filter(),
            "score": self._score_threshold(),
            "fii_dii": self._checkbox_on("inst_filter_cb"),
            "price100": self._checkbox_on("price_filter_cb"),
            "fund0": self._checkbox_on("fund_filter_cb"),
            "scanned": len(all_src),
            "above": sum(1 for r in all_src if score_of(r) >= threshold),
        }

        def _bg():
            try:
                os.makedirs(REPORTS_DIR, exist_ok=True)
                if any("_news_items" not in r for r in news_targets):
                    self._log("Fetching news sentiment for exported stocks...")
                else:
                    self._log("Exporting report (news already prefetched)...")
                html = generate_html_report(
                    results_snapshot,
                    title=safe_title,
                    threshold=threshold,
                    fetch_news=True,
                    meta=[f"🌐 {universe_name}", f"📊 {tf_label}"],
                    filters=filters,
                    news_targets=news_targets,
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

        from ..backend.report import prune_old

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
                prune_old(REPORTS_DIR, "scanner_results_*.csv", 4)
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
