"""Cache-card UI — ``CacheViewMixin`` for ``scanner.app.ScannerApp``.

Owns the three sidebar cache cards (dead-symbol, enrichment, price): status
snapshotting, the off-thread refresh, and the clear/prune card actions.

The methods are mixins: they run against the ``ScannerApp`` instance (via
the MRO) and rely on state/controls created elsewhere (``self._safe_update``,
``self._log``, ``self._toast``, …).
"""

import logging
import threading

logger = logging.getLogger(__name__)


class CacheViewMixin:
    # ── Cache management ───────────────────────────────────────────────

    def _cache_ui_controls(self, kind: str):
        """(label, clear-button) for a cache card; None before the sidebar exists."""
        if kind == "neg":
            lbl = getattr(self, "cache_status_lbl", None)
            btn = getattr(self, "cache_clear_btn", None)
        elif kind == "enrich":
            lbl = getattr(self, "enrich_cache_status_lbl", None)
            btn = getattr(self, "enrich_cache_clear_btn", None)
        else:
            lbl = getattr(self, "price_cache_status_lbl", None)
            btn = getattr(self, "price_cache_prune_btn", None)
        if lbl is None:
            return None
        return lbl, btn

    def _cache_ui_snapshot(self, kind: str):
        """Compute (label text, button visible) for a cache card.

        Touches only caches — no controls — so it is safe off the UI thread.
        """
        if self._cache_ui_controls(kind) is None:
            return None
        from ..api import cache_manager

        if kind in ("neg", "enrich"):
            title = "Dead-symbol cache" if kind == "neg" else "Enrichment cache"
            try:
                if kind == "neg":
                    n = len(cache_manager.negative_load())
                    ttl_h = cache_manager.negative_ttl_hours()
                else:
                    n = cache_manager.enrichment_size()
                    ttl_h = cache_manager.ENRICHMENT_CACHE_TTL_HOURS
            except Exception:
                logger.info("Failed to load %s info", title, exc_info=True)
                n, ttl_h = 0, 24
            text = f"{title}: {n} (auto-resets ~{ttl_h}h)" if n else f"{title}: empty"
            return text, bool(n)

        # price cache has a different shape (stale count)
        try:
            h = cache_manager.cache_health()
            n, stale = h["price_entries"], h["stale_entries"]
        except Exception:
            logger.info("Failed to load price cache health", exc_info=True)
            n, stale = 0, 0
        if not n:
            text = "Price cache: empty"
        elif stale:
            text = f"Price cache: {n} ({stale} stale — auto-prunes on next scan)"
        else:
            text = f"Price cache: {n} (clean)"
        return text, bool(stale)

    def _apply_cache_ui(self, kind: str, text: str, visible: bool) -> None:
        """Push a computed snapshot onto the cache card controls (UI thread)."""
        controls = self._cache_ui_controls(kind)
        if controls is None:
            return
        lbl, btn = controls
        lbl.value = text
        if btn is not None:
            btn.visible = visible

    def _refresh_cache_ui(self, kind: str):
        """Shared status/clear wiring for the three cache cards (UI thread)."""
        snap = self._cache_ui_snapshot(kind)
        if snap is not None:
            self._apply_cache_ui(kind, *snap)

    def _refresh_cache_ui_bg(self, *kinds: str) -> None:
        """Compute cache-card status off-thread, then apply on the UI thread."""
        kinds = kinds or ("neg", "enrich", "price")

        def _bg():
            snaps = {k: self._cache_ui_snapshot(k) for k in kinds}

            def _apply():
                for k, snap in snaps.items():
                    if snap is not None:
                        self._apply_cache_ui(k, *snap)

            self._safe_update(_apply)

        t = threading.Thread(target=_bg, daemon=True)
        # Joinable handle: the snapshot reads sqlite/parquet, and a test
        # teardown that closes db conns under a still-running reader crashes
        # natively — tests join this before the db fixture tears down.
        self._cache_refresh_thread = t
        t.start()

    def _refresh_neg_cache_ui(self):
        self._refresh_cache_ui("neg")

    def _finish_cache_action(self, refresh, log_msg, toast_msg, is_error=False):
        """Log + toast + re-read a cache card — shared by the card actions."""
        self._log(log_msg)
        if toast_msg is not None:
            self._toast(toast_msg, "error" if is_error else "success")
        refresh()
        self.page.update()

    def _clear_negative_cache(self, e=None):
        def _bg():
            is_error = False
            try:
                from ..api import cache_manager

                cache_manager.negative_update(
                    clears=list(cache_manager.negative_load().keys())
                )
                msg = "Cleared dead-symbol cache — fallback will re-attempt all symbols"
                toast = "Dead-symbol cache cleared"
            except Exception as ex:
                msg = toast = f"Could not clear dead-symbol cache: {ex}"
                is_error = True
            self._safe_update(
                lambda: self._finish_cache_action(
                    self._refresh_neg_cache_ui, msg, toast, is_error
                )
            )

        threading.Thread(target=_bg, daemon=True).start()

    def _refresh_enrich_cache_ui(self):
        self._refresh_cache_ui("enrich")

    def _clear_enrichment_cache(self, e=None):
        def _bg():
            is_error = False
            try:
                from ..api import cache_manager

                cache_manager.enrichment_clear()
                msg = "Cleared enrichment cache — next scan will re-fetch phase-2 data"
                toast = "Enrichment cache cleared"
            except Exception as ex:
                msg = toast = f"Could not clear enrichment cache: {ex}"
                is_error = True
            self._safe_update(
                lambda: self._finish_cache_action(
                    self._refresh_enrich_cache_ui, msg, toast, is_error
                )
            )

        threading.Thread(target=_bg, daemon=True).start()

    def _refresh_price_cache_ui(self):
        self._refresh_cache_ui("price")

    def _prune_price_cache(self, e=None):
        def _bg():
            is_error = False
            try:
                from ..api import cache_manager

                removed = cache_manager.prune_stale_cache(force=True)
                if removed:
                    msg = f"Pruned {removed} price-cache entrie(s) (expired + over cap)"
                    toast = f"Pruned {removed} price-cache entries"
                else:
                    msg = "Price cache clean — nothing to prune"
                    toast = None
            except Exception as ex:
                msg = toast = f"Could not prune price cache: {ex}"
                is_error = True
            self._safe_update(
                lambda: self._finish_cache_action(
                    self._refresh_price_cache_ui, msg, toast, is_error
                )
            )

        threading.Thread(target=_bg, daemon=True).start()
