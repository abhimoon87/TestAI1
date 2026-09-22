"""
News prefetcher for the HMAxEMA Scanner.

Fetches news for top-scored results after a scan completes, attaching
``_news_items``, ``_article_count`` and ``_sentiment_score`` to each row
so clicking a ticker renders instantly.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from ..shared.constants import score_of
from .report import fetch_news_batch

logger = logging.getLogger(__name__)


class NewsPrefetcher:
    """Epoch-guarded async news fetcher.

    Each scan increments the epoch; if a prefetch is still in flight when
    the next scan starts, it detects the stale epoch and drops its updates.
    """

    def __init__(
        self,
        get_results: callable,
        results_lock: threading.RLock,
        get_epoch: callable,
        get_cancelled: callable,
        on_update: callable | None = None,
        on_log: callable | None = None,
    ):
        self._get_results = get_results
        self._results_lock = results_lock
        self._get_epoch = get_epoch
        self._get_cancelled = get_cancelled
        self._on_update = on_update
        self._on_log = on_log

    def prefetch(self, top_n: int = 50) -> None:
        """Fetch news for the top ``top_n`` scored rows.

        Runs in a daemon thread. Each targeted row dict gains
        ``_news_items`` (provider-keyed story dicts) plus
        ``_article_count`` / ``_sentiment_score`` when stories were found.
        """
        epoch = self._get_epoch()
        try:
            with self._results_lock:
                pool = list(self._get_results())
            by_ticker = {r.get("ticker"): r for r in pool if r.get("ticker")}
            if not by_ticker:
                return
            targets = [
                r
                for r in sorted(by_ticker.values(), key=score_of, reverse=True)[:top_n]
                if "_news_items" not in r
            ]
            if not targets:
                return
            if self._on_log:
                self._on_log("Prefetching top-50 news\u2026")
            news_map = fetch_news_batch([r["ticker"] for r in targets])
            if not news_map:
                return
            attached = 0
            with self._results_lock:
                if epoch != self._get_epoch() or self._get_cancelled():
                    return  # a newer scan owns the results now
                live = {
                    r.get("ticker"): r for r in self._get_results() if r.get("ticker")
                }
                updates: dict[str, dict] = {}
                for ticker, items in news_map.items():
                    row = live.get(ticker)
                    if row is None or "_news_items" in row:
                        continue
                    u: dict[str, Any] = {"_news_items": items}
                    n = len(items)
                    if n:
                        if "_article_count" not in row:
                            u["_article_count"] = n
                        if "_sentiment_score" not in row:
                            tone = {"Good": 1.0, "Bad": -1.0}
                            u["_sentiment_score"] = (
                                sum(
                                    tone.get(i.get("sentiment", "Neutral"), 0.0)
                                    for i in items
                                )
                                / n
                            )
                    updates[ticker] = u
                for ticker, u in updates.items():
                    row = live[ticker]
                    row.update(u)
                    attached += 1
            if attached and self._on_update:
                self._on_update()
            if self._on_log and attached:
                self._on_log(
                    f"News prefetched for {attached}/{len(targets)} top stocks"
                )
        except Exception as e:
            if self._on_log:
                self._on_log(f"News prefetch failed: {e!s}")

    def start(self, top_n: int = 50) -> None:
        """Launch prefetch in a daemon thread."""
        threading.Thread(target=self.prefetch, args=(top_n,), daemon=True).start()
