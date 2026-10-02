"""
Scanner Engine — Pure Python headless scanning logic.
Extracted from app.py for testability and reusability.
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from requests.exceptions import RequestException

from ..api.cache_manager import (
    enrichment_cache_size,
    enrichment_get,
    enrichment_put,
    enrichment_stats,
    negative_skip_count,
    record_enrichment_hit,
    record_enrichment_miss,
    reset_enrichment_counts,
    reset_negative_skips,
)
from ..api.data_fetcher import (
    _DaemonThreadPoolExecutor,
    fetch_batch_yfinance,
    fetch_batch_yfinance_stream,
    fetch_fundamentals,
    fetch_index_data,
)
from ..api.providers import TIMEOUT as _TIMEOUT
from ..api.providers import call_with_timeout as _call_with_timeout
from ..shared.constants import (
    DIRECTIONAL_TREND_FILTERS,
    ENRICH_OVERALL_TIMEOUT,
    ENRICH_PROVIDER_TIMEOUT,
    ENRICH_TOP_N,
    LARGE_UNIVERSE_THRESHOLD,
    POOR_RATINGS,
    SPARK_BARS,
    STALE_MEMBER_MAX_AGE_DAYS,
    TICKER_TIMEOUT,
)
from ..shared.trace import trace
from ..shared.universes import (
    SUSPENDED_OR_DELISTED,
    UNIVERSES,
    get_universe,
    strip_dead_members,
)
from .scoring import (
    bullish_candle_pattern,
    check_filter,
    compute_scores,
    get_direction,
)
from .settings_store import get_api_key, load_api_config

logger = logging.getLogger(__name__)


def _find_stale_members(batch_data: dict, max_age_days: float | None = None):
    """Universe members whose latest bar is older than ``max_age_days``.

    Suspended/delisted names (e.g. GSPL, halted May 2026; TATAMETALI, merged
    into Tata Steel in 2024) still occupy universe lists and are re-fetched on
    every scan, but their data ends long before today.  Returns
    ``[(ticker, 'YYYY-MM-DD'), ...]`` sorted oldest-first.
    """
    from datetime import timedelta

    if not batch_data:
        return []
    days = max_age_days if max_age_days is not None else STALE_MEMBER_MAX_AGE_DAYS
    cutoff = datetime.now().date() - timedelta(days=days)
    stale = []
    for t, df in batch_data.items():
        try:
            last = df.index[-1]
            if not isinstance(last, datetime):
                import pandas as pd

                last = pd.Timestamp(last)
            if last.date() < cutoff:
                stale.append((str(t), last.date().isoformat()))
        except (TypeError, ValueError, AttributeError) as e:
            logger.debug("Stale check failed for %s: %s", t, e)
            continue
    return sorted(stale, key=lambda x: x[1])


def _stale_members_message(stale_members: list, max_age_days: int | None = None) -> str:
    """Human-readable warning for a stale-member list (oldest first)."""
    days = max_age_days if max_age_days is not None else STALE_MEMBER_MAX_AGE_DAYS
    names = ", ".join(f"{t} ({d})" for t, d in stale_members[:5])
    if len(stale_members) > 5:
        names += f" +{len(stale_members) - 5} more"
    return (
        f"{len(stale_members)} universe member(s) have stale data "
        f"(last bar > {days}d old): {names} — "
        "suspended/delisted? They are fetched each scan but never trade "
        "the current window."
    )


def rating_ok_for_trend_filter(trend_filter: str, combined_rating: str | None) -> bool:
    """Whether a stock with `combined_rating` may be shown under `trend_filter`.

    Non-directional filters ("All") show every rating; directional filters
    drop POOR/WEAK-rated stocks. Missing/unknown ratings are kept — never
    hide data on a defensive default.
    """
    if trend_filter not in DIRECTIONAL_TREND_FILTERS:
        return True
    return (combined_rating or "") not in POOR_RATINGS


def _score_ticker(
    ticker: str,
    df: Any,
    *,
    settings: dict,
    timeframe: str,
    index_df: Any,
    trend_filter: str,
    is_large: bool,
    enrich: Callable,
    use_enrichment_cache: bool = True,
) -> tuple[dict | None, str]:
    """Score one ticker — shared by scan() and scan_stream().

    Runs the crossover filter, applies the trend-direction filter, attaches
    fundamentals for small universes, computes the category score and applies
    the rating gate. Returns ``(scores, direction)`` on success, or
    ``(None, reason)`` where reason is one of: empty / filtered / no_score /
    poor_rating / error.

    ``use_enrichment_cache`` gates cache *reads* on the small-universe path
    (callers whose enrich callable is a no-op, like the CLI small loop, pass
    False to keep output independent of cache state; writes still occur but
    are no-ops when enrich yields no provider keys).
    """
    try:
        if df is None or df.empty:
            return None, "empty"
        filter_result = check_filter(
            df,
            fast_ma_type=settings.get("fast_ma_type", "HMA"),
            fast_ma_len=settings.get("fast_ma_len", 40),
            slow_ma_type=settings.get("slow_ma_type", "EMA"),
            slow_ma_len=settings.get("slow_ma_len", 50),
            crossover_lookback=settings.get("crossover_lookback", 20),
            settings=settings,
        )
        if filter_result is None:
            return None, "filtered"
        direction = get_direction(filter_result)
        if trend_filter in ("Bullish Only", "Bullish + Candle") and direction != "Bull":
            return None, "filtered"
        if trend_filter == "Bearish Only" and direction != "Bear":
            return None, "filtered"
        if trend_filter == "Bullish + Candle" and bullish_candle_pattern(df) is None:
            return None, "filtered"
        # Skip fundamentals/enrichment in fast mode phase 1 (large universes
        # attach them later in the top-200 pass and re-score there).
        if is_large:
            enriched = {**settings, "_skip_vp": True}
        else:
            enriched = _enrich_small_cached(
                ticker,
                df,
                settings,
                enrich,
                use_cache=use_enrichment_cache,
            )
            enriched["_skip_vp"] = False
        scores = compute_scores(
            df, timeframe=timeframe, index_df=index_df, settings=enriched
        )
        if scores is None:
            return None, "no_score"
        if not rating_ok_for_trend_filter(trend_filter, scores.get("combined_rating")):
            return None, "poor_rating"
        scores["ticker"] = ticker
        scores["trend_dir"] = direction
        scores["trend_color"] = direction.lower()
        # Mini sparkline data — the last ~1 month of closes on the scan's
        # timeframe. Attached once here; the fast-mode re-score path only
        # updates existing keys, so the tail survives the top-200 pass.
        try:
            if "close" in df.columns:
                closes = df["close"].dropna()
                px = [float(x) for x in closes.tail(SPARK_BARS)]
                if len(px) >= 2:
                    scores["px_tail"] = px
        except (TypeError, ValueError):
            logger.debug("Sparkline px_tail build failed for %s", ticker, exc_info=True)
        # Keep enrichment keys for later if not large
        if not is_large:
            for k, v in enriched.items():
                if k.startswith("_") and k not in scores:
                    scores[k] = v
        return scores, direction
    except Exception as e:
        logger.info("Scoring failed for %s: %s", ticker, e)
        return None, "error"


# Defaults live enrichment sets unconditionally for these keys (mirrors
# _enrich_with_providers); used to backfill cache entries that predate a
# provider so older rows score identically to fresh live enrichment.
_PROVIDER_DEFAULTS = {
    "_sentiment_score": 0.0,
    "_article_count": 0,
    "_sentiment_source": "none",
    "_insider_score": 0.0,
    "_insider_source": "none",
}


def _enrich_small_cached(ticker, df, settings, enrich, use_cache=True):
    """Enrichment-cache read-through for the small-universe (full) path.

    A hit skips ~3s of provider calls per ticker: cached fundamentals are
    attached to ``df`` and cached provider keys are merged over settings.
    Missing keys get the same defaults live enrichment would set. A miss
    runs the live path verbatim and populates the cache, mirroring the
    phase-2 writer (same key extraction, same skip-empty guard).
    """
    if use_cache:
        cached = enrichment_get(ticker)
        if cached is not None:
            providers = dict(cached.get("providers") or {})
            for key, default in _PROVIDER_DEFAULTS.items():
                if key not in providers:
                    providers[key] = default
            fund = cached.get("fundamentals")
            if fund is not None:
                df.attrs["_fundamentals"] = fund
            return {**settings, **providers}
    if df.attrs.get("_fundamentals") is None:
        try:
            fund = fetch_fundamentals(ticker)
            if fund is not None:
                df.attrs["_fundamentals"] = fund
        except (RequestException, ValueError, KeyError) as e:
            logger.debug("Fundamentals fetch failed for %s: %s", ticker, e)
    enriched = enrich(ticker, settings)
    provider_keys = {
        k: v for k, v in enriched.items() if k.startswith("_") and k not in settings
    }
    enrichment_put(ticker, provider_keys, df.attrs.get("_fundamentals"))
    return enriched


def _guarded_score(score_fn, item):
    """Run one scoring unit, converting worker crashes to (None, "error")."""
    try:
        return score_fn(item)
    except Exception as e:
        ticker = item[0] if item else "?"
        logger.info("Scoring failed for %s: %s", ticker, e)
        return None, "error"


def _parallel_score(items, score_fn, cancel_event, max_workers=8):
    """Score ``(ticker, df)`` items concurrently, preserving input order.

    Network-bound provider work scales ~linearly with workers; CPU-bound
    pandas scoring benefits where numpy releases the GIL. Returns
    ``(ordered, cancelled)`` where ``ordered`` maps input index ->
    ``(scores, direction)`` for every item that finished, and ``cancelled``
    mirrors ``cancel_event``. Cancel is polled every 0.5s (same cadence as
    the large-universe loop) so Stop stays responsive; items that never ran
    are simply absent from ``ordered``.
    """
    from concurrent.futures import FIRST_COMPLETED, wait

    n = len(items)
    ordered: dict[int, tuple] = {}
    if n == 0:
        return ordered, False
    executor = _DaemonThreadPoolExecutor(max_workers=max(1, min(max_workers, n)))
    futs = {
        executor.submit(_guarded_score, score_fn, item): i
        for i, item in enumerate(items)
    }
    pending = set(futs)
    cancelled = False
    while pending:
        if cancel_event is not None and cancel_event.is_set():
            executor.shutdown(wait=False, cancel_futures=True)
            cancelled = True
            break
        completed, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
        for fut in completed:
            try:
                ordered[futs[fut]] = fut.result()
            except Exception as e:  # e.g. CancelledError after a late cancel
                logger.info("Parallel scoring worker failed: %s", e)
                ordered[futs[fut]] = (None, "error")
    if not cancelled:
        executor.shutdown(wait=True)
    return ordered, cancelled


def _tally(scores, direction, results, counts) -> tuple[int, int]:
    """Account one (scores, direction) pair. Returns (filtered, poor) increments."""
    if scores is None:
        return (direction == "filtered"), (direction == "poor_rating")
    counts[direction] = counts.get(direction, 0) + 1
    results.append(scores)
    return 0, 0


def _make_scorer(
    settings, timeframe, index_df, trend_filter, is_large, enrich, use_cache=True
):
    """Phase-1 per-ticker scorer shared by scan()/scan_stream()/the CLI."""

    def _score_one(item):
        ticker, df = item
        return _score_ticker(
            ticker,
            df,
            settings=settings,
            timeframe=timeframe,
            index_df=index_df,
            trend_filter=trend_filter,
            is_large=is_large,
            enrich=enrich,
            use_enrichment_cache=use_cache,
        )

    return _score_one


def _enrich_top(
    results,
    batch_data,
    settings,
    timeframe,
    index_df,
    enrich,
    cancel_event=None,
    progress=None,
    log=None,
):
    """Phase-2 body shared by scan()/scan_stream()/CLI.

    Sorts, enriches the global top-200 in place, and returns
    ``(enriched_top, merged_results)``. ``_finalize_scan`` (or the CLI)
    re-sorts the merged list afterwards because enrichment moves scores.
    """
    results.sort(key=lambda x: x.get("total", 0) or 0, reverse=True)
    top_n = min(ENRICH_TOP_N, len(results))
    top = results[:top_n]
    rest = results[top_n:]
    if progress:
        progress(0.82, f"Enriching top {top_n} stocks...")
    if log:
        log(f"Enriching top {top_n} of {len(results)} with fundamentals/sentiment...")
    enriched_top = _enrich_rows_in_place(
        top,
        batch_data,
        settings=settings,
        timeframe=timeframe,
        index_df=index_df,
        enrich=enrich,
        cancel_event=cancel_event,
        progress_callback=progress,
    )
    return enriched_top, enriched_top + rest


def _log_scan_outcome(
    result, universe, elapsed, tickers, settings, prefix, cancel_event=None
) -> None:
    """Shared wind-down + completion logging for scan()/scan_stream().

    A stop requested while the loop was between/inside batches (generator
    stop or late poll exit, not a loop-top check) must always surface as
    cancelled.
    """
    if cancel_event is not None and cancel_event.is_set():
        result.cancelled = True
    if result.cancelled:
        logger.info(
            "%s_cancelled: universe=%s elapsed=%.1fs results=%d",
            prefix,
            universe,
            elapsed,
            len(result.results),
        )
    elif result.error:
        logger.info(
            "%s_error: universe=%s elapsed=%.1fs error=%s",
            prefix,
            universe,
            elapsed,
            result.error,
        )
    else:
        logger.info(
            "%s_completed: universe=%s elapsed=%.1fs tickers=%d results=%d "
            "passed=%d filtered=%d bull=%d bear=%d",
            prefix,
            universe,
            elapsed,
            len(tickers),
            len(result.results),
            len(
                [
                    r
                    for r in result.results
                    if r["total"] >= settings.get("min_score", 50)
                ]
            ),
            result.filtered_out,
            result.direction_counts.get("Bull", 0),
            result.direction_counts.get("Bear", 0),
        )


def _enrich_rows_in_place(
    rows: list[dict],
    batch_data: dict,
    *,
    settings: dict,
    timeframe: str,
    index_df: Any,
    enrich: Callable,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[dict]:
    """Phase-2 body shared by scan()/scan_stream(): enrich rows, then re-score.

    Runs the per-ticker provider enrichment on each row, attaches fundamentals
    to the source DataFrame if missing, and recomputes the score so totals /
    ratings reflect the fundamentals that phase-1 fast mode skipped. Mutates
    each row dict in place ("_"-prefixed provider keys are added) and returns
    the enriched rows in the same order. When `cancel_event` is set, returns
    promptly with whatever rows finished so far — unfinished rows keep their
    phase-1 scores (daemon workers finish in the background).
    """
    from concurrent.futures import FIRST_COMPLETED, wait

    # Per-ticker hard ceiling — prevents one hung NSE server from blocking
    # the entire enrichment for minutes.  Each ticker runs 5 provider calls
    # (15 s each via ENRICH_PROVIDER_TIMEOUT) + fundamentals + re-score,
    # so 60 s gives ample headroom while still bounding worst-case stalls.

    # Shared executor for the 5 per-ticker provider calls — reused across
    # all enrichment workers to avoid creating/destroying an executor per ticker.
    _provider_executor = _DaemonThreadPoolExecutor(max_workers=5)

    def _enrich_one(r):
        ticker = r["ticker"]
        provider_keys: dict = {}
        try:
            # Provider results barely change intraday — replay the disk
            # cache on repeat scans instead of re-running the 5 parallel
            # provider fetches per ticker (the dominant full-market cost).
            cached = enrichment_get(ticker)
            if cached is not None:
                record_enrichment_hit()
                enriched = dict(settings)
                enriched.update(cached.get("providers") or {})
            else:
                record_enrichment_miss()
                try:
                    enriched = enrich(ticker, settings, executor=_provider_executor)
                except TypeError:
                    enriched = enrich(ticker, settings)
                provider_keys = {
                    k: v
                    for k, v in enriched.items()
                    if k.startswith("_") and k not in settings
                }
            for k, v in enriched.items():
                if k.startswith("_") and k not in r:
                    r[k] = v
            # Phase-1 fast mode scored WITHOUT fundamentals — attach
            # them now and recompute so totals/ratings reflect real data.
            df = batch_data.get(ticker)
            if df is not None and not df.empty:
                if df.attrs.get("_fundamentals") is None:
                    fund = None
                    try:
                        if cached is not None:
                            fund = cached.get(
                                "fundamentals"
                            )  # may be None = known-none
                        else:
                            fund = _call_with_timeout(
                                lambda: fetch_fundamentals(ticker),
                                timeout=TICKER_TIMEOUT,
                            )
                            if fund is not None and fund is not _TIMEOUT:
                                if provider_keys:
                                    enrichment_put(ticker, provider_keys, fund)
                        if fund is not None and fund is not _TIMEOUT:
                            df.attrs["_fundamentals"] = fund
                    except (RequestException, ValueError, KeyError) as e:
                        logger.debug("Fundamentals fetch failed for %s: %s", ticker, e)
                try:
                    recomputed = _call_with_timeout(
                        lambda: compute_scores(
                            df,
                            timeframe=timeframe,
                            index_df=index_df,
                            settings={
                                **settings,
                                "_skip_vp": True,
                                # Provider "_"-keys (flows, delivery, …) so
                                # score_bar's institutional bonus applies in
                                # phase-2 like fundamentals do via df.attrs.
                                **{
                                    k: v
                                    for k, v in enriched.items()
                                    if k.startswith("_")
                                },
                            },
                        ),
                        timeout=TICKER_TIMEOUT,
                    )
                except Exception:
                    logger.info("Re-score failed for %s", ticker, exc_info=True)
                    recomputed = None
                if recomputed is not None and recomputed is not _TIMEOUT:
                    r.update(recomputed)
        except Exception as e:
            logger.info("Top enrichment failed for %s: %s", ticker, e)
        return r

    import time as _time

    reset_enrichment_counts()
    total = len(rows)
    done_count = 0
    executor = _DaemonThreadPoolExecutor(max_workers=8)
    future_to_row = {executor.submit(_enrich_one, r): r for r in rows}
    cancelled = False
    deadline = _time.monotonic() + ENRICH_OVERALL_TIMEOUT
    pending = set(future_to_row)
    while pending:
        if cancel_event is not None and cancel_event.is_set():
            executor.shutdown(wait=False, cancel_futures=True)
            cancelled = True
            break
        if _time.monotonic() > deadline:
            logger.warning(
                "Phase-2 enrichment exceeded %ds deadline — "
                "returning %d/%d enriched rows",
                ENRICH_OVERALL_TIMEOUT,
                done_count,
                total,
            )
            executor.shutdown(wait=False, cancel_futures=True)
            break
        # Poll every 0.5s instead of blocking on executor.map — lets Stop
        # interrupt even this phase.
        completed, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
        if not completed:
            continue
        for fut in completed:
            fut.result()  # _enrich_one already caught its own exceptions
            done_count += 1
        if progress_callback:
            try:
                frac = 0.85 + 0.15 * (done_count / max(total, 1))
                progress_callback(
                    min(frac, 0.99),
                    f"Enriching {done_count}/{total} with fundamentals…",
                )
            except Exception:
                logger.info("Enrichment progress callback failed", exc_info=True)
    if not cancelled:
        executor.shutdown(wait=True)
    _provider_executor.shutdown(wait=False)
    stats = enrichment_stats()
    hits, misses = stats["hits"], stats["misses"]
    if hits or misses:
        logger.info(
            "Enrichment cache: %d/%d rows served from cache",
            hits,
            hits + misses,
        )
    if progress_callback:
        try:
            progress_callback(0.99, "Finalizing scan…")
        except Exception:
            logger.info("Finalizing progress callback failed", exc_info=True)
    return rows  # mutated in place; unfinished rows keep phase-1 scores


class ScanResult:
    """Container for scan results."""

    def __init__(self):
        self.results: list[dict[str, Any]] = []
        self.filtered_out = 0
        self.direction_counts = {"Bull": 0, "Bear": 0}
        self.cancelled = False
        self.error: str | None = None
        self.warnings: list[str] = []


def _build_scan_warnings(
    settings: dict,
    total: int,
    results: list,
    passed: int,
    min_score_default: float = 50.0,
) -> list[str]:
    """Warn when the crossover filter is broad but the candidate set is weak.

    A large passed set with few scored / entry names is the signature of a
    loose MA configuration (e.g. HMA20xEMA40): it screens breadth, and the
    backtests show such filters trade far worse than the default
    40x50 set.  Returns human-readable warnings (empty when the scan is
    healthy or too small to judge).
    """
    warnings = []
    if not results or total <= 0:
        return warnings
    min_score = float(settings.get("min_score", min_score_default))
    entry_ct = len([r for r in results if r.get("entry_signal")])
    ratio = len(results) / total
    fl = settings.get("fast_ma_len", "?")
    sl = settings.get("slow_ma_len", "?")
    if ratio < 0.30 or len(results) < 25:
        return warnings
    if entry_ct == 0:
        warnings.append(
            f"{len(results)}/{total} stocks pass the crossover filter but NONE has a valid "
            f"entry signal (only {passed} score {min_score:.0f}+). HMA{fl}xEMA{sl} is screening "
            "breadth, not tradeable signals."
        )
    elif entry_ct < 5 or passed < max(5, int(0.15 * len(results))):
        warnings.append(
            f"{len(results)}/{total} stocks pass the crossover filter but only {entry_ct} have "
            f"entry signals and {passed} score {min_score:.0f}+. A broad MA filter "
            f"(HMA{fl}xEMA{sl}) mostly surfaces weak setups — validate with a "
            "backtest before acting."
        )
    return warnings


class ScannerEngine:
    """
    Headless scanner engine for HMA/EMA multi-score swing strategy.

    Usage:
        engine = ScannerEngine()
        progress_cb = lambda p, t: print(f"{p:.0%}: {t}")
        log_cb = lambda msg: print(msg)
        result = engine.scan(universe="NIFTY 50", settings=settings,
                            progress_callback=progress_cb, log_callback=log_cb)
    """

    def __init__(self):
        self._cancel_event = threading.Event()
        self._progress_callback: Callable[[float, str], None] | None = None
        self._log_callback: Callable[[str], None] | None = None

    def cancel(self):
        """Signal the scan to cancel."""
        self._cancel_event.set()

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates: callback(progress: 0.0-1.0, text: str)"""
        self._progress_callback = callback

    def set_log_callback(self, callback: Callable[[str], None]):
        """Set callback for log messages: callback(message: str)"""
        self._log_callback = callback

    def _progress(self, value: float, text: str = ""):
        if self._progress_callback:
            try:
                self._progress_callback(value, text)
            except Exception as e:
                # A failing progress sink must never kill the scan.
                logger.info("Progress callback failed: %s", e)

    def _log(self, msg: str):
        if self._log_callback:
            try:
                self._log_callback(msg)
            except Exception as e:
                # A failing log sink (e.g. a console that can't encode ✓) must
                # never abort the scan and discard already-scored results.
                logger.debug("Log callback failed: %s", e)

    def _prepare_scan(
        self,
        universe: str,
        settings: dict,
        period: str,
        timeframe: str,
        trend_filter: str,
        index_symbol: str,
        scan_label: str = "SCAN",
    ) -> tuple[list[str], Any, bool]:
        """Shared setup for scan() and scan_stream().

        Resolves the universe, strips dead members, logs the header,
        fetches the index and caches API keys.

        Returns ``(tickers, index_df, is_large)``.
        """
        try:
            tickers = get_universe(universe)
        except KeyError:
            tickers = UNIVERSES.get(universe, [])

        tickers, dead_members = strip_dead_members(tickers)
        if dead_members:
            self._log(
                f"Skipping {len(dead_members)} suspended/delisted member(s): "
                + ", ".join(f"{t} ({SUSPENDED_OR_DELISTED[t]})" for t in dead_members),
            )

        tf_names = {"D": "Daily", "W": "Weekly", "M": "Monthly"}

        self._log("\n" + "=" * 50)
        self._log(
            f"START {scan_label} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._log("=" * 50)
        self._log(f"Starting scan: {universe} ({len(tickers)} stocks)")
        self._log(
            f"Timeframe: {tf_names.get(timeframe, timeframe)} | Period: {period} | Filter: {trend_filter}"
        )
        self._log(
            f"FastMA={settings.get('fast_ma_type', 'HMA')}{settings.get('fast_ma_len', 40)} "
            f"SlowMA={settings.get('slow_ma_type', 'EMA')}{settings.get('slow_ma_len', 50)} "
            f"RSI={settings.get('rsi_len', 14)} Threshold={settings.get('min_score', 50)}"
        )

        self._progress(0.0, f"Fetching {index_symbol} index...")
        index_df = fetch_index_data(f"^{index_symbol}", period=period)
        if index_df is not None:
            self._log(f"{index_symbol} index loaded ({len(index_df)} bars)")
        else:
            self._log(f"Warning: {index_symbol} index unavailable, using proxy for RS")

        is_large = len(tickers) > LARGE_UNIVERSE_THRESHOLD
        if is_large:
            self._log(
                f"Large universe fast mode: {len(tickers)} stocks — technicals first, enrich top {ENRICH_TOP_N} only"
            )

        return tickers, index_df, is_large

    def _finalize_scan(
        self,
        result: ScanResult,
        settings: dict,
        tickers: list[str],
        results: list[dict],
        filtered_out: int,
        poor_rating_hidden: int,
        direction_counts: dict,
        trend_filter: str,
        batch_data: dict | None = None,
        scan_label: str = "Scan Complete",
    ) -> None:
        """Shared teardown for scan() and scan_stream().

        Sorts results, populates the ScanResult, logs the summary, and
        attaches warnings (broad-filter, stale members).
        """
        results.sort(key=lambda x: x.get("total", 0) or 0, reverse=True)
        passed = len(
            [r for r in results if r["total"] >= settings.get("min_score", 50)]
        )

        result.results = results
        result.filtered_out = filtered_out
        result.direction_counts = direction_counts

        self._log("\n\u2501" * 25 + f" {scan_label} ")
        if batch_data is not None:
            missing_final = [t for t in tickers if t not in batch_data]
            self._log(f"  Total tickers: {len(tickers)} | fetched: {len(batch_data)}")
            if missing_final:
                self._log(
                    f"  \u26a0 {len(missing_final)} tickers unavailable on all providers "
                    "(yfinance, jugaad-data, nselib)"
                )
        else:
            self._log(f"  Total stocks:  {len(tickers)}")
        self._log(f"  Filtered out:  {filtered_out}")
        if poor_rating_hidden:
            self._log(
                f"  Poor-rated hidden: {poor_rating_hidden} ({trend_filter} filter)"
            )
        neg_skips = negative_skip_count()
        if neg_skips:
            self._log(f"  Dead-symbols skipped (negative cache): {neg_skips}")
        self._log(
            f"  Passed filter: {len(results)} ({direction_counts.get('Bull', 0)} Bull, {direction_counts.get('Bear', 0)} Bear)"
        )
        self._log(f"  Scored {settings.get('min_score', 50)}+: {passed}")

        if not result.cancelled:
            result.warnings = _build_scan_warnings(
                settings, len(tickers), results, passed
            )
            stale_days = float(
                settings.get("stale_member_max_age_days") or STALE_MEMBER_MAX_AGE_DAYS
            )
            stale_members = _find_stale_members(
                batch_data or {}, max_age_days=stale_days
            )
            if stale_members:
                result.warnings.append(
                    _stale_members_message(stale_members, max_age_days=stale_days)
                )
            for w in result.warnings:
                self._log(f"  \u26a0 {w}")

        stats = enrichment_stats()
        neg = negative_skip_count()
        if stats["hits"] or stats["misses"] or neg:
            total_reqs = stats["hits"] + stats["misses"]
            hit_pct = (stats["hits"] / total_reqs * 100) if total_reqs else 0
            self._log(
                f"  Cache: enrichment {stats['hits']}/{total_reqs} hits ({hit_pct:.0f}%)"
                f" | negative skips: {neg}"
            )
            logger.info(
                "cache_stats: enrichment_hits=%d enrichment_misses=%d "
                "negative_skips=%d price_cache_size=%d",
                stats["hits"],
                stats["misses"],
                neg,
                enrichment_cache_size(),
            )

    def _enrich_with_providers(
        self,
        ticker: str,
        settings: dict,
        executor=None,
    ) -> dict:
        """
        Enrich settings with data from provider modules.
        Runs all 4 per-ticker providers in parallel via ThreadPoolExecutor.
        If an executor is provided, reuses it; otherwise creates a temporary one.
        """
        from concurrent.futures import as_completed

        enriched = settings.copy()

        # Load API keys (cached per scan, not per-ticker)
        api_config = load_api_config()

        # Define per-ticker provider functions
        def _fetch_sentiment():
            from ..api.market_sentiment import fetch_sentiment

            return fetch_sentiment(
                ticker,
                marketaux_key=get_api_key("MARKETAUX_API_KEY", api_config),
                newsapi_key=get_api_key("NEWS_API_KEY", api_config),
                gnews_key=get_api_key("GNEWS_API_KEY", api_config),
            )

        def _fetch_indian_market():
            from ..api.indian_market import fetch_indian_market_data

            return fetch_indian_market_data(ticker)

        def _fetch_indian_fundamentals():
            from ..api.indian_fundamentals import fetch_indian_fundamentals

            return fetch_indian_fundamentals(ticker)

        def _fetch_insider():
            from ..api.insider_data import fetch_insider_data

            return fetch_insider_data(
                ticker,
                aletheia_key=get_api_key("ALETHEIA_API_KEY", api_config),
                congress_key=get_api_key("CONGRESS_API_KEY", api_config),
            )

        # Launch all 4 providers in parallel
        futures = {}
        _own_executor = executor is None
        if _own_executor:
            executor = ThreadPoolExecutor(max_workers=4)
        try:
            futures[executor.submit(_fetch_sentiment)] = "sentiment"
            futures[executor.submit(_fetch_indian_market)] = "india"
            futures[executor.submit(_fetch_indian_fundamentals)] = "india_fund"
            futures[executor.submit(_fetch_insider)] = "insider"

            try:
                for future in as_completed(futures, timeout=ENRICH_PROVIDER_TIMEOUT):
                    category = futures[future]
                    try:
                        result = future.result(timeout=5)
                        if category == "sentiment":
                            enriched["_sentiment_score"] = result.get(
                                "sentiment_score", 0.0
                            )
                            enriched["_article_count"] = result.get("article_count", 0)
                            enriched["_sentiment_source"] = result.get("source", "none")
                        elif category == "india":
                            delivery = result.get("delivery")
                            if delivery:
                                enriched["_delivery_pct"] = delivery.delivery_pct
                                enriched["_delivery_change_pct"] = (
                                    delivery.delivery_change_pct
                                )
                                enriched["_delivery_source"] = "nse"
                            fii_dii = result.get("fii_dii")
                            if fii_dii:
                                enriched["_fii_is_buying"] = fii_dii.fii_is_buying
                                enriched["_dii_is_buying"] = fii_dii.dii_is_buying
                                enriched["_fii_net"] = fii_dii.fii_net
                                enriched["_dii_net"] = fii_dii.dii_net
                                enriched["_institutional_source"] = "nse"
                        elif category == "india_fund":
                            trendlyne = result.get("trendlyne")
                            if trendlyne and trendlyne.promoter_holding:
                                enriched["_promoter_holding"] = round(
                                    float(trendlyne.promoter_holding), 2
                                )
                            screener = result.get("screener")
                            if screener:
                                if screener.industry_pe and screener.stock_pe:
                                    enriched["_pe_relative_to_industry"] = (
                                        screener.stock_pe / screener.industry_pe
                                    )
                                enriched["_is_quality_stock"] = screener.is_quality
                                enriched["_valuation_source"] = result.get(
                                    "source", "none"
                                )
                        elif category == "insider":
                            enriched["_insider_score"] = result.get(
                                "insider_score", 0.0
                            )
                            enriched["_insider_source"] = result.get("source", "none")
                    except (
                        RequestException,
                        ValueError,
                        KeyError,
                        AttributeError,
                        TimeoutError,
                    ) as e:
                        logger.debug(
                            "Provider %s failed for %s: %s", category, ticker, e
                        )
            except TimeoutError:
                logger.debug(
                    "Provider enrichment timed out for %s after %ds",
                    ticker,
                    ENRICH_PROVIDER_TIMEOUT,
                )
        finally:
            if _own_executor:
                executor.shutdown(wait=False)

        return enriched

    @trace(level=logging.INFO, log_args=True)
    def scan(
        self,
        universe: str,
        settings: dict,
        period: str = "1y",
        timeframe: str = "D",
        trend_filter: str = "All",
        index_symbol: str = "NSEI",
    ) -> ScanResult:
        """
        Run a full scan.

        Args:
            universe: Universe name (key in UNIVERSES or dynamic universe)
            settings: Scanner settings dict
            period: Data period (e.g., "1y", "3y")
            timeframe: "D" (daily), "W" (weekly), "M" (monthly)
            trend_filter: "All", "Bullish Only", "Bearish Only",
                  "Bullish + Candle" (bullish crossover + latest-bar
                  bullish-engulfing/hammer). Directional views also hide
                  POOR/WEAK-rated stocks.
            index_symbol: Index symbol for relative strength (e.g., "NSEI")

        Returns:
            ScanResult with results list and metadata
        """
        self._cancel_event.clear()
        result = ScanResult()
        tickers: list[str] = []  # bound before try so the outcome log never NameErrors
        import time as _time

        t_start = _time.monotonic()
        logger.info(
            "scan_started: universe=%s period=%s timeframe=%s trend=%s min_score=%s",
            universe,
            period,
            timeframe,
            trend_filter,
            settings.get("min_score", 50),
        )

        try:
            tickers, index_df, is_large = self._prepare_scan(
                universe,
                settings,
                period,
                timeframe,
                trend_filter,
                index_symbol,
                scan_label="SCAN",
            )

            # Batch download all stocks — yfinance chunks, then a per-ticker
            # fallback pass (jugaad-data/nselib) for anything yfinance missed.
            self._progress(0.05, f"Batch downloading {len(tickers)} stocks...")
            self._log(
                f"Batch downloading {len(tickers)} stocks via yfinance (fallback: jugaad-data/nselib)..."
            )
            reset_negative_skips()
            batch_data = fetch_batch_yfinance(
                tickers,
                period=period,
                timeframe=timeframe,
                cancel_event=self._cancel_event,
                on_fallback_progress=lambda done, total: self._progress(
                    0.05 + 0.05 * (done / max(total, 1)),
                    f"Fallback fetch {done}/{total} (jugaad/nselib)",
                ),
            )
            if len(batch_data) < len(tickers):
                self._log(
                    f"Batch download complete: {len(batch_data)}/{len(tickers)} stocks fetched "
                    f"({len(tickers) - len(batch_data)} unavailable on all providers)"
                )
            else:
                self._log(
                    f"Batch download complete: {len(batch_data)}/{len(tickers)} stocks fetched"
                )

            results = []
            total = len(batch_data)
            filtered_out = 0
            poor_rating_hidden = 0
            direction_counts = {"Bull": 0, "Bear": 0}

            # ── Phase 1: Fast technical scoring (no enrichment) ────────────
            # Use ThreadPool for CPU-bound scoring on large universes
            _score_one = _make_scorer(
                settings,
                timeframe,
                index_df,
                trend_filter,
                is_large,
                self._enrich_with_providers,
            )

            if is_large and total > 200:
                # Parallel scoring for large universes — poll cancel every 0.5s
                from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

                max_workers = min(8, (total // 50) + 2)
                self._log(
                    f"Parallel scoring {total} stocks with {max_workers} workers..."
                )
                executor = ThreadPoolExecutor(max_workers=max_workers)
                future_to_ticker = {
                    executor.submit(_score_one, item): item[0]
                    for item in batch_data.items()
                }
                cancelled_loop = False
                pending = set(future_to_ticker)
                done = 0
                while pending:
                    if self._cancel_event.is_set():
                        executor.shutdown(wait=False, cancel_futures=True)
                        cancelled_loop = True
                        result.cancelled = True
                        break
                    completed, pending = wait(
                        pending, timeout=0.5, return_when=FIRST_COMPLETED
                    )
                    if not completed:
                        continue
                    for future in completed:
                        done += 1
                        if done % 50 == 0 or done <= 5:
                            self._progress(
                                0.1 + (done / total * 0.7), f"Scoring {done}/{total}"
                            )
                        scores, direction = future.result()
                        f, p = _tally(scores, direction, results, direction_counts)
                        filtered_out += f
                        poor_rating_hidden += p
                        if scores is None:
                            continue
                        if len(results) % 20 == 0 or len(results) <= 5:
                            tag = (
                                "\u2713"
                                if scores["total"] >= settings.get("min_score", 50)
                                else "\u2717"
                            )
                            self._log(
                                f"  {tag} {scores['ticker']}: {scores['total']:.1f}/100 ({direction})"
                            )
                if not cancelled_loop:
                    executor.shutdown(wait=True)
            else:
                # Small universes run full per-ticker enrichment (~3s of
                # provider calls each) — parallelize across tickers the way
                # phase 2 does. Input order is preserved for the logs below.
                items = list(batch_data.items())
                ordered, _pcancelled = _parallel_score(
                    items, _score_one, self._cancel_event
                )
                if _pcancelled:
                    self._log("\n\u23f9  Scan cancelled by user")
                    result.cancelled = True
                done = 0
                for i, (ticker, _df) in enumerate(items, 1):
                    res = ordered.get(i - 1)
                    if res is None:
                        continue  # cancelled before this item ran
                    scores, direction = res
                    done += 1
                    progress = 0.1 + (done / total * 0.9) if total > 0 else 0.5
                    self._progress(progress, f"[{done}/{total}] {ticker}")

                    f, p = _tally(scores, direction, results, direction_counts)
                    filtered_out += f
                    poor_rating_hidden += p
                    if scores is None:
                        continue

                    if len(results) % 10 == 0 or len(results) <= 5:
                        score_val = scores["total"]
                        tag = (
                            "\u2713"
                            if score_val >= settings.get("min_score", 50)
                            else "\u2717"
                        )
                        self._log(
                            f"  {tag} {ticker}: {score_val:.1f}/100 ({direction})"
                        )

            # ── Phase 2: Enrich top 200 for large universes ───────────────
            if is_large and results and not self._cancel_event.is_set():
                _, results = _enrich_top(
                    results,
                    batch_data,
                    settings,
                    timeframe,
                    index_df,
                    self._enrich_with_providers,
                    cancel_event=self._cancel_event,
                    progress=self._progress,
                    log=self._log,
                )

            self._progress(0.95, "Finalizing scan...")

            self._finalize_scan(
                result,
                settings,
                tickers,
                results,
                filtered_out,
                poor_rating_hidden,
                direction_counts,
                trend_filter,
                batch_data=batch_data,
                scan_label="Scan Complete",
            )

        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            self._log(f"\nERROR: {type(e).__name__}: {e}")
            logger.exception("Scan failed")

        elapsed = _time.monotonic() - t_start
        _log_scan_outcome(
            result, universe, elapsed, tickers, settings, "scan", self._cancel_event
        )
        return result

    @trace(level=logging.INFO, log_args=True)
    def scan_stream(
        self,
        universe: str,
        settings: dict,
        period: str = "1y",
        timeframe: str = "D",
        trend_filter: str = "All",
        index_symbol: str = "NSEI",
        on_batch: Callable[[list[dict]], None] | None = None,
    ) -> ScanResult:
        """
        Streaming scan — yields results batch-by-batch to the grid.

        Instead of fetching all ~5900 then scoring, we stream via
        fetch_batch_yfinance_stream: each parallel batch (~200-1000 tickers)
        is scored immediately and on_batch(chunk_results) is called so the
        UI can append to the table without waiting.

        For large universes (>500) we stream technical-only scores per chunk,
        then after all chunks enrich the global top-200 and emit an update
        batch (existing rows are replaced in-place in the UI).
        """
        batch_cb = on_batch
        self._cancel_event.clear()
        result = ScanResult()
        tickers: list[str] = []  # bound before try so the outcome log never NameErrors
        import time as _time

        t_start = _time.monotonic()
        logger.info(
            "stream_scan_started: universe=%s period=%s timeframe=%s trend=%s min_score=%s",
            universe,
            period,
            timeframe,
            trend_filter,
            settings.get("min_score", 50),
        )

        try:
            tickers, index_df, is_large = self._prepare_scan(
                universe,
                settings,
                period,
                timeframe,
                trend_filter,
                index_symbol,
                scan_label="STREAM SCAN",
            )

            results: list[dict] = []
            batch_data_all: dict = {}
            filtered_out = 0
            poor_rating_hidden = 0
            direction_counts = {"Bull": 0, "Bear": 0}
            total_tickers = len(tickers)
            fetched_so_far = 0
            batch_idx = 0

            _score_one = _make_scorer(
                settings,
                timeframe,
                index_df,
                trend_filter,
                is_large,
                self._enrich_with_providers,
            )

            # ── Stream per parallel batch ─────────────────────────────────
            reset_negative_skips()
            for chunk_data in fetch_batch_yfinance_stream(
                tickers,
                period=period,
                timeframe=timeframe,
                cancel_event=self._cancel_event,
                on_fallback_progress=lambda done, total: self._progress(
                    0.05 + 0.05 * (done / max(total, 1)),
                    f"Fallback fetch {done}/{total} (jugaad/nselib)",
                ),
            ):
                if self._cancel_event.is_set():
                    result.cancelled = True
                    self._log("\n⏹  Scan cancelled by user")
                    break
                if not chunk_data:
                    continue
                batch_idx += 1
                batch_data_all.update(chunk_data)
                fetched_so_far += len(chunk_data)
                self._progress(
                    0.05 + (fetched_so_far / max(total_tickers, 1) * 0.05),
                    f"Batch {batch_idx}: {len(chunk_data)} downloaded ({fetched_so_far}/{total_tickers})",
                )
                self._log(
                    f"Batch {batch_idx} received: {len(chunk_data)} tickers (cumulative {fetched_so_far}) — scoring..."
                )

                chunk_results: list[dict] = []
                # For large universes use parallel scoring per chunk (smaller
                # pool) — poll cancel every 0.5s so Stop isn't held by scoring.
                if is_large and len(chunk_data) > 20:
                    from concurrent.futures import (
                        FIRST_COMPLETED,
                        wait,
                    )

                    max_w = min(4, (len(chunk_data) // 25) + 1)
                    ex = _DaemonThreadPoolExecutor(max_workers=max_w)
                    futs = {
                        ex.submit(_score_one, item): item[0]
                        for item in chunk_data.items()
                    }
                    cancelled_loop = False
                    pending = set(futs)
                    while pending:
                        if self._cancel_event.is_set():
                            ex.shutdown(wait=False, cancel_futures=True)
                            cancelled_loop = True
                            break
                        completed, pending = wait(
                            pending, timeout=0.5, return_when=FIRST_COMPLETED
                        )
                        if not completed:
                            continue
                        for fut in completed:
                            scores, direction = fut.result()
                            f, p = _tally(
                                scores, direction, chunk_results, direction_counts
                            )
                            filtered_out += f
                            poor_rating_hidden += p
                    if not cancelled_loop:
                        ex.shutdown(wait=True)
                else:
                    # Small universes run full per-ticker enrichment (~3s of
                    # provider calls each) — parallelize across tickers the
                    # way phase 2 does. Input order is preserved so the batch
                    # callback sees the same sequence as a sequential run.
                    items = list(chunk_data.items())
                    ordered, _ = _parallel_score(items, _score_one, self._cancel_event)
                    for idx, (ticker, _df) in enumerate(items):
                        res = ordered.get(idx)
                        if res is None:
                            continue  # cancelled before this item ran
                        scores, direction = res
                        f, p = _tally(
                            scores, direction, chunk_results, direction_counts
                        )
                        filtered_out += f
                        poor_rating_hidden += p

                if chunk_results:
                    # Keep global results sorted incrementally for top-200 calc later
                    results.extend(chunk_results)
                    # Progress: 0.1 - 0.8 range proportional to fetched
                    prog = 0.1 + (fetched_so_far / max(total_tickers, 1) * 0.7)
                    self._progress(
                        prog,
                        f"Scored {len(results)} passed ({fetched_so_far}/{total_tickers} fetched)",
                    )
                    for r in chunk_results[:3]:
                        tag = (
                            "✓" if r["total"] >= settings.get("min_score", 50) else "✗"
                        )
                        self._log(
                            f"  {tag} {r['ticker']}: {r['total']:.1f}/100 ({r['trend_dir']})"
                        )
                    if len(chunk_results) > 3:
                        self._log(f"  ... +{len(chunk_results) - 3} more in this batch")
                    if batch_cb:
                        try:
                            batch_cb(chunk_results)
                        except Exception as e:
                            logger.info("on_batch callback failed: %s", e)

            # ── Phase 2: Enrich top 200 for large universes (update in place) ─
            if is_large and results and not self._cancel_event.is_set():
                enriched_top, results = _enrich_top(
                    results,
                    batch_data_all,
                    settings,
                    timeframe,
                    index_df,
                    self._enrich_with_providers,
                    cancel_event=self._cancel_event,
                    progress=self._progress,
                    log=self._log,
                )
                if batch_cb and enriched_top:
                    try:
                        # Send enriched top as update batch — UI will replace existing tickers
                        batch_cb(enriched_top)
                        self._log(
                            f"Top {len(enriched_top)} enrichment complete — grid updated"
                        )
                    except Exception as e:
                        logger.info("on_batch enrichment callback failed: %s", e)

            self._finalize_scan(
                result,
                settings,
                tickers,
                results,
                filtered_out,
                poor_rating_hidden,
                direction_counts,
                trend_filter,
                batch_data=batch_data_all,
                scan_label="Stream Scan Complete",
            )

        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            self._log(f"\nERROR: {type(e).__name__}: {e}")
            logger.exception("Stream scan failed")

        elapsed = _time.monotonic() - t_start
        _log_scan_outcome(
            result,
            universe,
            elapsed,
            tickers,
            settings,
            "stream_scan",
            self._cancel_event,
        )
        return result
