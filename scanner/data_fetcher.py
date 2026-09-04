"""
Data fetcher for Indian stocks using multi-source provider with fallback.

Provider chain:
  1. jugaad-data — NSE official API
  2. yfinance — Yahoo Finance
  3. nselib — NSE library

All data is cached to disk to avoid repeated API calls.
"""

import json
import logging
import os
import threading
import time

import pandas as pd

from .trace import trace
from ._index_utils import _normalize_daily_index  # noqa: F401

logger = logging.getLogger(__name__)

from .data_providers import (
    DataProvider,
    _get_cached,
    _set_cached,
    prune_stale_cache,
)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample daily OHLCV data to weekly or monthly bars.

    Args:
        df: Daily OHLCV DataFrame with columns [open, high, low, close, volume]
        timeframe: 'D' (daily, no change), 'W' (weekly), 'M' (monthly)

    Returns:
        Resampled DataFrame with same columns
    """
    if timeframe == "D" or df is None or df.empty:
        return df

    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            # If index can't be converted, try to find a date column
            for col in ["date", "Date", "DATE"]:
                if col in df.columns:
                    df.index = pd.to_datetime(df[col])
                    df = df.drop(columns=[col])
                    break
            else:
                return df  # Can't resample without datetime index

    # Normalize stamps onto tz-naive IST trade dates before bucketing, so a
    # UTC-close frame does not land in the wrong weekly/monthly bucket.
    df = _normalize_daily_index(df)
    if len(df) < 2:
        return df

    if timeframe == "W":
        rule = "W"
    elif timeframe == "M":
        rule = "ME"  # Month End
    else:
        return df

    # Build aggregation rules based on available columns
    agg_rules = {}
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            agg_rules[col] = {"open": "first", "high": "max", "low": "min", "close": "last"}[col]
    if "volume" in df.columns:
        agg_rules["volume"] = "sum"

    resampled = df.resample(rule).agg(agg_rules).dropna()

    return resampled


# ── Global provider instance ───────────────────────────────────────────────
_provider = None
_provider_lock = threading.Lock()


def _get_provider() -> DataProvider:
    """Get or create the global data provider (thread-safe)."""
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = DataProvider()
    return _provider


def _extend_period_for_timeframe(period: str, timeframe: str) -> str:
    """Extend data period for higher timeframes.

    yfinance only provides daily data. When resampling to weekly/monthly,
    we need more daily bars so the result has enough bars for analysis.
    Monthly needs 60+ bars, so 1y daily (~252 bars) -> 5y daily (~1260 bars) -> 60 months.
    """
    if timeframe == "D":
        return period
    if timeframe == "W":
        return {"6mo": "1y", "1y": "2y", "2y": "5y", "5y": "5y"}.get(period, "2y")
    if timeframe == "M":
        return {"6mo": "2y", "1y": "5y", "2y": "5y", "5y": "5y"}.get(period, "5y")
    return period


def fetch_stock_data(ticker: str, period: str = "1y", timeframe: str = "D",
                     retries: int = 2) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for an Indian NSE stock.

    Uses multi-source provider with automatic fallback:
      jugaad-data -> yfinance -> nselib

    Args:
        ticker: NSE symbol (e.g., "RELIANCE", "TCS")
        period: Data period ("6mo", "1y", "2y", "5y")
        timeframe: Bar interval - 'D' (daily), 'W' (weekly), 'M' (monthly)
        retries: Number of retry attempts

    Returns:
        DataFrame with columns [open, high, low, close, volume] or None
    """
    provider = _get_provider()
    download_period = _extend_period_for_timeframe(period, timeframe)

    for attempt in range(retries):
        try:
            df = provider.fetch_stock(ticker, download_period)
            if df is not None and not df.empty and len(df) >= 50:
                # Resample to requested timeframe
                df = resample_ohlcv(df, timeframe)
                if df is None or df.empty:
                    continue
                # Attach fundamentals
                fund = provider.fetch_fundamentals(ticker)
                if fund is not None:
                    df.attrs['_fundamentals'] = fund
                return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                logger.warning("Failed to fetch %s: %s", ticker, e)

    return None


def fetch_index_data(ticker: str = "^NSEI", period: str = "1y") -> pd.DataFrame | None:
    """
    Fetch NIFTY 50 index data for relative strength comparison.

    Uses multi-source provider with automatic fallback.

    Args:
        ticker: Index ticker ("^NSEI" for NIFTY 50)
        period: Data period

    Returns:
        DataFrame with OHLCV data or None
    """
    provider = _get_provider()
    return provider.fetch_index(ticker, period)


def fetch_fundamentals(ticker: str) -> dict | None:
    """
    Fetch fundamental data for a stock.

    Uses multi-source provider:
      yfinance -> nselib

    Returns dict with pe_ratio, eps_growth, rev_growth, roe.
    """
    provider = _get_provider()
    return provider.fetch_fundamentals(ticker)


CHUNK = 200  # ~200 * 8 chars avg + commas ≈ 1.6k URL < 8k limit; safe for Yahoo
MAX_PARALLEL_CHUNKS = 8  # parallel chunk downloads — 8×200 = 1600 tickers in flight
SLEEP_BETWEEN_BATCH = 0.3  # throttle between parallel batches to avoid 429
FALLBACK_WORKERS = 8  # per-ticker fallback threads for tickers yfinance missed
FALLBACK_PROVIDER_TIMEOUT = 10.0  # per-provider cap (s) in the fallback pass — dead symbols fail fast
FALLBACK_FILTER_MIN_MISSING = 25  # only consult the NSE list above this many misses (filter pays off at scale)

# ── Negative cache: symbols with no data on any provider ────────────────────
# A symbol that fails the whole NSE fallback chain once is very likely dead
# (delisted / suspended / permanently renamed) — re-attempting it on every
# scan costs up to the full provider timeout per attempt. Remember failures
# on disk (.cache/dead_symbols.json) and skip re-attempts for a day.
NEGATIVE_CACHE_TTL_HOURS = 24
_NEGATIVE_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".cache", "dead_symbols.json"
)
_negative_cache: dict[str, float] | None = None  # ticker -> epoch (lazy-loaded)
_negative_lock = threading.Lock()


def _negative_cache_load() -> dict[str, float]:
    """Load unexpired entries from disk once per process."""
    global _negative_cache
    with _negative_lock:
        if _negative_cache is None:
            cache: dict[str, float] = {}
            try:
                with open(_NEGATIVE_CACHE_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                now = time.time()
                # Read the override directly — negative_cache_ttl_hours() takes
                # the same (non-reentrant) lock and would deadlock here.
                cache = {
                    k: ts for k, ts in raw.items()
                    if isinstance(ts, (int, float))
                    and now - ts < _negative_cache_ttl_hours * 3600
                }
            except Exception:
                logger.debug("Negative cache load failed (missing/corrupt file)", exc_info=True)
            _negative_cache = cache
        return _negative_cache


def _negative_cache_save() -> None:
    """Persist the in-memory cache (caller must hold the lock)."""
    try:
        os.makedirs(os.path.dirname(_NEGATIVE_CACHE_PATH), exist_ok=True)
        tmp = _NEGATIVE_CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_negative_cache, f)
        os.replace(tmp, _NEGATIVE_CACHE_PATH)
    except Exception:
        logger.debug("Negative-cache write failed", exc_info=True)


def _negative_cache_contains(ticker: str) -> bool:
    """True when the ticker was marked dead within the TTL window."""
    cache = _negative_cache_load()
    with _negative_lock:
        ts = cache.get(ticker)
        if ts is None:
            return False
        # Direct read: negative_cache_ttl_hours() re-locks -> deadlock.
        if time.time() - ts < _negative_cache_ttl_hours * 3600:
            return True
        cache.pop(ticker, None)
        return False


def _negative_cache_update(marks: list | None = None, clears: list | None = None) -> None:
    """Mark symbols as dead (no data on any provider) and/or clear others."""
    if not marks and not clears:
        return
    cache = _negative_cache_load()
    now = time.time()
    with _negative_lock:
        changed = False
        for t in marks or []:
            cache[t] = now
            changed = True
        for t in clears or []:
            if cache.pop(t, None) is not None:
                changed = True
        if changed:
            _negative_cache_save()


# ── Runtime TTL override + per-scan skip counter ────────────────────────────
# The TTL is configurable from the GUI settings; the skip counter lets the
# engine report how many dead symbols were skipped in the scan summary.
_negative_cache_ttl_hours: float = NEGATIVE_CACHE_TTL_HOURS
_neg_cache_skips = 0
_neg_cache_skips_lock = threading.Lock()


def set_negative_cache_ttl_hours(hours: float | None) -> None:
    """Override the expiry window (hours); None restores the default."""
    global _negative_cache_ttl_hours
    with _negative_lock:
        _negative_cache_ttl_hours = (
            max(0.5, float(hours)) if hours else NEGATIVE_CACHE_TTL_HOURS
        )


def negative_cache_ttl_hours() -> float:
    """Current expiry window in hours (default or runtime override)."""
    with _negative_lock:
        return _negative_cache_ttl_hours


def reset_negative_cache_skip_count() -> None:
    """Zero the per-scan skip counter (call before a scan's fetch pass)."""
    global _neg_cache_skips
    with _neg_cache_skips_lock:
        _neg_cache_skips = 0


def negative_cache_skip_count() -> int:
    """How many symbols the current scan skipped via the negative cache."""
    with _neg_cache_skips_lock:
        return _neg_cache_skips


def _record_negative_cache_skips(n: int) -> None:
    """Accumulate skipped symbols for the scan summary."""
    global _neg_cache_skips
    if n <= 0:
        return
    with _neg_cache_skips_lock:
        _neg_cache_skips += n


# ── Enrichment cache: phase-2 provider results ──────────────────────────────
# The top-200 phase-2 pass runs 5 parallel provider fetches (sentiment,
# social, delivery/FII-DII, screener, insider) plus fundamentals per ticker
# — the dominant cost on full-market scans (~6 min of the ~9 min total).
# Those values barely change intraday, so remember them on disk
# (.cache/enrichment_cache.json) and replay them on repeat scans within the
# TTL window (default 24h) instead of re-hitting the providers.
ENRICHMENT_CACHE_TTL_HOURS = 24
_ENRICHMENT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".cache", "enrichment_cache.json"
)
_enrichment_cache: dict | None = None  # ticker -> {ts, providers, fundamentals}
_enrichment_lock = threading.Lock()
_enrichment_hits = 0
_enrichment_misses = 0
_enrichment_counts_lock = threading.Lock()


def _enrichment_cache_load() -> dict:
    """Load unexpired entries from disk once per process."""
    global _enrichment_cache
    with _enrichment_lock:
        if _enrichment_cache is None:
            cache: dict = {}
            try:
                with open(_ENRICHMENT_CACHE_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                now = time.time()
                for k, entry in raw.items():
                    if (
                        isinstance(entry, dict)
                        and isinstance(entry.get("ts"), (int, float))
                        and now - entry["ts"] < ENRICHMENT_CACHE_TTL_HOURS * 3600
                    ):
                        cache[k] = entry
            except Exception:
                logger.debug("Enrichment cache load failed (missing/corrupt file)", exc_info=True)
            _enrichment_cache = cache
        return _enrichment_cache


def _enrichment_cache_save() -> None:
    """Persist the in-memory cache (caller must hold the lock)."""
    try:
        os.makedirs(os.path.dirname(_ENRICHMENT_CACHE_PATH), exist_ok=True)
        tmp = _ENRICHMENT_CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_enrichment_cache, f)
        os.replace(tmp, _ENRICHMENT_CACHE_PATH)
    except Exception:
        logger.debug("Enrichment-cache write failed", exc_info=True)


def _enrichment_cache_get(ticker: str) -> dict | None:
    """Fresh cached entry for a ticker (providers + fundamentals), else None."""
    cache = _enrichment_cache_load()
    with _enrichment_lock:
        entry = cache.get(ticker)
        if entry is None:
            return None
        if time.time() - entry.get("ts", 0) < ENRICHMENT_CACHE_TTL_HOURS * 3600:
            return entry
        cache.pop(ticker, None)  # expired — evict
        return None


def _enrichment_cache_put(
    ticker: str, providers: dict, fundamentals: dict | None
) -> None:
    """Store provider keys + fundamentals for a ticker (atomic write).

    Entries without any provider data are not cached — a transient provider
    outage must not freeze scores into the cache.
    """
    if not providers:
        return
    cache = _enrichment_cache_load()
    with _enrichment_lock:
        cache[ticker] = {
            "ts": time.time(),
            "providers": providers,
            "fundamentals": fundamentals,
        }
        _enrichment_cache_save()


def enrichment_cache_clear() -> None:
    """Wipe the enrichment cache (memory + disk)."""
    global _enrichment_cache
    with _enrichment_lock:
        _enrichment_cache = {}
        _enrichment_cache_save()


def enrichment_cache_size() -> int:
    """Number of tickers currently held in the enrichment cache."""
    # Load first: _enrichment_cache_load() takes the same (non-reentrant)
    # lock, so calling it while holding it would deadlock.
    cache = _enrichment_cache_load()
    with _enrichment_lock:
        return len(cache)


def reset_enrichment_cache_counts() -> None:
    """Zero the per-scan hit/miss counters (call before an enrichment pass)."""
    global _enrichment_hits, _enrichment_misses
    with _enrichment_counts_lock:
        _enrichment_hits = 0
        _enrichment_misses = 0


def enrichment_cache_hits() -> int:
    """Rows this scan served from the enrichment cache."""
    with _enrichment_counts_lock:
        return _enrichment_hits


def enrichment_cache_misses() -> int:
    """Rows this scan fetched from the providers (and cached)."""
    with _enrichment_counts_lock:
        return _enrichment_misses


def _record_enrichment_cache_hit() -> None:
    """Increment the per-scan cache-hit counter (worker threads)."""
    global _enrichment_hits
    with _enrichment_counts_lock:
        _enrichment_hits += 1


def _record_enrichment_cache_miss() -> None:
    """Increment the per-scan cache-miss counter (worker threads)."""
    global _enrichment_misses
    with _enrichment_counts_lock:
        _enrichment_misses += 1


def _nse_membership_set() -> set | None:
    """Current NSE mainboard symbols (upper-cased), or None when unknown.

    Uses symbol_fetcher's 4h-cached nselib list — the same NSE list that
    builds the FULL MARKET universe. Returns None when the list cannot be
    resolved confidently; callers then attempt every missed ticker (the
    unfiltered behavior) instead of dropping symbols wrongly.
    """
    try:
        from .symbol_fetcher import fetch_nse_mainboard

        symbols = fetch_nse_mainboard()
        if symbols and len(symbols) > 500:
            return {str(s).strip().upper() for s in symbols}
    except Exception as e:
        logger.debug("NSE mainboard list unavailable: %s", e)
    return None


def _fetch_fallback_batch(
    tickers: list,
    period: str = "1y",
    timeframe: str = "D",
    cancel_event: threading.Event | None = None,
    on_progress=None,
) -> dict:
    """
    Fetch OHLCV for tickers the yfinance batch pass missed, using the
    DataProvider fallback chain restricted to NSE-native sources:
    jugaad-data -> nselib. yfinance is deliberately skipped — it just
    failed at batch level (rate limit / outage), so retrying it per
    ticker would only slow recovery down.

    Results are normalized and validated exactly like the batch path
    (>= 50 bars after optional resample), so callers can treat recovered
    frames identically to regular chunk results.

    Args:
        tickers: Tickers to recover.
        period: Data period (e.g. "1y").
        timeframe: Bar interval 'D' (daily), 'W' (weekly), 'M' (monthly).
        cancel_event: When set, stops the pass early and returns partial results.
        on_progress: Optional callback on_progress(done, total).

    Returns:
        Dict mapping recovered ticker -> normalized OHLCV DataFrame.
    """
    if not tickers:
        return {}

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    provider = _get_provider()
    download_period = _extend_period_for_timeframe(period, timeframe)
    total = len(tickers)
    workers = min(FALLBACK_WORKERS, total)
    recovered: dict = {}
    no_data: list = []  # tickers with NO data on any fallback provider -> dead
    done = 0

    def _fetch_one(t: str):
        try:
            df = provider.fetch_stock(
                t, download_period,
                skip=("yfinance",),
                provider_timeout=FALLBACK_PROVIDER_TIMEOUT,
            )
            if df is None or df.empty:
                return "dead", t, None
            df = resample_ohlcv(df, timeframe)
            if timeframe == "D":
                df = _normalize_daily_index(df)
            if df is not None and not df.empty and len(df) >= 50:
                return "ok", t, df
            # Data exists but too few bars — a miss, but NOT a dead symbol
            # (e.g. recent IPO), so don't negative-cache it.
            return "short", t, None
        except Exception as e:
            logger.debug("Fallback fetch failed for %s: %s", t, e)
            return "dead", t, None

    executor = ThreadPoolExecutor(max_workers=workers)
    future_to_ticker = {executor.submit(_fetch_one, t): t for t in tickers}
    try:
        # Daemon threads: on cancel we don't join, so in-flight provider calls
        # must not keep the process alive afterwards.
        for t in executor._threads:
            t.daemon = True
    except Exception:
        pass
    cancelled_fb = False
    pending = set(future_to_ticker)
    while pending:
        if cancel_event is not None and cancel_event.is_set():
            executor.shutdown(wait=False, cancel_futures=True)
            cancelled_fb = True
            logger.info(
                "Fallback fetch cancelled — %d/%d recovered",
                len(recovered), total,
            )
            break
        # Poll every 0.5s instead of blocking on as_completed — lets Stop
        # return promptly even while slow provider calls are still running.
        completed, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
        if not completed:
            continue
        for future in completed:
            done += 1
            status, t, df = future.result()
            if status == "ok":
                recovered[t] = df
            elif status == "dead":
                no_data.append(t)
            if on_progress is not None:
                try:
                    on_progress(done, total)
                except Exception:
                    pass
    if not cancelled_fb:
        executor.shutdown(wait=True)

    # Remember what really has no data (skip future scans) and forget any
    # symbol that just came back (e.g. suspension lifted since TTL expiry).
    if no_data or recovered:
        _negative_cache_update(marks=no_data, clears=list(recovered))
    return recovered


def fetch_batch_yfinance_stream(
    tickers: list, period: str = "1y", timeframe: str = "D",
    cancel_event: threading.Event | None = None,
    on_fallback_progress=None,
):
    """
    Streaming generator that yields one dict per parallel batch.

    Each yielded dict is {ticker: DataFrame} for ~200-1000 tickers (one
    outer batch of up to MAX_PARALLEL_CHUNKS chunks). Caller can render
    incrementally instead of waiting for all ~5900.

    After the yfinance chunks are exhausted, any ticker the batch pass
    missed (Yahoo rate limit / no data on Yahoo) is retried through the
    NSE-native providers (jugaad-data -> nselib) and, if anything was
    recovered, yielded once more as a final batch.

    Usage:
        for chunk_data in fetch_batch_yfinance_stream(tickers):
            # process chunk_data and update UI
    """
    if not tickers:
        return

    # Sweep unreachable cache entries from previous days (once/hour/process)
    # so day-keyed pkl files never accumulate across scans.
    prune_stale_cache()

    # Deduplicate while preserving order
    seen = set()
    uniq_tickers = []
    for t in tickers:
        u = str(t).strip().upper()
        if u and u not in seen:
            seen.add(u)
            uniq_tickers.append(u)
    tickers = uniq_tickers
    seen_tickers = set()  # union of tickers returned across yfinance chunks

    try:
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        import pandas as pd
        import yfinance as yf

        download_period = _extend_period_for_timeframe(period, timeframe)
        total = len(tickers)
        chunks = [tickers[i : i + CHUNK] for i in range(0, total, CHUNK)]
        logger.info(
            "Batch streaming %d stocks via yfinance in %d chunk(s) of %d (parallel x%d)...",
            total,
            len(chunks),
            CHUNK,
            MAX_PARALLEL_CHUNKS,
        )

        def _fetch_chunk(chunk: list, ci: int) -> dict:
            # Cache-first: replay fresh daily bars from the per-ticker disk
            # cache and download only what's missing/expired — repeat scans
            # (within the price-cache TTL) skip re-downloading unchanged
            # symbols. Frames are cached as DAILY bars keyed by download_period
            # so every timeframe shares one download; resampling happens here.
            chunk_results: dict = {}
            to_download: list = []
            for t in chunk:
                try:
                    daily = _get_cached(t, download_period, "cache")
                except Exception as e:
                    logger.debug("Batch cache read failed for %s: %s", t, e)
                    daily = None
                # Legacy cache entries can hold UTC-close stamps; normalize so
                # all tickers share one trade-date calendar.
                if daily is not None and not daily.empty:
                    daily = _normalize_daily_index(daily)
                if daily is not None and not daily.empty:
                    df = daily if timeframe == "D" else resample_ohlcv(daily, timeframe)
                    if df is not None and len(df) >= 50:
                        chunk_results[t] = df
                        continue
                to_download.append(t)
            cached_hits = len(chunk_results)
            if not to_download:
                logger.info(
                    "Chunk %d/%d done: %d tickers (all served from disk cache)",
                    ci, len(chunks), len(chunk_results),
                )
                return chunk_results
            if cached_hits:
                logger.info(
                    "Chunk %d/%d: %d served from disk cache, downloading %d via yfinance",
                    ci, len(chunks), cached_hits, len(to_download),
                )

            yf_tickers = [f"{t}.NS" for t in to_download]
            ticker_map = {f"{t}.NS": t for t in to_download}
            try:
                data = yf.download(
                    yf_tickers,
                    period=download_period,
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    timeout=15,
                )
            except Exception as e:
                logger.warning("Chunk %d/%d download failed: %s", ci, len(chunks), e)
                return chunk_results
            if data is None or data.empty:
                logger.debug("Chunk %d/%d returned empty", ci, len(chunks))
                return chunk_results
            multi_idx = isinstance(data.columns, pd.MultiIndex)
            single = len(to_download) == 1
            for yf_ticker, orig_ticker in ticker_map.items():
                try:
                    if single and not multi_idx:
                        df = data.copy()
                    elif multi_idx:
                        if yf_ticker not in data.columns.get_level_values(0):
                            continue
                        df = data[yf_ticker].copy()
                    else:
                        df = data.copy()
                    if df is None or df.empty or len(df) < 20:
                        continue
                    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.columns = ["open", "high", "low", "close", "volume"]
                    df = df.dropna()
                    daily = _normalize_daily_index(df)  # on-disk cache unit
                    df = daily
                    if timeframe != "D":
                        df = resample_ohlcv(df, timeframe)
                    if df is not None and len(df) >= 50:
                        chunk_results[orig_ticker] = df
                        try:
                            _set_cached(orig_ticker, download_period, "cache", daily)
                        except Exception as e:
                            logger.debug(
                                "Batch cache write failed for %s: %s", orig_ticker, e
                            )
                except Exception as e:
                    logger.debug("Skipping %s in chunk %d: %s", orig_ticker, ci, e)
                    continue
            logger.info("Chunk %d/%d done: %d tickers", ci, len(chunks), len(chunk_results))
            return chunk_results

        cumulative = 0
        for batch_start in range(0, len(chunks), MAX_PARALLEL_CHUNKS):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Batch download cancelled between batches")
                break
            batch = chunks[batch_start : batch_start + MAX_PARALLEL_CHUNKS]
            batch_indices = list(range(batch_start + 1, batch_start + len(batch) + 1))
            batch_results: dict = {}
            cancelled_batch = False
            executor = ThreadPoolExecutor(max_workers=len(batch))
            future_to_ci = {executor.submit(_fetch_chunk, chunk, ci): ci for chunk, ci in zip(batch, batch_indices)}
            try:
                # Daemon threads: on cancel we don't join in-flight chunk
                # downloads (which can take ~60s under Yahoo rate limits).
                for t in executor._threads:
                    t.daemon = True
            except Exception:
                pass
            pending = set(future_to_ci)
            while pending:
                if cancel_event is not None and cancel_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    cancelled_batch = True
                    logger.info("Batch download cancelled — dropping in-flight chunks")
                    break
                # Poll every 0.5s instead of blocking on as_completed — lets
                # Stop return promptly even while chunks are mid-download.
                completed, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                if not completed:
                    continue
                for future in completed:
                    ci = future_to_ci[future]
                    try:
                        chunk_res = future.result()
                        batch_results.update(chunk_res)
                    except Exception as e:
                        logger.warning("Chunk %d failed in parallel batch: %s", ci, e)
            if not cancelled_batch:
                executor.shutdown(wait=True)
            cumulative += len(batch_results)
            seen_tickers.update(batch_results)
            logger.info(
                "Batch %d/%d done: %d this batch, %d/%d cumulative",
                (batch_start // MAX_PARALLEL_CHUNKS) + 1,
                (len(chunks) + MAX_PARALLEL_CHUNKS - 1) // MAX_PARALLEL_CHUNKS,
                len(batch_results),
                cumulative,
                total,
            )
            yield batch_results
            if batch_start + MAX_PARALLEL_CHUNKS < len(chunks):
                time.sleep(SLEEP_BETWEEN_BATCH)

        # ── Fallback: recover tickers the yfinance pass missed ───────────
        missing = [t for t in tickers if t not in seen_tickers]
        if missing and not (cancel_event is not None and cancel_event.is_set()):
            # Negative cache first: symbols that failed the whole NSE chain on
            # a recent scan are almost certainly dead — skip them cheaply.
            known_dead = [t for t in missing if _negative_cache_contains(t)]
            candidates = [t for t in missing if t not in known_dead]
            if known_dead:
                _record_negative_cache_skips(len(known_dead))
                logger.info(
                    "Skipping %d known-dead symbols via negative cache (marked within the last %dh)",
                    len(known_dead), negative_cache_ttl_hours(),
                )
            skipped: list = []
            # jugaad-data and nselib only serve NSE mainboard equities, so
            # BSE-only symbols (present in FULL MARKET) would each burn two
            # failed provider chains per scan. When the missed set is large
            # enough to matter, filter against nselib's (4h-cached) mainboard
            # list; when the list can't be resolved, attempt everything.
            if len(missing) >= FALLBACK_FILTER_MIN_MISSING:
                nse_members = _nse_membership_set()
                if nse_members is not None:
                    skipped = [t for t in candidates if t not in nse_members]
                    candidates = [t for t in candidates if t in nse_members]
            if skipped:
                logger.info(
                    "Skipping %d non-NSE (BSE-only / delisted) symbols in fallback",
                    len(skipped),
                )
            if not candidates:
                logger.warning(
                    "yfinance missed %d/%d tickers but none are recoverable (BSE-only or known-dead) — nothing to recover",
                    len(missing), total,
                )
            else:
                logger.info(
                    "yfinance missed %d/%d tickers — attempting %d via jugaad-data/nselib...",
                    len(missing), total, len(candidates),
                )
                recovered = _fetch_fallback_batch(
                    candidates,
                    period=period,
                    timeframe=timeframe,
                    cancel_event=cancel_event,
                    on_progress=on_fallback_progress,
                )
                if recovered:
                    logger.info(
                        "Fallback recovered %d/%d attempted tickers",
                        len(recovered), len(candidates),
                    )
                    yield recovered
                else:
                    logger.warning(
                        "Fallback fetch returned nothing for %d missed tickers",
                        len(candidates),
                    )

    except ImportError:
        logger.warning("yfinance not available for batch download")
        return
    except Exception:
        logger.exception("Batch streaming failed")
        return


@trace(level=logging.INFO, log_args=True, log_result=False)
def fetch_batch_yfinance(
    tickers: list,
    period: str = "1y",
    timeframe: str = "D",
    cancel_event=None,
    on_fallback_progress=None,
) -> dict:
    """
    Fast batch fetch using yfinance download — chunked for 6k symbols.

    Yahoo's URL limit (~8k chars) caps a single download to ~200-300 tickers.
    For 5,900 symbols we chunk into 200-symbol batches, with a short
    throttle between chunks to avoid 429. Much faster than individual fetches
    and scales linearly.

    Args:
        tickers: List of NSE/BSE symbols (plain, without .NS)
        period: Data period
        timeframe: 'D' daily, 'W' weekly, 'M' monthly

    Returns:
        Dict mapping ticker -> DataFrame
    """
    # Backward-compat wrapper: collect streaming batches (yfinance chunks,
    # then a final fallback batch for tickers yfinance missed).
    results: dict = {}
    total = len(tickers) if tickers else 0
    for batch_data in fetch_batch_yfinance_stream(
        tickers, period, timeframe,
        cancel_event=cancel_event,
        on_fallback_progress=on_fallback_progress,
    ):
        results.update(batch_data)
    logger.info(
        "Batch download complete: %d/%d stocks (parallel chunked + NSE fallback)",
        len(results), total,
    )
    return results
