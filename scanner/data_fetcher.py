"""
Data fetcher for Indian stocks using multi-source provider with fallback.

Provider chain:
  1. jugaad-data — NSE official API
  2. yfinance — Yahoo Finance
  3. nselib — NSE library

All data is cached to disk to avoid repeated API calls.
"""

import logging
import time

import pandas as pd

from .trace import trace

logger = logging.getLogger(__name__)

from .data_providers import DataProvider

# ── OHLCV Resampling ───────────────────────────────────────────────────────

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

    # Remove timezone info if present to avoid issues
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

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


def _get_provider() -> DataProvider:
    """Get or create the global data provider."""
    global _provider
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
                    object.__setattr__(df, '_fundamentals', fund)
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


def fetch_batch_yfinance_stream(tickers: list, period: str = "1y", timeframe: str = "D"):
    """
    Streaming generator that yields one dict per parallel batch.

    Each yielded dict is {ticker: DataFrame} for ~200-1000 tickers (one
    outer batch of up to MAX_PARALLEL_CHUNKS chunks). Caller can render
    incrementally instead of waiting for all ~5900.

    Usage:
        for chunk_data in fetch_batch_yfinance_stream(tickers):
            # process chunk_data and update UI
    """
    if not tickers:
        return

    # Deduplicate while preserving order
    seen = set()
    uniq_tickers = []
    for t in tickers:
        u = str(t).strip().upper()
        if u and u not in seen:
            seen.add(u)
            uniq_tickers.append(u)
    tickers = uniq_tickers

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

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
            yf_tickers = [f"{t}.NS" for t in chunk]
            ticker_map = {f"{t}.NS": t for t in chunk}
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
                return {}
            if data is None or data.empty:
                logger.debug("Chunk %d/%d returned empty", ci, len(chunks))
                return {}
            chunk_results: dict = {}
            multi_idx = isinstance(data.columns, pd.MultiIndex)
            for yf_ticker, orig_ticker in ticker_map.items():
                try:
                    if len(chunk) == 1 and not multi_idx:
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
                    if timeframe != "D":
                        df = resample_ohlcv(df, timeframe)
                    if df is not None and len(df) >= 50:
                        chunk_results[orig_ticker] = df
                except Exception as e:
                    logger.debug("Skipping %s in chunk %d: %s", orig_ticker, ci, e)
                    continue
            logger.info("Chunk %d/%d done: %d tickers", ci, len(chunks), len(chunk_results))
            return chunk_results

        cumulative = 0
        for batch_start in range(0, len(chunks), MAX_PARALLEL_CHUNKS):
            batch = chunks[batch_start : batch_start + MAX_PARALLEL_CHUNKS]
            batch_indices = list(range(batch_start + 1, batch_start + len(batch) + 1))
            batch_results: dict = {}
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                future_to_ci = {executor.submit(_fetch_chunk, chunk, ci): ci for chunk, ci in zip(batch, batch_indices)}
                for future in as_completed(future_to_ci):
                    ci = future_to_ci[future]
                    try:
                        chunk_res = future.result()
                        batch_results.update(chunk_res)
                    except Exception as e:
                        logger.warning("Chunk %d failed in parallel batch: %s", ci, e)
            cumulative += len(batch_results)
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

    except ImportError:
        logger.warning("yfinance not available for batch download")
        return
    except Exception:
        logger.exception("Batch streaming failed")
        return


@trace(level=logging.INFO, log_args=True, log_result=False)
def fetch_batch_yfinance(tickers: list, period: str = "1y", timeframe: str = "D") -> dict:
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
    # Backward-compat wrapper: collect streaming batches
    results: dict = {}
    total = len(tickers) if tickers else 0
    for batch_data in fetch_batch_yfinance_stream(tickers, period, timeframe):
        results.update(batch_data)
    logger.info("Batch download complete: %d/%d stocks (parallel chunked)", len(results), total)
    return results


def fetch_stock_fast(ticker: str, period: str = "1y", timeframe: str = "D") -> pd.DataFrame | None:
    """
    Fast single stock fetch using cache-first approach.
    Fundamentals are NOT fetched here — handled by ScannerEngine enrichment.
    """
    provider = _get_provider()
    download_period = _extend_period_for_timeframe(period, timeframe)
    try:
        df = provider.fetch_stock(ticker, download_period)
        if df is not None and not df.empty and len(df) >= 50:
            df = resample_ohlcv(df, timeframe)
            if df is not None and not df.empty:
                return df
    except Exception as e:
        logger.debug("fetch_stock_fast failed for %s: %s", ticker, e)
    return None
