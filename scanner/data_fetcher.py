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
from typing import Optional

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
        except (ValueError, TypeError):
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
                     retries: int = 2) -> Optional[pd.DataFrame]:
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
                    df._fundamentals = fund
                return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                logger.warning("Failed to fetch %s: %s", ticker, e)

    return None


def fetch_index_data(ticker: str = "^NSEI", period: str = "1y") -> Optional[pd.DataFrame]:
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


def fetch_fundamentals(ticker: str) -> Optional[dict]:
    """
    Fetch fundamental data for a stock.

    Uses multi-source provider:
      yfinance -> nselib

    Returns dict with pe_ratio, eps_growth, rev_growth, roe.
    """
    provider = _get_provider()
    return provider.fetch_fundamentals(ticker)


def fetch_batch_yfinance(tickers: list, period: str = "1y", timeframe: str = "D") -> dict:
    """
    Fast batch fetch using yfinance download (single API call).
    This is MUCH faster than individual fetches.

    Args:
        tickers: List of NSE symbols
        period: Data period
        timeframe: 'D' daily, 'W' weekly, 'M' monthly

    Returns:
        Dict mapping ticker -> DataFrame
    """
    try:
        import yfinance as yf
        import pandas as pd

        download_period = _extend_period_for_timeframe(period, timeframe)

        # Add .NS suffix for NSE stocks
        yf_tickers = [f"{t}.NS" for t in tickers]
        ticker_map = {f"{t}.NS": t for t in tickers}  # Map back to original

        # Single batch download
        logger.info("Batch downloading %d stocks via yfinance...", len(tickers))
        data = yf.download(yf_tickers, period=download_period, group_by="ticker",
                           auto_adjust=True, progress=False, threads=True)

        results = {}
        multi_idx = isinstance(data.columns, pd.MultiIndex)
        for yf_ticker, orig_ticker in ticker_map.items():
            try:
                if len(tickers) == 1 and not multi_idx:
                    # Single ticker returned as a flat (non-MultiIndex) DataFrame
                    df = data.copy()
                elif multi_idx:
                    if yf_ticker not in data.columns.get_level_values(0):
                        continue
                    df = data[yf_ticker].copy()
                else:
                    # Single ticker but returned flat anyway
                    df = data.copy()

                if df is None or df.empty or len(df) < 20:
                    continue

                # Normalize columns
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = ["open", "high", "low", "close", "volume"]
                df = df.dropna()

                # Resample to the requested analysis timeframe
                # ('D' = daily, no change; 'W' weekly; 'M' monthly)
                if timeframe != "D":
                    df = resample_ohlcv(df, timeframe)

                if df is not None and len(df) >= 50:
                    results[orig_ticker] = df
            except Exception as e:
                logger.debug("Skipping %s in batch: %s", orig_ticker, e)
                continue

        logger.info("Batch download complete: %d/%d stocks", len(results), len(tickers))
        return results

    except ImportError:
        logger.warning("yfinance not available for batch download")
        return {}
    except Exception as e:
        logger.error("Batch download failed: %s", e)
        return {}


def fetch_stock_fast(ticker: str, period: str = "1y", timeframe: str = "D") -> Optional[pd.DataFrame]:
    """
    Fast single stock fetch using cache-first approach.
    """
    provider = _get_provider()
    download_period = _extend_period_for_timeframe(period, timeframe)
    try:
        df = provider.fetch_stock(ticker, download_period)
        if df is not None and not df.empty and len(df) >= 50:
            df = resample_ohlcv(df, timeframe)
            if df is not None and not df.empty:
                fund = provider.fetch_fundamentals(ticker)
                if fund is not None:
                    df._fundamentals = fund
                return df
    except Exception as e:
        logger.debug("fetch_stock_fast failed for %s: %s", ticker, e)
    return None
