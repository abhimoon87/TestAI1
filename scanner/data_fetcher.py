"""
Data fetcher for Indian stocks using multi-source provider with fallback.

Provider chain:
  1. jugaad-data — NSE official API
  2. yfinance — Yahoo Finance
  3. nselib — NSE library

All data is cached to disk to avoid repeated API calls.
"""

import time
import pandas as pd
from typing import Optional

from data_providers import DataProvider


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

    for attempt in range(retries):
        try:
            df = provider.fetch_stock(ticker, period)
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
                print(f"  Failed to fetch {ticker}: {e}")

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


def fetch_batch(tickers: list, period: str = "1y", delay: float = 0.1,
                fetch_fund: bool = True) -> dict:
    """
    Fetch data for multiple tickers with rate limiting.

    Args:
        tickers: List of NSE symbols
        period: Data period
        delay: Delay between requests (seconds)
        fetch_fund: Whether to fetch fundamentals data

    Returns:
        Dict mapping ticker -> DataFrame (with _fundamentals attr if fetch_fund=True)
    """
    provider = _get_provider()
    results = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{total}] Fetching {ticker}...", end="", flush=True)
        try:
            df = provider.fetch_stock(ticker, period)
            if df is not None and not df.empty:
                if fetch_fund:
                    fund = provider.fetch_fundamentals(ticker)
                    if fund is not None:
                        df._fundamentals = fund
                results[ticker] = df
                src = provider.last_provider or "?"
                print(f" OK ({len(df)} bars, src:{src})")
            else:
                print(f" no data")
        except Exception as e:
            print(f" error: {e}")

        if i < total:
            time.sleep(delay)

    return results
