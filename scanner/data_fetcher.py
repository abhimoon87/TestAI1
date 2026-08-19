"""
Data fetcher for Indian stocks using yfinance.
Handles NSE (.NS) suffix, fundamentals, and provides caching.
"""

import time
import pandas as pd
import yfinance as yf
from typing import Optional


def fetch_stock_data(ticker: str, period: str = "1y", retries: int = 2) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for an Indian NSE stock.

    Args:
        ticker: NSE symbol (e.g., "RELIANCE", "TCS")
        period: Data period ("6mo", "1y", "2y", "5y")
        retries: Number of retry attempts

    Returns:
        DataFrame with columns [open, high, low, close, volume] or None
    """
    nse_ticker = f"{ticker}.NS"

    for attempt in range(retries):
        try:
            stock = yf.Ticker(nse_ticker)
            df = stock.history(period=period, auto_adjust=True)

            if df is None or df.empty or len(df) < 50:
                return None

            # Standardize columns
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df = df.dropna()

            # Attach fundamentals
            fund = fetch_fundamentals(ticker)
            if fund is not None:
                df._fundamentals = fund

            return df

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"  ✗ Failed to fetch {ticker}: {e}")
                return None

    return None


def fetch_fundamentals(ticker: str) -> Optional[dict]:
    """
    Fetch fundamental data for a stock using yfinance .info.
    Returns dict with pe_ratio, eps_growth, rev_growth, roe.
    """
    nse_ticker = f"{ticker}.NS"
    try:
        stock = yf.Ticker(nse_ticker)
        info = stock.info

        if not info:
            return None

        # Check if we got valid data
        pe_ratio = info.get("trailingPE")

        # EPS Growth
        eps_growth = info.get("earningsGrowth")
        if eps_growth is not None:
            eps_growth = eps_growth * 100 if abs(eps_growth) < 100 else eps_growth
        else:
            earnings_q = info.get("earningsQuarterlyGrowth")
            if earnings_q is not None:
                eps_growth = earnings_q * 100 if abs(earnings_q) < 100 else earnings_q

        # Revenue Growth
        rev_growth = info.get("revenueGrowth")
        if rev_growth is not None:
            rev_growth = rev_growth * 100 if abs(rev_growth) < 100 else rev_growth

        # ROE
        roe = info.get("returnOnEquity")
        if roe is not None:
            roe = roe * 100 if abs(roe) < 100 else roe

        return {
            "pe_ratio": pe_ratio,
            "eps_growth": eps_growth,
            "rev_growth": rev_growth,
            "roe": roe,
        }

    except Exception:
        return None


def fetch_index_data(ticker: str = "^NSEI", period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Fetch NIFTY 50 index data for relative strength comparison.

    Args:
        ticker: Index ticker ("^NSEI" for NIFTY 50, "^NSEBANK" for Bank Nifty)
        period: Data period

    Returns:
        DataFrame with OHLCV data or None
    """
    try:
        index = yf.Ticker(ticker)
        df = index.history(period=period, auto_adjust=True)

        if df is None or df.empty:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        return df.dropna()

    except Exception as e:
        print(f"  ✗ Failed to fetch index {ticker}: {e}")
        return None


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
    results = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{total}] Fetching {ticker}...", end="", flush=True)
        df = fetch_stock_data(ticker, period=period)
        if df is not None:
            # Attach fundamentals if requested and not already present
            if fetch_fund and (not hasattr(df, '_fundamentals') or df._fundamentals is None):
                fund = fetch_fundamentals(ticker)
                if fund is not None:
                    df._fundamentals = fund
            results[ticker] = df
            print(f" ✓ ({len(df)} bars)")
        else:
            print(f" ✗ (no data)")

        if i < total:
            time.sleep(delay)

    return results
