"""
Multi-source data provider with fallback chain for Indian stock market.

Provider priority for OHLCV:
  1. jugaad-data — NSE official API, no auth needed
  2. yfinance — Yahoo Finance, no auth needed
  3. nselib — NSE library, no auth needed

Provider priority for Fundamentals:
  1. yfinance .info — Detailed financial data
  2. nselib pe_ratio — Bulk P/E ratio for all stocks

All providers normalize data to a common DataFrame format:
  columns = [open, high, low, close, volume]
"""

import os
import json
import time
import hashlib
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

# ── Cache Directory ────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_TTL_HOURS = 4  # Cache expires after 4 hours


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(ticker: str, period: str, provider: str) -> str:
    """Generate a cache file key."""
    today = date.today().isoformat()
    raw = f"{ticker}_{period}_{provider}_{today}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cached(ticker: str, period: str, provider: str) -> Optional[pd.DataFrame]:
    """Retrieve cached data if fresh enough."""
    _ensure_cache_dir()
    key = _cache_key(ticker, period, provider)
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
    meta_file = os.path.join(CACHE_DIR, f"{key}.meta")

    if not os.path.exists(cache_file) or not os.path.exists(meta_file):
        return None

    try:
        with open(meta_file, "r") as f:
            meta = json.load(f)
        cached_time = datetime.fromisoformat(meta["timestamp"])
        age_hours = (datetime.now() - cached_time).total_seconds() / 3600

        if age_hours > CACHE_TTL_HOURS:
            return None

        return pd.read_pickle(cache_file)
    except Exception:
        return None


def _set_cached(ticker: str, period: str, provider: str, df: pd.DataFrame):
    """Store data in cache."""
    _ensure_cache_dir()
    key = _cache_key(ticker, period, provider)
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
    meta_file = os.path.join(CACHE_DIR, f"{key}.meta")

    try:
        df.to_pickle(cache_file)
        with open(meta_file, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "rows": len(df)}, f)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER: jugaad-data (NSE Official API)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_jugaad(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from NSE via jugaad-data. No auth needed."""
    try:
        from jugaad_data.nse import stock_df
        from datetime import date, timedelta

        period_days = {"6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
        days = period_days.get(period, 365)
        end = date.today()
        start = end - timedelta(days=days)

        df = stock_df(symbol=ticker, from_date=start, to_date=end, series="EQ")

        if df is None or df.empty:
            return None

        # Normalize columns
        df = df.rename(columns={
            "OPEN": "open", "HIGH": "high", "LOW": "low",
            "CLOSE": "close", "VOLUME": "volume",
            "DATE": "date"
        })

        # Set DATE as index (needed for resampling)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df.index.name = None

        # Keep required columns
        cols = ["open", "high", "low", "close", "volume"]
        for c in cols:
            if c not in df.columns:
                return None

        df = df[cols].copy()
        df = df.dropna()

        return df

    except ImportError:
        return None
    except Exception:
        return None


def _fetch_jugaad_index(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Fetch index data from NSE via jugaad-data."""
    try:
        from jugaad_data.nse import index_df
        from datetime import date, timedelta

        period_days = {"6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
        days = period_days.get(period, 365)
        end = date.today()
        start = end - timedelta(days=days)

        # Map index tickers to NSE index names
        index_map = {
            "^NSEI": "NIFTY 50",
            "^NSEBANK": "NIFTY BANK",
            "NIFTY 50": "NIFTY 50",
        }
        index_name = index_map.get(ticker, ticker)

        df = index_df(symbol=index_name, from_date=start, to_date=end)

        if df is None or df.empty:
            return None

        # Normalize columns
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if "open" in cl:
                col_map[c] = "open"
            elif "high" in cl:
                col_map[c] = "high"
            elif "low" in cl:
                col_map[c] = "low"
            elif "close" in cl:
                col_map[c] = "close"
            elif "volume" in cl or "turnover" in cl:
                col_map[c] = "volume"

        df = df.rename(columns=col_map)
        cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in cols if c in df.columns]

        if len(available) < 4:
            return None

        df = df[available].copy()
        df = df.dropna()

        return df

    except ImportError:
        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER: yfinance (Yahoo Finance)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_yfinance(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from Yahoo Finance."""
    try:
        import yfinance as yf

        nse_ticker = f"{ticker}.NS"
        stock = yf.Ticker(nse_ticker)
        df = stock.history(period=period, auto_adjust=True)

        if df is None or df.empty:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.dropna()

        return df

    except ImportError:
        return None
    except Exception:
        return None


def _fetch_yfinance_index(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Fetch index data from Yahoo Finance."""
    try:
        import yfinance as yf

        index = yf.Ticker(ticker)
        df = index.history(period=period, auto_adjust=True)

        if df is None or df.empty:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        return df.dropna()

    except ImportError:
        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER: nselib (NSE Library)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_nselib(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from NSE via nselib."""
    try:
        from nselib import capital_market

        period_days = {"6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
        days = period_days.get(period, 365)
        end = date.today()
        start = end - timedelta(days=days)

        from_date = start.strftime("%d-%m-%Y")
        to_date = end.strftime("%d-%m-%Y")

        df = capital_market.price_volume_and_deliverable_position_data(
            symbol=ticker, from_date=from_date, to_date=to_date
        )

        if df is None or df.empty:
            return None

        # Normalize columns
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if "open" in cl:
                col_map[c] = "open"
            elif "high" in cl:
                col_map[c] = "high"
            elif "low" in cl:
                col_map[c] = "low"
            elif "close" in cl or "last" in cl:
                col_map[c] = "close"
            elif "quantity" in cl or "volume" in cl or "traded" in cl:
                col_map[c] = "volume"

        df = df.rename(columns=col_map)
        cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in cols if c in df.columns]

        if len(available) < 4:
            return None

        df = df[available].copy()
        df = df.dropna()

        return df

    except ImportError:
        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL DATA PROVIDERS
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_fundamentals_yfinance(ticker: str) -> Optional[dict]:
    """Fetch fundamentals from yfinance .info."""
    try:
        import yfinance as yf

        nse_ticker = f"{ticker}.NS"
        info = yf.Ticker(nse_ticker).info

        if not info:
            return None

        pe_ratio = info.get("trailingPE")

        eps_growth = info.get("earningsGrowth")
        if eps_growth is not None:
            eps_growth = eps_growth * 100 if abs(eps_growth) < 100 else eps_growth
        else:
            earnings_q = info.get("earningsQuarterlyGrowth")
            if earnings_q is not None:
                eps_growth = earnings_q * 100 if abs(earnings_q) < 100 else earnings_q

        rev_growth = info.get("revenueGrowth")
        if rev_growth is not None:
            rev_growth = rev_growth * 100 if abs(rev_growth) < 100 else rev_growth

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


def _fetch_fundamentals_nselib(ticker: str) -> Optional[dict]:
    """Fetch P/E ratio from nselib (bulk data, single call for all stocks)."""
    try:
        from nselib import capital_market

        today = date.today()
        to_date = today.strftime("%d-%m-%Y")

        df = capital_market.pe_ratio(trade_date=to_date)

        if df is None or df.empty:
            return None

        sym_col = [c for c in df.columns if "symbol" in c.lower()][0]
        r = df[df[sym_col].str.strip() == ticker]

        if r.empty:
            return None

        pe_val = r.iloc[0].get("SYMBOLP/E") or r.iloc[0].get("ADJUSTEDP/E")
        if pe_val is not None:
            pe_val = float(pe_val)

        return {
            "pe_ratio": pe_val,
            "eps_growth": None,
            "rev_growth": None,
            "roe": None,
        }

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROVIDER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class DataProvider:
    """
    Multi-source data provider with automatic fallback.

    Usage:
        provider = DataProvider()
        df = provider.fetch_stock("RELIANCE", period="1y")
        fund = provider.fetch_fundamentals("RELIANCE")
    """

    def __init__(self, use_cache: bool = True):
        """
        Args:
            use_cache: Whether to use disk cache for API responses
        """
        self.use_cache = use_cache

        # Track which provider was last used (for UI display)
        self.last_provider = None
        self.last_error = None

    def fetch_stock(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data with provider fallback chain.

        Priority:
          1. Cache (if enabled)
          2. jugaad-data
          3. yfinance
          4. nselib
        """
        self.last_provider = None
        self.last_error = None

        # Check cache first
        if self.use_cache:
            cached = _get_cached(ticker, period, "cache")
            if cached is not None:
                self.last_provider = "cache"
                return cached

        # Provider chain
        providers = [
            ("jugaad", lambda: _fetch_jugaad(ticker, period)),
            ("yfinance", lambda: _fetch_yfinance(ticker, period)),
            ("nselib", lambda: _fetch_nselib(ticker, period)),
        ]

        for name, fetch_fn in providers:
            try:
                df = fetch_fn()
                if df is not None and not df.empty and len(df) >= 50:
                    self.last_provider = name
                    if self.use_cache:
                        _set_cached(ticker, period, "cache", df)
                    return df
            except Exception as e:
                self.last_error = f"{name}: {str(e)}"
                continue

        self.last_error = "All providers failed"
        return None

    def fetch_index(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch index data with provider fallback."""
        self.last_provider = None

        if self.use_cache:
            cached = _get_cached(ticker, period, "index_cache")
            if cached is not None:
                self.last_provider = "cache"
                return cached

        providers = [
            ("jugaad", lambda: _fetch_jugaad_index(ticker, period)),
            ("yfinance", lambda: _fetch_yfinance_index(ticker, period)),
        ]

        for name, fetch_fn in providers:
            try:
                df = fetch_fn()
                if df is not None and not df.empty:
                    self.last_provider = name
                    if self.use_cache:
                        _set_cached(ticker, period, "index_cache", df)
                    return df
            except Exception:
                continue

        return None

    def fetch_fundamentals(self, ticker: str) -> Optional[dict]:
        """
        Fetch fundamental data with provider fallback.

        Priority:
          1. yfinance
          2. nselib
        """
        self.last_provider = None

        providers = [
            ("yfinance", lambda: _fetch_fundamentals_yfinance(ticker)),
            ("nselib", lambda: _fetch_fundamentals_nselib(ticker)),
        ]

        for name, fetch_fn in providers:
            try:
                fund = fetch_fn()
                if fund is not None:
                    self.last_provider = name
                    return fund
            except Exception:
                continue

        return None

    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            _ensure_cache_dir()
