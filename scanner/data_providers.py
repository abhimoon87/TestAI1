"""
Multi-source data provider with fallback chain for Indian stock market.

Provider priority for OHLCV:
  1. jugaad-data — NSE official API, no auth needed
  2. yfinance — Yahoo Finance, no auth needed
  3. nselib — NSE library, no auth needed

Provider priority for Fundamentals:
  1. Finnhub — Institutional-grade data (free tier)
  2. Twelve Data — Stock market data with fundamentals (free tier: 800 calls/day)
  3. Alpha Vantage — Technical indicators + fundamentals (free API key)
  4. yfinance .info — Detailed financial data
  5. nselib pe_ratio — Bulk P/E ratio for all stocks

All providers normalize data to a common DataFrame format:
  columns = [open, high, low, close, volume]
"""

import logging
import os
import json
import time
import hashlib
from datetime import date, datetime, timedelta
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Load API keys from config file if exists
_config_file = Path(__file__).parent / "api_config.json"
if _config_file.exists():
    try:
        with open(_config_file, "r") as f:
            _config = json.load(f)
            for key, value in _config.items():
                if key not in os.environ:  # Don't override existing env vars
                    os.environ[key] = value
    except Exception as e:
        logger.debug("Could not load api_config.json: %s", e)

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
    except Exception as e:
        logger.debug("Cache read failed for %s: %s", ticker, e)
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
    except Exception as e:
        logger.debug("Cache write failed for %s: %s", ticker, e)


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
    except Exception as e:
        logger.debug("jugaad-data failed for %s: %s", ticker, e)
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
    except Exception as e:
        logger.debug("jugaad-data index failed for %s: %s", ticker, e)
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
    except Exception as e:
        logger.debug("yfinance failed for %s: %s", ticker, e)
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
    except Exception as e:
        logger.debug("yfinance index failed for %s: %s", ticker, e)
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
    except Exception as e:
        logger.debug("nselib failed for %s: %s", ticker, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL DATA PROVIDERS
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_fundamentals_finnhub(ticker: str) -> Optional[dict]:
    """Fetch fundamentals from Finnhub (free tier, institutional-grade)."""
    try:
        import requests

        # Finnhub uses .NS suffix for NSE stocks
        finnhub_ticker = f"{ticker}.NS"
        api_key = os.environ.get("FINNHUB_API_KEY", "")

        if not api_key:
            return None

        # Get basic financials
        url = f"https://finnhub.io/api/v1/stock/metric?symbol={finnhub_ticker}&metric=all&token={api_key}"
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if not data or "metric" not in data:
            return None

        metric = data["metric"]

        pe_ratio = metric.get("peTTM") or metric.get("peBasicExclExtraTTM")
        roe = metric.get("roeTTM")
        if roe is not None:
            roe = roe * 100 if abs(roe) <= 1 else roe

        # Get earnings data for growth
        eps_growth = None
        rev_growth = None

        try:
            earnings_url = f"https://finnhub.io/api/v1/stock/earnings?symbol={finnhub_ticker}&token={api_key}"
            earnings_resp = requests.get(earnings_url, timeout=10)
            earnings_data = earnings_resp.json()

            if earnings_data and len(earnings_data) >= 2:
                # Compare latest two quarters
                latest = earnings_data[0]
                prev = earnings_data[1]
                if prev.get("eps") and prev["eps"] != 0 and latest.get("eps"):
                    eps_growth = ((latest["eps"] - prev["eps"]) / abs(prev["eps"])) * 100
                if prev.get("revenue") and prev["revenue"] != 0 and latest.get("revenue"):
                    rev_growth = ((latest["revenue"] - prev["revenue"]) / abs(prev["revenue"])) * 100
        except Exception as e:
            logger.debug("Finnhub earnings fetch failed for %s: %s", ticker, e)

        return {
            "pe_ratio": pe_ratio,
            "eps_growth": eps_growth,
            "rev_growth": rev_growth,
            "roe": roe,
        }

    except Exception as e:
        logger.debug("Finnhub fundamentals failed for %s: %s", ticker, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER: Twelve Data (Free tier: 800 calls/day, 8/min)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_fundamentals_twelve_data(ticker: str) -> Optional[dict]:
    """Fetch fundamentals from Twelve Data (free tier)."""
    try:
        import requests

        api_key = os.environ.get("TWELVEDATA_API_KEY", "")
        if not api_key:
            return None

        # Twelve Data uses .NS suffix for NSE stocks
        symbol = f"{ticker}.NS"

        # Get fundamental data
        url = f"https://api.twelvedata.com/fundamentals?symbol={symbol}&apikey={api_key}"
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if not data or "fundamentals" not in data:
            return None

        fund = data["fundamentals"]

        # Extract metrics from different sections
        overview = fund.get("overview", {})
        valuation = fund.get("valuation", {})
        financials = fund.get("financials", {})

        # P/E ratio
        pe_ratio = None
        pe_raw = valuation.get("pe_ratio") or overview.get("pe_ratio")
        if pe_raw and pe_raw != "-":
            try:
                pe_ratio = float(pe_raw)
            except (ValueError, TypeError):
                pass

        # ROE
        roe = None
        roe_raw = financials.get("return_on_equity") or overview.get("roe")
        if roe_raw and roe_raw != "-":
            try:
                roe_val = float(roe_raw)
                roe = roe_val * 100 if abs(roe_val) <= 1 else roe_val
            except (ValueError, TypeError):
                pass

        # EPS growth (from earnings)
        eps_growth = None
        eps_raw = overview.get("eps_growth") or financials.get("eps_growth")
        if eps_raw and eps_raw != "-":
            try:
                eps_val = float(eps_raw)
                eps_growth = eps_val * 100 if abs(eps_val) <= 1 else eps_val
            except (ValueError, TypeError):
                pass

        # Revenue growth
        rev_growth = None
        rev_raw = overview.get("revenue_growth") or financials.get("revenue_growth")
        if rev_raw and rev_raw != "-":
            try:
                rev_val = float(rev_raw)
                rev_growth = rev_val * 100 if abs(rev_val) <= 1 else rev_val
            except (ValueError, TypeError):
                pass

        if pe_ratio is None and roe is None and eps_growth is None:
            return None

        return {
            "pe_ratio": pe_ratio,
            "eps_growth": eps_growth,
            "rev_growth": rev_growth,
            "roe": roe,
        }

    except Exception as e:
        logger.debug("Twelve Data fundamentals failed for %s: %s", ticker, e)
        return None


def _fetch_fundamentals_alpha_vantage(ticker: str) -> Optional[dict]:
    """Fetch fundamentals from Alpha Vantage (free API key)."""
    try:
        import requests

        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            return None

        # Get overview data
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}.NS&apikey={api_key}"
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if not data or "PERatio" not in data:
            return None

        pe_raw = data.get("PERatio")
        pe_ratio = float(pe_raw) if pe_raw is not None else None
        roe_raw = data.get("ReturnOnEquityTTM")
        roe = float(roe_raw) * 100 if roe_raw is not None else None
        eps_raw = data.get("EPSGrowthTTM")
        eps_growth = float(eps_raw) * 100 if eps_raw is not None else None
        rev_raw = data.get("RevenueGrowthTTM")
        rev_growth = float(rev_raw) * 100 if rev_raw is not None else None

        return {
            "pe_ratio": pe_ratio,
            "eps_growth": eps_growth,
            "rev_growth": rev_growth,
            "roe": roe,
        }

    except Exception as e:
        logger.debug("Alpha Vantage failed for %s: %s", ticker, e)
        return None


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
            eps_growth = eps_growth * 100 if abs(eps_growth) <= 1 else eps_growth
        else:
            earnings_q = info.get("earningsQuarterlyGrowth")
            if earnings_q is not None:
                eps_growth = earnings_q * 100 if abs(earnings_q) <= 1 else earnings_q

        rev_growth = info.get("revenueGrowth")
        if rev_growth is not None:
            rev_growth = rev_growth * 100 if abs(rev_growth) <= 1 else rev_growth

        roe = info.get("returnOnEquity")
        if roe is not None:
            roe = roe * 100 if abs(roe) <= 1 else roe

        return {
            "pe_ratio": pe_ratio,
            "eps_growth": eps_growth,
            "rev_growth": rev_growth,
            "roe": roe,
        }

    except Exception as e:
        logger.debug("yfinance fundamentals failed for %s: %s", ticker, e)
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

    except Exception as e:
        logger.debug("nselib P/E failed for %s: %s", ticker, e)
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
        
        # Metrics for observability
        self._metrics = {
            "stock_fetches": {"total": 0, "success": 0, "cache_hits": 0},
            "index_fetches": {"total": 0, "success": 0, "cache_hits": 0},
            "provider_stats": {},  # per-provider: success, failures, avg_latency
            "errors": [],  # recent errors
        }

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
        import time
        start_time = time.time()

        self._metrics["stock_fetches"]["total"] += 1

        # Check cache first
        if self.use_cache:
            cached = _get_cached(ticker, period, "cache")
            if cached is not None:
                self.last_provider = "cache"
                self._metrics["stock_fetches"]["cache_hits"] += 1
                self._metrics["stock_fetches"]["success"] += 1
                return cached

        # Provider chain
        providers = [
            ("jugaad", lambda: _fetch_jugaad(ticker, period)),
            ("yfinance", lambda: _fetch_yfinance(ticker, period)),
            ("nselib", lambda: _fetch_nselib(ticker, period)),
        ]

        for name, fetch_fn in providers:
            provider_start = time.time()
            try:
                df = fetch_fn()
                latency = time.time() - provider_start
                if df is not None and not df.empty and len(df) >= 50:
                    self.last_provider = name
                    if self.use_cache:
                        _set_cached(ticker, period, "cache", df)
                    self._metrics["stock_fetches"]["success"] += 1
                    self._record_provider_metric(name, True, latency)
                    return df
            except Exception as e:
                latency = time.time() - provider_start
                self.last_error = f"{name}: {str(e)}"
                self._record_provider_metric(name, False, latency)
                self._metrics["errors"].append(f"{name}: {str(e)}")
                if len(self._metrics["errors"]) > 20:
                    self._metrics["errors"] = self._metrics["errors"][-20:]
                continue

        self._metrics["errors"].append("All providers failed")
        self.last_error = "All providers failed"
        return None

    def _record_provider_metric(self, provider: str, success: bool, latency: float):
        """Record provider metrics for observability."""
        if provider not in self._metrics["provider_stats"]:
            self._metrics["provider_stats"][provider] = {"success": 0, "failures": 0, "total_latency": 0.0}
        stats = self._metrics["provider_stats"][provider]
        if success:
            stats["success"] += 1
        else:
            stats["failures"] += 1
        stats["total_latency"] += latency

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
            except Exception as e:
                logger.debug("Index provider %s failed for %s: %s", name, ticker, e)
                continue

        return None

    def fetch_fundamentals(self, ticker: str) -> Optional[dict]:
        """
        Fetch fundamental data with provider fallback.

        Priority:
          1. Finnhub (institutional-grade, free tier)
          2. Alpha Vantage (technical indicators + fundamentals)
          3. yfinance (detailed financial data)
          4. nselib (bulk P/E ratio)
        """
        self.last_provider = None

        providers = [
            ("finnhub", lambda: _fetch_fundamentals_finnhub(ticker)),
            ("twelve_data", lambda: _fetch_fundamentals_twelve_data(ticker)),
            ("alpha_vantage", lambda: _fetch_fundamentals_alpha_vantage(ticker)),
            ("yfinance", lambda: _fetch_fundamentals_yfinance(ticker)),
            ("nselib", lambda: _fetch_fundamentals_nselib(ticker)),
        ]

        for name, fetch_fn in providers:
            try:
                fund = fetch_fn()
                if fund is not None:
                    self.last_provider = name
                    return fund
            except Exception as e:
                logger.debug("Fundamentals provider %s failed for %s: %s", name, ticker, e)
                continue

        return None

    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            _ensure_cache_dir()

    def get_metrics(self) -> dict:
        """Return provider metrics for monitoring/debugging."""
        # Calculate average latencies
        provider_stats = {}
        for name, stats in self._metrics["provider_stats"].items():
            total = stats["success"] + stats["failures"]
            avg_latency = stats["total_latency"] / total if total > 0 else 0.0
            provider_stats[name] = {
                "success": stats["success"],
                "failures": stats["failures"],
                "success_rate": stats["success"] / total if total > 0 else 0.0,
                "avg_latency_sec": round(avg_latency, 3),
            }
        return {
            "stock_fetches": self._metrics["stock_fetches"],
            "index_fetches": self._metrics["index_fetches"],
            "provider_stats": provider_stats,
            "recent_errors": self._metrics["errors"][-10:],
        }
