"""
Multi-source data provider with fallback chain for Indian stock market.

Provider priority for OHLCV:
  1. jugaad-data — NSE official API, no auth needed
  2. yfinance — Yahoo Finance, no auth needed
  3. nselib — NSE library, no auth needed

Provider priority for Fundamentals:
  1. Finnhub — Institutional-grade data (free tier)
  2. Alpha Vantage — Technical indicators + fundamentals (free API key)
  3. yfinance .info — Detailed financial data
  4. nselib pe_ratio — Bulk P/E ratio for all stocks

All providers normalize data to a common DataFrame format:
  columns = [open, high, low, close, volume]
"""

import glob
import hashlib
import json
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

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


def _normalize_cache_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map a frame onto the canonical tz-naive IST trade-date calendar.

    The canonical implementation lives in ``data_fetcher``, which imports
    THIS module -- so it is lazy-imported here to avoid the cycle.  Applying
    it on every cache write AND read means no code path (batch or per-ticker,
    yfinance or jugaad/nselib fallback) can surface the UTC-close 18:30
    stamps that once made cross-ticker date unions double-count every day.
    """
    from .data_fetcher import _normalize_daily_index  # lazy: data_fetcher imports us
    return _normalize_daily_index(df)


_PRUNE_LOCK = threading.Lock()
_last_prune_ts = 0.0
PRUNE_INTERVAL_SECONDS = 3600  # at most one stale-cache sweep per process per hour


def prune_stale_cache(force: bool = False) -> int:
    """Delete cache entries from earlier days -- unreachable dead weight.

    The cache key embeds ``date.today()`` (see ``_cache_key``), so an entry
    written on any previous day can never be read again -- but one file per
    (ticker, period, provider, day) stays on disk forever unless pruned.
    Sweeping is rate-limited per process (``PRUNE_INTERVAL_SECONDS``) so scan
    starts stay cheap; only non-today files are ever deleted, so a sweep can
    not race a concurrent writer or reader (both use today's key).

    Returns the number of stale (pkl, meta) pairs removed.
    """
    global _last_prune_ts
    now = time.time()
    with _PRUNE_LOCK:
        if not force and now - _last_prune_ts < PRUNE_INTERVAL_SECONDS:
            return 0
        _last_prune_ts = now

    today = date.today().isoformat()
    removed = 0
    try:
        for meta in glob.glob(os.path.join(CACHE_DIR, "*.meta")):
            try:
                with open(meta, "r") as f:
                    ts = json.load(f).get("timestamp", "")
            except Exception:
                continue  # unreadable/corrupt meta -- leave the pair alone
            if ts[:10] == today:
                continue
            try:
                os.remove(meta[:-5] + ".pkl")
            except OSError:
                pass
            try:
                os.remove(meta)
            except OSError:
                pass
            removed += 1
    except Exception as e:
        logger.debug("Stale-cache prune failed: %s", e)
    if removed:
        logger.info("Pruned %d stale cache entrie(s) from previous days", removed)
    return removed


def cache_health() -> dict:
    """Price-cache census: fresh vs stale pkl+meta pairs on disk.

    Returns ``{price_entries, stale_entries, last_prune}`` where
    ``price_entries`` is the TOTAL pair count on disk (today's reachable
    entries plus every other day's unreachable leftovers) and
    ``stale_entries`` is the unreachable subset.  ``last_prune`` is an ISO
    timestamp of the last ``prune_stale_cache`` sweep in this process (""
    when never pruned).
    """
    fresh = stale = 0
    today = date.today().isoformat()
    try:
        for meta in glob.glob(os.path.join(CACHE_DIR, "*.meta")):
            try:
                with open(meta, "r") as f:
                    ts = json.load(f).get("timestamp", "")
            except Exception:
                continue
            if ts[:10] == today:
                fresh += 1
            else:
                stale += 1
    except Exception:
        pass
    with _PRUNE_LOCK:
        last_ts = _last_prune_ts
    last_prune = datetime.fromtimestamp(last_ts).isoformat(timespec="minutes") if last_ts > 0 else ""
    return {"price_entries": fresh + stale, "stale_entries": stale,
            "last_prune": last_prune}


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(ticker: str, period: str, provider: str) -> str:
    """Generate a cache file key."""
    today = date.today().isoformat()
    raw = f"{ticker}_{period}_{provider}_{today}"
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()


def _get_cached(ticker: str, period: str, provider: str) -> pd.DataFrame | None:
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

        return _normalize_cache_frame(pd.read_pickle(cache_file))
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
        df = _normalize_cache_frame(df)
        df.to_pickle(cache_file)
        with open(meta_file, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "rows": len(df)}, f)
    except Exception as e:
        logger.debug("Cache write failed for %s: %s", ticker, e)


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER: jugaad-data (NSE Official API)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_jugaad(ticker: str, period: str) -> pd.DataFrame | None:
    """Fetch OHLCV from NSE via jugaad-data. No auth needed."""
    try:
        from datetime import date, timedelta

        from jugaad_data.nse import stock_df

        period_days = {"6mo": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
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

        # Set DATE as index (needed for resampling). jugaad returns rows
        # newest-first — flip to ascending so .iloc[-1] is the latest bar.
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df.index.name = None
            df = df.sort_index()

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


def _fetch_jugaad_index(ticker: str, period: str) -> pd.DataFrame | None:
    """Fetch index data from NSE via jugaad-data."""
    try:
        from datetime import date, timedelta

        from jugaad_data.nse import index_df

        period_days = {"6mo": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
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
        date_col = None
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
            elif "date" in cl:
                # index_df() reports the trading day under HistoricalDate.
                # Keep it so it can become the DatetimeIndex below — without
                # this the frame has a RangeIndex and Relative-Strength
                # date alignment silently breaks (epoch-1970 dates).
                date_col = c

        df = df.rename(columns=col_map)
        cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in cols if c in df.columns]

        if len(available) < 4:
            return None

        # Pull the date column out first, then keep only OHLCV.
        if date_col is not None and date_col not in available:
            df = df[[date_col] + available].copy()
        else:
            df = df[available].copy()

        if date_col is not None and date_col in df.columns:
            dates = pd.to_datetime(df[date_col])
            df = df.drop(columns=[date_col])
            df.index = dates
            df.index.name = None

        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            return None

        # jugaad returns rows newest-first — flip to ascending so .iloc[-1]
        # and date-mask alignment in the scorers use the latest bar.
        df = df.sort_index()

        return df.dropna()

    except ImportError:
        return None
    except Exception as e:
        logger.debug("jugaad-data index failed for %s: %s", ticker, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER: yfinance (Yahoo Finance)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_yfinance(ticker: str, period: str) -> pd.DataFrame | None:
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


def _fetch_yfinance_index(ticker: str, period: str) -> pd.DataFrame | None:
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

def _fetch_nselib(ticker: str, period: str) -> pd.DataFrame | None:
    """Fetch OHLCV from NSE via nselib."""
    try:
        from nselib import capital_market

        period_days = {"6mo": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
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

def _fetch_fundamentals_finnhub(ticker: str) -> dict | None:
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
            roe = roe * 100 if abs(roe) < 1 else roe

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


def _fetch_fundamentals_alpha_vantage(ticker: str) -> dict | None:
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

        pe_ratio = float(data.get("PERatio", 0)) or None
        roe = float(data.get("ReturnOnEquityTTM", 0)) * 100 if data.get("ReturnOnEquityTTM") else None
        eps_growth = float(data.get("EPSGrowthTTM", 0)) * 100 if data.get("EPSGrowthTTM") else None
        rev_growth = float(data.get("RevenueGrowthTTM", 0)) * 100 if data.get("RevenueGrowthTTM") else None

        return {
            "pe_ratio": pe_ratio,
            "eps_growth": eps_growth,
            "rev_growth": rev_growth,
            "roe": roe,
        }

    except Exception as e:
        logger.debug("Alpha Vantage failed for %s: %s", ticker, e)
        return None


def _fetch_fundamentals_yfinance(ticker: str) -> dict | None:
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

    except Exception as e:
        logger.debug("yfinance fundamentals failed for %s: %s", ticker, e)
        return None


def _fetch_fundamentals_nselib(ticker: str) -> dict | None:
    """Fetch P/E ratio from nselib (bulk data, single call for all stocks)."""
    try:
        from nselib import capital_market

        today = date.today()
        to_date = today.strftime("%d-%m-%Y")

        df = capital_market.pe_ratio(trade_date=to_date)

        if df is None or df.empty:
            return None

        sym_col = next(c for c in df.columns if "symbol" in c.lower())
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

# Sentinel + runner used to bound individual provider calls (e.g. in the
# batch fallback pass) so one hung request cannot stall a worker forever.
_TIMEOUT = object()


def _call_with_timeout(fn, timeout: float):
    """Run fn on a daemon thread; return _TIMEOUT if it exceeds timeout.

    The worker thread keeps running in the background when it times out (it
    is a daemon, so it can never block interpreter shutdown). Callers treat
    a _TIMEOUT return exactly like a provider that returned nothing and move
    on to the next provider in the chain.
    """
    box: dict = {}

    def _run():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - surfaced to caller below
            box["error"] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return _TIMEOUT
    if "error" in box:
        raise box["error"]
    return box.get("value")


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

    def fetch_stock(self, ticker: str, period: str = "1y",
                    skip: tuple[str, ...] = (),
                    provider_timeout: float | None = None) -> pd.DataFrame | None:
        """
        Fetch OHLCV data with provider fallback chain.

        Priority:
          1. Cache (if enabled)
          2. jugaad-data
          3. yfinance
          4. nselib

        Args:
            skip: Provider names to exclude from the chain. Used by the
                  batch-download fallback path, where yfinance just failed
                  at scale (rate limit / outage) and should not be retried
                  per ticker.
            provider_timeout: When set, each provider call is capped at this
                  many seconds. A provider that exceeds the cap is treated
                  like one that returned no data (its thread keeps running
                  in the background as a daemon). Used by the batch fallback
                  so dead symbols fail fast instead of stalling a worker.
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
            if name in skip:
                continue
            try:
                if provider_timeout:
                    df = _call_with_timeout(fetch_fn, provider_timeout)
                    if df is _TIMEOUT:
                        self.last_error = f"{name}: timed out after {provider_timeout}s"
                        continue
                else:
                    df = fetch_fn()
                if df is not None and not df.empty and len(df) >= 50:
                    self.last_provider = name
                    if self.use_cache:
                        _set_cached(ticker, period, "cache", df)
                    return df
            except Exception as e:
                self.last_error = f"{name}: {e!s}"
                continue

        self.last_error = "All providers failed"
        return None

    def fetch_index(self, ticker: str, period: str = "1y") -> pd.DataFrame | None:
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

    def fetch_fundamentals(self, ticker: str) -> dict | None:
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
