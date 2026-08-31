"""
Dynamic symbol fetcher for NSE/BSE universes.
Fetches live symbol lists from NSE via nselib.
BSE support is limited due to anti-scraping measures on bseindia.com.
"""

import logging
import time
from datetime import date, timedelta
from functools import lru_cache
from typing import Optional

import nselib.capital_market as cm

logger = logging.getLogger(__name__)

# Cache TTL in seconds (4 hours)
CACHE_TTL_SECONDS = 4 * 3600

# In-memory cache with timestamps
_cache = {}
_cache_timestamps = {}


def _is_cache_valid(key: str) -> bool:
    """Check if cached data is still valid."""
    if key not in _cache_timestamps:
        return False
    return (time.time() - _cache_timestamps[key]) < CACHE_TTL_SECONDS


def _cache_get(key: str) -> Optional[list]:
    """Get cached value if valid."""
    if _is_cache_valid(key):
        return _cache.get(key)
    return None


def _cache_set(key: str, value: list):
    """Set cache value with timestamp."""
    _cache[key] = value
    _cache_timestamps[key] = time.time()


def _fetch_with_cache(key: str, fetch_func, fallback: list = None) -> list:
    """Fetch data with caching and static fallback."""
    # Try cache first
    cached = _cache_get(key)
    if cached is not None:
        return cached
    
    # Try live fetch
    try:
        result = fetch_func()
        if result:
            _cache_set(key, result)
            return result
    except Exception as e:
        logger.warning("Live fetch failed for %s: %s", key, e)
    
    # Return fallback if available
    if fallback is not None:
        logger.info("Using static fallback for %s", key)
        return fallback
    
    return []


# Static fallbacks (updated periodically)
_STATIC_FALLBACKS = {
    "mainboard": [],  # Will be populated from universes if needed
    "fno": [],
    "sme": [],
    "nifty50": [],
    "niftynext50": [],
    "midcap150": [],
    "smallcap250": [],
}


def _get_static_fallback(key: str) -> list:
    """Get static fallback from universes module."""
    if key == "mainboard":
        try:
            from .universes import CASH_MARKET
            return CASH_MARKET
        except Exception:
            return []
    elif key == "fno":
        try:
            from .universes import FNO_STOCKS
            return FNO_STOCKS
        except Exception:
            return []
    elif key == "nifty50":
        try:
            from .universes import NIFTY_50
            return NIFTY_50
        except Exception:
            return []
    elif key == "niftynext50":
        try:
            from .universes import NIFTY_NEXT_50
            return NIFTY_NEXT_50
        except Exception:
            return []
    elif key == "midcap150":
        try:
            from .universes import NIFTY_MIDCAP_100
            return NIFTY_MIDCAP_100
        except Exception:
            return []
    elif key == "smallcap250":
        try:
            from .universes import NIFTY_SMALLCAP_100
            return NIFTY_SMALLCAP_100
        except Exception:
            return []
    return []


def fetch_nse_mainboard() -> list[str]:
    """Fetch all NSE mainboard equity symbols (with caching)."""
    return _fetch_with_cache("mainboard", 
        lambda: cm.equity_list()["SYMBOL"].str.strip().tolist(),
        fallback=_get_static_fallback("mainboard"))


def fetch_nse_fno() -> list[str]:
    """Fetch NSE F&O eligible equity symbols (with caching)."""
    return _fetch_with_cache("fno",
        lambda: cm.fno_equity_list()["symbol"].str.strip().tolist(),
        fallback=_get_static_fallback("fno"))


def fetch_nse_sme(trade_date: Optional[date] = None) -> list[str]:
    """
    Fetch NSE SME platform equity symbols.
    
    Args:
        trade_date: Trading date. Defaults to most recent weekday.
    """
    # Use local variable to avoid mutating caller's date object
    check_date = trade_date if trade_date is not None else date.today()
    while check_date.weekday() >= 5:
        check_date -= timedelta(days=1)
    
    for days_back in range(10):
        d = check_date - timedelta(days=days_back)
        try:
            df = cm.sme_band_complete(trade_date=d.strftime("%d-%m-%Y"))
            if "Symbol" in df.columns:
                return df["Symbol"].str.strip().tolist()
            elif "symbol" in df.columns:
                return df["symbol"].str.strip().tolist()
            elif "SYMBOL" in df.columns:
                return df["SYMBOL"].str.strip().tolist()
        except (KeyError, ValueError, ConnectionError, TimeoutError) as e:
            logger.debug("SME fetch failed for %s: %s", d, e)
            continue
    logger.warning("SME symbol fetch failed for all recent dates")
    return _get_static_fallback("sme")


def fetch_nse_index_list(index_name: str) -> list[str]:
    """
    Fetch NSE index constituent symbols.
    
    Args:
        index_name: One of 'nifty50', 'niftynext50', 'midcap150', 'smallcap250'
    """
    func_map = {
        "nifty50": cm.nifty50_equity_list,
        "niftynext50": cm.niftynext50_equity_list,
        "midcap150": cm.niftymidcap150_equity_list,
        "smallcap250": cm.niftysmallcap250_equity_list,
    }
    if index_name not in func_map:
        raise ValueError(f"Unknown index: {index_name}. Valid: {list(func_map.keys())}")
    
    key = index_name
    fallback = _get_static_fallback(key)
    return _fetch_with_cache(key, 
        lambda: _fetch_index_list_raw(func_map[index_name]),
        fallback=fallback)


def _fetch_index_list_raw(func) -> list[str]:
    """Internal function to fetch index list without caching."""
    df = func()
    if "Symbol" in df.columns:
        return df["Symbol"].str.strip().tolist()
    elif "symbol" in df.columns:
        return df["symbol"].str.strip().tolist()
    elif "SYMBOL" in df.columns:
        return df["SYMBOL"].str.strip().tolist()
    return []


def fetch_all_nse_symbols() -> dict:
    """
    Fetch all NSE symbol lists at once.
    
    Returns:
        Dict with keys: mainboard, fno, sme, nifty50, niftynext50, midcap150, smallcap250
    """
    return {
        "mainboard": fetch_nse_mainboard(),
        "fno": fetch_nse_fno(),
        "sme": fetch_nse_sme(),
        "nifty50": fetch_nse_index_list("nifty50"),
        "niftynext50": fetch_nse_index_list("niftynext50"),
        "midcap150": fetch_nse_index_list("midcap150"),
        "smallcap250": fetch_nse_index_list("smallcap250"),
    }


def get_unique_nse_symbols() -> list[str]:
    """
    Get all unique NSE symbols across all segments (with caching).
    """
    return _fetch_with_cache("unique_nse",
        lambda: _compute_unique_nse(),
        fallback=_get_static_fallback("unique"))


def _compute_unique_nse() -> list[str]:
    """Internal function to compute unique symbols without caching."""
    all_data = fetch_all_nse_symbols()
    unique = set()
    for symbols in all_data.values():
        unique.update(s.upper().strip() for s in symbols if isinstance(s, str))
    return sorted(unique)


def _get_static_fallback(key: str) -> list:
    """Get static fallback from universes module."""
    if key == "mainboard":
        try:
            from .universes import CASH_MARKET
            return CASH_MARKET
        except Exception:
            return []
    elif key == "fno":
        try:
            from .universes import FNO_STOCKS
            return FNO_STOCKS
        except Exception:
            return []
    elif key == "nifty50":
        try:
            from .universes import NIFTY_50
            return NIFTY_50
        except Exception:
            return []
    elif key == "niftynext50":
        try:
            from .universes import NIFTY_NEXT_50
            return NIFTY_NEXT_50
        except Exception:
            return []
    elif key == "midcap150":
        try:
            from .universes import NIFTY_MIDCAP_100
            return NIFTY_MIDCAP_100
        except Exception:
            return []
    elif key == "smallcap250":
        try:
            from .universes import NIFTY_SMALLCAP_100
            return NIFTY_SMALLCAP_100
        except Exception:
            return []
    elif key == "sme":
        return []
    elif key == "unique":
        try:
            from .universes import CASH_MARKET
            return CASH_MARKET
        except Exception:
            return []
    return []


# ── BSE Support (Limited) ───────────────────────────────────────────────────
# BSE (bseindia.com) blocks automated access. No reliable public API for equity lists.
# Workarounds:
#   - Static universes in universes.py: BSE SENSEX, BSE MIDCAP, BSE SMALLCAP
#   - Cross-listing: NSE mainboard (2,559) covers most liquid BSE names
#   - yfinance supports BSE data via .BO suffix but has no listing function
#   - For full BSE universe (~5,500 symbols), use a paid data provider or manual CSV

def fetch_bse_static_universes() -> dict:
    """
    Return static BSE universes from universes.py.
    These are manually maintained and should be updated periodically.
    """
    try:
        from .universes import BSE_SENSEX, BSE_MIDCAP, BSE_SMALLCAP
        return {
            "BSE SENSEX": BSE_SENSEX,
            "BSE MIDCAP": BSE_MIDCAP,
            "BSE SMALLCAP": BSE_SMALLCAP,
        }
    except ImportError as e:
        logger.warning("Failed to import BSE static universes: %s", e)
        return {}


MAX_BSE_VALIDATE = 20

def validate_bse_symbols(symbols: list[str], max_check: int = MAX_BSE_VALIDATE) -> list[str]:
    """
    Validate BSE symbols by checking if they exist on yfinance (.BO suffix).
    Limited to max_check to avoid rate limits.
    Uses fast_info for lightweight check instead of full info.
    """
    import yfinance as yf
    valid = []
    for sym in symbols[:max_check]:
        try:
            ticker = yf.Ticker(f"{sym}.BO")
            # Use fast_info for lightweight existence check (no HTTP)
            fi = ticker.fast_info
            if fi and fi.get("symbol"):
                valid.append(sym)
        except (KeyError, ValueError, ConnectionError, TimeoutError) as e:
            logger.debug("BSE validation failed for %s: %s", sym, e)
            continue
    return valid


if __name__ == "__main__":
    all_data = fetch_all_nse_symbols()
    for k, v in all_data.items():
        print(f"{k}: {len(v)} symbols")
    unique = get_unique_nse_symbols()
    print(f"Unique NSE symbols: {len(unique)}")