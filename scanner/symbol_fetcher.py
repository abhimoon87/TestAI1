"""
Dynamic symbol fetcher for NSE/BSE universes.
Fetches live symbol lists from NSE via nselib.
BSE support is limited due to anti-scraping measures on bseindia.com.
"""

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import nselib.capital_market as cm

logger = logging.getLogger(__name__)

# Cache TTL in seconds (4 hours)
CACHE_TTL_SECONDS = 4 * 3600

# In-memory cache with timestamps
_cache: dict[str, list] = {}
_cache_timestamps: dict[str, float] = {}

# Disk cache for persistence across restarts
_DISK_CACHE_FILE = Path(__file__).parent / ".cache" / "symbols.json"


def _load_disk_cache():
    try:
        if _DISK_CACHE_FILE.exists():
            data = json.loads(_DISK_CACHE_FILE.read_text(encoding="utf-8"))
            now = time.time()
            for k, v in data.items():
                ts = v.get("_ts", 0)
                if now - ts < CACHE_TTL_SECONDS and isinstance(v.get("data"), list):
                    _cache[k] = v["data"]
                    _cache_timestamps[k] = ts
            if _cache:
                logger.debug("Disk symbol cache loaded: %d keys", len(_cache))
    except Exception as e:
        logger.debug("Disk cache load failed: %s", e)


def _save_disk_cache():
    try:
        _DISK_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: {"data": v, "_ts": _cache_timestamps.get(k, 0)} for k, v in _cache.items()}
        _DISK_CACHE_FILE.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as e:
        logger.debug("Disk cache save failed: %s", e)


# Load on import
_load_disk_cache()


def _is_cache_valid(key: str) -> bool:
    """Check if cached data is still valid."""
    if key not in _cache_timestamps:
        return False
    return (time.time() - _cache_timestamps[key]) < CACHE_TTL_SECONDS


def _cache_get(key: str) -> list | None:
    """Get cached value if valid."""
    if _is_cache_valid(key):
        return _cache.get(key)
    return None


def _cache_set(key: str, value: list):
    """Set cache value with timestamp (persists to disk)."""
    _cache[key] = value
    _cache_timestamps[key] = time.time()
    try:
        _save_disk_cache()
    except Exception:
        pass


def _fetch_with_cache(key: str, fetch_func, fallback: list | None = None) -> list:
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


def fetch_nse_sme(trade_date: date | None = None) -> list[str]:
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
    elif key == "bse_all":
        try:
            from .universes import BSE_MIDCAP, BSE_SENSEX, BSE_SMALLCAP, CASH_MARKET
            # Approximate BSE ALL as static BSE + NSE cash (covers dual-listed)
            return sorted(set(BSE_SENSEX + BSE_MIDCAP + BSE_SMALLCAP + CASH_MARKET))
        except Exception:
            return []
    elif key in ("all_market", "unique_nse", "all_nse"):
        try:
            from .universes import NIFTY_BROAD
            return NIFTY_BROAD
        except Exception:
            return []
    return []


# ── BSE Support — Full Market (~5,900 unique) ───────────────────────────────
# BSE blocks scraping, but we can fetch via:
#   1. BSE India API (with browser headers)
#   2. GitHub mirrors of BSE/NSE lists (fallback)
#   3. Static universes (last resort)

BSE_CSV_MIRRORS = [
    # GitHub mirrors — raw CSVs with BSE symbols (Security Code / Symbol)
    "https://raw.githubusercontent.com/pushkar-anand/bse-nse/master/bse.csv",
    "https://raw.githubusercontent.com/jignesh91/nse-bse-list/master/bse.csv",
]

NSE_CSV_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"


def _fetch_bse_via_api() -> list[str]:
    """Try BSE India JSON API (requires browser-like headers)."""
    import requests

    url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
    params = {"Group": "", "Scripcode": "", "industry": "", "segment": "Equity", "status": "Active"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.bseindia.com/",
        "Accept": "application/json",
    }
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # API returns {"Table": [{"SCRIP_CD": "...", "Scrip_Name": "...", "SYMBOL": "..."}]}
    symbols: list[str] = []
    table = data.get("Table") or data.get("table") or []
    if isinstance(table, list):
        for row in table:
            # Prefer SYMBOL / Scrip_Name / SCRIP_CD
            sym = (row.get("SYMBOL") or row.get("Scrip_Name") or row.get("SCRIP_CD") or "").strip().upper()
            # SYMBOL may be like "RELIANCE" or empty; SCRIP_CD is numeric code, skip if numeric
            if sym and not sym.isdigit() and len(sym) <= 20 and " " not in sym:
                symbols.append(sym)
    return sorted(set(symbols))


def _fetch_bse_via_csv_mirror() -> list[str]:
    """Fetch BSE symbols from GitHub CSV mirrors."""
    import csv
    import io

    import requests

    for url in BSE_CSV_MIRRORS:
        try:
            resp = requests.get(url, timeout=12)
            resp.raise_for_status()
            text = resp.text
            reader = csv.DictReader(io.StringIO(text))
            # Try common column names: SYMBOL, Symbol, Scrip Code, Security Code
            col = None
            if reader.fieldnames:
                for cand in ["SYMBOL", "Symbol", "symbol", "Scrip Code", "Security Code", "SC_CODE"]:
                    if cand in reader.fieldnames:
                        col = cand
                        break
                if col is None:
                    col = reader.fieldnames[0]
            symbols = []
            # Need to re-read after sniffing header
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                sym = str(row.get(col, "")).strip().upper()
                if sym and not sym.isdigit() and len(sym) <= 20 and " " not in sym:
                    # Some CSVs include .BO suffix or extra spaces
                    sym = sym.replace(".BO", "").replace(".NS", "").strip()
                    if sym:
                        symbols.append(sym)
            if len(symbols) > 1000:
                return sorted(set(symbols))
        except Exception as e:
            logger.debug("BSE CSV mirror failed %s: %s", url, e)
            continue
    return []


def _fetch_nse_via_csv() -> list[str]:
    """Fetch NSE symbols via archives.nseindia.com CSV (fallback)."""
    import csv
    import io

    import requests

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(NSE_CSV_URL, headers=headers, timeout=12)
        resp.raise_for_status()
        text = resp.text
        reader = csv.DictReader(io.StringIO(text))
        col = "SYMBOL" if "SYMBOL" in (reader.fieldnames or []) else (reader.fieldnames[0] if reader.fieldnames else "SYMBOL")
        reader = csv.DictReader(io.StringIO(text))
        symbols = [str(r.get(col, "")).strip().upper() for r in reader if str(r.get(col, "")).strip()]
        symbols = [s for s in symbols if s and len(s) <= 20 and " " not in s]
        if len(symbols) > 1000:
            return sorted(set(symbols))
    except Exception as e:
        logger.debug("NSE CSV fetch failed: %s", e)
    return []


def fetch_bse_all_live() -> list[str]:
    """Fetch all BSE Active Equity symbols (~4,000-5,500). Cached 4h."""
    def _do():
        # Try live API first
        try:
            syms = _fetch_bse_via_api()
            if len(syms) > 1000:
                logger.info("BSE API returned %d symbols", len(syms))
                return syms
        except Exception as e:
            logger.debug("BSE API failed: %s", e)
        # Try CSV mirrors
        syms = _fetch_bse_via_csv_mirror()
        if len(syms) > 1000:
            logger.info("BSE CSV mirror returned %d symbols", len(syms))
            return syms
        # Try NSE CSV as proxy (covers many BSE dual-listed)
        syms = _fetch_nse_via_csv()
        if syms:
            logger.info("BSE fallback via NSE CSV: %d symbols", len(syms))
            return syms
        return []

    return _fetch_with_cache("bse_all", _do, fallback=_get_static_fallback("bse_all"))


def fetch_bse_static_universes() -> dict:
    """
    Return static BSE universes from universes.py.
    These are manually maintained and should be updated periodically.
    """
    try:
        from .universes import BSE_MIDCAP, BSE_SENSEX, BSE_SMALLCAP
        return {
            "BSE SENSEX": BSE_SENSEX,
            "BSE MIDCAP": BSE_MIDCAP,
            "BSE SMALLCAP": BSE_SMALLCAP,
        }
    except ImportError as e:
        logger.warning("Failed to import BSE static universes: %s", e)
        return {}


def fetch_all_market_symbols() -> list[str]:
    """
    Fetch all unique listed symbols across NSE+BSE (~5,900).

    Combines:
      - NSE mainboard + SME + all indices (via nselib / CSV)
      - BSE all Active (via BSE API / CSV mirrors)
    Deduplicates (dual-listed) and caches 4h. Falls back to static
    NIFTY_BROAD (~207) if live fetch fails, so scan never breaks.
    """
    def _do() -> list[str]:
        nse = set()
        bse = set()
        # NSE — try live mainboard + SME
        try:
            nse_main = fetch_nse_mainboard()
            if nse_main:
                nse.update(s.upper() for s in nse_main if s)
        except Exception as e:
            logger.debug("NSE mainboard fetch in all-market failed: %s", e)
        try:
            # Also include SME and index constituents for coverage
            unique_nse = get_unique_nse_symbols()
            if unique_nse and len(unique_nse) > len(nse):
                nse.update(s.upper() for s in unique_nse if s)
        except Exception as e:
            logger.debug("Unique NSE fetch in all-market failed: %s", e)
        # BSE — try live all
        try:
            bse_list = fetch_bse_all_live()
            if bse_list:
                bse.update(s.upper() for s in bse_list if s)
        except Exception as e:
            logger.debug("BSE all fetch in all-market failed: %s", e)

        combined = nse | bse
        # If combined is still small (<500), add static BSE + NSE BROAD as floor
        if len(combined) < 500:
            try:
                from .universes import BSE_MIDCAP, BSE_SENSEX, BSE_SMALLCAP, NIFTY_BROAD
                combined.update(s.upper() for s in (NIFTY_BROAD + BSE_SENSEX + BSE_MIDCAP + BSE_SMALLCAP))
            except Exception:
                pass
        logger.info("All-market combined: NSE %d + BSE %d = %d unique (target ~5900)", len(nse), len(bse), len(combined))
        return sorted(combined)

    # Use 4h cache; fallback to NIFTY_BROAD so scan still works offline
    return _fetch_with_cache("all_market", _do, fallback=_get_static_fallback("all_market"))


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