"""
Dynamic symbol fetcher for NSE/BSE universes.
Fetches live symbol lists from NSE via nselib.
BSE support is limited due to anti-scraping measures on bseindia.com.
"""

from datetime import date, timedelta
from typing import Optional

import nselib.capital_market as cm


def fetch_nse_mainboard() -> list[str]:
    """Fetch all NSE mainboard equity symbols."""
    df = cm.equity_list()
    return df["SYMBOL"].str.strip().tolist()


def fetch_nse_fno() -> list[str]:
    """Fetch NSE F&O eligible equity symbols."""
    df = cm.fno_equity_list()
    return df["symbol"].str.strip().tolist()


def fetch_nse_sme(trade_date: Optional[date] = None) -> list[str]:
    """
    Fetch NSE SME platform equity symbols.
    
    Args:
        trade_date: Trading date. Defaults to most recent weekday.
    """
    if trade_date is None:
        trade_date = date.today()
        # Find most recent weekday
        while trade_date.weekday() >= 5:
            trade_date -= timedelta(days=1)
    
    for days_back in range(10):
        d = trade_date - timedelta(days=days_back)
        try:
            df = cm.sme_band_complete(trade_date=d.strftime("%d-%m-%Y"))
            if "Symbol" in df.columns:
                return df["Symbol"].str.strip().tolist()
            elif "symbol" in df.columns:
                return df["symbol"].str.strip().tolist()
            elif "SYMBOL" in df.columns:
                return df["SYMBOL"].str.strip().tolist()
        except Exception:
            continue
    return []


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
    
    df = func_map[index_name]()
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
    Get all unique NSE symbols across all segments.
    """
    all_data = fetch_all_nse_symbols()
    unique = set()
    for symbols in all_data.values():
        unique.update(s.upper().strip() for s in symbols if isinstance(s, str))
    return sorted(unique)


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
    except Exception:
        return {}


def validate_bse_symbols(symbols: list[str], max_check: int = 50) -> list[str]:
    """
    Validate BSE symbols by checking if they exist on yfinance (.BO suffix).
    Limited to max_check to avoid rate limits.
    """
    import yfinance as yf
    valid = []
    for sym in symbols[:max_check]:
        try:
            ticker = yf.Ticker(f"{sym}.BO")
            # Quick check - try to get info
            info = ticker.info
            if info and info.get("symbol"):
                valid.append(sym)
        except Exception:
            continue
    return valid


if __name__ == "__main__":
    all_data = fetch_all_nse_symbols()
    for k, v in all_data.items():
        print(f"{k}: {len(v)} symbols")
    unique = get_unique_nse_symbols()
    print(f"Unique NSE symbols: {len(unique)}")