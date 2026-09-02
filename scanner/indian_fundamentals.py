"""
Trendlyne & Screener.in Data Providers
Free Indian market data providers — fundamentals, peer comparison, technicals.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

# ── Cache ───────────────────────────────────────────────────────────────────

_FUND_CACHE: dict[str, tuple[dict, float]] = {}
_FUND_CACHE_TTL = 6 * 3600  # 6 hours


def _cache_get(key: str) -> dict | None:
    if key in _FUND_CACHE:
        result, ts = _FUND_CACHE[key]
        if time.time() - ts < _FUND_CACHE_TTL:
            return result
    return None


def _cache_set(key: str, value: dict):
    _FUND_CACHE[key] = (value, time.time())


# ── Trendlyne Fundamentals (Free, No Key) ──────────────────────────────────

@dataclass
class TrendlyneFundamentals:
    """Fundamental data from Trendlyne (free, no API key)."""
    ticker: str
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    roe: float | None = None
    roce: float | None = None
    dividend_yield: float | None = None
    debt_to_equity: float | None = None
    promoter_holding: float | None = None
    promoter_change: float | None = None  # Change in promoter holding
    market_cap: float | None = None
    enterprise_value: float | None = None
    peg_ratio: float | None = None
    cached: bool = False


def fetch_trendlyne_fundamentals(ticker: str) -> TrendlyneFundamentals | None:
    """
    Fetch fundamental data using Yahoo Finance (reliable, free).
    Trendlyne blocks automated access, so we use Yahoo as primary.
    
    Args:
        ticker: NSE ticker symbol (e.g., "RELIANCE")
    
    Returns:
        TrendlyneFundamentals or None
    """
    cache_k = hashlib.md5(f"trendlyne:{ticker}".encode(), usedforsecurity=False).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return TrendlyneFundamentals(**cached, cached=True)

    try:
        import yfinance as yf
        
        # Add .NS suffix for NSE stocks if not present
        yf_ticker = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
        stock = yf.Ticker(yf_ticker)
        info = stock.info
        
        if not info or info.get("regularMarketPrice") is None:
            logger.debug("Yahoo Finance: no data for %s", ticker)
            return None

        result = {}
        
        if info.get("trailingPE"):
            result["pe_ratio"] = float(info["trailingPE"])
        if info.get("priceToBook"):
            result["pb_ratio"] = float(info["priceToBook"])
        if info.get("returnOnEquity"):
            result["roe"] = float(info["returnOnEquity"]) * 100
        if info.get("returnOnCapitalEmployed"):
            result["roce"] = float(info["returnOnCapitalEmployed"]) * 100
        if info.get("dividendYield"):
            result["dividend_yield"] = float(info["dividendYield"]) * 100
        if info.get("debtToEquity"):
            result["debt_to_equity"] = float(info["debtToEquity"])
        if info.get("heldPercentInsiders"):
            result["promoter_holding"] = float(info["heldPercentInsiders"]) * 100

        if not result:
            return None

        fund = TrendlyneFundamentals(ticker=ticker, **result)
        _cache_set(cache_k, {"ticker": ticker, **result})
        return fund

    except Exception as e:
        logger.debug("Yahoo Fundamentals fetch failed for %s: %s", ticker, e)
        return None


# ── Screener.in Peer Comparison (Free, No Key) ────────────────────────────

@dataclass
class PeerComparison:
    """Peer comparison data from Screener.in."""
    ticker: str
    industry: str
    stock_pe: float | None = None
    industry_pe: float | None = None
    stock_roe: float | None = None
    industry_roe: float | None = None
    stock_roce: float | None = None
    industry_roce: float | None = None
    is_cheap_vs_peers: bool = False  # PE below industry average
    is_quality: bool = False  # ROE above industry average
    cached: bool = False


def fetch_peer_comparison(ticker: str) -> PeerComparison | None:
    """
    Fetch peer comparison from Screener.in (free, no API key).
    
    Args:
        ticker: NSE ticker symbol (e.g., "RELIANCE")
    
    Returns:
        PeerComparison or None
    """
    cache_k = hashlib.md5(f"screener:{ticker}".encode(), usedforsecurity=False).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return PeerComparison(**cached, cached=True)

    try:
        url = f"https://www.screener.in/company/{ticker}/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        html = resp.text

        result = {}

        # Extract industry
        industry_match = re.search(r'Industry\s*[:=]\s*([A-Za-z\s&]+)', html)
        if industry_match:
            result["industry"] = industry_match.group(1).strip()

        # Extract PE
        pe_match = re.search(r'Stock\s*PE\s*[:=]\s*(\d+\.?\d*)', html)
        if pe_match:
            result["stock_pe"] = float(pe_match.group(1))

        industry_pe_match = re.search(r'Industry\s*PE\s*[:=]\s*(\d+\.?\d*)', html)
        if industry_pe_match:
            result["industry_pe"] = float(industry_pe_match.group(1))

        # Extract ROE
        roe_match = re.search(r'Return\s*on\s*Equity\s*[:=]\s*(\d+\.?\d*)%?', html)
        if roe_match:
            result["stock_roe"] = float(roe_match.group(1))

        if not result:
            return None

        # Determine if cheap vs peers
        if "stock_pe" in result and "industry_pe" in result:
            result["is_cheap_vs_peers"] = result["stock_pe"] < result["industry_pe"]

        # Determine if quality
        if "stock_roe" in result:
            result["is_quality"] = result["stock_roe"] > 15.0

        peer = PeerComparison(ticker=ticker, **result)
        _cache_set(cache_k, {"ticker": ticker, **result})
        return peer

    except Exception as e:
        logger.debug("Screener.in fetch failed for %s: %s", ticker, e)
        return None


# ── Yahoo Finance Valuation (Free, No Key) ─────────────────────────────────

@dataclass
class YahooValuation:
    """Valuation data from Yahoo Finance (free, no API key)."""
    ticker: str
    pe_trailing: float | None = None
    pe_forward: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    peg_ratio: float | None = None
    dividend_yield: float | None = None
    profit_margin: float | None = None
    roe: float | None = None
    beta: float | None = None
    intrinsic_value: float | None = None  # Graham number if calculable
    cached: bool = False


def fetch_yahoo_valuation(ticker: str) -> YahooValuation | None:
    """
    Fetch valuation data from Yahoo Finance (free, no API key).
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE")
    
    Returns:
        YahooValuation or None
    """
    cache_k = hashlib.md5(f"yahoo_val:{ticker}".encode(), usedforsecurity=False).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return YahooValuation(**cached, cached=True)

    try:
        import yfinance as yf
        
        nse_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        stock = yf.Ticker(nse_ticker)
        info = stock.info
        
        if not info:
            return None

        result = {}

        # Extract valuation metrics
        for field_name, key in [
            ("pe_trailing", "trailingPE"),
            ("pe_forward", "forwardPE"),
            ("pb_ratio", "priceToBook"),
            ("ps_ratio", "priceToSalesTrailing12Months"),
            ("peg_ratio", "pegRatio"),
            ("dividend_yield", "dividendYield"),
            ("profit_margin", "profitMargins"),
            ("roe", "returnOnEquity"),
            ("beta", "beta"),
        ]:
            val = info.get(key)
            if val is not None:
                result[field_name] = float(val)

        # Calculate Graham Number if we have EPS and P/B
        eps = info.get("trailingEps")
        pb = result.get("pb_ratio")
        if eps and pb and eps > 0:
            # Graham Number = sqrt(22.5 * EPS * Book Value per Share)
            # Approximate: Book Value = Price / PB
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if price and pb > 0:
                book_value = price / pb
                graham = (22.5 * eps * book_value) ** 0.5
                result["intrinsic_value"] = round(graham, 2)

        if not result:
            return None

        val = YahooValuation(ticker=ticker, **result)
        _cache_set(cache_k, {"ticker": ticker, **result})
        return val

    except Exception as e:
        logger.debug("Yahoo valuation fetch failed for %s: %s", ticker, e)
        return None


# ── Unified Fundamentals Fetcher ───────────────────────────────────────────

def fetch_indian_fundamentals(ticker: str) -> dict:
    """
    Fetch all fundamental data from Indian market sources.
    
    Returns:
        {
            "trendlyne": TrendlyneFundamentals | None,
            "screener": PeerComparison | None,
            "yahoo_valuation": YahooValuation | None,
            "source": str,
        }
    """
    trendlyne = fetch_trendlyne_fundamentals(ticker)
    screener = fetch_peer_comparison(ticker)
    yahoo_val = fetch_yahoo_valuation(ticker)

    sources = []
    if trendlyne:
        sources.append("trendlyne")
    if screener:
        sources.append("screener")
    if yahoo_val:
        sources.append("yahoo_valuation")

    return {
        "trendlyne": trendlyne,
        "screener": screener,
        "yahoo_valuation": yahoo_val,
        "source": "+".join(sources) if sources else "none",
    }
