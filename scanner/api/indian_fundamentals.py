"""
Trendlyne & Screener.in Data Providers
Free Indian market data providers — fundamentals, peer comparison, technicals.
"""

import datetime
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from urllib.parse import quote

import requests

from ..shared import db
from ..shared.cache import TTLCache
from .yahoo_symbol import yf_quote_variants

logger = logging.getLogger(__name__)

# ── Cache ───────────────────────────────────────────────────────────────────

_FUND_CACHE: TTLCache[dict] = TTLCache(ttl=6 * 3600, namespace="indian_fundamentals")


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
    cache_k = hashlib.md5(
        f"trendlyne:{ticker}".encode(), usedforsecurity=False
    ).hexdigest()
    cached = _FUND_CACHE.get(cache_k)
    if cached:
        return TrendlyneFundamentals(**cached, cached=True)

    try:
        import yfinance as yf

        # .NS first, then one .BO retry -- BSE-only names never resolve on .NS
        info = {}
        for yf_ticker in yf_quote_variants(ticker):
            info = yf.Ticker(yf_ticker).info or {}
            if info and info.get("regularMarketPrice") is not None:
                break

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
        _FUND_CACHE.set(cache_k, {"ticker": ticker, **result})
        return fund

    except Exception as e:
        logger.info("Yahoo Fundamentals fetch failed for %s: %s", ticker, e)
        return None


# ── Promoter holding history ──────────────────────────────────────────────
# ponytail: no free quarterly shareholding series exists — remember our own
# snapshots. Day one has no baseline (renders "No change"); once a second
# snapshot lands, total/recent deltas drive the arrows. Promoter moves are
# rare (quarterly), so "No change" is the normal, correct state.

_PROMOTER_HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".cache", "promoter_history.json"
)
_PROMOTER_KEEP = 12
_promoter_lock = threading.Lock()


def _load_promoter_history() -> dict:
    """History blob from the kv store; empty → one-time legacy file read."""
    data = db.kv_get_json("promoter_history", "all")
    if isinstance(data, dict):
        return data
    try:
        with open(_PROMOTER_HISTORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    return data if isinstance(data, dict) else {}


def record_promoter_snapshot(ticker: str, pct: float) -> list[list]:
    """
    Append today's promoter holding % for ``ticker`` (once per day).

    Returns the snapshot series ``[[iso_date, pct], ...]`` oldest first,
    pruned to the last 12 entries. Invalid/zero pct returns [] unchanged.
    """
    if not pct or pct <= 0:
        return []
    today = datetime.date.today().isoformat()
    with _promoter_lock:
        data = _load_promoter_history()
        series = [
            e
            for e in data.get(ticker, [])
            if isinstance(e, list) and len(e) == 2 and isinstance(e[1], (int, float))
        ]
        if series and series[-1][0] == today:
            return series
        series.append([today, round(float(pct), 3)])
        series = series[-_PROMOTER_KEEP:]
        data[ticker] = series
        try:
            db.kv_put_json("promoter_history", "all", data)
        except Exception:
            logger.debug("Promoter-history write failed", exc_info=True)
        return series


def promoter_position_change(series: list[list]) -> tuple[float | None, float | None]:
    """
    (total_pp, recent_pp) change in promoter holding across snapshots,
    in percentage points. None until a second snapshot exists.
    """
    if len(series) < 2:
        return None, None
    total = round(float(series[-1][1]) - float(series[0][1]), 3)
    recent = round(float(series[-1][1]) - float(series[-2][1]), 3)
    return total, recent


# ── Per-stock shareholding pattern (Screener.in quarterly) ────────────────
# Screener mirrors the SEBI-mandated quarterly filings: 12 quarters oldest →
# newest in one GET, so total/recent deltas work from day one — no snapshot
# history needed. Cached a week (quarters move slowly); the snapshot path
# above stays as the fallback when Screener has no page for a ticker.

_SHP_CACHE: TTLCache[dict] = TTLCache(ttl=7 * 86400, namespace="shareholding")
_SHP_KEYS = ("promoters", "foreign_institutions", "domestic_institutions")


def parse_shareholding_html(html: str) -> dict | None:
    """
    Quarterly shareholding series from a Screener company page.

    Returns {"quarter": "Jun 2026", "series": {key: {latest, total, recent}}}
    where total/recent are percentage-point moves over the 12-quarter series
    and the latest quarter, or None when the table is missing.
    """
    if 'id="quarterly-shp"' not in html:
        return None
    # Split on the content <div>, not the bare id — the tab buttons carry
    # data-tab-id="quarterly-shp" earlier in the page and would truncate us.
    region = html.split('<div id="quarterly-shp">', 1)[1].split(
        '<div id="yearly-shp">', 1
    )[0]
    head = re.search(r"<thead>(.*?)</thead>", region, re.DOTALL)
    if not head:
        return None
    quarters = [
        re.sub(r"<[^>]+>", "", m).strip()
        for m in re.findall(r"<th[^>]*>(.*?)</th>", head.group(1), re.DOTALL)
    ]
    out: dict = {"quarter": quarters[-1] if quarters else None, "series": {}}
    for key in _SHP_KEYS:
        m = re.search(
            rf"showShareholders\(\s*'{key}'\s*,\s*'quarterly'.*?</tr>",
            region,
            re.DOTALL,
        )
        if not m:
            continue
        vals = [
            float(v) for v in re.findall(r"<td[^>]*>\s*([\d.]+)%\s*</td>", m.group(0))
        ]
        if not vals:
            continue
        out["series"][key] = {
            "latest": vals[-1],
            "total": round(vals[-1] - vals[0], 2) if len(vals) > 1 else None,
            "recent": round(vals[-1] - vals[-2], 2) if len(vals) > 1 else None,
        }
    return out if out["series"] else None


def _merge_shareholding(screener: dict | None, lens: dict | None) -> dict:
    """Promoters from Market Lens (primary), FII/DII from Screener.in."""
    out = {
        "quarter": (screener or {}).get("quarter") or (lens or {}).get("quarter"),
        "series": dict((screener or {}).get("series") or {}),
    }
    lens_series = (lens or {}).get("series") or {}
    if "promoters" in lens_series:
        out["series"]["promoters"] = lens_series["promoters"]
    return out


def fetch_shareholding_pattern(ticker: str, cache_only: bool = False) -> dict | None:
    """Latest quarterly shareholding % for ``ticker`` — merged, both sources.

    Screener.in supplies FII/DII (and promoters when Market Lens is down);
    Market Lens supplies promoters as primary. Cached a week either way.
    """
    cache_k = hashlib.md5(f"shp:{ticker}".encode(), usedforsecurity=False).hexdigest()
    cached = _SHP_CACHE.get(cache_k)
    if cached:
        return cached
    if cache_only:
        return None
    screener = None
    try:
        slug = quote(ticker, safe="")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        for path in ("consolidated/", ""):
            resp = requests.get(
                f"https://www.screener.in/company/{slug}/{path}",
                headers=headers,
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            parsed = parse_shareholding_html(resp.text)
            if parsed:
                screener = parsed
                break
    except Exception as e:
        logger.info("Shareholding fetch failed for %s: %s", ticker, e)

    lens = None
    try:
        from .market_lens import get_shareholding, ml_shareholding_shape

        raw = get_shareholding(ticker)
        if raw:
            lens = ml_shareholding_shape(raw)
    except Exception as e:
        logger.info("Market Lens shareholding failed for %s: %s", ticker, e)

    if screener is None and lens is None:
        return None
    merged = _merge_shareholding(screener, lens)
    _SHP_CACHE.set(cache_k, merged)
    return merged


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
    cache_k = hashlib.md5(
        f"screener:{ticker}".encode(), usedforsecurity=False
    ).hexdigest()
    cached = _FUND_CACHE.get(cache_k)
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
        industry_match = re.search(r"Industry\s*[:=]\s*([A-Za-z\s&]+)", html)
        if industry_match:
            result["industry"] = industry_match.group(1).strip()

        # Extract PE
        pe_match = re.search(r"Stock\s*PE\s*[:=]\s*(\d+\.?\d*)", html)
        if pe_match:
            result["stock_pe"] = float(pe_match.group(1))

        industry_pe_match = re.search(r"Industry\s*PE\s*[:=]\s*(\d+\.?\d*)", html)
        if industry_pe_match:
            result["industry_pe"] = float(industry_pe_match.group(1))

        # Extract ROE
        roe_match = re.search(r"Return\s*on\s*Equity\s*[:=]\s*(\d+\.?\d*)%?", html)
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
        _FUND_CACHE.set(cache_k, {"ticker": ticker, **result})
        return peer

    except Exception as e:
        logger.info("Screener.in fetch failed for %s: %s", ticker, e)
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
    cache_k = hashlib.md5(
        f"yahoo_val:{ticker}".encode(), usedforsecurity=False
    ).hexdigest()
    cached = _FUND_CACHE.get(cache_k)
    if cached:
        return YahooValuation(**cached, cached=True)

    try:
        import yfinance as yf

        info = {}
        for yf_ticker in yf_quote_variants(ticker):
            info = yf.Ticker(yf_ticker).info or {}
            if info:
                break

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
        _FUND_CACHE.set(cache_k, {"ticker": ticker, **result})
        return val

    except Exception as e:
        logger.info("Yahoo valuation fetch failed for %s: %s", ticker, e)
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
