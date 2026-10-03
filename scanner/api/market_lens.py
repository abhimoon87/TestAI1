"""
NSE Market Lens provider — official NSE screener API.

Data from https://marketlens.nseindia.com/api (free, keyless JSON behind
Akamai). Politeness: browser-ish headers, per-endpoint TTL caches, at most
one profile request per ticker per cache window. Every fetch degrades to
None on failure — a scan never depends on this source.

Field caveats (verified live): dma20/returnOnEquity/avgVolume/industryPe/
netProfit-in-list-view are broken or zeroed — never read them. Reliable:
peRatio, marketCap, eps, margins, promoterHolding, week52*, returns,
sector/subSector, deliveryPercentage. Bulk endpoints are unusable as a
universe source: the list endpoint caps at 100 rows (page>=2 empty) and
the 13 coarse sector lists cover only 688 stocks while profiles use a
fine taxonomy ("Diversified FMCG") those lists don't know — so bulk
access stays per-ticker profile fetches, which negative-cache on 404.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests

from ..shared.cache import TTLCache

logger = logging.getLogger(__name__)

_BASE = "https://marketlens.nseindia.com/api"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://marketlens.nseindia.com/",
}

_STOCK_CACHE: TTLCache[dict] = TTLCache(ttl=24 * 3600, namespace="marketlens_stock")
_META_CACHE: TTLCache = TTLCache(ttl=7 * 86400, namespace="marketlens_meta")
_INDEX_CACHE: TTLCache = TTLCache(ttl=900, namespace="marketlens_index")
_PRICE_CACHE: TTLCache = TTLCache(ttl=900, namespace="marketlens_price")

_MISS = "__ml_miss__"  # 404 sentinel — callers isinstance-filter it to None


def _clean_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    for suffix in (".NSE", ".BSE", ".NS", ".BO"):
        ticker = ticker.replace(suffix, "")
    return ticker


def _get_json(path: str, cache: TTLCache, cache_k: str):
    cached = cache.get(cache_k)
    if cached is not None:
        return cached
    try:
        resp = requests.get(f"{_BASE}{path}", headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        body = resp.json()
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status == 404:
            # Definitive miss (renamed/unknown ticker) — cache the sentinel so
            # bulk scans don't refetch every unknown symbol each run.
            cache.set(cache_k, _MISS)
            return _MISS
        logger.info("Market Lens %s failed: %s", path, e)
        return None
    except (requests.RequestException, ValueError) as e:
        logger.info("Market Lens %s failed: %s", path, e)
        return None
    if not isinstance(body, dict) or not body.get("success"):
        logger.debug("Market Lens %s returned an error payload", path)
        return None
    data = body.get("data")
    if not data:
        return None
    cache.set(cache_k, data)
    return data


def get_stock(ticker: str) -> dict | None:
    """Full per-stock profile (sector, PE, mcap, promoter holding, …)."""
    t = _clean_ticker(ticker)
    if not t:
        return None
    data = _get_json(f"/stocks/{t}", _STOCK_CACHE, _STOCK_CACHE.make_key(t))
    return data if isinstance(data, dict) else None


def get_quarterly(ticker: str) -> list | None:
    """Last 5 quarters: date, totalIncome, profitBeforeTax, netProfitLoss, eps."""
    t = _clean_ticker(ticker)
    if not t:
        return None
    data = _get_json(
        f"/stocks/{t}/quarterly-financials", _META_CACHE, _META_CACHE.make_key("q", t)
    )
    return data if isinstance(data, list) else None


def get_peers(ticker: str) -> list | None:
    """Peer stocks in the same sector (fields partly zeroed — display only)."""
    t = _clean_ticker(ticker)
    if not t:
        return None
    data = _get_json(
        f"/stocks/{t}/peers", _META_CACHE, _META_CACHE.make_key("peers", t)
    )
    return data if isinstance(data, list) else None


def get_shareholding(ticker: str) -> list | None:
    """Last 5 quarters of {quarterEnd, promoters, publicHolding}.

    NOTE: no FII/DII split — those stay on the Screener.in scrape.
    """
    t = _clean_ticker(ticker)
    if not t:
        return None
    data = _get_json(
        f"/stocks/{t}/shareholding",
        _META_CACHE,
        _META_CACHE.make_key("shp", t),
    )
    return data if isinstance(data, list) else None


def get_indices() -> list | None:
    """Index snapshots: [{ticker, name, value, change, updatedAt}]."""
    data = _get_json("/indices", _INDEX_CACHE, _INDEX_CACHE.make_key("indices"))
    return data if isinstance(data, list) else None


def get_price_history(ticker: str, period: str = "1Y") -> list | None:
    """Daily {date, price, volume} rows (ascending) for 1M/6M/1Y/5Y.

    Close + volume only — no OHLC; consumers must synthesize the rest.
    """
    t = _clean_ticker(ticker)
    if not t:
        return None
    data = _get_json(
        f"/stocks/{t}/price-history?period={period}",
        _PRICE_CACHE,
        _PRICE_CACHE.make_key("ph", t, period),
    )
    return data if isinstance(data, list) else None


# ── Shareholding → consumer shape ───────────────────────────────────────────


def yoy_growth(quarterly: list | None, field: str) -> float | None:
    """YoY growth % from the newest row (index 0) vs the year-ago row (4).

    Rows are newest-first (verified live); needs the full 5-quarter window.
    """
    if not quarterly or len(quarterly) < 5:
        return None
    try:
        new = float(quarterly[0].get(field))
        old = float(quarterly[4].get(field))
    except (TypeError, ValueError, AttributeError):
        return None
    if not old:
        return None
    return round((new - old) / abs(old) * 100, 2)


def _quarter_end(value: str):
    try:
        return datetime.strptime(str(value), "%d-%b-%Y")
    except (TypeError, ValueError):
        return None


def ml_shareholding_shape(raw: list) -> dict | None:
    """Map Market Lens's 5-quarter list to the app's shareholding shape.

    Produces ``{quarter, series: {promoters: {latest, total, recent}}}`` —
    promoters only; ``foreign_institutions``/``domestic_institutions`` come
    from Screener.in. Consumers already tolerate partial series.
    """
    rows = [r for r in raw if isinstance(r, dict) and _quarter_end(r.get("quarterEnd"))]
    if not rows:
        return None
    rows.sort(key=lambda r: _quarter_end(r["quarterEnd"]), reverse=True)
    values = []
    for r in rows:
        try:
            values.append(float(r.get("promoters")))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    total = round(values[0] - values[-1], 2) if len(values) >= 2 else None
    recent = round(values[0] - values[1], 2) if len(values) >= 2 else None
    return {
        "quarter": _quarter_end(rows[0]["quarterEnd"]).strftime("%b %Y"),
        "series": {
            "promoters": {
                "latest": round(values[0], 2),
                "total": total,
                "recent": recent,
            }
        },
    }


# ── Universe pre-filter ─────────────────────────────────────────────────────


def filter_universe(
    tickers: list,
    sectors: str = "",
    pe_max: float = 0.0,
    mcap_min_cr: float = 0.0,
) -> list:
    """Drop tickers failing the Market Lens criteria; no criteria → unchanged.

    ponytail: per-ticker profile fetch, no count ceiling — the bulk endpoints
    can't replace it (list caps at 100 with broken pagination; the 13 coarse
    sector lists cover only 688 stocks and miss fine-taxonomy names like
    "Diversified FMCG"). Profiles TTL-cache 24h and unknown tickers
    negative-cache on 404, so only the first scan of a universe pays the
    request cost; drop the executor to 4 workers if Akamai throttles bursts.
    """
    want_sectors = {s.strip().lower() for s in sectors.split(",") if s.strip()}
    if not want_sectors and pe_max <= 0 and mcap_min_cr <= 0:
        return tickers

    with ThreadPoolExecutor(max_workers=8) as ex:
        stocks = list(ex.map(get_stock, tickers))

    kept = []
    for ticker, stock in zip(tickers, stocks, strict=True):
        if stock is None:
            continue  # can't verify criteria → drop
        if want_sectors:
            haystack = f"{stock.get('sector', '')} {stock.get('subSector', '')}".lower()
            if not any(w in haystack for w in want_sectors):
                continue
        if pe_max > 0:
            try:
                pe = float(stock.get("peRatio") or 0)
            except (TypeError, ValueError):
                pe = 0
            if pe <= 0 or pe > pe_max:
                continue
        if mcap_min_cr > 0:
            try:
                mcap = float(stock.get("marketCap") or 0)
            except (TypeError, ValueError):
                mcap = 0
            if mcap / 1e7 < mcap_min_cr:
                continue
        kept.append(ticker)
    return kept
