"""
Premium Finance Providers — API Key Required
Enhanced financial data from premium providers for Indian stock market analysis.

Providers:
  - Marketstack: Real-time intraday data
  - EOD Historical Data: 150+ exchanges, fundamentals
  - Financial Modeling Prep: Financial statements, ratios
  - IEX Cloud: Real-time US + India data
  - Polygon: Historical OHLCV
  - StockData: News + sentiment
  - Styvio: Stock sentiment scores
  - Halal Terminal: Shariah screening (India)
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

# ── Cache ───────────────────────────────────────────────────────────────────

_PREMIUM_CACHE: dict[str, tuple[dict, float]] = {}
_PREMIUM_CACHE_TTL = 6 * 3600  # 6 hours


def _cache_get(key: str) -> dict | None:
    if key in _PREMIUM_CACHE:
        result, ts = _PREMIUM_CACHE[key]
        if time.time() - ts < _PREMIUM_CACHE_TTL:
            return result
    return None


def _cache_set(key: str, value: dict):
    _PREMIUM_CACHE[key] = (value, time.time())


# ── Marketstack — Real-Time Market Data ─────────────────────────────────────

@dataclass
class MarketstackData:
    """Real-time market data from Marketstack."""
    ticker: str
    current_price: float
    open_price: float
    high_price: float
    low_price: float
    volume: int
    change: float
    change_pct: float
    timestamp: str
    cached: bool = False


def fetch_marketstack_data(
    ticker: str,
    api_key: str | None = None,
) -> MarketstackData | None:
    """
    Fetch real-time market data from Marketstack (requires API key).
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE.BSE")
        api_key: Marketstack API key
    
    Returns:
        MarketstackData or None
    """
    if not api_key:
        return None

    cache_k = hashlib.md5(f"marketstack:{ticker}".encode(), usedforsecurity=False).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return MarketstackData(**cached, cached=True)

    try:
        url = "https://api.marketstack.com/v1/eod/latest"
        params = {
            "access_key": api_key,
            "symbols": ticker,
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        entries = data.get("data", [])
        if not entries:
            return None

        entry = entries[0]
        result = MarketstackData(
            ticker=ticker,
            current_price=entry.get("close", 0),
            open_price=entry.get("open", 0),
            high_price=entry.get("high", 0),
            low_price=entry.get("low", 0),
            volume=entry.get("volume", 0),
            change=entry.get("close", 0) - entry.get("open", 0),
            change_pct=((entry.get("close", 0) - entry.get("open", 0)) / entry.get("open", 1)) * 100,
            timestamp=entry.get("date", ""),
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "current_price": result.current_price,
            "open_price": result.open_price,
            "high_price": result.high_price,
            "low_price": result.low_price,
            "volume": result.volume,
            "change": result.change,
            "change_pct": result.change_pct,
            "timestamp": result.timestamp,
        })

        return result

    except Exception as e:
        logger.debug("Marketstack fetch failed for %s: %s", ticker, e)
        return None


# ── EOD Historical Data ────────────────────────────────────────────────────

@dataclass
class EODData:
    """Historical market data from EOD."""
    ticker: str
    current_price: float
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    eps: float | None = None
    dividend_yield: float | None = None
    market_cap: float | None = None
    cached: bool = False


def fetch_eod_data(
    ticker: str,
    api_key: str | None = None,
) -> EODData | None:
    """
    Fetch historical market data from EOD (requires API key).
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE.NSE")
        api_key: EOD API key
    
    Returns:
        EODData or None
    """
    if not api_key:
        return None

    cache_k = hashlib.md5(f"eod:{ticker}".encode(), usedforsecurity=False).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return EODData(**cached, cached=True)

    try:
        url = f"https://eodhistoricaldata.com/api/eod/{ticker}"
        params = {
            "api_token": api_key,
            "fmt": "json",
            "period": "d",
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return None

        last = data[-1] if isinstance(data, list) else data
        result = EODData(
            ticker=ticker,
            current_price=last.get("adjusted_close", last.get("close", 0)),
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "current_price": result.current_price,
            "pe_ratio": result.pe_ratio,
            "pb_ratio": result.pb_ratio,
            "eps": result.eps,
            "dividend_yield": result.dividend_yield,
            "market_cap": result.market_cap,
        })

        return result

    except Exception as e:
        logger.debug("EOD fetch failed for %s: %s", ticker, e)
        return None


# ── Financial Modeling Prep ─────────────────────────────────────────────────

@dataclass
class FMPData:
    """Financial data from Financial Modeling Prep."""
    ticker: str
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    eps: float | None = None
    roe: float | None = None
    roa: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    profit_margin: float | None = None
    market_cap: float | None = None
    cached: bool = False


def fetch_fmp_data(
    ticker: str,
    api_key: str | None = None,
) -> FMPData | None:
    """
    Fetch financial data from Financial Modeling Prep (requires API key).
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE.NS")
        api_key: FMP API key
    
    Returns:
        FMPData or None
    """
    if not api_key:
        return None

    cache_k = hashlib.md5(f"fmp:{ticker}".encode(), usedforsecurity=False).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return FMPData(**cached, cached=True)

    try:
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}"
        params = {"apikey": api_key}

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return None

        profile = data[0] if isinstance(data, list) else data
        result = FMPData(
            ticker=ticker,
            pe_ratio=profile.get("pe"),
            pb_ratio=profile.get("pb"),
            eps=profile.get("lastDiv"),
            roe=profile.get("roe"),
            market_cap=profile.get("mktCap"),
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "pe_ratio": result.pe_ratio,
            "pb_ratio": result.pb_ratio,
            "eps": result.eps,
            "roe": result.roe,
            "market_cap": result.market_cap,
        })

        return result

    except Exception as e:
        logger.debug("FMP fetch failed for %s: %s", ticker, e)
        return None


# ── IEX Cloud ───────────────────────────────────────────────────────────────

@dataclass
class IEXData:
    """Market data from IEX Cloud."""
    ticker: str
    current_price: float
    previous_close: float
    change: float
    change_pct: float
    volume: int
    market_cap: float | None = None
    pe_ratio: float | None = None
    week52_high: float | None = None
    week52_low: float | None = None
    cached: bool = False


def fetch_iex_data(
    ticker: str,
    api_key: str | None = None,
) -> IEXData | None:
    """
    Fetch market data from IEX Cloud (requires API key).
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE")
        api_key: IEX API key
    
    Returns:
        IEXData or None
    """
    if not api_key:
        return None

    cache_k = hashlib.md5(f"iex:{ticker}".encode(), usedforsecurity=False).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return IEXData(**cached, cached=True)

    try:
        url = f"https://cloud.iexapis.com/stable/stock/{ticker}/quote"
        params = {"token": api_key}

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result = IEXData(
            ticker=ticker,
            current_price=data.get("latestPrice", 0),
            previous_close=data.get("previousClose", 0),
            change=data.get("change", 0),
            change_pct=data.get("changePercent", 0) * 100,
            volume=data.get("volume", 0),
            market_cap=data.get("marketCap"),
            pe_ratio=data.get("peRatio"),
            week52_high=data.get("week52High"),
            week52_low=data.get("week52Low"),
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "current_price": result.current_price,
            "previous_close": result.previous_close,
            "change": result.change,
            "change_pct": result.change_pct,
            "volume": result.volume,
            "market_cap": result.market_cap,
            "pe_ratio": result.pe_ratio,
            "week52_high": result.week52_high,
            "week52_low": result.week52_low,
        })

        return result

    except Exception as e:
        logger.debug("IEX fetch failed for %s: %s", ticker, e)
        return None


# ── Polygon — Historical Data ───────────────────────────────────────────────

@dataclass
class PolygonData:
    """Historical market data from Polygon."""
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: str
    cached: bool = False


def fetch_polygon_data(
    ticker: str,
    api_key: str | None = None,
    days: int = 1,
) -> list[PolygonData] | None:
    """
    Fetch historical data from Polygon (requires API key).
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE")
        api_key: Polygon API key
        days: Number of days to fetch
    
    Returns:
        List of PolygonData or None
    """
    if not api_key:
        return None

    cache_k = hashlib.md5(f"polygon:{ticker}:{days}".encode(), usedforsecurity=False).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return [PolygonData(**item) for item in cached.get("bars", [])]

    try:
        from datetime import date, timedelta

        end = date.today()
        start = end - timedelta(days=days + 5)
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        params = {"apiKey": api_key, "limit": days}

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for bar in data.get("results", [])[-days:]:
            results.append(PolygonData(
                ticker=ticker,
                open=bar.get("o", 0),
                high=bar.get("h", 0),
                low=bar.get("l", 0),
                close=bar.get("c", 0),
                volume=bar.get("v", 0),
                timestamp=str(bar.get("t", "")),
            ))

        if results:
            _cache_set(cache_k, {"bars": [
                {"ticker": r.ticker, "open": r.open, "high": r.high,
                 "low": r.low, "close": r.close, "volume": r.volume,
                 "timestamp": r.timestamp}
                for r in results
            ]})

        return results if results else None

    except Exception as e:
        logger.debug("Polygon fetch failed for %s: %s", ticker, e)
        return None


# ── StockData — News + Sentiment ────────────────────────────────────────────

@dataclass
class StockDataNews:
    """News + sentiment from StockData."""
    ticker: str
    sentiment_score: float
    article_count: int
    top_headlines: list[str] = field(default_factory=list)
    cached: bool = False


def fetch_stockdata_news(
    ticker: str,
    api_key: str | None = None,
) -> StockDataNews | None:
    """
    Fetch news + sentiment from StockData (requires API key).
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE")
        api_key: StockData API key
    
    Returns:
        StockDataNews or None
    """
    if not api_key:
        return None

    cache_k = hashlib.md5(f"stockdata:{ticker}".encode(), usedforsecurity=False).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return StockDataNews(**cached, cached=True)

    try:
        url = "https://stockdata.org/api/v1/news"
        params = {"ticker": ticker, "api_token": api_key}

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("data", [])
        headlines = [a.get("title", "") for a in articles[:10]]

        # Simple keyword sentiment on headlines
        from .market_sentiment import _keyword_sentiment
        combined = " ".join(headlines)
        sentiment = _keyword_sentiment(combined) if combined.strip() else 0.0

        result = StockDataNews(
            ticker=ticker,
            sentiment_score=sentiment,
            article_count=len(articles),
            top_headlines=headlines[:5],
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "sentiment_score": result.sentiment_score,
            "article_count": result.article_count,
            "top_headlines": result.top_headlines,
        })

        return result

    except Exception as e:
        logger.debug("StockData fetch failed for %s: %s", ticker, e)
        return None


# ── Styvio — Stock Sentiment ────────────────────────────────────────────────

@dataclass
class StyvioData:
    """Stock sentiment from Styvio."""
    ticker: str
    sentiment_score: float
    sentiment_label: str  # "bullish", "bearish", "neutral"
    confidence: float
    cached: bool = False


def fetch_styvio_data(
    ticker: str,
    api_key: str | None = None,
) -> StyvioData | None:
    """
    Fetch stock sentiment from Styvio (requires API key).
    
    Args:
        ticker: Stock ticker
        api_key: Styvio API key
    
    Returns:
        StyvioData or None
    """
    if not api_key:
        return None

    cache_k = hashlib.md5(f"styvio:{ticker}".encode(), usedforsecurity=False).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return StyvioData(**cached, cached=True)

    try:
        url = f"https://api.styvio.com/sentiment/{ticker}"
        headers = {"Authorization": f"Bearer {api_key}"}

        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result = StyvioData(
            ticker=ticker,
            sentiment_score=data.get("sentiment_score", 0),
            sentiment_label=data.get("sentiment_label", "neutral"),
            confidence=data.get("confidence", 0.5),
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "sentiment_score": result.sentiment_score,
            "sentiment_label": result.sentiment_label,
            "confidence": result.confidence,
        })

        return result

    except Exception as e:
        logger.debug("Styvio fetch failed for %s: %s", ticker, e)
        return None


# ── Halal Terminal — Shariah Screening ──────────────────────────────────────

@dataclass
class ShariahData:
    """Shariah compliance data from Halal Terminal."""
    ticker: str
    is_shariah_compliant: bool
    screening_method: str
    purification_required: bool
    zakat_amount: float | None = None
    cached: bool = False


def fetch_shariah_data(
    ticker: str,
    api_key: str | None = None,
) -> ShariahData | None:
    """
    Fetch Shariah compliance data from Halal Terminal (requires API key).
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE")
        api_key: Halal Terminal API key
    
    Returns:
        ShariahData or None
    """
    if not api_key:
        return None

    cache_k = hashlib.md5(f"shariah:{ticker}".encode(), usedforsecurity=False).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return ShariahData(**cached, cached=True)

    try:
        url = f"https://api.halalterminal.com/v1/screen/{ticker}"
        params = {"api_key": api_key}

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result = ShariahData(
            ticker=ticker,
            is_shariah_compliant=data.get("is_compliant", False),
            screening_method=data.get("method", "aaoifi"),
            purification_required=data.get("purification_required", False),
            zakat_amount=data.get("zakat_amount"),
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "is_shariah_compliant": result.is_shariah_compliant,
            "screening_method": result.screening_method,
            "purification_required": result.purification_required,
            "zakat_amount": result.zakat_amount,
        })

        return result

    except Exception as e:
        logger.debug("Shariah fetch failed for %s: %s", ticker, e)
        return None


# ── Unified Premium Finance Fetcher ─────────────────────────────────────────

def fetch_premium_finance(
    ticker: str,
    api_keys: dict | None = None,
) -> dict:
    """
    Fetch all premium finance data for a ticker.
    
    Args:
        ticker: Stock ticker
        api_keys: Dict of API keys
    
    Returns:
        {
            "marketstack": MarketstackData | None,
            "eod": EODData | None,
            "fmp": FMPData | None,
            "iex": IEXData | None,
            "polygon": list[PolygonData] | None,
            "stockdata": StockDataNews | None,
            "styvio": StyvioData | None,
            "shariah": ShariahData | None,
            "source": str,
        }
    """
    if api_keys is None:
        api_keys = {}

    marketstack = fetch_marketstack_data(ticker, api_keys.get("MARKETAUX_API_KEY"))
    eod = fetch_eod_data(ticker, api_keys.get("EOD_API_KEY"))
    fmp = fetch_fmp_data(ticker, api_keys.get("FMP_API_KEY"))
    iex = fetch_iex_data(ticker, api_keys.get("IEX_API_KEY"))
    polygon = fetch_polygon_data(ticker, api_keys.get("POLYGON_API_KEY"))
    stockdata = fetch_stockdata_news(ticker, api_keys.get("STOCKDATA_API_KEY"))
    styvio = fetch_styvio_data(ticker, api_keys.get("STYVIO_API_KEY"))
    shariah = fetch_shariah_data(ticker, api_keys.get("HALAL_API_KEY"))

    sources = []
    if marketstack:
        sources.append("marketstack")
    if eod:
        sources.append("eod")
    if fmp:
        sources.append("fmp")
    if iex:
        sources.append("iex")
    if polygon:
        sources.append("polygon")
    if stockdata:
        sources.append("stockdata")
    if styvio:
        sources.append("styvio")
    if shariah:
        sources.append("shariah")

    return {
        "marketstack": marketstack,
        "eod": eod,
        "fmp": fmp,
        "iex": iex,
        "polygon": polygon,
        "stockdata": stockdata,
        "styvio": styvio,
        "shariah": shariah,
        "source": "+".join(sources) if sources else "none",
    }
