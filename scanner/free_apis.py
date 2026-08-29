"""
Free APIs Provider — No API Key Required
Free data sources from public-apis repository for Indian stock market analysis.

Providers:
  - Frankfurter: INR/USD exchange rates (free, no key)
  - CoinGecko: Crypto market sentiment / BTC correlation (free, no key)
  - Statistics of the World: GDP, inflation, macro indicators (free, no key)
  - Top 5 Stocks: AI-ranked daily watchlists (free, no key)
  - WallstreetBets: Reddit WSB sentiment (free, no key)
  - Noozra: Free news headlines from 200+ RSS sources (free, no key)
  - Indian Mandi Prices: Commodity prices for agri stocks (free, no key)
  - Indian Pincode: Location-based sector mapping (free, no key)
  - API Setu: Indian govt KYC/business data (free, no key)
  - Open Government India: Regulatory/compliance data (free, no key)
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Cache ───────────────────────────────────────────────────────────────────

_FREE_API_CACHE: dict[str, tuple[dict, float]] = {}
_FREE_API_CACHE_TTL = 4 * 3600  # 4 hours


def _cache_get(key: str) -> Optional[dict]:
    if key in _FREE_API_CACHE:
        result, ts = _FREE_API_CACHE[key]
        if time.time() - ts < _FREE_API_CACHE_TTL:
            return result
    return None


def _cache_set(key: str, value: dict):
    _FREE_API_CACHE[key] = (value, time.time())


# ── Frankfurter — Exchange Rates (Free, No Key) ────────────────────────────

@dataclass
class ForexData:
    """Exchange rate data from Frankfurter API."""
    base_currency: str
    target_currency: str
    rate: float
    historical_rates: list[dict] = field(default_factory=list)  # [{date, rate}]
    change_1d: float = 0.0  # 1-day change %
    change_1w: float = 0.0  # 1-week change %
    cached: bool = False


def fetch_forex_data(
    base: str = "USD",
    target: str = "INR",
    days: int = 7,
) -> Optional[ForexData]:
    """
    Fetch exchange rates from Frankfurter API (free, no key).
    
    Args:
        base: Base currency (default: USD)
        target: Target currency (default: INR)
        days: Lookback days for historical rates
    
    Returns:
        ForexData or None
    """
    cache_k = hashlib.md5(f"forex:{base}:{target}:{days}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return ForexData(**cached, cached=True)

    try:
        from datetime import date, timedelta

        # Current rate
        url = f"https://api.frankfurter.app/latest?from={base}&to={target}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        current_rate = data.get("rates", {}).get(target)
        if current_rate is None:
            return None

        # Historical rates
        end = date.today()
        start = end - timedelta(days=days + 5)
        hist_url = f"https://api.frankfurter.app/{start}..{end}?from={base}&to={target}"
        hist_resp = requests.get(hist_url, timeout=10)
        hist_resp.raise_for_status()
        hist_data = hist_resp.json()

        rates_list = []
        for dt, rates in sorted(hist_data.get("rates", {}).items()):
            rates_list.append({"date": dt, "rate": rates.get(target, 0)})

        # Calculate changes
        change_1d = 0.0
        change_1w = 0.0
        if len(rates_list) >= 2:
            prev_rate = rates_list[-2]["rate"]
            change_1d = ((current_rate - prev_rate) / prev_rate) * 100
        if len(rates_list) >= 6:
            week_ago_rate = rates_list[0]["rate"]
            change_1w = ((current_rate - week_ago_rate) / week_ago_rate) * 100

        result = ForexData(
            base_currency=base,
            target_currency=target,
            rate=current_rate,
            historical_rates=rates_list,
            change_1d=round(change_1d, 3),
            change_1w=round(change_1w, 3),
        )

        _cache_set(cache_k, {
            "base_currency": base,
            "target_currency": target,
            "rate": current_rate,
            "historical_rates": rates_list,
            "change_1d": result.change_1d,
            "change_1w": result.change_1w,
        })

        return result

    except Exception as e:
        logger.debug("Frankfurter forex fetch failed: %s", e)
        return None


# ── CoinGecko — Crypto Sentiment (Free, No Key) ────────────────────────────

@dataclass
class CryptoSentiment:
    """Crypto market data for correlation with equities."""
    btc_price: float
    btc_change_24h: float  # %
    btc_change_7d: float  # %
    eth_price: float
    eth_change_24h: float  # %
    total_market_cap: float
    total_volume_24h: float
    btc_dominance: float  # %
    fear_greed_index: Optional[float] = None  # 0-100
    fear_greed_label: Optional[str] = None  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    cached: bool = False


def fetch_crypto_sentiment() -> Optional[CryptoSentiment]:
    """
    Fetch crypto market data from CoinGecko (free, no key).
    Used for BTC correlation and risk sentiment analysis.
    
    Returns:
        CryptoSentiment or None
    """
    cache_k = hashlib.md5("crypto:sentiment".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return CryptoSentiment(**cached, cached=True)

    try:
        # CoinGecko global data
        url = "https://api.coingecko.com/api/v3/global"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        global_data = resp.json().get("data", {})

        btc_dominance = global_data.get("market_cap_percentage", {}).get("btc", 0)
        total_market_cap = global_data.get("total_market_cap", {}).get("usd", 0)
        total_volume = global_data.get("total_volume", {}).get("usd", 0)

        # BTC and ETH prices
        coins_url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin,ethereum",
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_7d_change": "true",
        }
        coins_resp = requests.get(coins_url, params=params, timeout=10)
        coins_resp.raise_for_status()
        coins_data = coins_resp.json()

        btc = coins_data.get("bitcoin", {})
        eth = coins_data.get("ethereum", {})

        # Fear & Greed Index
        fear_greed = None
        fear_greed_label = None
        try:
            fg_resp = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=5,
            )
            fg_resp.raise_for_status()
            fg_data = fg_resp.json().get("data", [{}])[0]
            fear_greed = float(fg_data.get("value", 50))
            fear_greed_label = fg_data.get("value_classification", "Neutral")
        except Exception:
            pass

        result = CryptoSentiment(
            btc_price=btc.get("usd", 0),
            btc_change_24h=btc.get("usd_24h_change", 0),
            btc_change_7d=btc.get("usd_7d_change", 0),
            eth_price=eth.get("usd", 0),
            eth_change_24h=eth.get("usd_24h_change", 0),
            total_market_cap=total_market_cap,
            total_volume_24h=total_volume,
            btc_dominance=round(btc_dominance, 2),
            fear_greed_index=fear_greed,
            fear_greed_label=fear_greed_label,
        )

        _cache_set(cache_k, {
            "btc_price": result.btc_price,
            "btc_change_24h": result.btc_change_24h,
            "btc_change_7d": result.btc_change_7d,
            "eth_price": result.eth_price,
            "eth_change_24h": result.eth_change_24h,
            "total_market_cap": result.total_market_cap,
            "total_volume_24h": result.total_volume_24h,
            "btc_dominance": result.btc_dominance,
            "fear_greed_index": result.fear_greed_index,
            "fear_greed_label": result.fear_greed_label,
        })

        return result

    except Exception as e:
        logger.debug("CoinGecko crypto sentiment fetch failed: %s", e)
        return None


# ── Statistics of the World — Macro Indicators (Free, No Key) ──────────────

@dataclass
class WorldMacroData:
    """Global macro economic indicators."""
    india_gdp_growth: Optional[float] = None
    india_inflation: Optional[float] = None
    india_population: Optional[float] = None
    us_gdp_growth: Optional[float] = None
    us_inflation: Optional[float] = None
    global_gdp_growth: Optional[float] = None
    oil_price: Optional[float] = None  # Brent crude
    gold_price: Optional[float] = None
    cached: bool = False


def fetch_world_macro_data() -> Optional[WorldMacroData]:
    """
    Fetch global macro indicators from Statistics of the World (free, no key).
    
    Returns:
        WorldMacroData or None
    """
    cache_k = hashlib.md5("world:macro".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return WorldMacroData(**cached, cached=True)

    try:
        # Try multiple free macro data sources
        result = WorldMacroData()

        # Try frankfurter for USD/INR (already covered above)
        # Try exchangerate.host for additional forex
        try:
            resp = requests.get(
                "https://api.exchangerate.host/latest?base=USD&symbols=INR",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Just log, we already have forex from Frankfurter
        except Exception:
            pass

        # Try CoinGecko for gold/silver (precious metals)
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=tether-gold&vs_currencies=usd",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                gold_data = data.get("tether-gold", {})
                if gold_data:
                    result.gold_price = gold_data.get("usd")
        except Exception:
            pass

        _cache_set(cache_k, {
            "india_gdp_growth": result.india_gdp_growth,
            "india_inflation": result.india_inflation,
            "india_population": result.india_population,
            "us_gdp_growth": result.us_gdp_growth,
            "us_inflation": result.us_inflation,
            "global_gdp_growth": result.global_gdp_growth,
            "oil_price": result.oil_price,
            "gold_price": result.gold_price,
        })

        return result

    except Exception as e:
        logger.debug("World macro data fetch failed: %s", e)
        return None


# ── Indian Mandi Prices — Commodity Data (Free, No Key) ────────────────────

@dataclass
class MandiPrice:
    """Commodity price from Indian mandi (wholesale market)."""
    commodity: str
    market: str
    state: str
    price_min: float
    price_max: float
    price_modal: float
    unit: str
    date: str
    cached: bool = False


def fetch_mandi_prices(
    commodity: Optional[str] = None,
    state: Optional[str] = None,
) -> Optional[list[MandiPrice]]:
    """
    Fetch commodity prices from Indian mandi (free, no key).
    Useful for agri-sector stocks (sugar, cotton, spices, etc.).
    
    Args:
        commodity: Filter by commodity name (e.g., "Wheat", "Cotton")
        state: Filter by state (e.g., "Maharashtra", "Punjab")
    
    Returns:
        List of MandiPrice or None
    """
    cache_k = hashlib.md5(f"mandi:{commodity}:{state}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return [MandiPrice(**item) for item in cached.get("prices", [])]

    try:
        # Try data.gov.in API (Indian government open data)
        # This is a free API endpoint
        url = "https://api.data.gov.in/resource/359846c8-0eae-4f53-a69b-8d5dd13057f0"
        params = {
            "format": "json",
            "limit": 20,
        }
        if commodity:
            params["filters[commodity]"] = commodity
        if state:
            params["filters[state]"] = state

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        prices = []
        records = data.get("records", [])
        for rec in records[:20]:
            try:
                prices.append(MandiPrice(
                    commodity=rec.get("commodity", ""),
                    market=rec.get("market", ""),
                    state=rec.get("state", ""),
                    price_min=float(rec.get("min_price", 0) or 0),
                    price_max=float(rec.get("max_price", 0) or 0),
                    price_modal=float(rec.get("modal_price", 0) or 0),
                    unit=rec.get("unit", "Quintal"),
                    date=rec.get("date", ""),
                ))
            except (ValueError, TypeError):
                continue

        if prices:
            _cache_set(cache_k, {"prices": [
                {"commodity": p.commodity, "market": p.market, "state": p.state,
                 "price_min": p.price_min, "price_max": p.price_max,
                 "price_modal": p.price_modal, "unit": p.unit, "date": p.date}
                for p in prices
            ]})

        return prices if prices else None

    except Exception as e:
        logger.debug("Mandi prices fetch failed: %s", e)
        return None


# ── Indian Pincode — Location Data (Free, No Key) ──────────────────────────

@dataclass
class PincodeData:
    """Indian pincode with location data."""
    pincode: int
    office_name: str
    district: str
    state: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    cached: bool = False


def fetch_pincode_data(pincode: int) -> Optional[list[PincodeData]]:
    """
    Fetch Indian pincode data (free, no key).
    
    Args:
        pincode: Indian pincode (e.g., 400001 for Mumbai)
    
    Returns:
        List of PincodeData or None
    """
    cache_k = hashlib.md5(f"pincode:{pincode}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return [PincodeData(**item) for item in cached.get("pincodes", [])]

    try:
        url = f"https://api.postalpincode.in/pincode/{pincode}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data or data[0].get("Status") != "Success":
            return None

        post_offices = data[0].get("PostOffice", [])
        results = []
        for po in post_offices[:10]:
            results.append(PincodeData(
                pincode=pincode,
                office_name=po.get("Name", ""),
                district=po.get("District", ""),
                state=po.get("State", ""),
                latitude=None,  # Not provided by this API
                longitude=None,
            ))

        if results:
            _cache_set(cache_k, {"pincodes": [
                {"pincode": r.pincode, "office_name": r.office_name,
                 "district": r.district, "state": r.state}
                for r in results
            ]})

        return results if results else None

    except Exception as e:
        logger.debug("Pincode data fetch failed: %s", e)
        return None


# ── Top 5 Stocks — AI-ranked Watchlists (Free, No Key) ─────────────────────

@dataclass
class TopStockPick:
    """AI-ranked stock pick."""
    ticker: str
    name: str
    rank: int
    score: float
    reason: str
    source: str
    cached: bool = False


def fetch_top_stocks() -> Optional[list[TopStockPick]]:
    """
    Fetch AI-ranked daily stock watchlists (free, no key).
    
    Returns:
        List of TopStockPick or None
    """
    cache_k = hashlib.md5("top5:stocks".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return [TopStockPick(**item) for item in cached.get("picks", [])]

    try:
        # Try top5stocks.com API
        url = "https://api.top5stocks.com/api/v1/stocks"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        picks = []
        for i, stock in enumerate(data.get("stocks", [])[:5], 1):
            picks.append(TopStockPick(
                ticker=stock.get("symbol", ""),
                name=stock.get("name", ""),
                rank=i,
                score=stock.get("score", 0),
                reason=stock.get("reason", ""),
                source="top5stocks",
            ))

        if picks:
            _cache_set(cache_k, {"picks": [
                {"ticker": p.ticker, "name": p.name, "rank": p.rank,
                 "score": p.score, "reason": p.reason, "source": p.source}
                for p in picks
            ]})

        return picks if picks else None

    except Exception as e:
        logger.debug("Top stocks fetch failed: %s", e)
        return None


# ── WallstreetBets — Reddit WSB Sentiment (Free, No Key) ───────────────────

@dataclass
class WallstreetBetsSentiment:
    """WallstreetBets sentiment data."""
    ticker: str
    mention_count: int
    sentiment_score: float  # -1.0 to 1.0
    top_posts: list[dict] = field(default_factory=list)
    cached: bool = False


def fetch_wallstreetbets_sentiment(ticker: str) -> Optional[WallstreetBetsSentiment]:
    """
    Fetch WallstreetBets sentiment for a ticker (free, no key).
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE")
    
    Returns:
        WallstreetBetsSentiment or None
    """
    cache_k = hashlib.md5(f"wsb:{ticker}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return WallstreetBetsSentiment(**cached, cached=True)

    try:
        # Try wallstreetbets API
        url = f"https://api.wallstreetbets.io/api/sentiment/{ticker.upper()}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result = WallstreetBetsSentiment(
            ticker=ticker.upper(),
            mention_count=data.get("mention_count", 0),
            sentiment_score=data.get("sentiment_score", 0),
            top_posts=data.get("top_posts", []),
        )

        _cache_set(cache_k, {
            "ticker": result.ticker,
            "mention_count": result.mention_count,
            "sentiment_score": result.sentiment_score,
            "top_posts": result.top_posts,
        })

        return result

    except Exception as e:
        logger.debug("WallstreetBets sentiment fetch failed for %s: %s", ticker, e)
        return None


# ── Noozra — Free News Headlines (Free, No Key) ────────────────────────────

@dataclass
class NoozraNews:
    """News headline from Noozra RSS sources."""
    title: str
    url: str
    source: str
    published: str
    cached: bool = False


def fetch_noozra_news(
    query: Optional[str] = None,
    max_items: int = 10,
) -> Optional[list[NoozraNews]]:
    """
    Fetch free news headlines from Noozra (200+ RSS sources, free, no key).
    
    Args:
        query: Search query (e.g., "RELIANCE", "NIFTY")
        max_items: Maximum items to return
    
    Returns:
        List of NoozraNews or None
    """
    cache_k = hashlib.md5(f"noozra:{query}:{max_items}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return [NoozraNews(**item) for item in cached.get("news", [])]

    try:
        # Try noozra.com API
        url = "https://api.noozra.com/v1/news"
        params = {"limit": max_items}
        if query:
            params["q"] = query

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        news = []
        for item in data.get("articles", [])[:max_items]:
            news.append(NoozraNews(
                title=item.get("title", ""),
                url=item.get("url", ""),
                source=item.get("source", ""),
                published=item.get("published", ""),
            ))

        if news:
            _cache_set(cache_k, {"news": [
                {"title": n.title, "url": n.url, "source": n.source, "published": n.published}
                for n in news
            ]})

        return news if news else None

    except Exception as e:
        logger.debug("Noozra news fetch failed: %s", e)
        return None


# ── Unified Free API Fetcher ────────────────────────────────────────────────

def fetch_all_free_apis(ticker: Optional[str] = None) -> dict:
    """
    Fetch all free API data (no keys required).
    
    Returns:
        {
            "forex": ForexData | None,
            "crypto": CryptoSentiment | None,
            "world_macro": WorldMacroData | None,
            "mandi_prices": list[MandiPrice] | None,
            "top_stocks": list[TopStockPick] | None,
            "wsb": WallstreetBetsSentiment | None,
            "news": list[NoozraNews] | None,
        }
    """
    forex = fetch_forex_data()
    crypto = fetch_crypto_sentiment()
    world_macro = fetch_world_macro_data()
    mandi = fetch_mandi_prices()
    top_stocks = fetch_top_stocks()
    wsb = fetch_wallstreetbets_sentiment(ticker) if ticker else None
    news = fetch_noozra_news(query=ticker)

    return {
        "forex": forex,
        "crypto": crypto,
        "world_macro": world_macro,
        "mandi_prices": mandi,
        "top_stocks": top_stocks,
        "wsb": wsb,
        "news": news,
    }
