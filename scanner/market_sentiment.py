"""
Market Sentiment Provider
Combines MarketAux (ticker-tagged sentiment) + NewsAPI/GNews (headlines)
for news-based sentiment scoring.
"""

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Sentiment Word Lists ────────────────────────────────────────────────────

POSITIVE_WORDS = {
    "surge", "surges", "surging", "rally", "rallies", "rallying", "gain", "gains",
    "gaining", "bullish", "buy", "buying", "upgrade", "upgrades", "outperform",
    "beat", "beats", "beating", "record", "high", "strong", "strength", "profit",
    "profits", "profitable", "growth", "growing", "positive", "optimistic",
    "recovery", "recovering", "breakout", "momentum", "upside", "boom", "soar",
    "soars", "soaring", "jump", "jumps", "jumping", "climb", "climbs", "climbing",
    "advance", "advances", "rising", "upbeat", "exceeds", "exceed", "superior",
    "dividend", "buyback", "expansion", "innovative", "leader", "dominant",
}

NEGATIVE_WORDS = {
    "crash", "crashes", "crashing", "plunge", "plunges", "plunging", "drop",
    "drops", "dropping", "bearish", "sell", "selling", "downgrade", "downgrades",
    "underperform", "miss", "misses", "missing", "loss", "losses", "loss-making",
    "decline", "declines", "declining", "negative", "pessimistic", "recession",
    "recessionary", "breakdown", "weakness", "weak", "downside", "bust", "slump",
    "slumps", "slumping", "fall", "falls", "falling", "retreat", "retreats",
    "decline", "downturn", "crisis", "debt", "default", "bankruptcy", "insolvent",
    "fraud", "scandal", "investigation", "lawsuit", "penalty", "fine", "warning",
    "cut", "cuts", "cutting", "reduce", "reduces", "reducing", "layoff", "layoffs",
    "restructure", "restructuring", "impairment", "write-down", "overvalued",
}

# ── Cache ───────────────────────────────────────────────────────────────────

_SENTIMENT_CACHE: dict[str, tuple[dict, float]] = {}
_SENTIMENT_CACHE_TTL = 4 * 3600  # 4 hours


def _cache_key(ticker: str, source: str) -> str:
    return hashlib.md5(f"{ticker}:{source}".encode()).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    if key in _SENTIMENT_CACHE:
        result, ts = _SENTIMENT_CACHE[key]
        if time.time() - ts < _SENTIMENT_CACHE_TTL:
            return result
    return None


def _cache_set(key: str, value: dict):
    _SENTIMENT_CACHE[key] = (value, time.time())


# ── Simple Keyword Sentiment ───────────────────────────────────────────────

def _keyword_sentiment(text: str) -> float:
    """Compute sentiment from text using keyword matching. Returns -1.0 to 1.0."""
    if not text:
        return 0.0
    words = set(re.findall(r'\b\w+\b', text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


# ── MarketAux Provider ─────────────────────────────────────────────────────

@dataclass
class MarketAuxSentiment:
    """News sentiment from MarketAux API (ticker-tagged)."""
    ticker: str
    sentiment_score: float  # -1.0 to 1.0
    article_count: int
    sources: list[str] = field(default_factory=list)
    cached: bool = False


def fetch_marketaux_sentiment(
    ticker: str,
    api_key: Optional[str] = None,
    days: int = 7,
) -> Optional[MarketAuxSentiment]:
    """
    Fetch news sentiment for a ticker from MarketAux.
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE.NS")
        api_key: MarketAux API key (or env MARKETAUX_API_KEY)
        days: Lookback period in days
    
    Returns:
        MarketAuxSentiment or None on failure
    """
    api_key = api_key or os.environ.get("MARKETAUX_API_KEY")
    if not api_key:
        logger.debug("MarketAux: no API key, skipping")
        return None

    # Check cache
    cache_k = _cache_key(ticker, "marketaux")
    cached = _cache_get(cache_k)
    if cached:
        return MarketAuxSentiment(**cached, cached=True)

    # Strip .NS/.BO suffix for MarketAux
    symbol = ticker.replace(".NS", "").replace(".BO", "")

    try:
        from datetime import datetime, timedelta
        date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        url = "https://api.marketaux.com/v1/entity/search"
        params = {"search": symbol, "api_token": api_key}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        entities = data.get("data", [])
        if not entities:
            logger.debug("MarketAux: no entity found for %s", symbol)
            return None

        entity_id = entities[0].get("entity_id")
        if not entity_id:
            return None

        # Fetch news for entity
        url = "https://api.marketaux.com/v1/news"
        params = {
            "entity_ids": entity_id,
            "api_token": api_key,
            "published_after": date_from,
            "limit": 50,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        news_data = resp.json()

        articles = news_data.get("data", [])
        if not articles:
            return MarketAuxSentiment(
                ticker=ticker, sentiment_score=0.0,
                article_count=0, sources=[], cached=False,
            )

        # Compute sentiment from article titles + descriptions
        sentiments = []
        sources = set()
        for article in articles:
            title = article.get("title", "")
            desc = article.get("description", "")
            text = f"{title} {desc}"
            sentiments.append(_keyword_sentiment(text))
            source = article.get("source", "")
            if source:
                sources.add(source)

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        result = MarketAuxSentiment(
            ticker=ticker,
            sentiment_score=round(avg_sentiment, 3),
            article_count=len(articles),
            sources=list(sources)[:5],
            cached=False,
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "sentiment_score": result.sentiment_score,
            "article_count": result.article_count,
            "sources": result.sources,
        })

        return result

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning("MarketAux failed for %s: %s", ticker, e)
        return None


# ── NewsAPI Provider ───────────────────────────────────────────────────────

@dataclass
class NewsAPISentiment:
    """News sentiment from NewsAPI.org."""
    ticker: str
    sentiment_score: float
    article_count: int
    top_headlines: list[str] = field(default_factory=list)
    cached: bool = False


def fetch_newsapi_sentiment(
    ticker: str,
    api_key: Optional[str] = None,
    days: int = 7,
) -> Optional[NewsAPISentiment]:
    """
    Fetch news sentiment from NewsAPI.org.
    
    Args:
        ticker: Stock ticker
        api_key: NewsAPI key (or env NEWSAPI_KEY)
        days: Lookback days
    
    Returns:
        NewsAPISentiment or None
    """
    api_key = api_key or os.environ.get("NEWSAPI_KEY")
    if not api_key:
        logger.debug("NewsAPI: no API key, skipping")
        return None

    cache_k = _cache_key(ticker, "newsapi")
    cached = _cache_get(cache_k)
    if cached:
        return NewsAPISentiment(**cached, cached=True)

    symbol = ticker.replace(".NS", "").replace(".BO", "")

    try:
        from datetime import datetime, timedelta
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 50,
            "apiKey": api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("articles", [])
        if not articles:
            return NewsAPISentiment(
                ticker=ticker, sentiment_score=0.0,
                article_count=0, top_headlines=[], cached=False,
            )

        sentiments = []
        headlines = []
        for article in articles[:20]:
            title = article.get("title", "") or ""
            desc = article.get("description", "") or ""
            text = f"{title} {desc}"
            sentiments.append(_keyword_sentiment(text))
            if title and len(headlines) < 3:
                headlines.append(title[:120])

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        result = NewsAPISentiment(
            ticker=ticker,
            sentiment_score=round(avg_sentiment, 3),
            article_count=len(articles),
            top_headlines=headlines,
            cached=False,
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "sentiment_score": result.sentiment_score,
            "article_count": result.article_count,
            "top_headlines": result.top_headlines,
        })

        return result

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning("NewsAPI failed for %s: %s", ticker, e)
        return None


# ── GNews Provider (Free, no key) ──────────────────────────────────────────

@dataclass
class GNewsSentiment:
    """News sentiment from GNews API (free tier)."""
    ticker: str
    sentiment_score: float
    article_count: int
    cached: bool = False


def fetch_gnews_sentiment(
    ticker: str,
    api_key: Optional[str] = None,
    days: int = 7,
) -> Optional[GNewsSentiment]:
    """
    Fetch news sentiment from GNews (free, 100 requests/day).
    
    Args:
        ticker: Stock ticker
        api_key: GNews API key (or env GNEWS_API_KEY)
        days: Lookback days
    
    Returns:
        GNewsSentiment or None
    """
    api_key = api_key or os.environ.get("GNEWS_API_KEY")
    if not api_key:
        logger.debug("GNews: no API key, skipping")
        return None

    cache_k = _cache_key(ticker, "gnews")
    cached = _cache_get(cache_k)
    if cached:
        return GNewsSentiment(**cached, cached=True)

    symbol = ticker.replace(".NS", "").replace(".BO", "")

    try:
        from datetime import datetime, timedelta
        when = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")

        url = "https://gnews.io/api/v4/search"
        params = {
            "q": symbol,
            "from": when,
            "lang": "en",
            "max": 10,
            "token": api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("articles", [])
        if not articles:
            return GNewsSentiment(
                ticker=ticker, sentiment_score=0.0,
                article_count=0, cached=False,
            )

        sentiments = []
        for article in articles:
            title = article.get("title", "")
            desc = article.get("description", "")
            text = f"{title} {desc}"
            sentiments.append(_keyword_sentiment(text))

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        result = GNewsSentiment(
            ticker=ticker,
            sentiment_score=round(avg_sentiment, 3),
            article_count=len(articles),
            cached=False,
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "sentiment_score": result.sentiment_score,
            "article_count": result.article_count,
        })

        return result

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning("GNews failed for %s: %s", ticker, e)
        return None


# ── Yahoo Finance News Provider (Free, No Key) ─────────────────────────────

@dataclass
class YFinanceNewsSentiment:
    """News sentiment from Yahoo Finance (free, no API key required)."""
    ticker: str
    sentiment_score: float
    article_count: int
    top_headlines: list[str] = field(default_factory=list)
    cached: bool = False


def fetch_yfinance_news_sentiment(ticker: str) -> Optional[YFinanceNewsSentiment]:
    """
    Fetch news sentiment from Yahoo Finance (free, no API key).
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE.NS")
    
    Returns:
        YFinanceNewsSentiment or None
    """
    cache_k = _cache_key(ticker, "yfinance_news")
    cached = _cache_get(cache_k)
    if cached:
        return YFinanceNewsSentiment(**cached, cached=True)

    try:
        import yfinance as yf
        
        nse_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        stock = yf.Ticker(nse_ticker)
        news = stock.news  # Returns list of dicts with title, publisher, link, etc.
        
        if not news:
            return YFinanceNewsSentiment(
                ticker=ticker, sentiment_score=0.0,
                article_count=0, top_headlines=[], cached=False,
            )

        sentiments = []
        headlines = []
        
        for article in news[:20]:  # Analyze up to 20 articles
            title = article.get("title", "")
            publisher = article.get("publisher", "")
            # Combine title + publisher for sentiment
            text = f"{title} {publisher}"
            score = _keyword_sentiment(text)
            sentiments.append(score)
            
            if title and len(headlines) < 3:
                headlines.append(title[:120])

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        result = YFinanceNewsSentiment(
            ticker=ticker,
            sentiment_score=round(avg_sentiment, 3),
            article_count=len(news),
            top_headlines=headlines,
            cached=False,
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "sentiment_score": result.sentiment_score,
            "article_count": result.article_count,
            "top_headlines": result.top_headlines,
        })

        return result

    except Exception as e:
        logger.debug("Yahoo Finance news failed for %s: %s", ticker, e)
        return None


# ── Unified Sentiment Fetcher ──────────────────────────────────────────────

def fetch_sentiment(
    ticker: str,
    marketaux_key: Optional[str] = None,
    newsapi_key: Optional[str] = None,
    gnews_key: Optional[str] = None,
) -> dict:
    """
    Fetch news sentiment from multiple sources with fallback.
    
    Priority:
      1. MarketAux (ticker-tagged, best quality)
      2. NewsAPI (80k+ sources)
      3. GNews (free tier)
      4. Yahoo Finance (free, no key - fallback)
    
    Returns:
        {
            "sentiment_score": float,  # -1.0 to 1.0
            "article_count": int,
            "source": str,  # "marketaux" | "newsapi" | "gnews" | "yfinance" | "none"
            "top_headlines": list[str],
        }
    """
    # Try MarketAux first (best: ticker-tagged)
    ma = fetch_marketaux_sentiment(ticker, marketaux_key)
    if ma and ma.article_count > 0:
        return {
            "sentiment_score": ma.sentiment_score,
            "article_count": ma.article_count,
            "source": "marketaux",
            "top_headlines": [],
        }

    # Try NewsAPI
    na = fetch_newsapi_sentiment(ticker, newsapi_key)
    if na and na.article_count > 0:
        return {
            "sentiment_score": na.sentiment_score,
            "article_count": na.article_count,
            "source": "newsapi",
            "top_headlines": na.top_headlines,
        }

    # Try GNews
    gn = fetch_gnews_sentiment(ticker, gnews_key)
    if gn and gn.article_count > 0:
        return {
            "sentiment_score": gn.sentiment_score,
            "article_count": gn.article_count,
            "source": "gnews",
            "top_headlines": [],
        }

    # Try Yahoo Finance (free fallback, no key needed)
    yf_news = fetch_yfinance_news_sentiment(ticker)
    if yf_news and yf_news.article_count > 0:
        return {
            "sentiment_score": yf_news.sentiment_score,
            "article_count": yf_news.article_count,
            "source": "yfinance",
            "top_headlines": yf_news.top_headlines,
        }

    # ── Noozra RSS (free, no key) ──────────────────────────────────────
    try:
        from .free_apis import fetch_noozra_news
        noozra = fetch_noozra_news(query=ticker, max_items=10)
        if noozra:
            # Keyword sentiment on headlines
            headlines = [n.title for n in noozra]
            combined_text = " ".join(headlines)
            kw_score = _keyword_sentiment(combined_text)
            if combined_text.strip():
                return {
                    "sentiment_score": kw_score,
                    "article_count": len(noozra),
                    "source": "noozra",
                    "top_headlines": headlines[:5],
                }
    except Exception as e:
        logger.debug("Noozra news fetch failed for %s: %s", ticker, e)

    # No sentiment data
    return {
        "sentiment_score": 0.0,
        "article_count": 0,
        "source": "none",
        "top_headlines": [],
    }
