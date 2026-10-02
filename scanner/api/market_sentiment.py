"""
Market Sentiment Provider
Headline keyword scoring across news providers with fallback:
MarketAux -> NewsAPI -> GNews -> Yahoo Finance -> Noozra RSS.
"""

import logging
import os
import re
from datetime import datetime, timedelta

import requests

from ..shared.cache import TTLCache

logger = logging.getLogger(__name__)

# ── Sentiment Word Lists ────────────────────────────────────────────────

POSITIVE_WORDS = {
    "surge",
    "surges",
    "surging",
    "rally",
    "rallies",
    "rallying",
    "gain",
    "gains",
    "gaining",
    "bullish",
    "buy",
    "buying",
    "upgrade",
    "upgrades",
    "outperform",
    "beat",
    "beats",
    "beating",
    "record",
    "high",
    "strong",
    "strength",
    "profit",
    "profits",
    "profitable",
    "growth",
    "growing",
    "positive",
    "optimistic",
    "recovery",
    "recovering",
    "breakout",
    "momentum",
    "upside",
    "boom",
    "soar",
    "soars",
    "soaring",
    "jump",
    "jumps",
    "jumping",
    "climb",
    "climbs",
    "climbing",
    "advance",
    "advances",
    "rising",
    "upbeat",
    "exceeds",
    "exceed",
    "superior",
    "dividend",
    "buyback",
    "expansion",
    "innovative",
    "leader",
    "dominant",
}

NEGATIVE_WORDS = {
    "crash",
    "crashes",
    "crashing",
    "plunge",
    "plunges",
    "plunging",
    "drop",
    "drops",
    "dropping",
    "bearish",
    "sell",
    "selling",
    "downgrade",
    "downgrades",
    "underperform",
    "miss",
    "misses",
    "missing",
    "loss",
    "losses",
    "loss-making",
    "decline",
    "declines",
    "declining",
    "negative",
    "pessimistic",
    "recession",
    "recessionary",
    "breakdown",
    "weakness",
    "weak",
    "downside",
    "bust",
    "slump",
    "slumps",
    "slumping",
    "fall",
    "falls",
    "falling",
    "retreat",
    "retreats",
    "downturn",
    "crisis",
    "debt",
    "default",
    "bankruptcy",
    "insolvent",
    "fraud",
    "scandal",
    "investigation",
    "lawsuit",
    "penalty",
    "fine",
    "warning",
    "cut",
    "cuts",
    "cutting",
    "reduce",
    "reduces",
    "reducing",
    "layoff",
    "layoffs",
    "restructure",
    "restructuring",
    "impairment",
    "write-down",
    "overvalued",
}

# ── Cache ─────────────────────────────────────────────────────

_SENTIMENT_CACHE: TTLCache[dict] = TTLCache(ttl=4 * 3600, namespace="market_sentiment")


def _cache_key(ticker: str, source: str) -> str:
    return _SENTIMENT_CACHE.make_key(ticker, source)


# ── Simple Keyword Sentiment ───────────────────────────────────


def _keyword_sentiment(text: str) -> float:
    """Compute sentiment from text using keyword matching. Returns -1.0 to 1.0."""
    if not text:
        return 0.0
    words = set(re.findall(r"\b\w+\b", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


# ── Provider article fetchers (return raw article lists) ───────


def _title_desc(article: dict) -> str:
    return f"{article.get('title', '')} {article.get('description', '')}"


def _lookback(days: int, fmt: str) -> str:
    return (datetime.now() - timedelta(days=days)).strftime(fmt)


def _marketaux_articles(ticker: str, api_key: str) -> list:
    """MarketAux: entity lookup, then ticker-tagged news (articles key 'data')."""
    symbol = ticker.replace(".NS", "").replace(".BO", "")
    resp = requests.get(
        "https://api.marketaux.com/v1/entity/search",
        params={"search": symbol, "api_token": api_key},
        timeout=10,
    )
    resp.raise_for_status()
    entities = resp.json().get("data", [])
    if not entities or not entities[0].get("entity_id"):
        return []
    resp = requests.get(
        "https://api.marketaux.com/v1/news",
        params={
            "entity_ids": entities[0]["entity_id"],
            "api_token": api_key,
            "published_after": _lookback(7, "%Y-%m-%d"),
            "limit": 50,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def _newsapi_articles(ticker: str, api_key: str) -> list:
    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": ticker.replace(".NS", "").replace(".BO", ""),
            "from": _lookback(7, "%Y-%m-%d"),
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 50,
            "apiKey": api_key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("articles", [])


def _gnews_articles(ticker: str, api_key: str) -> list:
    resp = requests.get(
        "https://gnews.io/api/v4/search",
        params={
            "q": ticker.replace(".NS", "").replace(".BO", ""),
            "from": _lookback(7, "%Y-%m-%dT00:00:00Z"),
            "lang": "en",
            "max": 10,
            "token": api_key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("articles", [])


def _yfinance_articles(ticker: str, api_key: str | None = None) -> list:
    import yfinance as yf

    nse_ticker = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
    return yf.Ticker(nse_ticker).news or []


# ── Shared provider flow: cache -> fetch -> keyword-score -> cache ─


def _fetch_headline_provider(
    ticker: str,
    source: str,
    fetch_articles,
    text_of,
    api_key: str | None,
    score_n: int | None = None,
) -> dict | None:
    """Score one provider's articles, or None when it yields nothing.

    Returns {"sentiment_score", "article_count"} and caches it for 4h.
    Failures are logged and treated as "provider unavailable".
    """
    cache_k = _cache_key(ticker, source)
    cached = _SENTIMENT_CACHE.get(cache_k)
    if cached:
        return cached
    try:
        articles = fetch_articles(ticker, api_key)
        if not articles:
            return None
        scored = articles if score_n is None else articles[:score_n]
        scores = [_keyword_sentiment(text_of(a)) for a in scored]
        result = {
            "sentiment_score": round(sum(scores) / len(scores), 3),
            "article_count": len(articles),
        }
        _SENTIMENT_CACHE.set(cache_k, result)
        return result
    except Exception as e:
        logger.warning("%s failed for %s: %s", source, ticker, e)
        return None


# ── Unified Sentiment Fetcher ────────────────────────────────────────────────


def fetch_sentiment(
    ticker: str,
    marketaux_key: str | None = None,
    newsapi_key: str | None = None,
    gnews_key: str | None = None,
) -> dict:
    """Headline keyword sentiment with provider fallback.

    Priority: MarketAux (ticker-tagged) -> NewsAPI -> GNews -> Yahoo Finance
    -> Noozra RSS. Returns {sentiment_score, article_count, source};
    source="none" when every provider fails.
    """
    providers = (
        # (source, api_key, fetch_articles, text_of, score_n)
        (
            "marketaux",
            marketaux_key or os.environ.get("MARKETAUX_API_KEY"),
            _marketaux_articles,
            _title_desc,
            None,
        ),
        (
            "newsapi",
            newsapi_key or os.environ.get("NEWSAPI_KEY"),
            _newsapi_articles,
            _title_desc,
            20,
        ),
        (
            "gnews",
            gnews_key or os.environ.get("GNEWS_API_KEY"),
            _gnews_articles,
            _title_desc,
            None,
        ),
        (
            "yfinance",
            None,
            _yfinance_articles,
            lambda a: f"{a.get('title', '')} {a.get('publisher', '')}",
            20,
        ),
    )
    for source, key, fetch_articles, text_of, score_n in providers:
        if source != "yfinance" and not key:
            continue  # key-gated providers without a key are skipped
        result = _fetch_headline_provider(
            ticker, source, fetch_articles, text_of, key, score_n
        )
        if result:
            return {**result, "source": source}

    # Noozra RSS fallback (free, no key)
    try:
        from .free_apis import fetch_noozra_news

        noozra = fetch_noozra_news(query=ticker, max_items=10)
        if noozra:
            combined_text = " ".join(n.title for n in noozra)
            if combined_text.strip():
                return {
                    "sentiment_score": _keyword_sentiment(combined_text),
                    "article_count": len(noozra),
                    "source": "noozra",
                }
    except Exception as e:
        logger.info("Noozra news fetch failed for %s: %s", ticker, e)

    return {"sentiment_score": 0.0, "article_count": 0, "source": "none"}
