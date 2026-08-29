"""
Social Sentiment Provider
Fetches Reddit + Twitter/X data for social momentum scoring.
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

# ── Positive/Negative Word Lists (Social/Financial) ────────────────────────

SOCIAL_POSITIVE = {
    "moon", "mooning", "bullish", "buy", "buying", "long", "hold", "hodl",
    "rocket", "🚀", "💎", "💎🙌", "undervalued", "cheap", "discount",
    "breakout", "squeeze", "rally", "gain", "profit", "bull", "up",
    "strong", "beat", "upgrade", "outperform", "accumulate",
}

SOCIAL_NEGATIVE = {
    "bearish", "sell", "selling", "short", "overvalued", "expensive",
    "dump", "dumping", "crash", "plunge", "loss", "bear", "down",
    "weak", "miss", "downgrade", "underperform", "avoid", "panic",
    "bagholder", "rekt", "bubble", "scam", "fraud",
}

# ── Cache ───────────────────────────────────────────────────────────────────

_SOCIAL_CACHE: dict[str, tuple[dict, float]] = {}
_SOCIAL_CACHE_TTL = 4 * 3600  # 4 hours


def _cache_get(key: str) -> Optional[dict]:
    if key in _SOCIAL_CACHE:
        result, ts = _SOCIAL_CACHE[key]
        if time.time() - ts < _SOCIAL_CACHE_TTL:
            return result
    return None


def _cache_set(key: str, value: dict):
    _SOCIAL_CACHE[key] = (value, time.time())


def _social_sentiment(text: str) -> float:
    """Simple keyword sentiment for social text. Returns -1.0 to 1.0."""
    if not text:
        return 0.0
    words = set(re.findall(r'\b\w+\b', text.lower()))
    emojis = set(re.findall(r'[\U0001f600-\U0001f9ff]|[\U0001f300-\U0001f5ff]|[\U0001f680-\U0001f6ff]|[\U0001f900-\U0001f9ff]', text))
    all_tokens = words | emojis
    pos = len(all_tokens & SOCIAL_POSITIVE)
    neg = len(all_tokens & SOCIAL_NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


# ── Reddit Provider ────────────────────────────────────────────────────────

@dataclass
class RedditSentiment:
    """Social sentiment from Reddit."""
    ticker: str
    mention_count: int
    sentiment_score: float  # -1.0 to 1.0
    bullish_pct: float  # % of bullish posts
    bearish_pct: float  # % of bearish posts
    top_posts: list[dict] = field(default_factory=list)  # [{title, score, url}]
    subreddits: list[str] = field(default_factory=list)
    cached: bool = False


def fetch_reddit_sentiment(
    ticker: str,
    subreddits: Optional[list[str]] = None,
    limit: int = 25,
) -> Optional[RedditSentiment]:
    """
    Fetch Reddit sentiment for a ticker.
    Uses Reddit's public JSON API (no auth required for read).
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE")
        subreddits: Subreddits to search (default: IndianStreetBets, stocks, wallstreetbets)
        limit: Max posts per subreddit
    
    Returns:
        RedditSentiment or None
    """
    if subreddits is None:
        subreddits = [
            "IndianStreetBets",
            "IndianStockMarket",
            "stocks",
            "wallstreetbets",
            "investing",
        ]

    # Strip suffixes for Reddit search
    symbol = ticker.replace(".NS", "").replace(".BO", "")

    cache_k = hashlib.md5(f"reddit:{symbol}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return RedditSentiment(**cached, cached=True)

    all_posts = []
    sentiments = []
    bullish = 0
    bearish = 0

    headers = {"User-Agent": "StockScanner/1.0"}

    for sub in subreddits:
        try:
            url = f"https://www.reddit.com/r/{sub}/search.json"
            params = {
                "q": symbol,
                "restrict_sr": "on",
                "sort": "new",
                "t": "week",
                "limit": limit,
            }
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 429:
                logger.debug("Reddit rate limited for r/%s", sub)
                time.sleep(2)
                continue
            resp.raise_for_status()
            data = resp.json()

            posts = data.get("data", {}).get("children", [])
            for post in posts:
                p_data = post.get("data", {})
                title = p_data.get("title", "")
                selftext = p_data.get("selftext", "")[:500]
                text = f"{title} {selftext}"
                score = _social_sentiment(text)

                sentiments.append(score)
                if score > 0.1:
                    bullish += 1
                elif score < -0.1:
                    bearish += 1

                all_posts.append({
                    "title": title[:120],
                    "score": p_data.get("score", 0),
                    "url": f"https://reddit.com{p_data.get('permalink', '')}",
                    "subreddit": sub,
                    "sentiment": round(score, 3),
                })

        except (requests.RequestException, KeyError, ValueError) as e:
            logger.debug("Reddit search failed for r/%s: %s", sub, e)
            continue

    if not sentiments:
        return RedditSentiment(
            ticker=ticker, mention_count=0, sentiment_score=0.0,
            bullish_pct=0.0, bearish_pct=0.0, top_posts=[],
            subreddits=[], cached=False,
        )

    avg_sentiment = sum(sentiments) / len(sentiments)
    total = bullish + bearish + 1
    bull_pct = bullish / total
    bear_pct = bearish / total

    # Sort by score and take top 5
    all_posts.sort(key=lambda x: abs(x.get("score", 0)), reverse=True)
    top = all_posts[:5]

    result = RedditSentiment(
        ticker=ticker,
        mention_count=len(sentiments),
        sentiment_score=round(avg_sentiment, 3),
        bullish_pct=round(bull_pct, 3),
        bearish_pct=round(bear_pct, 3),
        top_posts=top,
        subreddits=list(set(p["subreddit"] for p in all_posts)),
        cached=False,
    )

    _cache_set(cache_k, {
        "ticker": ticker,
        "mention_count": result.mention_count,
        "sentiment_score": result.sentiment_score,
        "bullish_pct": result.bullish_pct,
        "bearish_pct": result.bearish_pct,
        "top_posts": result.top_posts,
        "subreddits": result.subreddits,
    })

    return result


# ── Twitter/X Provider ─────────────────────────────────────────────────────

@dataclass
class TwitterSentiment:
    """Social sentiment from Twitter/X."""
    ticker: str
    mention_count: int
    sentiment_score: float  # -1.0 to 1.0
    avg_retweets: float
    avg_likes: float
    top_tweets: list[dict] = field(default_factory=list)
    cached: bool = False


def fetch_twitter_sentiment(
    ticker: str,
    api_key: Optional[str] = None,
    max_results: int = 20,
) -> Optional[TwitterSentiment]:
    """
    Fetch Twitter/X sentiment for a ticker.
    Uses GetXAPI or TweetAPI (third-party Twitter data providers).
    
    Args:
        ticker: Stock ticker
        api_key: API key for GetXAPI/TweetAPI (or env TWITTER_API_KEY)
        max_results: Max tweets to analyze
    
    Returns:
        TwitterSentiment or None
    """
    api_key = api_key or os.environ.get("TWITTER_API_KEY")
    if not api_key:
        logger.debug("Twitter: no API key, skipping")
        return None

    symbol = ticker.replace(".NS", "").replace(".BO", "")

    cache_k = hashlib.md5(f"twitter:{symbol}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return TwitterSentiment(**cached, cached=True)

    try:
        # Try GetXAPI first
        url = "https://api.getxapi.com/v2/tweets/search"
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {
            "query": f"${symbol} OR #{symbol} stock",
            "max_results": max_results,
            "sort": "recent",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        tweets = data.get("data", data.get("tweets", []))
        if not tweets:
            return TwitterSentiment(
                ticker=ticker, mention_count=0, sentiment_score=0.0,
                avg_retweets=0, avg_likes=0, top_tweets=[], cached=False,
            )

        sentiments = []
        total_retweets = 0
        total_likes = 0
        top_tweets = []

        for tweet in tweets[:max_results]:
            text = tweet.get("text", tweet.get("full_text", ""))
            score = _social_sentiment(text)
            sentiments.append(score)

            retweets = tweet.get("retweet_count", 0)
            likes = tweet.get("favorite_count", tweet.get("like_count", 0))
            total_retweets += retweets
            total_likes += likes

            top_tweets.append({
                "text": text[:200],
                "retweets": retweets,
                "likes": likes,
                "sentiment": round(score, 3),
            })

        n = len(sentiments) or 1
        avg_sentiment = sum(sentiments) / n

        # Sort by engagement
        top_tweets.sort(key=lambda x: x.get("retweets", 0) + x.get("likes", 0), reverse=True)

        result = TwitterSentiment(
            ticker=ticker,
            mention_count=len(sentiments),
            sentiment_score=round(avg_sentiment, 3),
            avg_retweets=round(total_retweets / n, 1),
            avg_likes=round(total_likes / n, 1),
            top_tweets=top_tweets[:5],
            cached=False,
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "mention_count": result.mention_count,
            "sentiment_score": result.sentiment_score,
            "avg_retweets": result.avg_retweets,
            "avg_likes": result.avg_likes,
            "top_tweets": result.top_tweets,
        })

        return result

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning("Twitter sentiment failed for %s: %s", ticker, e)
        return None


# ── Unified Social Sentiment ───────────────────────────────────────────────

def fetch_social_sentiment(
    ticker: str,
    twitter_api_key: Optional[str] = None,
    subreddits: Optional[list[str]] = None,
) -> dict:
    """
    Fetch social sentiment from Reddit + Twitter with fallback.
    
    Returns:
        {
            "social_score": float,  # -1.0 to 1.0
            "mention_count": int,
            "source": str,  # "reddit+twitter" | "reddit" | "twitter" | "none"
            "reddit": RedditSentiment | None,
            "twitter": TwitterSentiment | None,
        }
    """
    reddit = fetch_reddit_sentiment(ticker, subreddits)
    twitter = fetch_twitter_sentiment(ticker, twitter_api_key)

    # Weight Reddit more heavily (more relevant for Indian stocks)
    scores = []
    weights = []
    mention_total = 0

    if reddit and reddit.mention_count > 0:
        scores.append(reddit.sentiment_score)
        weights.append(2.0)  # Reddit weighted 2x
        mention_total += reddit.mention_count

    if twitter and twitter.mention_count > 0:
        scores.append(twitter.sentiment_score)
        weights.append(1.0)
        mention_total += twitter.mention_count

    if not scores:
        return {
            "social_score": 0.0,
            "mention_count": 0,
            "source": "none",
            "reddit": None,
            "twitter": None,
        }

    weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    sources = []
    if reddit and reddit.mention_count > 0:
        sources.append("reddit")
    if twitter and twitter.mention_count > 0:
        sources.append("twitter")

    return {
        "social_score": round(weighted_score, 3),
        "mention_count": mention_total,
        "source": "+".join(sources),
        "reddit": reddit,
        "twitter": twitter,
    }
