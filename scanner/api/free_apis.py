"""
Free APIs Provider — No API Key Required

Providers:
  - Noozra: Free news headlines from 200+ RSS sources (free, no key)
"""

import hashlib
import logging
from dataclasses import dataclass

import requests

from ..shared.cache import TTLCache

logger = logging.getLogger(__name__)

# ── Cache ───────────────────────────────────────────────────────────────────

_FREE_API_CACHE: TTLCache[dict] = TTLCache(ttl=4 * 3600, namespace="free_api")


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
    query: str | None = None,
    max_items: int = 10,
) -> list[NoozraNews] | None:
    """
    Fetch free news headlines from Google News RSS (free, no key).
    Fallback for Noozra API (domain may be unavailable).

    Args:
        query: Search query (e.g., "RELIANCE", "NIFTY")
        max_items: Maximum items to return

    Returns:
        List of NoozraNews or None
    """
    cache_k = hashlib.md5(
        f"news:{query}:{max_items}".encode(), usedforsecurity=False
    ).hexdigest()
    cached = _FREE_API_CACHE.get(cache_k)
    if cached:
        return [NoozraNews(**item) for item in cached.get("news", [])]

    try:
        # Use Google News RSS feed (free, no key)
        import xml.etree.ElementTree as ET

        search_query = query or "Indian stock market"
        url = f"https://news.google.com/rss/search?q={search_query}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"

        resp = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        news = []

        for item in root.findall(".//item")[:max_items]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            source = item.findtext("source", "")
            pub_date = item.findtext("pubDate", "")

            news.append(
                NoozraNews(
                    title=title[:200],
                    url=link,
                    source=source or "Google News",
                    published=pub_date,
                )
            )

        if news:
            _FREE_API_CACHE.set(
                cache_k,
                {
                    "news": [
                        {
                            "title": n.title,
                            "url": n.url,
                            "source": n.source,
                            "published": n.published,
                        }
                        for n in news
                    ]
                },
            )

        return news if news else None

    except Exception as e:
        logger.info("News fetch failed: %s", e)
        return None
