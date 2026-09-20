"""
Provider registry for the HMAxEMA Scanner.

Provides a clean public API for the 5 per-ticker data providers that
run during Phase-2 enrichment.  Each provider is lazily imported to
keep startup fast.

Usage in ``scanner_engine._enrich_with_providers``:
    from .providers import run_all_providers
    results = run_all_providers(ticker, settings, api_config)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

TIMEOUT = object()  # Sentinel for timed-out provider calls


def call_with_timeout(fn, timeout: float):
    """Run *fn* on a daemon thread; return ``TIMEOUT`` if it exceeds *timeout*."""
    result = [None]
    exc = [None]

    def _target():
        try:
            result[0] = fn()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        logger.debug("Provider %s timed out after %.1fs", getattr(fn, "__name__", "?"), timeout)
        return TIMEOUT
    if exc[0] is not None:
        raise exc[0]
    return result[0]


def fetch_sentiment(ticker: str, api_config: dict) -> dict:
    """Market sentiment (news sentiment score)."""
    from ..market_sentiment import fetch_sentiment as _fetch
    from ...backend.settings_store import get_api_key
    return _fetch(
        ticker,
        marketaux_key=get_api_key("MARKETAUX_API_KEY", api_config),
        newsapi_key=get_api_key("NEWS_API_KEY", api_config),
        gnews_key=get_api_key("GNEWS_API_KEY", api_config),
    )


def fetch_social(ticker: str, api_config: dict) -> dict:
    """Social sentiment (Twitter/X mentions)."""
    from ...backend.settings_store import get_api_key
    from ..social_sentiment import fetch_social_sentiment as _fetch
    return _fetch(
        ticker,
        twitter_api_key=get_api_key("TWITTER_API_KEY", api_config),
    )


def fetch_indian_market(ticker: str) -> dict:
    """Indian market data (delivery, FII/DII, 52-week)."""
    from ..indian_market import fetch_indian_market_data as _fetch
    return _fetch(ticker)


def fetch_indian_fundamentals(ticker: str) -> dict:
    """Indian fundamentals (P/E, quality, valuation)."""
    from ..indian_fundamentals import fetch_indian_fundamentals as _fetch
    return _fetch(ticker)


def fetch_insider(ticker: str, api_config: dict) -> dict:
    """Insider activity data."""
    from ..insider_data import fetch_insider_data as _fetch
    from ...backend.settings_store import get_api_key
    return _fetch(
        ticker,
        aletheia_key=get_api_key("ALETHEIA_API_KEY", api_config),
        congress_key=get_api_key("CONGRESS_API_KEY", api_config),
    )


def fetch_shariah(ticker: str, api_config: dict) -> Any:
    """Shariah compliance data."""
    from ..premium_finance import fetch_shariah_data as _fetch
    from ...backend.settings_store import get_api_key
    return _fetch(
        ticker,
        api_key=get_api_key("HALAL_API_KEY", api_config),
    )


# Provider registry: (name, fetch_fn, setting_key, result_mapper)
# result_mapper maps provider result dict → enriched keys dict
PROVIDERS = [
    ("sentiment", fetch_sentiment, "use_market_sentiment", {
        "_sentiment_score": lambda r: r.get("sentiment_score", 0.0),
        "_article_count": lambda r: r.get("article_count", 0),
        "_sentiment_source": lambda r: r.get("source", "none"),
    }),
    ("social", fetch_social, "use_social_sentiment", {
        "_social_score": lambda r: r.get("social_score", 0.0),
        "_mention_count": lambda r: r.get("mention_count", 0),
        "_social_source": lambda r: r.get("source", "none"),
    }),
    ("india", fetch_indian_market, "use_indian_market", {
        "_delivery_pct": lambda r: r.get("delivery").delivery_pct if r.get("delivery") else None,
        "_delivery_change_pct": lambda r: r.get("delivery").delivery_change_pct if r.get("delivery") else None,
        "_delivery_source": lambda r: "nse" if r.get("delivery") else None,
        "_fii_is_buying": lambda r: r.get("fii_dii").fii_is_buying if r.get("fii_dii") else None,
        "_dii_is_buying": lambda r: r.get("fii_dii").dii_is_buying if r.get("fii_dii") else None,
        "_fii_net": lambda r: r.get("fii_dii").fii_net if r.get("fii_dii") else None,
        "_dii_net": lambda r: r.get("fii_dii").dii_net if r.get("fii_dii") else None,
        "_institutional_source": lambda r: "nse" if r.get("fii_dii") else None,
        "_52w_position": lambda r: r.get("week52").position_in_range if r.get("week52") else None,
        "_52w_pct_from_high": lambda r: r.get("week52").pct_from_52w_high if r.get("week52") else None,
        "_52w_source": lambda r: "nse" if r.get("week52") else None,
    }),
    ("india_fund", fetch_indian_fundamentals, "use_indian_fundamentals", {
        "_pe_relative_to_industry": lambda r: (
            r.get("screener").stock_pe / r.get("screener").industry_pe
            if r.get("screener") and r.get("screener").industry_pe and r.get("screener").stock_pe
            else None
        ),
        "_is_quality_stock": lambda r: r.get("screener").is_quality if r.get("screener") else None,
        "_valuation_source": lambda r: r.get("source", "none"),
    }),
    ("insider", fetch_insider, "use_insider_data", {
        "_insider_score": lambda r: r.get("insider_score", 0.0),
        "_insider_source": lambda r: r.get("source", "none"),
    }),
]
