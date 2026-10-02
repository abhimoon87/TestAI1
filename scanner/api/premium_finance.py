"""
Shariah compliance screening — API Key Required.
"""

import hashlib
import logging
from dataclasses import dataclass

import requests

from ..shared.cache import TTLCache

logger = logging.getLogger(__name__)

_PREMIUM_CACHE: TTLCache[dict] = TTLCache(ttl=6 * 3600, namespace="premium_finance")


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

    cache_k = hashlib.md5(
        f"shariah:{ticker}".encode(), usedforsecurity=False
    ).hexdigest()
    cached = _PREMIUM_CACHE.get(cache_k)
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

        _PREMIUM_CACHE.set(
            cache_k,
            {
                "ticker": ticker,
                "is_shariah_compliant": result.is_shariah_compliant,
                "screening_method": result.screening_method,
                "purification_required": result.purification_required,
                "zakat_amount": result.zakat_amount,
            },
        )

        return result

    except Exception as e:
        logger.info("Shariah fetch failed for %s: %s", ticker, e)
        return None
