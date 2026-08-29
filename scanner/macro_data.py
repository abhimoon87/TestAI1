"""
Macro Economic Data Provider
Fetches FRED + EconPulse data for market regime detection.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ── Cache ───────────────────────────────────────────────────────────────────

_MACRO_CACHE: dict[str, tuple[dict, float]] = {}
_MACRO_CACHE_TTL = 6 * 3600  # 6 hours (macro data changes slowly)


def _cache_get(key: str) -> Optional[dict]:
    if key in _MACRO_CACHE:
        result, ts = _MACRO_CACHE[key]
        if time.time() - ts < _MACRO_CACHE_TTL:
            return result
    return None


def _cache_set(key: str, value: dict):
    _MACRO_CACHE[key] = (value, time.time())


# ── FRED Provider ──────────────────────────────────────────────────────────

@dataclass
class FredData:
    """Economic data from Federal Reserve (FRED)."""
    fed_funds_rate: Optional[float] = None
    unemployment_rate: Optional[float] = None
    cpi: Optional[float] = None
    gdp_growth: Optional[float] = None
    treasury_10y: Optional[float] = None
    treasury_2y: Optional[float] = None
    yield_curve_spread: Optional[float] = None  # 10Y - 2Y
    is_yield_curve_inverted: bool = False
    pmi: Optional[float] = None
    last_updated: str = ""
    cached: bool = False


def fetch_fred_data(api_key: Optional[str] = None) -> Optional[FredData]:
    """
    Fetch key economic indicators from FRED.
    
    Args:
        api_key: FRED API key (or env FRED_API_KEY)
    
    Returns:
        FredData or None on failure
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.debug("FRED: no API key, skipping")
        return None

    cache_k = hashlib.md5("fred:indicators".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return FredData(**cached, cached=True)

    series_ids = {
        "fed_funds_rate": "FEDFUNDS",
        "unemployment_rate": "UNRATE",
        "cpi": "CPIAUCSL",
        "gdp_growth": "A191RL1Q225SBEA",
        "treasury_10y": "DGS10",
        "treasury_2y": "DGS2",
        "pmi": "MANEMP",
    }

    result = {}
    try:
        for field_name, series_id in series_ids.items():
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            observations = data.get("observations", [])
            if observations:
                val = observations[0].get("value")
                if val and val != ".":
                    result[field_name] = float(val)

        # Compute yield curve spread
        if "treasury_10y" in result and "treasury_2y" in result:
            spread = result["treasury_10y"] - result["treasury_2y"]
            result["yield_curve_spread"] = round(spread, 3)
            result["is_yield_curve_inverted"] = spread < 0

        from datetime import datetime
        result["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        fred = FredData(**result)
        _cache_set(cache_k, result)
        return fred

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning("FRED fetch failed: %s", e)
        return None


# ── EconPulse Provider ─────────────────────────────────────────────────────

@dataclass
class EconPulseData:
    """Live economic data from EconPulse."""
    cpi_yoy: Optional[float] = None
    ppi_yoy: Optional[float] = None
    treasury_10y: Optional[float] = None
    breakeven_inflation: Optional[float] = None
    oil_wti: Optional[float] = None
    gold_price: Optional[float] = None
    inr_usd: Optional[float] = None
    last_updated: str = ""
    cached: bool = False


def fetch_econpulse_data(api_key: Optional[str] = None) -> Optional[EconPulseData]:
    """
    Fetch live economic data from EconPulse.
    
    Args:
        api_key: EconPulse API key (or env ECONPULSE_API_KEY)
    
    Returns:
        EconPulseData or None
    """
    api_key = api_key or os.environ.get("ECONPULSE_API_KEY")
    if not api_key:
        logger.debug("EconPulse: no API key, skipping")
        return None

    cache_k = hashlib.md5("econpulse:indicators".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return EconPulseData(**cached, cached=True)

    try:
        url = "https://api.econpulse.com/v1/latest"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result = {}
        for key in ["cpi_yoy", "ppi_yoy", "treasury_10y", "breakeven_inflation",
                     "oil_wti", "gold_price", "inr_usd"]:
            val = data.get(key)
            if val is not None:
                result[key] = float(val)

        from datetime import datetime
        result["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        ep = EconPulseData(**result)
        _cache_set(cache_k, result)
        return ep

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning("EconPulse fetch failed: %s", e)
        return None


# ── Econdb Provider (Free, no key) ────────────────────────────────────────

@dataclass
class EcondbData:
    """Global macroeconomic data from Econdb."""
    india_gdp_growth: Optional[float] = None
    india_inflation: Optional[float] = None
    india_policy_rate: Optional[float] = None
    us_10y_yield: Optional[float] = None
    crude_oil: Optional[float] = None
    last_updated: str = ""
    cached: bool = False


def fetch_econdb_data() -> Optional[EcondbData]:
    """
    Fetch macro data from Econdb (free, no API key required).
    
    Returns:
        EcondbData or None
    """
    cache_k = hashlib.md5("econdb:global".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return EcondbData(**cached, cached=True)

    try:
        # Econdb provides pre-built macro bundles
        url = "https://api.econdb.com/v1/series/US10Y,CL1.1,INREALGDPGR,INCPALTTM01IXNBY,INRREPRTD"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result = {}
        series = data.get("series", [])
        for s in series:
            ticker = s.get("ticker", "")
            values = s.get("data", [])
            if values:
                latest = values[-1].get("value") if isinstance(values[-1], dict) else values[-1]
                if latest is not None:
                    if "US10Y" in ticker:
                        result["us_10y_yield"] = float(latest)
                    elif "CL1" in ticker:
                        result["crude_oil"] = float(latest)
                    elif "INREALGDPGR" in ticker:
                        result["india_gdp_growth"] = float(latest)
                    elif "INCPALTTM" in ticker:
                        result["india_inflation"] = float(latest)
                    elif "INRREPRTD" in ticker:
                        result["india_policy_rate"] = float(latest)

        from datetime import datetime
        result["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        edb = EcondbData(**result)
        _cache_set(cache_k, result)
        return edb

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning("Econdb fetch failed: %s", e)
        return None


# ── Market Regime Detection ────────────────────────────────────────────────

@dataclass
class MarketRegime:
    """Detected market regime from macro data."""
    regime: str  # "risk_on" | "risk_off" | "recession" | "neutral"
    confidence: float  # 0.0 to 1.0
    signals: list[str] = field(default_factory=list)
    recommended_score_threshold: int = 50  # minimum score to pass filter
    active_categories: list[str] = field(default_factory=list)  # empty = all active


def detect_market_regime(
    fred: Optional[FredData] = None,
    econpulse: Optional[EconPulseData] = None,
    econdb: Optional[EcondbData] = None,
) -> MarketRegime:
    """
    Detect market regime from macro economic data.
    
    Regimes:
    - risk_on: Bull market, loose monetary policy, positive growth
    - risk_off: Defensive, tight policy, negative signals
    - recession: Yield curve inverted, high unemployment, low growth
    - neutral: Mixed signals, default regime
    """
    signals = []
    risk_on_score = 0
    risk_off_score = 0
    recession_score = 0

    # Yield curve analysis
    if fred and fred.yield_curve_spread is not None:
        spread = fred.yield_curve_spread
        if spread < -0.5:
            signals.append(f"Yield curve deeply inverted ({spread:.2f}%)")
            recession_score += 3
        elif spread < 0:
            signals.append(f"Yield curve inverted ({spread:.2f}%)")
            recession_score += 2
        elif spread > 1.0:
            signals.append(f"Yield curve steep ({spread:.2f}%)")
            risk_on_score += 2
        else:
            signals.append(f"Yield curve normal ({spread:.2f}%)")

    # Fed funds rate
    if fred and fred.fed_funds_rate is not None:
        rate = fred.fed_funds_rate
        if rate < 2.0:
            signals.append(f"Low fed funds rate ({rate:.2f}%) — accommodative")
            risk_on_score += 2
        elif rate > 5.0:
            signals.append(f"High fed funds rate ({rate:.2f}%) — restrictive")
            risk_off_score += 2
        else:
            signals.append(f"Moderate fed funds rate ({rate:.2f}%)")

    # Unemployment
    if fred and fred.unemployment_rate is not None:
        unemp = fred.unemployment_rate
        if unemp > 6.0:
            signals.append(f"High unemployment ({unemp:.1f}%)")
            recession_score += 2
        elif unemp < 4.0:
            signals.append(f"Low unemployment ({unemp:.1f}%)")
            risk_on_score += 1

    # GDP growth
    if fred and fred.gdp_growth is not None:
        gdp = fred.gdp_growth
        if gdp < -1.0:
            signals.append(f"Negative GDP growth ({gdp:.1f}%)")
            recession_score += 3
        elif gdp < 0:
            signals.append(f"Slightly negative GDP ({gdp:.1f}%)")
            risk_off_score += 1
        elif gdp > 3.0:
            signals.append(f"Strong GDP growth ({gdp:.1f}%)")
            risk_on_score += 2

    # India-specific
    if econdb and econdb.india_gdp_growth is not None:
        in_gdp = econdb.india_gdp_growth
        if in_gdp > 6.0:
            signals.append(f"India GDP strong ({in_gdp:.1f}%)")
            risk_on_score += 1
        elif in_gdp < 4.0:
            signals.append(f"India GDP weak ({in_gdp:.1f}%)")
            risk_off_score += 1

    # Crude oil (negative for India importer)
    oil = None
    if econdb and econdb.crude_oil:
        oil = econdb.crude_oil
    elif econpulse and econpulse.oil_wti:
        oil = econpulse.oil_wti

    if oil is not None:
        if oil > 100:
            signals.append(f"High crude oil (${oil:.0f}) — negative for India")
            risk_off_score += 1
        elif oil < 60:
            signals.append(f"Low crude oil (${oil:.0f}) — positive for India")
            risk_on_score += 1

    # Determine regime
    max_score = max(risk_on_score, risk_off_score, recession_score)
    total = risk_on_score + risk_off_score + recession_score + 1  # avoid /0

    if recession_score >= 4:
        regime = "recession"
        confidence = min(recession_score / total, 1.0)
        threshold = 65  # higher bar in recession
        categories = ["trend", "momentum", "rsi", "fundamentals", "sentiment"]
    elif risk_off_score > risk_on_score and risk_off_score >= 3:
        regime = "risk_off"
        confidence = min(risk_off_score / total, 1.0)
        threshold = 55
        categories = []
    elif risk_on_score > risk_off_score and risk_on_score >= 3:
        regime = "risk_on"
        confidence = min(risk_on_score / total, 1.0)
        threshold = 45
        categories = []
    else:
        regime = "neutral"
        confidence = 0.5
        threshold = 50
        categories = []

    return MarketRegime(
        regime=regime,
        confidence=round(confidence, 2),
        signals=signals,
        recommended_score_threshold=threshold,
        active_categories=categories,
    )


# ── Unified Macro Fetcher ──────────────────────────────────────────────────

def fetch_macro_data(
    fred_key: Optional[str] = None,
    econpulse_key: Optional[str] = None,
) -> dict:
    """
    Fetch all macro data and detect market regime.
    
    Returns:
        {
            "fred": FredData | None,
            "econpulse": EconPulseData | None,
            "econdb": EcondbData | None,
            "regime": MarketRegime,
        }
    """
    fred = fetch_fred_data(fred_key)
    econpulse = fetch_econpulse_data(econpulse_key)
    econdb = fetch_econdb_data()

    regime = detect_market_regime(fred, econpulse, econdb)

    return {
        "fred": fred,
        "econpulse": econpulse,
        "econdb": econdb,
        "regime": regime,
    }
