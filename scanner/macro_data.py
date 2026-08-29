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


# ── Yahoo Finance Macro Provider (Free, No Key) ───────────────────────────

@dataclass
class YahooMacroData:
    """Free macro data from Yahoo Finance (no API key required)."""
    us_10y_yield: Optional[float] = None
    us_2y_yield: Optional[float] = None
    vix: Optional[float] = None
    crude_oil_wti: Optional[float] = None
    gold_price: Optional[float] = None
    inr_usd: Optional[float] = None
    nifty_50: Optional[float] = None
    sensex: Optional[float] = None
    last_updated: str = ""
    cached: bool = False


def fetch_yahoo_macro_data() -> Optional[YahooMacroData]:
    """
    Fetch free macro data from Yahoo Finance (no API key required).
    
    Returns:
        YahooMacroData or None
    """
    cache_k = hashlib.md5("yahoo:macro".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return YahooMacroData(**cached, cached=True)

    try:
        import yfinance as yf
        
        # Yahoo Finance ticker symbols for macro data
        tickers = {
            "us_10y_yield": "^TNX",   # US 10-Year Treasury Yield
            "us_2y_yield": "^IRX",    # US 13-Week Treasury Bill (proxy for 2Y)
            "vix": "^VIX",            # CBOE Volatility Index
            "crude_oil_wti": "CL=F",  # WTI Crude Oil Futures
            "gold_price": "GC=F",     # Gold Futures
            "inr_usd": "INR=X",       # USD/INR exchange rate
            "nifty_50": "^NSEI",      # NIFTY 50 Index
            "sensex": "^BSESN",       # SENSEX Index
        }
        
        result = {}
        for field_name, ticker_symbol in tickers.items():
            try:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.fast_info
                price = info.get("lastPrice") or info.get("last_price")
                if price is not None:
                    result[field_name] = float(price)
            except Exception as e:
                logger.debug("Yahoo Finance %s failed: %s", ticker_symbol, e)
                continue

        from datetime import datetime
        result["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        ym = YahooMacroData(**result)
        _cache_set(cache_k, result)
        return ym

    except Exception as e:
        logger.warning("Yahoo Finance macro fetch failed: %s", e)
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
    yahoo: Optional[YahooMacroData] = None,
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

    # VIX analysis (from Yahoo Finance - free)
    if yahoo and yahoo.vix is not None:
        vix = yahoo.vix
        if vix > 30:
            signals.append(f"High VIX ({vix:.1f}) — fear elevated")
            risk_off_score += 2
        elif vix > 20:
            signals.append(f"Moderate VIX ({vix:.1f}) — normal volatility")
        elif vix < 15:
            signals.append(f"Low VIX ({vix:.1f}) — complacency")
            risk_on_score += 1

    # Yield curve analysis (from FRED or Yahoo)
    spread = None
    if fred and fred.yield_curve_spread is not None:
        spread = fred.yield_curve_spread
    elif yahoo and yahoo.us_10y_yield is not None and yahoo.us_2y_yield is not None:
        spread = yahoo.us_10y_yield - yahoo.us_2y_yield
    
    if spread is not None:
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


# ── Frankfurter — Exchange Rates (Free, No Key) ────────────────────────────

@dataclass
class ForexData:
    """Exchange rate data from Frankfurter API."""
    base_currency: str
    target_currency: str
    rate: float
    historical_rates: list[dict] = field(default_factory=list)
    change_1d: float = 0.0
    change_1w: float = 0.0
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
    cached = _MACRO_CACHE.get(cache_k)
    if cached:
        result, ts = cached
        if time.time() - ts < _MACRO_CACHE_TTL:
            return ForexData(**result, cached=True)

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

        _MACRO_CACHE[cache_k] = ({
            "base_currency": base,
            "target_currency": target,
            "rate": current_rate,
            "historical_rates": rates_list,
            "change_1d": result.change_1d,
            "change_1w": result.change_1w,
        }, time.time())

        return result

    except Exception as e:
        logger.debug("Frankfurter forex fetch failed: %s", e)
        return None


# ── CoinGecko — Crypto Sentiment (Free, No Key) ────────────────────────────

@dataclass
class CryptoSentiment:
    """Crypto market data for correlation with equities."""
    btc_price: float
    btc_change_24h: float
    btc_change_7d: float
    eth_price: float
    eth_change_24h: float
    total_market_cap: float
    total_volume_24h: float
    btc_dominance: float
    fear_greed_index: Optional[float] = None
    fear_greed_label: Optional[str] = None
    cached: bool = False


def fetch_crypto_sentiment() -> Optional[CryptoSentiment]:
    """
    Fetch crypto market data from CoinGecko (free, no key).
    Used for BTC correlation and risk sentiment analysis.
    
    Returns:
        CryptoSentiment or None
    """
    cache_k = hashlib.md5("crypto:sentiment".encode()).hexdigest()
    cached = _MACRO_CACHE.get(cache_k)
    if cached:
        result, ts = cached
        if time.time() - ts < _MACRO_CACHE_TTL:
            return CryptoSentiment(**result, cached=True)

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

        _MACRO_CACHE[cache_k] = ({
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
        }, time.time())

        return result

    except Exception as e:
        logger.debug("CoinGecko crypto sentiment fetch failed: %s", e)
        return None


# ── Unified Macro Fetcher ──────────────────────────────────────────────────

def fetch_macro_data(
    fred_key: Optional[str] = None,
    econpulse_key: Optional[str] = None,
) -> dict:
    """
    Fetch all macro data and detect market regime.
    
    Sources:
      - FRED (requires API key)
      - EconPulse (requires API key)
      - Econdb (free, no key)
      - Yahoo Finance (free, no key) — VIX, yields, oil, gold, INR, indices
      - Frankfurter (free, no key) — INR/USD exchange rates
      - CoinGecko (free, no key) — Crypto sentiment, BTC correlation
    
    Returns:
        {
            "fred": FredData | None,
            "econpulse": EconPulseData | None,
            "econdb": EcondbData | None,
            "yahoo": YahooMacroData | None,
            "forex": ForexData | None,
            "crypto": CryptoSentiment | None,
            "regime": MarketRegime,
        }
    """
    fred = fetch_fred_data(fred_key)
    econpulse = fetch_econpulse_data(econpulse_key)
    econdb = fetch_econdb_data()
    yahoo = fetch_yahoo_macro_data()
    forex = fetch_forex_data()
    crypto = fetch_crypto_sentiment()

    regime = detect_market_regime(fred, econpulse, econdb, yahoo)

    return {
        "fred": fred,
        "econpulse": econpulse,
        "econdb": econdb,
        "yahoo": yahoo,
        "forex": forex,
        "crypto": crypto,
        "regime": regime,
    }
