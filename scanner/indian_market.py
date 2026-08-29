"""
Indian Market Data Provider
Free data from NSE India — delivery volume, FII/DII activity, 52-week data.
All data is fetched without API keys using nselib or direct NSE API calls.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Cache ───────────────────────────────────────────────────────────────────

_INDIA_CACHE: dict[str, tuple[dict, float]] = {}
_INDIA_CACHE_TTL = 4 * 3600  # 4 hours


def _cache_get(key: str) -> Optional[dict]:
    if key in _INDIA_CACHE:
        result, ts = _INDIA_CACHE[key]
        if time.time() - ts < _INDIA_CACHE_TTL:
            return result
    return None


def _cache_set(key: str, value: dict):
    _INDIA_CACHE[key] = (value, time.time())


# ── Delivery Volume Data ───────────────────────────────────────────────────

@dataclass
class DeliveryData:
    """Delivery volume data for a stock from NSE."""
    ticker: str
    delivery_pct: float  # Delivery volume as % of total volume
    delivery_volume: int
    total_volume: int
    delivery_change_pct: float  # Change vs previous day
    is_high_delivery: bool  # > 60% delivery
    cached: bool = False


def fetch_delivery_data(ticker: str, days: int = 5) -> Optional[DeliveryData]:
    """
    Fetch delivery volume data from NSE (free, no API key).
    
    High delivery % indicates institutional/strong hands buying.
    
    Args:
        ticker: NSE ticker symbol (e.g., "RELIANCE")
        days: Lookback days for trend
    
    Returns:
        DeliveryData or None
    """
    cache_k = hashlib.md5(f"delivery:{ticker}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return DeliveryData(**cached, cached=True)

    try:
        from nselib import capital_market
        
        # Fetch price volume and deliverable position data
        from datetime import date, timedelta
        end = date.today()
        start = end - timedelta(days=days + 5)  # Extra days for buffer
        
        df = capital_market.price_volume_and_deliverable_position_data(
            symbol=ticker,
            from_date=start.strftime("%d-%m-%Y"),
            to_date=end.strftime("%d-%m-%Y")
        )
        
        if df is None or df.empty:
            return None

        # Find deliverable columns
        delivery_col = None
        volume_col = None
        for col in df.columns:
            cl = col.lower().strip()
            if "deliverable" in cl or "delivery" in cl:
                delivery_col = col
            elif "quantity" in cl or "volume" in cl or "traded" in cl:
                volume_col = col

        if not delivery_col or not volume_col:
            return None

        # Get latest data
        df = df.sort_index(ascending=False)  # Most recent first
        latest = df.iloc[0]
        
        delivery_vol = int(latest.get(delivery_col, 0) or 0)
        total_vol = int(latest.get(volume_col, 0) or 0)
        
        if total_vol == 0:
            return None

        delivery_pct = (delivery_vol / total_vol) * 100

        # Calculate change vs previous day
        if len(df) >= 2:
            prev = df.iloc[1]
            prev_delivery = int(prev.get(delivery_col, 0) or 0)
            prev_total = int(prev.get(volume_col, 0) or 0)
            if prev_total > 0:
                prev_pct = (prev_delivery / prev_total) * 100
                delivery_change = delivery_pct - prev_pct
            else:
                delivery_change = 0.0
        else:
            delivery_change = 0.0

        result = DeliveryData(
            ticker=ticker,
            delivery_pct=round(delivery_pct, 2),
            delivery_volume=delivery_vol,
            total_volume=total_vol,
            delivery_change_pct=round(delivery_change, 2),
            is_high_delivery=delivery_pct > 60.0,
            cached=False,
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "delivery_pct": result.delivery_pct,
            "delivery_volume": result.delivery_volume,
            "total_volume": result.total_volume,
            "delivery_change_pct": result.delivery_change_pct,
            "is_high_delivery": result.is_high_delivery,
        })

        return result

    except Exception as e:
        logger.debug("Delivery data fetch failed for %s: %s", ticker, e)
        return None


# ── FII/DII Activity Data ──────────────────────────────────────────────────

@dataclass
class FIIDIIActivity:
    """FII/DII activity data from NSE."""
    date: str
    fii_buy: float
    fii_sell: float
    fii_net: float  # Positive = net buying
    dii_buy: float
    dii_sell: float
    dii_net: float  # Positive = net buying
    fii_is_buying: bool
    dii_is_buying: bool
    cached: bool = False


def fetch_fii_dii_activity(days: int = 5) -> Optional[FIIDIIActivity]:
    """
    Fetch FII/DII activity from NSE (free, no API key).
    
    FII (Foreign Institutional Investors) = "hot money"
    DII (Domestic Institutional Investors) = local institutions
    
    Returns:
        FIIDIIActivity or None
    """
    cache_k = hashlib.md5("fii_dii:activity".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return FIIDIIActivity(**cached, cached=True)

    try:
        from nselib import derivatives_market
        
        # Fetch FII/DII data
        df = derivatives_market.fii_dii_data()
        
        if df is None or df.empty:
            return None

        # Sort by date descending
        if "Date" in df.columns:
            df = df.sort_values("Date", ascending=False)
        
        latest = df.iloc[0]
        
        # Extract FII data
        fii_buy = float(latest.get("FII Buy", latest.get("FII_Buy", 0)) or 0)
        fii_sell = float(latest.get("FII Sell", latest.get("FII_Sell", 0)) or 0)
        fii_net = fii_buy - fii_sell
        
        # Extract DII data
        dii_buy = float(latest.get("DII Buy", latest.get("DII_Buy", 0)) or 0)
        dii_sell = float(latest.get("DII Sell", latest.get("DII_Sell", 0)) or 0)
        dii_net = dii_buy - dii_sell
        
        date_str = str(latest.get("Date", ""))

        result = FIIDIIActivity(
            date=date_str,
            fii_buy=fii_buy,
            fii_sell=fii_sell,
            fii_net=round(fii_net, 2),
            dii_buy=dii_buy,
            dii_sell=dii_sell,
            dii_net=round(dii_net, 2),
            fii_is_buying=fii_net > 0,
            dii_is_buying=dii_net > 0,
            cached=False,
        )

        _cache_set(cache_k, {
            "date": date_str,
            "fii_buy": result.fii_buy,
            "fii_sell": result.fii_sell,
            "fii_net": result.fii_net,
            "dii_buy": result.dii_buy,
            "dii_sell": result.dii_sell,
            "dii_net": result.dii_net,
            "fii_is_buying": result.fii_is_buying,
            "dii_is_buying": result.dii_is_buying,
        })

        return result

    except Exception as e:
        logger.debug("FII/DII data fetch failed: %s", e)
        return None


# ── 52-Week High/Low Data ──────────────────────────────────────────────────

@dataclass
class Week52Data:
    """52-week high/low data for a stock."""
    ticker: str
    current_price: float
    week52_high: float
    week52_low: float
    week52_high_date: str
    week52_low_date: str
    pct_from_52w_high: float  # Negative = below high
    pct_from_52w_low: float  # Positive = above low
    position_in_range: float  # 0 = at low, 100 = at high
    is_near_52w_high: bool  # Within 10% of high
    is_near_52w_low: bool  # Within 10% of low
    cached: bool = False


def fetch_52week_data(ticker: str) -> Optional[Week52Data]:
    """
    Fetch 52-week high/low data from Yahoo Finance (free, no API key).
    
    Args:
        ticker: Stock ticker (e.g., "RELIANCE")
    
    Returns:
        Week52Data or None
    """
    cache_k = hashlib.md5(f"52week:{ticker}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return Week52Data(**cached, cached=True)

    try:
        import yfinance as yf
        
        nse_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        stock = yf.Ticker(nse_ticker)
        info = stock.info
        
        if not info:
            return None

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        week52_high = info.get("fiftyTwoWeekHigh")
        week52_low = info.get("fiftyTwoWeekLow")
        
        if not all([current_price, week52_high, week52_low]):
            return None

        current_price = float(current_price)
        week52_high = float(week52_high)
        week52_low = float(week52_low)

        # Calculate percentages
        pct_from_high = ((current_price - week52_high) / week52_high) * 100
        pct_from_low = ((current_price - week52_low) / week52_low) * 100
        
        # Position in range (0-100)
        range_size = week52_high - week52_low
        if range_size > 0:
            position = ((current_price - week52_low) / range_size) * 100
        else:
            position = 50.0

        result = Week52Data(
            ticker=ticker,
            current_price=round(current_price, 2),
            week52_high=round(week52_high, 2),
            week52_low=round(week52_low, 2),
            week52_high_date=info.get("fiftyTwoWeekHighDate", ""),
            week52_low_date=info.get("fiftyTwoWeekLowDate", ""),
            pct_from_52w_high=round(pct_from_high, 2),
            pct_from_52w_low=round(pct_from_low, 2),
            position_in_range=round(position, 2),
            is_near_52w_high=pct_from_high > -10.0,  # Within 10% of high
            is_near_52w_low=pct_from_low < 10.0,  # Within 10% of low
            cached=False,
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "current_price": result.current_price,
            "week52_high": result.week52_high,
            "week52_low": result.week52_low,
            "week52_high_date": result.week52_high_date,
            "week52_low_date": result.week52_low_date,
            "pct_from_52w_high": result.pct_from_52w_high,
            "pct_from_52w_low": result.pct_from_52w_low,
            "position_in_range": result.position_in_range,
            "is_near_52w_high": result.is_near_52w_high,
            "is_near_52w_low": result.is_near_52w_low,
        })

        return result

    except Exception as e:
        logger.debug("52-week data fetch failed for %s: %s", ticker, e)
        return None


# ── Industry PE Data ────────────────────────────────────────────────────────

@dataclass
class IndustryPEData:
    """Industry PE ratio data from NSE."""
    ticker: str
    stock_pe: float
    industry_pe: float
    industry_name: str
    pe_relative_to_industry: float  # Stock PE / Industry PE
    is_cheap: bool  # Stock PE < Industry PE
    cached: bool = False


def fetch_industry_pe(ticker: str) -> Optional[IndustryPEData]:
    """
    Fetch industry PE comparison from NSE (free, no API key).
    
    Args:
        ticker: NSE ticker symbol
    
    Returns:
        IndustryPEData or None
    """
    cache_k = hashlib.md5(f"industry_pe:{ticker}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return IndustryPEData(**cached, cached=True)

    try:
        import yfinance as yf
        
        nse_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        stock = yf.Ticker(nse_ticker)
        info = stock.info
        
        if not info:
            return None

        stock_pe = info.get("trailingPE") or info.get("forwardPE")
        industry_pe = info.get("industryPE") or info.get("trailingPE")
        industry_name = info.get("industry", "Unknown")
        
        if not stock_pe or not industry_pe:
            return None

        stock_pe = float(stock_pe)
        industry_pe = float(industry_pe)
        
        if industry_pe == 0:
            return None

        relative_pe = stock_pe / industry_pe

        result = IndustryPEData(
            ticker=ticker,
            stock_pe=round(stock_pe, 2),
            industry_pe=round(industry_pe, 2),
            industry_name=industry_name,
            pe_relative_to_industry=round(relative_pe, 3),
            is_cheap=relative_pe < 1.0,
            cached=False,
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "stock_pe": result.stock_pe,
            "industry_pe": result.industry_pe,
            "industry_name": result.industry_name,
            "pe_relative_to_industry": result.pe_relative_to_industry,
            "is_cheap": result.is_cheap,
        })

        return result

    except Exception as e:
        logger.debug("Industry PE fetch failed for %s: %s", ticker, e)
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
    cached = _INDIA_CACHE.get(cache_k)
    if cached:
        result, ts = cached
        if time.time() - ts < _INDIA_CACHE_TTL:
            return [MandiPrice(**item) for item in result.get("prices", [])]

    try:
        # Try data.gov.in API (Indian government open data)
        # This API can be slow, use longer timeout
        url = "https://api.data.gov.in/resource/359846c8-0eae-4f53-a69b-8d5dd13057f0"
        params = {
            "format": "json",
            "limit": 20,
        }
        if commodity:
            params["filters[commodity]"] = commodity
        if state:
            params["filters[state]"] = state

        # Retry up to 2 times with increasing timeout
        for attempt in range(2):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.Timeout:
                if attempt < 1:
                    time.sleep(2)
                    continue
                raise
        else:
            return None

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
            _INDIA_CACHE[cache_k] = ({"prices": [
                {"commodity": p.commodity, "market": p.market, "state": p.state,
                 "price_min": p.price_min, "price_max": p.price_max,
                 "price_modal": p.price_modal, "unit": p.unit, "date": p.date}
                for p in prices
            ]}, time.time())

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
    cached = _INDIA_CACHE.get(cache_k)
    if cached:
        result, ts = cached
        if time.time() - ts < _INDIA_CACHE_TTL:
            return [PincodeData(**item) for item in result.get("pincodes", [])]

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
                latitude=None,
                longitude=None,
            ))

        if results:
            _INDIA_CACHE[cache_k] = ({"pincodes": [
                {"pincode": r.pincode, "office_name": r.office_name,
                 "district": r.district, "state": r.state}
                for r in results
            ]}, time.time())

        return results if results else None

    except Exception as e:
        logger.debug("Pincode data fetch failed: %s", e)
        return None


# ── Unified Indian Market Data ─────────────────────────────────────────────

def fetch_indian_market_data(ticker: str) -> dict:
    """
    Fetch all Indian market data for a ticker.
    
    Returns:
        {
            "delivery": DeliveryData | None,
            "fii_dii": FIIDIIActivity | None,
            "week52": Week52Data | None,
            "industry_pe": IndustryPEData | None,
            "source": str,
        }
    """
    delivery = fetch_delivery_data(ticker)
    fii_dii = fetch_fii_dii_activity()
    week52 = fetch_52week_data(ticker)
    industry_pe = fetch_industry_pe(ticker)

    sources = []
    if delivery:
        sources.append("delivery")
    if fii_dii:
        sources.append("fii_dii")
    if week52:
        sources.append("week52")
    if industry_pe:
        sources.append("industry_pe")

    return {
        "delivery": delivery,
        "fii_dii": fii_dii,
        "week52": week52,
        "industry_pe": industry_pe,
        "source": "+".join(sources) if sources else "none",
    }
