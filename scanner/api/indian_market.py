"""
Indian Market Data Provider
Free data from NSE India — delivery volume, FII/DII activity, 52-week data.
All data is fetched without API keys using nselib or direct NSE API calls.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass

import pandas as pd

from ..shared.cache import TTLCache
from .yahoo_symbol import yf_quote_variants

logger = logging.getLogger(__name__)

_NSE_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# ── Cache ───────────────────────────────────────────────────────────────────

_INDIA_CACHE: TTLCache[dict] = TTLCache(ttl=4 * 3600, namespace="indian_market")


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


def _num(v) -> float | None:
    try:
        return float(str(v).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def _delivery_from_df(df, ticker: str) -> DeliveryData | None:
    """Derive delivery % from an nselib price-volume-delivery frame.

    Prefers NSE's own ``%DlyQttoTradedQty`` column; falls back to
    DeliverableQty / TotalTradedQuantity ratio. Rows are picked newest-first
    by the Date column (nselib may concatenate oldest chunk first).
    """
    if df is None or df.empty:
        return None

    pct_col = del_col = vol_col = None
    for c in df.columns:
        cl = str(c).lower()
        if pct_col is None and "dlyqtto" in cl:
            pct_col = c
        elif (
            vol_col is None
            and ("quantity" in cl or "volume" in cl or "traded" in cl)
            and "deliver" not in cl
            and "dly" not in cl
        ):
            vol_col = c
        elif del_col is None and "deliverable" in cl:
            del_col = c

    work = df
    if "Date" in df.columns:
        parsed = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
        if parsed.notna().any():
            work = df.assign(_dt=parsed).sort_values(
                "_dt", ascending=False, kind="stable"
            )

    def pct_of(r) -> float | None:
        if r is None:
            return None
        if pct_col is not None:
            p = _num(r.get(pct_col))
            if p is not None and 0 < p <= 100:
                return p
        dv = _num(r.get(del_col)) if del_col else None
        tv = _num(r.get(vol_col)) if vol_col else None
        if dv is None or not tv:
            return None
        p = dv / tv * 100
        return p if 0 < p <= 100 else None

    row = work.iloc[0]
    pct = pct_of(row)
    if pct is None:
        return None
    prev_pct = pct_of(work.iloc[1] if len(work) > 1 else None)
    dv = _num(row.get(del_col)) if del_col else None
    tv = _num(row.get(vol_col)) if vol_col else None
    return DeliveryData(
        ticker=ticker,
        delivery_pct=round(pct, 2),
        delivery_volume=int(dv or 0),
        total_volume=int(tv or 0),
        delivery_change_pct=round(pct - prev_pct, 2) if prev_pct is not None else 0.0,
        is_high_delivery=pct > 60.0,
        cached=False,
    )


def fetch_delivery_data(ticker: str, days: int = 5) -> DeliveryData | None:
    """
    Fetch delivery volume data from NSE (free, no API key).

    High delivery % indicates institutional/strong hands buying.

    Args:
        ticker: NSE ticker symbol (e.g., "RELIANCE")
        days: Lookback days for trend

    Returns:
        DeliveryData or None
    """
    cache_k = hashlib.md5(
        f"delivery:{ticker}".encode(), usedforsecurity=False
    ).hexdigest()
    cached = _INDIA_CACHE.get(cache_k)
    if cached:
        return DeliveryData(**cached, cached=True)

    try:
        from datetime import date, timedelta

        from nselib import capital_market

        end = date.today()
        start = end - timedelta(days=days + 5)  # Extra days for buffer

        df = capital_market.price_volume_and_deliverable_position_data(
            symbol=ticker,
            from_date=start.strftime("%d-%m-%Y"),
            to_date=end.strftime("%d-%m-%Y"),
        )
        result = _delivery_from_df(df, ticker)
        if result is None:
            logger.info("No usable delivery data for %s", ticker)
            return None

        _INDIA_CACHE.set(
            cache_k,
            {
                "ticker": result.ticker,
                "delivery_pct": result.delivery_pct,
                "delivery_volume": result.delivery_volume,
                "total_volume": result.total_volume,
                "delivery_change_pct": result.delivery_change_pct,
                "is_high_delivery": result.is_high_delivery,
            },
        )

        return result

    except Exception as e:
        logger.info("Delivery data fetch failed for %s: %s", ticker, e)
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


def fetch_fii_dii_activity(days: int = 5) -> FIIDIIActivity | None:
    """
    Fetch FII/DII activity from NSE (free, no API key).

    FII (Foreign Institutional Investors) = "hot money"
    DII (Domestic Institutional Investors) = local institutions

    Returns:
        FIIDIIActivity or None
    """
    cache_k = hashlib.md5(b"fii_dii:activity", usedforsecurity=False).hexdigest()
    cached = _INDIA_CACHE.get(cache_k)
    if cached:
        return FIIDIIActivity(**cached, cached=True)

    try:
        # nselib 2.5.x dropped the old derivatives_market.fii_dii_data() —
        # go straight to the NSE endpoint (returns the latest day only).
        import requests

        resp = requests.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers=_NSE_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return None
        by_cat = {
            str(r.get("category", "")).upper().strip(): r
            for r in data
            if isinstance(r, dict)
        }
        # NSE labels the row "FII/FPI" these days; older payloads used "FII".
        fii_rec = next(
            (r for k, r in by_cat.items() if k.startswith("FII") or k == "FPI"),
            None,
        )
        dii_rec = by_cat.get("DII")
        if not fii_rec or not dii_rec:
            return None

        def _val(rec: dict, key: str) -> float:
            try:
                return float(rec.get(key) or 0)
            except (TypeError, ValueError):
                return 0.0

        fii_buy = _val(fii_rec, "buyValue")
        fii_sell = _val(fii_rec, "sellValue")
        dii_buy = _val(dii_rec, "buyValue")
        dii_sell = _val(dii_rec, "sellValue")
        fii_net = round(fii_buy - fii_sell, 2)
        dii_net = round(dii_buy - dii_sell, 2)
        date_str = str(fii_rec.get("date", ""))

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

        _INDIA_CACHE.set(
            cache_k,
            {
                "date": date_str,
                "fii_buy": result.fii_buy,
                "fii_sell": result.fii_sell,
                "fii_net": result.fii_net,
                "dii_buy": result.dii_buy,
                "dii_sell": result.dii_sell,
                "dii_net": result.dii_net,
                "fii_is_buying": result.fii_is_buying,
                "dii_is_buying": result.dii_is_buying,
            },
        )

        return result

    except Exception as e:
        logger.info("FII/DII data fetch failed: %s", e)
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


def fetch_52week_data(ticker: str) -> Week52Data | None:
    """
    Fetch 52-week high/low data from Yahoo Finance (free, no API key).

    Args:
        ticker: Stock ticker (e.g., "RELIANCE")

    Returns:
        Week52Data or None
    """
    cache_k = hashlib.md5(
        f"52week:{ticker}".encode(), usedforsecurity=False
    ).hexdigest()
    cached = _INDIA_CACHE.get(cache_k)
    if cached:
        return Week52Data(**cached, cached=True)

    try:
        import yfinance as yf

        info = {}
        for yf_ticker in yf_quote_variants(ticker):
            info = yf.Ticker(yf_ticker).info or {}
            if info:
                break

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

        _INDIA_CACHE.set(
            cache_k,
            {
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
            },
        )

        return result

    except Exception as e:
        logger.info("52-week data fetch failed for %s: %s", ticker, e)
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
            "source": str,
        }
    """
    delivery = fetch_delivery_data(ticker)
    fii_dii = fetch_fii_dii_activity()
    week52 = fetch_52week_data(ticker)

    sources = []
    if delivery:
        sources.append("delivery")
    if fii_dii:
        sources.append("fii_dii")
    if week52:
        sources.append("week52")

    return {
        "delivery": delivery,
        "fii_dii": fii_dii,
        "week52": week52,
        "source": "+".join(sources) if sources else "none",
    }


# ponytail: near-zero nets make % explode — floor the denominator
_PCT_CHANGE_FLOOR = 10_000


def _pct_change(now: float, then: float, floor: float = _PCT_CHANGE_FLOOR) -> float:
    """Signed % change of a net position; |baseline| floored against noise."""
    base = abs(then) if abs(then) >= floor else floor
    return round((now - then) / base * 100, 2)


# ── Cash-flow history (moneycontrol) ─────────────────
# NSE's fiidiiTradeReact endpoint only returns today. Moneycontrol's
# __NEXT_DATA__ payload carries the same fiiCM/diiCM cash numbers for the
# last 60 sessions in one GET (no form).

_FLOW_CACHE: TTLCache[list] = TTLCache(ttl=12 * 3600, namespace="mc_fiidii")
_FLOW_URL = "https://www.moneycontrol.com/markets/fii-dii-data/"


def fetch_fii_dii_history(limit: int = 60, cache_only: bool = False) -> list | None:
    """
    Daily FII/DII cash-market net flows (₹ Cr), newest first.

    Returns [{"date": "2026-09-25", "fii": -3693.93, "dii": 2838.17}, ...]
    or None when the page/parse fails. ``cache_only=True`` skips the network.
    """
    cache_k = hashlib.md5(b"fii_dii:history", usedforsecurity=False).hexdigest()
    cached = _FLOW_CACHE.get(cache_k)
    if cached:
        return cached
    if cache_only:
        return None

    try:
        import requests

        resp = requests.get(
            _FLOW_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15
        )
        resp.raise_for_status()
        m = re.search(
            r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text, re.DOTALL
        )
        if not m:
            return None
        rows = (
            json.loads(m.group(1))
            .get("props", {})
            .get("pageProps", {})
            .get("FiiDiiData", {})
            .get("fiiDiiData")
        )
        if not isinstance(rows, list):
            return None

        out = [
            {
                "date": str(r.get("date", "")),
                "fii": _num(r.get("fiiCM")) or 0.0,
                "dii": _num(r.get("diiCM")) or 0.0,
            }
            for r in rows
            if isinstance(r, dict)
        ]
        if len(out) < 2:
            return None
        out = out[:limit]
        _FLOW_CACHE.set(cache_k, out)
        return out
    except Exception as e:
        logger.info("Moneycontrol FII/DII history fetch failed: %s", e)
        return None


def flow_window_pcts(
    history: list, key: str, total_n: int = 20, recent_n: int = 5, floor: float = 100.0
) -> tuple[float | None, float | None]:
    """(total, recent) % change of window *sums* over newest-first history.

    total = last total_n vs prior total_n sessions, recent likewise.
    Returns (None, None) until enough history exists; floor keeps the
    denominator sane when a window sum hovers near zero. The total window
    shrinks to half the history when fewer than 2×total_n rows are served
    (moneycontrol hands out ~30 sessions).
    """
    if not history:
        return None, None
    # min() guarantees 2*total_n <= len(history) — the window always exists.
    total_n = min(total_n, len(history) // 2)
    now = sum(r[key] for r in history[:total_n])
    then = sum(r[key] for r in history[total_n : 2 * total_n])
    total = _pct_change(now, then, floor=floor)
    if len(history) < 2 * recent_n:
        recent = None
    else:
        now = sum(r[key] for r in history[:recent_n])
        then = sum(r[key] for r in history[recent_n : 2 * recent_n])
        recent = _pct_change(now, then, floor=floor)
    return total, recent
