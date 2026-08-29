"""
Insider Data Provider
Fetches insider trading + institutional activity from Aletheia + CongressInvests + Filingrail.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Cache ───────────────────────────────────────────────────────────────────

_INSIDER_CACHE: dict[str, tuple[dict, float]] = {}
_INSIDER_CACHE_TTL = 6 * 3600  # 6 hours


def _cache_get(key: str) -> Optional[dict]:
    if key in _INSIDER_CACHE:
        result, ts = _INSIDER_CACHE[key]
        if time.time() - ts < _INSIDER_CACHE_TTL:
            return result
    return None


def _cache_set(key: str, value: dict):
    _INSIDER_CACHE[key] = (value, time.time())


# ── Aletheia Provider ──────────────────────────────────────────────────────

@dataclass
class InsiderTrade:
    """Single insider trade record."""
    insider_name: str
    title: str  # CEO, CFO, Director, etc.
    transaction_type: str  # "buy" | "sell"
    shares: int
    price: float
    value: float
    date: str
    filing_url: str = ""


@dataclass
class AletheiaInsider:
    """Insider trading data from Aletheia."""
    ticker: str
    net_insider_activity: float  # positive = net buying, negative = net selling
    total_buys: int
    total_sells: int
    buy_value: float
    sell_value: float
    recent_trades: list[InsiderTrade] = field(default_factory=list)
    insider_score: float = 0.0  # -1.0 to 1.0
    cached: bool = False


def fetch_aletheia_insider(
    ticker: str,
    api_key: Optional[str] = None,
    days: int = 90,
) -> Optional[AletheiaInsider]:
    """
    Fetch insider trading data from Aletheia.
    
    Args:
        ticker: Stock ticker
        api_key: Aletheia API key (or env ALETHEIA_API_KEY)
        days: Lookback period
    
    Returns:
        AletheiaInsider or None
    """
    api_key = api_key or os.environ.get("ALETHEIA_API_KEY")
    if not api_key:
        logger.debug("Aletheia: no API key, skipping")
        return None

    cache_k = hashlib.md5(f"aletheia:{ticker}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return AletheiaInsider(**cached, cached=True)

    symbol = ticker.replace(".NS", "").replace(".BO", "")

    try:
        url = f"https://api.aletheia.com/v1/insider-trading/{symbol}"
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"days": days, "limit": 50}
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        trades_raw = data.get("trades", data.get("data", []))
        trades = []
        buy_value = 0
        sell_value = 0
        total_buys = 0
        total_sells = 0

        for t in trades_raw:
            ttype = t.get("transaction_type", "").lower()
            shares = int(t.get("shares", t.get("quantity", 0)))
            price = float(t.get("price", t.get("share_price", 0)))
            value = shares * price

            trade = InsiderTrade(
                insider_name=t.get("insider_name", t.get("reporter_name", "Unknown")),
                title=t.get("title", t.get("officer_title", "")),
                transaction_type=ttype,
                shares=shares,
                price=price,
                value=value,
                date=t.get("filing_date", t.get("transaction_date", "")),
                filing_url=t.get("filing_url", ""),
            )
            trades.append(trade)

            if "buy" in ttype or "purchase" in ttype:
                buy_value += value
                total_buys += 1
            elif "sell" in ttype or "sale" in ttype:
                sell_value += value
                total_sells += 1

        # Compute insider score
        total_value = buy_value + sell_value
        if total_value > 0:
            insider_score = (buy_value - sell_value) / total_value
        else:
            insider_score = 0.0

        # Net activity (positive = net buying)
        net = buy_value - sell_value

        result = AletheiaInsider(
            ticker=ticker,
            net_insider_activity=round(net, 2),
            total_buys=total_buys,
            total_sells=total_sells,
            buy_value=round(buy_value, 2),
            sell_value=round(sell_value, 2),
            recent_trades=trades[:10],
            insider_score=round(max(-1.0, min(1.0, insider_score)), 3),
            cached=False,
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "net_insider_activity": result.net_insider_activity,
            "total_buys": result.total_buys,
            "total_sells": result.total_sells,
            "buy_value": result.buy_value,
            "sell_value": result.sell_value,
            "recent_trades": [
                {
                    "insider_name": t.insider_name,
                    "title": t.title,
                    "transaction_type": t.transaction_type,
                    "shares": t.shares,
                    "price": t.price,
                    "value": t.value,
                    "date": t.date,
                    "filing_url": t.filing_url,
                }
                for t in trades[:10]
            ],
            "insider_score": result.insider_score,
        })

        return result

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning("Aletheia insider fetch failed for %s: %s", ticker, e)
        return None


# ── CongressInvests Provider ───────────────────────────────────────────────

@dataclass
class CongressionalTrade:
    """Single congressional stock trade."""
    member: str
    party: str
    transaction_type: str  # "purchase" | "sale"
    asset: str
    amount_range: str  # "$1,001 - $15,000" etc.
    disclosure_date: str
    transaction_date: str
    url: str = ""


@dataclass
class CongressInvestsData:
    """Congressional trading data."""
    ticker: str
    recent_trades: list[CongressionalTrade] = field(default_factory=list)
    net_congressional_activity: float = 0.0  # positive = net buying
    buy_count: int = 0
    sell_count: int = 0
    congressional_score: float = 0.0  # -1.0 to 1.0
    cached: bool = False


def fetch_congress_invests(
    ticker: str,
    api_key: Optional[str] = None,
) -> Optional[CongressInvestsData]:
    """
    Fetch congressional stock trade data from CongressInvests.
    
    Args:
        ticker: Stock ticker
        api_key: CongressInvests API key (or env CONGRESSINVESTS_API_KEY)
    
    Returns:
        CongressInvestsData or None
    """
    api_key = api_key or os.environ.get("CONGRESSINVESTS_API_KEY")
    if not api_key:
        logger.debug("CongressInvests: no API key, skipping")
        return None

    cache_k = hashlib.md5(f"congress:{ticker}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return CongressInvestsData(**cached, cached=True)

    symbol = ticker.replace(".NS", "").replace(".BO", "")

    try:
        url = "https://api.congressinvests.com/v1/trades"
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"ticker": symbol, "limit": 20}
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        trades_raw = data.get("trades", data.get("data", []))
        trades = []
        buy_count = 0
        sell_count = 0

        for t in trades_raw:
            ttype = t.get("transaction_type", "").lower()
            trade = CongressionalTrade(
                member=t.get("member_name", t.get("representative", "")),
                party=t.get("party", ""),
                transaction_type=ttype,
                asset=t.get("asset_description", t.get("asset", "")),
                amount_range=t.get("amount_range", t.get("amount", "")),
                disclosure_date=t.get("disclosure_date", ""),
                transaction_date=t.get("transaction_date", t.get("disclosure_date", "")),
                url=t.get("url", t.get("source_url", "")),
            )
            trades.append(trade)

            if "purchase" in ttype or "buy" in ttype:
                buy_count += 1
            elif "sale" in ttype or "sell" in ttype:
                sell_count += 1

        total = buy_count + sell_count
        if total > 0:
            score = (buy_count - sell_count) / total
        else:
            score = 0.0

        result = CongressInvestsData(
            ticker=ticker,
            recent_trades=trades[:10],
            net_congressional_activity=float(buy_count - sell_count),
            buy_count=buy_count,
            sell_count=sell_count,
            congressional_score=round(max(-1.0, min(1.0, score)), 3),
            cached=False,
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "recent_trades": [
                {
                    "member": t.member,
                    "party": t.party,
                    "transaction_type": t.transaction_type,
                    "asset": t.asset,
                    "amount_range": t.amount_range,
                    "disclosure_date": t.disclosure_date,
                    "transaction_date": t.transaction_date,
                    "url": t.url,
                }
                for t in trades[:10]
            ],
            "net_congressional_activity": result.net_congressional_activity,
            "buy_count": result.buy_count,
            "sell_count": result.sell_count,
            "congressional_score": result.congressional_score,
        })

        return result

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning("CongressInvests fetch failed for %s: %s", ticker, e)
        return None


# ── SEC EDGAR Provider (Free, no key) ──────────────────────────────────────

@dataclass
class SECFiling:
    """SEC EDGAR filing record."""
    form_type: str
    filed_date: str
    description: str
    url: str


@dataclass
class SECEdgData:
    """SEC EDGAR data for a ticker."""
    ticker: str
    cik: str
    recent_filings: list[SECFiling] = field(default_factory=list)
    has_10k: bool = False
    has_10q: bool = False
    has_8k: bool = False
    insider_filings: int = 0  # Form 4 count
    filing_score: float = 0.0  # -1.0 to 1.0
    cached: bool = False


def fetch_sec_edgar(
    ticker: str,
    company_name: Optional[str] = None,
) -> Optional[SECEdgData]:
    """
    Fetch SEC EDGAR data (free, no API key).
    Only works for US-listed Indian companies (e.g., INFY, WIT, HDB).
    
    Args:
        ticker: Stock ticker
        company_name: Company name for CIK lookup (optional)
    
    Returns:
        SECEdgData or None
    """
    cache_k = hashlib.md5(f"sec_edgar:{ticker}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return SECEdgData(**cached, cached=True)

    # Strip Indian suffixes
    symbol = ticker.replace(".NS", "").replace(".BO", "")

    headers = {"User-Agent": "StockScanner/1.0 contact@example.com"}

    try:
        # Step 1: Find CIK by ticker
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&dateRange=custom&startdt=2024-01-01&enddt=2026-12-31&forms=10-K,10-Q,8-K,4"
        # Use the full-text search API
        search_url = "https://efts.sec.gov/LATEST/search-index"
        
        # Alternative: Use company tickers endpoint
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(tickers_url, headers=headers, timeout=10)
        resp.raise_for_status()
        tickers_data = resp.json()

        cik = None
        for item in tickers_data.values() if isinstance(tickers_data, dict) else tickers_data:
            if isinstance(item, dict):
                if item.get("ticker", "").upper() == symbol.upper():
                    cik = str(item.get("cik_str", "")).zfill(10)
                    break

        if not cik:
            logger.debug("SEC EDGAR: no CIK found for %s", symbol)
            return None

        # Step 2: Fetch recent filings
        filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(filings_url, headers=headers, timeout=10)
        resp.raise_for_status()
        filings_data = resp.json()

        recent = filings_data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        descriptions = recent.get("primaryDocDescription", [])
        accessions = recent.get("accessionNumber", [])

        filings = []
        has_10k = False
        has_10q = False
        has_8k = False
        insider_count = 0

        for i in range(min(len(forms), 20)):
            form = forms[i]
            date = dates[i] if i < len(dates) else ""
            desc = descriptions[i] if i < len(descriptions) else ""
            acc = accessions[i] if i < len(accessions) else ""
            
            filing = SECFiling(
                form_type=form,
                filed_date=date,
                description=desc,
                url=f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc.replace('-', '')}/{desc}.htm",
            )
            filings.append(filing)

            if form == "10-K":
                has_10k = True
            elif form == "10-Q":
                has_10q = True
            elif form == "8-K":
                has_8k = True
            elif form == "4":
                insider_count += 1

        # Score: recent 10-K/10-Q = positive, many Form 4 = neutral, 8-K = attention
        filing_score = 0.0
        if has_10k:
            filing_score += 0.3
        if has_10q:
            filing_score += 0.2
        if insider_count > 3:
            filing_score += 0.1  # Some insider activity
        if has_8k:
            filing_score += 0.1  # Material event filed

        result = SECEdgData(
            ticker=ticker,
            cik=cik,
            recent_filings=filings[:10],
            has_10k=has_10k,
            has_10q=has_10q,
            has_8k=has_8k,
            insider_filings=insider_count,
            filing_score=round(min(1.0, filing_score), 3),
            cached=False,
        )

        _cache_set(cache_k, {
            "ticker": ticker,
            "cik": cik,
            "recent_filings": [
                {
                    "form_type": f.form_type,
                    "filed_date": f.filed_date,
                    "description": f.description,
                    "url": f.url,
                }
                for f in filings[:10]
            ],
            "has_10k": has_10k,
            "has_10q": has_10q,
            "has_8k": has_8k,
            "insider_filings": insider_count,
            "filing_score": result.filing_score,
        })

        return result

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning("SEC EDGAR fetch failed for %s: %s", ticker, e)
        return None


# ── Unified Insider Fetcher ────────────────────────────────────────────────

def fetch_insider_data(
    ticker: str,
    aletheia_key: Optional[str] = None,
    congress_key: Optional[str] = None,
    company_name: Optional[str] = None,
) -> dict:
    """
    Fetch insider + institutional data from multiple sources.
    
    Returns:
        {
            "insider_score": float,  # -1.0 to 1.0
            "aletheia": AletheiaInsider | None,
            "congress": CongressInvestsData | None,
            "sec_edgar": SECEdgData | None,
            "source": str,
        }
    """
    aletheia = fetch_aletheia_insider(ticker, aletheia_key)
    congress = fetch_congress_invests(ticker, congress_key)
    sec = fetch_sec_edgar(ticker, company_name)

    # Weighted combination
    scores = []
    weights = []

    if aletheia:
        scores.append(aletheia.insider_score)
        weights.append(3.0)  # Insider data most relevant

    if congress:
        scores.append(congress.congressional_score)
        weights.append(1.0)  # Congressional trades less relevant for Indian stocks

    if sec:
        scores.append(sec.filing_score)
        weights.append(1.0)

    if not scores:
        return {
            "insider_score": 0.0,
            "aletheia": None,
            "congress": None,
            "sec_edgar": None,
            "source": "none",
        }

    weighted = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    sources = []
    if aletheia:
        sources.append("aletheia")
    if congress:
        sources.append("congress")
    if sec:
        sources.append("sec_edgar")

    return {
        "insider_score": round(weighted, 3),
        "aletheia": aletheia,
        "congress": congress,
        "sec_edgar": sec,
        "source": "+".join(sources),
    }
