"""
HTML report generator for stock scanner results.
Produces a sortable, filterable table with color-coded scores and news sentiment.
"""

import html as _html
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from ..shared.cache import TTLCache
from ..shared.constants import has_fii_dii, score_of
from ..shared.detail_specs import SCORE_CATS, fmt_pct, ma_chip, signal_specs

logger = logging.getLogger(__name__)

# Reports regenerate on demand — keep each ticker's yfinance news for 4h so
# repeated report/detail renders don't re-hit Yahoo per ticker.
_NEWS_CACHE: TTLCache = TTLCache(ttl=4 * 3600, namespace="stock_news")


def _sparkline_svg(closes: list, width: int = 100, height: int = 22) -> str:
    """Smooth cubic-bezier sparkline SVG from a list of closing prices."""
    px = [v for v in closes if v is not None and isinstance(v, (int, float))]
    if len(px) < 2:
        return ""
    up = px[-1] >= px[0]
    stroke = "#10b981" if up else "#f87171"
    fill = "rgba(52,211,153,0.15)" if up else "rgba(248,113,113,0.15)"
    lo, hi = min(px), max(px)
    span = (hi - lo) or 1.0
    n = len(px)
    # Large detail chart: app-style cyan line, left gutter with y-axis labels.
    detail = width > 200
    pad_l = 46 if detail else 0
    if detail:
        stroke, fill = "#14b8a6", "rgba(20,184,166,0.15)"
    coords = []
    for i, v in enumerate(px):
        x = pad_l + i / max(n - 1, 1) * (width - pad_l)
        y = height - 2 - (v - lo) / span * (height - 4)
        coords.append((x, y))
    # Build smooth cubic bezier path
    d = f"M{coords[0][0]:.1f},{coords[0][1]:.1f}"
    for i in range(1, len(coords)):
        x0, y0 = coords[i - 1]
        x1, y1 = coords[i]
        mx = (x0 + x1) / 2
        d += f"C{mx:.1f},{y0:.1f} {mx:.1f},{y1:.1f} {x1:.1f},{y1:.1f}"
    fill_d = d + f"L{width},{height}L{pad_l},{height}Z"
    extras = ""
    if detail:
        # Faint gridlines + y-axis ticks at 5 values (hi → lo), app parity.
        grid = ""
        for i in range(5):
            f = i / 4
            y = 2 + f * (height - 4)
            grid += (
                f'<line class="grid" x1="{pad_l}" y1="{y:.0f}" x2="{width}" '
                f'y2="{y:.0f}" stroke="rgba(90,122,106,0.16)" stroke-width="1"/>'
                f'<text class="axis-lbl" x="{pad_l - 6}" y="{y:.0f}" '
                f'text-anchor="end" dominant-baseline="middle">'
                f"{hi - f * span:,.0f}</text>"
            )
        ex = min(coords[-1][0], width - 3)
        dot = (
            f'<circle cx="{ex:.1f}" cy="{coords[-1][1]:.1f}" r="3.5" fill="{stroke}" '
            f'style="filter:drop-shadow(0 0 4px {stroke})"/>'
        )
        extras = grid + dot
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block">'
        f'<path d="{fill_d}" fill="{fill}" />'
        f"{extras}"
        f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round" />'
        f"</svg>"
    )


# ─── Sentiment keywords ──────────────────────────────────────────────────────
SENTIMENT_GOOD = frozenset(
    [
        "profit",
        "growth",
        "record",
        "gain",
        "surge",
        "rally",
        "strong",
        "beat",
        "bullish",
        "outperform",
        "order",
        "deal",
        "buy",
        "upgrade",
        "partner",
        "expand",
        "launch",
        "innovate",
        "dividend",
        "revenue",
        "acquire",
        "breakout",
        "resilient",
        "optimistic",
        "recovery",
        "momentum",
        "approval",
        "milestone",
        "boom",
        "soar",
        "jump",
        "climb",
    ]
)

SENTIMENT_BAD = frozenset(
    [
        "loss",
        "decline",
        "crash",
        "drop",
        "fall",
        "weak",
        "bearish",
        "underperform",
        "sell",
        "downgrade",
        "fraud",
        "lawsuit",
        "investigation",
        "debt",
        "recession",
        "warning",
        "cut",
        "slump",
        "miss",
        "risk",
        "concern",
        "delay",
        "ban",
        "penalty",
        "probe",
        "resign",
        "volatile",
        "crisis",
        "shortage",
        "slowdown",
        "shrink",
        "tumble",
    ]
)


def _sentiment(title: str, summary: str = "") -> str:
    """Simple keyword-based sentiment: Good / Bad / Neutral."""
    text = (title + " " + summary).lower()
    # Split on whitespace and hyphens to catch hyphenated words
    words = set(re.split(r"[\s\-]+", text))
    g = len(words & SENTIMENT_GOOD)
    b = len(words & SENTIMENT_BAD)
    if g > b:
        return "Good"
    if b > g:
        return "Bad"
    return "Neutral"


def _parse_date(date_str: str) -> datetime | None:
    """Parse ISO date string to datetime, return None on failure."""
    if not date_str:
        return None
    clean = date_str.rstrip("Z").strip()[:19]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            continue
    return None


def _news_id(ticker: str) -> str:
    """DOM/JS-safe element id fragment for a ticker.

    Tickers are interpolated into ``id="news-..."`` and an inline
    ``onclick="toggleNews('...')"`` handler; HTML-escaping alone does not
    make a string safe for the JS single-quoted context, so restrict to
    ``[A-Za-z0-9_]`` (NSE/BSE symbols are alphanumeric).
    """
    return re.sub(r"[^A-Za-z0-9_]", "_", str(ticker or "UNKNOWN"))


def fetch_stock_news(ticker: str, max_items: int = 10, months_back: int = 2) -> list:
    """
    Fetch recent news for a stock from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g. 'RELIANCE.NS')
        max_items: Maximum news items to return
        months_back: Only include news from the last N months

    Returns:
        List of dicts with 'title', 'summary', 'date', 'publisher',
        'url', 'sentiment'
    """
    cache_k = _NEWS_CACHE.make_key(ticker, str(max_items), str(months_back))
    cached = _NEWS_CACHE.get(cache_k)
    if cached is not None:
        return cached
    try:
        import yfinance as yf

        # Auto-append .NS suffix for Indian stocks if not present
        yf_ticker = (
            ticker
            if any(ticker.endswith(s) for s in (".NS", ".BO", ".NSE", ".BSE"))
            else f"{ticker}.NS"
        )
        t = yf.Ticker(yf_ticker)
        news_items = t.news or []
        cutoff = datetime.now() - timedelta(days=months_back * 30)
        results = []
        for item in news_items:
            content = item.get("content", {})
            title = content.get("title", "")
            summary = content.get("summary", "")
            pub_date = content.get("pubDate", "")
            provider = content.get("provider", {}).get("displayName", "")
            dt = _parse_date(pub_date)
            if dt and dt < cutoff:
                continue
            results.append(
                {
                    "title": title,
                    "summary": summary,
                    "date": dt.strftime("%Y-%m-%d") if dt else "—",
                    "publisher": provider,
                    "url": (
                        (content.get("canonicalUrl") or {}).get("url")
                        or (content.get("clickThroughUrl") or {}).get("url")
                        or item.get("link")
                        or ""
                    ),
                    "sentiment": _sentiment(title, summary),
                }
            )
            if len(results) >= max_items:
                break
        _NEWS_CACHE.set(cache_k, results)  # failures skip the try → uncached
        return results
    except Exception as e:
        logger.info("News fetch failed for %s: %s", ticker, e)
        return []


def _fetch_news_parallel(
    tickers: list[str],
    max_items: int = 10,
    months_back: int = 2,
    max_workers: int = 8,
    fetch_fn=None,
) -> dict[str, list]:
    """
    Fetch news for multiple tickers in parallel.

    Args:
        tickers: Tickers to fetch news for.
        max_items / months_back: Passed through to the per-ticker fetcher.
        max_workers: Parallelism cap.
        fetch_fn: Optional per-ticker callable ``(ticker, max_items,
            months_back) -> list``; defaults to :func:`fetch_stock_news`.

    Returns:
        Dict mapping ticker -> list of news item dicts (``[]`` on failure).
    """
    fetch_fn = fetch_fn or fetch_stock_news
    news_map: dict[str, list] = {}
    if not tickers:
        return news_map

    def _fetch_one(ticker: str) -> tuple[str, list]:
        return ticker, fetch_fn(ticker, max_items, months_back)

    workers = min(max_workers, len(tickers))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_one, t): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    t, items = future.result()
                    news_map[t] = items
                except Exception as e:
                    logger.info("Parallel news fetch failed for %s: %s", ticker, e)
                    news_map[ticker] = []
    except Exception as e:
        logger.info("ThreadPoolExecutor failed: %s", e)
        # Fallback: sequential fetch (never let one ticker abort the rest)
        for t in tickers:
            try:
                news_map[t] = fetch_fn(t, max_items, months_back)
            except Exception as e2:
                logger.info("Sequential news fetch failed for %s: %s", t, e2)
                news_map[t] = []

    return news_map


def fetch_news_for_ticker(
    ticker: str, max_items: int = 10, months_back: int = 2
) -> list:
    """News for one ticker in the results-panel shape, NSE then BSE fallback.

    Tries the ``.NS`` suffix first (Indian primary listing); when that returns
    no stories it retries ``.BO`` (BSE-only names). Items carry the
    ``provider`` key the results panel reads (a copy of ``fetch_stock_news``
    's ``publisher``). Never raises — returns ``[]`` on failure.
    """
    if any(ticker.upper().endswith(s) for s in (".NS", ".BO", ".NSE", ".BSE")):
        candidates = [ticker]
    else:
        candidates = [f"{ticker}.NS", f"{ticker}.BO"]
    for cand in candidates:
        items = fetch_stock_news(cand, max_items, months_back)
        if not items:
            continue
        for item in items:
            # The GUI news panel reads ``provider`` (fetch_stock_news emits
            # ``publisher`` for the HTML report template).
            item["provider"] = item.pop("publisher", "")
        return items
    return []


def fetch_news_batch(
    tickers: list[str], max_items: int = 10, months_back: int = 2, max_workers: int = 6
) -> dict[str, list]:
    """Parallel batch version of :func:`fetch_news_for_ticker`.

    Returns a dict mapping each input ticker to its provider-keyed news list
    (``[]`` when nothing was found or the fetch failed).
    """
    return _fetch_news_parallel(
        tickers,
        max_items=max_items,
        months_back=months_back,
        max_workers=max_workers,
        fetch_fn=fetch_news_for_ticker,
    )


def _css_block() -> str:
    """CSS rules for the report <style> block (plain string, no f-string)."""
    css = """    :root {
        /* Emerald (GUI) dark-theme palette — keeps the exported report visually
           identical to the app: same surfaces, borders and accent colors. */
        color-scheme: dark;
        --bg: #0a0f0c; --surface: #111a16; --surface2: #152019; --surface3: #1a2a22;
        --border: #1f3328; --border-light: #2a4a3a; --text: #e8f5ee; --text-dim: #8ca89a; --text-faint: #5a7a6a;
        --green: #10b981; --lime: #a3e635; --orange: #fb923c; --red: #f87171;
        --blue: #2dd4bf; --cyan: #14b8a6; --yellow: #facc15;
        /* Row zebra / component tints for the dark-only report */
        --focus-ring: rgba(52,211,153,0.15); --row-alt: #0d1512;
        --ticker-hover-bg: rgba(52,211,153,0.12);
        --ticker-hover-c: #ffffff; --track: rgba(255,255,255,0.06); --row-line: rgba(31,51,40,0.6);
        --news-line: rgba(31,51,40,0.5);
        --radius: 12px; --radius-sm: 8px;
        /* Local-first stacks: no CDN font request, falls back to the OS UI font
           (same system-ui the app renders with) and system mono for numbers. */
        --mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    ::selection { background: rgba(16,185,129,0.30); color: #ffffff; }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 99px; border: 2px solid var(--bg); }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
    body { font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; background: radial-gradient(1200px 600px at 0% -10%, #0d3328 0%, var(--bg) 55%), var(--bg); color: var(--text); padding: 24px; line-height: 1.5; min-height: 100vh;
           -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }

    /* ─── Hero banner: slim tinted card — title + status left, meta
           pills right (one band, no dead space), market box beside it.
           Soft top-right sheen + staggered rise-in. ── */
    .hero { display: flex; align-items: center; gap: 12px; margin-bottom: 10px;
            animation: rise 0.45s cubic-bezier(0.2, 0.7, 0.3, 1) both; }
    .hero-pill { flex: 1; min-width: 0; position: relative; overflow: hidden;
                 background: linear-gradient(100deg, rgba(16,185,129,0.20), rgba(13,148,136,0.07) 45%, rgba(16,185,129,0) 75%), var(--surface2);
                 border: 1px solid var(--border); border-radius: 14px; padding: 11px 16px;
                 box-shadow: 0 4px 16px rgba(0,0,0,0.28); }
    .hero-pill::after { content: ''; position: absolute; inset: 0; pointer-events: none;
                        background: radial-gradient(120% 220% at 100% -40%, rgba(255,255,255,0.10), transparent 55%); }
    .hero-row { display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap; }
    .hero-text { min-width: 0; }
    .hero-title { font-size: 17px; font-weight: 700; color: #ffffff; letter-spacing: -0.01em; line-height: 1.3; }
    .hero-sub { color: rgba(209,250,229,0.85); font-size: 11.5px; margin-top: 3px; }
    .meta { display: flex; gap: 6px; flex-wrap: wrap; }
    .meta span { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.13); border-radius: 99px;
                 padding: 4px 11px; font-size: 10px; font-weight: 600; color: #d1fae5; white-space: nowrap; }
    .market-box { flex: none; width: 175px; text-align: right; background: var(--surface3);
                  border: 1px solid var(--border-light); border-radius: 12px; padding: 6px 14px;
                  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05); }
    .market-label { font-size: 9px; font-weight: 700; color: rgba(209,250,229,0.65);
                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .market-value { font-size: 19px; font-weight: 700; color: #ffffff; line-height: 1.3; font-variant-numeric: tabular-nums; }
    .market-change { font-size: 9.5px; color: rgba(209,250,229,0.8); line-height: 1.5; }

    /* ─── Overview: 4 compact stat cards in one strip ── */
    .overview { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; align-items: stretch; margin-bottom: 16px;
                animation: rise 0.45s cubic-bezier(0.2, 0.7, 0.3, 1) 0.06s both; }
    .summary { display: contents; }
    .stat { display: flex; align-items: center; gap: 10px; background: linear-gradient(180deg, var(--surface) 0%, var(--surface2) 100%); border: 1px solid var(--border); border-radius: var(--radius); padding: 9px 12px 9px 14px; position: relative; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.22);
            transition: transform 0.2s ease, box-shadow 0.2s ease; }
    .stat:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0,0,0,0.3); }
    .stat::before { content: ''; position: absolute; top: 0; bottom: 0; left: 0; width: 3px; background: var(--accent, var(--green)); opacity: 0.9; border-radius: 0 2px 2px 0; }
    .stat.green { --accent: var(--green); } .stat.lime { --accent: var(--lime); } .stat.cyan { --accent: var(--cyan); } .stat.orange { --accent: var(--orange); } .stat.red { --accent: var(--red); }
    .stat .num { flex: none; font-family: var(--mono); font-size: 1.3em; font-weight: 700; line-height: 1; text-align: left; color: var(--accent, var(--text));
                 font-variant-numeric: tabular-nums; }
    .stat-meta { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
    .stat .label { color: var(--text-dim); font-size: 0.6em; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; }
    .stat .sub { color: var(--text-faint); font-size: 0.6em; }
    .filters { margin-bottom: 16px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
               background: rgba(255,255,255,0.03); border: 1px solid var(--border); padding: 10px 14px; border-radius: var(--radius);
               backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
               animation: rise 0.45s cubic-bezier(0.2, 0.7, 0.3, 1) 0.12s both; }
    .filters input[type="text"] {
        background: var(--bg) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%238ca89a' stroke-width='2.2' stroke-linecap='round'><path d='M11 11m-7.5 0a7.5 7.5 0 1 0 15 0 7.5 7.5 0 1 0-15 0M16.6 16.6L21 21'/></svg>") no-repeat 11px center;
        border: 1px solid var(--border); color: var(--text); width: 220px;
        padding: 7px 14px 7px 33px; border-radius: 99px; font-family: inherit; font-size: 0.82em;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .filters input[type="text"]::placeholder { color: var(--text-faint); }
    .filters input[type="text"]:focus { outline: none; border-color: var(--green); box-shadow: 0 0 0 3px var(--focus-ring); }
    .chip-group { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
    .chip-label { font-size: 0.62em; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-faint); margin-right: 2px; }
    .chip { background: var(--bg); border: 1px solid var(--border); color: var(--text-dim); cursor: pointer;
            padding: 5px 12px; border-radius: 99px; font-family: inherit; font-size: 0.74em; font-weight: 600;
            transition: color 0.15s, border-color 0.15s, background 0.15s, transform 0.1s; }
    .chip:hover { color: var(--text); border-color: var(--border-light); transform: translateY(-1px); }
    .chip:active { transform: scale(0.96); }
    .chip.active { background: rgba(52,211,153,0.14); border-color: rgba(52,211,153,0.45); color: var(--green); }
    input:focus-visible, .chip:focus-visible, a:focus-visible {
        outline: 2px solid var(--green); outline-offset: 2px; }
    /* App section header (views_layout.section_header): title, filter
       checkboxes, active chips, scanned/above count on the right */
    .results-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 10px 8px 6px;
                    animation: rise 0.45s cubic-bezier(0.2, 0.7, 0.3, 1) 0.16s both; }
    .results-title { font-size: 16px; font-weight: 700; color: var(--text); letter-spacing: -0.01em; }
    .cb { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text-dim); cursor: pointer; }
    .cb:has(input:checked) { color: var(--text); }
    .cb input { width: 14px; height: 14px; margin: 0; accent-color: var(--green); cursor: pointer; }
    .filter-chips { display: flex; gap: 6px; flex-wrap: wrap; }
    .filter-chip { background: var(--surface3); border: 1px solid var(--border); border-radius: 6px;
                   padding: 2px 6px; font-family: inherit; font-size: 10px; font-weight: 600; color: var(--text-dim);
                   cursor: pointer; transition: color 0.15s, border-color 0.15s, background 0.15s; }
    .filter-chip:hover { color: var(--text); border-color: var(--border-light); background: var(--bg); }
    .count-label { margin-left: auto; font-size: 11px; color: var(--text-dim); white-space: nowrap; }
    .table-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                  animation: fadeIn 0.5s ease 0.2s both; }
    table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 11px; min-width: 1250px;
            font-variant-numeric: tabular-nums; }
    /* Header — app `_make_header_row`: card2 bar, 10px bold, cyan@55 until sorted */
    th { background: linear-gradient(180deg, #1e3028, var(--surface3)); color: rgba(20,184,166,0.55); padding: 0 10px; text-align: left; font-weight: 700;
         font-size: 10px; letter-spacing: normal; border-bottom: 1px solid var(--border); cursor: pointer; user-select: none;
         position: sticky; top: 0; z-index: 2; white-space: nowrap; height: 34px; transition: color 0.15s, background 0.15s; }
    th::after { content: " ▲▼"; }
    /* Frozen identity columns: header + rank/ticker stay put on h-scroll */
    th:first-child { left: 0; border-top-left-radius: var(--radius); }
    th:last-child { border-top-right-radius: var(--radius); }
    th:hover { color: var(--cyan); }
    th.sorted-asc, th.sorted-desc { color: var(--green); }
    th.sorted-asc::after { content: " ▲"; }
    th.sorted-desc::after { content: " ▼"; }
    /* Rows — app `_create_row_controls`: 34px lines, rating rail, odd-rank zebra,
       card bg when the score clears the threshold, card2 on hover */
    td { padding: 0 10px; border-bottom: 1px solid var(--row-line); vertical-align: middle; height: 34px; }
    tbody tr:not(.news-row) td { white-space: nowrap; }
    tbody tr { transition: background 0.15s; }
    tbody tr:not(.news-row) { background: var(--bg); }
    tbody tr:not(.news-row).row-alt { background: var(--row-alt); }
    tbody tr.highlight:not(.news-row) { background: var(--surface2); }
    tbody tr.highlight:not(.news-row) td { border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
    tbody tr:not(.news-row):hover { background: var(--surface3); }
    tbody tr:last-child td:first-child { border-bottom-left-radius: var(--radius); }
    tbody tr:last-child td:last-child { border-bottom-right-radius: var(--radius); }
    /* Sticky first two columns — 19 columns need ~1750px, so the identity
       cells (rank rail + ticker) must not scroll away from the row. */
    td.rank { position: sticky; left: 0; z-index: 1; background: inherit; width: 52px; color: var(--text-faint); padding-left: 12px; }
    td.rank::before { content: ''; position: absolute; left: 0; top: 8px; bottom: 8px; width: 3px; border-radius: 2px;
                      background: var(--rail, transparent); }
    td.ticker { position: sticky; left: 52px; z-index: 1; background: inherit; }
    .ticker { color: var(--text); font-weight: 700; cursor: pointer; font-size: 12px; }
    .ticker.above { color: var(--green); }
    .ticker:hover { color: var(--ticker-hover-c); text-decoration: none; background: var(--ticker-hover-bg); border-radius: 4px; }
    .copy-t { background: none; border: 0; padding: 0 0 0 4px; color: var(--text-faint); font-size: 10px; cursor: pointer; }
    .copy-t:hover { color: var(--green); }
    .score { font-size: 14px; font-weight: 700; color: var(--red); }
    .score.s-excellent { color: var(--green); }
    .score.s-good { color: var(--lime); }
    .score.s-moderate { color: var(--orange); }
    .rating { font-weight: 700; color: var(--red); border-radius: 6px; }
    .rating.r-excellent { color: var(--green); background: rgba(16,185,129,0.12); }
    .rating.r-good { color: var(--lime); background: rgba(163,230,53,0.09); }
    .rating.r-moderate { color: var(--orange); background: rgba(251,146,60,0.12); }
    .rating.r-poor { color: var(--red); background: rgba(248,113,113,0.12); }
    .entry { border-radius: 6px; color: var(--text-dim); }
    .entry.yes { color: var(--green); font-weight: 700; background: rgba(16,185,129,0.14); }
    .price { font-size: 12px; font-weight: 700; }
    .bull { color: var(--green); }
    .bear { color: var(--red); }
    .sc-orange { color: var(--orange); }
    .sc-lime { color: var(--lime); }
    .sc-fund { color: #ffe600; }
    .rval-ok { color: var(--green); }
    .rval-off { color: var(--orange); }
    .rval-none { color: var(--text-faint); }
    .sideways { color: var(--orange); }
    .trending { color: var(--green); }
    .ma-cross { color: var(--green); }
    .ma-bull { color: var(--lime); }
    .ma-bear { color: var(--red); }
    .poc-above { color: var(--green); }
    .poc-below { color: var(--red); }
    .bothma-yes { color: var(--green); }
    .bothma-no { color: var(--text-dim); }
    .footer { margin-top: 22px; padding-top: 14px; border-top: 1px solid var(--border); color: var(--text-faint);
              font-size: 0.72em; letter-spacing: 0.03em; text-align: center; }

    /* ─── News panel styles ─────────────────────────────── */
    .news-row td { padding: 0; border-bottom: 1px solid var(--border); }
    .news-panel {
        background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
        border-left: 3px solid var(--cyan);
        padding: 10px 16px;
        margin: 4px 12px 8px 40px;
        border-radius: var(--radius-sm);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }
    .news-summary-line {
        color: var(--text);
        font-size: 0.85em;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border);
    }
    .news-item {
        padding: 6px 0;
        border-bottom: 1px solid var(--news-line);
    }
    .news-item:last-child { border-bottom: none; }
    .news-sentiment {
        font-weight: bold;
        font-size: 0.8em;
        margin-right: 6px;
    }
    .news-sentiment.good { color: var(--green); }
    .news-sentiment.bad { color: var(--red); }
    .news-sentiment.neutral { color: var(--text-dim); }
    .news-date { color: var(--text-dim); font-size: 0.78em; margin-right: 8px; }
    .news-pub { color: var(--cyan); font-size: 0.78em; }
    .news-title { color: var(--text); font-size: 0.85em; margin-top: 3px; font-weight: bold; }
    .news-summary { color: var(--text-dim); font-size: 0.78em; margin-top: 2px; }
    .news-good { color: var(--green); font-weight: bold; }
    .news-bad { color: var(--red); font-weight: bold; }
    .news-neutral { color: var(--text-dim); font-weight: bold; }

    /* ─── Trade reasons ────────────────────────────────────── */
    .reason-item {
        color: var(--text);
        font-size: 0.82em;
        padding: 3px 0;
        line-height: 1.4;
    }
    .reason-icon {
        font-weight: bold;
        margin-right: 4px;
    }

    /* ─── Stock detail (mirrors the app's detail panel) ─── */
    .detail-grid { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 16px; margin-bottom: 12px; }
    .detail-col { display: flex; flex-direction: column; gap: 12px; }
    .detail-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 12px 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.25); }
    .detail-title { color: var(--cyan); font-weight: 700; font-size: 0.85em; margin: 0 0 8px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
    .axis-lbl { font-size: 10px; fill: var(--text-faint); font-family: var(--mono); }
    .chart-box { position: relative; background: var(--surface2); border-radius: 6px; padding: 8px; }
    .chart-hi, .chart-lo { position: absolute; right: 10px; font-size: 0.7em; color: var(--text-dim); font-family: var(--mono); }
    .chart-hi { top: 6px; }
    .chart-lo { bottom: 6px; }
    .signals { display: flex; flex-wrap: wrap; gap: 6px; }
    .signal { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; min-width: 64px; box-shadow: 0 3px 12px rgba(0,0,0,0.22); }
    .signal-label { display: block; font-size: 0.65em; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
    .signal-val { font-weight: 700; font-size: 0.95em; color: var(--text); }
    .bd-row { display: grid; grid-template-columns: 86px minmax(0,1fr) 52px; gap: 8px; align-items: center; font-size: 0.78em; color: var(--text-dim); margin-bottom: 5px; }
    .bd-track { height: 8px; background: var(--track); border-radius: 99px; overflow: hidden; }
    .bd-fill { display: block; height: 100%; border-radius: 99px; }
    .bd-val { text-align: right; color: var(--text); font-family: var(--mono); }
    .inst-tiles { display: flex; flex-wrap: wrap; gap: 6px; }
    .inst-tile { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; box-shadow: 0 3px 12px rgba(0,0,0,0.22); }
    .inst-label { display: block; font-size: 0.65em; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
    .inst-val { font-weight: 700; color: var(--text); }
    .inst-sub { display: block; font-size: 0.7em; }
    .inst-hint { font-size: 0.7em; color: var(--text-faint); margin-top: 4px; }
    .news-more { display: inline-block; color: var(--cyan); font-size: 0.8em; margin-top: 6px; text-decoration: none; }
    .news-more:hover { text-decoration: underline; }

    /* ─── Sparkline column ─────────────────────────────────── */
    .spark-cell { padding: 4px 6px; }

    /* ─── Content-fit: JS hides FIT_ORDER columns until the table
           fits its container (viewport tiers alone can't — 23
           columns need ~2200px at full width). ───────────────── */
    .fit-hidden { display: none; }

    /* ─── Entrance + expand motion (staggered rise for sections),
           reduced-motion guard below switches everything off ──── */
    @keyframes fadeUp { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: none; } }
    @keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .news-row.open .news-panel { animation: fadeUp 0.25s ease both; }
    @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

    /* ─── Responsive column tiers — mirrors the GUI width tiers
           (ui_kit: wide ≥1600, standard ≥1400, compact below).
           .c-t1 drops first (diagnostics), .c-t2 adds on narrow
           (score components); sort/filter JS indexes are unaffected. */
    @media (max-width: 1599px) { th.c-t1, td.c-t1 { display: none; } }
    @media (max-width: 1399px) {
        th.c-t2, td.c-t2 { display: none; }
        table { min-width: 940px; }
    }
    @media (max-width: 1100px) {
        .detail-grid { grid-template-columns: minmax(0,1fr); }
    }
    @media (max-width: 860px) {
        .overview { grid-template-columns: repeat(2, minmax(0,1fr)); }
    }
    @media (max-width: 720px) {
        body { padding: 12px; }
        .filters input[type="text"] { width: 100%; }
        .hero { flex-wrap: wrap; }
        .market-box { display: none; }
    }"""

    light_vars = """:root {
        color-scheme: light;
        --bg: #f4f6f9; --surface: #ffffff; --surface2: #f0f2f6; --surface3: #e7eaf0;
        --border: #d7dbe3; --border-light: #c3c9d4; --text: #151823; --text-dim: #4a5063; --text-faint: #767d92;
        --green: #0e9f6e; --lime: #4d7c0f; --orange: #d97706; --red: #dc2626;
        --blue: #2563eb; --cyan: #0891b2; --yellow: #ca8a04;
        --focus-ring: rgba(14,159,110,0.15); --row-alt: #eef1f6;
        --ticker-hover-bg: rgba(14,159,110,0.15);
        --ticker-hover-c: #151823;
        --track: rgba(0,0,0,0.08); --row-line: rgba(0,0,0,0.09);
        --news-line: rgba(0,0,0,0.08);
    }
"""
    return (
        css
        + """
    @page { margin: 12mm; }
    @media print {
"""
        + light_vars
        + """        body { background: #fff; padding: 0; }
        .filters { display: none; }
        th { position: static; background: var(--surface2); backdrop-filter: none; -webkit-backdrop-filter: none; }
        td.rank, td.ticker { position: static; }
        .copy-t { display: none; }
        .table-wrap { border: none; }
        .hero-pill, .filters { background: var(--surface); }
        * { animation: none !important; }
        th.fit-hidden, td.fit-hidden { display: table-cell; }  /* full table on paper */
        .stat, .table-wrap, .news-panel, .chart-box, .signal, .inst-tile, .detail-card { box-shadow: none; }
        a { color: inherit; text-decoration: none; }
    }
    """
    )


def _js_block() -> str:
    """Client-side JS: column sorting, news toggling, row filtering."""
    return """let sortDir = {};
function sortTable(col) {
    const table = document.getElementById("stockTable");
    const tbody = table.tBodies[0];
    const allRows = Array.from(tbody.rows);
    const th = table.tHead.rows[0].cells[col];

    sortDir[col] = sortDir[col] === "asc" ? "desc" : "asc";
    const dir = sortDir[col];

    for (let cell of table.tHead.rows[0].cells) {
        cell.classList.remove("sorted-asc", "sorted-desc");
    }
    th.classList.add(dir === "asc" ? "sorted-asc" : "sorted-desc");

    // Build pairs: each data row + its following news row (if any) - keep them together
    const pairs = [];
    for (let i = 0; i < allRows.length; i++) {
        const row = allRows[i];
        if (row.classList.contains("news-row")) continue;
        const newsRow = (i + 1 < allRows.length && allRows[i + 1].classList.contains("news-row")) ? allRows[i + 1] : null;
        pairs.push({dataRow: row, newsRow: newsRow});
        if (newsRow) i++;
    }

    const ratingOrder = {"EXCELLENT":4,"GOOD":3,"MODERATE":2,"POOR":1};

    pairs.sort((a, b) => {
        let aCell = a.dataRow.cells[col];
        let bCell = b.dataRow.cells[col];
        let aVal = aCell ? aCell.textContent.trim() : "";
        let bVal = bCell ? bCell.textContent.trim() : "";
        let aText = aVal.replace(/<[^>]*>/g, "").trim();
        let bText = bVal.replace(/<[^>]*>/g, "").trim();

        // Rating column (3) — custom order, not alphabetical
        if (col === 3) {
            let aR = ratingOrder[aText.toUpperCase()] || 0;
            let bR = ratingOrder[bText.toUpperCase()] || 0;
            return dir === "asc" ? aR - bR : bR - aR;
        }

        let aNum = parseFloat(aText.replace(/[^0-9.\\-]/g, ""));
        let bNum = parseFloat(bText.replace(/[^0-9.\\-]/g, ""));
        let aIsNum = !isNaN(aNum) && /[0-9]/.test(aText);
        let bIsNum = !isNaN(bNum) && /[0-9]/.test(bText);
        if (aIsNum && bIsNum) {
            return dir === "asc" ? aNum - bNum : bNum - aNum;
        }
        return dir === "asc" ? aText.localeCompare(bText) : bText.localeCompare(aText);
    });

    // Re-append in sorted order, keeping news rows attached to their parent
    pairs.forEach(p => {
        tbody.appendChild(p.dataRow);
        if (p.newsRow) tbody.appendChild(p.newsRow);
    });
    restripe();
}

// App grid parity: zebra by visible rank (odd ranks on the alt band).
function restripe() {
    const rows = document.querySelectorAll("#stockTable tbody tr:not(.news-row)");
    let n = 0;
    rows.forEach(r => {
        if (r.style.display === "none") return;
        r.classList.toggle("row-alt", n % 2 === 0);
        n++;
    });
}

function toggleNews(tickerId) {
    const newsRow = document.getElementById("news-" + tickerId);
    if (!newsRow) return;

    // Collapse any other open news panels
    document.querySelectorAll(".news-row").forEach(row => {
        if (row.id !== "news-" + tickerId) {
            row.style.display = "none";
            row.classList.remove("open");
        }
    });

    // Toggle this one (open class drives the fade-in)
    const opening = newsRow.style.display === "none";
    newsRow.style.display = opening ? "" : "none";
    newsRow.classList.toggle("open", opening);
}

function copyTicker(e, el) {
    e.stopPropagation();  // don't expand the news panel
    const ta = document.createElement("textarea");
    ta.value = el.closest("tr").dataset.ticker;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");  // deprecated but prompt-free on file://
    ta.remove();
    const old = el.textContent;
    el.textContent = "✓";
    setTimeout(() => { el.textContent = old; }, 1000);
}

function setFilter(id, val, el) {
    document.getElementById(id).value = val;
    el.parentElement.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
    el.classList.add("active");
    filterTable();
}

function clearFilter(which) {
    if (which === "search") {
        document.getElementById("search").value = "";
    } else if (which === "rating") {
        document.getElementById("chip-rating").hidden = true;
    } else if (which === "score") {
        // Re-click the Score group's "All" chip so the active pill moves too.
        const allBtn = document.getElementById("chip-score-all");
        if (allBtn) { allBtn.click(); return; }  // setFilter runs filterTable
    }
    filterTable();
}

// Status line + chips track the live filter state (app _render_current_page).
function syncResultsHead(shown) {
    const el = document.getElementById("countLabel");
    if (!el) return;
    const search = document.getElementById("search").value.trim().toUpperCase();
    const minScore = parseFloat(document.getElementById("minScore").value) || 0;
    const chipR = document.getElementById("chip-rating");
    const ratingOn = chipR && !chipR.hidden ? chipR.dataset.rating : "";
    const parts = [];
    if (search) parts.push("'" + search + "'");
    if (ratingOn) parts.push("rating " + ratingOn.charAt(0) + ratingOn.slice(1).toLowerCase());
    if (minScore > 0) parts.push("score ≥ " + minScore);
    if (document.getElementById("f-fii").checked) parts.push("FII/DII data");
    if (document.getElementById("f-price").checked) parts.push("close < ₹100");
    if (document.getElementById("f-fund").checked) parts.push("fund 0");
    const base = el.dataset.scanned + " scanned  |  " + el.dataset.above
               + " above " + el.dataset.threshold + "+";
    el.textContent = parts.length
        ? base + "  |  filter: " + parts.join(", ") + " (" + shown + ")"
        : base;
    const chipQ = document.getElementById("chip-search");
    chipQ.hidden = !search;
    if (search) chipQ.textContent = "'" + search + "'  ✕";
    const chipS = document.getElementById("chip-score");
    chipS.hidden = !(minScore > 0);
    if (minScore > 0) chipS.textContent = "score ≥ " + minScore;
}

function filterTable() {
    const search = document.getElementById("search").value.trim().toLowerCase();
    const minScore = parseFloat(document.getElementById("minScore").value);
    const trendFilter = document.getElementById("trendFilter").value;
    const signalFilter = document.getElementById("signalFilter").value;
    const newsFilter = document.getElementById("newsFilter").value;
    const chipR = document.getElementById("chip-rating");
    const activeRating = chipR && !chipR.hidden ? chipR.dataset.rating : "";
    const fiiOn = document.getElementById("f-fii").checked;
    const priceOn = document.getElementById("f-price").checked;
    const fundOn = document.getElementById("f-fund").checked;
    const rows = document.getElementById("stockTable").tBodies[0].rows;
    let shown = 0;

    for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        // Skip news rows — they follow their parent
        if (row.classList.contains("news-row")) continue;

        const ticker = row.cells[1].textContent.toLowerCase();
        const score = parseFloat(row.getAttribute("data-score") || row.cells[2].textContent);
        const trend = row.cells[18].textContent;
        const maBull = row.getAttribute("data-ma-bull") === "true";
        const abovePoc = row.getAttribute("data-above-poc") === "true";
        const bothMa = row.getAttribute("data-both-ma") === "true";
        const crossed = row.getAttribute("data-crossed") === "true";
        const hasEntry = row.getAttribute("data-entry") === "true";
        const rating = row.getAttribute("data-rating") || "";
        const price = parseFloat(row.getAttribute("data-price")) || 0;
        const fund = parseFloat(row.getAttribute("data-fund")) || 0;
        const fii = row.getAttribute("data-fii") === "1";

        const matchSearch = ticker.includes(search);
        const matchScore = score >= minScore;
        const matchTrend = !trendFilter || trend.includes(trendFilter);
        const matchRating = !activeRating || rating === activeRating;
        const matchFii = !fiiOn || fii;
        const matchPrice = !priceOn || price >= 100;
        const matchFund = !fundOn || fund > 0;

        let matchSignal = true;
        if (signalFilter === "entry") matchSignal = hasEntry;
        else if (signalFilter === "both_ma+poc") matchSignal = bothMa && abovePoc;
        else if (signalFilter === "ma+poc") matchSignal = maBull && abovePoc;
        else if (signalFilter === "crossed") matchSignal = crossed;
        else if (signalFilter === "ma_bull") matchSignal = maBull;
        else if (signalFilter === "above_poc") matchSignal = abovePoc;
        else if (signalFilter === "both_ma") matchSignal = bothMa;

        let matchNews = true;
        if (newsFilter) {
            const newsRow = rows[i + 1];
            if (newsRow && newsRow.classList.contains("news-row")) {
                const newsText = newsRow.textContent.toLowerCase();
                if (newsFilter === "good_news") matchNews = newsText.includes("good");
                else if (newsFilter === "bad_news") matchNews = newsText.includes("bad");
            } else {
                matchNews = false;
            }
        }

        const visible = matchSearch && matchScore && matchTrend && matchSignal
                     && matchNews && matchRating && matchFii && matchPrice && matchFund;
        if (visible) shown++;
        row.style.display = visible ? "" : "none";
        // News rows default to hidden — collapse the one that follows
        if (i + 1 < rows.length && rows[i + 1].classList.contains("news-row")) {
            rows[i + 1].style.display = "none";
        }
    }
    restripe();
    syncResultsHead(shown);
}

function syncColspan() {
    // Responsive tiers hide columns — keep the detail/news row's span in step.
    const ths = document.querySelectorAll("#stockTable thead th");
    let n = 0;
    ths.forEach(th => { if (th.offsetParent !== null) n++; });
    if (n > 0) document.querySelectorAll(".news-row > td").forEach(td => { td.colSpan = n; });
}

// Content-fit: hide the least useful columns until the table fits its
// container — viewport media queries alone can't (19 columns need
// ~1750px). Indexes cheapest-first: chop/vol diagnostics, then readouts,
// then POC/Both MA. Sort/filter indexes are positional so
// display:none is safe.
const FIT_ORDER = [17, 16, 14, 15, 7, 8, 12];
function fitColumns() {
    const table = document.getElementById("stockTable");
    const wrap = table.closest(".table-wrap");
    document.querySelectorAll(".fit-hidden").forEach(e => e.classList.remove("fit-hidden"));
    const ths = table.tHead.rows[0].cells;
    for (let i = 0; i < FIT_ORDER.length && table.offsetWidth > wrap.clientWidth; i++) {
        const idx = FIT_ORDER[i];
        ths[idx].classList.add("fit-hidden");
        table.querySelectorAll(`tbody td:nth-child(${idx + 1})`).forEach(td => td.classList.add("fit-hidden"));
    }
}

function relayout() { fitColumns(); syncColspan(); }
let relayoutTimer;
window.addEventListener("resize", () => {
    clearTimeout(relayoutTimer);
    relayoutTimer = setTimeout(relayout, 80);
});
window.addEventListener("load", relayout);  // web fonts change metrics
filterTable();  // apply export-time filters + recount before first paint
relayout();"""


def _chip(gid: str, val: str, label: str, active: bool = False, cid: str = "") -> str:
    """One filter pill; writes the value into the hidden input via setFilter()."""
    cls = "chip active" if active else "chip"
    id_attr = f' id="{cid}"' if cid else ""
    return (
        f'<button type="button" class="{cls}"{id_attr} '
        f"onclick=\"setFilter('{gid}','{val}',this)\">{label}</button>"
    )


def _chip_group(label: str, chips: str) -> str:
    return (
        f'<div class="chip-group"><span class="chip-label">{label}</span>{chips}</div>'
    )


def _summary_header_html(
    title: str,
    now: str,
    results: list,
    passed: list,
    failed: list,
    threshold: float,
    meta: list | None = None,
    filters: dict | None = None,
) -> str:
    """Hero banner, overview band (stats), and filter chips."""
    meta_spans = "".join(f"<span>{_html.escape(str(m))}</span>" for m in (meta or []))
    search_val = _html.escape(str((filters or {}).get("search") or ""))
    # Score group starts at the exported slider value when present, else the
    # same default the old <select> preselected (70+ if threshold ≥ 70,
    # 50+ if exactly 50, otherwise All).
    f_score = float((filters or {}).get("score") or 0)
    default_min = (
        int(f_score)
        if f_score > 0
        else (70 if threshold >= 70 else 50 if threshold == 50 else 0)
    )
    score_chips = "".join(
        _chip(
            "minScore",
            str(v),
            lbl,
            v == default_min,
            cid="chip-score-all" if v == 0 else "",
        )
        for v, lbl in ((0, "All"), (70, "70+"), (50, "50+"), (30, "30+"))
    )
    trend_chips = "".join(
        _chip("trendFilter", v, lbl)
        for v, lbl in (("", "All"), ("Bull", "Bull"), ("Bear", "Bear"))
    )
    signal_chips = "".join(
        _chip("signalFilter", v, lbl)
        for v, lbl in (
            ("", "Any"),
            ("entry", "Entry"),
            ("both_ma+poc", "Close&gt;MA+POC"),
            ("ma+poc", "MA+POC"),
            ("crossed", "Fresh X"),
            ("ma_bull", "MA Bull"),
            ("above_poc", "Above POC"),
            ("both_ma", "Both MA"),
        )
    )
    news_chips = "".join(
        _chip("newsFilter", v, lbl)
        for v, lbl in (("", "All"), ("good_news", "Good"), ("bad_news", "Bad"))
    )
    overview_cls = "" if results else " solo"
    # App hero status line (views_results._update_hero_status), verbatim.
    if results:
        entry_ct = sum(1 for r in results if r.get("entry_signal"))
        status = (
            f"{len(results)} stocks passed the crossover filter  ·  "
            f"{len(passed)} scored {threshold:.0f}+  ·  "
            f"{entry_ct} ENTRY signals"
        )
    else:
        status = (
            "Set your universe on the left, then RUN SCAN — "
            "HMA×EMA crossover • 10-factor score • news sentiment"
        )
    return f"""<header class="hero">
  <div class="hero-pill">
    <div class="hero-row">
      <div class="hero-text">
        <h1 class="hero-title">Find Your Next Swing Trade</h1>
        <div class="hero-sub">{status}</div>
      </div>
      <div class="meta"><span>🎯 Threshold {threshold:g}+</span><span>📦 {len(results)} total</span>{meta_spans}</div>
    </div>
  </div>
  <div class="market-box">
    <div class="market-label">{_html.escape(title)}</div>
    <div class="market-value">{len(results)}</div>
    <div class="market-change">⚡ Generated locally<br>⏱ {now}</div>
  </div>
</header>

<div class="overview{overview_cls}">
    <div class="summary">
        <div class="stat green">
            <div class="num">{len(passed)}</div>
            <div class="stat-meta">
                <div class="label">Passed · {threshold:g}+</div>
                <div class="sub">{len(passed) / max(len(results), 1) * 100:.0f}% hit rate</div>
            </div>
        </div>
        <div class="stat red">
            <div class="num">{len(failed)}</div>
            <div class="stat-meta">
                <div class="label">Below threshold</div>
                <div class="sub">filtered out</div>
            </div>
        </div>
        <div class="stat cyan">
            <div class="num">{len(results)}</div>
            <div class="stat-meta">
                <div class="label">Total scanned</div>
                <div class="sub">{len({r.get("ticker", "")[:3] for r in results})} sectors</div>
            </div>
        </div>
        <div class="stat lime">
            <div class="num">{passed[0]["total"] if passed else 0:.0f}</div>
            <div class="stat-meta">
                <div class="label">Highest score</div>
                <div class="sub">{passed[0]["ticker"] if passed else "—"}</div>
            </div>
        </div>
    </div>
</div>

<div class="filters">
    <input type="text" id="search" value="{search_val}" placeholder="Search ticker..." oninput="filterTable()">
    <input type="hidden" id="minScore" value="{default_min}">
    <input type="hidden" id="trendFilter" value="">
    <input type="hidden" id="signalFilter" value="">
    <input type="hidden" id="newsFilter" value="">
    {_chip_group("Score", score_chips)}
    {_chip_group("Trend", trend_chips)}
    {_chip_group("Signal", signal_chips)}
    {_chip_group("News", news_chips)}
</div>"""


def _results_head_html(results: list, threshold: float, filters: dict | None) -> str:
    """App section header — interactive like the app's: real checkboxes,
    clearable filter chips, and a status line that recounts on every change.

    ``filters`` carries the export-time grid state (app ``_export_html``):
    ``search``, ``rating`` (uppercase, "ALL" = off), ``score`` (>0 = on),
    ``fii_dii`` / ``price100`` / ``fund0`` toggles, and pre-filter
    ``scanned`` / ``above`` counts. None (CLI) renders all toggles off.
    Strings mirror views_results._render_current_page; _js_block re-applies
    the same rules client-side so the table starts in the exported state.
    """
    f = filters or {}
    scanned = int(f.get("scanned") or len(results))
    above = int(
        f.get("above")
        if f.get("above") is not None
        else sum(1 for r in results if (r.get("total", 0) or 0) >= threshold)
    )
    # App _on_search_change stores upper — display parts/match the same way.
    search = str(f.get("search") or "").upper()
    rating = str(f.get("rating") or "ALL").upper()
    score = float(f.get("score") or 0)

    def _matches(r: dict) -> bool:
        """App _row_matches_filters over the exported rows — the visible count."""
        if search and search not in str(r.get("ticker", "")).upper():
            return False
        if (
            rating != "ALL"
            and str(r.get("combined_rating") or "POOR").upper() != rating
        ):
            return False
        if score_of(r) < score:
            return False
        if f.get("fii_dii") and not has_fii_dii(r):
            return False
        if f.get("price100") and float(r.get("close") or 0) < 100:
            return False
        if f.get("fund0") and float(r.get("fundamentals") or 0) <= 0:
            return False
        return True

    shown = sum(1 for r in results if _matches(r))

    parts = []
    if search:
        parts.append(f"'{_html.escape(search)}'")
    if rating != "ALL":
        parts.append(f"rating {rating.title()}")
    if score > 0:
        parts.append(f"score ≥ {score:.0f}")
    if f.get("fii_dii"):
        parts.append("FII/DII data")
    if f.get("price100"):
        parts.append("close < ₹100")
    if f.get("fund0"):
        parts.append("fund 0")
    suffix = f"  |  filter: {', '.join(parts)} ({shown})" if parts else ""
    status = f"{scanned} scanned  |  {above} above {threshold:.0f}+{suffix}"

    def _cb(cid: str, label: str, on: bool) -> str:
        checked = " checked" if on else ""
        return (
            f'<label class="cb"><input type="checkbox" id="{cid}"{checked} '
            f'onchange="filterTable()">{label}</label>'
        )

    cbs = (
        _cb("f-fii", "Show stocks with FII/DII data", bool(f.get("fii_dii")))
        + _cb("f-price", "Hide stocks below ₹100", bool(f.get("price100")))
        + _cb("f-fund", "Hide zero fundamental score", bool(f.get("fund0")))
    )

    search_chip = f"'{_html.escape(search)}'  ✕" if search else ""
    rating_chip = f"rating {rating.title()}  ✕" if rating != "ALL" else ""
    score_chip = f"score ≥ {score:.0f}" if score > 0 else ""
    chips = (
        '<button type="button" class="filter-chip" id="chip-search" '
        f"onclick=\"clearFilter('search')\"{' hidden' if not search else ''}>"
        f"{search_chip}</button>"
        '<button type="button" class="filter-chip" id="chip-rating" '
        f'data-rating="{rating if rating != "ALL" else ""}" '
        f"onclick=\"clearFilter('rating')\""
        f"{' hidden' if rating == 'ALL' else ''}>{rating_chip}</button>"
        '<button type="button" class="filter-chip" id="chip-score" '
        f"onclick=\"clearFilter('score')\"{' hidden' if score <= 0 else ''}>"
        f"{score_chip}</button>"
    )

    return f"""<div class="results-head">
    <span class="results-title">Scan Results</span>
    {cbs}
    <div class="filter-chips">{chips}</div>
    <span class="count-label" id="countLabel" data-scanned="{scanned}" data-above="{above}" data-threshold="{threshold:.0f}">{status}</span>
</div>"""


# (label, drop-tier) for the results table — labels are RESULT_COLS verbatim
# so the grid and the export read identically. Tier 1 hides at ≤1599px,
# tier 2 adds on ≤1399px — same boundaries and same order as
# scanner.ui.ui_kit.COL_HIDE_ORDER (diagnostics first, score components
# last). Sort indexes are positional, so order here is load-bearing.
_REPORT_COLS: list[tuple[str, int | None]] = [
    ("#", None),
    ("Ticker", None),
    ("Score", None),
    ("Rating", None),
    ("ENTRY", None),
    ("Price", None),
    ("MA", None),
    ("POC", None),
    ("Both MA", None),
    ("Vol/10", 2),
    ("RS/10", 2),
    ("F/20", 2),
    ("RSI Val", None),
    ("1M%", None),
    ("Dir", 1),
    ("ADX", 1),
    ("Volatility", None),
    ("Chop", 1),
    ("1M", None),
]


def _table_head_html() -> str:
    """Static table wrapper, header row, and <tbody> opening tag."""
    ths = []
    for i, (label, tier) in enumerate(_REPORT_COLS):
        cls = f' class="c-t{tier}"' if tier else ""
        # The spark column is visual-only — no sort handler.
        onclick = f' onclick="sortTable({i})"' if label != "1M" else ""
        ths.append(f"<th{cls}{onclick}>{label}</th>")
    return f"""<div class="table-wrap">
<table id="stockTable">
<thead>
<tr>
{chr(10).join(ths)}
</tr>
</thead>
<tbody>"""


def _trade_reasons_html(r: dict) -> str:
    """Trade-reasons panel HTML (empty string when no reasons)."""
    from ..shared.trade_reasons import build_trade_reasons

    trade_reasons = build_trade_reasons(r)
    if not trade_reasons:
        return ""
    reason_items = ""
    for reason_text in trade_reasons:
        is_risk = reason_text.startswith("Risk:")
        icon_color = "var(--orange)" if is_risk else "var(--green)"
        icon_char = "⚠" if is_risk else "✓"
        reason_items += f'<div class="reason-item"><span class="reason-icon" style="color:{icon_color}">{icon_char}</span> {_html.escape(reason_text)}</div>'
    return f'<div class="detail-title">Why this trade?</div>{reason_items}'


# Report palette for shared signal-spec color roles ("" = inherit text).
_ROLE_CSS = {
    "green": "var(--green)",
    "lime": "var(--lime)",
    "red": "var(--red)",
    "orange": "var(--orange)",
    "text": "",
}


def _key_signals_html(r: dict) -> str:
    """Signal chips — shared specs, report palette."""
    out = ""
    for label, value, role in signal_specs(r):
        css = _ROLE_CSS[role]
        style = f' style="color:{css}"' if css else ""
        out += (
            f'<div class="signal"><span class="signal-label">{label}</span>'
            f'<span class="signal-val"{style}>{_html.escape(value)}</span></div>'
        )
    return out


def _breakdown_html(r: dict) -> str:
    """10 category bars — shared SCORE_CATS with the app's breakdown."""
    rows = ""
    for label, key, mx, _role, css in SCORE_CATS:
        score = float(r.get(key) or 0)
        pct = min(score / mx, 1.0) * 100 if mx else 0.0
        rows += (
            f'<div class="bd-row"><span>{label}</span>'
            f'<span class="bd-track"><span class="bd-fill" '
            f'style="width:{pct:.0f}%;background:{css}"></span></span>'
            f'<span class="bd-val">{score:g}/{mx}</span></div>'
        )
    return f'<div class="bd-list">{rows}</div>'


def _fmt_cr(v) -> str:
    """Flow as a signed Cr string, e.g. ₹-3,694 Cr / ₹+2,838 Cr (app parity)."""
    try:
        return f"₹{float(v):+,.0f} Cr"
    except (TypeError, ValueError):
        return "-"


def _inst_tile_html(
    label: str, value: str, total, recent, state: str | None = None
) -> str:
    """One tile: label → [arrow] value → total/recent sub-line (app `_inst_tile`)."""
    arrow = ""
    if state == "missing":
        sub, color = "n/a", "var(--text-dim)"
    elif total is None:
        sub, color = "no history", "var(--text-dim)"
    elif total == 0:
        sub, color = "No change", "var(--text-dim)"
    else:
        up = total > 0
        color = "var(--green)" if up else "var(--red)"
        arrow = (
            f'<span style="color:{color};font-size:10px">{"▲" if up else "▼"}</span>'
        )
        sub = f"total {fmt_pct(total)}"
        if recent is not None:
            sub += f" · recent {fmt_pct(recent)}"
    return (
        f'<div class="inst-tile"><span class="inst-label">{label}</span>'
        f'<span class="inst-val">{arrow}{value}</span>'
        f'<span class="inst-sub" style="color:{color}">{sub}</span></div>'
    )


def _institutional_html(r: dict, flow: list | None) -> str:
    """Per-stock shareholding tiles — the app's `_pct_tile` fallback chain.

    series % → promoter snapshot % → market flow ₹Cr → scan-time markers;
    always renders (missing sources become ``n/a`` tiles), like the app card.
    """
    shp = r.get("_shareholding") or {}
    series = shp.get("series") or {}

    def _tile(label, key, flow_key=None, marker_key=None, snap=None):
        e = series.get(key)
        if e:
            return _inst_tile_html(
                label,
                f"{float(e['latest']):.1f}%",
                e.get("total"),
                e.get("recent"),
            )
        if snap:
            return _inst_tile_html(label, f"{float(snap):.1f}%", None, None)
        if flow_key and flow:
            from ..api.indian_market import flow_window_pcts

            total, recent = flow_window_pcts(flow, flow_key)
            return _inst_tile_html(label, _fmt_cr(flow[0][flow_key]), total, recent)
        net = r.get(marker_key) if marker_key else None
        if net is not None:
            return _inst_tile_html(label, _fmt_cr(net), None, None)
        return _inst_tile_html(label, "—", None, None, state="missing")

    tiles = (
        _tile("FII", "foreign_institutions", flow_key="fii", marker_key="_fii_net")
        + _tile("DII", "domestic_institutions", flow_key="dii", marker_key="_dii_net")
        + _tile("Promoters", "promoters", snap=r.get("_promoter_holding"))
    )
    hint = "Shareholding"
    if shp.get("quarter"):
        hint += f" · {shp['quarter']}"
    return (
        f'<div class="inst-tiles">{tiles}</div>'
        f'<div class="inst-hint">{_html.escape(hint)}</div>'
    )


def _detail_sections_html(r: dict, reasons_html: str, flow: list | None) -> str:
    """Chart, signals, breakdown, reasons cards + full-width institutional
    card — mirrors the app's detail composition."""
    px = [v for v in (r.get("px_tail") or []) if isinstance(v, (int, float))]
    chart = ""
    if len(px) >= 2:
        chart = (
            '<div class="detail-title">Price (last 20 closes)</div>'
            f'<div class="chart-box">{_sparkline_svg(px, 560, 170)}'
            f'<span class="chart-hi">{max(px):.1f}</span>'
            f'<span class="chart-lo">{min(px):.1f}</span></div>'
        )
    signals = (
        '<div class="detail-title">Key Signals</div>'
        f'<div class="signals">{_key_signals_html(r)}</div>'
    )
    breakdown = f'<div class="detail-title">Score Breakdown</div>{_breakdown_html(r)}'
    chart_card = f'<div class="detail-card">{chart}</div>' if chart else ""
    signals_card = f'<div class="detail-card">{signals}</div>'
    breakdown_card = f'<div class="detail-card">{breakdown}</div>'
    reasons_card = (
        f'<div class="detail-card">{reasons_html}</div>' if reasons_html else ""
    )
    inst = _institutional_html(r, flow)
    inst_block = (
        '<div class="detail-card"><div class="detail-title">Institutional '
        f"Positioning</div>{inst}</div>"
        if inst
        else ""
    )
    return (
        '<div class="detail-grid">'
        f'<div class="detail-col">{chart_card}{signals_card}</div>'
        f'<div class="detail-col">{breakdown_card}{reasons_card}</div>'
        "</div>" + inst_block
    )


def _news_panel_html(
    ticker: str, news_items: list, fetch_news: bool, preamble_html: str
) -> str:
    """Expandable detail row: app-style sections + reasons + news."""
    news_rows_html = ""
    if news_items:
        good_count = sum(
            1
            for n in news_items
            if isinstance(n, dict) and n.get("sentiment") == "Good"
        )
        bad_count = sum(
            1 for n in news_items if isinstance(n, dict) and n.get("sentiment") == "Bad"
        )
        neutral_count = sum(
            1
            for n in news_items
            if isinstance(n, dict) and n.get("sentiment") == "Neutral"
        )
        summary_parts = []
        if good_count:
            summary_parts.append(f'<span class="news-good">{good_count} Good</span>')
        if bad_count:
            summary_parts.append(f'<span class="news-bad">{bad_count} Bad</span>')
        if neutral_count:
            summary_parts.append(
                f'<span class="news-neutral">{neutral_count} Neutral</span>'
            )

        for n in news_items:
            if not isinstance(n, dict):
                continue
            sent = str(n.get("sentiment", "Neutral"))
            sent_cls = _html.escape(sent.lower())
            safe_title = _html.escape(str(n.get("title", "")))
            safe_summary = _html.escape(str(n.get("summary", ""))[:200])
            safe_publisher = _html.escape(
                str(n.get("publisher") or n.get("provider") or "")
            )
            safe_date = _html.escape(str(n.get("date", "")))
            safe_sentiment = _html.escape(sent)
            news_rows_html += f"""
                        <div class="news-item">
                            <span class="news-sentiment {sent_cls}">[{safe_sentiment}]</span>
                            <span class="news-date">{safe_date}</span>
                            <span class="news-pub">{safe_publisher}</span>
                            <div class="news-title">{safe_title}</div>
                            <div class="news-summary">{safe_summary}</div>
                        </div>"""
        news_summary_html = (
            f'<div class="news-summary-line">{" | ".join(summary_parts)}</div>'
        )
    elif fetch_news:
        news_summary_html = ""
        news_rows_html = '<div class="news-item"><span class="news-title" style="color:var(--text-dim)">No recent news found</span></div>'
    else:
        news_summary_html = ""
        news_rows_html = ""

    # Always build the expandable panel: detail sections + reasons + news
    panel_content = preamble_html
    if news_summary_html:
        panel_content += news_summary_html
    if news_rows_html:
        panel_content += news_rows_html + (
            f'<a class="news-more" href="https://finance.yahoo.com/quote/'
            f'{_html.escape(ticker)}/news/" target="_blank" rel="noopener">'
            f"More {_html.escape(ticker)} news →</a>"
        )

    if panel_content:
        return f"""
                <tr class="news-row" id="news-{_news_id(ticker)}" style="display:none">
                    <td colspan="23">
                        <div class="news-panel">
                            {panel_content}
                        </div>
                    </td>
                </tr>"""
    return ""


def _result_row_html(
    r: dict, threshold: float, news_map: dict, fetch_news: bool, flow, rank: int
) -> str:
    """Render one result as a data row plus its expandable news row.

    Cell text, sizes and colors mirror the app grid row-for-row
    (views_results._row_specs / _ROW_STYLE / _create_row_controls).
    """
    score = r.get("total", 0) or 0
    ticker = str(r.get("ticker", "?"))
    above = score >= threshold
    rating = str(r.get("combined_rating") or "POOR")
    entry = bool(r.get("entry_signal"))
    ma_text, ma_role = ma_chip(r)
    ma_cls = {"green": "ma-cross", "lime": "ma-bull", "red": "ma-bear"}[ma_role]
    above_poc = bool(r.get("above_poc"))
    both_ma = bool(r.get("close_above_both_ma"))
    rsi_val = r.get("rsi_val")
    pc1m = r.get("pc1m", 0) or 0
    trend_dir = r.get("trend_dir") or ""
    sideways = bool(r.get("is_sideways"))
    sideways_reasons = _html.escape(", ".join(r.get("sideways_reasons", [])))
    row_cls = " ".join(
        cls for cls, keep in (("highlight", above), ("row-alt", bool(rank % 2))) if keep
    )
    rail = {
        "EXCELLENT": "var(--green)",
        "GOOD": "var(--lime)",
        "MODERATE": "var(--orange)",
    }.get(rating, "var(--red)")
    rsi_cls = (
        "rval-none"
        if rsi_val is None
        else "rval-ok"
        if 40 <= rsi_val <= 70
        else "rval-off"
    )
    ticker_cls = "ticker above" if above else "ticker"
    entry_cls = "entry yes" if entry else "entry"
    poc_cls = "poc-above" if above_poc else "poc-below"
    bothma_cls = "bothma-yes" if both_ma else "bothma-no"

    reasons_html = _trade_reasons_html(r)
    detail_html = _detail_sections_html(r, reasons_html, flow)
    news_items = news_map.get(ticker, []) if fetch_news else []
    # Rows outside news_targets carry no news block — claiming "no news"
    # for a ticker we never fetched would be a lie.
    news_html = _news_panel_html(
        ticker, news_items, fetch_news and ticker in news_map, detail_html
    )

    return f"""
        <tr class="{row_cls}"
            style="--rail: {rail}"
            data-ma-bull="{"true" if r.get("ma_bullish") else "false"}"
            data-above-poc="{"true" if above_poc else "false"}"
            data-both-ma="{"true" if both_ma else "false"}"
            data-crossed="{"true" if r.get("ma_crossed_above") else "false"}"
            data-entry="{"true" if entry else "false"}"
            data-ticker="{_html.escape(ticker)}"
            data-score="{float(score):g}"
            data-rating="{_html.escape(rating.upper())}"
            data-price="{float(r.get("close") or 0):g}"
            data-fund="{float(r.get("fundamentals") or 0):g}"
            data-fii="{"1" if has_fii_dii(r) else "0"}">
            <td class="rank">{rank}</td>
            <td class="{ticker_cls}" onclick="toggleNews('{_news_id(ticker)}')">{_html.escape(ticker)}<button type="button" class="copy-t" onclick="copyTicker(event,this)" title="Copy ticker">⧉</button></td>
            <td class="score s-{_score_class(score)}">{score:.0f}</td>
            <td class="rating r-{rating.lower()}">{_html.escape(rating)}</td>
            <td class="{entry_cls}">{"YES" if entry else "--"}</td>
            <td class="num price">₹{r.get("close", 0) or 0:.0f}</td>
            <td class="{ma_cls}">{_html.escape(ma_text)}</td>
            <td class="{poc_cls}">{"Above" if above_poc else "Below"}</td>
            <td class="{bothma_cls}">{"YES" if both_ma else "NO"}</td>
            <td class="num sc-orange c-t2">{r.get("volume", 0) or 0:.0f}</td>
            <td class="num sc-lime c-t2">{r.get("rel_str", 0) or 0:.0f}</td>
            <td class="num sc-fund c-t2">{r.get("fundamentals", 0) or 0:.0f}</td>
            <td class="num {rsi_cls}">{"—" if rsi_val is None else f"{rsi_val:.1f}"}</td>
            <td class="num {"bull" if pc1m > 0 else "bear"}">{pc1m:+.1f}%</td>
            <td class="c-t1 {"bull" if trend_dir == "Bull" else "bear"}">{("^ " if trend_dir == "Bull" else "v ") + (trend_dir or "?")}</td>
            <td class="num c-t1">{r.get("adx_val", 0) or 0:.0f}</td>
            <td>{r.get("volat_stat") or "—"}</td>
            <td class="c-t1 {"sideways" if sideways else "trending"}" title="{sideways_reasons}">{"Chop" if sideways else "OK"}</td>
            <td class="spark-cell">{_sparkline_svg(r.get("px_tail") or [])}</td>
        </tr>
        {news_html}"""


def generate_html_report(
    results: list,
    title: str = "HMAxEMA Stock Scanner",
    threshold: float = 50.0,
    fetch_news: bool = True,
    meta: list | None = None,
    filters: dict | None = None,
    news_targets: list | None = None,
) -> str:
    """
    Generate a complete HTML report from scan results.

    Args:
        results: List of dicts with 'ticker', 'total', scores, and metadata
        title: Report title
        threshold: Minimum score threshold
        fetch_news: Whether to fetch news sentiment for each stock
        meta: Extra metadata chips (universe, timeframe, ...) for the header
        filters: Export-time grid filter state for the section header
            (search/rating/score/fii_dii/price100/fund0 + scanned/above
            counts); None (CLI) renders the header with toggles off
        news_targets: Tickers to fetch missing news for (defaults to all
            ``results``). Rows outside this set still render their detail
            panel — they just carry no news block.

    Returns:
        Complete HTML string
    """
    # Sort by total score descending (non-mutating, tolerant of bad rows)
    results = sorted(results, key=lambda x: x.get("total", 0) or 0, reverse=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    passed = [r for r in results if (r.get("total", 0) or 0) >= threshold]
    failed = [r for r in results if (r.get("total", 0) or 0) < threshold]

    # ── News: reuse rows' prefetched _news_items; fetch only the rest ──
    news_map: dict[str, list] = {
        str(r.get("ticker", "?")): list(r["_news_items"] or [])
        for r in results
        if "_news_items" in r
    }
    if fetch_news:
        scope = results if news_targets is None else news_targets
        missing = [
            t
            for t in dict.fromkeys(str(r.get("ticker", "?")) for r in scope)
            if t not in news_map
        ]
        if missing:
            news_map.update(_fetch_news_parallel(missing))

    rows_html = ""
    # One cached FII/DII flow read per report — the app's cache-only fallback.
    flow = None
    try:
        from ..api.indian_market import fetch_fii_dii_history

        flow = fetch_fii_dii_history(cache_only=True)
    except Exception:
        logger.debug("FII/DII flow cache read failed", exc_info=True)
    for i, r in enumerate(results, 1):
        rows_html += _result_row_html(r, threshold, news_map, fetch_news, flow, i)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_html.escape(title)}</title>
<style>
{_css_block()}
</style>
</head>
<body>

{_summary_header_html(title, now, results, passed, failed, threshold, meta, filters)}

{_results_head_html(results, threshold, filters)}

{_table_head_html()}
{rows_html}
</tbody>
</table>
</div>

<div class="footer">
    Generated by HMAxEMA Stock Scanner &nbsp;|&nbsp; Scoring engine mirrors the Pine Script indicator<br>
    Click any ticker to expand trade reasons &amp; news sentiment
</div>

<script>
{_js_block()}
</script>
</body>
</html>"""


def _score_class(score: float) -> str:
    """Return CSS class for score coloring."""
    if score >= 70:
        return "excellent"
    if score >= 50:
        return "good"
    if score >= 30:
        return "moderate"
    return "poor"


def prune_old(directory: str, pattern: str, keep: int) -> None:
    """Delete all but the newest ``keep`` files matching ``pattern`` in dir."""
    import glob as _glob
    import os as _os

    try:
        files = sorted(
            _glob.glob(_os.path.join(directory, pattern)),
            key=_os.path.getmtime,
            reverse=True,
        )
    except OSError:
        return
    for old in files[keep:]:
        try:
            _os.remove(old)
        except OSError:
            pass


def save_report(
    html: str, filename: str = "scanner_report.html", max_reports: int = 4
) -> str:
    """Save HTML report to file and keep only the last max_reports files."""
    import os as _os

    # Atomic save so a crash never leaves a truncated report behind
    report_dir = _os.path.dirname(filename) or "."
    try:
        _os.makedirs(report_dir, exist_ok=True)
    except OSError:
        pass
    tmp = filename + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(html)
    _os.replace(tmp, filename)

    # Keep only the newest max_reports (pattern also covers the bare default name)
    prune_old(report_dir, "scanner_report*.html", max_reports)

    return filename
