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
from ..shared.detail_specs import SCORE_CATS, fmt_pct, ma_chip, signal_specs, spark_move

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
           identical to the app: same surfaces, borders and accent colors.
           Deepened for a modern, high-contrast canvas with glass surfaces. */
        color-scheme: dark;
        --bg: #060a08; --surface: #0e1512; --surface2: #121d17; --surface3: #18271f;
        --border: #1d3128; --border-light: #2d5142; --text: #eafaf2; --text-dim: #93ab9e; --text-faint: #5e7d6e;
        --green: #10b981; --lime: #a3e635; --orange: #fb923c; --red: #f87171;
        --blue: #2dd4bf; --cyan: #14b8a6; --yellow: #facc15;
        /* Row zebra / component tints for the dark-only report */
        --focus-ring: rgba(52,211,153,0.18); --row-alt: #0a110e;
        --ticker-hover-bg: rgba(52,211,153,0.14);
        --ticker-hover-c: #ffffff; --track: rgba(255,255,255,0.07); --row-line: rgba(31,51,40,0.65);
        --news-line: rgba(31,51,40,0.5);
        --radius: 14px; --radius-sm: 10px;
        --hair: rgba(255,255,255,0.06);
        --glass: rgba(255,255,255,0.045);
        --row-h: 36px;
        --accent-grad: linear-gradient(135deg, rgba(16,185,129,0.30), rgba(20,184,166,0.10) 55%, rgba(16,185,129,0) 100%);
        /* Local-first stacks: no CDN font request, falls back to the OS UI font
           (same system-ui the app renders with) and system mono for numbers. */
        --mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    ::selection { background: rgba(16,185,129,0.32); color: #ffffff; }
    ::-webkit-scrollbar { width: 11px; height: 11px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 99px; border: 2px solid var(--bg); }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
    body { font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
           background:
               radial-gradient(900px 520px at 10% -12%, rgba(16,185,129,0.16), transparent 62%),
               radial-gradient(780px 460px at 96% -16%, rgba(20,184,166,0.12), transparent 64%),
               radial-gradient(1100px 700px at 50% 120%, rgba(16,185,129,0.06), transparent 70%),
               var(--bg);
           color: var(--text); padding: 26px; line-height: 1.5; min-height: 100vh;
           -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }

    /* ─── Hero banner: slim tinted card — title + status left, meta
           pills right (one band, no dead space), market box beside it.
           Soft top-right sheen + staggered rise-in. ── */
    .hero { display: flex; align-items: stretch; gap: 12px; margin-bottom: 12px;
            animation: rise 0.5s cubic-bezier(0.2, 0.7, 0.3, 1) both; }
    .hero-pill { flex: 1; min-width: 0; position: relative; overflow: hidden;
                 background: linear-gradient(100deg, rgba(16,185,129,0.22), rgba(13,148,136,0.08) 45%, rgba(16,185,129,0) 78%), var(--glass);
                 border: 1px solid var(--border); border-radius: var(--radius); padding: 13px 18px;
                 backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
                 box-shadow: 0 12px 32px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.06); }
    .hero-pill::after { content: ''; position: absolute; inset: 0; pointer-events: none;
                        background: radial-gradient(130% 240% at 100% -50%, rgba(255,255,255,0.11), transparent 55%); }
    .hero-row { display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap; }
    .hero-text { min-width: 0; }
    .hero-title { font-size: 21px; font-weight: 700; color: #ffffff; letter-spacing: -0.025em; line-height: 1.22;
                  background: linear-gradient(92deg, #ffffff 25%, #a7f3d0 100%); -webkit-background-clip: text;
                  background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-sub { color: rgba(209,250,229,0.85); font-size: 12px; margin-top: 4px; letter-spacing: 0.01em; }
    .meta { display: flex; gap: 6px; flex-wrap: wrap; }
    .meta span { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.14); border-radius: 99px;
                 padding: 4px 11px; font-size: 10px; font-weight: 600; color: #d1fae5; white-space: nowrap;
                 box-shadow: 0 1px 2px rgba(0,0,0,0.25); }
    .market-box { flex: none; width: 178px; text-align: right; position: relative; overflow: hidden;
                  background: linear-gradient(160deg, var(--surface3), var(--surface2));
                  border: 1px solid var(--border-light); border-radius: var(--radius); padding: 7px 15px;
                  box-shadow: 0 10px 26px rgba(0,0,0,0.34), inset 0 1px 0 rgba(255,255,255,0.06); }
    .market-box::before { content: ''; position: absolute; inset: 0;
                          background: radial-gradient(90% 120% at 100% 0%, rgba(52,211,153,0.16), transparent 60%); }
    .market-label { position: relative; font-size: 9px; font-weight: 700; color: rgba(209,250,229,0.7);
                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; letter-spacing: 0.06em;
                    text-transform: uppercase; }
    .market-value { position: relative; font-size: 21px; font-weight: 700; color: #ffffff; line-height: 1.25;
                    font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
    .market-change { position: relative; font-size: 9.5px; color: rgba(209,250,229,0.8); line-height: 1.55; }

    /* ─── Toolbar: density / columns / export — all <button>, no
           native form controls (report keeps exactly 3 checkboxes).
           position+z-index is load-bearing: the filling `rise` animation
           makes .toolbar a stacking context, which traps .tb-pop's own
           z-index — without this the dropdowns paint *under* .overview /
           .filters / .table-wrap (they come later in tree order). ── */
    .toolbar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-bottom: 12px;
               position: relative; z-index: 50;
               animation: rise 0.5s cubic-bezier(0.2,0.7,0.3,1) 0.04s both; }
    .tb-group { display: flex; gap: 4px; align-items: center; background: var(--glass);
                border: 1px solid var(--border); border-radius: 99px; padding: 4px;
                backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
                box-shadow: 0 6px 18px rgba(0,0,0,0.26); }
    .tb-btn { background: transparent; border: 1px solid transparent; color: var(--text-dim); cursor: pointer;
              padding: 5px 13px; border-radius: 99px; font-family: inherit; font-size: 11px; font-weight: 600;
              letter-spacing: 0.02em; transition: color .16s, background .16s, border-color .16s, transform .1s;
              white-space: nowrap; }
    .tb-btn:hover { color: var(--text); background: rgba(255,255,255,0.06); }
    .tb-btn:active { transform: scale(0.96); }
    .tb-btn.on { color: var(--green); background: rgba(52,211,153,0.14); border-color: rgba(52,211,153,0.42); }
    .tb-sep { width: 1px; align-self: stretch; background: var(--border); margin: 0 2px; }
    .tb-menu { position: relative; }
    .tb-pop { position: absolute; top: calc(100% + 8px); left: 0; z-index: 40; min-width: 190px;
              background: linear-gradient(180deg, var(--surface3), var(--surface2));
              border: 1px solid var(--border-light); border-radius: var(--radius-sm); padding: 8px;
              box-shadow: 0 18px 44px rgba(0,0,0,0.5); backdrop-filter: blur(16px);
              -webkit-backdrop-filter: blur(16px); display: none; }
    .tb-pop.open { display: block; animation: fadeUp .16s ease both; }
    .tb-pop h5 { margin: 2px 4px 7px; font-size: 9.5px; font-weight: 700; letter-spacing: 0.1em;
                 text-transform: uppercase; color: var(--text-faint); }
    .tb-opt { display: flex; width: 100%; align-items: center; gap: 8px; background: transparent; border: 0;
              color: var(--text-dim); cursor: pointer; padding: 6px 8px; border-radius: 7px;
              font-family: inherit; font-size: 11.5px; text-align: left; transition: background .14s, color .14s; }
    .tb-opt:hover { background: rgba(255,255,255,0.06); color: var(--text); }
    .tb-opt .dot { width: 7px; height: 7px; border-radius: 99px; background: var(--border-light); flex: none;
                   box-shadow: 0 0 0 1px rgba(0,0,0,0.3); transition: background .14s, box-shadow .14s; }
    .tb-opt.on { color: var(--green); }
    .tb-opt.on .dot { background: var(--green); box-shadow: 0 0 8px rgba(16,185,129,0.8); }
    .tb-kbd { margin-left: auto; font-family: var(--mono); font-size: 9.5px; color: var(--text-faint);
              background: rgba(0,0,0,0.35); border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; }

    /* ─── Overview: 4 compact stat cards in one strip ── */
    .overview { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; align-items: stretch; margin-bottom: 16px;
                animation: rise 0.5s cubic-bezier(0.2, 0.7, 0.3, 1) 0.08s both; }
    .summary { display: contents; }
    .stat { display: flex; align-items: center; gap: 11px; position: relative; overflow: hidden;
            background: linear-gradient(180deg, var(--surface2) 0%, var(--surface) 100%);
            border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 13px 10px 15px;
            box-shadow: 0 10px 26px rgba(0,0,0,0.32), inset 0 1px 0 rgba(255,255,255,0.05);
            backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease; }
    .stat:hover { transform: translateY(-2px); border-color: var(--border-light);
                  box-shadow: 0 16px 34px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.06); }
    .stat::after { content: ''; position: absolute; inset: 0; pointer-events: none;
                   background: radial-gradient(120% 200% at 100% -60%, rgba(255,255,255,0.07), transparent 60%); }
    .stat::before { content: ''; position: absolute; top: 0; bottom: 0; left: 0; width: 3px;
                    background: linear-gradient(180deg, var(--accent, var(--green)), transparent 92%); opacity: 0.95; }
    .stat.green { --accent: var(--green); } .stat.lime { --accent: var(--lime); } .stat.cyan { --accent: var(--cyan); } .stat.orange { --accent: var(--orange); } .stat.red { --accent: var(--red); }
    .stat .num { position: relative; flex: none; font-family: var(--mono); font-size: 1.75em; font-weight: 700; line-height: 1; text-align: left; color: var(--accent, var(--text));
                 font-variant-numeric: tabular-nums; letter-spacing: -0.04em;
                 text-shadow: 0 0 18px color-mix(in srgb, var(--accent, var(--green)) 45%, transparent); }
    .stat-meta { position: relative; display: flex; flex-direction: column; gap: 3px; min-width: 0; }
    .stat .label { color: var(--text-dim); font-size: 0.66em; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; }
    .stat .sub { color: var(--text-faint); font-size: 0.68em; letter-spacing: 0.01em; }
    .filters { margin-bottom: 14px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
               background: var(--glass); border: 1px solid var(--border); padding: 10px 14px; border-radius: var(--radius);
               backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
               box-shadow: 0 8px 24px rgba(0,0,0,0.28);
               animation: rise 0.5s cubic-bezier(0.2, 0.7, 0.3, 1) 0.14s both; }
    .filters input[type="text"] {
        background: rgba(0,0,0,0.34) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%2393ab9e' stroke-width='2.2' stroke-linecap='round'><path d='M11 11m-7.5 0a7.5 7.5 0 1 0 15 0 7.5 7.5 0 1 0-15 0M16.6 16.6L21 21'/></svg>") no-repeat 12px center;
        border: 1px solid var(--border-light); color: var(--text); width: 244px;
        padding: 8px 34px 8px 34px; border-radius: 99px; font-family: inherit; font-size: 0.84em;
        transition: border-color 0.2s, box-shadow 0.2s, width 0.2s;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.28);
    }
    .filters input[type="text"]::placeholder { color: var(--text-faint); }
    .filters input[type="text"]:focus { outline: none; border-color: var(--green); width: 280px;
                                        box-shadow: 0 0 0 3px var(--focus-ring), inset 0 2px 6px rgba(0,0,0,0.28); }
    .chip-group { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
    .chip-label { font-size: 0.62em; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-faint); margin-right: 2px; }
    .chip { background: rgba(0,0,0,0.28); border: 1px solid var(--border); color: var(--text-dim); cursor: pointer;
            padding: 5px 13px; border-radius: 99px; font-family: inherit; font-size: 0.74em; font-weight: 600;
            transition: color .15s, border-color .15s, background .15s, transform .1s, box-shadow .15s;
            box-shadow: 0 1px 2px rgba(0,0,0,0.24); }
    .chip:hover { color: var(--text); border-color: var(--border-light); transform: translateY(-1px);
                  box-shadow: 0 5px 14px rgba(0,0,0,0.34); }
    .chip:active { transform: scale(0.96); }
    .chip.active { background: linear-gradient(135deg, rgba(52,211,153,0.24), rgba(16,185,129,0.12));
                   border-color: rgba(52,211,153,0.55); color: var(--green);
                   box-shadow: 0 0 0 1px rgba(52,211,153,0.18), 0 5px 16px rgba(16,185,129,0.20); }
    input:focus-visible, .chip:focus-visible, a:focus-visible, .tb-btn:focus-visible, .tb-opt:focus-visible {
        outline: 2px solid var(--green); outline-offset: 2px; }
    /* App section header (views_layout.section_header): title, filter
       checkboxes, active chips, scanned/above count on the right */
    .results-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 10px 8px 6px;
                    animation: rise 0.5s cubic-bezier(0.2, 0.7, 0.3, 1) 0.18s both; }
    .results-title { font-size: 16px; font-weight: 700; color: var(--text); letter-spacing: -0.015em; }
    .results-title::before { content: '›'; color: var(--green); margin-right: 8px; font-weight: 400; }
    /* Toggles: pill chips with a custom check — keeps the native input
       (id/handler are load-bearing for tests + filterTable), only its look. */
    .cb { display: inline-flex; align-items: center; gap: 7px; font-size: 10.5px; font-weight: 600;
          color: var(--text-dim); cursor: pointer; background: rgba(0,0,0,0.26);
          border: 1px solid var(--border); padding: 5px 11px 5px 8px; border-radius: 99px;
          box-shadow: 0 1px 2px rgba(0,0,0,0.24);
          transition: color .15s, background .15s, border-color .15s, transform .1s, box-shadow .15s; }
    .cb:hover { color: var(--text); border-color: var(--border-light);
                box-shadow: 0 5px 14px rgba(0,0,0,0.34); transform: translateY(-1px); }
    .cb:active { transform: scale(0.97); }
    .cb:has(input:checked) { color: var(--green); background: rgba(52,211,153,0.12);
                             border-color: rgba(52,211,153,0.45);
                             box-shadow: 0 0 0 1px rgba(52,211,153,0.16), 0 5px 16px rgba(16,185,129,0.18); }
    .cb input { appearance: none; -webkit-appearance: none; width: 13px; height: 13px; margin: 0; flex: none;
                cursor: pointer; background: rgba(0,0,0,0.35); border: 1.5px solid var(--border-light);
                border-radius: 4px; transition: background .15s, border-color .15s, box-shadow .15s; }
    .cb:hover input { border-color: var(--green); }
    .cb input:checked { border-color: var(--green); box-shadow: 0 0 8px rgba(16,185,129,0.55);
                        background: var(--green) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='9' height='9' viewBox='0 0 24 24' fill='none' stroke='%23052e22' stroke-width='4.5' stroke-linecap='round' stroke-linejoin='round'><path d='M20 6 9 17l-5-5'/></svg>") no-repeat center; }
    .filter-chips { display: flex; gap: 6px; flex-wrap: wrap; }
    .filter-chip { background: rgba(52,211,153,0.10); border: 1px solid rgba(52,211,153,0.35); border-radius: 99px;
                   padding: 3px 10px; font-family: inherit; font-size: 10px; font-weight: 700; color: var(--green);
                   cursor: pointer; transition: color .15s, border-color .15s, background .15s, transform .1s; }
    .filter-chip:hover { color: #ffffff; border-color: rgba(52,211,153,0.7); background: rgba(52,211,153,0.20); transform: translateY(-1px); }
    .count-label { margin-left: auto; font-size: 11px; color: var(--text-dim); white-space: nowrap;
                   font-variant-numeric: tabular-nums; background: rgba(0,0,0,0.28); border: 1px solid var(--border);
                   border-radius: 99px; padding: 4px 12px; }
    .table-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
                  box-shadow: 0 20px 50px rgba(0,0,0,0.42), inset 0 1px 0 rgba(255,255,255,0.04);
                  animation: fadeIn 0.55s ease 0.22s both; }
    table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 11.5px; min-width: 1250px;
            font-variant-numeric: tabular-nums; }
    /* Header — app `_make_header_row`: card2 bar, 10px bold, cyan@55 until sorted */
    th { background: linear-gradient(180deg, #1f3229, var(--surface3)); color: rgba(45,212,191,0.62); padding: 0 11px; text-align: left; font-weight: 700;
         font-size: 10px; letter-spacing: 0.045em; text-transform: uppercase; border-bottom: 1px solid var(--border-light); cursor: pointer; user-select: none;
         position: sticky; top: 0; z-index: 3; white-space: nowrap; height: 36px; transition: color 0.15s, background 0.15s;
         backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); }
    th::after { content: " ▲▼"; opacity: 0.35; font-size: 8px; }
    /* Frozen identity columns: header + rank/ticker stay put on h-scroll */
    th:first-child { left: 0; z-index: 4; border-top-left-radius: var(--radius); }
    th:last-child { border-top-right-radius: var(--radius); }
    th:hover { color: var(--cyan); background: linear-gradient(180deg, #24382e, var(--surface3)); }
    th:hover::after { opacity: 0.7; }
    th.sorted-asc, th.sorted-desc { color: var(--green); }
    th.sorted-asc::after { content: " ▲"; opacity: 1; }
    th.sorted-desc::after { content: " ▼"; opacity: 1; }
    /* Rows — app `_create_row_controls`: lines, rating rail, odd-rank zebra,
       card bg when the score clears the threshold, card2 on hover */
    td { padding: 0 11px; border-bottom: 1px solid var(--row-line); vertical-align: middle; height: var(--row-h); }
    tbody tr:not(.news-row) td { white-space: nowrap; }
    tbody tr { transition: background 0.15s, box-shadow 0.15s; }
    tbody tr:not(.news-row) { background: var(--bg); }
    tbody tr:not(.news-row).row-alt { background: var(--row-alt); }
    tbody tr.highlight:not(.news-row) { background: var(--surface2); }
    tbody tr.highlight:not(.news-row) td { border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
    tbody tr:not(.news-row):hover { background: var(--surface3); box-shadow: inset 3px 0 0 rgba(52,211,153,0.55); }
    tbody tr:not(.news-row):hover td { border-bottom-color: var(--border-light); }
    tbody tr:last-child td:first-child { border-bottom-left-radius: var(--radius); }
    tbody tr:last-child td:last-child { border-bottom-right-radius: var(--radius); }
    /* Sticky first two columns — 19 columns need ~1750px, so the identity
       cells (rank rail + ticker) must not scroll away from the row. */
    td.rank { position: sticky; left: 0; z-index: 2; background: inherit; width: 52px; color: var(--text-faint); padding-left: 12px;
              box-shadow: 1px 0 0 var(--row-line); }
    td.rank::before { content: ''; position: absolute; left: 0; top: 8px; bottom: 8px; width: 3px; border-radius: 0 2px 2px 0;
                      background: var(--rail, transparent); box-shadow: 0 0 10px var(--rail, transparent); }
    td.ticker { position: sticky; left: 52px; z-index: 2; background: inherit; box-shadow: 1px 0 0 var(--row-line);
                padding-right: 26px; }  /* room for the absolute copy button */
    .ticker { color: var(--text); font-weight: 700; cursor: pointer; font-size: 12.5px; letter-spacing: 0.01em; }
    .ticker.above { color: var(--green); text-shadow: 0 0 14px rgba(16,185,129,0.35); }
    .ticker:hover { color: var(--ticker-hover-c); text-decoration: none; background: var(--ticker-hover-bg); border-radius: 5px; }
    /* Copy pinned to the ticker cell's right edge, vertically centred —
       app parity (Row: expand ticker + badge + copy). Absolute works because
       td.ticker is position: sticky, i.e. already a positioned ancestor. */
    .copy-t { position: absolute; right: 6px; top: 0; bottom: 0; display: inline-flex; align-items: center;
              background: none; border: 0; padding: 0; color: var(--text-faint); font-size: 10px; cursor: pointer;
              transition: color .14s, transform .14s; }
    .copy-t:hover { color: var(--green); transform: scale(1.18); }
    /* Score — colour-graded pill so strength reads at a glance */
    .score { font-size: 14.5px; font-weight: 800; color: var(--red); font-family: var(--mono); letter-spacing: -0.03em; }
    .score.s-excellent { color: var(--green); text-shadow: 0 0 16px rgba(16,185,129,0.45); }
    .score.s-good { color: var(--lime); text-shadow: 0 0 14px rgba(163,230,53,0.35); }
    .score.s-moderate { color: var(--orange); }
    .rating { font-weight: 700; color: var(--red); border-radius: 7px; letter-spacing: 0.03em; font-size: 10.5px; }
    .rating.r-excellent { color: var(--green); background: linear-gradient(135deg, rgba(16,185,129,0.22), rgba(16,185,129,0.08)); box-shadow: 0 0 0 1px rgba(16,185,129,0.28); }
    .rating.r-good { color: var(--lime); background: linear-gradient(135deg, rgba(163,230,53,0.18), rgba(163,230,53,0.06)); box-shadow: 0 0 0 1px rgba(163,230,53,0.24); }
    .rating.r-moderate { color: var(--orange); background: linear-gradient(135deg, rgba(251,146,60,0.20), rgba(251,146,60,0.07)); box-shadow: 0 0 0 1px rgba(251,146,60,0.26); }
    .rating.r-poor { color: var(--red); background: linear-gradient(135deg, rgba(248,113,113,0.20), rgba(248,113,113,0.07)); box-shadow: 0 0 0 1px rgba(248,113,113,0.26); }
    .entry { border-radius: 7px; color: var(--text-dim); font-weight: 600; letter-spacing: 0.04em; }
    /* Glossy raised chip: top highlight + bottom shade beat a flat slab for depth. */
    .entry.yes { color: #04150f; font-weight: 800; letter-spacing: 0.07em;
                 background: linear-gradient(180deg, #6ee7b7 0%, #34d399 42%, #10b981 100%);
                 box-shadow: inset 0 1px 0 rgba(255,255,255,0.55), inset 0 -3px 8px rgba(4,21,15,0.20),
                             0 4px 14px rgba(16,185,129,0.32); }
    .price { font-size: 12.5px; font-weight: 700; font-family: var(--mono); letter-spacing: -0.02em; }
    .bull { color: var(--green); font-weight: 600; }
    .bear { color: var(--red); font-weight: 600; }
    .sc-orange { color: var(--orange); }
    .sc-lime { color: var(--lime); }
    .sc-fund { color: #ffe600; }
    .rval-ok { color: var(--green); }
    .rval-off { color: var(--orange); }
    .rval-none { color: var(--text-faint); }
    .sideways { color: var(--orange); font-weight: 600; }
    .trending { color: var(--green); font-weight: 600; }
    .ma-cross { color: var(--green); }
    .ma-bull { color: var(--lime); }
    .ma-bear { color: var(--red); }
    .poc-above { color: var(--green); }
    .poc-below { color: var(--red); }
    .bothma-yes { color: var(--green); }
    .bothma-no { color: var(--text-dim); }
    .footer { margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border); color: var(--text-faint);
              font-size: 0.72em; letter-spacing: 0.04em; text-align: center; }

    /* ─── News panel styles ─────────────────────────────── */
    .news-row td { padding: 0; border-bottom: 1px solid var(--border); }
    .news-panel {
        background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
        border-left: 3px solid var(--cyan);
        padding: 14px 18px;
        margin: 6px 14px 10px 42px;
        border-radius: var(--radius-sm);
        box-shadow: 0 14px 34px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative; overflow: hidden;
    }
    .news-panel::after { content: ''; position: absolute; inset: 0; pointer-events: none;
                         background: radial-gradient(140% 200% at 0% 0%, rgba(20,184,166,0.10), transparent 55%); }
    .news-summary-line {
        color: var(--text);
        font-size: 0.85em;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
        position: relative;
    }
    .news-item {
        padding: 7px 0;
        border-bottom: 1px solid var(--news-line);
        position: relative;
    }
    .news-item:last-child { border-bottom: none; }
    .news-sentiment {
        display: inline-block;
        font-weight: 700;
        font-size: 0.7em;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-right: 7px;
        padding: 2px 7px;
        border-radius: 99px;
    }
    .news-sentiment.good { color: var(--green); background: rgba(16,185,129,0.16); box-shadow: 0 0 0 1px rgba(16,185,129,0.3); }
    .news-sentiment.bad { color: var(--red); background: rgba(248,113,113,0.16); box-shadow: 0 0 0 1px rgba(248,113,113,0.3); }
    .news-sentiment.neutral { color: var(--text-dim); background: rgba(255,255,255,0.06); box-shadow: 0 0 0 1px var(--border); }
    .news-date { color: var(--text-dim); font-size: 0.78em; margin-right: 8px; font-variant-numeric: tabular-nums; }
    .news-pub { color: var(--cyan); font-size: 0.78em; }
    .news-title { color: var(--text); font-size: 0.86em; margin-top: 4px; font-weight: 700; }
    .news-summary { color: var(--text-dim); font-size: 0.78em; margin-top: 3px; line-height: 1.5; }
    .news-good { color: var(--green); font-weight: 700; }
    .news-bad { color: var(--red); font-weight: 700; }
    .news-neutral { color: var(--text-dim); font-weight: 700; }

    /* ─── Trade reasons ────────────────────────────────────── */
    .reason-item {
        color: var(--text);
        font-size: 0.82em;
        padding: 4px 0;
        line-height: 1.45;
    }
    .reason-icon {
        font-weight: 700;
        margin-right: 5px;
        color: var(--green);
    }

    /* ─── Stock detail (mirrors the app's detail panel) ─── */
    .detail-grid { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 16px; margin-bottom: 12px; }
    .detail-col { display: flex; flex-direction: column; gap: 12px; }
    .detail-card { background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
                   border: 1px solid var(--border); border-radius: var(--radius); padding: 11px 13px 13px;
                   box-shadow: 0 14px 34px rgba(0,0,0,0.36), inset 0 1px 0 rgba(255,255,255,0.05); }
    .detail-title { color: var(--cyan); font-weight: 700; font-size: 0.8em; margin: 0 0 9px; padding-bottom: 7px;
                    border-bottom: 1px solid var(--border); letter-spacing: 0.05em; text-transform: uppercase; }
    .axis-lbl { font-size: 10px; fill: var(--text-faint); font-family: var(--mono); }
    .chart-box { position: relative; background: rgba(0,0,0,0.30); border: 1px solid var(--border);
                 border-radius: var(--radius-sm); padding: 8px; box-shadow: inset 0 2px 8px rgba(0,0,0,0.3); }
    .chart-hi, .chart-lo { position: absolute; right: 10px; font-size: 0.7em; color: var(--text-dim); font-family: var(--mono); }
    .chart-hi { top: 6px; }
    .chart-lo { bottom: 6px; }
    .signals { display: flex; flex-wrap: wrap; gap: 7px; }
    .signal { background: rgba(0,0,0,0.26); border: 1px solid var(--border); border-radius: var(--radius-sm);
              padding: 6px 11px; min-width: 66px; box-shadow: 0 5px 16px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.04);
              transition: border-color .16s, transform .16s; }
    .signal:hover { border-color: var(--border-light); transform: translateY(-1px); }
    .signal-label { display: block; font-size: 0.62em; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.07em; }
    .signal-val { font-weight: 700; font-size: 1em; color: var(--text); font-variant-numeric: tabular-nums; }
    .bd-row { display: grid; grid-template-columns: 86px minmax(0,1fr) 52px; gap: 8px; align-items: center; font-size: 0.78em; color: var(--text-dim); margin-bottom: 6px; }
    .bd-track { height: 8px; background: var(--track); border-radius: 99px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.4); }
    .bd-fill { display: block; height: 100%; border-radius: 99px; box-shadow: 0 0 10px rgba(16,185,129,0.5); }
    .bd-val { text-align: right; color: var(--text); font-family: var(--mono); font-variant-numeric: tabular-nums; }
    .inst-tiles { display: flex; flex-wrap: wrap; gap: 7px; }
    .inst-tile { background: rgba(0,0,0,0.26); border: 1px solid var(--border); border-radius: var(--radius-sm);
                 padding: 7px 11px; box-shadow: 0 5px 16px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.04); }
    .inst-label { display: block; font-size: 0.62em; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.07em; }
    .inst-val { font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }
    .inst-sub { display: block; font-size: 0.7em; }
    .inst-hint { font-size: 0.7em; color: var(--text-faint); margin-top: 5px; }
    .news-more { display: inline-block; color: var(--cyan); font-size: 0.8em; margin-top: 8px; text-decoration: none;
                 font-weight: 600; transition: color .14s; }
    .news-more:hover { color: var(--green); text-decoration: underline; }

    /* ─── Sparkline column ─────────────────────────────────── */
    .spark-cell { padding: 4px 7px; }

    /* ─── Content-fit: JS hides FIT_ORDER columns until the table
           fits its container (viewport tiers alone can't — 23
           columns need ~2200px at full width). ───────────────── */
    .fit-hidden { display: none; }
    /* User column menu uses its own class so it can't fight fitColumns(). */
    .user-hidden { display: none; }

    /* ─── Density: Comfort (default) / Compact row rhythm ──── */
    body.dense { --row-h: 27px; font-size: 13.5px; }
    body.dense table { font-size: 11px; }
    body.dense .score { font-size: 13px; }
    body.dense .ticker { font-size: 11.5px; }
    body.dense .stat { padding: 7px 11px 7px 13px; }
    body.dense .stat .num { font-size: 1.2em; }
    body.dense .detail-card { padding: 9px 11px 11px; }

    /* ─── Scroll-to-top pill ───────────────────────────────── */
    #toTop { position: fixed; right: 22px; bottom: 22px; z-index: 60;
             background: linear-gradient(135deg, rgba(16,185,129,0.95), rgba(13,148,136,0.95));
             color: #04150e; border: 0; border-radius: 99px; width: 42px; height: 42px; cursor: pointer;
             font-size: 17px; font-weight: 800; box-shadow: 0 12px 30px rgba(16,185,129,0.45);
             opacity: 0; visibility: hidden; transform: translateY(12px) scale(0.9);
             transition: opacity .22s, transform .22s, visibility .22s; }
    #toTop.show { opacity: 1; visibility: visible; transform: none; }
    #toTop:hover { transform: translateY(-2px); box-shadow: 0 16px 36px rgba(16,185,129,0.55); }

    /* ─── Toast (copy / export feedback) ───────────────────── */
    #toast { position: fixed; left: 50%; bottom: 26px; z-index: 70; transform: translate(-50%, 20px);
             background: linear-gradient(180deg, var(--surface3), var(--surface2)); color: var(--text);
             border: 1px solid var(--border-light); border-radius: 99px; padding: 9px 20px;
             font-size: 12px; font-weight: 600; box-shadow: 0 16px 40px rgba(0,0,0,0.5);
             opacity: 0; visibility: hidden; transition: opacity .2s, transform .2s, visibility .2s;
             backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); }
    #toast.show { opacity: 1; visibility: visible; transform: translate(-50%, 0); }

    /* ─── Search hit highlight in the ticker cell ──────────── */
    mark.hit { background: rgba(250,204,21,0.28); color: inherit; border-radius: 3px; padding: 0 1px;
               box-shadow: 0 0 0 1px rgba(250,204,21,0.35); }

    /* ─── Entrance + expand motion (staggered rise for sections),
           reduced-motion guard below switches everything off ──── */
    @keyframes fadeUp { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: none; } }
    @keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes popIn { 0% { opacity: 0; transform: scale(0.94); } 100% { opacity: 1; transform: scale(1); } }
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
        .results-head { gap: 8px; }
    }
    @media (max-width: 860px) {
        .overview { grid-template-columns: repeat(2, minmax(0,1fr)); }
        .toolbar { gap: 8px; }
    }
    @media (max-width: 720px) {
        body { padding: 12px; }
        .filters input[type="text"], .filters input[type="text"]:focus { width: 100%; }
        .hero { flex-wrap: wrap; }
        .market-box { display: none; }
        .count-label { margin-left: 0; }
        .tb-pop { left: auto; right: 0; }
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
        .toolbar, #toTop, #toast { display: none; }
        .filters { display: none; }
        th { position: static; background: var(--surface2); backdrop-filter: none; -webkit-backdrop-filter: none; }
        td.rank, td.ticker { position: static; }
        td.rank, td.ticker { box-shadow: none; }
        .copy-t { display: none; }
        .table-wrap { border: none; }
        .hero-pill, .filters { background: var(--surface); backdrop-filter: none; -webkit-backdrop-filter: none; }
        * { animation: none !important; transition: none !important; }
        th.fit-hidden, td.fit-hidden { display: table-cell; }  /* full table on paper */
        th.user-hidden, td.user-hidden { display: table-cell; }  /* and every picked column */
        .stat, .table-wrap, .news-panel, .chart-box, .signal, .inst-tile, .detail-card,
        .count-label, .meta span, .market-box { box-shadow: none; }
        .rating, .entry.yes, .filter-chip, .news-sentiment { box-shadow: none !important; background: transparent !important; }
        a { color: inherit; text-decoration: none; }
        .footer { margin-top: 14px; }
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

    // First click seeds the app's default direction (desc, except #/Ticker);
    // switching columns resets the others, mirroring app _on_sort.
    if (sortDir[col] === undefined) {
        sortDir[col] = (col === 0 || col === 1) ? "asc" : "desc";
    } else {
        sortDir[col] = sortDir[col] === "asc" ? "desc" : "asc";
    }
    const dir = sortDir[col];

    for (let i = 0; i < table.tHead.rows[0].cells.length; i++) {
        const cell = table.tHead.rows[0].cells[i];
        cell.classList.remove("sorted-asc", "sorted-desc");
        if (i !== col) delete sortDir[i];
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

    const ratingOrder = {"EXCELLENT":4,"GOOD":3,"MODERATE":2,"POOR":1,"WEAK":0};
    const isNum = t => /^[-+]?[\u20b9$]?[0-9,]+(?:[.][0-9]+)?[ ]*%?$/.test(t);
    const flip = dir === "asc" ? 1 : -1;

    pairs.sort((a, b) => {
        let aCell = a.dataRow.cells[col];
        let bCell = b.dataRow.cells[col];
        let aVal = aCell ? aCell.textContent.trim() : "";
        let bVal = bCell ? bCell.textContent.trim() : "";
        let aText = aVal.replace(/<[^>]*>/g, "").trim();
        let bText = bVal.replace(/<[^>]*>/g, "").trim();

        // Semantic columns mirror app _get_sort_key — never the cell text.
        if (col === 3 || col === 6 || col === 7 || col === 14 || col === 17 || col === 18) {
            let aR, bR;
            if (col === 3) {  // Rating: custom order, not alphabetical
                aR = ratingOrder[aText.toUpperCase()] || 0;
                bR = ratingOrder[bText.toUpperCase()] || 0;
            } else if (col === 6) {  // MA: fresh crossover > bullish > bearish
                const dA = a.dataRow.dataset, dB = b.dataRow.dataset;
                aR = dA.crossed === "true" ? 200 - (Number(dA.cbars) || 0) : (dA.maBull === "true" ? 1 : 0);
                bR = dB.crossed === "true" ? 200 - (Number(dB.cbars) || 0) : (dB.maBull === "true" ? 1 : 0);
            } else if (col === 7) {  // POC: Above = 1 (lexical A<B would invert desc)
                aR = aCell && aCell.classList.contains("poc-above") ? 1 : 0;
                bR = bCell && bCell.classList.contains("poc-above") ? 1 : 0;
            } else if (col === 14) {  // Dir: Bull = 1
                aR = aCell && aCell.classList.contains("bull") ? 1 : 0;
                bR = bCell && bCell.classList.contains("bull") ? 1 : 0;
            } else if (col === 17) {  // Chop: sideways = 1
                aR = aCell && aCell.classList.contains("sideways") ? 1 : 0;
                bR = bCell && bCell.classList.contains("sideways") ? 1 : 0;
            } else {  // Spark (18): drawn % move
                aR = Number(a.dataRow.dataset.spark) || 0;
                bR = Number(b.dataRow.dataset.spark) || 0;
            }
            return (aR - bR) * flip;
        }

        // Whole-cell numeric (₹, %, commas). Text like "3MINDIA" fails the
        // test and stays alphabetical; N/A and — always sort after numbers,
        // in both directions.
        const aOk = isNum(aText), bOk = isNum(bText);
        if (aOk !== bOk) return aOk ? -1 : 1;
        if (aOk) {
            let aNum = parseFloat(aText.replace(/[^0-9.-]/g, ""));
            let bNum = parseFloat(bText.replace(/[^0-9.-]/g, ""));
            if (!isNaN(aNum) && !isNaN(bNum)) return (aNum - bNum) * flip;
        }
        return aText.localeCompare(bText) * flip;
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
        const trend = row.cells[14].textContent;  // Dir cell (^ Bull / v Bear)
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

/* ── Toolbar: density / column menu ──────────────────────────── */
function setDensity(mode) {
    const compact = mode === "compact";
    document.body.classList.toggle("dense", compact);
    const a = document.getElementById("tb-dens-comfort");
    const b = document.getElementById("tb-dens-compact");
    if (a) a.classList.toggle("on", !compact);
    if (b) b.classList.toggle("on", compact);
    try { localStorage.setItem("rep-dens", mode); } catch (e) {}
    relayout();
}
function closePops() {
    document.querySelectorAll(".tb-pop.open").forEach(p => p.classList.remove("open"));
    document.querySelectorAll(".tb-menu > .tb-btn.on").forEach(b => b.classList.remove("on"));
}
function togglePop(id, btn) {
    const pop = document.getElementById(id);
    if (!pop) return;
    const willOpen = !pop.classList.contains("open");
    closePops();
    if (willOpen) {
        pop.classList.add("open");
        if (btn) btn.classList.add("on");
    }
}
document.addEventListener("click", (e) => { if (!e.target.closest(".tb-menu")) closePops(); });
// user-hidden is independent of fit-hidden: the column menu is what the user
// picked, fitColumns() is what overflow forced — keep the two orthogonal.
function toggleCol(i, btn) {
    const show = !btn.classList.contains("on");
    btn.classList.toggle("on", show);
    const table = document.getElementById("stockTable");
    const th = table.tHead.rows[0].cells[i];
    if (th) th.classList.toggle("user-hidden", !show);
    table.querySelectorAll(`tbody td:nth-child(${i + 1})`).forEach(td => td.classList.toggle("user-hidden", !show));
    syncColspan();
}
function showAllCols() {
    document.querySelectorAll(".tb-opt[data-col]").forEach(b => b.classList.add("on"));
    document.querySelectorAll(".user-hidden").forEach(e => e.classList.remove("user-hidden"));
    syncColspan();
}

/* ── Toast ──────────────────────────────────────────────────── */
let toastTimer;
function toast(msg) {
    let t = document.getElementById("toast");
    if (!t) { t = document.createElement("div"); t.id = "toast"; document.body.appendChild(t); }
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), 1900);
}

/* ── CSV export of the rows currently on screen ─────────────── */
function exportCSV() {
    const table = document.getElementById("stockTable");
    const head = table.tHead.rows[0];
    const cols = [];
    for (let i = 0; i < head.cells.length; i++) if (head.cells[i].offsetParent !== null) cols.push(i);
    const esc = v => '"' + String(v).replace(/\\s+/g, " ").replace(/"/g, '""').trim() + '"';
    const lines = [cols.map(i => esc(head.cells[i].textContent)).join(",")];
    for (const r of table.tBodies[0].rows) {
        if (r.classList.contains("news-row") || r.offsetParent === null) continue;
        lines.push(cols.map(i => esc(r.cells[i] ? r.cells[i].textContent : "")).join(","));
    }
    const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "");
    const blob = new Blob(["\\uFEFF" + lines.join("\\r\\n")], { type: "text/csv;charset=utf-8;" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "scanner_report_" + stamp + ".csv";
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
    toast("Exported " + (lines.length - 1) + " rows");
}

/* ── Scroll-to-top pill ─────────────────────────────────────── */
const toTop = document.createElement("button");
toTop.id = "toTop";
toTop.type = "button";
toTop.title = "Back to top";
toTop.textContent = "\\u2191";
toTop.onclick = () => window.scrollTo({ top: 0, behavior: "smooth" });
document.body.appendChild(toTop);
window.addEventListener("scroll", () => {
    toTop.classList.toggle("show", window.scrollY > 460);
}, { passive: true });

/* ── Row focus + keyboard shortcuts ─────────────────────────── */
let focusIdx = -1;
function moveFocus(dir) {
    const rows = Array.from(document.querySelectorAll("#stockTable tbody tr:not(.news-row)"))
        .filter(r => r.offsetParent !== null);
    if (!rows.length) return;
    rows.forEach(r => r.classList.remove("highlight"));
    focusIdx = Math.max(0, Math.min(rows.length - 1, focusIdx + dir));
    rows[focusIdx].classList.add("highlight");
    rows[focusIdx].scrollIntoView({ block: "center", behavior: "smooth" });
}
document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    const typing = tag === "input" || tag === "textarea" || tag === "select";
    if (e.key === "Escape") { closePops(); if (typing) e.target.blur(); return; }
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "/") { e.preventDefault(); document.getElementById("search").focus(); return; }
    const k = e.key.toLowerCase();
    if (k === "d") { setDensity(document.body.classList.contains("dense") ? "compact" : "comfort"); return; }
    if (k === "c") { const b = document.querySelector("#pop-cols").previousElementSibling; togglePop("pop-cols", b); e.preventDefault(); return; }
    if (k === "e") { exportCSV(); return; }
    if (/^[1-9]$/.test(e.key)) { const c = parseInt(e.key, 10) - 1;
        if (document.getElementById("stockTable").tHead.rows[0].cells[c]) sortTable(c); return; }
    if (e.key === "ArrowDown") { e.preventDefault(); moveFocus(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); moveFocus(-1); }
});

/* ── Restore stored prefs after the static markup has painted ── */
try {
    if (localStorage.getItem("rep-dens") === "compact") setDensity("compact");
} catch (e) {}

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


def _toolbar_html() -> str:
    """Control bar under the hero: density, column menu, CSV, keys.

    Every control is a <button> — the report contract allows exactly three
    checkboxes, and those all live in the results-head filter row.
    """
    # Index 0 (#) and 1 (Ticker) are locked: the sticky left columns rely on
    # fixed offsets, and losing the identity column makes rows unreadable.
    col_opts = "".join(
        f'<button type="button" class="tb-opt on" data-col="{i}" '
        f'onclick="toggleCol({i},this)">'
        f'<span class="dot"></span>{_html.escape(label)}</button>'
        for i, (label, _tier) in enumerate(_REPORT_COLS)
        if i >= 2
    )
    keys = (
        ("/", "Focus search"),
        ("↑ ↓", "Move row focus"),
        ("1 – 9", "Sort by column"),
        ("D", "Comfort / compact"),
        ("C", "Column menu"),
        ("E", "Export CSV"),
        ("Esc", "Close popup"),
    )
    # Each row shows the action then its key badge.
    key_rows = "".join(
        f'<button type="button" class="tb-opt" style="cursor:default">{_html.escape(label)}'
        f'<span class="tb-kbd">{_html.escape(k)}</span></button>'
        for k, label in keys
    )
    return f"""<div class="toolbar" role="toolbar" aria-label="Report controls">
  <div class="tb-group" role="group" aria-label="Row density">
    <button type="button" class="tb-btn on" id="tb-dens-comfort" onclick="setDensity('comfort')">Comfort</button>
    <button type="button" class="tb-btn" id="tb-dens-compact" onclick="setDensity('compact')">Compact</button>
  </div>
  <span class="tb-sep"></span>
  <div class="tb-menu">
    <button type="button" class="tb-btn" onclick="togglePop('pop-cols',this)">Columns ▾</button>
    <div class="tb-pop" id="pop-cols">
      <h5>Visible columns</h5>
      {col_opts}
      <button type="button" class="tb-opt" onclick="showAllCols()"><span class="dot"></span>Show all</button>
    </div>
  </div>
  <button type="button" class="tb-btn" onclick="exportCSV()" title="Download the visible rows as CSV">Export CSV</button>
  <div class="tb-menu">
    <button type="button" class="tb-btn" onclick="togglePop('pop-keys',this)" title="Keyboard shortcuts">Shortcuts</button>
    <div class="tb-pop" id="pop-keys">
      <h5>Keyboard</h5>
      {key_rows}
    </div>
  </div>
</div>"""


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

{_toolbar_html()}

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
        # Every header sorts positionally, spark included — parity with the
        # app grid, which sorts col 18 by the drawn move (spark_move).
        ths.append(f'<th{cls} onclick="sortTable({i})">{label}</th>')
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
                    <td colspan="19">
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
    fund_pts = r.get("fundamentals", 0) or 0
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
            data-fii="{"1" if has_fii_dii(r) else "0"}"
            data-cbars="{r.get("crossover_bars_ago") or 0}"
            data-spark="{spark_move(r):g}">
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
            <td class="num sc-fund c-t2">{"N/A" if not fund_pts else f"{fund_pts:.0f}"}</td>
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
