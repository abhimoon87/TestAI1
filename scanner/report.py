"""
HTML report generator for stock scanner results.
Produces a sortable, filterable table with color-coded scores and news sentiment.
"""

import html as _html
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ─── Sentiment keywords ──────────────────────────────────────────────────────
SENTIMENT_GOOD = frozenset([
    "profit", "growth", "record", "gain", "surge", "rally",
    "strong", "beat", "bullish", "outperform", "order", "deal",
    "buy", "upgrade", "partner", "expand", "launch", "innovate",
    "dividend", "revenue", "acquire", "breakout", "resilient",
    "optimistic", "recovery", "momentum", "approval", "milestone",
    "boom", "soar", "jump", "climb",
])

SENTIMENT_BAD = frozenset([
    "loss", "decline", "crash", "drop", "fall", "weak",
    "bearish", "underperform", "sell", "downgrade", "fraud",
    "lawsuit", "investigation", "debt", "recession", "warning",
    "cut", "slump", "miss", "risk", "concern", "delay",
    "ban", "penalty", "probe", "resign", "volatile", "crisis",
    "shortage", "slowdown", "shrink", "tumble",
])


def _sentiment(title: str, summary: str = "") -> str:
    """Simple keyword-based sentiment: Good / Bad / Neutral."""
    text = (title + " " + summary).lower()
    # Split on whitespace and hyphens to catch hyphenated words
    import re
    words = set(re.split(r'[\s\-]+', text))
    g = len(words & SENTIMENT_GOOD)
    b = len(words & SENTIMENT_BAD)
    if g > b:
        return "Good"
    elif b > g:
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


def fetch_stock_news(ticker: str, max_items: int = 10,
                     months_back: int = 2) -> list:
    """
    Fetch recent news for a stock from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g. 'RELIANCE.NS')
        max_items: Maximum news items to return
        months_back: Only include news from the last N months

    Returns:
        List of dicts with 'title', 'summary', 'date', 'publisher', 'sentiment'
    """
    try:
        import yfinance as yf
        # Auto-append .NS suffix for Indian stocks if not present
        yf_ticker = ticker if any(ticker.endswith(s) for s in ('.NS', '.BO', '.NSE', '.BSE')) else f'{ticker}.NS'
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
            results.append({
                "title": title,
                "summary": summary,
                "date": dt.strftime("%Y-%m-%d") if dt else "—",
                "publisher": provider,
                "sentiment": _sentiment(title, summary),
            })
            if len(results) >= max_items:
                break
        return results
    except Exception as e:
        logger.debug("News fetch failed for %s: %s", ticker, e)
        return []


def _fetch_news_parallel(tickers: list[str], max_items: int = 10,
                         months_back: int = 2,
                         max_workers: int = 8,
                         fetch_fn=None) -> dict[str, list]:
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
                    logger.debug("Parallel news fetch failed for %s: %s", ticker, e)
                    news_map[ticker] = []
    except Exception as e:
        logger.debug("ThreadPoolExecutor failed: %s", e)
        # Fallback: sequential fetch
        for t in tickers:
            news_map[t] = fetch_fn(t, max_items, months_back)

    return news_map


def fetch_news_for_ticker(ticker: str, max_items: int = 10,
                          months_back: int = 2) -> list:
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


def fetch_news_batch(tickers: list[str], max_items: int = 10,
                     months_back: int = 2,
                     max_workers: int = 6) -> dict[str, list]:
    """Parallel batch version of :func:`fetch_news_for_ticker`.

    Returns a dict mapping each input ticker to its provider-keyed news list
    (``[]`` when nothing was found or the fetch failed).
    """
    return _fetch_news_parallel(
        tickers, max_items=max_items, months_back=months_back,
        max_workers=max_workers, fetch_fn=fetch_news_for_ticker,
    )


def generate_html_report(results: list, title: str = "HMAxEMA Stock Scanner",
                         threshold: float = 50.0,
                         fetch_news: bool = True) -> str:
    """
    Generate a complete HTML report from scan results.

    Args:
        results: List of dicts with 'ticker', 'total', scores, and metadata
        title: Report title
        threshold: Minimum score threshold
        fetch_news: Whether to fetch news sentiment for each stock

    Returns:
        Complete HTML string
    """
    # Sort by total score descending (non-mutating)
    results = sorted(results, key=lambda x: x["total"], reverse=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    passed = [r for r in results if r["total"] >= threshold]
    failed = [r for r in results if r["total"] < threshold]

    # ── Pre-fetch news for all tickers in parallel ───────────────────────
    news_map: dict[str, list] = {}
    if fetch_news:
        tickers = [r["ticker"] for r in results]
        news_map = _fetch_news_parallel(tickers)

    rows_html = ""
    for r in results:
        score = r["total"]
        ticker = r["ticker"]

        # Use combined rating if available, else fall back to score-based
        combined_rating = r.get('combined_rating', None)
        if combined_rating:
            rating_lower = _html.escape(combined_rating.lower())
            badge = f'<span class="badge {rating_lower}">{_html.escape(combined_rating)}</span>'
        else:
            if score >= 70:
                badge = '<span class="badge excellent">EXCELLENT</span>'
            elif score >= 50:
                badge = '<span class="badge good">GOOD</span>'
            elif score >= 30:
                badge = '<span class="badge moderate">MODERATE</span>'
            else:
                badge = '<span class="badge poor">POOR</span>'

        trend_icon = "▲" if r["trend_dir"] == "Bull" else "▼"
        trend_class = "bull" if "bull" in r["trend_color"] else "bear"
        rs_icon = "+" if (r.get("pc1m", 0) or 0) > 0 else ""

        # MA signal
        ma_bullish = r.get('ma_bullish', False)
        ma_crossed = r.get('ma_crossed_above', False)
        crossover_ago = r.get('crossover_bars_ago', -1)
        if ma_crossed:
            freshness_cls = 'fresh' if crossover_ago <= 2 else 'stale'
            ma_html = f'<span class="ma-cross">^ CROSS</span> <span class="{freshness_cls}">({crossover_ago} bars)</span>'
        elif ma_bullish:
            ma_html = '<span class="ma-bull">^ BULL</span>'
        else:
            ma_html = '<span class="ma-bear">v BEAR</span>'

        # POC signal
        above_poc = r.get('above_poc', False)
        vp_poc = r.get('vp_poc', 0)
        if above_poc:
            poc_html = f'<span class="poc-above">ABOVE</span> <span style="color:var(--text-dim);font-size:0.8em">{vp_poc}</span>'
        else:
            poc_html = f'<span class="poc-below">BELOW</span> <span style="color:var(--text-dim);font-size:0.8em">{vp_poc}</span>'

        # Close above both MAs
        close_above_both = r.get('close_above_both_ma', False)
        if close_above_both:
            bothma_html = '<span class="bothma-yes">YES</span>'
        else:
            bothma_html = '<span class="bothma-no">NO</span>'

        sideways = r.get('is_sideways', False)
        sideways_cls = 'sideways' if sideways else 'trending'
        sideways_reasons = _html.escape(', '.join(r.get('sideways_reasons', [])))
        sideways_label = '⚠ Chop' if sideways else '✓ Trend'

        # ─── News sentiment (pre-fetched in parallel) ─────────────────────
        news_html = ""
        if fetch_news:
            news_items = news_map.get(ticker, [])
            if news_items:
                good_count = sum(1 for n in news_items if n["sentiment"] == "Good")
                bad_count = sum(1 for n in news_items if n["sentiment"] == "Bad")
                neutral_count = sum(1 for n in news_items if n["sentiment"] == "Neutral")
                summary_parts = []
                if good_count:
                    summary_parts.append(f'<span class="news-good">{good_count} Good</span>')
                if bad_count:
                    summary_parts.append(f'<span class="news-bad">{bad_count} Bad</span>')
                if neutral_count:
                    summary_parts.append(f'<span class="news-neutral">{neutral_count} Neutral</span>')

                news_rows = ""
                for n in news_items:
                    sent_cls = _html.escape(n["sentiment"].lower())
                    safe_title = _html.escape(n["title"])
                    safe_summary = _html.escape(n["summary"][:200])
                    safe_publisher = _html.escape(n["publisher"])
                    safe_date = _html.escape(n["date"])
                    safe_sentiment = _html.escape(n["sentiment"])
                    news_rows += f"""
                        <div class="news-item">
                            <span class="news-sentiment {sent_cls}">[{safe_sentiment}]</span>
                            <span class="news-date">{safe_date}</span>
                            <span class="news-pub">{safe_publisher}</span>
                            <div class="news-title">{safe_title}</div>
                            <div class="news-summary">{safe_summary}</div>
                        </div>"""

                news_html = f"""
                <tr class="news-row" id="news-{ticker.replace(".", "_")}" style="display:none">
                    <td colspan="20">
                        <div class="news-panel">
                            <div class="news-summary-line">{" | ".join(summary_parts)}</div>
                            {news_rows}
                        </div>
                    </td>
                </tr>"""
            else:
                news_html = f"""
                <tr class="news-row" id="news-{ticker.replace(".", "_")}" style="display:none">
                    <td colspan="20">
                        <div class="news-panel">
                            <div class="news-item"><span class="news-title">No recent news found</span></div>
                        </div>
                    </td>
                </tr>"""

        rows_html += f"""
        <tr class="{'highlight' if score >= threshold else ''}" 
            data-ma-bull="{'true' if ma_bullish else 'false'}" 
            data-above-poc="{'true' if above_poc else 'false'}"
            data-both-ma="{'true' if close_above_both else 'false'}"
            data-crossed="{'true' if ma_crossed else 'false'}"
            data-ticker="{_html.escape(ticker)}">
            <td class="ticker" onclick="toggleNews('{_html.escape(ticker.replace(".", "_"))}')">{_html.escape(ticker)}</td>
            <td class="score score-{_score_class(score)}">{score:.1f}</td>
            <td>{badge}</td>
            <td class="num">{r.get('close', '—')}</td>
            <td class="num">{ma_html}</td>
            <td class="num">{poc_html}</td>
            <td class="num">{bothma_html}</td>
            <td class="num bar-cell">
                <div class="bar-container">
                    <div class="bar" style="width: {r['trend'] / 20 * 100:.0f}%"></div>
                </div>
                <span class="bar-val">{r['trend']}/20</span>
            </td>
            <td class="num bar-cell">
                <div class="bar-container">
                    <div class="bar mom" style="width: {r['momentum'] / 15 * 100:.0f}%"></div>
                </div>
                <span class="bar-val">{r['momentum']}/15</span>
            </td>
            <td class="num bar-cell">
                <div class="bar-container">
                    <div class="bar rsi" style="width: {r['rsi'] / 8 * 100:.0f}%"></div>
                </div>
                <span class="bar-val">{r['rsi']}/8</span>
            </td>
            <td class="num bar-cell">
                <div class="bar-container">
                    <div class="bar macd" style="width: {r['macd'] / 7 * 100:.0f}%"></div>
                </div>
                <span class="bar-val">{r['macd']}/7</span>
            </td>
            <td class="num bar-cell">
                <div class="bar-container">
                    <div class="bar vol" style="width: {r['volume'] / 10 * 100:.0f}%"></div>
                </div>
                <span class="bar-val">{r['volume']}/10</span>
            </td>
            <td class="num bar-cell">
                <div class="bar-container">
                    <div class="bar rs" style="width: {r['rel_str'] / 10 * 100:.0f}%"></div>
                </div>
                <span class="bar-val">{r['rel_str']}/10</span>
            </td>
            <td class="num bar-cell">
                <div class="bar-container">
                    <div class="bar fund" style="width: {r.get('fundamentals', 0) / 20 * 100:.0f}%"></div>
                </div>
                <span class="bar-val">{r.get('fundamentals', 0)}/20</span>
            </td>
            <td class="num">{r.get('rsi_val', '—')}</td>
            <td class="num">{r.get('adx_val', '—')}</td>
            <td class="num {'bull' if (r.get('pc1m') or 0) > 0 else 'bear'}">{rs_icon}{r.get('pc1m', '—')}%</td>
            <td><span class="{trend_class}">{trend_icon} {r['trend_dir']}</span></td>
            <td>{r.get('volat_stat', '—')}</td>
            <td><span class="{sideways_cls}" title="{sideways_reasons}">{sideways_label}</span></td>
        </tr>
        {news_html}"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
    :root {{
        /* Aurora (GUI) dark-theme palette — keeps the exported report visually
           identical to the app: same surfaces, borders and accent colors. */
        --bg: #0f0f13; --surface: #16161b; --surface2: #1c1c22; --surface3: #24242c;
        --border: #2b2b34; --border-light: #3a3a46; --text: #e9eaf0; --text-dim: #8e93a8; --text-faint: #5c6178;
        --green: #34d399; --lime: #a3e635; --orange: #fb923c; --red: #f87171;
        --blue: #60a5fa; --cyan: #22d3ee; --yellow: #facc15;
        /* Theme-aware component tints (overridden for the light variant) */
        --focus-ring: rgba(52,211,153,0.15); --row-hover: rgba(52,211,153,0.06);
        --row-hl: rgba(52,211,153,0.10); --ticker-hover-bg: rgba(52,211,153,0.12);
        --ticker-hover-c: #ffffff; --score-glow: 0 0 8px rgba(52,211,153,0.35);
        --track: rgba(255,255,255,0.06); --row-line: rgba(43,43,52,0.6);
        --news-line: rgba(43,43,52,0.5);
        --radius: 12px; --radius-sm: 8px;
    }}
    /* Light variant (button in the header) mirrors the GUI Aurora light theme. */
    body.light {{
        --bg: #f0f5f1; --surface: #ffffff; --surface2: #eef5f0; --surface3: #e6efe8;
        --border: #d1e3d6; --border-light: #e6efe8; --text: #0f2318; --text-dim: #5a7a65; --text-faint: #8aa89a;
        --green: #059669; --lime: #65a30d; --orange: #d97706; --red: #dc2626;
        --blue: #0284c7; --cyan: #0e7490; --yellow: #ca8a04;
        --focus-ring: rgba(5,150,105,0.18); --row-hover: rgba(5,150,105,0.07);
        --row-hl: rgba(5,150,105,0.11); --ticker-hover-bg: rgba(5,150,105,0.10);
        --ticker-hover-c: #065f46; --score-glow: none;
        --track: rgba(15,35,24,0.08); --row-line: rgba(209,227,214,0.9);
        --news-line: rgba(209,227,214,0.85);
        background: radial-gradient(1200px 600px at 0% -10%, #dcefe3 0%, var(--bg) 55%), var(--bg);
    }}
    .theme-toggle {{
        margin-left: auto; background: var(--surface); color: var(--text);
        border: 1px solid var(--border); border-radius: 20px; padding: 6px 14px;
        font-size: 0.8em; font-weight: 600; cursor: pointer;
        font-family: 'Inter', sans-serif;
        transition: border-color 0.2s, color 0.2s, background 0.2s;
    }}
    .theme-toggle:hover {{ border-color: var(--green); color: var(--green); }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; background: radial-gradient(1200px 600px at 0% -10%, #1b2f4d 0%, var(--bg) 55%), var(--bg); color: var(--text); padding: 24px; line-height: 1.5; min-height: 100vh; }}
    h1 {{ font-family: 'Inter', sans-serif; font-size: 1.6em; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 4px;
         background: linear-gradient(90deg, #38bdf8, #a78bfa); -webkit-background-clip: text; background-clip: text; color: transparent; }}
    .subtitle {{ color: var(--text-dim); font-size: 0.9em; margin-bottom: 6px; }}
    .meta {{ color: var(--text-faint); font-size: 0.8em; margin-bottom: 18px; display: flex; gap: 12px; flex-wrap: wrap; }}
    .meta span {{ background: var(--surface); border: 1px solid var(--border); padding: 4px 10px; border-radius: 20px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 20px; }}
    .stat {{ background: linear-gradient(180deg, var(--surface) 0%, var(--surface2) 100%); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 14px; position: relative; overflow: hidden; }}
    .stat::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: var(--accent, var(--green)); opacity: 0.9; }}
    .stat.green {{ --accent: var(--green); }} .stat.lime {{ --accent: var(--lime); }} .stat.cyan {{ --accent: var(--cyan); }} .stat.orange {{ --accent: var(--orange); }} .stat.red {{ --accent: var(--red); }}
    .stat .num {{ font-family: 'JetBrains Mono', monospace; font-size: 1.9em; font-weight: 700; line-height: 1; }}
    .stat .label {{ color: var(--text-dim); font-size: 0.7em; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 6px; }}
    .stat .sub {{ color: var(--text-faint); font-size: 0.7em; margin-top: 2px; }}
    .filters {{ margin-bottom: 18px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; background: var(--surface); border: 1px solid var(--border); padding: 10px 12px; border-radius: var(--radius); }}
    .filters input, .filters select {{
        background: var(--bg); border: 1px solid var(--border); color: var(--text);
        padding: 8px 14px; border-radius: 8px; font-family: 'Inter', sans-serif; font-size: 0.85em; transition: border-color 0.2s, box-shadow 0.2s;
    }}
    .filters input:focus, .filters select:focus {{ outline: none; border-color: var(--green); box-shadow: 0 0 0 3px var(--focus-ring); }}
    .filters input {{ width: 280px; }}
    .table-wrap {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.8em; min-width: 1100px; }}
    th {{ background: var(--surface2); color: var(--text-dim); padding: 10px 8px; text-align: left; font-weight: 600; font-size: 0.75em; letter-spacing: 0.06em; text-transform: uppercase;
          border-bottom: 1px solid var(--border); cursor: pointer; user-select: none; position: sticky; top: 0; white-space: nowrap; transition: color 0.15s, background 0.15s; }}
    th:hover {{ color: var(--green); background: var(--surface3); }}
    th.sorted-asc::after {{ content: " ▲"; color: var(--green); }}
    th.sorted-desc::after {{ content: " ▼"; color: var(--green); }}
    td {{ padding: 9px 8px; border-bottom: 1px solid var(--row-line); vertical-align: middle; }}
    tbody tr {{ transition: background 0.15s; }}
    tbody tr:hover {{ background: var(--row-hover); }}
    tbody tr.highlight {{ background: var(--row-hl) !important; box-shadow: inset 3px 0 0 var(--green); }}
    .ticker {{ color: var(--green); font-weight: 700; cursor: pointer; font-family: 'JetBrains Mono', monospace; }}
    .ticker:hover {{ color: var(--ticker-hover-c); text-decoration: none; background: var(--ticker-hover-bg); padding: 2px 6px; border-radius: 4px; margin: -2px -6px; }}
    .score {{ font-family: 'JetBrains Mono', monospace; font-size: 1.15em; font-weight: 700; }}
    .score-excellent {{ color: var(--green); text-shadow: var(--score-glow); }}
    .score-good {{ color: var(--lime); }}
    .score-moderate {{ color: var(--orange); }}
    .score-poor {{ color: var(--red); }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; font-family: 'JetBrains Mono', monospace; }}
    .bull {{ color: var(--green); font-weight: 600; }}
    .bear {{ color: var(--red); font-weight: 600; }}
    .bar-cell {{ white-space: nowrap; }}
    .bar-container {{ display: inline-block; width: 52px; height: 6px; background: var(--track);
                      border-radius: 99px; vertical-align: middle; margin-right: 6px; overflow: hidden; }}
    .bar {{ height: 100%; border-radius: 99px; background: var(--green); transition: width 0.4s cubic-bezier(0.22,1,0.36,1); }}
    .bar.mom {{ background: var(--cyan); }}
    .bar.rsi {{ background: var(--blue); }}
    .bar.macd {{ background: #a78bfa; }}
    .bar.vol {{ background: var(--orange); }}
    .bar.rs {{ background: var(--lime); }}
    .bar.fund {{ background: #ffe600; }}
    .sideways {{ color: var(--orange); font-weight: bold; }}
    .trending {{ color: var(--green); font-weight: bold; }}
    .bar-val {{ color: var(--text-dim); font-size: 0.9em; }}
    .badge {{ padding: 3px 10px; border-radius: 20px; font-size: 0.7em; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; border: 1px solid transparent; }}
    .badge.excellent {{ background: rgba(52,211,153,0.15); color: var(--green); border-color: rgba(52,211,153,0.25); }}
    .badge.good {{ background: rgba(163,230,53,0.12); color: var(--lime); border-color: rgba(163,230,53,0.2); }}
    .badge.moderate {{ background: rgba(251,146,60,0.12); color: var(--orange); border-color: rgba(251,146,60,0.2); }}
    .badge.poor {{ background: rgba(248,113,113,0.1); color: var(--red); border-color: rgba(248,113,113,0.18); }}
    .ma-cross {{ color: var(--green); font-weight: bold; font-size: 0.9em; }}
    .ma-bull {{ color: var(--lime); font-weight: bold; font-size: 0.9em; }}
    .ma-bear {{ color: var(--red); font-weight: bold; font-size: 0.9em; }}
    .poc-above {{ color: var(--green); font-weight: bold; font-size: 0.9em; }}
    .poc-below {{ color: var(--red); font-weight: bold; font-size: 0.9em; }}
    .bothma-yes {{ color: var(--green); font-weight: bold; font-size: 0.9em; }}
    .bothma-no {{ color: var(--text-dim); font-size: 0.9em; }}
    .fresh {{ color: var(--green); }}
    .stale {{ color: var(--text-dim); }}
    .footer {{ margin-top: 20px; color: var(--text-dim); font-size: 0.75em; text-align: center; }}

    /* ─── News panel styles ─────────────────────────────── */
    .news-row td {{ padding: 0; border-bottom: 1px solid var(--border); }}
    .news-panel {{
        background: var(--surface);
        border-left: 3px solid var(--cyan);
        padding: 10px 16px;
        margin: 4px 12px 8px 40px;
        border-radius: 4px;
    }}
    .news-summary-line {{
        color: var(--text);
        font-size: 0.85em;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border);
    }}
    .news-item {{
        padding: 6px 0;
        border-bottom: 1px solid var(--news-line);
    }}
    .news-item:last-child {{ border-bottom: none; }}
    .news-sentiment {{
        font-weight: bold;
        font-size: 0.8em;
        margin-right: 6px;
    }}
    .news-sentiment.good {{ color: var(--green); }}
    .news-sentiment.bad {{ color: var(--red); }}
    .news-sentiment.neutral {{ color: var(--text-dim); }}
    .news-date {{ color: var(--text-dim); font-size: 0.78em; margin-right: 8px; }}
    .news-pub {{ color: var(--cyan); font-size: 0.78em; }}
    .news-title {{ color: var(--text); font-size: 0.85em; margin-top: 3px; font-weight: bold; }}
    .news-summary {{ color: var(--text-dim); font-size: 0.78em; margin-top: 2px; }}
    .news-good {{ color: var(--green); font-weight: bold; }}
    .news-bad {{ color: var(--red); font-weight: bold; }}
    .news-neutral {{ color: var(--text-dim); font-weight: bold; }}
</style>
</head>
<body>

<div style="display:flex; align-items:center; gap:14px; margin-bottom:10px;">
  <div style="width:38px; height:38px; background: linear-gradient(135deg, var(--green), var(--cyan)); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px;">◈</div>
  <div>
    <h1 style="margin:0;">{_html.escape(title)}</h1>
    <div class="subtitle">HMA × EMA Swing System — 10-factor score · Volume Profile · News Sentiment</div>
  </div>
  <div style="flex:1"></div>
  <button id="themeToggle" class="theme-toggle" onclick="toggleTheme()" title="Toggle light / dark theme">☾ Light</button>
</div>
<div class="meta"><span>⏱ {now}</span><span>🎯 Threshold {threshold}+</span><span>📦 {len(results)} total</span><span>⚡ Generated locally</span></div>

<div class="summary">
    <div class="stat green">
        <div class="num">{len(passed)}</div>
        <div class="label">Passed · {threshold}+</div>
        <div class="sub">{len(passed)/max(len(results),1)*100:.0f}% hit rate</div>
    </div>
    <div class="stat red">
        <div class="num">{len(failed)}</div>
        <div class="label">Below threshold</div>
        <div class="sub">filtered out</div>
    </div>
    <div class="stat cyan">
        <div class="num">{len(results)}</div>
        <div class="label">Total scanned</div>
        <div class="sub">{len({r.get('ticker','')[:3] for r in results})} sectors</div>
    </div>
    <div class="stat lime">
        <div class="num">{passed[0]['total'] if passed else 0:.0f}</div>
        <div class="label">Highest score</div>
        <div class="sub">{passed[0]['ticker'] if passed else '—'}</div>
    </div>
</div>

<div class="filters">
    <input type="text" id="search" placeholder="Search ticker..." oninput="filterTable()">
    <select id="minScore" onchange="filterTable()">
        <option value="0">All scores</option>
        <option value="70" {"selected" if threshold >= 70 else ""}>70+ EXCELLENT</option>
        <option value="50" {"selected" if threshold == 50 else ""}>50+ GOOD</option>
        <option value="30">30+ MODERATE</option>
    </select>
    <select id="trendFilter" onchange="filterTable()">
        <option value="">All trends</option>
        <option value="Bull">Bullish only</option>
        <option value="Bear">Bearish only</option>
    </select>
    <select id="signalFilter" onchange="filterTable()">
        <option value="">All signals</option>
        <option value="both_ma+poc">Close &gt; Both MA + POC</option>
        <option value="ma+poc">MA Bull + POC</option>
        <option value="crossed">Fresh Crossover</option>
        <option value="ma_bull">MA Bullish only</option>
        <option value="above_poc">Above POC only</option>
        <option value="both_ma">Close &gt; Both MA</option>
    </select>
    <select id="newsFilter" onchange="filterTable()">
        <option value="">All</option>
        <option value="good_news">With Good News</option>
        <option value="bad_news">With Bad News</option>
    </select>
</div>

<div class="table-wrap">
<table id="stockTable">
<thead>
<tr>
    <th onclick="sortTable(0)">Ticker</th>
    <th onclick="sortTable(1)">Score</th>
    <th onclick="sortTable(2)">Rating</th>
    <th onclick="sortTable(3)">Price</th>
    <th onclick="sortTable(4)">MA Signal</th>
    <th onclick="sortTable(5)">POC</th>
    <th onclick="sortTable(6)">Both MA</th>
    <th onclick="sortTable(7)">Trend</th>
    <th onclick="sortTable(8)">Momentum</th>
    <th onclick="sortTable(9)">RSI</th>
    <th onclick="sortTable(10)">MACD</th>
    <th onclick="sortTable(11)">Volume</th>
    <th onclick="sortTable(12)">RS</th>
    <th onclick="sortTable(13)">Fund</th>
    <th onclick="sortTable(14)">RSI Val</th>
    <th onclick="sortTable(15)">ADX</th>
    <th onclick="sortTable(16)">1M Chg</th>
    <th onclick="sortTable(17)">Trend</th>
    <th onclick="sortTable(18)">Volatility</th>
    <th onclick="sortTable(19)">Sideways</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>

<div class="footer">
    Generated by HMAxEMA Stock Scanner &nbsp;|&nbsp; Scoring engine mirrors the Pine Script indicator<br>
    Click any ticker to expand/collapse news sentiment
</div>

<script>
const THEME_KEY = "hmareport-theme";

function applyTheme(light) {{
    document.body.classList.toggle("light", light);
    const btn = document.getElementById("themeToggle");
    if (btn) btn.textContent = light ? "☀ Dark" : "☾ Light";
}}

function toggleTheme() {{
    const light = document.body.classList.contains("light");
    applyTheme(!light);
    try {{ localStorage.setItem(THEME_KEY, light ? "dark" : "light"); }} catch (e) {{}}
}}

// Restore the saved preference (survives closing the report).
try {{ if (localStorage.getItem(THEME_KEY) === "light") applyTheme(true); }} catch (e) {{}}

let sortDir = {{}};
function sortTable(col) {{
    const table = document.getElementById("stockTable");
    const tbody = table.tBodies[0];
    const allRows = Array.from(tbody.rows);
    const th = table.tHead.rows[0].cells[col];

    sortDir[col] = sortDir[col] === "asc" ? "desc" : "asc";
    const dir = sortDir[col];

    for (let cell of table.tHead.rows[0].cells) {{
        cell.classList.remove("sorted-asc", "sorted-desc");
    }}
    th.classList.add(dir === "asc" ? "sorted-asc" : "sorted-desc");

    // Build pairs: each data row + its following news row (if any) - keep them together
    const pairs = [];
    for (let i = 0; i < allRows.length; i++) {{
        const row = allRows[i];
        if (row.classList.contains("news-row")) continue;
        const newsRow = (i + 1 < allRows.length && allRows[i + 1].classList.contains("news-row")) ? allRows[i + 1] : null;
        pairs.push({{dataRow: row, newsRow: newsRow}});
        if (newsRow) i++;
    }}

    const ratingOrder = {{"EXCELLENT":4,"GOOD":3,"MODERATE":2,"POOR":1}};

    pairs.sort((a, b) => {{
        let aCell = a.dataRow.cells[col];
        let bCell = b.dataRow.cells[col];
        let aVal = aCell ? aCell.textContent.trim() : "";
        let bVal = bCell ? bCell.textContent.trim() : "";
        let aText = aVal.replace(/<[^>]*>/g, "").trim();
        let bText = bVal.replace(/<[^>]*>/g, "").trim();

        // Rating column (2) — custom order, not alphabetical
        if (col === 2) {{
            let aR = ratingOrder[aText.toUpperCase()] || 0;
            let bR = ratingOrder[bText.toUpperCase()] || 0;
            return dir === "asc" ? aR - bR : bR - aR;
        }}

        let aNum = parseFloat(aText.replace(/[^0-9.\\-]/g, ""));
        let bNum = parseFloat(bText.replace(/[^0-9.\\-]/g, ""));
        let aIsNum = !isNaN(aNum) && /[0-9]/.test(aText);
        let bIsNum = !isNaN(bNum) && /[0-9]/.test(bText);
        if (aIsNum && bIsNum) {{
            return dir === "asc" ? aNum - bNum : bNum - aNum;
        }}
        return dir === "asc" ? aText.localeCompare(bText) : bText.localeCompare(aText);
    }});

    // Re-append in sorted order, keeping news rows attached to their parent
    pairs.forEach(p => {{
        tbody.appendChild(p.dataRow);
        if (p.newsRow) tbody.appendChild(p.newsRow);
    }});
}}

function toggleNews(tickerId) {{
    const newsRow = document.getElementById("news-" + tickerId);
    if (!newsRow) return;

    // Collapse any other open news panels
    document.querySelectorAll(".news-row").forEach(row => {{
        if (row.id !== "news-" + tickerId) {{
            row.style.display = "none";
        }}
    }});

    // Toggle this one
    newsRow.style.display = newsRow.style.display === "none" ? "" : "none";
}}

function filterTable() {{
    const search = document.getElementById("search").value.toLowerCase();
    const minScore = parseFloat(document.getElementById("minScore").value);
    const trendFilter = document.getElementById("trendFilter").value;
    const signalFilter = document.getElementById("signalFilter").value;
    const newsFilter = document.getElementById("newsFilter").value;
    const rows = document.getElementById("stockTable").tBodies[0].rows;

    for (let i = 0; i < rows.length; i++) {{
        const row = rows[i];
        // Skip news rows — they follow their parent
        if (row.classList.contains("news-row")) continue;

        const ticker = row.cells[0].textContent.toLowerCase();
        const score = parseFloat(row.cells[1].textContent);
        const trend = row.cells[17].textContent;
        const maBull = row.getAttribute("data-ma-bull") === "true";
        const abovePoc = row.getAttribute("data-above-poc") === "true";
        const bothMa = row.getAttribute("data-both-ma") === "true";
        const crossed = row.getAttribute("data-crossed") === "true";

        const matchSearch = ticker.includes(search);
        const matchScore = score >= minScore;
        const matchTrend = !trendFilter || trend.includes(trendFilter);

        let matchSignal = true;
        if (signalFilter === "both_ma+poc") matchSignal = bothMa && abovePoc;
        else if (signalFilter === "ma+poc") matchSignal = maBull && abovePoc;
        else if (signalFilter === "crossed") matchSignal = crossed;
        else if (signalFilter === "ma_bull") matchSignal = maBull;
        else if (signalFilter === "above_poc") matchSignal = abovePoc;
        else if (signalFilter === "both_ma") matchSignal = bothMa;

        let matchNews = true;
        if (newsFilter) {{
            const newsRow = rows[i + 1];
            if (newsRow && newsRow.classList.contains("news-row")) {{
                const newsText = newsRow.textContent.toLowerCase();
                if (newsFilter === "good_news") matchNews = newsText.includes("good");
                else if (newsFilter === "bad_news") matchNews = newsText.includes("bad");
            }} else {{
                matchNews = false;
            }}
        }}

        const visible = matchSearch && matchScore && matchTrend && matchSignal && matchNews;
        row.style.display = visible ? "" : "none";
        // Also hide/show the news row that follows
        if (i + 1 < rows.length && rows[i + 1].classList.contains("news-row")) {{
            rows[i + 1].style.display = "none";
        }}
    }}
}}
</script>
</body>
</html>"""

    return html


def _score_class(score: float) -> str:
    """Return CSS class for score coloring."""
    if score >= 70:
        return "excellent"
    elif score >= 50:
        return "good"
    elif score >= 30:
        return "moderate"
    return "poor"


def save_report(html: str, filename: str = "scanner_report.html",
                max_reports: int = 4) -> str:
    """Save HTML report to file and keep only the last max_reports files."""
    import glob as _glob
    import os as _os

    # Save the report
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    # Clean up old reports — keep only the newest max_reports
    report_dir = _os.path.dirname(filename) or "."
    pattern = _os.path.join(report_dir, "scanner_report_*.html")
    reports = sorted(_glob.glob(pattern), key=_os.path.getmtime, reverse=True)
    for old in reports[max_reports:]:
        try:
            _os.remove(old)
        except OSError:
            pass

    return filename
