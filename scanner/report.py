"""
HTML report generator for stock scanner results.
Produces a sortable, filterable table with color-coded scores and news sentiment.
"""

from datetime import datetime, timedelta
from typing import Optional

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
    words = set((title + " " + summary).lower().split())
    g = len(words & SENTIMENT_GOOD)
    b = len(words & SENTIMENT_BAD)
    if g > b:
        return "Good"
    elif b > g:
        return "Bad"
    return "Neutral"


def _parse_date(date_str: str) -> Optional[datetime]:
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
    except Exception:
        return []


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
    # Sort by total score descending
    results.sort(key=lambda x: x["total"], reverse=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    passed = [r for r in results if r["total"] >= threshold]
    failed = [r for r in results if r["total"] < threshold]

    rows_html = ""
    for r in results:
        score = r["total"]
        ticker = r["ticker"]

        # Use combined rating if available, else fall back to score-based
        combined_rating = r.get('combined_rating', None)
        if combined_rating:
            rating_lower = combined_rating.lower()
            badge = f'<span class="badge {rating_lower}">{combined_rating}</span>'
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
        sideways_reasons = ', '.join(r.get('sideways_reasons', []))
        sideways_label = f'⚠ Chop' if sideways else '✓ Trend'

        # ─── News sentiment (fetched per stock) ────────────────────────────
        news_html = ""
        if fetch_news:
            news_items = fetch_stock_news(ticker)
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
                    sent_cls = n["sentiment"].lower()
                    news_rows += f"""
                        <div class="news-item">
                            <span class="news-sentiment {sent_cls}">[{n["sentiment"]}]</span>
                            <span class="news-date">{n["date"]}</span>
                            <span class="news-pub">{n["publisher"]}</span>
                            <div class="news-title">{n["title"]}</div>
                            <div class="news-summary">{n["summary"][:200]}</div>
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
            data-ticker="{ticker}">
            <td class="ticker" onclick="toggleNews('{ticker.replace(".", "_")}')">{ticker}</td>
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
<title>{title}</title>
<style>
    :root {{
        --bg: #0a1a10; --surface: #0f2a1a; --surface2: #153520;
        --border: #1a4a2a; --text: #c8d8c0; --text-dim: #6a8a6a;
        --green: #00ff88; --lime: #aaff00; --orange: #ffaa00; --red: #ff4444;
        --blue: #00aaff; --cyan: #00ddcc;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; background: var(--bg); color: var(--text); padding: 20px; }}
    h1 {{ color: var(--green); font-size: 1.4em; margin-bottom: 5px; }}
    .meta {{ color: var(--text-dim); font-size: 0.85em; margin-bottom: 15px; }}
    .summary {{ display: flex; gap: 20px; margin-bottom: 15px; flex-wrap: wrap; }}
    .stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 18px; }}
    .stat .num {{ font-size: 1.8em; font-weight: bold; }}
    .stat .label {{ color: var(--text-dim); font-size: 0.8em; margin-top: 2px; }}
    .filters {{ margin-bottom: 15px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    .filters input, .filters select {{
        background: var(--surface); border: 1px solid var(--border); color: var(--text);
        padding: 6px 12px; border-radius: 4px; font-family: inherit; font-size: 0.85em;
    }}
    .filters input {{ width: 250px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
    th {{ background: var(--surface2); color: var(--cyan); padding: 8px 6px; text-align: left;
          border-bottom: 2px solid var(--border); cursor: pointer; user-select: none; position: sticky; top: 0; }}
    th:hover {{ color: var(--green); }}
    th.sorted-asc::after {{ content: " ▲"; color: var(--green); }}
    th.sorted-desc::after {{ content: " ▼"; color: var(--green); }}
    td {{ padding: 6px; border-bottom: 1px solid var(--border); }}
    tr:hover {{ background: var(--surface2); }}
    tr.highlight {{ background: rgba(0,255,136,0.06); }}
    .ticker {{ color: var(--green); font-weight: bold; cursor: pointer; }}
    .ticker:hover {{ text-decoration: underline; color: #ffffff; }}
    .score {{ font-size: 1.1em; font-weight: bold; }}
    .score-excellent {{ color: var(--green); }}
    .score-good {{ color: var(--lime); }}
    .score-moderate {{ color: var(--orange); }}
    .score-poor {{ color: var(--red); }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .bull {{ color: var(--green); }}
    .bear {{ color: var(--red); }}
    .bar-cell {{ white-space: nowrap; }}
    .bar-container {{ display: inline-block; width: 45px; height: 8px; background: var(--surface);
                      border-radius: 3px; vertical-align: middle; margin-right: 4px; }}
    .bar {{ height: 100%; border-radius: 3px; background: var(--green); transition: width 0.3s; }}
    .bar.mom {{ background: var(--cyan); }}
    .bar.rsi {{ background: var(--blue); }}
    .bar.macd {{ background: #aa88ff; }}
    .bar.vol {{ background: var(--orange); }}
    .bar.rs {{ background: var(--lime); }}
    .bar.fund {{ background: #ffe600; }}
    .sideways {{ color: var(--orange); font-weight: bold; }}
    .trending {{ color: var(--green); font-weight: bold; }}
    .bar-val {{ color: var(--text-dim); font-size: 0.9em; }}
    .badge {{ padding: 2px 8px; border-radius: 3px; font-size: 0.75em; font-weight: bold; }}
    .badge.excellent {{ background: rgba(0,255,136,0.15); color: var(--green); }}
    .badge.good {{ background: rgba(170,255,0,0.15); color: var(--lime); }}
    .badge.moderate {{ background: rgba(255,170,0,0.15); color: var(--orange); }}
    .badge.poor {{ background: rgba(255,68,68,0.15); color: var(--red); }}
    .ma-cross {{ color: #00ff88; font-weight: bold; font-size: 0.9em; }}
    .ma-bull {{ color: #aaff00; font-weight: bold; font-size: 0.9em; }}
    .ma-bear {{ color: #ff4444; font-weight: bold; font-size: 0.9em; }}
    .poc-above {{ color: #00ff88; font-weight: bold; font-size: 0.9em; }}
    .poc-below {{ color: #ff4444; font-weight: bold; font-size: 0.9em; }}
    .bothma-yes {{ color: #00ff88; font-weight: bold; font-size: 0.9em; }}
    .bothma-no {{ color: #6a8a6a; font-size: 0.9em; }}
    .fresh {{ color: #00ff88; }}
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
        border-bottom: 1px solid rgba(26,74,42,0.5);
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

<h1>📊 {title}</h1>
<div class="meta">Scanned: {now} &nbsp;|&nbsp; Threshold: {threshold}+ &nbsp;|&nbsp; Total stocks: {len(results)}</div>

<div class="summary">
    <div class="stat">
        <div class="num" style="color: var(--green)">{len(passed)}</div>
        <div class="label">Passed ({threshold}+)</div>
    </div>
    <div class="stat">
        <div class="num" style="color: var(--red)">{len(failed)}</div>
        <div class="label">Below threshold</div>
    </div>
    <div class="stat">
        <div class="num" style="color: var(--cyan)">{len(results)}</div>
        <div class="label">Total scanned</div>
    </div>
    <div class="stat">
        <div class="num" style="color: var(--lime)">{passed[0]['total'] if passed else 0:.1f}</div>
        <div class="label">Highest score</div>
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

<div class="footer">
    Generated by HMAxEMA Stock Scanner &nbsp;|&nbsp; Scoring engine mirrors the Pine Script indicator<br>
    Click any ticker to expand/collapse news sentiment
</div>

<script>
let sortDir = {{}};
function sortTable(col) {{
    const table = document.getElementById("stockTable");
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    const th = table.tHead.rows[0].cells[col];

    sortDir[col] = sortDir[col] === "asc" ? "desc" : "asc";
    const dir = sortDir[col];

    for (let cell of table.tHead.rows[0].cells) {{
        cell.classList.remove("sorted-asc", "sorted-desc");
    }}
    th.classList.add(dir === "asc" ? "sorted-asc" : "sorted-desc");

    rows.sort((a, b) => {{
        let aVal = a.cells[col].textContent.trim();
        let bVal = b.cells[col].textContent.trim();
        let aText = aVal.replace(/<[^>]*>/g, "").trim();
        let bText = bVal.replace(/<[^>]*>/g, "").trim();
        let aNum = parseFloat(aText.replace(/[+%]/g, ""));
        let bNum = parseFloat(bText.replace(/[+%]/g, ""));
        if (!isNaN(aNum) && !isNaN(bNum)) {{
            return dir === "asc" ? aNum - bNum : bNum - aNum;
        }}
        return dir === "asc" ? aText.localeCompare(bText) : bText.localeCompare(aText);
    }});

    rows.forEach(row => tbody.appendChild(row));
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
