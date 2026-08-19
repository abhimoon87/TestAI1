"""
HTML report generator for stock scanner results.
Produces a sortable, filterable table with color-coded scores.
"""

from datetime import datetime


def generate_html_report(results: list, title: str = "HMAxEMA Stock Scanner",
                         threshold: float = 50.0) -> str:
    """
    Generate a complete HTML report from scan results.

    Args:
        results: List of dicts with 'ticker', 'total', scores, and metadata
        title: Report title
        threshold: Minimum score threshold

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

        sideways = r.get('is_sideways', False)
        sideways_cls = 'sideways' if sideways else 'trending'
        sideways_reasons = ', '.join(r.get('sideways_reasons', []))
        sideways_label = f'⚠ Chop' if sideways else '✓ Trend'

        rows_html += f"""
        <tr class="{'highlight' if score >= threshold else ''}">
            <td class="ticker">{r['ticker']}</td>
            <td class="score score-{_score_class(score)}">{score:.1f}</td>
            <td>{badge}</td>
            <td class="num">{r.get('close', '—')}</td>
            <td class="num bar-cell">
                <div class="bar-container">
                    <div class="bar" style="width: {r['trend'] / 15 * 100:.0f}%"></div>
                </div>
                <span class="bar-val">{r['trend']}/15</span>
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
        </tr>"""

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
    .summary {{ display: flex; gap: 20px; margin-bottom: 15px; }}
    .stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 18px; }}
    .stat .num {{ font-size: 1.8em; font-weight: bold; }}
    .stat .label {{ color: var(--text-dim); font-size: 0.8em; margin-top: 2px; }}
    .filters {{ margin-bottom: 15px; display: flex; gap: 10px; align-items: center; }}
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
    .ticker {{ color: var(--green); font-weight: bold; }}
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
    .footer {{ margin-top: 20px; color: var(--text-dim); font-size: 0.75em; text-align: center; }}
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
    <input type="text" id="search" placeholder="🔍 Search ticker..." oninput="filterTable()">
    <select id="minScore" onchange="filterTable()">
        <option value="0">All scores</option>
        <option value="70" {"selected" if threshold >= 70 else ""}>70+ EXCELLENT</option>
        <option value="50" {"selected" if threshold == 50 else ""}>50+ GOOD</option>
        <option value="30">30+ MODERATE</option>
    </select>
    <select id="trendFilter" onchange="filterTable()">
        <option value="">All trends</option>
        <option value="Bull">▲ Bullish only</option>
        <option value="Bear">▼ Bearish only</option>
    </select>
</div>

<table id="stockTable">
<thead>
<tr>
    <th onclick="sortTable(0)">Ticker</th>
    <th onclick="sortTable(1)">Score</th>
    <th onclick="sortTable(2)">Rating</th>
    <th onclick="sortTable(3)">Price (₹)</th>
    <th onclick="sortTable(4)">Trend</th>
    <th onclick="sortTable(5)">Momentum</th>
    <th onclick="sortTable(6)">RSI</th>
    <th onclick="sortTable(7)">MACD</th>
    <th onclick="sortTable(8)">Volume</th>
    <th onclick="sortTable(9)">RS</th>
    <th onclick="sortTable(10)">Fund</th>
    <th onclick="sortTable(11)">RSI Val</th>
    <th onclick="sortTable(12)">ADX</th>
    <th onclick="sortTable(13)">1M Chg</th>
    <th onclick="sortTable(14)">Trend</th>
    <th onclick="sortTable(15)">Volatility</th>
    <th onclick="sortTable(16)">Sideways</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>

<div class="footer">
    Generated by HMAxEMA Stock Scanner &nbsp;|&nbsp; Scoring engine mirrors the Pine Script indicator
</div>

<script>
let sortDir = {{}};
function sortTable(col) {{
    const table = document.getElementById("stockTable");
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    const th = table.tHead.rows[0].cells[col];

    // Toggle direction
    sortDir[col] = sortDir[col] === "asc" ? "desc" : "asc";
    const dir = sortDir[col];

    // Clear sorted classes
    for (let cell of table.tHead.rows[0].cells) {{
        cell.classList.remove("sorted-asc", "sorted-desc");
    }}
    th.classList.add(dir === "asc" ? "sorted-asc" : "sorted-desc");

    rows.sort((a, b) => {{
        let aVal = a.cells[col].textContent.trim();
        let bVal = b.cells[col].textContent.trim();
        let aNum = parseFloat(aVal.replace(/[+%]/g, ""));
        let bNum = parseFloat(bVal.replace(/[+%]/g, ""));
        if (!isNaN(aNum) && !isNaN(bNum)) {{
            return dir === "asc" ? aNum - bNum : bNum - aNum;
        }}
        return dir === "asc" ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    }});

    rows.forEach(row => tbody.appendChild(row));
}}

function filterTable() {{
    const search = document.getElementById("search").value.toLowerCase();
    const minScore = parseFloat(document.getElementById("minScore").value);
    const trendFilter = document.getElementById("trendFilter").value;
    const rows = document.getElementById("stockTable").tBodies[0].rows;

    for (let row of rows) {{
        const ticker = row.cells[0].textContent.toLowerCase();
        const score = parseFloat(row.cells[1].textContent);
        const trend = row.cells[13].textContent;

        const matchSearch = ticker.includes(search);
        const matchScore = score >= minScore;
        const matchTrend = !trendFilter || trend.includes(trendFilter);

        row.style.display = (matchSearch && matchScore && matchTrend) ? "" : "none";
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


def save_report(html: str, filename: str = "scanner_report.html") -> str:
    """Save HTML report to file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    return filename
