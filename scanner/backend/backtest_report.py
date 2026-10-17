"""HTML report and CSV export for the backtest engine.

Self-contained dark dashboard: SVG charts with crosshair tooltips,
monthly heatmap, distribution histogram, sortable/filterable tables.
No external assets — opens offline in any browser.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from .backtest_models import TradeResult

if TYPE_CHECKING:
    from .backtest import BacktestEngine


def _generate_trade_chart(
    t: TradeResult, stock_data: dict, chart_w: int = 300, chart_h: int = 110
) -> str:
    """SVG sparkline for one trade: price path, entry/SL/target levels, markers."""
    ticker = t.ticker
    if ticker not in stock_data:
        return ""

    df = stock_data[ticker]
    mask = (df.index >= t.entry_date) & (df.index <= t.exit_date)
    trade_df = df[mask]
    if len(trade_df) < 2:
        return ""

    closes = trade_df["close"].values
    n = len(closes)

    all_prices = list(closes) + [t.entry_price, t.exit_price]
    if t.target_price > 0:
        all_prices.append(t.target_price)
    if t.stop_loss > 0:
        all_prices.append(t.stop_loss)

    min_p = min(all_prices)
    max_p = max(all_prices)
    p_range = max_p - min_p if max_p > min_p else 1
    padding = p_range * 0.1
    min_p -= padding
    max_p += padding
    p_range = max_p - min_p

    margin_l, margin_r, margin_t, margin_b = 40, 10, 10, 18
    inner_w = chart_w - margin_l - margin_r
    inner_h = chart_h - margin_t - margin_b

    def sx(i):
        return margin_l + int(i / max(n - 1, 1) * inner_w)

    def sy(p):
        return margin_t + int((1 - (p - min_p) / p_range) * inner_h)

    parts = [f'<svg width="{chart_w}" height="{chart_h}" class="spark" font-family="ui-monospace,monospace">']

    for i in range(4):
        gy = margin_t + int(i / 3 * inner_h)
        gp = max_p - (i / 3 * p_range)
        parts.append(
            f'<line x1="{margin_l}" y1="{gy}" x2="{chart_w - margin_r}" y2="{gy}" stroke="#182430" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{margin_l - 4}" y="{gy + 3}" fill="#5c7186" font-size="7" text-anchor="end">{gp:.0f}</text>'
        )

    if t.target_price > min_p:
        ty = sy(t.target_price)
        parts.append(
            f'<line x1="{margin_l}" y1="{ty}" x2="{chart_w - margin_r}" y2="{ty}" stroke="#10b981" stroke-width="0.7" stroke-dasharray="3,3" opacity="0.55"/>'
        )
        parts.append(
            f'<text x="{chart_w - margin_r - 2}" y="{ty - 3}" fill="#10b981" font-size="7" text-anchor="end" opacity="0.85">TGT</text>'
        )

    if t.stop_loss > min_p:
        sly = sy(t.stop_loss)
        parts.append(
            f'<line x1="{margin_l}" y1="{sly}" x2="{chart_w - margin_r}" y2="{sly}" stroke="#f43f5e" stroke-width="0.7" stroke-dasharray="3,3" opacity="0.55"/>'
        )
        parts.append(
            f'<text x="{chart_w - margin_r - 2}" y="{sly + 10}" fill="#f43f5e" font-size="7" text-anchor="end" opacity="0.85">SL</text>'
        )

    ey = sy(t.entry_price)
    parts.append(
        f'<line x1="{margin_l}" y1="{ey}" x2="{chart_w - margin_r}" y2="{ey}" stroke="#38bdf8" stroke-width="0.7" opacity="0.45"/>'
    )
    parts.append(
        f'<text x="{margin_l + 2}" y="{ey - 3}" fill="#38bdf8" font-size="7" opacity="0.85">IN</text>'
    )

    price_points = " ".join(f"{sx(i)},{sy(c)}" for i, c in enumerate(closes))
    line_color = "#34d399" if t.pnl > 0 else "#fb7185"
    parts.append(f'<polyline points="{price_points}" fill="none" stroke="{line_color}" stroke-width="1.3"/>')

    trail_idx = -1
    for i, c in enumerate(closes):
        if c >= t.target_price:
            trail_idx = i
            break
    if trail_idx >= 0:
        tx, ty = sx(trail_idx), sy(closes[trail_idx])
        parts.append(
            f'<polygon points="{tx},{ty - 4} {tx + 4},{ty} {tx},{ty + 4} {tx - 4},{ty}" fill="#facc15" stroke="#facc15" stroke-width="0.5"/>'
        )

    parts.append(f'<circle cx="{sx(0)}" cy="{sy(closes[0])}" r="3" fill="#38bdf8" stroke="#0d131a" stroke-width="1"/>')
    parts.append(
        f'<circle cx="{sx(n - 1)}" cy="{sy(closes[-1])}" r="3" fill="{line_color}" stroke="#0d131a" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{sx(n - 1) + 5}" y="{sy(closes[-1]) + 3}" fill="#c9d6e2" font-size="7">Rs.{t.exit_price:,.0f}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _inr(v: float) -> str:
    """Compact rupee label: 1.3Cr / 67.5L / 123k."""
    a = abs(v)
    if a >= 1e7:
        return f"{v / 1e7:.2f}Cr"
    if a >= 1e5:
        return f"{v / 1e5:.1f}L"
    if a >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


# Config keys shown in the Run Configuration section (order matters).
_CFG_KEYS = [
    "timeframe",
    "entry_trigger",
    "w_fast_len",
    "w_slow_len",
    "confirm_weeks",
    "score_threshold",
    "exit_mode",
    "sl_pct",
    "rr_ratio",
    "be_act_r",
    "atr_mult",
    "struct_buf",
    "struct_cap_mult",
    "risk_pct",
    "commission_pct",
    "vp_weekly_lookback",
    "vp_rows",
    "warmup_bars",
    "position_size_pct",
    "max_positions",
    "max_positions_per_sector",
    "initial_capital",
    "fast_ma_len",
    "slow_ma_len",
]

_CSS = r"""
:root{
  --bg:#070a0e; --panel:#0d131a; --panel2:#101822; --line:#1a2532;
  --txt:#dbe6f0; --muted:#7d93a8; --dim:#5c7186;
  --pos:#34d399; --neg:#fb7185; --acc:#5eead4; --sky:#38bdf8; --amber:#facc15;
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
  background:
    radial-gradient(1100px 480px at 18% -8%, rgba(56,189,248,.10), transparent 60%),
    radial-gradient(900px 420px at 85% -12%, rgba(94,234,212,.09), transparent 55%),
    var(--bg);
  color:var(--txt);font-size:15px;
}
.num{font-family:ui-monospace,'Cascadia Mono',Consolas,monospace}
.pos{color:var(--pos)} .neg{color:var(--neg)} .muted{color:var(--muted)}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px 70px}

/* ── sticky nav ─────────────────────────── */
nav{position:sticky;top:0;z-index:60;backdrop-filter:blur(12px);
  background:rgba(7,10,14,.82);border-bottom:1px solid var(--line)}
.navin{max-width:1180px;margin:0 auto;padding:10px 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.brand{font-weight:700;font-size:.95em;letter-spacing:.4px}
.brand em{color:var(--acc);font-style:normal}
.navlinks{display:flex;gap:4px;flex-wrap:wrap}
.navlinks a{color:var(--muted);text-decoration:none;font-size:.78em;padding:5px 11px;border-radius:99px;
  border:1px solid transparent;transition:.15s}
.navlinks a:hover{color:var(--acc);border-color:#134e4a;background:#0b1f1e}
.navlinks a.active{color:var(--acc);border-color:#134e4a;background:#0b1f1e}
nav .prog{position:absolute;left:0;bottom:-1px;height:2px;width:0;
  background:linear-gradient(90deg,var(--acc),var(--sky));transition:width .12s linear}

/* ── hero ───────────────────────────────── */
header.hero{padding:44px 0 26px;display:grid;grid-template-columns:1.45fr 1fr;gap:30px;align-items:center}
@media (max-width:880px){header.hero{grid-template-columns:1fr}}
.hcard{background:linear-gradient(160deg,#101b26,#0c1218);border:1px solid var(--line);
  border-radius:18px;padding:16px 18px 13px;box-shadow:0 14px 34px rgba(0,0,0,.4);
  position:relative;overflow:hidden}
.hcard::after{content:'';position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(420px 120px at 70% 0%,rgba(94,234,212,.10),transparent 65%)}
.hcard .hh{display:flex;justify-content:space-between;font-size:.68em;letter-spacing:.16em;
  text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.hcard svg{display:block;width:100%;height:auto}
.hcap{display:flex;gap:14px;align-items:baseline;justify-content:center;margin-top:9px;
  font-family:ui-monospace,monospace;font-size:.8em}
.hcap .from{color:var(--dim)}
.hcap .to{font-weight:800;color:var(--pos);font-size:1.3em;text-shadow:0 0 22px rgba(52,211,153,.35)}
.kicker{font-size:.72em;letter-spacing:.22em;text-transform:uppercase;color:var(--acc);margin-bottom:10px}
h1{font-size:2.1em;font-weight:800;letter-spacing:-.5px}
h1 span{background:linear-gradient(90deg,var(--acc),var(--sky));-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--muted);font-size:.9em;margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.pill{background:var(--panel2);border:1px solid var(--line);border-radius:99px;padding:4px 12px;font-size:.85em}
.pill b{color:var(--txt);font-weight:600}

/* ── panels & sections ──────────────────── */
section{margin-top:34px;scroll-margin-top:68px}
.js section{opacity:0;transform:translateY(16px)}
.js section.in{opacity:1;transform:none;transition:opacity .5s ease,transform .5s ease}
@media (prefers-reduced-motion:reduce){
  *{animation:none!important;transition:none!important}
  .js section{opacity:1!important;transform:none!important}
}
.ptitle{display:flex;gap:12px;align-items:baseline;margin-bottom:16px}
.idx{font:600 10px ui-monospace,monospace;color:var(--acc);background:#0b2320;
  border:1px solid #14524c;padding:3px 8px;border-radius:7px;letter-spacing:.1em}
.ptitle h2{font-size:1.05em;font-weight:700;letter-spacing:.02em}
.ptitle .hint{color:var(--dim);font-size:.78em;margin-left:auto}
.panel{background:linear-gradient(180deg,#0e151d,#0c1218);border:1px solid var(--line);
  border-radius:18px;padding:20px 22px;box-shadow:0 10px 30px rgba(0,0,0,.35)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media (max-width:920px){.cols{grid-template-columns:1fr}}

/* ── KPI scoreboard ─────────────────────── */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(172px,1fr));gap:14px}
.kpi{--kc:#5eead4;background:linear-gradient(180deg,#101822,#0c1218);border:1px solid var(--line);border-radius:16px;
  padding:16px 18px;transition:.18s;position:relative;overflow:hidden}
.kpi:hover{transform:translateY(-3px);border-color:#24485a;box-shadow:0 14px 30px rgba(0,0,0,.45)}
.kpi::before{content:'';position:absolute;inset:0 0 auto;height:2px;opacity:.6;transition:.2s;
  background:linear-gradient(90deg,var(--kc),transparent 85%)}
.kpi:hover::before{opacity:1}
.kpi .lab{font-size:.66em;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.kpi .lab::before{content:'';display:inline-block;width:6px;height:6px;border-radius:2px;
  background:var(--kc);margin-right:7px;vertical-align:1px}
.kpi .val{font-size:1.75em;font-weight:800;margin-top:6px;letter-spacing:-.5px;white-space:nowrap}
.kpi .val.rs{font-size:1.35em}
.kpi .foot{font-size:.76em;color:var(--dim);margin-top:5px}
.kpi.hero{grid-column:span 2;background:linear-gradient(135deg,#0d2320,#0c1218 65%);border-color:#14524c}
.kpi.hero .val{font-size:2.6em;text-shadow:0 0 34px rgba(52,211,153,.35)}
@media (max-width:640px){.kpi.hero{grid-column:span 1}}
.mbars{display:flex;flex-direction:column;gap:5px;margin-top:10px}
.mbar{display:grid;grid-template-columns:28px 1fr auto;gap:8px;align-items:center;font-size:.68em;color:var(--muted)}
.mbar .t{background:#111a24;border-radius:99px;height:6px;overflow:hidden}
.mbar .t i{display:block;height:100%;border-radius:99px;transition:width .6s ease}
.mbar .n{font-family:ui-monospace,monospace;color:var(--txt)}
.dots{display:flex;gap:3px;margin-top:11px;align-items:center;flex-wrap:wrap}
.dots i{width:7px;height:7px;border-radius:99px;display:block}
.dots .sep{color:var(--dim);font-size:.65em;margin:0 5px}

/* ── charts ─────────────────────────────── */
.chart{position:relative}
.chart svg{display:block;width:100%;height:auto}
.tip{position:absolute;top:8px;display:none;transform:translateX(-50%);pointer-events:none;
  background:#0b1117ee;border:1px solid #24485a;border-radius:10px;padding:7px 11px;
  font-size:.78em;line-height:1.5;white-space:nowrap;z-index:5;box-shadow:0 8px 22px rgba(0,0,0,.5)}
.tip b{color:var(--acc)}
.legend{display:flex;gap:18px;font-size:.78em;color:var(--muted);margin-top:10px;flex-wrap:wrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:6px;vertical-align:-1px}

/* ── heatmap ────────────────────────────── */
.hm-scroll{overflow-x:auto}
table.hm{border-collapse:separate;border-spacing:4px;width:100%;font-size:.8em}
table.hm th{color:var(--dim);font-weight:600;font-size:.9em;padding:3px 6px;text-align:center}
table.hm td{padding:7px 4px;text-align:center;border-radius:8px;font-family:ui-monospace,monospace;
  min-width:46px;color:#e7f2ec;transition:.15s}
table.hm td:hover{outline:1.5px solid var(--acc);transform:scale(1.08)}
table.hm td.empty{background:#0d131a;color:#2b3947}
table.hm td.ytot{background:#131c26;font-weight:700;color:var(--txt);min-width:58px}
table.hm th.ycol,table.hm td.ycol{position:sticky;left:0;background:#0e151d;z-index:2}
.hmleg{display:flex;gap:10px;align-items:center;margin-top:14px;font-size:.75em;
  color:var(--dim);flex-wrap:wrap}
.hmleg .g{width:170px;height:9px;border-radius:99px;
  background:linear-gradient(90deg,rgba(251,113,133,.85),#131c26 50%,rgba(52,211,153,.85))}

/* ── tables ─────────────────────────────── */
.tscroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.86em}
th{color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.06em;font-size:.74em;
  padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap;
  background:#0e151d;position:sticky;top:0}
table.sortable th{cursor:pointer;user-select:none}
table.sortable th:hover{color:var(--acc)}
table.sortable th::after{content:'↕';opacity:.35;margin-left:6px;font-size:.85em}
table.sortable th[data-dir="asc"]::after{content:'↑';opacity:1;color:var(--acc)}
table.sortable th[data-dir="desc"]::after{content:'↓';opacity:1;color:var(--acc)}
td{padding:9px 12px;border-bottom:1px solid #141d27;vertical-align:top}
tbody tr{transition:.12s}
.zebra tbody tr:nth-child(even) td{background:#0e151c}
tbody tr:hover td,.zebra tbody tr:hover td{background:#111a24}
tbody tr:hover td:first-child{box-shadow:inset 3px 0 0 var(--acc)}
.tscroll.tall{max-height:78vh;overflow:auto}
.tnum{font-family:ui-monospace,monospace}
.bar{background:#111a24;border-radius:99px;height:8px;width:130px;overflow:hidden;display:inline-block;vertical-align:middle}
.bar i{display:block;height:100%;border-radius:99px}
.score{display:inline-block;min-width:34px;text-align:center;padding:2px 7px;border-radius:7px;
  font-weight:700;font-size:.92em;color:#08131a}

/* ── exit / sector bars ─────────────────── */
.hbar{display:grid;grid-template-columns:120px 1fr 74px;gap:10px;align-items:center;
  font-size:.84em;padding:7px 0;border-bottom:1px solid #141d27}
.hbar:last-child{border-bottom:none}
.hbar .track{background:#111a24;border-radius:99px;height:12px;overflow:hidden}
.hbar .track i{display:block;height:100%;border-radius:99px;transition:width .5s ease}
.hbar .v{text-align:right;font-family:ui-monospace,monospace}

/* ── leaderboard / donut / histogram ────── */
.lead{display:grid;grid-template-columns:1fr 1fr;gap:10px 26px}
@media (max-width:920px){.lead{grid-template-columns:1fr}}
.lrow{display:flex;gap:12px;align-items:center;padding:9px 12px;border-radius:12px;background:#0f1720;
  border:1px solid transparent;transition:.15s}
.lrow:hover{border-color:#24485a;transform:translateX(4px)}
.lrow .rank{font:700 11px ui-monospace,monospace;color:var(--dim);width:18px}
.lrow .tk{font-weight:700;flex:1}
.lrow .tk small{display:block;font-weight:400;color:var(--dim);font-size:.8em}
.lrow .pc{font-family:ui-monospace,monospace;font-weight:800;font-size:1.05em}
.donutwrap{display:flex;gap:24px;align-items:center;flex-wrap:wrap}
.donut{position:relative;width:150px;height:150px;flex:none}
.donut .mid{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.donut .mid b{font-size:1.7em;font-family:ui-monospace,monospace}
.donut .mid span{font-size:.65em;color:var(--muted);letter-spacing:.14em;text-transform:uppercase}
.statlist{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;flex:1;min-width:240px}
.statlist div{background:#0f1720;border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.statlist b{display:block;font-size:1.2em;font-family:ui-monospace,monospace;margin-top:3px}
.statlist span{font-size:.66em;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}

/* ── filters ────────────────────────────── */
.filters{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
.filters input,.filters select{background:#0f1720;border:1px solid var(--line);color:var(--txt);
  border-radius:10px;padding:8px 12px;font-size:.85em;outline:none;transition:.15s}
.filters input:focus,.filters select:focus{border-color:var(--acc)}
.filters input{min-width:170px}
.fcount{margin-left:auto;font-size:.8em;color:var(--muted);font-family:ui-monospace,monospace}

/* ── config chips ───────────────────────── */
.cfg{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:9px}
.cfg .c{display:flex;justify-content:space-between;background:#0f1720;border:1px solid var(--line);
  border-radius:11px;padding:9px 14px;font-size:.84em}
.cfg .k{color:var(--muted)}
.cfg .v{color:var(--txt);font-weight:600;font-family:ui-monospace,monospace}
footer{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--dim);font-size:.78em;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}
.spark{background:#0b1017;border-radius:8px}
.hist{width:100%;height:auto;display:block}
.fempty{padding:24px 10px;text-align:center;color:var(--muted);font-size:.86em}
#toTop{position:fixed;right:22px;bottom:22px;width:42px;height:42px;border-radius:99px;
  background:#101822ee;border:1px solid #24485a;color:var(--acc);font-size:1.1em;cursor:pointer;
  opacity:0;pointer-events:none;transition:.2s;backdrop-filter:blur(8px);z-index:70}
#toTop:hover{background:#134e4a;transform:translateY(-2px)}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:#1f2c3b;border-radius:99px;border:2px solid var(--bg)}
::-webkit-scrollbar-track{background:transparent}
"""

_JS = r"""
(function(){
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // reveal-on-scroll + nav scrollspy + scroll progress + back-to-top
  try{
    var secs = document.querySelectorAll('section[id]');
    var links = {};
    document.querySelectorAll('.navlinks a').forEach(function(a){ links[a.getAttribute('href').slice(1)] = a; });
    if('IntersectionObserver' in window){
      var ro = new IntersectionObserver(function(es){
        es.forEach(function(e){ if(e.isIntersecting){ e.target.classList.add('in'); ro.unobserve(e.target); } });
      }, {threshold:.05});
      secs.forEach(function(s){ ro.observe(s); });
      var so = new IntersectionObserver(function(es){
        es.forEach(function(e){
          var L = links[e.target.id]; if(!L || !e.isIntersecting) return;
          document.querySelectorAll('.navlinks a').forEach(function(x){ x.classList.remove('active'); });
          L.classList.add('active');
        });
      }, {rootMargin:'-45% 0px -50% 0px'});
      secs.forEach(function(s){ so.observe(s); });
    } else {
      secs.forEach(function(s){ s.classList.add('in'); });
    }
    var prog = document.getElementById('prog'), toTop = document.getElementById('toTop');
    function onScroll(){
      var h = document.documentElement.scrollHeight - window.innerHeight;
      var p = h > 0 ? Math.min(window.scrollY / h, 1) : 0;
      if(prog) prog.style.width = (p * 100) + '%';
      if(toTop){
        toTop.style.opacity = p > 0.15 ? '1' : '0';
        toTop.style.pointerEvents = p > 0.15 ? 'auto' : 'none';
      }
    }
    window.addEventListener('scroll', onScroll, {passive:true});
    onScroll();
    if(toTop) toTop.addEventListener('click', function(){ window.scrollTo({top:0, behavior:'smooth'}); });
  }catch(err){
    document.querySelectorAll('section').forEach(function(s){ s.classList.add('in'); });
  }

  // count-up KPI numbers (server already renders final values)
  if(!reduce){
    function fmtRs(a){
      a = Math.abs(a);
      if(a>=1e7) return (a/1e7).toFixed(2)+'Cr';
      if(a>=1e5) return (a/1e5).toFixed(1)+'L';
      if(a>=1e3) return Math.round(a/1e3)+'k';
      return String(Math.round(a));
    }
    document.querySelectorAll('[data-count]').forEach(function(el){
      var target = parseFloat(el.getAttribute('data-count'));
      var kind = el.getAttribute('data-kind') || 'num';
      if(!isFinite(target)) return;
      function fmt(x){
        if(kind==='pct') return (target>=0?'+':'')+x.toFixed(1)+'%';
        if(kind==='rsC') return 'Rs.'+fmtRs(x);
        if(kind==='rsP') return (x>=0?'+':'-')+'Rs.'+fmtRs(x);
        if(kind==='int') return String(Math.round(x));
        return x.toFixed(2);
      }
      var t0 = performance.now(), dur = 750;
      function step(t){
        var p = Math.min((t-t0)/dur,1), e = 1-Math.pow(1-p,3);
        el.textContent = fmt(target*e);
        if(p<1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    });
  }

    // equity / drawdown crosshair tooltips (data area: m .. m+w in viewBox units)
  var EQ = window.EQ || [];
  function wire(id){
    var box = document.getElementById(id); if(!box || EQ.length<2) return;
    var svg = box.querySelector('svg'), tip = box.querySelector('.tip'),
        line = box.querySelector('.cross');
    if(!svg) return;
    var m = parseFloat(svg.getAttribute('data-m')) || 0,
        w = parseFloat(svg.getAttribute('data-w')) || 1000;
    box.addEventListener('mousemove', function(e){
      var r = box.getBoundingClientRect();
      var px = Math.min(Math.max((e.clientX-r.left)/r.width,0),1)*1000;
      var frac = Math.min(Math.max((px-m)/w,0),1);
      var i = Math.round(frac*(EQ.length-1));
      var d = EQ[i];
      if(line){ line.setAttribute('x1',px); line.setAttribute('x2',px); line.style.opacity=.55; }
      tip.innerHTML = '<b>'+d[0]+'</b><br>Equity Rs.'+Number(d[1]).toLocaleString('en-US')
        +'<br>Drawdown '+d[2].toFixed(1)+'%';
      tip.style.display='block';
      tip.style.left = (px/10)+'%';
      tip.style.transform = px<160 ? 'translateX(0)' : (px>840 ? 'translateX(-100%)' : 'translateX(-50%)');
    });
    box.addEventListener('mouseleave', function(){
      tip.style.display='none';
      if(line) line.style.opacity=0;
    });
  }
  wire('eqChart'); wire('ddChart');

  // sortable tables
  document.querySelectorAll('table.sortable thead th').forEach(function(th, idx){
    th.addEventListener('click', function(){
      var tb = th.closest('table').querySelector('tbody');
      var dir = th.getAttribute('data-dir')==='asc' ? -1 : 1;
      th.closest('table').querySelectorAll('thead th').forEach(function(o){ o.removeAttribute('data-dir'); });
      th.setAttribute('data-dir', dir===1 ? 'asc':'desc');
      var rows = Array.prototype.slice.call(tb.rows);
      rows.sort(function(a,b){
        var A = a.cells[idx], B = b.cells[idx];
        var na = parseFloat(A.getAttribute('data-num')), nb = parseFloat(B.getAttribute('data-num'));
        if(!isNaN(na) && !isNaN(nb)) return (na-nb)*dir;
        return A.textContent.trim().localeCompare(B.textContent.trim())*dir;
      });
      rows.forEach(function(r){ tb.appendChild(r); });
    });
  });

  // trade log filters
  function wireFilters(){
    var q = document.getElementById('fQ'); if(!q) return;
    var yr = document.getElementById('fYear'), rs = document.getElementById('fReason'),
        oc = document.getElementById('fOutcome'), cnt = document.getElementById('fCount'),
        body = document.getElementById('tradeBody');
    function apply(){
      var qq = q.value.trim().toUpperCase(), y = yr.value, r = rs.value, o = oc.value, shown = 0;
      Array.prototype.forEach.call(body.rows, function(tr){
        var d = tr.dataset, ok = true;
        if(qq && d.t.indexOf(qq)<0) ok = false;
        if(ok && y && d.y!==y) ok = false;
        if(ok && r && d.r!==r) ok = false;
        if(ok && o==='w' && !(parseFloat(d.p)>0)) ok = false;
        if(ok && o==='l' && !(parseFloat(d.p)<=0)) ok = false;
        tr.style.display = ok ? '' : 'none';
        if(ok) shown++;
      });
      cnt.textContent = 'showing '+shown+' / '+body.rows.length;
      var fe = document.getElementById('fEmpty');
      if(fe) fe.style.display = shown ? 'none' : 'block';
    }
    [q,yr,rs,oc].forEach(function(el){ el.addEventListener('input', apply); el.addEventListener('change', apply); });
    apply();
  }
  wireFilters();
})();
"""


def generate_html_report(
    engine: BacktestEngine,
    metrics: dict,
    filepath: str = "Reports/backtest_report.html",
):
    """Generate a self-contained HTML backtest dashboard."""
    import html as html_mod

    m = metrics
    if m.get("total_trades", 0) == 0:
        print("  No trades to report in HTML.")
        return

    esc = html_mod.escape

    # ── run metadata ────────────────────────────────────────────────────────
    settings = engine.settings or {}
    tf = settings.get("timeframe", "daily")
    trigger = settings.get("entry_trigger", "close")
    label = settings.get("universe_label") or f"{len(engine.stocks)} stocks"
    eq_days = [d for d, _ in engine.equity_curve]
    eq_vals = [v for _, v in engine.equity_curve]
    sim_start = eq_days[0].strftime("%Y-%m-%d") if eq_days else "-"
    sim_end = eq_days[-1].strftime("%Y-%m-%d") if eq_days else "-"
    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    # ── trade stats ─────────────────────────────────────────────────────────
    tsorted = sorted(engine.trades, key=lambda t: t.exit_date)
    pnl_pcts = [t.pnl_pct for t in tsorted]

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    expectancy = _mean(pnl_pcts)
    avg_hold = _mean([float(t.days_held) for t in tsorted])
    best_trade = max(pnl_pcts, default=0.0)
    worst_trade = min(pnl_pcts, default=0.0)
    median_trade = sorted(pnl_pcts)[len(pnl_pcts) // 2] if pnl_pcts else 0.0
    max_win_streak = max_loss_streak = cur_w = cur_l = 0
    for t in tsorted:
        if t.pnl > 0:
            cur_w += 1
            cur_l = 0
            max_win_streak = max(max_win_streak, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_loss_streak = max(max_loss_streak, cur_l)

    # ── drawdown series + JSON payload for crosshair ────────────────────────
    dds: list[float] = []
    peak = float("-inf")
    for v in eq_vals:
        peak = max(peak, v)
        dds.append(v / peak - 1 if peak else 0.0)
    eq_json = json.dumps(
        [[d.strftime("%Y-%m-%d"), round(v, 2), round(dd * 100, 2)]
         for d, v, dd in zip(eq_days, eq_vals, dds)]
    )

    # ── hero mini equity card ───────────────────────────────────────────────
    hero_svg = (
        '<div class="hcard"><div class="hh"><span>equity curve</span>'
        f"<span>{sim_start} → {sim_end}</span></div>"
    )
    if len(eq_vals) > 1:
        hw, hht = 400, 92
        hmin, hmax = min(eq_vals), max(eq_vals)
        hrange = (hmax - hmin) or 1.0
        hn = len(eq_vals)
        hpts = " ".join(
            f"{i / (hn - 1) * hw:.1f},{(1 - (v - hmin) / hrange) * (hht - 8) + 4:.1f}"
            for i, v in enumerate(eq_vals)
        )
        hy_end = (1 - (eq_vals[-1] - hmin) / hrange) * (hht - 8) + 4
        hero_svg += (
            f'<svg viewBox="0 0 {hw} {hht}" font-family="ui-monospace,monospace">'
            f'<defs><linearGradient id="hg" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="#34d399" stop-opacity=".32"/>'
            f'<stop offset="1" stop-color="#34d399" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="0,{hht} {hpts} {hw},{hht}" fill="url(#hg)"/>'
            f'<polyline points="{hpts}" fill="none" stroke="#34d399" stroke-width="2"/>'
            f'<circle cx="{hw}" cy="{hy_end:.1f}" r="3.5" fill="#34d399"/>'
            f"</svg>"
            f'<div class="hcap"><span class="from">Rs.{_inr(m["initial_capital"])}</span>'
            f'<span class="muted">→</span>'
            f'<span class="to">Rs.{_inr(m["final_value"])}</span></div>'
        )
    hero_svg += "</div>"

    # ── KPI cards ───────────────────────────────────────────────────────────
    def _mbars(*rows):
        out = ['<div class="mbars">']
        for lab, w, col, txt in rows:
            out.append(
                f'<div class="mbar"><span>{lab}</span>'
                f'<span class="t"><i style="width:{w:.1f}%;background:{col}"></i></span>'
                f'<span class="n">{txt}</span></div>'
            )
        out.append("</div>")
        return "".join(out)

    def kpi(lab, val_html, foot, count=None, kind=None, hero=False,
            hex="#5eead4", viz=""):
        cls = "kpi hero" if hero else "kpi"
        vcls = "val num"
        if count is not None:
            vcls += " pos" if count >= 0 else " neg"
        if kind in ("rs", "rsC", "rsP"):
            vcls += " rs"
        cnt = (
            f' data-count="{count}" data-kind="{kind}"'
            if count is not None and kind
            else ""
        )
        return (
            f'<div class="{cls}" style="--kc:{hex}"><div class="lab">{lab}</div>'
            f'<div class="{vcls}"{cnt}>{val_html}</div>{viz}'
            f'<div class="foot">{foot}</div></div>'
        )

    win_hold = _mean([float(t.days_held) for t in tsorted if t.pnl > 0])
    loss_hold = _mean([float(t.days_held) for t in tsorted if t.pnl <= 0])
    hold_max = max(win_hold, loss_hold, 1.0)
    aw, al = abs(m["avg_win_pct"]), abs(m["avg_loss_pct"])
    pf_max = max(aw, al, 1.0)
    taken, missed = m["signals_taken"], m["signals_generated"] - m["signals_taken"]
    conv = m["signal_conversion"]
    dots_html = (
        '<div class="dots">'
        + '<i style="background:var(--pos)"></i>' * max_win_streak
        + '<span class="sep">vs</span>'
        + '<i style="background:var(--neg)"></i>' * max_loss_streak
        + "</div>"
    )
    v_eq = m["final_value"]
    v_pnl = m["total_pnl"]

    kpis = (
        kpi(
            "Total Return",
            f'{m["total_return_pct"]:+.1f}%',
            f'CAGR <b class="num">{m["annual_return_pct"]:+.1f}%</b> · {m["years"]:.1f} yrs',
            count=m["total_return_pct"], kind="pct", hero=True, hex="#34d399",
        )
        + kpi(
            "Final Equity",
            f'Rs.{_inr(v_eq)}',
            f'from Rs.{_inr(m["initial_capital"])} · exact '
            f'<b class="num">Rs.{v_eq:,.0f}</b>',
            count=v_eq, kind="rsC", hex="#34d399",
        )
        + kpi(
            "Max Drawdown",
            f'-{m["max_drawdown_pct"]:.1f}%',
            f'Rs.{_inr(m["max_drawdown"])} peak-to-trough',
            count=-m["max_drawdown_pct"], kind="pct", hex="#fb7185",
            viz=_mbars(
                ("dd", m["max_drawdown_pct"], "#fb7185", f'-{m["max_drawdown_pct"]:.1f}%'),
                ("run", max(0, 100 - m["max_drawdown_pct"]), "#1a2532", ""),
            ),
        )
        + kpi(
            "Sharpe",
            f'{m["sharpe_ratio"]:.2f}',
            f'Sortino <b class="num">{m["sortino_ratio"]:.2f}</b>',
            count=m["sharpe_ratio"], kind="ratio", hex="#38bdf8",
        )
        + kpi(
            "Win Rate",
            f'{m["win_rate"]:.1f}%',
            f'{m["total_trades"]} trades closed',
            count=m["win_rate"], kind="pct", hex="#34d399",
            viz=_mbars(
                ("win", m["win_rate"], "#34d399", f'{m["win_rate"]:.1f}%'),
                ("loss", 100 - m["win_rate"], "#fb7185", f'{100 - m["win_rate"]:.1f}%'),
            ),
        )
        + kpi(
            "Profit Factor",
            f'{m["profit_factor"]:.2f}',
            f'avg win <b class="num pos">{m["avg_win_pct"]:+.1f}%</b> / '
            f'loss <b class="num neg">{m["avg_loss_pct"]:+.1f}%</b>',
            count=m["profit_factor"], kind="ratio", hex="#38bdf8",
            viz=_mbars(
                ("win", aw / pf_max * 100, "#34d399", f'+{aw:.1f}%'),
                ("loss", al / pf_max * 100, "#fb7185", f'-{al:.1f}%'),
            ),
        )
        + kpi(
            "Expectancy",
            f'{expectancy:+.1f}%',
            f'median {median_trade:+.1f}% · per trade',
            count=expectancy, kind="pct", hex="#34d399",
        )
        + kpi(
            "Signals",
            f'{taken}',
            f'{m["signals_generated"]} generated · {conv:.1f}% conversion',
            count=taken, kind="int", hex="#facc15",
            viz=_mbars(
                ("took", max(conv, 1.5), "#38bdf8", str(taken)),
                ("skip", 100 - max(conv, 1.5), "#1a2532", str(missed)),
            ),
        )
        + kpi(
            "Avg Hold",
            f'{avg_hold:.0f}d',
            f'best <b class="num pos">{best_trade:+.1f}%</b> / '
            f'worst <b class="num neg">{worst_trade:+.1f}%</b>',
            count=avg_hold, kind="int", hex="#facc15",
            viz=_mbars(
                ("win", win_hold / hold_max * 100, "#34d399", f'{win_hold:.0f}d'),
                ("loss", loss_hold / hold_max * 100, "#fb7185", f'{loss_hold:.0f}d'),
            ),
        )
        + kpi(
            "Streaks",
            f'{max_win_streak}<span class="muted" style="font-size:.6em"> / {max_loss_streak}</span>',
            "max wins / losses in a row",
            hex="#facc15", viz=dots_html,
        )
        + kpi(
            "Winner Score",
            f'{m["avg_winner_score"]:.0f}',
            f'losers avg {m["avg_loser_score"]:.0f}',
            count=m["avg_winner_score"], kind="int", hex="#5eead4",
        )
        + kpi(
            "Net P&L",
            f'{"+" if v_pnl >= 0 else "-"}Rs.{_inr(abs(v_pnl))}',
            f'Rs.{v_pnl:+,.0f} net of commission',
            count=v_pnl, kind="rsP", hex="#34d399",
        )
    )

    # ── yearly returns (chain from initial capital) ─────────────────────────
    year_end: dict[int, float] = {}
    for d, v in engine.equity_curve:
        year_end[d.year] = v
    trades_by_year: dict[int, list] = {}
    for t in engine.trades:
        trades_by_year.setdefault(t.entry_date.year, []).append(t)
    yearly: list[tuple[int, float, float, int, str]] = []
    prev_val = m["initial_capital"]
    for y in sorted(year_end):
        val = year_end[y]
        ret = (val / prev_val - 1) * 100 if prev_val else 0.0
        prev_val = val
        ts = trades_by_year.get(y, [])
        wins = sum(1 for t in ts if t.pnl > 0)
        wr = f"{wins / len(ts) * 100:.0f}%" if ts else "-"
        yearly.append((y, ret, val, len(ts), wr))
    max_abs_year = max((abs(r) for _, r, *_ in yearly), default=1.0) or 1.0

    yearly_rows = ""
    for y, ret, val, n_tr, wr in yearly:
        c = "var(--pos)" if ret >= 0 else "var(--neg)"
        w = abs(ret) / max_abs_year * 100
        yearly_rows += (
            f'<tr><td class="tnum" data-num="{y}"><b>{y}</b></td>'
            f'<td class="tnum" data-num="{ret}"><span style="color:{c};font-weight:700">{ret:+.1f}%</span></td>'
            f'<td><span class="bar"><i style="width:{w:.1f}%;background:{c}"></i></span></td>'
            f'<td class="tnum" data-num="{val}">Rs.{_inr(val)}</td>'
            f'<td class="tnum" data-num="{n_tr}">{n_tr}</td>'
            f'<td class="tnum">{wr}</td></tr>'
        )

    # ── monthly heatmap ─────────────────────────────────────────────────────
    month_val: dict[tuple[int, int], float] = {}
    for d, v in engine.equity_curve:
        month_val[(d.year, d.month)] = v  # chronological — last wins
    month_rets: list[tuple[tuple[int, int], float]] = []
    prev = m["initial_capital"]
    for key in sorted(month_val):
        v = month_val[key]
        month_rets.append((key, (v / prev - 1) * 100 if prev else 0.0))
        prev = v
    year_ret: dict[int, float] = {}
    for (yy, _mm), r in month_rets:
        year_ret[yy] = ((1 + year_ret.get(yy, 0.0)) * (1 + r / 100) - 1) * 100
    max_abs_m = max((abs(r) for _, r in month_rets), default=1.0) or 1.0

    by_year: dict[int, dict[int, float]] = {}
    for (yy, mm), r in month_rets:
        by_year.setdefault(yy, {})[mm] = r
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    hm_rows = ""
    for yy in sorted(by_year):
        cells = ""
        for mm in range(1, 13):
            if mm in by_year[yy]:
                r = by_year[yy][mm]
                a = 0.10 + 0.75 * min(abs(r) / max_abs_m, 1.0)
                rgb = "52,211,153" if r >= 0 else "251,113,133"
                cells += (
                    f'<td style="background:rgba({rgb},{a:.2f})" '
                    f'title="{yy}-{mm:02d}: {r:+.1f}%">{r:+.1f}</td>'
                )
            else:
                cells += '<td class="empty">·</td>'
        yr = year_ret.get(yy, 0.0)
        yrgb = "52,211,153" if yr >= 0 else "251,113,133"
        hm_rows += (
            f'<tr><td class="ycol"><b>{yy}</b></td>{cells}'
            f'<td class="ytot" style="color:{yrgb}" title="FY {yy}: {yr:+.1f}%">'
            f'<b>{yr:+.1f}%</b></td></tr>'
        )
    hm_head = '<tr><th class="ycol">Year</th>' + "".join(
        f"<th>{n}</th>" for n in month_names
    ) + "<th>Year</th></tr>"

    # ── exits ───────────────────────────────────────────────────────────────
    exit_items = sorted(m["exit_reasons"].items(), key=lambda x: -x[1])
    max_ex = max((c for _, c in exit_items), default=1) or 1
    exit_rows = ""
    for reason, count in exit_items:
        pct = count / m["total_trades"] * 100
        w = count / max_ex * 100
        exit_rows += (
            f'<div class="hbar"><span>{esc(str(reason))}</span>'
            f'<span class="track"><i style="width:{w:.1f}%;background:linear-gradient(90deg,#0ea5e9,#5eead4)"></i></span>'
            f'<span class="v">{count} · {pct:.0f}%</span></div>'
        )

    # ── sectors ─────────────────────────────────────────────────────────────
    sect: dict[str, dict] = {}
    for t in engine.trades:
        d_ = sect.setdefault(t.sector, {"n": 0, "pnl": 0.0, "w": 0})
        d_["n"] += 1
        d_["pnl"] += t.pnl
        d_["w"] += 1 if t.pnl > 0 else 0
    max_abs_sp = max((abs(s["pnl"]) for s in sect.values()), default=1.0) or 1.0
    sector_html = ""
    for name, s in sorted(sect.items(), key=lambda kv: -kv[1]["pnl"]):
        c = "var(--pos)" if s["pnl"] >= 0 else "var(--neg)"
        w = abs(s["pnl"]) / max_abs_sp * 100
        sector_html += (
            f'<div class="hbar"><span title="{esc(str(name))}"><b>{esc(str(name))}</b>'
            f'<span class="muted" style="font-size:.85em"> {s["n"]}t · {s["w"] / s["n"] * 100:.0f}%</span></span>'
            f'<span class="track"><i style="width:{w:.1f}%;background:{c}"></i></span>'
            f'<span class="v" style="color:{c}">{s["pnl"]:+,.0f}</span></div>'
        )

    # ── leaderboard ─────────────────────────────────────────────────────────
    by_pnl = sorted(engine.trades, key=lambda t: -t.pnl_pct)
    top5 = by_pnl[:5]
    worst5 = by_pnl[-5:][::-1]

    def lead_rows(ts):
        out = ""
        for i, t in enumerate(ts, 1):
            c = "pos" if t.pnl_pct >= 0 else "neg"
            out += (
                f'<div class="lrow"><span class="rank">#{i}</span>'
                f'<span class="tk">{esc(t.ticker)}<small>{esc(t.sector)} · '
                f'{t.entry_date.strftime("%y-%m-%d")}→{t.exit_date.strftime("%y-%m-%d")}</small></span>'
                f'<span class="pc {c}">{t.pnl_pct:+.1f}%</span></div>'
            )
        return out

    lead_html = (
        f'<div><div class="ptitle" style="margin-bottom:10px">'
        f'<span class="idx">WIN</span><h2 style="font-size:.9em">Top winners</h2></div>'
        f'<div class="lead" style="grid-template-columns:1fr">{lead_rows(top5)}</div></div>'
        f'<div><div class="ptitle" style="margin-bottom:10px">'
        f'<span class="idx" style="color:var(--neg);background:#2a0f17;border-color:#7f1d2e">LOSS</span>'
        f'<h2 style="font-size:.9em">Top losers</h2></div>'
        f'<div class="lead" style="grid-template-columns:1fr">{lead_rows(worst5)}</div></div>'
    )

    # ── win donut + histogram ───────────────────────────────────────────────
    wr = m["win_rate"]
    circ = 2 * 3.14159265 * 52
    dash = circ * wr / 100
    donut = (
        f'<div class="donut"><svg width="150" height="150" viewBox="0 0 150 150">'
        f'<circle cx="75" cy="75" r="52" fill="none" stroke="#1a2532" stroke-width="15"/>'
        f'<circle cx="75" cy="75" r="52" fill="none" stroke="url(#wg)" stroke-width="15" '
        f'stroke-linecap="round" stroke-dasharray="{dash:.1f} {circ:.1f}" '
        f'transform="rotate(-90 75 75)"/>'
        f'<defs><linearGradient id="wg" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="#34d399"/><stop offset="1" stop-color="#38bdf8"/>'
        f"</linearGradient></defs></svg>"
        f'<div class="mid"><b>{wr:.1f}%</b><span>win rate</span></div></div>'
    )
    statlist = (
        f'<div class="statlist">'
        f"<div><span>Avg win</span><b class=\"pos\">{m['avg_win_pct']:+.1f}%</b></div>"
        f"<div><span>Avg loss</span><b class=\"neg\">{m['avg_loss_pct']:+.1f}%</b></div>"
        f"<div><span>Best</span><b class=\"pos\">{best_trade:+.1f}%</b></div>"
        f"<div><span>Worst</span><b class=\"neg\">{worst_trade:+.1f}%</b></div>"
        f"<div><span>Median</span><b>{median_trade:+.1f}%</b></div>"
        f"<div><span>Avg hold</span><b>{avg_hold:.0f}d</b></div>"
        f"</div>"
    )

    edges = [(-1e9, -30), (-30, -15), (-15, -5), (-5, 0), (0, 5),
             (5, 15), (15, 30), (30, 50), (50, 100), (100, 1e9)]
    labels = ["<-30", "-30..-15", "-15..-5", "-5..0", "0..5",
              "5..15", "15..30", "30..50", "50..100", ">100"]
    counts = [0] * len(edges)
    for p in pnl_pcts:
        for i, (lo, hi) in enumerate(edges):
            if lo < p <= hi:
                counts[i] += 1
                break
    max_c = max(counts, default=1) or 1
    hist_svg = ['<svg viewBox="0 0 1000 230" class="hist" font-family="ui-monospace,monospace">']
    for i, (cnt, lab) in enumerate(zip(counts, labels)):
        bw = 1000 / len(counts)
        x = i * bw + 8
        h = cnt / max_c * 160
        y = 185 - h
        col = "#fb7185" if i < 4 else "#34d399"
        if cnt:
            hist_svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 16:.1f}" height="{h:.1f}" '
                f'rx="6" fill="{col}" opacity=".82"><title>{lab}: {cnt} trades</title></rect>'
            )
            hist_svg.append(
                f'<text x="{x + (bw - 16) / 2:.1f}" y="{y - 6:.1f}" fill="#c9d6e2" '
                f'font-size="13" text-anchor="middle">{cnt}</text>'
            )
        else:
            hist_svg.append(
                f'<rect x="{x:.1f}" y="183" width="{bw - 16:.1f}" height="2" rx="1" fill="#1a2532"/>'
            )
        hist_svg.append(
            f'<text x="{x + (bw - 16) / 2:.1f}" y="207" fill="#5c7186" font-size="11.5" '
            f'text-anchor="middle">{lab}</text>'
        )
    hist_svg.append(
        '<text x="500" y="226" fill="#5c7186" font-size="11" text-anchor="middle">'
        "trade P&amp;L %</text></svg>"
    )
    hist_html = "".join(hist_svg)

    # ── config chips ────────────────────────────────────────────────────────
    cfg_html = ""
    for k in _CFG_KEYS:
        if k in settings:
            cfg_html += (
                f'<div class="c"><span class="k">{k.replace("_", " ").capitalize()}</span>'
                f'<span class="v">{esc(str(settings[k]))}</span></div>'
            )

    # ── per-stock table ─────────────────────────────────────────────────────
    sector_of = {t.ticker: t.sector for t in engine.trades}
    stock_rows = ""
    for ticker in sorted(m["stock_stats"], key=lambda t: -m["stock_stats"][t]["total_pnl"]):
        s = m["stock_stats"][ticker]
        c = "pos" if s["total_pnl"] > 0 else "neg"
        stock_rows += (
            f'<tr><td><b>{esc(ticker)}</b></td>'
            f'<td class="muted">{esc(str(sector_of.get(ticker, "-")))}</td>'
            f'<td class="tnum" data-num="{s["trades"]}">{s["trades"]}</td>'
            f'<td class="tnum" data-num="{s["win_rate"]}">{s["win_rate"]:.0f}%</td>'
            f'<td class="tnum {c}" data-num="{s["total_pnl"]}">Rs.{s["total_pnl"]:+,.0f}</td>'
            f'<td class="tnum {c}" data-num="{s["avg_pnl_pct"]}">{s["avg_pnl_pct"]:+.1f}%</td></tr>'
        )

    # ── trade log (filterable + sortable + sparklines) ───────────────────────
    stock_data_for_charts = {st.ticker: st.df for st in engine.stocks}
    trade_rows = ""
    for t in sorted(engine.trades, key=lambda x: x.entry_date):
        c = "pos" if t.pnl > 0 else "neg"
        sbg = "#34d399" if t.entry_score >= 70 else "#facc15" if t.entry_score >= 50 else "#7d93a8"
        chart_svg = _generate_trade_chart(t, stock_data_for_charts)
        trade_rows += (
            f'<tr data-t="{esc(t.ticker.upper())}" data-y="{t.entry_date.year}" '
            f'data-r="{esc(t.exit_reason)}" data-p="{t.pnl_pct:.2f}">'
            f'<td><b>{esc(t.ticker)}</b><br>'
            f'<span class="muted" style="font-size:.8em">{esc(t.sector)}</span></td>'
            f'<td class="tnum" data-num="{t.entry_date.toordinal()}">{t.entry_date.strftime("%Y-%m-%d")}'
            f'<br><span class="muted" style="font-size:.8em">Rs.{t.entry_price:,.0f}</span></td>'
            f'<td data-num="{t.entry_score}"><span class="score" style="background:{sbg}">{t.entry_score:.0f}</span></td>'
            f'<td class="tnum" data-num="{t.exit_date.toordinal()}">{t.exit_date.strftime("%Y-%m-%d")}'
            f'<br><span class="muted" style="font-size:.8em">Rs.{t.exit_price:,.0f}</span></td>'
            f'<td class="tnum" data-num="{t.days_held}">{t.days_held}d</td>'
            f'<td class="tnum {c}" data-num="{t.pnl_pct}"><b>{t.pnl_pct:+.1f}%</b></td>'
            f'<td class="tnum {c}" data-num="{t.pnl}">{t.pnl:+,.0f}</td>'
            f'<td><span class="muted" style="font-size:.85em">{esc(t.exit_reason)}</span></td>'
            f'<td>{chart_svg}</td></tr>'
        )

    years_opts = sorted({t.entry_date.year for t in engine.trades}, reverse=True)
    year_opts_html = '<option value="">all years</option>' + "".join(
        f'<option value="{y}">{y}</option>' for y in years_opts
    )
    reason_opts_html = '<option value="">all reasons</option>' + "".join(
        f'<option value="{esc(str(r))}">{esc(str(r))} ({c})</option>' for r, c in exit_items
    )

    # ── equity + drawdown charts (viewBox 1000-wide for JS crosshair) ───────
    if len(eq_vals) > 1:
        W, H = 1000, 320
        mL, mR, mT, mB = 92, 16, 18, 34
        iw, ih = W - mL - mR, H - mT - mB
        min_v, max_v = min(eq_vals), max(eq_vals)
        v_range = max_v - min_v or 1.0
        n = len(eq_vals)

        def sx(i):
            return mL + i / max(n - 1, 1) * iw

        def sy(v):
            return mT + (1 - (v - min_v) / v_range) * ih

        eqp = [" ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(eq_vals))]
        area_pts = f"{sx(0):.1f},{sy(min_v):.1f} {eqp[0]} {sx(n - 1):.1f},{sy(min_v):.1f}"
        grid = ""
        for i in range(5):
            gv = min_v + i / 4 * v_range
            gy = sy(gv)
            grid += (
                f'<line x1="{mL}" y1="{gy:.1f}" x2="{W - mR}" y2="{gy:.1f}" stroke="#182430" stroke-width="0.7"/>'
                f'<text x="{mL - 8}" y="{gy + 4:.1f}" fill="#5c7186" font-size="12" text-anchor="end">Rs.{_inr(gv)}</text>'
            )
        span_days = (eq_days[-1] - eq_days[0]).days if len(eq_days) > 1 else 0
        fmt = "%b %Y" if span_days < 730 else "%Y"
        xticks = ""
        for i in range(6):
            idx = int(i / 5 * (n - 1))
            xticks += (
                f'<text x="{sx(idx):.1f}" y="{H - 10}" fill="#5c7186" font-size="12" '
                f'text-anchor="middle">{eq_days[idx].strftime(fmt)}</text>'
            )
        ic_y = sy(m["initial_capital"])
        equity_svg = (
            f'<svg viewBox="0 0 {W} {H}" data-m="{mL}" data-w="{W - mR - mL}" font-family="ui-monospace,monospace">'
            f'<defs><linearGradient id="eqFill" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="#34d399" stop-opacity=".28"/>'
            f'<stop offset="1" stop-color="#34d399" stop-opacity="0"/></linearGradient></defs>'
            f"{grid}{xticks}"
            f'<polygon points="{area_pts}" fill="url(#eqFill)"/>'
            f'<line x1="{mL}" y1="{ic_y:.1f}" x2="{W - mR}" y2="{ic_y:.1f}" '
            f'stroke="#7d93a8" stroke-width="1" stroke-dasharray="6" opacity=".7"/>'
            f'<text x="{W - mR}" y="{ic_y - 6:.1f}" fill="#7d93a8" font-size="11" text-anchor="end">initial</text>'
            f'<polyline points="{" ".join(eqp)}" fill="none" stroke="#34d399" stroke-width="2.2"/>'
            f'<line class="cross" x1="0" y1="0" x2="0" y2="{H}" stroke="#5eead4" '
            f'stroke-width="1" style="opacity:0"/>'
            f"</svg>"
        )

        dds_min = min(dds, default=-0.01) or -0.01
        dH = 170
        dT, dB = 14, 30

        def ddy(d):
            return dT + (0 - d) / (0 - dds_min) * (dH - dT - dB)

        dpts = " ".join(f"{sx(i):.1f},{ddy(d):.1f}" for i, d in enumerate(dds))
        darea = f"{sx(0):.1f},{ddy(0):.1f} {dpts} {sx(n - 1):.1f},{ddy(0):.1f}"
        min_i = dds.index(dds_min)
        dd_svg = (
            f'<svg viewBox="0 0 {W} {dH}" data-m="{mL}" data-w="{W - mR - mL}" font-family="ui-monospace,monospace">'
            f'<polygon points="{darea}" fill="rgba(251,113,133,.18)"/>'
            f'<polyline points="{dpts}" fill="none" stroke="#fb7185" stroke-width="1.6"/>'
            f'<line x1="{mL}" y1="{ddy(0):.1f}" x2="{W - mR}" y2="{ddy(0):.1f}" '
            f'stroke="#5c7186" stroke-width="0.8" stroke-dasharray="5"/>'
            f'<text x="{mL - 8}" y="{ddy(0) + 4:.1f}" fill="#5c7186" font-size="12" text-anchor="end">0%</text>'
            f'<text x="{mL - 8}" y="{ddy(dds_min) + 4:.1f}" fill="#5c7186" font-size="12" '
            f'text-anchor="end">{dds_min * 100:.0f}%</text>'
            f'<circle cx="{sx(min_i):.1f}" cy="{ddy(dds_min):.1f}" r="4" fill="#fb7185"/>'
            f'<text x="{sx(min_i):.1f}" y="{ddy(dds_min) + 20:.1f}" fill="#fb7185" font-size="12" '
            f'text-anchor="middle">max {abs(dds_min) * 100:.1f}%</text>'
            f'<line class="cross" x1="0" y1="0" x2="0" y2="{dH}" stroke="#5eead4" '
            f'stroke-width="1" style="opacity:0"/>'
            f"</svg>"
        )
        charts_html = (
            '<div class="panel"><div class="chart" id="eqChart">'
            f'{equity_svg}<div class="tip"></div></div>'
            '<div class="legend"><span><i class="dot" style="background:#34d399"></i>equity</span>'
            '<span><i class="dot" style="background:#7d93a8"></i>initial capital</span>'
            '<span class="muted">hover for crosshair</span></div></div>'
            '<div class="panel" style="margin-top:14px"><div class="ptitle">'
            '<span class="idx">DD</span><h2 style="font-size:.9em">Underwater — drawdown from peak</h2></div>'
            '<div class="chart" id="ddChart">' + dd_svg + '<div class="tip"></div></div></div>'
        )
    else:
        charts_html = '<div class="panel"><p class="muted">No equity data</p></div>'
        eq_json = "[]"

    # ── assemble ────────────────────────────────────────────────────────────
    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<script>document.documentElement.classList.add('js')</script>
<title>Backtest — {esc(label)} · {tf}</title>
<style>{_CSS}</style>
</head>
<body>

<nav><div class="navin">
  <span class="brand">HMA<em>×</em>EMA · Backtest</span>
  <span class="navlinks">
    <a href="#overview" class="active">Overview</a><a href="#equity">Equity</a>
    <a href="#calendar">Calendar</a><a href="#breakdown">Breakdown</a>
    <a href="#leaders">Leaders</a><a href="#log">Trade log</a><a href="#config">Config</a>
  </span>
  <span class="pill" style="margin-left:auto"><b>{esc(label)}</b> · {tf} · {trigger}</span>
</div><i class="prog" id="prog"></i></nav>

<div class="wrap">

<header class="hero">
  <div class="heroText">
    <div class="kicker">Strategy performance report</div>
    <h1>HMA×EMA Swing <span>Backtest</span></h1>
    <div class="sub">
      <span class="pill"><b>{esc(label)}</b></span>
      <span class="pill">{tf} bars · {esc(str(trigger))} trigger</span>
      <span class="pill">{sim_start} → {sim_end}</span>
      <span class="pill"><b>{m["years"]:.1f}</b> years</span>
      <span class="pill">generated {generated}</span>
    </div>
  </div>
  {hero_svg}
</header>

<section id="overview">
  <div class="ptitle"><span class="idx">01</span><h2>Scoreboard</h2>
    <span class="hint">hover a card · numbers animate on load</span></div>
  <div class="kpis">{kpis}</div>
</section>

<section id="equity">
  <div class="ptitle"><span class="idx">02</span><h2>Equity &amp; drawdown</h2>
    <span class="hint">move the mouse over a chart</span></div>
  {charts_html}
</section>

<section id="calendar">
  <div class="ptitle"><span class="idx">03</span><h2>Monthly return calendar</h2>
    <span class="hint">green = up month · red = down · hover for exact %</span></div>
  <div class="panel hm-scroll"><table class="hm">{hm_head}{hm_rows}</table>
    <div class="hmleg"><span>-{max_abs_m:.1f}%</span><span class="g"></span><span>+{max_abs_m:.1f}%</span>
      <span style="margin-left:auto">hover a cell for the exact month</span></div>
  </div>
</section>

<section id="breakdown">
  <div class="ptitle"><span class="idx">04</span><h2>Breakdown</h2>
    <span class="hint">click column headers to sort</span></div>
  <div class="cols">
    <div class="panel">
      <div class="ptitle"><span class="idx">YR</span><h2 style="font-size:.9em">Yearly returns</h2></div>
      <div class="tscroll"><table class="sortable zebra">
        <thead><tr><th>Year</th><th>Return</th><th></th><th>Equity</th><th>Trades</th><th>Win%</th></tr></thead>
        <tbody>{yearly_rows}</tbody></table></div>
    </div>
    <div class="panel">
      <div class="ptitle"><span class="idx">EX</span><h2 style="font-size:.9em">Exit reasons</h2></div>
      {exit_rows}
    </div>
  </div>
  <div class="cols" style="margin-top:18px">
    <div class="panel">
      <div class="ptitle"><span class="idx">DS</span><h2 style="font-size:.9em">Win rate &amp; trade distribution</h2></div>
      <div class="donutwrap">{donut}{statlist}</div>
      <div style="margin-top:16px">{hist_html}</div>
    </div>
    <div class="panel">
      <div class="ptitle"><span class="idx">SC</span><h2 style="font-size:.9em">Sector P&amp;L</h2></div>
      {sector_html}
    </div>
  </div>
</section>

<section id="leaders">
  <div class="ptitle"><span class="idx">05</span><h2>Leaderboard</h2></div>
  <div class="cols">{lead_html}</div>
  <div class="panel" style="margin-top:18px">
    <div class="ptitle"><span class="idx">PS</span><h2 style="font-size:.9em">Per-stock performance</h2></div>
    <div class="tscroll tall"><table class="sortable zebra">
      <thead><tr><th>Stock</th><th>Sector</th><th>Trades</th><th>Win%</th>
        <th>Total P&amp;L</th><th>Avg P&amp;L%</th></tr></thead>
      <tbody>{stock_rows}</tbody></table></div>
  </div>
</section>

<section id="log">
  <div class="ptitle"><span class="idx">06</span><h2>Trade log</h2>
    <span class="hint">{len(engine.trades)} trades · filter &amp; sort</span></div>
  <div class="panel">
    <div class="filters">
      <input id="fQ" type="search" placeholder="🔍  search ticker…" spellcheck="false">
      <select id="fYear">{year_opts_html}</select>
      <select id="fReason">{reason_opts_html}</select>
      <select id="fOutcome"><option value="">wins + losses</option>
        <option value="w">winners only</option><option value="l">losers only</option></select>
      <span class="fcount" id="fCount"></span>
    </div>
    <div class="tscroll tall"><table class="sortable">
      <thead><tr><th>Stock</th><th>Entry</th><th>Score</th><th>Exit</th><th>Held</th>
        <th>P&amp;L%</th><th>P&amp;L Rs.</th><th>Reason</th><th>Path</th></tr></thead>
      <tbody id="tradeBody">{trade_rows}</tbody></table></div>
    <div class="fempty" id="fEmpty" style="display:none">No trades match the current filters.</div>
  </div>
</section>

<section id="config">
  <div class="ptitle"><span class="idx">07</span><h2>Run configuration</h2></div>
  <div class="panel"><div class="cfg">{cfg_html}</div></div>
</section>

<footer>
  <span>HMA×EMA Multi-Score Swing · Pine V4 port</span>
  <span>{esc(label)} · {sim_start} → {sim_end} · generated {generated}</span>
</footer>

</div>
<button id="toTop" title="Back to top" aria-label="Back to top">↑</button>
<script>window.EQ = {eq_json};</script>
<script>{_JS}</script>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"\n  HTML report saved: {filepath}")


def save_trades_csv(
    trades: list[TradeResult], filepath: str = "Reports/backtest_trades.csv"
):
    """Save all trades to a CSV file."""
    rows = [
        {
            "ticker": t.ticker,
            "sector": t.sector,
            "entry_date": t.entry_date.strftime("%Y-%m-%d"),
            "entry_price": round(t.entry_price, 2),
            "entry_score": round(t.entry_score, 1),
            "exit_date": t.exit_date.strftime("%Y-%m-%d"),
            "exit_price": round(t.exit_price, 2),
            "exit_reason": t.exit_reason,
            "shares": t.shares,
            "pnl": round(t.pnl, 2),
            "pnl_pct": round(t.pnl_pct, 2),
            "days_held": t.days_held,
            "investment": round(t.investment, 2),
        }
        for t in trades
    ]

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"  Trades CSV saved: {filepath}")


_SCAN_CSS = r"""
.scankpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.scankpi{background:linear-gradient(180deg,#101822,#0c1218);border:1px solid var(--line);
  border-radius:14px;padding:14px 16px;position:relative;overflow:hidden}
.scankpi::before{content:'';position:absolute;inset:0 0 auto;height:2px;
  background:linear-gradient(90deg,var(--kc,#5eead4),transparent 85%);opacity:.6}
.scankpi .lab{font-size:.66em;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.scankpi b{display:block;font-size:1.8em;font-family:ui-monospace,monospace;margin-top:4px}
.triggrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px}
.trigcard{background:linear-gradient(150deg,#0d2320,#0c1218 60%);border:1px solid #14524c;
  border-radius:16px;padding:16px 18px;transition:.18s}
.trigcard:hover{transform:translateY(-2px);box-shadow:0 14px 30px rgba(0,0,0,.45)}
.trigcard .tk{font-size:1.25em;font-weight:800}
.trigcard .tk small{font-size:.55em;font-weight:400;color:var(--muted);margin-left:8px}
.trigcard .row{display:flex;justify-content:space-between;font-size:.8em;
  color:var(--muted);margin-top:8px;font-family:ui-monospace,monospace}
.trigcard .row b{color:var(--txt)}
.gchip{display:inline-block;padding:2px 9px;border-radius:99px;font-size:.68em;
  border:1px solid #14524a;color:var(--pos);background:#0b1f1e;margin-top:9px}
.nbar{background:#111a24;border-radius:99px;height:7px;width:90px;display:inline-block;
  overflow:hidden;vertical-align:middle;margin-left:8px}
.nbar i{display:block;height:100%;background:linear-gradient(90deg,#0ea5e9,#5eead4);border-radius:99px}
.scorechip{display:inline-block;min-width:34px;text-align:center;padding:2px 7px;border-radius:7px;
  font-weight:700;font-size:.9em;color:#08131a;font-family:ui-monospace,monospace}
.empty{color:var(--dim);font-size:.85em;padding:6px 0}
"""


def generate_scan_html(
    rows: list,
    settings: dict,
    meta: dict,
    filepath: str = "Reports/scan_v2_latest.html",
):
    """Self-contained dark dashboard for scan_latest() output."""
    import html as html_mod

    esc = html_mod.escape
    s = settings
    as_of = meta.get("as_of")
    as_of_s = as_of.strftime("%Y-%m-%d") if as_of is not None else "-"
    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    confirm = int(s.get("confirm_weeks", 15))
    label = s.get("universe_label", "-")
    tf = s.get("timeframe", "daily")
    trigger = s.get("entry_trigger", "close")
    fill_note = "buy-stop fill at level" if trigger == "highbreak" else "fill next open"

    trig = sorted(
        (r for r in rows if r["state"] == "TRIGGERED"),
        key=lambda r: -(r["score"] or 0),
    )
    armed = sorted(
        (r for r in rows if r["state"] == "ARMED"), key=lambda r: -r["n"]
    )
    expired = sorted(
        (r for r in rows if r["state"] == "EXPIRED"),
        key=lambda r: r["cross_date"],
        reverse=True,
    )

    def _scorechip(score):
        if score is None:
            return '<span class="scorechip" style="background:#7d93a8">-</span>'
        bg = "#34d399" if score >= 70 else "#facc15" if score >= 50 else "#7d93a8"
        return f'<span class="scorechip" style="background:{bg}">{score:.0f}</span>'

    trig_html = ""
    for r in trig:
        lvl = (
            f'→ <b>Rs.{r["level"]:,.1f}</b>' if r["level"] is not None else ""
        )
        trig_html += (
            f'<div class="trigcard"><div class="tk">{esc(r["ticker"])}'
            f'<small>{esc(str(r["sector"]))}</small></div>'
            f'<div class="row"><span>score {_scorechip(r["score"])}</span>'
            f'<span>n={r["n"]}/{confirm}w</span></div>'
            f'<div class="row"><span>cross</span>'
            f'<b>{r["cross_date"].strftime("%Y-%m-%d")}</b></div>'
            f'<div class="row"><span>close Rs.{r["close"]:,.1f}</span>'
            f"<span>{lvl}</span></div>"
            f'<div class="gchip">{esc(r["note"])} &middot; {fill_note}</div></div>'
        )
    if not trig_html:
        trig_html = '<p class="empty">No stock clears every entry gate right now.</p>'

    armed_html = ""
    for r in armed:
        w = min(r["n"] / confirm * 100, 100) if confirm else 0
        sc = f'{r["score"]:.0f}' if r["score"] is not None else "-"
        armed_html += (
            f'<tr><td><b>{esc(r["ticker"])}</b><br>'
            f'<span class="muted" style="font-size:.8em">{esc(str(r["sector"]))}</span></td>'
            f'<td>{_scorechip(r["score"])}</td>'
            f'<td class="tnum"><b>{r["n"]}</b>/{confirm}w'
            f'<span class="nbar"><i style="width:{w:.0f}%"></i></span></td>'
            f'<td class="tnum">{r["cross_date"].strftime("%Y-%m-%d")}</td>'
            f'<td class="tnum">Rs.{r["close"]:,.1f}</td>'
            f'<td class="muted">{esc(r["note"])}</td></tr>'
        )
    if not armed_html:
        armed_html = (
            '<tr><td colspan="6" class="muted">No live bullish crossovers '
            "inside the confirm window.</td></tr>"
        )

    expired_html = ""
    for r in expired:
        expired_html += (
            f'<tr><td><b>{esc(r["ticker"])}</b></td>'
            f'<td class="muted">{esc(str(r["sector"]))}</td>'
            f'<td class="tnum">{r["cross_date"].strftime("%Y-%m-%d")}</td>'
            f'<td class="muted">{esc(r["note"])}</td></tr>'
        )
    if not expired_html:
        expired_html = (
            '<tr><td colspan="4" class="muted">None — no recent drops.</td></tr>'
        )

    cfg_html = ""
    for k in _CFG_KEYS:
        if k in s:
            cfg_html += (
                f'<div class="c"><span class="k">{k.replace("_", " ").capitalize()}</span>'
                f'<span class="v">{esc(str(s[k]))}</span></div>'
            )

    def _kpi(lab, val, hc):
        return (
            f'<div class="scankpi" style="--kc:{hc}"><div class="lab">{lab}</div>'
            f"<b>{val}</b></div>"
        )

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Scan — {esc(label)} · {as_of_s}</title>
<style>{_CSS}{_SCAN_CSS}</style>
</head>
<body>
<nav><div class="navin">
  <span class="brand">HMA<em>×</em>EMA · Live Scan</span>
  <span class="pill" style="margin-left:auto"><b>{esc(label)}</b> · {tf} · {trigger} · as of {as_of_s}</span>
</div></nav>

<div class="wrap">

<header class="hero" style="grid-template-columns:1fr">
  <div class="heroText">
    <div class="kicker">Latest crossovers &amp; entries</div>
    <h1>Live <span>Scan</span></h1>
    <div class="sub">
      <span class="pill"><b>{esc(label)}</b></span>
      <span class="pill">{tf} · confirm <b>{confirm}w</b></span>
      <span class="pill">score ≥ <b>{s.get("score_threshold", 50)}</b></span>
      <span class="pill">{trigger} trigger</span>
      <span class="pill">as of {as_of_s}</span>
      <span class="pill">generated {generated}</span>
    </div>
  </div>
</header>

<section id="overview" style="opacity:1;transform:none">
  <div class="scankpis">
    {_kpi("Entry now", len(trig), "#34d399")}
    {_kpi("Armed", len(armed), "#38bdf8")}
    {_kpi("Recently expired", len(expired), "#fb7185")}
    {_kpi("Scanned", meta.get("scanned", len(rows)), "#facc15")}
    {_kpi("Crosses in window", meta.get("crosses_in_window", 0), "#5eead4")}
  </div>
</section>

<section id="trig" style="opacity:1;transform:none;margin-top:28px">
  <div class="ptitle"><span class="idx">01</span><h2>Entry now — all gates green</h2>
    <span class="hint">{fill_note}</span></div>
  <div class="triggrid">{trig_html}</div>
</section>

<section id="armed" style="opacity:1;transform:none;margin-top:28px">
  <div class="ptitle"><span class="idx">02</span><h2>Armed — bullish cross, waiting</h2>
    <span class="hint">n = weeks since the crossover · expiry at {confirm}w</span></div>
  <div class="panel"><div class="tscroll"><table>
    <thead><tr><th>Stock</th><th>Score</th><th>Confirm</th><th>Cross</th>
      <th>Close</th><th>Waiting on</th></tr></thead>
    <tbody>{armed_html}</tbody></table></div></div>
</section>

<section id="expired" style="opacity:1;transform:none;margin-top:28px">
  <div class="ptitle"><span class="idx">03</span><h2>Recently expired</h2>
    <span class="hint">armed crosses that timed out or were cancelled</span></div>
  <div class="panel"><div class="tscroll"><table>
    <thead><tr><th>Stock</th><th>Sector</th><th>Cross</th><th>Dropped</th></tr></thead>
    <tbody>{expired_html}</tbody></table></div></div>
</section>

<section id="config" style="opacity:1;transform:none;margin-top:28px">
  <div class="ptitle"><span class="idx">04</span><h2>Run configuration</h2></div>
  <div class="panel"><div class="cfg">{cfg_html}</div></div>
</section>

<footer>
  <span>HMA×EMA Multi-Score Swing · live scan (signal state, no portfolio)</span>
  <span>{esc(label)} · as of {as_of_s} · generated {generated}</span>
</footer>

</div>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"  Scan HTML saved: {filepath}")
