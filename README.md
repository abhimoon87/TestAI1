# HMAxEMA Stock Scanner

A Python desktop application that screens **Indian stocks (NSE/BSE)** for swing-trading setups using an **HMA × EMA crossover** strategy with a **10-factor, 100-point scoring engine** — a Python re-implementation of the TradingView Pine Script in [`HMA_EMA_Swing_Strategy_v2.pine`](HMA_EMA_Swing_Strategy_v2.pine).

The app includes a dark-themed desktop GUI (Flet), an interactive CLI, a headless scan engine, HTML/CSV report export, and a full backtesting engine.

---

## Features

- **Scan universes** — NIFTY 50, BANK NIFTY, sector indices, FnO stocks, BSE indices, or the **live full market (~5,900 NSE+BSE symbols)** resolved dynamically from NSE.
- **Multi-model pipeline**:
  1. *Stock filter* — recent bullish MA crossover
  2. *Direction* — Bull / Bear
  3. *10-category scoring* — Trend, Momentum, RSI, MACD, Stochastic, OBV, Volume, Relative Strength, Volatility, Fundamentals (0–100)
- **Resilient data layer** — multi-provider fallback chains with disk caching, chunked batch downloads, and a per-ticker NSE fallback when Yahoo rate-limits a full-market scan.
- **Optional enrichment** — news/social sentiment, FII/DII delivery data, 52-week position, macro/forex/crypto regime, insider signals, Shariah compliance (all optional; most need free API keys).
- **HTML report** with sortable/filterable table, score bars and per-stock news sentiment; CSV export.
- **Backtesting engine** — walk-forward simulation of the entry/exit rules with position sizing, stop/target/trailing stops, ATR stops and sector rotation.
- **Tracing & logging** — rotating `scanner/trace.log`, `scanner/scan.log`, plus an uncaught-exception hook.

---

## Installation

Requires **Python 3.10+**.

```bash
pip install -r scanner/requirements.txt
```

The GUI also needs `Pillow` (already listed above) and a display server.

---

## Quick start

| What | How |
|---|---|
| Launch the GUI | `python -m scanner` (Windows: double-click `scanner/run.bat`, macOS/Linux: `scanner/run.sh`) |
| Interactive CLI scan | `python -m scanner --cli` or `python scanner/run_scanner.py` |
| Headless scan from code | `ScannerEngine().scan(...)` — see `scanner/scanner_engine.py` |
| Backtest NIFTY Alpha 50 | `python run_alpha_backtest.py` |
| Backtest with options | `python -m scanner.backtest --years 3` |

In the GUI: pick a **universe**, **timeframe** (Daily/Weekly/Monthly), **data period**, optional **trend filter**, and a **min score threshold**, then press **RUN SCAN**. Results stream into the table incrementally; use the header row to sort, the search box to filter, and the top-bar buttons to export **HTML** or **CSV**.

Settings chosen in the GUI are saved to `scanner/settings.json`.

---

## Architecture

```
HMA_EMA_Swing_Strategy_v2.pine   ← strategy spec (TradingView reference)

scanner/__main__.py ─┬─> scanner/app.py           Flet GUI ("Aurora") — scan wiring, page
                     │                             layout, pagination, exports, logging
                     │        views_layout.py       dashboard panes: rail / sidebar / main /
                     │                             right panel (+ summary & top-pick cards)
                     │        views_results.py      results grid (rows, sort), chart, news rows
                     │        views_settings.py     settings page + input builder
                     │        ui_kit.py             shared Flet primitives (view mixins use it)
                     └─> scanner/run_scanner.py   interactive CLI
run_alpha_backtest.py ─> scanner/backtest.py      backtesting engine

                              ┌────────────────────────────┐
                              │ scanner/scanner_engine.py   │  headless pipeline
                              │ universe → index → batch →  │  with cancel + progress
                              │ filter → enrich → score     │  callbacks
                              └──────────────┬─────────────┘
                                             │
        ┌────────────────────────────────────┼───────────────────────────────┐
        ▼                                    ▼                               ▼
universes.py                          data_fetcher.py                 scoring.py
static universes + sector map         chunked yfinance batch          filter + 10-category
symbol_fetcher.py                     (+ NSE fallback pass)           scoring, sideways
live NSE/BSE symbol lists             data_providers.py               filter, entry signals
                                      3-tier OHLCV fallback:          indicators.py
                                      jugaad-data → yfinance →        vectorized HMA/EMA/SMA/
                                      nselib (disk cache 4h)          KAMA/VWMA/RSI/MACD/ATR/
                                      fundamentals: Finnhub →         ADX/OBV/VP-POC
                                      Alpha Vantage → yfinance →
                                      nselib
```

### Module map

| Module | Responsibility |
|---|---|
| `scanner/app.py` | Flet GUI — `ScannerApp` wires the view mixins below; owns app lifecycle, scanning/events, cache UI, HTML/CSV export and the activity log |
| `scanner/views_layout.py` | `LayoutViewMixin` — dashboard panes (rail, sidebar, main area, right panel), summary + top-pick cards |
| `scanner/views_results.py` | `ResultsViewMixin` — paginated results grid, data-row/header builders, sorting keys, score chart, row-level news expansion, summary/hero updates |
| `scanner/views_settings.py` | `SettingsViewMixin` — declarative settings spec + settings-page builder/inputs |
| `scanner/ui_kit.py` | Shared Flet primitives (`_border_all`, `_glass_bg`, ...), `RESULT_COLS`, `_score_of` |
| `scanner/scanner_engine.py` | Headless scan orchestration (`scan`, `scan_stream`) shared by GUI/CLI; "fast mode" for >500 tickers (technicals first, enrich top 200) |
| `scanner/scoring.py` | Stock filter (MA crossover), Bull/Bear direction, 10-category scoring, weekly-HTF check, sideways filter, entry signals |
| `scanner/indicators.py` | Vectorized pandas/numpy indicators used by scoring & backtest |
| `scanner/data_fetcher.py` | Batch downloads (200/chunk, 8 parallel), weekly/monthly resampling, batch **fallback pass** via NSE providers |
| `scanner/data_providers.py` | `DataProvider` class with provider fallback chains + disk cache |
| `scanner/symbol_fetcher.py` | Live NSE/BSE symbol lists (4 h cache, static fallbacks) |
| `scanner/universes.py` | ~20 static universes, sector map/colors, live-universe resolution |
| `scanner/backtest.py` | Backtest engine (`python -m scanner.backtest`) |
| `scanner/report.py` | Self-contained HTML report generator + keyword news sentiment |
| `scanner/settings_store.py` | Canonical `DEFAULT_SETTINGS`, settings persistence, API-key registry/loading |
| `scanner/trace.py` | Rotating trace log, `@trace` decorator, custom TRACE level |
| `scanner/cache.py` | Shared in-memory TTL cache |
| `scanner/themes.py` | Light/dark theme definitions |

Enrichment modules (all optional, all guarded): `market_sentiment.py`, `social_sentiment.py`, `indian_market.py`, `indian_fundamentals.py`, `insider_data.py`, `macro_data.py`, `free_apis.py`, `premium_finance.py`.

### Scan pipeline (end to end)

1. **Resolve universe** — static list, or live NSE/BSE symbol fetch (falls back to static if offline).
2. **Fetch index** — NIFTY via `jugaad-data → yfinance` for relative-strength comparison.
3. **Batch download** — OHLCV for every ticker via `yfinance` in 200-symbol chunks (8 parallel, throttled). Tickers yfinance misses are retried (8 parallel, per-provider 10 s timeout) via **jugaad-data → nselib** — yfinance is deliberately skipped there since it just failed at scale. When a scan misses many tickers the recovery pass is filtered against nselib's NSE mainboard list, so BSE-only symbols (which neither NSE provider serves) are skipped instead of burning two failed calls each.
4. **Global enrichment (once)** — macro regime, forex, crypto fear/greed, commodity.
5. **Per-stock pipeline** — `check_filter()` (recent crossover) → direction → (optional) 5 parallel provider enrichments → `compute_scores()`.
6. **Fast mode (>500 tickers)** — technical scoring first with volume-profile POC skipped, then fundamentals/sentiment enrichment on the top 200 by score.
7. **Output** — GUI table / HTML report / CSV.

### Caching

- `scanner/.cache/` — per-ticker disk cache (pickle, 4 h TTL) written by the provider chain; symbol lists are cached here too.
- Module-level TTL caches for free APIs (`cache.py`).
- Full-market symbol lists survive restarts through the disk cache and fall back to the static universes.

### Cache hygiene (price data)

- **One trade-date calendar.** Every daily frame is normalized onto tz-naive IST midnights at the cache boundary (`data_providers._normalize_cache_frame` on both write and read), so the UTC-close 18:30 stamps some fallback providers return can never make cross-ticker date unions double-count (the FNO 2×-calendar bug) or drift the relative-strength date masks.
- **Auto-prune on scan start.** The cache key embeds the fetch date, so entries from previous days are unreachable and would accumulate forever. `fetch_batch_yfinance` sweeps them (`data_providers.prune_stale_cache`, rate-limited to once per hour per process) before the first chunk of any scan/backtest/walk-forward; a manual **Prune** button lives on the sidebar's **Price data** card, which also shows the live fresh/stale entry counts (`cache_health`).
- **Short frames are honest data.** Names whose history is shorter than the requested window (recent listings, or suspended/delisted names like GSPL — halted May 2026 — and TATAMETALI — merged into Tata Steel 2024) come back with whatever exists, contiguous and ending at the last trade day; they are **not** fetch truncation. Anything under 260 bars is dropped by the engine's warm-up gate.
- **Dead members are skipped, stale ones are warned.** GSPL and TATAMETALI are annotated in `universes.SUSPENDED_OR_DELISTED` (kept in their lists so published membership is intact) and every scan skips them with a log line — no more pointless re-fetching. If a *different* member's data ends more than `stale_member_max_age_days` ago (Settings → Output, cache & theme, default 45), the scan appends an amber warning under the results hero naming the member and its last bar date.
- **Keep the annotation current with the audit script.** Run `python -m scanner.audit_stale_members` (or the **Check stale members** button on the Settings page) roughly weekly to catch new suspensions: it reports stale-but-unannotated names (with a paste-ready `SUSPENDED_OR_DELISTED` snippet), annotated names whose trading resumed, and names with no data in the window — flagged separately when they are only dead-symbol-cache skips. It also caught real symbol bugs in the universe lists (AVALONLABS→AVALON, ASTER→ASTERDM, BIRLASOFT→BSOFT), so treat its "no data" section as a symbol-integrity check too.

- **Audit every static universe & auto-suggest renames.** `python -m scanner.audit_stale_members --all` checks every static universe in one union fetch (no repeat downloads) and prints a per-universe missing/members breakdown so a bad symbol is attributed to the exact list that carries it. Two safety passes run on any "no data" name before it is reported: a **live probe** re-attempts it through the per-ticker provider chain (bypassing the dead-symbol-cache skip that gates the batch fallback, so wrongly-marked symbols get a second chance), and a **rename search** matches it against the live NSE mainboard list (prefix rules for ASTER→ASTERDM / AVALONLABS→AVALON, fuzzy match for BIRLASOFT→BSOFT) and suggests only candidates verified to have data — section 5 of the report. Use `--no-probe --no-renames` for a pure report-only run, and `--json` for machine-readable output (the GUI's audit verdict also mentions how many renames were suggested).

- **Apply fixes with `--fix`.** Re-running an audit with `--fix` applies section 1 + 5 + 3 findings directly to `scanner/universes.py`: verified renames are rewritten everywhere (universe lists and `SECTOR_MAP`, via quoted-symbol replacement so `ASTER→ASTERDM` can never corrupt `ASTERMINDS`), stale-unannotated names are inserted into `SUSPENDED_OR_DELISTED` in the same style as the hand-written entries, and annotated-but-fresh names are removed. The edited text is `ast.parse`-validated before an atomic write, and a `universes.py.bak` of the pre-fix file is kept. Report-only stays the default — nothing is written without the flag. Add `--dry-run` to preview the exact lines (unified diff) without writing.

- **Apply from the GUI too.** The Settings page ships an **Apply fixes** button next to **Check stale members**. After an audit the verdict shows `— fixes ready` when renames or annotation updates exist and the button enables; clicking it opens a confirmation dialog listing exactly what will change (renames / annotation adds / removals). Confirming runs the same `apply_fixes` path in a background thread, logs the full summary (including the `.bak` location), reloads `universes.py`, and automatically re-runs the audit to confirm the file is clean. Restarting the app re-reads `universes.py`, so the fix is durable.

---

## Scoring engine (Pine parity)

The 10 categories and their maximum weights match the Pine Script:

| # | Category | Max | # | Category | Max |
|---|---|---|---|---|---|
| 1 | Trend | 15 | 6 | OBV | 5 |
| 2 | Momentum | 15 | 7 | Volume | 10 |
| 3 | RSI | 8 | 8 | Relative Strength | 10 |
| 4 | MACD | 7 | 9 | Volatility | 5 |
| 5 | Stochastic | 5 | 10 | Fundamentals | 20 |

**Total = 100.** See `scanner/scoring.py` for the authoritative category-by-category mapping. The Python scorer shares indicator math with the Pine script but has also grown deliberate extensions (e.g. volume-profile POC participation, crossover-recentness points, weekly higher-timeframe checks); where it differs from the Pine v2 file, `scoring.py` documents the difference rather than claiming exact duplication.

---

## API keys (optional)

All enrichment is **optional** — the scanner works keyless using free NSE/Yahoo data. Adding keys upgrades quality: sentiment sources, institutional data, fundamentals, macro, insider and Shariah checks.

**1. Create `scanner/api_config.json`:**

```json
{
  "FINNHUB_API_KEY": "cxxxxxxx",
  "MARKETAUX_API_KEY": "xxxxx"
}
```

**2. Or set environment variables** — env vars take precedence over the file:

```bash
export FINNHUB_API_KEY=cxxxxxxx
```

**3. Or run the interactive setup wizard:**

```bash
python -m scanner.setup_api_keys
```

Every known key, its purpose, and its free tier is registered in `API_KEY_REGISTRY` in `scanner/settings_store.py`. Current registry:

| Category | Keys |
|---|---|
| Finance | `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `TWELVE_DATA_API_KEY`, `EOD_API_KEY`, `FMP_API_KEY`, `IEX_API_KEY`, `POLYGON_API_KEY`, `STOCKDATA_API_KEY`, `STYVIO_API_KEY` |
| News | `MARKETAUX_API_KEY`, `NEWS_API_KEY`, `GNEWS_API_KEY` |
| NLP / sentiment | `MEANINGCLOUD_API_KEY`, `NLPCLOUD_API_KEY`, `HF_API_KEY`, `GROQ_API_KEY` |
| Insider | `ALETHEIA_API_KEY`, `CONGRESS_API_KEY` |
| Macro | `FRED_API_KEY`, `ECONPULSE_API_KEY`, `ECONDB_API_KEY` |
| ESG / Shariah / ML | `CARBON_INTERFACE_API_KEY`, `CLIMATIQ_API_KEY`, `HALAL_API_KEY`, `TIMEDOOR_API_KEY` |

Without keys, provider fetches return empty results and the scanner simply scores on technicals + yfinance fundamentals.

---

## Backtesting & walk-forward validation

### One-line walk-forward (CLI)

```bash
python -m scanner.walkforward --stop 5 --target 15 --min-adx 20
# Sweep the ADX gate on the TRAIN half, apply the winner out-of-sample:
python -m scanner.walkforward --stop 2 --target 10 --adx-sweep "0,15,20,25,30"
```

`scanner/walkforward.py` splits the simulation window at its midpoint (or
`--split-date YYYY-MM-DD`), runs three passes — **FULL / TRAIN / TEST** — and
prints per-window trades, return, win rate, profit factor and drawdown. The
TEST (out-of-sample) column is the only one that counts: full-window numbers
are in-sample and overstate edge. Options: `--regime`, `--rotation
[--rot-block]`, `--no-thursday`, `--tickers`, `--skip-{full,train,test}`.

It is also wired into the GUI: the **chart icon in the left rail** opens a
"Backtest — walk-forward validation" page that runs the protocol against the
*current* scanner settings (MA types/lengths, crossover lookback, ADX gate)
plus the risk parameters typed on the page, so a configuration can be checked
out-of-sample before it is saved.

### How the engine supports it

`BacktestEngine` accepts optional `sim_start` / `sim_end` settings (ISO dates)
that restrict the simulated calendar window while all indicators stay
precomputed causally on the full series — each half is an independent,
lookahead-free simulation, and positions left open at a window end are closed
at that window's last bar rather than the data's last bar.

### What the ADX gate did (2024-09 → 2026-09, NIFTY 50, HMA40×EMA50)

A 2026 losing-trade audit found 92% of trades stopped out at −2% with a 9%
win rate, and that entry score / day-of-week / sector did not separate winners
from losers. Weak-trend entries (ADX < 20) were ~60% more likely to lose, so a
`min_adx_entry` gate was added to **both** engines (backtest entry logic and the
live scanner's `entry_signal`) — screening and backtest now agree.

| Configuration (3y window) | FULL | TRAIN | TEST (out-of-sample) |
|---|---|---|---|
| S5/T15, no gate | −3.8% | −1.5% | −5.1% |
| S5/T15, ADX≥20 | +5.9% | −0.2% | −2.3% |
| S5/T12, ADX≥20 | +2.6% | — | +0.2% |
| S2/T10, ADX≥20 | −0.6% | — | **+3.2%** |
| S5/T8, ADX≥20 | +1.7% | — | **+3.8%** (PF 1.30) |

Findings, in order of confidence:

1. **The gate's direction holds out-of-sample.** ADX≥20 improved 13 of 15 risk
   configs on the TEST half (avg ≈ +2–3 pts) and was the best threshold on
   both halves (it is the standard ADX convention, not an overfit spike).
   But out-of-sample magnitudes were far smaller than full-window numbers —
   losses became smaller losses / break-even, not the +5% the full window
   suggested.
2. **Full-window ranking misleads.** S5/T15 was the full-window star (+5.9%)
   yet ranked 12th of 15 out-of-sample. Short-target configs (T8/T10/T12)
   dominated the OOS ranking with ADX≥20.
3. **Do not stack gates.** Adding the index-regime gate or sector rotation on
   top of ADX≥20 helped in at most one half and hurt elsewhere; rotation
   became inert once the ADX gate filtered the pool.
4. **The edge is NIFTY-50-specific and MA-set-specific.** The same S5/T8+ADX≥20
   config was negative on BANK NIFTY in every window (−4% OOS) and on the FNO
   universe, and the saved **loose 20×40 filter collapses**: S5/T8 full-window
   went +1.7% (40×50) to −16.9% (20×40), with the ADX gate no longer helping
   out-of-sample. The loose filter is only a broad candidate *screener*; it is
   not a tradeable parameter set. The GUI backtest page therefore offers an
   explicit MA-set toggle (current settings vs the tested 40×50 reference) and
   documented presets rather than implying one config works everywhere.
5. **Default scanner setting:** `min_adx_entry: 20` now gates the live
   `entry_signal` in the GUI grid and CLI scan, dropping weak-trend names
   (e.g. RELIANCE at ADX 14 and SBICARD at ADX 19 no longer show "entry YES").

Caveat: every number above is one 2-year window on one universe — the correct
next step before trusting any config is re-running the walk-forward on fresh
data (or a 5y window) and only saving settings whose TEST column is positive.

---

## Testing

```bash
# Offline unit tests (default)
python -m pytest

# Include tests that hit live network APIs
python -m pytest -m integration
```

Test files live in `scanner/tests/`; external APIs are mocked for the offline suite.

---

## Troubleshooting

- **GUI fails to start** — make sure `Pillow` is installed (`pip install Pillow`); the app imports it at startup.
- **Scan returns few/no stocks** — check `scanner/scan.log` and `scanner/trace.log`. If Yahoo is rate-limiting, the batch fallback (jugaad-data/nselib) kicks in automatically; if all three providers fail the ticker is skipped and reported.
- **No data for a specific stock** — BSE-only symbols without NSE listings may be unavailable from the free providers.
- **Reset everything** — Settings: delete `scanner/settings.json`. Cache: Settings → "Clear Cache" in the GUI, or delete `scanner/.cache/`.
