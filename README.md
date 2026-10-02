# HMAxEMA Stock Scanner

HMAxEMA is a desktop app that screens Indian stocks (NSE/BSE) for swing trades. It runs the HMA × EMA crossover strategy from the TradingView Pine script in [`HMA_EMA_Swing_Strategy_v2.pine`](HMA_EMA_Swing_Strategy_v2.pine), scoring each stock out of 100 across 10 factors.

There's a dark-themed GUI built with Flet, an interactive CLI, and a headless engine you can drive from your own code. Scans export to HTML or CSV, and a full backtesting engine is included.

---

## Features

- **Scan universes** — NIFTY 50, BANK NIFTY, sector indices, FnO stocks, BSE indices, or the full live market (~5,900 NSE+BSE symbols), resolved from NSE at scan time.
- **Three-stage pipeline**:
  1. *Stock filter* — recent bullish MA crossover
  2. *Direction* — Bull / Bear
  3. *10-category scoring* — Trend, Momentum, RSI, MACD, Stochastic, OBV, Volume, Relative Strength, Volatility, Fundamentals (0–100)
- **Data layer built for flaky providers** — providers fail over into each other, downloads are chunked and cached on disk, and when Yahoo rate-limits a full-market scan, missed tickers are retried through NSE providers one by one.
- **Optional enrichment** — news/social sentiment, FII/DII delivery data, 52-week position, macro/forex/crypto regime, insider signals, Shariah compliance. Skip any of it; most of it just needs a free API key.
- **HTML report** — sortable, filterable table with score bars and per-stock news sentiment. CSV export too.
- **Backtesting engine** — replays the entry/exit rules with position sizing, stop/target/trailing stops, ATR stops and sector rotation.
- **Tracing & logging** — rotating `AppLog/trace.log` and `AppLog/scan.log`, plus a hook that catches uncaught exceptions.

---

## Installation

Needs **Python 3.10+**.

```bash
pip install -r scanner/requirements.txt
```

The GUI also needs a display server.

---

## Quick start

| What | How |
|---|---|
| Launch the GUI | `python -m scanner` (Windows: double-click `scanner/run.bat`, macOS/Linux: `scanner/run.sh`) |
| Interactive CLI scan | `python -m scanner --cli` or `python -m scanner.backend.run_scanner` |
| Headless scan from code | `ScannerEngine().scan(...)` — see `scanner/backend/scanner_engine.py` |
| Backtest with options | `python -m scanner.backend.backtest --years 3` |

In the GUI, pick a universe, timeframe (Daily/Weekly/Monthly), data period, an optional trend filter, and a min score threshold, then hit RUN SCAN. Results stream into the table as they arrive — click a header to sort, use the search box to filter, and export HTML or CSV from the top bar. The Import watchlist button scans your own ticker list straight from a CSV/TXT file, and every news panel has a copy button for its ticker.

Whatever you choose is saved to `scanner/settings.json`.

If you live on the keyboard: `Ctrl+K` opens the command palette, `Ctrl+R` runs or stops a scan, `Ctrl+1/2` switches views, `Ctrl+E` exports HTML, `Ctrl+S` saves settings, and `/` jumps to the search box. Snackbars confirm when scans, exports, audits and backtests finish, the rail collapses the sidebar, and double-clicking a ticker opens its news and sentiment.

---

## Architecture

### Module map

| Module | Responsibility |
|---|---|
| `scanner/ui/app.py` | Flet GUI — `ScannerApp` wires the view mixins below; owns app lifecycle, scanning/events, cache UI, HTML/CSV export, activity log, palette, shortcuts and snackbars |
| `scanner/ui/views_layout.py` | `LayoutViewMixin` — dashboard panes (rail, sidebar, main area, right panel), summary + top-pick cards |
| `scanner/ui/views_results.py` | `ResultsViewMixin` — paginated results grid, data-row/header builders, sorting keys, score chart, row-level news expansion, summary/hero updates |
| `scanner/ui/views_settings.py` | `SettingsViewMixin` — declarative settings spec + settings-page builder/inputs |
| `scanner/ui/ui_kit.py` | Shared primitives (`_border_all`, `_glass_bg`, ...), palette matcher (`fuzzy_score`, `filter_actions`) |
| `scanner/backend/scanner_engine.py` | Headless scan orchestration (`scan`, `scan_stream`) shared by GUI/CLI; "fast mode" for >500 tickers (technicals first, enrich top 200) |
| `scanner/backend/scoring.py` | Stock filter (MA crossover), Bull/Bear direction, 10-category scoring, weekly-HTF check, sideways filter, entry signals |
| `scanner/shared/indicators.py` | Vectorized pandas/numpy indicators used by scoring & backtest |
| `scanner/api/data_fetcher.py` | Batch downloads (200/chunk, 8 parallel), weekly/monthly resampling, batch **fallback pass** via NSE providers |
| `scanner/api/data_providers.py` | `DataProvider` class with provider fallback chains + disk cache |
| `scanner/api/symbol_fetcher.py` | Live NSE/BSE symbol lists (4 h cache, static fallbacks) |
| `scanner/shared/universes.py` | ~20 static universes, sector map/colors, live-universe resolution |
| `scanner/backend/backtest.py` | Backtest engine (`python -m scanner.backend.backtest`) |
| `scanner/backend/report.py` | Self-contained HTML report generator + keyword news sentiment |
| `scanner/backend/settings_store.py` | Canonical `DEFAULT_SETTINGS`, settings persistence, API-key registry/loading |
| `scanner/shared/trace.py` | Rotating trace log, `@trace` decorator, custom TRACE level |
| `scanner/shared/cache.py` | Shared in-memory TTL cache |
| `scanner/shared/themes.py` | Dark theme definition (Emerald palette, single variant) |

Enrichment modules (all optional, all guarded, in `scanner/api/`): `market_sentiment.py`, `social_sentiment.py`, `indian_market.py`, `indian_fundamentals.py`, `insider_data.py`, `macro_data.py`, `free_apis.py`, `premium_finance.py`. Tests mirror the package layout under `scanner/tests/{api,backend,shared,ui}/`.

### Scan pipeline (end to end)

1. **Resolve universe** — static list, or live NSE/BSE symbol fetch (falls back to static if offline).
2. **Fetch index** — NIFTY via `jugaad-data → yfinance` for relative-strength comparison.
3. **Batch download** — OHLCV for every ticker through `yfinance`, 200 symbols at a time, 8 parallel with throttling. Anything yfinance misses gets retried through **jugaad-data → nselib** (again 8 parallel, 10 s timeout per provider) — yfinance is skipped on the retry since it just failed at scale. When a scan misses a lot of tickers, the recovery pass first checks nselib's NSE mainboard list, so BSE-only symbols don't waste two failing calls each.
4. **Global enrichment (once)** — macro regime, forex, crypto fear/greed, commodity.
5. **Per-stock pipeline** — `check_filter()` (recent crossover) → direction → (optional) 5 parallel provider enrichments → `compute_scores()`.
6. **Fast mode (>500 tickers)** — score the technicals first (volume-profile POC skipped), then enrich the top 200 by score with fundamentals and sentiment.
7. **Output** — GUI table / HTML report / CSV.

### Caching

- `scanner/.cache/` — per-ticker disk cache (parquet + JSON meta, 4 h TTL) written by the provider chain; symbol lists are cached here too.
- Module-level TTL caches for the free APIs (`cache.py`).
- Full-market symbol lists survive restarts through the disk cache and fall back to the static universes.

### Cache hygiene (price data)

- Daily frames are normalized onto tz-naive IST midnights at the cache boundary, so cross-ticker date unions stay consistent.
- Stale cache entries are pruned when a scan starts (rate-limited); the sidebar **Price data** card shows fresh/stale counts and has a manual **Prune**.
- Suspended/delisted names live in `universes.SUSPENDED_OR_DELISTED` and are skipped at scan time. Members whose data ends older than `stale_member_max_age_days` show an amber warning under the results hero.
- Keep annotations current with `python -m scanner.backend.audit_stale_members` (or **Check stale members** on Settings). `--all` audits every universe; `--fix` applies verified renames/annotation updates (writes `universes.py.bak` first). The Settings page **Apply fixes** button runs the same path.

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

**Total = 100.** See `scanner/backend/scoring.py` for the authoritative category-by-category mapping. The Python scorer shares indicator math with the Pine script, but it has also grown some deliberate extensions (volume-profile POC participation, crossover-recentness points, weekly higher-timeframe checks). Where it differs from the Pine v2 file, `scoring.py` documents the difference rather than pretending they're identical.

---

## API keys (optional)

Everything in this section is optional — the scanner runs fine without keys, using free NSE/Yahoo data. Keys just unlock better enrichment: sentiment sources, institutional data, fundamentals, macro, insider and Shariah checks.

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

Every known key, its purpose, and its free tier is registered in `API_KEY_REGISTRY` in `scanner/backend/settings_store.py`. Current registry:

| Category | Keys |
|---|---|
| Finance | `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEY` |
| News | `MARKETAUX_API_KEY`, `NEWS_API_KEY`, `GNEWS_API_KEY` |
| Social | `TWITTER_API_KEY` |
| Insider | `ALETHEIA_API_KEY`, `CONGRESS_API_KEY` |
| Macro | `FRED_API_KEY`, `ECONPULSE_API_KEY`, `ECONDB_API_KEY` |
| Shariah | `HALAL_API_KEY` |

Without keys, provider fetches return empty results and the scanner simply scores on technicals + yfinance fundamentals.

---

## Backtesting

The backtest engine (`scanner/backend/backtest.py`) replays the full strategy on historical daily data:

```bash
python -m scanner.backend.backtest --years 3   # custom lookback
```

Entry: fast MA crosses above slow MA, close above the crossover level, close above the volume-profile POC, and score above `min_score` (plus the ADX gate). Exit: stop loss → target → trailing stop after target, with optional position sizing, ATR stops and sector rotation. `--html` / `--csv` write the report and trade log.

`BacktestEngine` also takes optional `sim_start` / `sim_end` settings (ISO dates) that restrict the simulated calendar window while all indicators stay precomputed causally on the full series. Each window is an independent, lookahead-free simulation, and positions left open at a window end close at that window's last bar rather than the data's last bar.

**ADX gate:** entries require `min_adx_entry` (default 20) in both the live scanner and the backtest — weak-trend entries lost more often in a 2-year audit, and the gate held out-of-sample on NIFTY 50. It's a broad filter, not a tradeable parameter set, so re-run the backtest before trusting a config.

---

## Testing

```bash
# Offline unit tests (default)
python -m pytest

# Include tests that hit live network APIs
python -m pytest -m integration

# With coverage (config in .coveragerc)
pip install pytest-cov
python -m pytest --cov=scanner --cov-report=term-missing
```

Test files live in `scanner/tests/`; external APIs are mocked for the offline suite.

---

## Troubleshooting

- **GUI fails to start** — check `AppLog/trace.log` for the import error.
- **Scan returns few/no stocks** — check `AppLog/scan.log` and `AppLog/trace.log`. If Yahoo is rate-limiting, the batch fallback (jugaad-data/nselib) kicks in automatically; if all three providers fail, the ticker is skipped and reported.
- **No data for a specific stock** — BSE-only symbols without NSE listings may be unavailable from the free providers.
- **Reset everything** — Settings: delete `scanner/settings.json`. Cache: Settings → "Clear Cache" in the GUI, or delete `scanner/.cache/`.
