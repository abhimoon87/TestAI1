# HMAxEMA Stock Scanner

A Python desktop application that screens **Indian stocks (NSE/BSE)** for swing-trading setups using an **HMA × EMA crossover** strategy with a **10-factor, 100-point scoring engine** — a Python re-implementation of the TradingView Pine Script in [`HMA_EMA_Swing_Strategy_v2.pine`](HMA_EMA_Swing_Strategy_v2.pine).

The app includes a dark-themed desktop GUI (CustomTkinter), an interactive CLI, a headless scan engine, HTML/CSV report export, and a full backtesting engine.

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

scanner/__main__.py ─┬─> scanner/app.py           GUI ("Aurora" UI), exports, pagination
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
| `scanner/app.py` | CustomTkinter GUI — dashboard/settings views, live log, paginated results grid, HTML/CSV export |
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
