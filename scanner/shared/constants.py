"""
Shared constants for the HMAxEMA Scanner.

Single source of truth for magic numbers, thresholds, and column definitions
that are referenced across multiple modules. Each constant includes a brief
comment explaining its origin or purpose.
"""

# ── Results grid columns ─────────────────────────────────────────────────────
# The trailing "1M" column draws a mini sparkline of the last ~20 closes
# (see px_tail on each result row); "1M%" stays the numeric one-month change.
RESULT_COLS = [
    "#",
    "Ticker",
    "Score",
    "Rating",
    "ENTRY",
    "Price",
    "MA",
    "T/15",
    "M/15",
    "R/8",
    "V/7",
    "Vol/10",
    "RS/10",
    "F/20",
    "1M%",
    "Dir",
    "ADX",
    "Chop",
    "1M",
]

# ── Universe defaults ────────────────────────────────────────────────────────
UNIVERSE_SIZES = {
    "NSE ALL": 2567,
    "BSE ALL": 4500,
    "FULL MARKET": 5900,
}
UNIVERSE_DEFAULT_SIZE = 50

# ── Scanner engine thresholds ───────────────────────────────────────────────
# Universe size cutoff for fast-mode vs. small-mode parallel scoring
LARGE_UNIVERSE_THRESHOLD = 500
# Number of top-scored results to run Phase-2 enrichment on
ENRICH_TOP_N = 200
# Per-provider future.result() timeout (seconds) — prevents scan hang on stalled HTTP
ENRICH_PROVIDER_TIMEOUT = 15
# Hard ceiling (seconds) for the entire Phase-2 enrichment pass
ENRICH_OVERALL_TIMEOUT = 300
# Per-ticker hard ceiling for enrichment (seconds)
TICKER_TIMEOUT = 60
# Number of recent closes carried on each result row for the sparkline
SPARK_BARS = 20
# Default minimum score threshold
DEFAULT_MIN_SCORE = 50
# Two-phase enrichment kicks in above this many tickers
TWO_PHASE_THRESHOLD = 20
# Directional trend filters hide POOR/WEAK stocks
DIRECTIONAL_TREND_FILTERS = ("Bullish Only", "Bearish Only")
POOR_RATINGS = ("POOR", "WEAK")
# Stale member detection: default max age in days
STALE_MEMBER_MAX_AGE_DAYS = 45

# ── Data fetcher thresholds ─────────────────────────────────────────────────
# Chunk size for parallel yfinance downloads (~200 tickers per URL)
CHUNK_SIZE = 200
# Maximum parallel chunk downloads
MAX_PARALLEL_CHUNKS = 8
# Throttle between parallel batches (seconds) to avoid 429
SLEEP_BETWEEN_BATCH = 0.3
# Per-ticker fallback threads — keep low to avoid NSE rate-limiting
FALLBACK_WORKERS = 4
# Per-provider timeout in the fallback pass (seconds)
FALLBACK_PROVIDER_TIMEOUT = 10.0
# Hard ceiling for the entire fallback batch (seconds)
FALLBACK_OVERALL_TIMEOUT = 180
# Only consult the NSE list above this many misses
FALLBACK_FILTER_MIN_MISSING = 25
# Minimum bars required for a valid OHLCV DataFrame
MIN_BARS_VALIDATION = 50
# Minimum bars for a cache entry to be considered
MIN_BARS_CACHE_ENTRY = 20

# ── Scoring thresholds (from Pine Script v2 reference) ──────────────────────
# RSI peak score threshold
RSI_PEAK = 55.0
# RSI band boundaries for scoring
RSI_LOW = 40.0
RSI_HIGH = 70.0
# ADX threshold for trend confirmation
ADX_TREND_THRESHOLD = 25.0
# ATR percentage thresholds for volatility scoring
ATR_HIGH_PCT = 3.0
ATR_LOW_PCT = 1.0
# Momentum: strong move threshold (percentage)
STRONG_MOVE_PCT = 5.0
# Stochastic healthy band
STOCH_LOW = 20.0
STOCH_HIGH = 80.0
# Volume at crossover gate: multiplier for volume comparison
VOL_CROSSOVER_MULT = 0.8

# ── UI thresholds ────────────────────────────────────────────────────────────
# Top-scored rows whose news is prefetched after a scan
NEWS_PREFETCH_TOP = 50
# Rotating scan.log: keep at most this many lines in the UI log viewer
LOG_MAX_LINES = 500
# Rotate scan.log when it is older than this many hours
LOG_ROTATE_HOURS = 24


# ── Shared helpers (pure data, no Flet dependency) ───────────────────────────
def score_of(r: dict) -> float:
    """Total score of a result row, tolerant of missing/None values."""
    return r.get("total", 0) or 0
