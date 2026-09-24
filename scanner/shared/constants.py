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
# Directional trend filters hide POOR/WEAK stocks
DIRECTIONAL_TREND_FILTERS = ("Bullish Only", "Bearish Only")
POOR_RATINGS = ("POOR", "WEAK")
# Stale member detection: default max age in days
STALE_MEMBER_MAX_AGE_DAYS = 45

# ── UI thresholds ────────────────────────────────────────────────────────────
# Top-scored rows whose news is prefetched after a scan
NEWS_PREFETCH_TOP = 50
# Rotating scan.log: keep at most this many lines in the UI log viewer
LOG_MAX_LINES = 200


# ── Shared helpers (pure data, no Flet dependency) ───────────────────────────
def score_of(r: dict) -> float:
    """Total score of a result row, tolerant of missing/None values."""
    return r.get("total", 0) or 0
