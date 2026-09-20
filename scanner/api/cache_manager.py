"""
Unified cache manager for the HMAxEMA Scanner.

Provides a clean public API for the two caching subsystems that live
in ``data_fetcher`` (negative cache + enrichment cache).  Business logic
(``scanner_engine``) and the GUI (``app``) import from here instead of
reaching into ``data_fetcher`` internals.
"""

from .data_fetcher import (
    ENRICHMENT_CACHE_TTL_HOURS,
    NEGATIVE_CACHE_TTL_HOURS,
    _enrichment_cache_get,
    _enrichment_cache_put,
    _negative_cache_contains,
    _negative_cache_load,
    _negative_cache_update,
    _record_enrichment_cache_hit,
    _record_enrichment_cache_miss,
    _record_negative_cache_skips,
    enrichment_cache_clear,
    enrichment_cache_hits,
    enrichment_cache_misses,
    enrichment_cache_size,
    enrichment_cache_ttl_hours,
    negative_cache_skip_count,
    negative_cache_ttl_hours,
    reset_enrichment_cache_counts,
    reset_negative_cache_skip_count,
    set_enrichment_cache_ttl_hours,
    set_negative_cache_ttl_hours,
)
from .data_providers import cache_health as _cache_health
from .data_providers import prune_stale_cache as _prune_stale_cache

__all__ = [
    # ── Enrichment cache ────────────────────────────────────────
    "enrichment_get",
    "enrichment_put",
    "enrichment_clear",
    "enrichment_size",
    "enrichment_ttl_hours",
    "set_enrichment_ttl",
    "enrichment_stats",
    "reset_enrichment_counts",
    "record_enrichment_hit",
    "record_enrichment_miss",
    "ENRICHMENT_CACHE_TTL_HOURS",
    # ── Negative cache ──────────────────────────────────────────
    "negative_contains",
    "negative_load",
    "negative_update",
    "negative_skip_count",
    "reset_negative_skips",
    "record_negative_skips",
    "set_negative_ttl",
    "negative_ttl_hours",
    "NEGATIVE_CACHE_TTL_HOURS",
    # ── Price cache ─────────────────────────────────────────────
    "cache_health",
    "prune_stale_cache",
]

# ── Enrichment cache aliases ─────────────────────────────────────────────────
enrichment_get = _enrichment_cache_get
enrichment_put = _enrichment_cache_put
enrichment_clear = enrichment_cache_clear
enrichment_size = enrichment_cache_size
enrichment_ttl_hours = enrichment_cache_ttl_hours
set_enrichment_ttl = set_enrichment_cache_ttl_hours
reset_enrichment_counts = reset_enrichment_cache_counts
record_enrichment_hit = _record_enrichment_cache_hit
record_enrichment_miss = _record_enrichment_cache_miss


def enrichment_stats() -> dict:
    """Return current scan hit/miss counts."""
    return {
        "hits": enrichment_cache_hits(),
        "misses": enrichment_cache_misses(),
    }


# ── Negative cache aliases ───────────────────────────────────────────────────
negative_contains = _negative_cache_contains
negative_load = _negative_cache_load
negative_update = _negative_cache_update
negative_skip_count = negative_cache_skip_count
reset_negative_skips = reset_negative_cache_skip_count
record_negative_skips = _record_negative_cache_skips
set_negative_ttl = set_negative_cache_ttl_hours
negative_ttl_hours = negative_cache_ttl_hours


# ── Price cache aliases ─────────────────────────────────────────────────────
cache_health = _cache_health
prune_stale_cache = _prune_stale_cache
