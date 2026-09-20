"""Batch-scan speed: parallel small-universe scoring + enrichment cache.

Covers the two optimizations that target the measured bottleneck (~3.3s of
provider network per ticker vs ~20ms of scoring):

- ``_parallel_score``: order preservation, cancel responsiveness, crash
  isolation, and a wall-time proof it beats the sequential floor.
- ``_enrich_small_cached`` (via real ``_score_ticker``): cache hits skip
  provider + fundamentals calls with identical scores; provider-flag
  filtering; default backfill; ``use_enrichment_cache=False`` bypass.
- ``scan_stream`` small path end-to-end: exact counts, input-order batch
  payloads, one enrich call per scored ticker.
"""

import threading
import time
from typing import ClassVar

import numpy as np
import pandas as pd

import scanner.backend.scanner_engine as eng_mod
from scanner.api import data_fetcher
from scanner.backend.scanner_engine import (
    _parallel_score,
    _score_ticker,
)


def _crossover_df(seed=11):
    """140 bars, flat then sharp rise — passes check_filter (like the CLI fixture)."""
    rng = np.random.RandomState(seed)
    n = 140
    close = np.concatenate([np.full(120, 100.0), np.linspace(100, 150, 20)])
    high = close * 1.01
    low = close * 0.99
    open_ = close * 1.001
    volume = (rng.rand(n) * 1_000_000 + 500_000).astype(int)
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _flat_df():
    n = 140
    close = np.full(n, 100.0)
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(n, 1_000_000.0)},
        index=dates,
    )


def _isolated_cache(monkeypatch, tmp_path):
    """Point the enrichment disk cache at tmp and reset in-memory state."""
    monkeypatch.setattr(
        data_fetcher, "_ENRICHMENT_CACHE_PATH", str(tmp_path / "enrich.json"))
    monkeypatch.setattr(data_fetcher, "_enrichment_cache", None)


# ── _parallel_score ───────────────────────────────────────────────────

class TestParallelScore:
    def test_preserves_input_order(self):
        items = [(f"T{i}", i) for i in range(8)]

        def slow_fn(item):
            time.sleep(0.05 * (8 - item[1]))  # reverse completion order
            return ({"ticker": item[0]}, "Bull")

        ordered, cancelled = _parallel_score(items, slow_fn, threading.Event())
        assert cancelled is False
        assert [ordered[i][0]["ticker"] for i in range(8)] == [f"T{i}" for i in range(8)]

    def test_beats_sequential_floor(self):
        items = [(f"T{i}", None) for i in range(5)]

        def slow_fn(item):
            time.sleep(0.4)
            return ({"ticker": item[0]}, "Bull")

        t0 = time.perf_counter()
        ordered, _ = _parallel_score(items, slow_fn, threading.Event())
        elapsed = time.perf_counter() - t0
        assert len(ordered) == 5
        # Sequential floor would be 5 * 0.4 = 2.0s; parallel must beat it
        # with wide margin (expected ~0.4s + thread overhead).
        assert elapsed < 1.2, f"too slow: {elapsed:.2f}s"

    def test_cancel_returns_promptly_with_partial_results(self):
        items = [(f"T{i}", None) for i in range(10)]
        cancel = threading.Event()

        def slow_fn(item):
            time.sleep(0.5)
            return ({"ticker": item[0]}, "Bull")

        def fire():
            time.sleep(0.2)
            cancel.set()

        threading.Thread(target=fire, daemon=True).start()
        t0 = time.perf_counter()
        ordered, cancelled = _parallel_score(items, slow_fn, cancel)
        elapsed = time.perf_counter() - t0
        assert cancelled is True
        assert elapsed < 2.0  # far below the 5s full-batch cost
        assert all(v[1] == "Bull" for v in ordered.values())

    def test_worker_crash_isolated(self):
        items = [("OK", 1), ("BOOM", 2)]

        def flaky(item):
            if item[0] == "BOOM":
                raise RuntimeError("worker blew up")
            return ({"ticker": "OK"}, "Bull")

        ordered, cancelled = _parallel_score(items, flaky, threading.Event())
        assert cancelled is False
        assert ordered[0] == ({"ticker": "OK"}, "Bull")
        assert ordered[1] == (None, "error")

    def test_empty_input(self):
        assert _parallel_score([], lambda i: i, threading.Event()) == ({}, False)

    def test_none_cancel_event_runs_to_completion(self):
        ordered, cancelled = _parallel_score(
            [("A", 1)], lambda item: ({"ticker": "A"}, "Bear"), None)
        assert cancelled is False
        assert ordered == {0: ({"ticker": "A"}, "Bear")}


# ── _enrich_small_cached via real _score_ticker ───────────────────────

class TestSmallCacheReadthrough:
    SETTINGS: ClassVar[dict] = {"min_score": 50.0, "use_market_sentiment": True,
                "use_social_sentiment": True, "use_indian_market": True,
                "use_indian_fundamentals": True, "use_insider_data": True}

    def _score(self, ticker, df, enrich, fund, monkeypatch, use_cache=True):
        monkeypatch.setattr(eng_mod, "fetch_fundamentals", fund)
        return _score_ticker(
            ticker, df, settings=dict(self.SETTINGS), timeframe="D",
            index_df=df, trend_filter="All", is_large=False,
            global_data={}, enrich=enrich, use_enrichment_cache=use_cache)

    def test_miss_runs_live_and_populates_cache(self, monkeypatch, tmp_path):
        _isolated_cache(monkeypatch, tmp_path)
        calls = []

        def enrich(ticker, settings, gd):
            calls.append(ticker)
            return {"_sentiment_score": 0.7, "_article_count": 3}

        df = _crossover_df()
        out, _ = self._score(
            "PF11", df, enrich, lambda ticker: {"pe_ratio": 15.0}, monkeypatch)
        assert calls == ["PF11"]
        assert df.attrs.get("_fundamentals") == {"pe_ratio": 15.0}
        entry = data_fetcher._enrichment_cache_get("PF11")
        assert entry is not None
        assert entry["providers"] == {"_sentiment_score": 0.7, "_article_count": 3}
        assert entry["fundamentals"] == {"pe_ratio": 15.0}

    def test_hit_skips_providers_and_fundamentals_with_identical_scores(
            self, monkeypatch, tmp_path):
        _isolated_cache(monkeypatch, tmp_path)
        data_fetcher._enrichment_cache_put(
            "PF12", {"_sentiment_score": 0.9, "_insider_score": 5},
            {"pe_ratio": 20.0})

        def enrich(ticker, settings, gd):
            raise AssertionError("providers must not run on a cache hit")

        def fund(ticker):
            raise AssertionError("fundamentals must not run on a cache hit")

        df_hit = _crossover_df(seed=21)
        out_hit, _ = self._score("PF12", df_hit, enrich, fund, monkeypatch)
        # Same ticker, live path, for equality comparison
        data_fetcher.enrichment_cache_clear()
        df_live = _crossover_df(seed=21)
        live_calls = []

        def enrich_live(ticker, settings, gd):
            live_calls.append(ticker)
            return {"_sentiment_score": 0.9, "_insider_score": 5}

        out_live, _ = self._score(
            "PF12b", df_live, enrich_live, lambda ticker: {"pe_ratio": 20.0},
            monkeypatch)
        assert live_calls == ["PF12b"]
        assert df_hit.attrs.get("_fundamentals") == {"pe_ratio": 20.0}
        # Identical inputs (modulo ticker/fund source) score identically
        assert out_hit["total"] == out_live["total"]
        assert out_hit["combined_rating"] == out_live["combined_rating"]

    def test_disabled_flag_drops_stale_keys(self, monkeypatch, tmp_path):
        _isolated_cache(monkeypatch, tmp_path)
        data_fetcher._enrichment_cache_put(
            "PF13", {"_sentiment_score": 0.9, "_insider_score": 5}, None)

        def enrich(ticker, settings, gd):
            raise AssertionError("must not run on a cache hit")

        df = _crossover_df(seed=31)
        settings = dict(self.SETTINGS)
        settings["use_market_sentiment"] = False
        monkeypatch.setattr(eng_mod, "fetch_fundamentals",
                            lambda ticker: (_ for _ in ()).throw(
                                AssertionError("no fund call expected")))
        out, _ = _score_ticker(
            "PF13", df, settings=settings, timeframe="D",
            index_df=df, trend_filter="All", is_large=False,
            global_data={}, enrich=enrich, use_enrichment_cache=True)
        assert "_sentiment_score" not in out  # disabled provider stays out...
        assert out["_insider_score"] == 5  # ...while enabled keys replay

    def test_enabled_missing_keys_get_live_defaults(self, monkeypatch, tmp_path):
        _isolated_cache(monkeypatch, tmp_path)
        # Cache written while sentiment was disabled: no sentiment keys.
        data_fetcher._enrichment_cache_put("PF14", {"_insider_score": 1}, None)

        def enrich(ticker, settings, gd):
            raise AssertionError("must not run on a cache hit")

        df = _crossover_df(seed=41)
        out, _ = self._score(
            "PF14", df, enrich, lambda ticker: None, monkeypatch)
        assert out["_sentiment_score"] == 0.0  # live-equivalent default
        assert out["_insider_score"] == 1

    def test_bypass_flag_keeps_cli_deterministic(self, monkeypatch, tmp_path):
        _isolated_cache(monkeypatch, tmp_path)
        data_fetcher._enrichment_cache_put(
            "PF15", {"_sentiment_score": 0.9}, {"pe_ratio": 20.0})
        calls = []

        def enrich(ticker, settings, gd):
            calls.append(ticker)
            return {}

        df = _crossover_df(seed=51)
        self._score("PF15", df, enrich, lambda ticker: {"pe_ratio": 9.0},
                    monkeypatch, use_cache=False)
        assert calls == ["PF15"]  # cache ignored despite warm entry


# ── scan_stream small path end-to-end ─────────────────────────────────

class TestScanStreamSmallParallel:
    def test_counts_order_and_single_enrich_per_ticker(
            self, monkeypatch, tmp_path):
        _isolated_cache(monkeypatch, tmp_path)
        dfs = {"PS01": _crossover_df(seed=61), "PS02": _flat_df(),
               "PS03": _crossover_df(seed=63), "PS04": _flat_df()}
        tickers = list(dfs)
        enrich_calls = []
        batches = []

        # Patched onto the class, so it binds like a method (self first).
        def enrich(_self, ticker, settings, gd):
            enrich_calls.append(ticker)
            time.sleep(0.2)
            return {"_sentiment_score": 0.5}

        monkeypatch.setattr(eng_mod, "fetch_fundamentals",
                            lambda ticker: {"pe_ratio": 18.0})
        monkeypatch.setattr(eng_mod.ScannerEngine, "_enrich_with_providers", enrich)
        eng = eng_mod.ScannerEngine()
        monkeypatch.setattr(
            eng, "_prepare_scan",
            lambda *a, **k: (tickers, dfs["PS01"], {}, False))
        monkeypatch.setattr(
            eng_mod, "fetch_batch_yfinance_stream",
            lambda *a, **k: iter([dict(dfs)]))

        result = eng.scan_stream(
            "TEST", dict(TestSmallCacheReadthrough.SETTINGS),
            period="1y", timeframe="D", trend_filter="All",
            index_symbol="NSEI",
            on_batch=lambda chunk: batches.append([r["ticker"] for r in chunk]),
        )
        assert result.error is None
        assert not result.cancelled
        got = sorted(r["ticker"] for r in result.results)
        assert got == ["PS01", "PS03"]  # flats filtered, crossovers scored
        assert result.filtered_out == 2
        # Batch payload preserves input order; each scored ticker enriched once
        assert batches and batches[0] == ["PS01", "PS03"]
        assert sorted(enrich_calls) == ["PS01", "PS03"]
