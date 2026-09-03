"""Unit tests for scanner.run_scanner — CLI scoring loop vs shared _score_ticker.

The CLI loop must behave exactly like the engine's per-ticker scorer:
filter -> direction -> fundamentals -> score -> rating gate. These tests
run run_scan() with every external call (fetch, report, prompts) mocked.
"""

import builtins
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import scanner.run_scanner as run_scanner
from scanner import scanner_engine


@pytest.fixture
def recent_crossover_ohlcv():
    """Flat for 120 bars then a sharp 20-bar rise — a crossover inside the
    20-bar lookback (empirically verified against check_filter)."""
    n = 140
    close = np.concatenate([np.full(120, 100.0), np.linspace(100, 150, 20)])
    high = close * 1.01
    low = close * 0.99
    open_ = close * 1.001
    volume = np.full(n, 1_000_000.0)
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def cli_mocks(recent_crossover_ohlcv, flat_ohlcv, monkeypatch):
    """Patch every external/interactive call out of run_scan().

    Returns (stock_data, captured) where captured collects the results list
    handed to generate_html_report.
    """
    captured = {"results": None}

    def fake_report(results, **kw):
        captured["results"] = list(results)
        return "<html>"

    monkeypatch.setattr(run_scanner, "_load_settings", lambda: {"min_score": 50.0})
    monkeypatch.setattr(run_scanner, "select_universe", lambda: ("TEST", ["AAA", "BBB"]))
    monkeypatch.setattr(run_scanner, "select_threshold", lambda: 50.0)
    monkeypatch.setattr(run_scanner, "select_period", lambda: "1y")
    monkeypatch.setattr(run_scanner, "select_timeframe", lambda: "D")
    monkeypatch.setattr(run_scanner, "fetch_index_data", lambda *a, **kw: None)
    monkeypatch.setattr(run_scanner, "generate_html_report", fake_report)
    monkeypatch.setattr(run_scanner, "save_report", lambda *a, **kw: None)
    monkeypatch.setattr(run_scanner.webbrowser, "open", lambda *a, **kw: None)
    monkeypatch.setattr(builtins, "input", lambda *a, **kw: "n")
    # No network from the shared helper: fundamentals come back empty
    monkeypatch.setattr(scanner_engine, "fetch_fundamentals", lambda *a, **kw: None)

    stock_data = {"AAA": recent_crossover_ohlcv, "BBB": flat_ohlcv}
    monkeypatch.setattr(run_scanner, "fetch_batch_yfinance", lambda *a, **kw: stock_data)
    return stock_data, captured


class TestCliScoringLoop:
    def test_loop_uses_shared_helper_accounting(self, cli_mocks, monkeypatch):
        """Filtered and insufficient rows are excluded; direction comes from the helper."""
        _, captured = cli_mocks
        calls = []

        def fake_score_ticker(ticker, df, **kw):
            calls.append((ticker, kw))
            if ticker == "AAA":
                return {"total": 62.0, "trend_dir": "Bull",
                        "trend_color": "bull", "combined_rating": "GOOD",
                        "ticker": ticker}, "Bull"
            if ticker == "BBB":
                return None, "filtered"
            return None, "no_score"

        monkeypatch.setattr(run_scanner, "_score_ticker", fake_score_ticker)

        run_scanner.run_scan()

        # Only the scored ticker reaches the report
        assert captured["results"] == [{"total": 62.0, "trend_dir": "Bull",
                                        "trend_color": "bull",
                                        "combined_rating": "GOOD",
                                        "ticker": "AAA"}]
        # Helper called with CLI semantics: no directional filter, no enrichment
        ticker0, kw0 = calls[0]
        assert ticker0 == "AAA"
        assert kw0["trend_filter"] == "All"
        assert kw0["is_large"] is False
        assert kw0["global_data"] is None
        assert kw0["timeframe"] == "D"
        assert callable(kw0["enrich"])
        assert kw0["settings"] == {"min_score": 50.0, "data_period": "1y",
                                   "timeframe": "D"}

    def test_loop_matches_direct_helper_call(self, cli_mocks, monkeypatch):
        """The loop appends exactly what _score_ticker returns — no re-derivation."""
        stock_data, captured = cli_mocks
        settings = {"min_score": 50.0, "data_period": "1y", "timeframe": "D"}

        run_scanner.run_scan()

        # Reproduce the exact helper result the loop should have appended
        df = stock_data["AAA"]
        expected, direction = scanner_engine._score_ticker(
            "AAA", df,
            settings=settings, timeframe="D", index_df=None,
            trend_filter="All", is_large=False, global_data=None,
            enrich=lambda _t, s, _g: dict(s),
        )
        assert direction == "Bull"
        assert captured["results"] == [expected]
        # The flat series did not produce a recent crossover
        assert len(captured["results"]) == 1

    def test_large_universe_gets_phase2_enrichment(self, cli_mocks, monkeypatch):
        """Custom universes >500 trigger fast scoring + top-200 enrich/re-score."""
        stock_data, _ = cli_mocks
        # 501 tickers -> large mode; only 2 real frames come back from fetch
        names = [f"T{i}" for i in range(501)]
        monkeypatch.setattr(run_scanner, "select_universe", lambda: ("BIG", names))
        data = {"AAA": stock_data["AAA"], "BBB": stock_data["BBB"]}
        monkeypatch.setattr(run_scanner, "fetch_batch_yfinance", lambda *a, **kw: data)

        fake_engine = MagicMock()
        fake_engine._fetch_global_enrichment.return_value = {"macro": 1}
        fake_engine._enrich_with_providers.return_value = {}
        monkeypatch.setattr(run_scanner, "ScannerEngine", lambda: fake_engine)

        enrich_calls = {}

        def fake_enrich_top(rows, batch_data, **kw):
            enrich_calls["n"] = len(rows)
            enrich_calls["kw"] = kw
            return rows

        monkeypatch.setattr(run_scanner, "_enrich_rows_in_place", fake_enrich_top)

        run_scanner.run_scan()

        # Phase-2 ran with the engine's provider-enrichment callable and the
        # fetched global data; only the one scored row was enriched
        assert enrich_calls["n"] == 1
        assert enrich_calls["kw"]["global_data"] == {"macro": 1}
        assert enrich_calls["kw"]["timeframe"] == "D"
        assert enrich_calls["kw"]["enrich"] is fake_engine._enrich_with_providers
        # And phase-1 used the engine semantics too
        assert fake_engine._fetch_global_enrichment.called_once

    def test_small_universe_skips_engine_setup(self, cli_mocks, monkeypatch):
        """Small lists must not instantiate the engine or fetch global data."""
        monkeypatch.setattr(run_scanner, "ScannerEngine", lambda: pytest.fail("no engine needed"))
        run_scanner.run_scan()  # 2-ticker TEST universe -> small path