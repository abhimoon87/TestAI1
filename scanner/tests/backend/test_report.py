"""Unit tests for scanner.report — HTML report generation and helpers.

Tests cover:
  - _sentiment: keyword-based sentiment scoring
  - _parse_date: ISO date string parsing
  - _score_class: CSS class selection
  - generate_html_report: HTML output structure
  - save_report: file writing and old-report cleanup
  - fetch_stock_news: news fetching with mocked yfinance
"""

import os
import re
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from scanner.backend.report import (
    _NEWS_CACHE,
    _REPORT_COLS,
    SENTIMENT_BAD,
    SENTIMENT_GOOD,
    _css_block,
    _js_block,
    _parse_date,
    _score_class,
    _sentiment,
    _table_head_html,
    fetch_news_batch,
    fetch_news_for_ticker,
    fetch_stock_news,
    generate_html_report,
    prune_old,
    save_report,
)

# ── Helpers ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _news_cache_reset():
    """The in-memory TTLCache outlives the per-test sqlite db — reset it."""
    _NEWS_CACHE.clear()
    yield
    _NEWS_CACHE.clear()


def _make_score_result(ticker="RELIANCE", total=65.0, **overrides):
    """Create a minimal scoring result dict for report testing."""
    base = {
        "ticker": ticker,
        "total": total,
        "trend": 10.0,
        "momentum": 8.0,
        "rsi": 6.0,
        "macd": 5.0,
        "stoch": 4.0,
        "obv": 3.0,
        "volume": 7.0,
        "rel_str": 6.0,
        "volatility": 5.0,
        "fundamentals": 10.0,
        "ma_bullish": True,
        "close_above_both_ma": True,
        "ma_crossed_above": False,
        "crossover_bars_ago": -1,
        "above_poc": True,
        "vp_poc": 2450.0,
        "close": 2500.0,
        "trend_dir": "Bull",
        "trend_color": "bull",
        "rsi_val": 55.0,
        "adx_val": 30.0,
        "pc1m": 3.5,
        "pc3m": 8.2,
        "volat_stat": "Medium",
        "is_sideways": False,
        "sideways_reasons": [],
        "combined_rating": "GOOD",
        "entry_signal": True,
        "weekly_entry_signal": False,
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _no_flow_cache(monkeypatch):
    """Machine-local FII/DII flow cache must not leak into report assertions."""
    monkeypatch.setattr(
        "scanner.api.indian_market.fetch_fii_dii_history", lambda *a, **k: None
    )


# ══════════════════════════════════════════════════════════════════════════════
# _sentiment
# ══════════════════════════════════════════════════════════════════════════════


class TestSentiment:
    def test_good_headline(self):
        assert _sentiment("Stock surges on strong profit growth") == "Good"

    def test_bad_headline(self):
        assert _sentiment("Company faces fraud investigation") == "Bad"

    def test_neutral_headline(self):
        assert _sentiment("Board meets on Tuesday") == "Neutral"

    def test_empty_input(self):
        assert _sentiment("") == "Neutral"

    def test_mixed_more_good(self):
        assert _sentiment("profit loss profit growth") == "Good"

    def test_mixed_more_bad(self):
        assert _sentiment("loss decline loss crash") == "Bad"

    def test_equal_good_bad(self):
        assert _sentiment("profit loss") == "Neutral"

    def test_case_insensitive(self):
        assert _sentiment("PROFIT GROWTH SURGE") == "Good"

    def test_summary_contributes(self):
        assert _sentiment("Stock rises", "Company reports strong earnings") == "Good"

    def test_keyword_sets_are_frozensets(self):
        """SENTIMENT_GOOD and SENTIMENT_BAD should be frozensets for O(1) lookup."""
        assert isinstance(SENTIMENT_GOOD, frozenset)
        assert isinstance(SENTIMENT_BAD, frozenset)
        assert len(SENTIMENT_GOOD) > 0
        assert len(SENTIMENT_BAD) > 0


# ══════════════════════════════════════════════════════════════════════════════
# _parse_date
# ══════════════════════════════════════════════════════════════════════════════


class TestParseDate:
    def test_iso_datetime(self):
        result = _parse_date("2024-08-15T10:30:00")
        assert result == datetime(2024, 8, 15, 10, 30, 0)

    def test_iso_date_only(self):
        result = _parse_date("2024-08-15")
        assert result == datetime(2024, 8, 15)

    def test_with_z_suffix(self):
        result = _parse_date("2024-08-15T10:30:00Z")
        assert result == datetime(2024, 8, 15, 10, 30, 0)

    def test_empty_string(self):
        assert _parse_date("") is None

    def test_none(self):
        assert _parse_date(None) is None

    def test_invalid_format(self):
        assert _parse_date("not-a-date") is None

    def test_partial_date(self):
        # "2024-08" doesn't match either format
        assert _parse_date("2024-08") is None


# ══════════════════════════════════════════════════════════════════════════════
# _score_class
# ══════════════════════════════════════════════════════════════════════════════


class TestScoreClass:
    def test_excellent(self):
        assert _score_class(75.0) == "excellent"
        assert _score_class(100.0) == "excellent"

    def test_good(self):
        assert _score_class(55.0) == "good"
        assert _score_class(69.9) == "good"

    def test_moderate(self):
        assert _score_class(35.0) == "moderate"
        assert _score_class(49.9) == "moderate"

    def test_poor(self):
        assert _score_class(10.0) == "poor"
        assert _score_class(0.0) == "poor"
        assert _score_class(29.9) == "poor"

    def test_boundary_70(self):
        assert _score_class(70.0) == "excellent"

    def test_boundary_50(self):
        assert _score_class(50.0) == "good"

    def test_boundary_30(self):
        assert _score_class(30.0) == "moderate"


# ══════════════════════════════════════════════════════════════════════════════
# generate_html_report
# ══════════════════════════════════════════════════════════════════════════════


class TestGenerateHtmlReport:
    def test_returns_string(self):
        results = [_make_score_result()]
        html = generate_html_report(results, fetch_news=False)
        assert isinstance(html, str)

    def test_contains_title(self):
        results = [_make_score_result()]
        html = generate_html_report(results, title="My Scanner", fetch_news=False)
        assert "My Scanner" in html

    def test_contains_ticker(self):
        results = [_make_score_result(ticker="TCS")]
        html = generate_html_report(results, fetch_news=False)
        assert "TCS" in html

    def test_contains_score(self):
        results = [_make_score_result(total=72.5)]
        html = generate_html_report(results, fetch_news=False)
        # Grid parity: score renders `:.0f` (72.5 → 72, banker's like the app).
        assert '<td class="score s-excellent">72</td>' in html

    def test_contains_threshold(self):
        results = [_make_score_result()]
        html = generate_html_report(results, threshold=50.0, fetch_news=False)
        assert "Threshold 50+" in html

    def test_sorted_by_score_descending(self):
        results = [
            _make_score_result(ticker="A", total=30.0),
            _make_score_result(ticker="B", total=80.0),
            _make_score_result(ticker="C", total=50.0),
        ]
        html = generate_html_report(results, fetch_news=False)
        # B (80) should appear before A (30) in the HTML
        pos_b = html.index("B")
        pos_a = html.index('data-ticker="A"')
        assert pos_b < pos_a

    def test_highlight_class_for_passing(self):
        results = [_make_score_result(total=60.0)]
        html = generate_html_report(results, threshold=50.0, fetch_news=False)
        assert "highlight" in html

    def test_no_news_when_disabled(self):
        results = [_make_score_result()]
        html = generate_html_report(results, fetch_news=False)
        # When fetch_news=False, no news sentiment content should appear
        # in the expandable panels (trade reasons panels are always shown).
        # Scoped to news markers — ">Good</button>" would false-positive on
        # the filter chips, bare names on the CSS rules.
        assert 'class="news-sentiment' not in html
        assert 'class="news-item"' not in html
        assert "No recent news found" not in html

    def test_empty_results(self):
        html = generate_html_report([], fetch_news=False)
        assert isinstance(html, str)
        assert "Total scanned" in html

    def test_html_is_valid_structure(self):
        results = [_make_score_result()]
        html = generate_html_report(results, fetch_news=False)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<table" in html
        assert "<script>" in html

    def test_bull_trend_icon(self):
        results = [_make_score_result(trend_dir="Bull")]
        html = generate_html_report(results, fetch_news=False)
        assert "^ Bull" in html  # Dir cell text, grid parity

    def test_bear_trend_icon(self):
        results = [_make_score_result(trend_dir="Bear", trend_color="bear")]
        html = generate_html_report(results, fetch_news=False)
        assert "v Bear" in html

    def test_ma_cross_signal(self):
        results = [_make_score_result(ma_crossed_above=True, crossover_bars_ago=2)]
        html = generate_html_report(results, fetch_news=False)
        assert "^ X2" in html  # shared ma_chip text, not the old "^ CROSS" pill

    def test_sideways_label(self):
        results = [
            _make_score_result(is_sideways=True, sideways_reasons=["ADX", "Chop"])
        ]
        html = generate_html_report(results, fetch_news=False)
        assert "Chop" in html

    def test_news_fetching_mocked(self):
        """With news enabled, fetch_stock_news should be called."""
        mock_news = [
            {
                "title": "Stock rises",
                "summary": "",
                "date": "2024-08-15",
                "publisher": "Reuters",
                "sentiment": "Good",
            }
        ]
        results = [_make_score_result()]
        with patch("scanner.backend.report.fetch_stock_news", return_value=mock_news):
            html = generate_html_report(results, fetch_news=True)
        assert "news-panel" in html
        assert "Stock rises" in html

    def test_prefetched_news_reused_without_fetch(self):
        """Rows carrying _news_items never hit the network on export."""
        results = [
            _make_score_result(
                ticker="PRE1",
                _news_items=[
                    {
                        "title": "Prefetched story",
                        "summary": "",
                        "date": "2026-09-25",
                        "provider": "Yahoo",  # GUI/prefetch key, not publisher
                        "sentiment": "Good",
                    }
                ],
            )
        ]
        with patch(
            "scanner.backend.report._fetch_news_parallel",
            side_effect=AssertionError("should not fetch"),
        ):
            html = generate_html_report(results, fetch_news=True)
        assert "Prefetched story" in html
        assert "Yahoo" in html  # provider key renders in publisher slot

    def test_fetches_only_tickers_without_prefetch(self):
        results = [
            _make_score_result(ticker="PRE", _news_items=[]),
            _make_score_result(ticker="MISS"),
        ]
        with patch(
            "scanner.backend.report._fetch_news_parallel",
            return_value={},
        ) as m:
            generate_html_report(results, fetch_news=True)
        assert m.call_args[0][0] == ["MISS"]

    def test_row_cells_render_grid_style_values(self):
        """Price/fund cells render grid-style bare values (no bars, ₹ price)."""
        html = generate_html_report(
            [_make_score_result(fundamentals=13.0)], fetch_news=False
        )
        assert '<td class="num sc-fund c-t2">13</td>' in html
        assert "bar-val" not in html and "bar-container" not in html
        assert '<td class="num price">₹2500</td>' in html


class TestReportPolish:
    def test_meta_chips_rendered(self):
        html = generate_html_report(
            [_make_score_result()], fetch_news=False, meta=["NSE ALL", "Daily"]
        )
        assert "NSE ALL" in html
        assert "Daily" in html

    def test_direction_header_replaces_duplicate_trend(self):
        html = generate_html_report([_make_score_result()], fetch_news=False)
        assert ">Dir<" in html
        # Score columns only — scope to the table head so the filter chip
        # label ("Trend") doesn't count as a column.
        thead = html.split("<thead>")[1].split("</thead>")[0]
        assert thead.count(">Trend<") == 0
        assert ">Vol/10<" in thead  # abbreviated grid label

    def test_sticky_header_and_print_css(self):
        css = _css_block()
        assert "position: sticky" in css
        assert "overflow-x: auto" not in css  # was killing page-level sticky
        # print-only rules must live inside the @media print block, never leak
        head, _, print_block = css.partition("@media print {")
        assert print_block, "missing @media print block"
        assert ".filters { display: none; }" not in head
        assert "position: static" not in head
        assert "#fff" in print_block
        assert ":root {" in print_block  # light vars need their own scope

    def test_modern_polish_surface(self):
        """Frozen identity columns, tabular numbers, staged motion, no CDN."""
        css = _css_block()
        assert "td.ticker { position: sticky" in css  # rank+ticker stay on h-scroll
        assert "td.rank, td.ticker { position: static; }" in css  # released on paper
        assert "font-variant-numeric: tabular-nums" in css
        assert "color-scheme: dark" in css
        assert "@keyframes rise" in css and "@keyframes fadeIn" in css
        html = generate_html_report([_make_score_result()], fetch_news=False)
        assert "googleapis" not in html  # offline-first: no font CDN request

    def test_news_panel_links_to_yahoo(self):
        results = [
            _make_score_result(
                ticker="RELIANCE",
                _news_items=[
                    {"title": "T", "summary": "", "date": "2026-09-25", "provider": "P"}
                ],
            )
        ]
        html = generate_html_report(results, fetch_news=True)
        assert "https://finance.yahoo.com/quote/RELIANCE/news/" in html


class TestPruneOld:
    def test_keeps_newest_only(self, tmp_path):
        for i in range(6):
            f = tmp_path / f"scanner_results_2024010{i}.csv"
            f.write_text("x")
            os.utime(f, (1000000 + i, 1000000 + i))
        prune_old(str(tmp_path), "scanner_results_*.csv", 4)
        assert len(list(tmp_path.glob("scanner_results_*.csv"))) == 4

    def test_default_bare_name_participates_in_html_retention(self, tmp_path):
        bare = tmp_path / "scanner_report.html"
        bare.write_text("old")
        os.utime(bare, (1000000, 1000000))
        for i in range(4):
            f = tmp_path / f"scanner_report_2024081{i}.html"
            f.write_text("x")
            os.utime(f, (1000100 + i, 1000100 + i))
        save_report(
            "<html>new</html>",
            str(tmp_path / "scanner_report_20240820_120000.html"),
            max_reports=4,
        )
        remaining = list(tmp_path.glob("scanner_report*.html"))
        assert len(remaining) == 4
        assert bare not in remaining  # oldest — now covered by the pattern


# ══════════════════════════════════════════════════════════════════════════════
# save_report
# ══════════════════════════════════════════════════════════════════════════════


class TestSaveReport:
    def test_creates_file(self, tmp_path):
        filepath = str(tmp_path / "test_report.html")
        save_report("<html>test</html>", filepath)
        assert os.path.exists(filepath)
        with open(filepath) as f:
            assert f.read() == "<html>test</html>"

    def test_returns_filename(self, tmp_path):
        filepath = str(tmp_path / "report.html")
        result = save_report("<html></html>", filepath)
        assert result == filepath

    def test_cleans_old_reports(self, tmp_path):
        """Should keep only max_reports files."""
        # Create 6 old report files
        for i in range(6):
            fpath = tmp_path / f"scanner_report_2024081{i}_120000.html"
            fpath.write_text(f"<html>old {i}</html>")
            # Stagger modification times
            os.utime(fpath, (1000000 + i, 1000000 + i))

        # Save a new one — should keep only 4 (max_reports default)
        filepath = str(tmp_path / "scanner_report_20240820_120000.html")
        save_report("<html>new</html>", filepath, max_reports=4)

        remaining = list(tmp_path.glob("scanner_report_*.html"))
        assert len(remaining) <= 4

    def test_overwrites_existing(self, tmp_path):
        filepath = str(tmp_path / "report.html")
        save_report("<html>v1</html>", filepath)
        save_report("<html>v2</html>", filepath)
        with open(filepath) as f:
            assert f.read() == "<html>v2</html>"


# ══════════════════════════════════════════════════════════════════════════════
# fetch_stock_news (mocked)
# ══════════════════════════════════════════════════════════════════════════════


class TestFetchStockNews:
    def test_returns_list(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.news = []
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_stock_news("RELIANCE")

        assert isinstance(result, list)

    def test_adds_ns_suffix(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.news = []
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            fetch_stock_news("RELIANCE")

        mock_yf.Ticker.assert_called_once_with("RELIANCE.NS")

    def test_no_ns_suffix_if_present(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.news = []
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            fetch_stock_news("RELIANCE.NS")

        mock_yf.Ticker.assert_called_once_with("RELIANCE.NS")

    def test_parses_news_items(self):
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        mock_news = [
            {
                "content": {
                    "title": "Stock surges on profit growth",
                    "summary": "Company reports strong results",
                    "pubDate": now,
                    "provider": {"displayName": "Reuters"},
                }
            }
        ]
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.news = mock_news
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_stock_news("RELIANCE", months_back=2)

        assert len(result) == 1
        assert result[0]["title"] == "Stock surges on profit growth"
        assert result[0]["publisher"] == "Reuters"
        assert result[0]["sentiment"] == "Good"
        assert result[0]["url"] == ""

    def test_extracts_url(self):
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        mock_news = [
            {
                "content": {
                    "title": "New format",
                    "summary": "",
                    "pubDate": now,
                    "provider": {"displayName": "Reuters"},
                    "canonicalUrl": {"url": "https://example.com/new"},
                }
            },
            {
                "content": {
                    "title": "Old format",
                    "summary": "",
                    "pubDate": now,
                    "provider": {"displayName": "Bloomberg"},
                },
                "link": "https://example.com/old",
            },
        ]
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.news = mock_news
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_stock_news("RELIANCE")

        assert result[0]["url"] == "https://example.com/new"
        assert result[1]["url"] == "https://example.com/old"

    def test_filters_old_news(self):
        old_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%dT%H:%M:%S")
        mock_news = [
            {
                "content": {
                    "title": "Old news",
                    "summary": "",
                    "pubDate": old_date,
                    "provider": {"displayName": "BBC"},
                }
            }
        ]
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.news = mock_news
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_stock_news("RELIANCE", months_back=2)

        assert len(result) == 0

    def test_respects_max_items(self):
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        mock_news = [
            {
                "content": {
                    "title": f"News {i}",
                    "summary": "",
                    "pubDate": now,
                    "provider": {"displayName": "Pub"},
                }
            }
            for i in range(20)
        ]
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.news = mock_news
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_stock_news("RELIANCE", max_items=3)

        assert len(result) == 3

    def test_exception_returns_empty(self):
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = Exception("network error")

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_stock_news("RELIANCE")

        assert result == []

    def test_import_error_returns_empty(self):
        with patch.dict("sys.modules", {"yfinance": None}):
            fetch_stock_news("RELIANCE")  # must not raise when yfinance is absent

    def test_news_cached_between_calls(self):
        """Second call for the same ticker replays the cache (no yfinance hit)."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.news = []
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            first = fetch_stock_news("CACHEDNEWS")
            second = fetch_stock_news("CACHEDNEWS")

        assert first == second == []
        mock_yf.Ticker.assert_called_once_with("CACHEDNEWS.NS")


# ══════════════════════════════════════════════════════════════════════════════
# fetch_news_for_ticker / fetch_news_batch (NSE→BSE fallback, provider key)
# ══════════════════════════════════════════════════════════════════════════════


class TestFetchNewsForTicker:
    def _mock_yf(self, news_by_ticker):
        mock_yf = MagicMock()

        def _ticker(t):
            m = MagicMock()
            m.news = news_by_ticker.get(t, [])
            return m

        mock_yf.Ticker.side_effect = _ticker
        return mock_yf

    def _item(self, title="Story"):
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        return {
            "content": {
                "title": title,
                "summary": "",
                "pubDate": now,
                "provider": {"displayName": "Reuters"},
            }
        }

    def test_returns_provider_keyed_items_from_ns(self):
        mock_yf = self._mock_yf({"TCS.NS": [self._item("NSE story")]})
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_news_for_ticker("TCS")

        assert len(result) == 1
        assert result[0]["title"] == "NSE story"
        assert result[0]["provider"] == "Reuters"  # GUI panel key
        assert "publisher" not in result[0]

    def test_falls_back_to_bo_when_ns_empty(self):
        mock_yf = self._mock_yf({"TCS.NS": [], "TCS.BO": [self._item("BSE story")]})
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_news_for_ticker("TCS")

        assert len(result) == 1
        assert result[0]["title"] == "BSE story"

    def test_returns_empty_when_both_suffixes_silent(self):
        mock_yf = self._mock_yf({})
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_news_for_ticker("TCS")
        assert result == []

    def test_explicit_suffix_is_kept(self):
        mock_yf = self._mock_yf({"TCS.BO": [self._item()]})
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_news_for_ticker("TCS.BO")
        assert len(result) == 1
        mock_yf.Ticker.assert_called_once_with("TCS.BO")

    def test_fetch_failure_returns_empty(self):
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = Exception("network error")
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_news_for_ticker("TCS")
        assert result == []


class TestFetchNewsBatch:
    def test_maps_every_ticker_to_its_news(self):
        def _fake(t, max_items=10, months_back=2):
            if t in ("A", "B"):
                return [
                    {
                        "title": f"{t} story",
                        "summary": "",
                        "date": "2026-08-01",
                        "provider": "Reuters",
                        "sentiment": "Good",
                    }
                ]
            return []

        with patch("scanner.backend.report.fetch_news_for_ticker", side_effect=_fake):
            news_map = fetch_news_batch(["A", "B", "C"])

        assert set(news_map) == {"A", "B", "C"}
        assert news_map["A"][0]["title"] == "A story"
        assert news_map["A"][0]["provider"] == "Reuters"
        assert news_map["C"] == []

    def test_empty_input_returns_empty_map(self):
        assert fetch_news_batch([]) == {}


class TestDetailPanelParity:
    """Expanded stock detail mirrors the app's detail panel."""

    def _rich_row(self, **overrides):
        base = {
            "px_tail": [100.0, 102.5, 101.0, 105.0, 110.0],
            "atr_pct": 4.5,
            "_shareholding": {
                "quarter": "Jun 2026",
                "series": {
                    "foreign_institutions": {
                        "latest": 17.2,
                        "total": -5.41,
                        "recent": -1.48,
                    },
                    "domestic_institutions": {
                        "latest": 21.1,
                        "total": 5.11,
                        "recent": 0.64,
                    },
                    "promoters": {"latest": 50.5, "total": 0.21, "recent": 0.48},
                },
            },
        }
        base.update(overrides)
        return _make_score_result(**base)

    def test_app_sections_render(self):
        html = generate_html_report([self._rich_row()], fetch_news=False)
        for s in (
            "Price (last 20 closes)",
            "Key Signals",
            "Score Breakdown",
            "Institutional Positioning",
            "Shareholding · Jun 2026",
            "Stoch",  # breakdown category the table row lacks
            "Volatility",
            "Fundamental",
            "17.2%",  # FII shareholding tile
            "50.5%",  # promoter tile
            "4.50%",  # ATR% signal chip
        ):
            assert s in html
        # Market-wide sources stay out of the per-stock panel.
        assert "FPI" not in html
        assert "Smart Money" not in html

    def test_promoter_fallback_without_shareholding(self):
        html = generate_html_report(
            [_make_score_result(_promoter_holding=51.8)], fetch_news=False
        )
        assert "Institutional Positioning" in html
        assert "51.8%" in html

    def test_na_tiles_when_not_enriched(self):
        html = generate_html_report([_make_score_result()], fetch_news=False)
        # The app card always renders; unsourced tiles settle to n/a.
        assert "Institutional Positioning" in html
        assert "n/a" in html
        assert html.count('class="inst-tile"') == 3

    def test_flow_fallback_renders_cr_tiles(self, monkeypatch):
        rows = [{"date": "2026-09-25", "fii": -1000.0, "dii": 500.0}] * 20 + [
            {"date": "2026-09-24", "fii": 100.0, "dii": -50.0}
        ] * 20
        monkeypatch.setattr(
            "scanner.api.indian_market.fetch_fii_dii_history", lambda *a, **k: rows
        )
        html = generate_html_report([_make_score_result()], fetch_news=False)
        assert "₹-1,000 Cr" in html  # FII newest-day flow value
        assert "₹+500 Cr" in html  # DII newest-day flow value
        assert "total" in html and "recent" in html  # flow_window_pcts sub-line
        assert html.count('class="inst-tile"') == 3

    def test_marker_fallback_renders_cr_tiles(self):
        html = generate_html_report(
            [_make_score_result(_fii_net=2838.0, _dii_net=-3694.0)], fetch_news=False
        )
        assert "₹+2,838 Cr" in html
        assert "₹-3,694 Cr" in html
        assert "no history" in html

    def test_shareholding_series_beats_flow(self, monkeypatch):
        monkeypatch.setattr(
            "scanner.api.indian_market.fetch_fii_dii_history",
            lambda *a, **k: [{"date": "d", "fii": -1.0, "dii": 2.0}] * 40,
        )
        html = generate_html_report([self._rich_row()], fetch_news=False)
        assert "17.2%" in html  # series % wins over the flow fallback
        assert "₹-1 Cr" not in html

    def test_detail_cards_match_app_layout(self):
        html = generate_html_report([self._rich_row()], fetch_news=False)
        # chart, signals, breakdown, reasons, institutional
        assert html.count('class="detail-card"') == 5
        # Reasons sit in the right column; institutional is full-width below.
        assert html.index("Why this trade?") < html.index("Institutional Positioning")
        assert "reasons-panel" not in html  # flat panel wrapper removed
        css = _css_block()
        assert ".detail-card {" in css
        assert ".detail-col { display: flex" in css
        _, _, print_block = css.partition("@media print {")
        assert ".detail-card" in print_block  # shadows reset on paper

    def test_detail_chart_axis_labels_and_cyan_line(self):
        html = generate_html_report([self._rich_row()], fetch_news=False)
        assert html.count('<text class="axis-lbl"') == 5  # hi → lo y-axis ticks
        assert 'stroke="#14b8a6"' in html  # app-cyan detail line
        assert "#10b981" in html  # row sparklines keep direction colors
        assert ".axis-lbl" in _css_block()

    def test_chart_needs_px_tail(self):
        html = generate_html_report([_make_score_result(atr_pct=4.5)], fetch_news=False)
        assert "Price (last 20 closes)" not in html
        assert "Key Signals" in html


class TestResponsiveReport:
    """Report mirrors the GUI responsive tiers (ui_kit.width_tier)."""

    def test_tier_order_mirrors_gui_hide_order(self):
        from scanner.shared.constants import RESULT_COLS
        from scanner.ui.ui_kit import COL_HIDE_ORDER

        # Report labels are RESULT_COLS verbatim — mapping is the identity.
        assert [lbl for lbl, _ in _REPORT_COLS] == RESULT_COLS
        t1 = {lbl for lbl, tier in _REPORT_COLS if tier == 1}
        t2 = {lbl for lbl, tier in _REPORT_COLS if tier == 2}
        assert t1 == {"Chop", "Dir", "ADX"}
        assert t2 == {"Vol/10", "RS/10", "F/20"}
        assert t1 | t2 == set(COL_HIDE_ORDER)

    def test_header_tier_classes_keep_sort_indexes(self):
        head = _table_head_html()
        ths = re.findall(r"<th([^>]*)>([^<]+)</th>", head)
        assert [label for _, label in ths] == [lbl for lbl, _ in _REPORT_COLS]
        for i, ((attrs, label), (_, tier)) in enumerate(zip(ths, _REPORT_COLS)):
            if label == "1M":
                assert "sortTable" not in attrs  # spark column never sorts
            else:
                assert f"sortTable({i})" in attrs  # positional index intact
            assert ('class="c-t1"' in attrs) == (tier == 1)
            assert ('class="c-t2"' in attrs) == (tier == 2)

    def test_row_cells_carry_tier_classes(self):
        html = generate_html_report([_make_score_result()], fetch_news=False)
        assert html.count('<td class="num c-t1">') == 1  # ADX
        assert html.count('<td class="c-t1 ') == 2  # Dir + Chop (extra class each)
        assert len(re.findall(r'<td class="[^"]*c-t2', html)) == 3  # Vol/RS/Fund
        # Kept columns never get a drop class.
        assert '<td class="num price">' in html

    def test_media_queries_use_gui_breakpoints(self):
        css = _css_block()
        assert "@media (max-width: 1599px)" in css
        assert "@media (max-width: 1399px)" in css
        assert "th.c-t1, td.c-t1 { display: none; }" in css
        assert "th.c-t2, td.c-t2 { display: none; }" in css
        # Drop rules must live outside @media print, shadows reset inside it.
        head, _, print_block = css.partition("@media print {")
        assert "display: none" in head
        assert "box-shadow: none" in print_block

    def test_elevation_matches_app_polish(self):
        css = _css_block()
        assert css.count("box-shadow: 0") >= 5  # stat/table/news/detail surfaces
        assert "linear-gradient(135deg, var(--surface)" in css  # detail host

    def test_news_row_colspan_follows_visible_columns(self):
        js = _js_block()
        assert "function syncColspan()" in js
        assert 'addEventListener("resize"' in js  # debounced relayout
        assert "fitColumns(); syncColspan()" in js  # fit runs first

    def test_content_fit_hides_until_table_fits(self):
        js = _js_block()
        assert "function fitColumns()" in js
        assert "FIT_ORDER" in js
        assert 'addEventListener("load", relayout)' in js  # after web fonts
        css = _css_block()
        assert ".fit-hidden { display: none; }" in css
        head, _, print_block = css.partition("@media print {")
        assert "fit-hidden" in head
        assert "th.fit-hidden, td.fit-hidden { display: table-cell; }" in print_block

    def test_threshold_displayed_cleanly(self):
        html = generate_html_report(
            [_make_score_result()], fetch_news=False, threshold=55.00000000000001
        )
        assert "55.00000000000001" not in html
        assert "Threshold 55+" in html
        assert "Passed · 55+" in html

    def test_compact_stat_strip(self):
        html = generate_html_report([_make_score_result()], fetch_news=False)
        assert html.count('class="stat-meta"') == 4  # inline number + labels
        css = _css_block()
        assert ".summary { display: contents; }" in css  # stats as grid cells
        assert "grid-template-columns: repeat(4, minmax(0,1fr))" in css
        assert "histogram" not in css  # score-distribution strip removed
        assert (
            ".stat::before { content: ''; position: absolute; top: 0; bottom: 0; left: 0; width: 3px;"
            in css
        )

    def test_modern_grid_rows(self):
        css = _css_block()
        assert "tbody tr:not(.news-row) td { white-space: nowrap; }" in css
        assert "tbody tr:not(.news-row).row-alt" in css  # odd-rank zebra band
        assert "td.rank::before" in css  # rating accent rail
        # Compact rows keep news panels wrappable.
        assert "tbody tr:not(.news-row)" in css


class TestReportElegance:
    """Bold redesign: hero, overview band, filter chips, pills, motion."""

    def test_hero_header_renders(self):
        html = generate_html_report(
            [_make_score_result()], fetch_news=False, meta=["NSE ALL", "Daily"]
        )
        assert 'class="hero"' in html
        assert 'class="hero-pill"' in html
        assert 'class="market-box"' in html
        assert "Find Your Next Swing Trade" in html  # app hero title
        assert "ENTRY signals" in html  # app status line
        assert "<span>NSE ALL</span>" in html
        assert "<span>Daily</span>" in html
        assert "Generated locally" in html

    def test_filters_are_chips_not_selects(self):
        html = generate_html_report([_make_score_result()], fetch_news=False)
        assert "<select" not in html
        assert '<input type="text" id="search"' in html
        for fid in ("minScore", "trendFilter", "signalFilter", "newsFilter"):
            assert f'<input type="hidden" id="{fid}"' in html
        js = _js_block()
        assert "function setFilter" in js
        assert 'classList.add("active")' in js

    def test_results_head_shows_app_filter_context(self):
        html = generate_html_report(
            [_make_score_result(_fii_is_buying=True)],
            fetch_news=False,
            threshold=50.0,
            filters={
                "search": "",
                "rating": "GOOD",
                "score": 50.0,
                "fii_dii": True,
                "price100": False,
                "fund0": False,
                "scanned": 153,
                "above": 84,
            },
        )
        assert '<span class="results-title">Scan Results</span>' in html
        # Native checkboxes (interactive like the app): only FII/DII checked.
        assert html.count('type="checkbox"') == 3
        assert (
            '<input type="checkbox" id="f-fii" checked onchange="filterTable()">'
            in html
        )
        assert ">Show stocks with FII/DII data</label>" in html
        assert ">Hide stocks below ₹100</label>" in html
        assert ">Hide zero fundamental score</label>" in html
        # Chips are clearable buttons wired to clearFilter().
        assert 'id="chip-rating" data-rating="GOOD"' in html
        assert ">rating Good  ✕</button>" in html
        assert ">score ≥ 50</button>" in html
        assert "clearFilter('rating')" in html and "clearFilter('score')" in html
        # Status line: app spacing + dataset the live recount reads back.
        assert (
            'id="countLabel" data-scanned="153" data-above="84" data-threshold="50"'
        ) in html
        assert (
            ">153 scanned  |  84 above 50+  |  "
            "filter: rating Good, score ≥ 50, FII/DII data (1)</span>"
        ) in html

    def test_results_head_without_filters_has_no_suffix(self):
        html = generate_html_report(
            [_make_score_result()], fetch_news=False, threshold=50.0
        )
        assert 'id="f-fii" checked' not in html  # boxes start unchecked
        assert "| filter: " not in html
        assert ">1 scanned  |  1 above 50+</span>" in html

    def test_filter_controls_drive_client_side_filtering(self):
        """JS re-applies the app filter state: row data attrs + recount."""
        js = _js_block()
        assert "function clearFilter(" in js
        assert "function syncResultsHead(" in js
        assert 'getElementById("f-fii")' in js
        # Score-clear targets a stable id, not a string-matched onclick.
        assert 'getElementById("chip-score-all")' in js
        for attr in (
            "data-score",
            "data-rating",
            "data-price",
            "data-fund",
            "data-fii",
        ):
            assert attr in js
        assert "syncResultsHead(shown)" in js
        assert "filterTable();  // apply export-time filters" in js
        html = generate_html_report([_make_score_result()], fetch_news=False)
        assert 'id="chip-score-all"' in html
        # attrs ride the row's opening tag, after data-ticker
        row = html.split('data-ticker="RELIANCE"', 1)[1].split("</tr>", 1)[0]
        assert 'data-score="65"' in row  # raw score, not the rounded display
        assert 'data-rating="GOOD"' in row
        assert 'data-price="2500"' in row
        assert 'data-fund="10"' in row
        assert 'data-fii="0"' in row

    def test_search_filter_displays_uppercase_like_app(self):
        html = generate_html_report(
            [_make_score_result()],
            fetch_news=False,
            threshold=50.0,
            filters={"search": "tata"},
        )
        assert ">1 scanned  |  1 above 50+  |  filter: 'TATA' (0)</span>" in html
        assert "'TATA'  ✕</button>" in html

    def test_ticker_cell_has_copy_button(self):
        html = generate_html_report([_make_score_result()], fetch_news=False)
        assert (
            'class="copy-t" onclick="copyTicker(event,this)" title="Copy ticker">'
            in html
        )
        js = _js_block()
        assert "function copyTicker(" in js
        assert "dataset.ticker" in js

    def test_default_score_chip_matches_threshold(self):
        for threshold, value in ((70, "70"), (50, "50"), (45, "0")):
            html = generate_html_report(
                [_make_score_result()], fetch_news=False, threshold=threshold
            )
            assert f'<input type="hidden" id="minScore" value="{value}">' in html

    def test_score_and_entry_cells(self):
        html = generate_html_report([_make_score_result()], fetch_news=False)
        assert '<td class="score s-good">65</td>' in html  # total 65 → GOOD
        assert '<td class="entry yes">YES</td>' in html  # entry_signal True
        assert "score-excellent" not in html  # glow classes gone

    def test_overview_band_renders(self):
        html = generate_html_report([_make_score_result()], fetch_news=False)
        assert 'class="overview"' in html
        assert "Score distribution" not in html
        empty = generate_html_report([], fetch_news=False)
        assert 'class="overview solo"' in empty
        assert 'class="histogram"' not in empty

    def test_print_resets_glass_and_motion(self):
        css = _css_block()
        head, _, print_block = css.partition("@media print {")
        assert "backdrop-filter: blur" in head  # glass is screen-only
        assert "prefers-reduced-motion" in head
        assert "backdrop-filter: none" in print_block
        assert "animation: none" in print_block

    def test_detail_chart_gets_gridlines_and_dot(self):
        html = generate_html_report(
            [_make_score_result(px_tail=[100.0, 102.0, 101.5, 104.0])],
            fetch_news=False,
        )
        assert html.count('class="grid"') == 5  # big chart: 5 axis-tick lines
        assert html.count("<circle") == 1  # endpoint dot
        bare = generate_html_report([_make_score_result()], fetch_news=False)
        assert 'class="grid"' not in bare  # no px_tail → no detail chart
