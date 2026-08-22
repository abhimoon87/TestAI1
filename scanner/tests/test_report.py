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
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from scanner.report import (
    _sentiment,
    _parse_date,
    _score_class,
    generate_html_report,
    save_report,
    fetch_stock_news,
    SENTIMENT_GOOD,
    SENTIMENT_BAD,
)


# ── Helpers ────────────────────────────────────────────────────────────────


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
        assert "72.5" in html

    def test_contains_threshold(self):
        results = [_make_score_result()]
        html = generate_html_report(results, threshold=50.0, fetch_news=False)
        assert "50.0" in html

    def test_sorted_by_score_descending(self):
        results = [
            _make_score_result(ticker="A", total=30.0),
            _make_score_result(ticker="B", total=80.0),
            _make_score_result(ticker="C", total=50.0),
        ]
        html = generate_html_report(results, fetch_news=False)
        # B (80) should appear before A (30) in the HTML
        pos_b = html.index("B")
        pos_a = html.index("data-ticker=\"A\"")
        assert pos_b < pos_a

    def test_highlight_class_for_passing(self):
        results = [_make_score_result(total=60.0)]
        html = generate_html_report(results, threshold=50.0, fetch_news=False)
        assert "highlight" in html

    def test_no_news_when_disabled(self):
        results = [_make_score_result()]
        html = generate_html_report(results, fetch_news=False)
        # When fetch_news=False, no news data rows (with id=) should appear
        assert 'id="news-' not in html

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
        assert "▲" in html

    def test_bear_trend_icon(self):
        results = [_make_score_result(trend_dir="Bear", trend_color="bear")]
        html = generate_html_report(results, fetch_news=False)
        assert "▼" in html

    def test_ma_cross_signal(self):
        results = [_make_score_result(ma_crossed_above=True, crossover_bars_ago=2)]
        html = generate_html_report(results, fetch_news=False)
        assert "CROSS" in html

    def test_sideways_label(self):
        results = [_make_score_result(is_sideways=True, sideways_reasons=["ADX", "Chop"])]
        html = generate_html_report(results, fetch_news=False)
        assert "Chop" in html

    def test_news_fetching_mocked(self):
        """With news enabled, fetch_stock_news should be called."""
        mock_news = [{"title": "Stock rises", "summary": "", "date": "2024-08-15",
                       "publisher": "Reuters", "sentiment": "Good"}]
        results = [_make_score_result()]
        with patch("scanner.report.fetch_stock_news", return_value=mock_news):
            html = generate_html_report(results, fetch_news=True)
        assert "news-panel" in html
        assert "Stock rises" in html


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
        mock_news = [{
            "content": {
                "title": "Stock surges on profit growth",
                "summary": "Company reports strong results",
                "pubDate": now,
                "provider": {"displayName": "Reuters"},
            }
        }]
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

    def test_filters_old_news(self):
        old_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%dT%H:%M:%S")
        mock_news = [{
            "content": {
                "title": "Old news",
                "summary": "",
                "pubDate": old_date,
                "provider": {"displayName": "BBC"},
            }
        }]
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
            {"content": {"title": f"News {i}", "summary": "", "pubDate": now,
                          "provider": {"displayName": "Pub"}}}
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
            result = fetch_stock_news("RELIANCE")
        assert result == []
