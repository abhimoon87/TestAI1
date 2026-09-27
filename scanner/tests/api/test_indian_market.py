"""Tests for scanner.api.indian_market — FII/DII endpoint, cash-flow history
(moneycontrol), delivery volume and flow-window math."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scanner.api.indian_market import (
    _FLOW_CACHE,
    _INDIA_CACHE,
    _delivery_from_df,
    _pct_change,
    fetch_fii_dii_activity,
    fetch_fii_dii_history,
    flow_window_pcts,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    for cache in (_INDIA_CACHE, _FLOW_CACHE):
        cache.clear()
    yield
    for cache in (_INDIA_CACHE, _FLOW_CACHE):
        cache.clear()


def _resp(payload=None, text="", status=200):
    m = MagicMock()
    m.status_code = status
    m.text = text
    m.json.return_value = payload
    m.raise_for_status.return_value = None
    return m


_FIIDII_PAYLOAD = [
    {
        "category": "FII/FPI",
        "date": "25-Sep-2026",
        "buyValue": "1000.50",
        "sellValue": "800.25",
        "netValue": "200.25",
    },
    {
        "category": "DII",
        "date": "25-Sep-2026",
        "buyValue": "500.00",
        "sellValue": "750.00",
        "netValue": "-250.00",
    },
]


class TestFiiDiiEndpoint:
    """fiidiiTradeReact parse — replaces the dead nselib derivatives_market call."""

    def test_parses_fii_fpi_label_and_math(self):
        with patch("requests.get", return_value=_resp(payload=_FIIDII_PAYLOAD)) as g:
            a = fetch_fii_dii_activity()
        assert a is not None
        assert a.fii_buy == 1000.50
        assert a.fii_sell == 800.25
        assert a.fii_net == 200.25
        assert a.fii_is_buying is True
        assert a.dii_net == -250.00
        assert a.dii_is_buying is False
        assert a.date == "25-Sep-2026"
        assert g.call_count == 1

    def test_second_call_hits_cache(self):
        with patch("requests.get", return_value=_resp(payload=_FIIDII_PAYLOAD)) as g:
            fetch_fii_dii_activity()
            a = fetch_fii_dii_activity()
        assert a.cached is True
        assert g.call_count == 1

    def test_network_error_returns_none(self):
        with patch("requests.get", side_effect=OSError("down")):
            assert fetch_fii_dii_activity() is None

    def test_unknown_category_returns_none(self):
        with patch("requests.get", return_value=_resp(payload=[{"category": "X"}])):
            assert fetch_fii_dii_activity() is None

    def test_non_list_payload_returns_none(self):
        with patch("requests.get", return_value=_resp(payload={"oops": True})):
            assert fetch_fii_dii_activity() is None


class TestPctChange:
    def test_basic(self):
        assert _pct_change(50000, 25000) == 100.0

    def test_negative_baseline(self):
        assert _pct_change(-20000, -50000) == 60.0

    def test_floor_on_tiny_baseline(self):
        # (500-100)/10_000 — near-zero nets must not explode the %.
        assert _pct_change(500, 100) == 4.0


_NEXT_DATA_HTML = (
    '<html><script id="__NEXT_DATA__" type="application/json">'
    + json.dumps(
        {
            "props": {
                "pageProps": {
                    "FiiDiiData": {
                        "fiiDiiData": [
                            {
                                "date": "2026-09-25",
                                "fiiCM": "-3,693.93",
                                "diiCM": "2,838.17",
                            },
                            {"date": "2026-09-24", "fiiCM": "1,234.5", "diiCM": "-10"},
                        ]
                    }
                }
            }
        }
    )
    + "</script></html>"
)


class TestFlowHistory:
    """Moneycontrol __NEXT_DATA__ — 30 sessions of fiiCM/diiCM cash flows."""

    def test_parses_newest_first_floats(self):
        with patch("requests.get", return_value=_resp(text=_NEXT_DATA_HTML)) as g:
            hist = fetch_fii_dii_history()
            calls = g.call_count
            again = fetch_fii_dii_history()
        assert hist == [
            {"date": "2026-09-25", "fii": -3693.93, "dii": 2838.17},
            {"date": "2026-09-24", "fii": 1234.5, "dii": -10.0},
        ]
        assert again == hist
        assert g.call_count == calls  # cached

    def test_missing_next_data_returns_none(self):
        with patch("requests.get", return_value=_resp(text="<html></html>")):
            assert fetch_fii_dii_history() is None
        with patch("requests.get") as g:  # failures are not cached
            assert fetch_fii_dii_history(cache_only=True) is None
        g.assert_not_called()

    def test_network_error_returns_none(self):
        with patch("requests.get", side_effect=OSError("down")):
            assert fetch_fii_dii_history() is None


class TestFlowWindowPcts:
    """Window-sum % changes over newest-first history (flow tiles)."""

    @staticmethod
    def _hist(rows):
        return [{"date": str(i), "fii": f, "dii": 0.0} for i, f in enumerate(rows)]

    def test_ten_rows_total_equals_recent_window(self):
        # now: 5 × -1000 = -5000, then: 5 × -2000 = -10000 → +50.0%
        total, recent = flow_window_pcts(self._hist([-1000] * 5 + [-2000] * 5), "fii")
        assert total == 50.0
        assert recent == 50.0

    def test_short_history_shrinks_total_window(self):
        # 6 rows → total window 3 vs 3; recent (needs 10) stays None.
        total, recent = flow_window_pcts(self._hist([-1000] * 3 + [-2000] * 3), "fii")
        assert total == 50.0
        assert recent is None

    def test_empty_history_returns_none_pair(self):
        assert flow_window_pcts([], "fii") == (None, None)
        assert flow_window_pcts(None, "fii") == (None, None)

    def test_zero_windows_report_no_change(self):
        total, recent = flow_window_pcts(self._hist([0.0] * 10), "fii")
        assert (total, recent) == (0.0, 0.0)


_NSE_COLS = [
    "Symbol",
    "Series",
    "Date",
    "PrevClose",
    "OpenPrice",
    "HighPrice",
    "LowPrice",
    "LastPrice",
    "ClosePrice",
    "AveragePrice",
    "TotalTradedQuantity",
    "TurnoverInRs",
    "No.ofTrades",
    "DeliverableQty",
    "%DlyQttoTradedQty",
]


def _nse_df(rows):
    """Build an nselib-shaped frame; rows are dicts of column → value."""
    return pd.DataFrame([{c: r.get(c, 0) for c in _NSE_COLS} for r in rows])


# Two days, oldest-first (nselib may return concatenated chunks that way).
_DELIVERY_ROWS = [
    {
        "Date": "24-Sep-2026",
        "TotalTradedQuantity": 1297709,
        "No.ofTrades": 17530,
        "DeliverableQty": 556000,
        "%DlyQttoTradedQty": 40.0,
    },
    {
        "Date": "25-Sep-2026",
        "TotalTradedQuantity": 1070523,
        "No.ofTrades": 16819,
        "DeliverableQty": 484704,
        "%DlyQttoTradedQty": 45.28,
    },
]


class TestDeliveryFromDf:
    """Delivery % derivation — newest day, NSE's own %, sane fallbacks."""

    def test_prefers_nsdl_pct_column_and_newest_date(self):
        d = _delivery_from_df(_nse_df(_DELIVERY_ROWS), "LICHSGFIN")
        assert d is not None
        assert d.delivery_pct == 45.28  # NSE's own %, newest day — not a
        assert d.total_volume == 1070523  # last-match % col over volume (bug)
        assert d.delivery_volume == 484704
        assert d.delivery_change_pct == 5.28
        assert d.is_high_delivery is False

    def test_falls_back_to_ratio_without_pct_column(self):
        rows = [
            {k: v for k, v in r.items() if k != "%DlyQttoTradedQty"}
            for r in _DELIVERY_ROWS
        ]
        d = _delivery_from_df(_nse_df(rows), "X")
        assert d.delivery_pct == round(484704 / 1070523 * 100, 2)
        assert d.is_high_delivery is False

    def test_garbage_pct_falls_back_to_ratio(self):
        rows = [dict(r) for r in _DELIVERY_ROWS]
        rows[-1]["%DlyQttoTradedQty"] = 3696575.0  # impossible % → ratio wins
        d = _delivery_from_df(_nse_df(rows), "X")
        assert d.delivery_pct == round(484704 / 1070523 * 100, 2)

    def test_comma_strings_are_parsed(self):
        rows = [dict(r) for r in _DELIVERY_ROWS]
        rows[-1]["TotalTradedQuantity"] = "10,70,523"
        rows[-1]["DeliverableQty"] = "4,84,704"
        rows = [{k: v for k, v in r.items() if k != "%DlyQttoTradedQty"} for r in rows]
        d = _delivery_from_df(_nse_df(rows), "X")
        assert d.total_volume == 1070523
        assert d.delivery_volume == 484704

    def test_empty_or_unusable_frame_returns_none(self):
        assert _delivery_from_df(pd.DataFrame(), "X") is None
        assert _delivery_from_df(None, "X") is None
        bad = _nse_df([{"Date": "25-Sep-2026"}])  # all-zero quantities
        assert _delivery_from_df(bad, "X") is None
