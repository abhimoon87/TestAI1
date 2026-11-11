"""Insider data providers — offline unit tests for scanner.api.insider_data.

Every HTTP call is mocked; no test touches the network.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

import scanner.api.insider_data as insider_mod


@pytest.fixture(autouse=True)
def _insider_isolation(monkeypatch):
    """No leaked API keys and an empty TTLCache around every test."""
    monkeypatch.delenv("ALETHEIA_API_KEY", raising=False)
    monkeypatch.delenv("CONGRESS_API_KEY", raising=False)
    insider_mod._INSIDER_CACHE.clear()
    yield
    insider_mod._INSIDER_CACHE.clear()


def _resp(payload=None):
    """Mocked response: raise_for_status passes, json() returns payload."""
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = payload
    m.raise_for_status.return_value = None
    return m


def _err(msg="boom"):
    """Mocked response whose raise_for_status raises HTTPError."""
    m = MagicMock()
    m.raise_for_status.side_effect = requests.HTTPError(msg)
    return m


# ── Aletheia ────────────────────────────────────────────────────────────────

_ALETHEIA_PAYLOAD = {
    "trades": [
        {
            "insider_name": "Alice",
            "title": "CEO",
            "transaction_type": "BUY",
            "shares": 100,
            "price": 10.0,
            "filing_date": "2026-01-05",
            "filing_url": "https://example.com/f1",
        },
        {
            "insider_name": "Bob",
            "title": "CFO",
            "transaction_type": "Sell",
            "shares": 50,
            "price": 10.0,
            "filing_date": "2026-01-06",
            "filing_url": "https://example.com/f2",
        },
    ]
}


def test_aletheia_missing_api_key_returns_none():
    with patch("requests.get") as get:
        assert insider_mod.fetch_aletheia_insider("INFY") is None
    get.assert_not_called()


def test_aletheia_maps_trades_and_scores():
    with patch("requests.get", return_value=_resp(_ALETHEIA_PAYLOAD)) as get:
        out = insider_mod.fetch_aletheia_insider("INFY.NS", api_key="k")

    get.assert_called_once_with(
        "https://api.aletheia.com/v1/insider-trading/INFY",  # .NS/.BO stripped
        headers={"Authorization": "Bearer k"},
        params={"days": 90, "limit": 50},
        timeout=10,
    )
    assert out is not None and out.cached is False
    assert out.ticker == "INFY.NS"  # original ticker kept on the result
    assert out.total_buys == 1 and out.total_sells == 1
    assert out.buy_value == 1000.0 and out.sell_value == 500.0
    assert out.net_insider_activity == 500.0
    assert out.insider_score == 0.333  # (1000 - 500) / 1500
    first, second = out.recent_trades
    assert first.transaction_type == "buy"  # lowercased
    assert first.value == 1000.0  # shares * price
    assert second.transaction_type == "sell"


def test_aletheia_fallback_keys_and_data_envelope():
    """Alternate field names + the {"data": [...]} envelope are accepted."""
    payload = {
        "data": [
            {
                "quantity": 100,
                "share_price": 2.5,
                "reporter_name": "Jane",
                "officer_title": "CFO",
                "transaction_type": "purchase",
                "transaction_date": "2026-02-01",
                "filing_date": "2026-02-03",
            },
            {
                "quantity": 10,
                "share_price": 1.0,
                "reporter_name": "John",
                "officer_title": "Director",
                "transaction_type": "sale",
                "transaction_date": "2026-02-02",
            },
        ]
    }
    with patch("requests.get", return_value=_resp(payload)):
        out = insider_mod.fetch_aletheia_insider("T", api_key="k")
    assert out is not None
    first, second = out.recent_trades
    assert first.insider_name == "Jane" and first.title == "CFO"
    assert first.value == 250.0
    assert first.date == "2026-02-03"  # filing_date wins over transaction_date
    assert second.date == "2026-02-02"  # falls back to transaction_date
    assert out.buy_value == 250.0 and out.sell_value == 10.0


def test_aletheia_defaults_and_unclassified_trade():
    """Missing fields default to 0/""/Unknown; unclassified types count nowhere."""
    payload = {
        "trades": [
            {"transaction_type": "gift"},
            {"transaction_type": "PURCHASE", "shares": 1, "price": 1.0},
        ]
    }
    with patch("requests.get", return_value=_resp(payload)):
        out = insider_mod.fetch_aletheia_insider("T", api_key="k")
    assert out is not None
    gift, buy = out.recent_trades
    assert gift.insider_name == "Unknown"
    assert gift.title == "" and gift.date == "" and gift.filing_url == ""
    assert gift.shares == 0 and gift.price == 0.0 and gift.value == 0.0
    assert out.total_buys == 1 and out.total_sells == 0  # gift not counted
    assert out.buy_value == 1.0 and out.sell_value == 0.0
    assert out.net_insider_activity == 1.0
    assert out.insider_score == 1.0
    assert buy.transaction_type == "purchase"  # "PURCHASE" lowercased


def test_aletheia_empty_payload_zeroes():
    with patch("requests.get", return_value=_resp({})):
        out = insider_mod.fetch_aletheia_insider("EMPTY", api_key="k")
    assert out is not None and out.cached is False
    assert out.recent_trades == []
    assert out.total_buys == 0 and out.total_sells == 0
    assert out.buy_value == 0.0 and out.sell_value == 0.0
    assert out.net_insider_activity == 0.0
    assert out.insider_score == 0.0  # no value traded → score 0, not div-by-zero


def test_aletheia_caps_recent_trades_at_ten():
    payload = {"trades": [{"transaction_type": "buy", "shares": 1, "price": 1.0}] * 12}
    with patch("requests.get", return_value=_resp(payload)):
        out = insider_mod.fetch_aletheia_insider("T", api_key="k")
    assert out is not None
    assert len(out.recent_trades) == 10  # list trimmed to 10…
    assert out.total_buys == 12  # …but counts cover every trade
    assert out.buy_value == 12.0


def test_aletheia_http_error_returns_none():
    with patch("requests.get", side_effect=requests.RequestException("down")):
        assert insider_mod.fetch_aletheia_insider("T", api_key="k") is None


def test_aletheia_malformed_json_returns_none():
    bad = MagicMock()
    bad.json.side_effect = ValueError("not json")
    with patch("requests.get", return_value=bad):
        assert insider_mod.fetch_aletheia_insider("T", api_key="k") is None


def test_aletheia_second_call_served_from_cache():
    payload = {
        "trades": [
            {"insider_name": "Alice", "transaction_type": "buy", "shares": 1, "price": 1.0}
        ]
    }
    with patch("requests.get", return_value=_resp(payload)) as get:
        first = insider_mod.fetch_aletheia_insider("CACHET", api_key="k")
        second = insider_mod.fetch_aletheia_insider("CACHET", api_key="k")
    assert first is not None and second is not None
    assert first.cached is False and second.cached is True
    assert get.call_count == 1
    # cached rows rehydrate as plain dicts (the dataclass does not coerce them)
    assert second.recent_trades[0]["insider_name"] == "Alice"


# ── CongressInvests ─────────────────────────────────────────────────────────

_CONGRESS_PAYLOAD = {
    "trades": [
        {
            "member_name": "Rep A",
            "party": "D",
            "transaction_type": "purchase",
            "asset_description": "INFY",
            "amount_range": "$1,001 - $15,000",
            "disclosure_date": "2026-01-03",
            "transaction_date": "2026-01-01",
            "url": "https://example.com/c1",
        },
        {
            "member_name": "Rep B",
            "party": "R",
            "transaction_type": "Purchase",
            "asset_description": "INFY",
            "amount_range": "$1,001 - $15,000",
            "disclosure_date": "2026-01-04",
            "transaction_date": "2026-01-02",
            "url": "https://example.com/c2",
        },
        {
            "member_name": "Rep C",
            "party": "I",
            "transaction_type": "SALE",
            "asset_description": "INFY",
            "amount_range": "$15,001 - $50,000",
            "disclosure_date": "2026-01-05",
            "transaction_date": "2026-01-03",
            "url": "https://example.com/c3",
        },
    ]
}


def test_congress_missing_api_key_returns_none():
    with patch("requests.get") as get:
        assert insider_mod.fetch_congress_invests("RELIANCE") is None
    get.assert_not_called()


def test_congress_maps_trades_and_scores():
    with patch("requests.get", return_value=_resp(_CONGRESS_PAYLOAD)) as get:
        out = insider_mod.fetch_congress_invests("RELIANCE.NS", api_key="ck")

    get.assert_called_once_with(
        "https://api.congressinvests.com/v1/trades",
        headers={"Authorization": "Bearer ck"},
        params={"ticker": "RELIANCE", "limit": 20},
        timeout=10,
    )
    assert out is not None and out.cached is False
    assert out.ticker == "RELIANCE.NS"
    assert out.buy_count == 2 and out.sell_count == 1
    assert out.net_congressional_activity == 1.0
    assert out.congressional_score == 0.333  # (2 - 1) / 3
    assert [t.member for t in out.recent_trades] == ["Rep A", "Rep B", "Rep C"]
    assert out.recent_trades[0].amount_range == "$1,001 - $15,000"
    assert out.recent_trades[1].transaction_type == "purchase"  # lowercased
    assert out.recent_trades[2].transaction_type == "sale"  # "SALE" lowercased


def test_congress_fallback_keys_and_date_default():
    """representative/asset/amount/source_url keys; date falls back to disclosure."""
    payload = {
        "data": [
            {
                "representative": "Sen X",
                "party": "D",
                "transaction_type": "Sale",
                "asset": "Apple Inc",
                "amount": "$1,001 - $15,000",
                "disclosure_date": "2026-02-01",
            }
        ]
    }
    with patch("requests.get", return_value=_resp(payload)):
        out = insider_mod.fetch_congress_invests("T", api_key="ck")
    assert out is not None and len(out.recent_trades) == 1
    trade = out.recent_trades[0]
    assert trade.member == "Sen X"
    assert trade.asset == "Apple Inc"
    assert trade.amount_range == "$1,001 - $15,000"
    assert trade.disclosure_date == "2026-02-01"
    assert trade.transaction_date == "2026-02-01"  # defaults to disclosure_date
    assert trade.url == ""
    assert out.buy_count == 0 and out.sell_count == 1
    assert out.congressional_score == -1.0


def test_congress_empty_payload_zeroes():
    with patch("requests.get", return_value=_resp({})):
        out = insider_mod.fetch_congress_invests("EMPTY", api_key="ck")
    assert out is not None and out.cached is False
    assert out.recent_trades == []
    assert out.buy_count == 0 and out.sell_count == 0
    assert out.net_congressional_activity == 0.0
    assert out.congressional_score == 0.0


def test_congress_caps_recent_trades_at_ten():
    payload = {
        "trades": [
            {"member_name": f"Rep {i}", "transaction_type": "purchase"}
            for i in range(12)
        ]
    }
    with patch("requests.get", return_value=_resp(payload)):
        out = insider_mod.fetch_congress_invests("T", api_key="ck")
    assert out is not None
    assert len(out.recent_trades) == 10  # list trimmed to 10…
    assert out.buy_count == 12  # …counts still cover every trade
    assert out.congressional_score == 1.0


def test_congress_http_error_returns_none():
    with patch("requests.get", return_value=_err("500")):
        assert insider_mod.fetch_congress_invests("T", api_key="ck") is None


def test_congress_second_call_served_from_cache():
    payload = {"trades": [{"member_name": "Rep A", "transaction_type": "purchase"}]}
    with patch("requests.get", return_value=_resp(payload)) as get:
        first = insider_mod.fetch_congress_invests("CACHET", api_key="ck")
        second = insider_mod.fetch_congress_invests("CACHET", api_key="ck")
    assert first is not None and second is not None
    assert first.cached is False and second.cached is True
    assert get.call_count == 1
    # cached rows rehydrate as plain dicts (the dataclass does not coerce them)
    assert second.recent_trades[0]["member"] == "Rep A"


# ── SEC EDGAR ───────────────────────────────────────────────────────────────

_TICKERS_WIT = {"0": {"cik_str": 1163385, "ticker": "WIT", "title": "Infosys Ltd"}}

_FILINGS_WIT = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K", "4", "4", "4", "4"],
            "filingDate": [
                "2026-01-01",
                "2025-10-30",
                "2025-12-15",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
                "2026-01-05",
            ],
            "primaryDocDescription": [
                "Annual Report",
                "Quarterly Report",
                "Current Report",
                "Form 4",
                "Form 4",
                "Form 4",
                "Form 4",
            ],
            "accessionNumber": [
                "0001163385-26-000001",
                "0001163385-25-000009",
                "0001163385-25-000044",
                "0001163385-26-000002",
                "0001163385-26-000003",
                "0001163385-26-000004",
                "0001163385-26-000005",
            ],
        }
    }
}


def test_sec_edgar_two_step_lookup_and_scoring():
    """ticker map → CIK (zfill 10) → submissions; flags + score from first 20 forms."""
    with patch(
        "requests.get", side_effect=[_resp(_TICKERS_WIT), _resp(_FILINGS_WIT)]
    ) as get:
        out = insider_mod.fetch_sec_edgar("WIT.NS", company_name="Infosys")

    assert get.call_count == 2
    assert (
        get.call_args_list[0].args[0]
        == "https://www.sec.gov/files/company_tickers.json"
    )
    assert (
        get.call_args_list[1].args[0]
        == "https://data.sec.gov/submissions/CIK0001163385.json"
    )
    headers = {"User-Agent": "StockScanner/1.0 contact@example.com"}
    assert get.call_args_list[0].kwargs == {"headers": headers, "timeout": 10}

    assert out is not None and out.cached is False
    assert out.ticker == "WIT.NS" and out.cik == "0001163385"
    assert out.has_10k and out.has_10q and out.has_8k
    assert out.insider_filings == 4
    assert out.filing_score == 0.7  # 0.3 + 0.2 + 0.1 (8-K) + 0.1 (4 Form 4s)
    assert len(out.recent_filings) == 7
    assert out.recent_filings[0] == insider_mod.SECFiling(
        form_type="10-K",
        filed_date="2026-01-01",
        description="Annual Report",
        url="https://www.sec.gov/Archives/edgar/data/1163385/"
        "000116338526000001/Annual Report.htm",
    )


def test_sec_edgar_tickers_payload_as_list_and_empty_filings():
    """Tickers endpoint may be a list; an empty recent block zeroes everything."""
    tickers = [{"cik_str": 1163385, "ticker": "WIT"}]
    filings = {"filings": {"recent": {}}}
    with patch("requests.get", side_effect=[_resp(tickers), _resp(filings)]):
        out = insider_mod.fetch_sec_edgar("WIT")
    assert out is not None
    assert out.cik == "0001163385"
    assert out.recent_filings == []
    assert not (out.has_10k or out.has_10q or out.has_8k)
    assert out.insider_filings == 0 and out.filing_score == 0.0


def test_sec_edgar_no_cik_match_returns_none():
    with patch("requests.get", return_value=_resp(_TICKERS_WIT)) as get:
        assert insider_mod.fetch_sec_edgar("NOPE") is None
    assert get.call_count == 1  # never reaches the submissions request


def test_sec_edgar_ignores_forms_past_twenty():
    """Loop caps at 20 forms (flags/score) while recent_filings keeps only 10."""
    filings = {
        "filings": {
            "recent": {
                "form": ["4"] * 20 + ["10-K"] * 5,
                "filingDate": ["2026-01-01"] * 25,
                "primaryDocDescription": ["Form 4"] * 25,
                "accessionNumber": [f"0001163385-26-{i:06d}" for i in range(25)],
            }
        }
    }
    with patch("requests.get", side_effect=[_resp(_TICKERS_WIT), _resp(filings)]):
        out = insider_mod.fetch_sec_edgar("WIT")
    assert out is not None
    assert len(out.recent_filings) == 10
    assert out.insider_filings == 20
    assert out.has_10k is False  # 10-K sits at index 20, past the window
    assert out.filing_score == 0.1  # only the Form-4 bonus applies


def test_sec_edgar_short_parallel_arrays_default_to_empty():
    """Dates/descriptions/accessions shorter than forms → "" per filing."""
    filings = {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K"],
                "filingDate": ["2026-01-01"],
                "primaryDocDescription": [],
                "accessionNumber": ["0001163385-26-000001"],
            }
        }
    }
    with patch("requests.get", side_effect=[_resp(_TICKERS_WIT), _resp(filings)]):
        out = insider_mod.fetch_sec_edgar("WIT")
    assert out is not None and len(out.recent_filings) == 2
    first, second = out.recent_filings
    assert first.filed_date == "2026-01-01" and first.description == ""
    assert first.url == (
        "https://www.sec.gov/Archives/edgar/data/1163385/000116338526000001/.htm"
    )
    assert second.filed_date == "" and second.description == ""
    assert second.url == "https://www.sec.gov/Archives/edgar/data/1163385//.htm"
    assert out.has_10k and out.has_8k
    assert out.filing_score == 0.4  # 0.3 (10-K) + 0.1 (8-K)


def test_sec_edgar_first_request_error_returns_none():
    with patch("requests.get", side_effect=requests.HTTPError("403")):
        assert insider_mod.fetch_sec_edgar("WIT") is None


def test_sec_edgar_second_request_error_returns_none():
    with patch(
        "requests.get", side_effect=[_resp(_TICKERS_WIT), _err("503")]
    ) as get:
        assert insider_mod.fetch_sec_edgar("WIT") is None
    assert get.call_count == 2


def test_sec_edgar_second_call_served_from_cache():
    with patch(
        "requests.get", side_effect=[_resp(_TICKERS_WIT), _resp(_FILINGS_WIT)]
    ) as get:
        first = insider_mod.fetch_sec_edgar("WIT")
        second = insider_mod.fetch_sec_edgar("WIT")
    assert first is not None and second is not None
    assert first.cached is False and second.cached is True
    assert get.call_count == 2  # lookup + submissions only happen once
    # cached rows rehydrate as plain dicts (the dataclass does not coerce them)
    assert second.recent_filings[0]["form_type"] == "10-K"
    assert second.filing_score == 0.7


# ── Unified fetcher ─────────────────────────────────────────────────────────


def _aletheia(score):
    return insider_mod.AletheiaInsider(
        ticker="T",
        net_insider_activity=0.0,
        total_buys=0,
        total_sells=0,
        buy_value=0.0,
        sell_value=0.0,
        insider_score=score,
    )


def _congress(score):
    return insider_mod.CongressInvestsData(ticker="T", congressional_score=score)


def _sec(score):
    return insider_mod.SECEdgData(ticker="T", cik="0000000001", filing_score=score)


def test_insider_data_weighted_average_of_three(monkeypatch):
    """Weights are aletheia 3, congress 1, sec 1 → (0.5*3 + 1.0 + 0.25) / 5."""
    ale, con, sec = _aletheia(0.5), _congress(1.0), _sec(0.25)
    monkeypatch.setattr(insider_mod, "fetch_aletheia_insider", lambda *a: ale)
    monkeypatch.setattr(insider_mod, "fetch_congress_invests", lambda *a: con)
    monkeypatch.setattr(insider_mod, "fetch_sec_edgar", lambda *a: sec)
    out = insider_mod.fetch_insider_data("T")
    assert out["insider_score"] == 0.55
    assert out["source"] == "aletheia+congress+sec_edgar"
    assert out["aletheia"] is ale
    assert out["congress"] is con
    assert out["sec_edgar"] is sec


def test_insider_data_partial_sources_reweight(monkeypatch):
    """Missing congress drops its weight: (0.5*3 + 0.25) / 4."""
    ale, sec = _aletheia(0.5), _sec(0.25)
    monkeypatch.setattr(insider_mod, "fetch_aletheia_insider", lambda *a: ale)
    monkeypatch.setattr(insider_mod, "fetch_congress_invests", lambda *a: None)
    monkeypatch.setattr(insider_mod, "fetch_sec_edgar", lambda *a: sec)
    out = insider_mod.fetch_insider_data("T")
    assert out["insider_score"] == 0.438
    assert out["source"] == "aletheia+sec_edgar"
    assert out["congress"] is None


def test_insider_data_congress_only_keeps_its_score(monkeypatch):
    con = _congress(1.0)
    monkeypatch.setattr(insider_mod, "fetch_aletheia_insider", lambda *a: None)
    monkeypatch.setattr(insider_mod, "fetch_congress_invests", lambda *a: con)
    monkeypatch.setattr(insider_mod, "fetch_sec_edgar", lambda *a: None)
    out = insider_mod.fetch_insider_data("T")
    assert out["insider_score"] == 1.0
    assert out["source"] == "congress"
    assert out["aletheia"] is None and out["sec_edgar"] is None


def test_insider_data_all_sources_down(monkeypatch):
    for name in ("fetch_aletheia_insider", "fetch_congress_invests", "fetch_sec_edgar"):
        monkeypatch.setattr(insider_mod, name, lambda *a: None)
    out = insider_mod.fetch_insider_data("T")
    assert out == {
        "insider_score": 0.0,
        "aletheia": None,
        "congress": None,
        "sec_edgar": None,
        "source": "none",
    }


def test_insider_data_forwards_keys_and_company(monkeypatch):
    calls = {}

    def _spy(key):
        def _fetch(*args):
            calls[key] = args

        return _fetch

    monkeypatch.setattr(insider_mod, "fetch_aletheia_insider", _spy("aletheia"))
    monkeypatch.setattr(insider_mod, "fetch_congress_invests", _spy("congress"))
    monkeypatch.setattr(insider_mod, "fetch_sec_edgar", _spy("sec"))
    out = insider_mod.fetch_insider_data("T", "ak", "ck", "Co")
    assert out["source"] == "none"
    assert calls == {
        "aletheia": ("T", "ak"),
        "congress": ("T", "ck"),
        "sec": ("T", "Co"),
    }
