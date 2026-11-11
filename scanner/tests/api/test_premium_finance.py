"""Shariah screening — API-key gate, response mapping, cache, failures."""

from unittest.mock import MagicMock, patch

import pytest

import scanner.api.premium_finance as pf_mod


@pytest.fixture(autouse=True)
def _clear_cache():
    """Module-level TTLCache is in-memory: never leak entries across tests."""
    pf_mod._PREMIUM_CACHE.clear()
    yield
    pf_mod._PREMIUM_CACHE.clear()


def _resp(payload):
    r = MagicMock()
    r.json.return_value = payload
    return r


def test_no_api_key_returns_none_without_request():
    with patch.object(pf_mod.requests, "get") as get:
        assert pf_mod.fetch_shariah_data("RELIANCE") is None
        assert not get.called


def test_maps_response_fields():
    resp = _resp(
        {
            "is_compliant": True,
            "method": "shafii",
            "purification_required": True,
            "zakat_amount": 12.5,
        }
    )
    with patch.object(pf_mod.requests, "get", return_value=resp) as get:
        out = pf_mod.fetch_shariah_data("RELIANCE", api_key="k")

    assert out.ticker == "RELIANCE"
    assert out.is_shariah_compliant is True
    assert out.screening_method == "shafii"
    assert out.purification_required is True
    assert out.zakat_amount == 12.5
    assert out.cached is False
    assert get.call_args.kwargs["timeout"] == 10


def test_defaults_when_fields_missing():
    with patch.object(pf_mod.requests, "get", return_value=_resp({})):
        out = pf_mod.fetch_shariah_data("X", api_key="k")

    assert out.is_shariah_compliant is False
    assert out.screening_method == "aaoifi"
    assert out.purification_required is False
    assert out.zakat_amount is None


def test_second_call_served_from_cache():
    resp = _resp({"is_compliant": True, "method": "aaoifi"})
    with patch.object(pf_mod.requests, "get", return_value=resp) as get:
        first = pf_mod.fetch_shariah_data("TCS", api_key="k")
        second = pf_mod.fetch_shariah_data("TCS", api_key="k")

    assert get.call_count == 1  # only the first call hits the network
    assert first.cached is False
    assert second.cached is True
    assert second.is_shariah_compliant is True
    assert second.screening_method == "aaoifi"


def test_http_error_returns_none():
    resp = MagicMock()
    resp.raise_for_status.side_effect = RuntimeError("500")
    with patch.object(pf_mod.requests, "get", return_value=resp):
        assert pf_mod.fetch_shariah_data("Y", api_key="k") is None


def test_request_exception_returns_none():
    with patch.object(pf_mod.requests, "get", side_effect=OSError("no net")):
        assert pf_mod.fetch_shariah_data("Y", api_key="k") is None


def test_failed_fetch_is_not_cached():
    with patch.object(
        pf_mod.requests, "get", side_effect=OSError("no net")
    ) as get:
        assert pf_mod.fetch_shariah_data("Z", api_key="k") is None
        assert pf_mod.fetch_shariah_data("Z", api_key="k") is None
    assert get.call_count == 2  # retry allowed after a failure
