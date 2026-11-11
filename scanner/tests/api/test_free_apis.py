"""Free API fetchers — Frankfurter / CoinGecko / mandi / WSB / Google News.

Every HTTP call is mocked; nothing here touches the network. The shared
TTL cache is cleared around each test so cached hits never leak.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from scanner.api import free_apis


@pytest.fixture(autouse=True)
def _clear_free_api_cache():
    """Empty the free-api TTL cache (memory + kv backstop) around each test."""
    free_apis._FREE_API_CACHE.clear()
    yield
    free_apis._FREE_API_CACHE.clear()


def _resp(*, json_data=None, text=None, status=200, exc=None, json_exc=None):
    """Mock requests response: choose what json / text / raise_for_status do."""
    r = MagicMock()
    r.status_code = status
    if exc is not None:
        r.raise_for_status.side_effect = exc
    if json_exc is not None:
        r.json.side_effect = json_exc
    elif json_data is not None:
        r.json.return_value = json_data
    if text is not None:
        r.text = text
    return r


def _patch_get(*responses):
    """Patch requests.get for free_apis; items are mock responses or exceptions."""
    return patch.object(free_apis.requests, "get", side_effect=list(responses))


# ── Frankfurter — forex ──────────────────────────────────────────────────────


def test_forex_happy_path_maps_rate_history_and_changes():
    latest = _resp(json_data={"rates": {"INR": 84.0}})
    hist = _resp(
        json_data={
            "rates": {
                "2026-11-02": {"INR": 70.0},
                "2026-11-03": {"INR": 72.0},
                "2026-11-04": {"INR": 74.0},
                "2026-11-05": {"INR": 76.0},
                "2026-11-06": {"INR": 80.0},
                "2026-11-09": {"INR": 79.0},
            }
        }
    )
    with _patch_get(latest, hist) as get:
        out = free_apis.fetch_forex_data()
    assert out is not None
    assert out.base_currency == "USD" and out.target_currency == "INR"
    assert out.rate == 84.0
    assert out.cached is False
    # sorted by date; 1d uses the second-to-last row, 1w the first
    assert [r["rate"] for r in out.historical_rates] == [70.0, 72.0, 74.0, 76.0, 80.0, 79.0]
    assert out.change_1d == pytest.approx(5.0)  # (84 - 80) / 80
    assert out.change_1w == pytest.approx(20.0)  # (84 - 70) / 70
    assert get.call_count == 2
    assert get.call_args_list[0].args[0].endswith("latest?from=USD&to=INR")
    assert ".." in get.call_args_list[1].args[0]  # date-range URL


def test_forex_missing_target_rate_returns_none():
    # No rate for the target currency bails before the historical request.
    with _patch_get(_resp(json_data={"rates": {"EUR": 0.9}})) as get:
        assert free_apis.fetch_forex_data() is None
    assert get.call_count == 1


def test_forex_thin_history_yields_zero_changes():
    latest = _resp(json_data={"rates": {"INR": 84.0}})
    hist = _resp(json_data={"rates": {"2026-11-09": {"INR": 79.0}}})
    with _patch_get(latest, hist):
        out = free_apis.fetch_forex_data()
    assert out.historical_rates == [{"date": "2026-11-09", "rate": 79.0}]
    assert out.change_1d == 0.0 and out.change_1w == 0.0


def test_forex_hist_row_without_target_counts_as_zero():
    # Quirk: a row missing the target currency is recorded with rate 0.
    latest = _resp(json_data={"rates": {"INR": 84.0}})
    hist = _resp(
        json_data={"rates": {"2026-11-10": {"INR": 80.0}, "2026-11-11": {}}}
    )
    with _patch_get(latest, hist):
        out = free_apis.fetch_forex_data()
    assert [r["rate"] for r in out.historical_rates] == [80.0, 0]
    assert out.change_1d == pytest.approx(5.0)  # prev row still has a rate


def test_forex_zero_placeholder_prev_rate_returns_none():
    # Quirk: if the row used as "prev" is a 0 placeholder, change math divides
    # by zero and the whole fetch degrades to None.
    latest = _resp(json_data={"rates": {"INR": 84.0}})
    hist = _resp(
        json_data={"rates": {"2026-11-10": {}, "2026-11-11": {"INR": 84.0}}}
    )
    with _patch_get(latest, hist):
        assert free_apis.fetch_forex_data() is None


def test_forex_http_error_returns_none():
    with _patch_get(_resp(exc=requests.HTTPError("500"))):
        assert free_apis.fetch_forex_data() is None


def test_forex_malformed_json_returns_none():
    with _patch_get(_resp(json_exc=ValueError("nope"))):
        assert free_apis.fetch_forex_data() is None


def test_forex_second_call_served_from_cache():
    latest = _resp(json_data={"rates": {"INR": 84.0}})
    hist = _resp(json_data={"rates": {"2026-11-09": {"INR": 79.0}}})
    with _patch_get(latest, hist) as get:
        first = free_apis.fetch_forex_data()
        second = free_apis.fetch_forex_data()
    # 2 requests per fetch: only the first fetch touched the network
    assert get.call_count == 2
    assert first.cached is False and second.cached is True
    assert second.rate == first.rate
    assert second.historical_rates == first.historical_rates
    assert second.change_1d == first.change_1d


# ── CoinGecko — crypto sentiment ─────────────────────────────────────────────


def test_crypto_happy_path_maps_market_data():
    global_resp = _resp(
        json_data={
            "data": {
                "market_cap_percentage": {"btc": 54.567},
                "total_market_cap": {"usd": 2.5e12},
                "total_volume": {"usd": 1.1e11},
            }
        }
    )
    coins = _resp(
        json_data={
            "bitcoin": {"usd": 50000, "usd_24h_change": 1.25, "usd_7d_change": -3.5},
            "ethereum": {"usd": 3000, "usd_24h_change": 0.75},
        }
    )
    fng = _resp(json_data={"data": [{"value": "72", "value_classification": "Greed"}]})
    with _patch_get(global_resp, coins, fng) as get:
        out = free_apis.fetch_crypto_sentiment()
    assert out is not None
    assert out.btc_price == 50000
    assert out.btc_change_24h == 1.25
    assert out.btc_change_7d == -3.5
    assert out.eth_price == 3000
    assert out.eth_change_24h == 0.75
    assert out.total_market_cap == 2.5e12
    assert out.total_volume_24h == 1.1e11
    assert out.btc_dominance == 54.57  # rounded to 2dp
    assert out.fear_greed_index == 72.0  # arrives as a string, cast to float
    assert out.fear_greed_label == "Greed"
    assert out.cached is False
    assert get.call_count == 3
    assert get.call_args_list[1].kwargs["params"]["ids"] == "bitcoin,ethereum"
    assert get.call_args_list[2].kwargs["timeout"] == 5  # F&G has a shorter timeout


def test_crypto_defaults_when_fields_missing():
    with _patch_get(_resp(json_data={}), _resp(json_data={}), _resp(json_data={"data": [{}]})):
        out = free_apis.fetch_crypto_sentiment()
    assert out.btc_price == 0 and out.btc_change_24h == 0 and out.btc_change_7d == 0
    assert out.eth_price == 0 and out.eth_change_24h == 0
    assert out.total_market_cap == 0 and out.total_volume_24h == 0
    assert out.btc_dominance == 0
    assert out.fear_greed_index == 50.0  # inner default
    assert out.fear_greed_label == "Neutral"


def test_crypto_fear_greed_is_optional():
    """F&G outage or an empty payload leaves the index None (inner try/except)."""
    for fng in (_resp(exc=requests.HTTPError("429")), _resp(json_data={"data": []})):
        free_apis._FREE_API_CACHE.clear()  # each scenario must reach the network
        global_resp = _resp(json_data={"data": {"market_cap_percentage": {"btc": 50.0}}})
        coins = _resp(json_data={"bitcoin": {"usd": 1.0}})
        with _patch_get(global_resp, coins, fng):
            out = free_apis.fetch_crypto_sentiment()
        assert out is not None
        assert out.fear_greed_index is None
        assert out.fear_greed_label is None
        assert out.btc_price == 1.0


def test_crypto_http_error_returns_none():
    with _patch_get(_resp(exc=requests.HTTPError("500"))):
        assert free_apis.fetch_crypto_sentiment() is None


def test_crypto_second_call_served_from_cache():
    global_resp = _resp(json_data={"data": {"market_cap_percentage": {"btc": 50.0}}})
    coins = _resp(json_data={"bitcoin": {"usd": 1.0}})
    fng = _resp(json_data={"data": [{"value": "25", "value_classification": "Fear"}]})
    with _patch_get(global_resp, coins, fng) as get:
        first = free_apis.fetch_crypto_sentiment()
        second = free_apis.fetch_crypto_sentiment()
    # 3 requests per fetch: only the first fetch touched the network
    assert get.call_count == 3
    assert first.cached is False and second.cached is True
    assert second.btc_price == first.btc_price
    assert second.fear_greed_index == first.fear_greed_index


# ── Indian mandi prices ──────────────────────────────────────────────────────


def test_mandi_maps_records_with_defaults():
    records = [
        {
            "commodity": "Wheat",
            "market": "Azadpur",
            "state": "Delhi",
            "min_price": "2000",
            "max_price": "2400",
            "modal_price": "2200",
            "unit": "Quintal",
            "date": "09-11-2026",
        },
        {
            "commodity": "Cotton",
            "market": "Kota",
            "state": "Rajasthan",
            "min_price": None,
            "max_price": None,
            "modal_price": None,
        },
    ]
    with _patch_get(_resp(json_data={"records": records})) as get:
        out = free_apis.fetch_mandi_prices(commodity="Wheat", state="Delhi")
    assert out is not None
    assert [p.commodity for p in out] == ["Wheat", "Cotton"]
    first, second = out
    assert first.price_min == 2000.0 and first.price_modal == 2200.0
    assert first.unit == "Quintal" and first.date == "09-11-2026"
    # None numerics fall back to 0; missing strings to their defaults
    assert second.price_min == 0.0 and second.price_max == 0.0 and second.price_modal == 0.0
    assert second.unit == "Quintal" and second.date == ""
    params = get.call_args.kwargs["params"]
    assert params["filters[commodity]"] == "Wheat"
    assert params["filters[state]"] == "Delhi"
    assert params["limit"] == 20
    assert "api.data.gov.in" in get.call_args.args[0]


def test_mandi_empty_records_returns_none_and_omits_filters():
    with _patch_get(_resp(json_data={"records": []})) as get:
        assert free_apis.fetch_mandi_prices() is None
    params = get.call_args.kwargs["params"]
    assert "filters[commodity]" not in params
    assert "filters[state]" not in params


def test_mandi_caps_at_twenty_records():
    good = {
        "commodity": "Wheat",
        "market": "M",
        "state": "S",
        "min_price": "100",
        "max_price": "200",
        "modal_price": "150",
    }
    records = [dict(good, commodity=f"W{i}") for i in range(25)]
    with _patch_get(_resp(json_data={"records": records})):
        out = free_apis.fetch_mandi_prices()
    assert out is not None and len(out) == 20
    assert out[-1].commodity == "W19"


def test_mandi_skips_records_with_bad_numbers():
    bad = {"commodity": "Bad", "min_price": "not-a-number"}
    good = {"commodity": "Wheat", "min_price": "100"}
    with _patch_get(_resp(json_data={"records": [bad, good]})):
        out = free_apis.fetch_mandi_prices()
    assert out is not None
    assert [p.commodity for p in out] == ["Wheat"]
    assert out[0].price_max == 0.0  # missing max/modal default to 0


def test_mandi_http_error_returns_none():
    with _patch_get(_resp(exc=requests.HTTPError("500"))):
        assert free_apis.fetch_mandi_prices() is None


def test_mandi_second_call_served_from_cache():
    records = [{"commodity": "Wheat", "min_price": "100"}]
    with _patch_get(_resp(json_data={"records": records})) as get:
        first = free_apis.fetch_mandi_prices(commodity="Wheat")
        second = free_apis.fetch_mandi_prices(commodity="Wheat")
    assert get.call_count == 1
    assert [p.commodity for p in second] == [p.commodity for p in first]
    # Quirk: the mandi cache dict carries no cached flag, so it stays False.
    assert second[0].cached is False


# ── WallstreetBets sentiment ─────────────────────────────────────────────────


def test_wsb_happy_path_sentiment_and_truncation():
    ups_list = [500, 200, 300, -11, -50, 0, 0, 0, 0, 0, 999, 999]
    children = [
        {
            "data": {
                "title": "T" * 150 if i == 0 else f"Post {i}",
                "ups": ups,
                "permalink": f"/r/wallstreetbets/comments/{i}/x/",
            }
        }
        for i, ups in enumerate(ups_list)
    ]
    payload = {"data": {"children": children}}
    with _patch_get(_resp(json_data=payload)) as get:
        out = free_apis.fetch_wallstreetbets_sentiment("RELIANCE.NS")
    assert out is not None
    assert out.ticker == "RELIANCE.NS"  # only the search query strips .NS/.BO
    assert out.mention_count == 12  # counts every child
    assert len(out.top_posts) == 10  # but only the first 10 are scored/listed
    assert out.sentiment_score == pytest.approx(0.01)  # (3*0.1 - 2*0.1) / 10
    assert out.top_posts[0]["title"] == "T" * 120  # titles truncated to 120
    assert out.top_posts[0]["upvotes"] == 500
    assert out.top_posts[0]["url"] == "https://reddit.com/r/wallstreetbets/comments/0/x/"
    kwargs = get.call_args.kwargs
    assert kwargs["params"]["q"] == "RELIANCE"
    assert kwargs["params"] == {
        "q": "RELIANCE",
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",
        "limit": 25,
    }
    assert kwargs["timeout"] == 15
    assert kwargs["headers"]["User-Agent"].startswith("Mozilla")


def test_wsb_upvote_thresholds_are_strict():
    """100 is not >100 and -10 is not <-10, so both score 0."""
    for ups in (100, -10):
        child = {"data": {"title": "t", "ups": ups, "permalink": "/p"}}
        with _patch_get(_resp(json_data={"data": {"children": [child]}})):
            out = free_apis.fetch_wallstreetbets_sentiment(f"T{ups}")
        assert out.sentiment_score == 0.0
        assert out.mention_count == 1


def test_wsb_reddit_403_returns_none():
    blocked = _resp(status=403)
    with _patch_get(blocked):
        assert free_apis.fetch_wallstreetbets_sentiment("RELIANCE") is None
    blocked.raise_for_status.assert_not_called()  # 403 handled before raise


def test_wsb_no_mentions_returns_zero_sentiment():
    """Missing/empty children short-circuits to a zero object, never cached."""
    for payload in ({}, {"data": {"children": []}}):
        # two responses: the zero-mention result is never cached, so the
        # second call hits the network again
        with _patch_get(_resp(json_data=payload), _resp(json_data=payload)) as get:
            out = free_apis.fetch_wallstreetbets_sentiment("EMPTY")
            again = free_apis.fetch_wallstreetbets_sentiment("EMPTY")
        assert get.call_count == 2
        for res in (out, again):
            assert res.ticker == "EMPTY"
            assert res.mention_count == 0
            assert res.sentiment_score == 0.0
            assert res.top_posts == []
            assert res.cached is False


def test_wsb_http_error_returns_none():
    with _patch_get(requests.ConnectionError("boom")):
        assert free_apis.fetch_wallstreetbets_sentiment("TCS") is None


def test_wsb_second_call_served_from_cache():
    child = {"data": {"title": "t", "ups": 500, "permalink": "/p"}}
    payload = {"data": {"children": [child]}}
    with _patch_get(_resp(json_data=payload)) as get:
        first = free_apis.fetch_wallstreetbets_sentiment("TCS")
        second = free_apis.fetch_wallstreetbets_sentiment("TCS")
    assert get.call_count == 1
    assert first.cached is False and second.cached is True
    assert second.mention_count == first.mention_count
    assert second.sentiment_score == first.sentiment_score


# ── Noozra / Google News RSS ─────────────────────────────────────────────────


def _rss(*items):
    return '<rss version="2.0"><channel>' + "".join(items) + "</channel></rss>"


def _item(title, link="", source="", pub=""):
    """One RSS <item>; empty strings omit the optional element."""
    src = f"<source>{source}</source>" if source else ""
    date = f"<pubDate>{pub}</pubDate>" if pub else ""
    return f"<item><title>{title}</title><link>{link}</link>{src}{date}</item>"


def test_noozra_maps_items_with_defaults():
    xml = _rss(
        _item(
            "Wheat prices rise",
            link="http://a/1",
            source="Mint",
            pub="Mon, 09 Nov 2026 06:00:00 GMT",
        ),
        _item("Markets edge higher", link="http://a/2", pub="Sun, 08 Nov 2026"),
    )
    with _patch_get(_resp(text=xml)) as get:
        out = free_apis.fetch_noozra_news(query="NIFTY")
    assert out is not None and len(out) == 2
    first, second = out
    assert first.title == "Wheat prices rise"
    assert first.url == "http://a/1"
    assert first.source == "Mint"
    assert first.published == "Mon, 09 Nov 2026 06:00:00 GMT"
    assert second.source == "Google News"  # <source> missing
    assert second.published == "Sun, 08 Nov 2026"
    assert "NIFTY+when:7d" in get.call_args.args[0]
    assert get.call_args.kwargs["timeout"] == 15


def test_noozra_default_query_used_when_none():
    xml = _rss(_item("Only item", link="http://a"))
    with _patch_get(_resp(text=xml)) as get:
        out = free_apis.fetch_noozra_news()
    assert out is not None and len(out) == 1
    assert "Indian stock market+when:7d" in get.call_args.args[0]
    assert out[0].source == "Google News"  # missing <source> and <pubDate>
    assert out[0].published == ""


def test_noozra_respects_max_items_and_truncates_titles():
    xml = _rss(*[_item("H" * 250, link=f"http://a/{i}") for i in range(4)])
    with _patch_get(_resp(text=xml)):
        out = free_apis.fetch_noozra_news(query="BIG", max_items=3)
    assert out is not None and len(out) == 3
    assert all(len(n.title) == 200 for n in out)


def test_noozra_empty_feed_returns_none():
    with _patch_get(_resp(text=_rss())):
        assert free_apis.fetch_noozra_news(query="NONEWS") is None


def test_noozra_malformed_xml_returns_none():
    bad = "<rss><channel><item></channel></rss>"  # unclosed <item>
    with _patch_get(_resp(text=bad)):
        assert free_apis.fetch_noozra_news(query="BROKEN") is None


def test_noozra_http_error_returns_none():
    with _patch_get(_resp(exc=requests.HTTPError("404"))):
        assert free_apis.fetch_noozra_news(query="DOWN") is None


def test_noozra_second_call_served_from_cache():
    xml = _rss(_item("Cached item", link="http://a", source="S"))
    with _patch_get(_resp(text=xml)) as get:
        first = free_apis.fetch_noozra_news(query="CACHED")
        second = free_apis.fetch_noozra_news(query="CACHED")
    assert get.call_count == 1
    assert [n.title for n in second] == [n.title for n in first]
    # Quirk: the news cache dict carries no cached flag, so it stays False.
    assert second[0].cached is False
