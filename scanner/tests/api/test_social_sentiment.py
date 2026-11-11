"""Social sentiment — keyword scoring plus the Reddit/Twitter/unified fetchers.

All offline: ``requests.get`` and the twitter-cli fallback are mocked, and
the module-level social cache is cleared around every test.
"""

import hashlib
from unittest.mock import MagicMock

import pytest

import scanner.api.social_sentiment as ss
from scanner.api import channels, free_apis


@pytest.fixture(autouse=True)
def _clean_social_state(_isolate_db, monkeypatch):
    """Clear the shared social cache; never inherit an ambient API key.

    Depends on the per-test db isolation so the cache's sqlite backstop is
    wiped from the tmp db, not the real scanner.db.
    """
    ss._SOCIAL_CACHE.clear()
    monkeypatch.delenv("TWITTER_API_KEY", raising=False)
    yield
    ss._SOCIAL_CACHE.clear()


# ── _social_sentiment ──────────────────────────────────────────────────────


def test_sentiment_empty_or_keyword_free_text_is_neutral():
    assert ss._social_sentiment("") == 0.0
    assert ss._social_sentiment("nothing interesting here") == 0.0


def test_sentiment_positive_keywords_score_one():
    assert ss._social_sentiment("MOON bullish buy") == 1.0  # case-insensitive


def test_sentiment_negative_keywords_score_minus_one():
    assert ss._social_sentiment("crash dump scam") == -1.0


def test_sentiment_mixed_text_stays_between_bounds():
    assert ss._social_sentiment("buy crash") == 0.0  # one of each → 0
    # Tokens are a set: "buy moon crash" → (2 - 1) / 3
    assert ss._social_sentiment("buy moon crash") == pytest.approx(1 / 3)
    positive = "buy moon rocket rally up strong gain profit bullish beat"
    negative = "crash dump scam bearish sell overvalued"
    assert ss._social_sentiment(positive) == 1.0
    assert ss._social_sentiment(negative) == -1.0


def test_sentiment_counts_emoji_tokens():
    # 🚀 is in SOCIAL_POSITIVE; duplicate emoji collapse to one token.
    assert ss._social_sentiment("🚀🚀 moon") == 1.0
    assert ss._social_sentiment("🚀 crash") == 0.0


# ── _score_tweets ──────────────────────────────────────────────────────────


def test_score_tweets_maps_sorts_and_caches():
    tweets = [
        {"text": "moon rocket", "retweet_count": 10, "favorite_count": 20},
        {"text": "crash scam", "retweet_count": 2, "favorite_count": 0},
        {"text": "hold", "retweet_count": 0, "favorite_count": 5},
    ]
    res = ss._score_tweets("ABC", tweets, "unit-score-cache", 20)
    assert res.mention_count == 3
    assert res.sentiment_score == pytest.approx(0.333)  # (1 - 1 + 1) / 3
    assert res.avg_retweets == 4.0  # 12 / 3
    assert res.avg_likes == 8.3  # 25 / 3, rounded
    assert res.cached is False
    # Sorted by retweets + likes: 30, 5, 2
    assert [t["text"] for t in res.top_tweets] == ["moon rocket", "hold", "crash scam"]
    assert res.top_tweets[0] == {
        "text": "moon rocket",
        "retweets": 10,
        "likes": 20,
        "sentiment": 1.0,
    }
    assert ss._SOCIAL_CACHE.get("unit-score-cache") is not None


def test_score_tweets_empty_list_returns_zero_without_cache():
    res = ss._score_tweets("ABC", [], "unit-empty-cache", 20)
    assert res.mention_count == 0
    assert res.sentiment_score == 0.0
    assert res.avg_retweets == 0
    assert res.avg_likes == 0
    assert res.top_tweets == []
    assert res.cached is False
    assert ss._SOCIAL_CACHE.get("unit-empty-cache") is None


def test_score_tweets_skips_non_dicts_empty_and_non_string_text():
    tweets = ["raw", {"text": ""}, {"other": 1}, {"text": 123}]
    res = ss._score_tweets("ABC", tweets, "unit-skip-cache", 20)
    assert res.mention_count == 1  # only {"text": 123} survives, as "123"
    assert res.sentiment_score == 0.0
    assert res.top_tweets == [
        {"text": "123", "retweets": 0, "likes": 0, "sentiment": 0.0}
    ]


def test_score_tweets_alternate_keys_and_unparsable_counts():
    tweets = [
        {"full_text": "scam fraud", "retweets": 4, "likes": 6},
        {"text": "buy", "retweet_count": "many", "favorite_count": 9},
    ]
    res = ss._score_tweets("ABC", tweets, "unit-alt-cache", 20)
    assert res.mention_count == 2
    assert res.sentiment_score == 0.0  # -1.0 and +1.0 average out
    assert res.avg_retweets == 2.0  # (4 + 0) / 2
    assert res.avg_likes == 3.0  # (6 + 0) / 2 — a bad count zeroes BOTH
    assert res.top_tweets[0]["retweets"] == 4
    assert res.top_tweets[1] == {
        "text": "buy",
        "retweets": 0,
        "likes": 0,
        "sentiment": 1.0,
    }


def test_score_tweets_respects_max_results():
    tweets = [{"text": "buy"}, {"text": "crash"}, {"text": "sell"}]
    res = ss._score_tweets("ABC", tweets, "unit-cap-cache", 1)
    assert res.mention_count == 1  # only the first tweet is scored
    assert res.sentiment_score == 1.0


# ── fetch_reddit_sentiment ─────────────────────────────────────────────────

_BULL = {
    "data": {
        "title": "moon rocket buy breakout",
        "selftext": "",
        "score": 100,
        "permalink": "/r/IndianStreetBets/1a/",
    }
}
_BEAR = {
    "data": {
        "title": "crash dump scam",
        "selftext": "",
        "score": -5,
        "permalink": "/r/stocks/1b/",
    }
}


def _reddit_resp(payload, status=200):
    """requests.Response stand-in for Reddit's search.json."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    return resp


def _patch_reddit_happy(monkeypatch):
    """One bullish post on r/IndianStreetBets, one bearish on r/stocks."""
    calls = []

    def _get(url, **kwargs):
        calls.append((url, kwargs))
        children = [_BULL] if "IndianStreetBets" in url else [_BEAR]
        return _reddit_resp({"data": {"children": children}})

    monkeypatch.setattr(ss.requests, "get", _get)
    return calls


def test_reddit_happy_path_maps_fields_and_requests(monkeypatch):
    calls = _patch_reddit_happy(monkeypatch)
    res = ss.fetch_reddit_sentiment("RELIANCE.NS")
    assert res.cached is False
    assert res.ticker == "RELIANCE.NS"
    assert res.mention_count == 2
    assert res.sentiment_score == 0.0  # +1.0 and -1.0 average out
    assert res.bullish_pct == 0.333  # 1 / (1 + 1 + 1)
    assert res.bearish_pct == 0.333
    assert len(res.top_posts) == 2
    assert set(res.subreddits) == {"IndianStreetBets", "stocks"}
    by_sub = {p["subreddit"]: p for p in res.top_posts}
    assert by_sub["IndianStreetBets"]["sentiment"] == 1.0
    assert by_sub["stocks"]["sentiment"] == -1.0
    assert by_sub["stocks"]["score"] == -5
    assert by_sub["stocks"]["url"] == "https://reddit.com/r/stocks/1b/"
    assert "raw_score" not in by_sub["stocks"]
    # Both default subreddits hit; .NS stripped from the search query.
    assert {url for url, _ in calls} == {
        "https://www.reddit.com/r/IndianStreetBets/search.json",
        "https://www.reddit.com/r/stocks/search.json",
    }
    for _, kwargs in calls:
        assert kwargs["params"]["q"] == "RELIANCE"
        assert kwargs["params"]["limit"] == 25


def test_reddit_second_call_served_from_cache(monkeypatch):
    calls = _patch_reddit_happy(monkeypatch)
    first = ss.fetch_reddit_sentiment("RELIANCE.NS")
    second = ss.fetch_reddit_sentiment("RELIANCE.NS")
    assert len(calls) == 2  # one request per default subreddit, no more
    assert second.cached is True
    assert second.mention_count == first.mention_count == 2


def test_reddit_empty_children_returns_zeros(monkeypatch):
    resp = _reddit_resp({"data": {"children": []}})
    monkeypatch.setattr(ss.requests, "get", MagicMock(return_value=resp))
    res = ss.fetch_reddit_sentiment("EMPTYR")
    assert res.mention_count == 0
    assert res.sentiment_score == 0.0
    assert res.bullish_pct == 0.0
    assert res.bearish_pct == 0.0
    assert res.top_posts == []
    assert res.subreddits == []
    assert res.cached is False
    key = hashlib.md5(b"reddit:EMPTYR", usedforsecurity=False).hexdigest()
    assert ss._SOCIAL_CACHE.get(key) is None  # empty results are not cached


def test_reddit_http_error_returns_zeros(monkeypatch):
    resp = _reddit_resp({"data": {"children": []}})
    resp.raise_for_status.side_effect = ss.requests.HTTPError("500 Server Error")
    monkeypatch.setattr(ss.requests, "get", MagicMock(return_value=resp))
    res = ss.fetch_reddit_sentiment("REDDOWN")
    assert res.mention_count == 0
    assert resp.raise_for_status.call_count == 2  # both subreddits attempted


def test_reddit_rate_limit_returns_zeros_without_raising(monkeypatch):
    resp = _reddit_resp({}, status=403)
    monkeypatch.setattr(ss.requests, "get", MagicMock(return_value=resp))
    res = ss.fetch_reddit_sentiment("REDDENIED")
    assert res.mention_count == 0
    resp.raise_for_status.assert_not_called()  # 429/403 short-circuit
    resp.json.assert_not_called()


def test_reddit_network_exception_returns_zeros(monkeypatch):
    def _get(*a, **k):
        raise ss.requests.ConnectionError("offline")

    monkeypatch.setattr(ss.requests, "get", _get)
    res = ss.fetch_reddit_sentiment("REDOFFLINE")
    assert res is not None
    assert res.mention_count == 0


def test_reddit_missing_post_fields_use_defaults(monkeypatch):
    sparse = {"data": {"selftext": "crash dumping"}}
    resp = _reddit_resp({"data": {"children": [sparse]}})
    monkeypatch.setattr(ss.requests, "get", MagicMock(return_value=resp))
    res = ss.fetch_reddit_sentiment("SPARSE", subreddits=["stocks"])
    assert res.mention_count == 1
    assert res.sentiment_score == -1.0
    assert res.bearish_pct == 0.5  # 1 / (1 + 0 + 1)
    post = res.top_posts[0]
    assert post["title"] == ""
    assert post["score"] == 0
    assert post["url"] == "https://reddit.com"


# ── fetch_twitter_sentiment ────────────────────────────────────────────────


def _twitter_resp(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


def _patch_twitter_search(monkeypatch, tweets):
    """twitter-cli fallback stand-in; returns the mock."""
    search = MagicMock(return_value=tweets)
    monkeypatch.setattr(channels, "twitter_search", search)
    return search


def test_twitter_api_happy_path_maps_fields_and_caches(monkeypatch):
    resp = _twitter_resp(
        {
            "data": [
                {"text": "moon rocket", "retweet_count": 10, "favorite_count": 20},
                {"text": "crash scam", "retweet_count": 2, "favorite_count": 0},
                {"text": "hold", "retweet_count": 0, "favorite_count": 5},
            ]
        }
    )
    get = MagicMock(return_value=resp)
    monkeypatch.setattr(ss.requests, "get", get)
    res = ss.fetch_twitter_sentiment("TCS.NS", api_key="k1")
    assert res.cached is False
    assert res.ticker == "TCS.NS"
    assert res.mention_count == 3
    assert res.sentiment_score == pytest.approx(0.333)
    assert res.avg_retweets == 4.0
    assert res.avg_likes == 8.3
    assert [t["text"] for t in res.top_tweets] == [
        "moon rocket",
        "hold",
        "crash scam",
    ]
    args, kwargs = get.call_args
    assert args[0] == "https://api.getxapi.com/v2/tweets/search"
    assert kwargs["headers"] == {"Authorization": "Bearer k1"}
    assert kwargs["params"] == {
        "query": "$TCS OR #TCS stock",
        "max_results": 20,
        "sort": "recent",
    }
    again = ss.fetch_twitter_sentiment("TCS.NS", api_key="k1")
    assert again.cached is True
    assert get.call_count == 1  # second read served from cache


def test_twitter_api_connect_failure_falls_back_to_cli(monkeypatch):
    def _get(*a, **k):
        raise ss.requests.RequestException("api down")

    monkeypatch.setattr(ss.requests, "get", _get)
    search = _patch_twitter_search(
        monkeypatch,
        [{"text": "solid quarter buy", "retweet_count": 2, "favorite_count": 4}],
    )
    res = ss.fetch_twitter_sentiment("FALLBK", api_key="k1", cookies=("tok", "ct0"))
    search.assert_called_once_with("$FALLBK OR #FALLBK stock", 20, ("tok", "ct0"))
    assert res is not None
    assert res.mention_count == 1
    assert res.sentiment_score == 1.0


def test_twitter_unparseable_body_falls_back_to_cli(monkeypatch):
    resp = MagicMock()
    resp.json.side_effect = ValueError("not json")
    monkeypatch.setattr(ss.requests, "get", MagicMock(return_value=resp))
    search = _patch_twitter_search(monkeypatch, [{"text": "buy"}])
    res = ss.fetch_twitter_sentiment("JSONFAIL", api_key="k1", cookies=("t", "c"))
    search.assert_called_once()
    assert res is not None
    assert res.mention_count == 1


def test_twitter_without_key_skips_http_and_uses_cli(monkeypatch):
    get = MagicMock()
    monkeypatch.setattr(ss.requests, "get", get)
    search = _patch_twitter_search(monkeypatch, [{"text": "buy the breakout"}])
    res = ss.fetch_twitter_sentiment("NOKEY")
    get.assert_not_called()  # no key → API path never attempted
    search.assert_called_once_with("$NOKEY OR #NOKEY stock", 20, None)
    assert res is not None
    assert res.mention_count == 1


def test_twitter_no_working_source_returns_none(monkeypatch):
    def _get(*a, **k):
        raise ss.requests.RequestException("api down")

    monkeypatch.setattr(ss.requests, "get", _get)
    _patch_twitter_search(monkeypatch, None)
    assert ss.fetch_twitter_sentiment("NOTW", api_key="k1", cookies=("t", "c")) is None


def test_twitter_api_empty_data_returns_zero_without_cli(monkeypatch):
    resp = _twitter_resp({"data": []})
    monkeypatch.setattr(ss.requests, "get", MagicMock(return_value=resp))
    search = _patch_twitter_search(monkeypatch, [{"text": "should not run"}])
    res = ss.fetch_twitter_sentiment("APIEMPTY", api_key="k1")
    search.assert_not_called()  # empty data is a result, not a failure
    assert res is not None
    assert res.mention_count == 0
    assert res.avg_retweets == 0
    assert res.cached is False


def test_twitter_api_tweets_envelope_is_supported(monkeypatch):
    resp = _twitter_resp(
        {"tweets": [{"text": "moon", "retweet_count": 1, "favorite_count": 2}]}
    )
    monkeypatch.setattr(ss.requests, "get", MagicMock(return_value=resp))
    res = ss.fetch_twitter_sentiment("TWEETSENV", api_key="k1")
    assert res is not None
    assert res.mention_count == 1
    assert res.sentiment_score == 1.0
    assert res.avg_likes == 2.0


def test_twitter_api_body_without_tweets_returns_zero(monkeypatch):
    resp = _twitter_resp({"meta": {"result_count": 0}})
    monkeypatch.setattr(ss.requests, "get", MagicMock(return_value=resp))
    search = _patch_twitter_search(monkeypatch, [{"text": "should not run"}])
    res = ss.fetch_twitter_sentiment("NODATA", api_key="k1")
    search.assert_not_called()  # no data key → empty, not a failure
    assert res is not None
    assert res.mention_count == 0


# ── fetch_social_sentiment ─────────────────────────────────────────────────


def _reddit_obj(mentions=2, score=0.4):
    return ss.RedditSentiment(
        ticker="ABC",
        mention_count=mentions,
        sentiment_score=score,
        bullish_pct=0.5,
        bearish_pct=0.0,
    )


def _twitter_obj(mentions=1, score=1.0):
    return ss.TwitterSentiment(
        ticker="ABC",
        mention_count=mentions,
        sentiment_score=score,
        avg_retweets=3.0,
        avg_likes=5.0,
    )


def _patch_sources(monkeypatch, reddit, twitter, wsb=None):
    """Replace both fetchers and the WSB add-on; returns their mocks."""
    r_mock = MagicMock(return_value=reddit)
    t_mock = MagicMock(return_value=twitter)
    w_mock = MagicMock(return_value=wsb)
    monkeypatch.setattr(ss, "fetch_reddit_sentiment", r_mock)
    monkeypatch.setattr(ss, "fetch_twitter_sentiment", t_mock)
    monkeypatch.setattr(free_apis, "fetch_wallstreetbets_sentiment", w_mock)
    return r_mock, t_mock, w_mock


def test_social_combines_reddit_and_twitter_weighted(monkeypatch):
    """Reddit counts double: (0.4 * 2 + 1.0 * 1) / 3 == 0.6."""
    reddit = _reddit_obj()
    twitter = _twitter_obj()
    r_mock, t_mock, w_mock = _patch_sources(monkeypatch, reddit, twitter)
    out = ss.fetch_social_sentiment(
        "ABC",
        twitter_api_key="k1",
        subreddits=["stocks"],
        twitter_cookies=("t", "c"),
    )
    r_mock.assert_called_once_with("ABC", ["stocks"])
    t_mock.assert_called_once_with("ABC", "k1", cookies=("t", "c"))
    w_mock.assert_called_once_with("ABC")
    assert out["social_score"] == 0.6
    assert out["mention_count"] == 3
    assert out["source"] == "reddit+twitter"
    assert out["reddit"] is reddit
    assert out["twitter"] is twitter
    assert out["wsb"] is None


def test_social_reddit_only(monkeypatch):
    reddit = _reddit_obj()
    r_mock, _, w_mock = _patch_sources(monkeypatch, reddit, None)
    out = ss.fetch_social_sentiment("ABC")
    r_mock.assert_called_once_with("ABC", None)
    w_mock.assert_called_once_with("ABC")
    assert out["social_score"] == 0.4  # 0.4 * 2 / 2
    assert out["mention_count"] == 2
    assert out["source"] == "reddit"
    assert out["twitter"] is None
    assert out["wsb"] is None


def test_social_twitter_only(monkeypatch):
    twitter = _twitter_obj()
    _, _, _ = _patch_sources(monkeypatch, None, twitter)
    out = ss.fetch_social_sentiment("ABC")
    assert out["social_score"] == 1.0
    assert out["mention_count"] == 1
    assert out["source"] == "twitter"
    assert out["reddit"] is None


def test_social_no_sources_returns_none_dict(monkeypatch):
    _, _, w_mock = _patch_sources(monkeypatch, None, None)
    out = ss.fetch_social_sentiment("ABC")
    assert out == {
        "social_score": 0.0,
        "mention_count": 0,
        "source": "none",
        "reddit": None,
        "twitter": None,
    }
    assert "wsb" not in out  # early return skips the WSB lookup entirely
    w_mock.assert_not_called()


def test_social_zero_mention_reddit_counts_as_no_source(monkeypatch):
    _, _, w_mock = _patch_sources(monkeypatch, _reddit_obj(mentions=0), None)
    out = ss.fetch_social_sentiment("ABC")
    assert out["source"] == "none"
    assert out["reddit"] is None  # no-score path hardcodes None
    w_mock.assert_not_called()


def test_social_wsb_added_with_weight(monkeypatch):
    """WSB joins at 1.5x: (0.4 * 2 + 1.0 * 1.5) / 3.5 == 0.657."""
    wsb = free_apis.WallstreetBetsSentiment(
        ticker="ABC", mention_count=3, sentiment_score=1.0
    )
    _patch_sources(monkeypatch, _reddit_obj(), None, wsb=wsb)
    out = ss.fetch_social_sentiment("ABC")
    assert out["source"] == "reddit+wsb"
    assert out["mention_count"] == 5  # 2 reddit + 3 wsb
    assert out["social_score"] == round((0.4 * 2 + 1.0 * 1.5) / 3.5, 3)
    assert out["wsb"] is wsb


def test_social_wsb_zero_mentions_ignored(monkeypatch):
    wsb = free_apis.WallstreetBetsSentiment(
        ticker="ABC", mention_count=0, sentiment_score=1.0
    )
    _patch_sources(monkeypatch, _reddit_obj(), None, wsb=wsb)
    out = ss.fetch_social_sentiment("ABC")
    assert out["source"] == "reddit"
    assert out["social_score"] == 0.4  # WSB adds no weight
    assert out["mention_count"] == 2
    assert out["wsb"] is wsb


def test_social_wsb_failure_keeps_result(monkeypatch):
    _, _, w_mock = _patch_sources(monkeypatch, _reddit_obj(), None)
    w_mock.side_effect = RuntimeError("wsb down")
    out = ss.fetch_social_sentiment("ABC")
    assert out["source"] == "reddit"
    assert out["social_score"] == 0.4
    assert out["wsb"] is None  # WSB exceptions are swallowed
