"""Tests for scanner.api.channels — subprocess channel primitive plus the
twitter-cli / Jina Reader / Exa wrappers. All external calls are mocked
except the primitive's own failure paths (missing tool, nonzero exit,
timeout), which run a real child process on purpose."""

import json
import sys

import pytest

from scanner.api import channels
from scanner.api.channels import (
    exa_search,
    fetch_article,
    run_channel,
    twitter_search,
)
from scanner.api.social_sentiment import fetch_twitter_sentiment


@pytest.fixture(autouse=True)
def _no_twitter_api_key(monkeypatch):
    monkeypatch.delenv("TWITTER_API_KEY", raising=False)


class TestRunChannel:
    def test_missing_tool_returns_none(self):
        assert run_channel(["no_such_tool_xyz_123"]) is None

    def test_nonzero_exit_returns_none(self):
        assert run_channel([sys.executable, "-c", "import sys; sys.exit(3)"]) is None

    def test_timeout_returns_none(self):
        out = run_channel(
            [sys.executable, "-c", "import time; time.sleep(30)"], timeout=1
        )
        assert out is None

    def test_success_returns_stdout(self):
        out = run_channel([sys.executable, "-c", "print('hello')"])
        assert out is not None
        assert out.strip() == "hello"


class TestTwitterSearch:
    def test_requires_cookies(self):
        assert twitter_search("q", cookies=None) is None
        assert twitter_search("q", cookies=("", "ct0")) is None

    @pytest.mark.parametrize(
        "payload",
        ['[{"text": "a"}]', '{"tweets": [{"text": "a"}]}', '{"data": [{"text": "a"}]}'],
    )
    def test_parses_envelopes(self, monkeypatch, payload):
        monkeypatch.setattr(channels, "run_channel", lambda *a, **k: payload)
        assert twitter_search("q", cookies=("t", "c")) == [{"text": "a"}]

    @pytest.mark.parametrize("payload", ["not json", '{"other": 1}'])
    def test_bad_payload_returns_none(self, monkeypatch, payload):
        monkeypatch.setattr(channels, "run_channel", lambda *a, **k: payload)
        assert twitter_search("q", cookies=("t", "c")) is None

    def test_tool_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(channels, "run_channel", lambda *a, **k: None)
        assert twitter_search("q", cookies=("t", "c")) is None


class TestFetchArticle:
    def test_rejects_non_http_urls(self):
        assert fetch_article("") is None
        assert fetch_article("ftp://example.com/x") is None

    def test_fetches_and_caches(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            channels, "run_channel", lambda *a, **k: calls.append(a) or "article body"
        )
        assert fetch_article("https://example.com/jina-cache-test") == "article body"
        assert fetch_article("https://example.com/jina-cache-test") == "article body"
        assert len(calls) == 1


class TestExaSearch:
    def test_parses_results_envelope(self, monkeypatch):
        payload = json.dumps(
            {"results": [{"title": "T", "url": "https://u", "text": "s"}]}
        )
        monkeypatch.setattr(channels, "run_channel", lambda *a, **k: payload)
        assert exa_search("exa-query-results") == [
            {"title": "T", "url": "https://u", "snippet": "s"}
        ]

    def test_parses_live_text_blocks(self, monkeypatch):
        # Real --output json shape (verified against mcp.exa.ai): MCP content
        # blocks whose text is Title:/URL:/... sections split by '---' lines.
        text = (
            "Title: Press Release\n"
            "URL: https://example.com/pr\n"
            "Published: N/A\n"
            "Author: A\n"
            "Highlights:\n"
            "Profit jumped 10%.\n"
            "...\n"
            "\n---\n\n"
            "Title: Q2 update\n"
            "URL: https://example.com/q2\n"
            "Published: N/A\n"
            "Author: N/A\n"
            "Highlights:\n"
            "NII up 4.8%.\n"
        )
        payload = json.dumps({"content": [{"type": "text", "text": text}]})
        monkeypatch.setattr(channels, "run_channel", lambda *a, **k: payload)
        out = exa_search("exa-query-live-text")
        assert out is not None
        assert len(out) == 2
        assert out[0] == {
            "title": "Press Release",
            "url": "https://example.com/pr",
            "snippet": "Profit jumped 10%.",
        }
        assert out[1]["title"] == "Q2 update"
        assert out[1]["snippet"] == "NII up 4.8%."

    def test_parses_mcp_content_envelope(self, monkeypatch):
        inner = json.dumps([{"title": "T2", "url": "https://u2", "description": "d"}])
        payload = json.dumps({"content": [{"type": "text", "text": inner}]})
        monkeypatch.setattr(channels, "run_channel", lambda *a, **k: payload)
        assert exa_search("exa-query-mcp") == [
            {"title": "T2", "url": "https://u2", "snippet": "d"}
        ]

    def test_tool_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(channels, "run_channel", lambda *a, **k: None)
        assert exa_search("exa-query-fail") is None

    def test_caches_successful_results(self, monkeypatch):
        payload = json.dumps([{"title": "T", "url": "https://u", "text": "s"}])
        calls = []
        monkeypatch.setattr(
            channels, "run_channel", lambda *a, **k: calls.append(a) or payload
        )
        exa_search("exa-query-cache")
        exa_search("exa-query-cache")
        assert len(calls) == 1


class TestTwitterSentimentFallback:
    def test_cli_fallback_scores_tweets(self, monkeypatch):
        monkeypatch.setattr(
            channels,
            "twitter_search",
            lambda *a, **k: [
                {
                    "text": "great results, buying more",
                    "retweet_count": 4,
                    "favorite_count": 6,
                },
                {
                    "text": "weak guidance, trimming",
                    "retweet_count": 1,
                    "favorite_count": 0,
                },
            ],
        )
        res = fetch_twitter_sentiment("ZZFALLBACK1", cookies=("t", "c"))
        assert res is not None
        assert res.mention_count == 2
        assert res.avg_retweets == 2.5
        assert res.avg_likes == 3.0
        res2 = fetch_twitter_sentiment("ZZFALLBACK1", cookies=("t", "c"))
        assert res2 is not None and res2.cached

    def test_no_sources_returns_none(self, monkeypatch):
        monkeypatch.setattr(channels, "twitter_search", lambda *a, **k: None)
        assert fetch_twitter_sentiment("ZZFALLBACK2", cookies=None) is None

    def test_api_failure_falls_back_to_cli(self, monkeypatch):
        import scanner.api.social_sentiment as ss

        def _boom(*a, **k):
            raise ss.requests.RequestException("api down")

        monkeypatch.setattr(ss.requests, "get", _boom)
        monkeypatch.setattr(
            channels,
            "twitter_search",
            lambda *a, **k: [
                {"text": "solid quarter", "retweet_count": 2, "favorite_count": 3}
            ],
        )
        res = fetch_twitter_sentiment(
            "ZZFALLBACK3", api_key="test-key", cookies=("t", "c")
        )
        assert res is not None
        assert res.mention_count == 1

    def test_empty_tweet_list_returns_zero_object(self, monkeypatch):
        monkeypatch.setattr(channels, "twitter_search", lambda *a, **k: [])
        res = fetch_twitter_sentiment("ZZFALLBACK4", cookies=("t", "c"))
        assert res is not None
        assert res.mention_count == 0
        assert res.cached is False
