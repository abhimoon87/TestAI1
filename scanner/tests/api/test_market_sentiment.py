"""Tests for market_sentiment._keyword_sentiment — pure keyword scoring."""

from scanner.api.market_sentiment import _keyword_sentiment


class TestKeywordSentiment:
    def test_empty_is_zero(self):
        assert _keyword_sentiment("") == 0.0

    def test_positive_headline(self):
        assert _keyword_sentiment("Stocks surge on strong demand") > 0

    def test_negative_headline(self):
        assert _keyword_sentiment("Shares crash after results") < 0

    def test_neutral_when_no_keywords(self):
        assert _keyword_sentiment("The company scheduled a meeting") == 0.0

    def test_balanced_is_zero(self):
        # one positive ("surge") + one negative ("crash") → net zero
        assert _keyword_sentiment("surge then crash") == 0.0
