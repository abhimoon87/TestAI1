from scanner.api.yahoo_symbol import yf_quote_variants


def test_bare_ticker_tries_ns_then_bo():
    assert yf_quote_variants("RELIANCE") == ["RELIANCE.NS", "RELIANCE.BO"]


def test_suffixed_tickers_pass_through():
    assert yf_quote_variants("X.NS") == ["X.NS"]
    assert yf_quote_variants("X.BO") == ["X.BO"]
