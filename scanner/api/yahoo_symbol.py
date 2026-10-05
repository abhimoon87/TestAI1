"""Yahoo Finance symbol variants for mixed NSE/BSE universes."""


def yf_quote_variants(ticker: str) -> list[str]:
    """quoteSummary variants: .NS first, then .BO.

    BSE-only listings never resolve on .NS (HTTP 404); bare tickers get
    one .BO retry after .NS fails. Already-suffixed symbols pass through
    unchanged.
    """
    if ticker.endswith((".NS", ".BO")):
        return [ticker]
    return [f"{ticker}.NS", f"{ticker}.BO"]
