"""Static universe fallback — last-resort symbol lists must resolve.

Guards the import path (scanner.shared.universes): a wrong relative import
here only surfaces when live NSE fetches fail, which is exactly when the
fallback matters.
"""

from unittest.mock import MagicMock, patch

import pytest

from scanner.api.symbol_fetcher import (
    _fetch_bse_via_csv_mirror,
    _get_static_fallback,
    _parse_bse_master_csv,
)

_KEYS = (
    "mainboard",
    "fno",
    "nifty50",
    "niftynext50",
    "midcap150",
    "smallcap250",
    "unique",
    "bse_all",
    "all_market",
)


def test_static_fallback_lists_resolve():
    for key in _KEYS:
        assert _get_static_fallback(key), f"{key} fallback empty (import broken?)"


_BSE_CSV_HEADER = "Security Code,Security Id,Security Name,Status,ISIN No"


def _mirror_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp


def test_bse_mirror_keeps_only_active_equities():
    """Full scrip masters carry debt/delisted rows — filter to Active 5xxxxx
    equities (what Yahoo .BO can actually serve) and strip BSE's * markers."""
    rows = [
        _BSE_CSV_HEADER,
        "500325,RELIANCE,RELIANCE INDUSTRIES LTD,Active,INE002A01018",
        "500004,TPAEC,TORRENT POWER AEC LTD,Delisted,INE424A01014",
        "901002,960TML22,Structured note,Active,INE000A01001",
        "500500,INSECTICID*,INSECTICIDES INDIA LTD,Active,INE004A01004",
        "500600,NIFTYETF,NIFTY 50 ETF,Active,INF204KA01J5",
        "500002,ABB,ABB India Limited,Active,INE117A01022",
    ]
    rows += [f"510{i:03d},SYM{i},Company {i},Active,INE00{i:06d}" for i in range(4000)]
    with patch("requests.get", return_value=_mirror_response("\n".join(rows))):
        symbols = _fetch_bse_via_csv_mirror()

    assert len(symbols) >= 4000
    assert "RELIANCE" in symbols and "ABB" in symbols
    assert "INSECTICID" in symbols and "INSECTICID*" not in symbols
    assert "NIFTYETF" not in symbols  # ETF (INF ISIN) — stock-only scope
    assert "TPAEC" not in symbols  # delisted
    assert "960TML22" not in symbols  # non-equity (9xxxxx scrip code)


def test_bse_mirror_rejects_partial_lists():
    """A small list is a degraded source pretending to be the full BSE list."""
    rows = [_BSE_CSV_HEADER] + [
        f"510{i:03d},SYM{i},Company {i},Active,INE00{i:06d}" for i in range(500)
    ]
    with patch("requests.get", return_value=_mirror_response("\n".join(rows))):
        assert _fetch_bse_via_csv_mirror() == []


def test_bse_master_csv_parser_filters_status_and_instrument():
    """Official equity-master CSV -> Active Equity rows only."""
    body = (
        "Security Code,Issuer Name,Security Id,Security Name,Status,"
        "Group,Face Value,ISIN No,Instrument\n"
        "500325,RELIANCE INDUSTRIES LTD,RELIANCE,RELIANCE INDUSTRIES LTD,"
        "Active,A,10,INE002A01018,Equity\n"
        "599999,OLD CO,OLDCO,OLD CO LTD,Delisted,X,10,INE001A01001,Equity\n"
        "588888,SUSP CO,SUSPCO,SUSP CO LTD,Suspended,X,10,INE002A01002,Equity\n"
        "577777,PREF CO,PREFCO,PREF CO LTD,Active,A,10,INE003A01003,PreferenceShares\n"
        "566667,JUNK CO,JUNK,JUNK CO,Active,A,10,INE004A01004,-\n"
        "566668,STAR CO,STAR*,STAR CO,Active,A,10,INE005A01005,Equity\n"
        "555555,ETF ISSUER,NIFTYETF,NIFTY 50 ETF,Active,A,10,INF204KA01J5,Equity\n"
    )
    assert _parse_bse_master_csv(body) == ["RELIANCE", "STAR"]


def test_bse_master_parser_rejects_non_csv_body():
    """Akamai denial / homepage HTML must raise, not silently yield []."""
    with pytest.raises(ValueError, match="unexpected BSE master body"):
        _parse_bse_master_csv("<HTML><HEAD><TITLE>Access Denied</TITLE>")
