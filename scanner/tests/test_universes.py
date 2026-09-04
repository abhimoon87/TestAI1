"""Unit tests for scanner.universes — stock universe definitions.

Tests verify:
  - All universe lists contain valid ticker strings
  - UNIVERSES map covers all defined lists
  - No empty universes
  - Key stocks appear in expected universes
  - Combined universes include their constituents
"""

import pytest

from scanner.universes import (
    BANK_NIFTY,
    BSE_SENSEX,
    CASH_MARKET,
    FNO_STOCKS,
    NIFTY_50,
    NIFTY_AUTO,
    NIFTY_ENERGY,
    NIFTY_FINANCIAL,
    NIFTY_IT,
    NIFTY_METAL,
    NIFTY_MIDCAP_100,
    NIFTY_NEXT_50,
    NIFTY_PHARMA,
    NIFTY_REALTY,
    NIFTY_SMALLCAP_100,
    SUSPENDED_OR_DELISTED,
    UNIVERSES,
    dead_member_reason,
    strip_dead_members,
)

# ══════════════════════════════════════════════════════════════════════════════
# List Properties
# ══════════════════════════════════════════════════════════════════════════════


class TestUniverseLists:
    """Verify each universe list has valid content."""

    @pytest.mark.parametrize("name,universe", [
        ("NIFTY_50", NIFTY_50),
        ("BANK_NIFTY", BANK_NIFTY),
        ("NIFTY_NEXT_50", NIFTY_NEXT_50),
        ("NIFTY_MIDCAP_100", NIFTY_MIDCAP_100),
        ("NIFTY_SMALLCAP_100", NIFTY_SMALLCAP_100),
        ("FNO_STOCKS", FNO_STOCKS),
        ("BSE_SENSEX", BSE_SENSEX),
        ("NIFTY_IT", NIFTY_IT),
        ("NIFTY_PHARMA", NIFTY_PHARMA),
        ("NIFTY_AUTO", NIFTY_AUTO),
        ("NIFTY_METAL", NIFTY_METAL),
        ("NIFTY_REALTY", NIFTY_REALTY),
        ("NIFTY_ENERGY", NIFTY_ENERGY),
        ("NIFTY_FINANCIAL", NIFTY_FINANCIAL),
    ])
    def test_not_empty(self, name, universe):
        assert len(universe) > 0, f"{name} is empty"

    @pytest.mark.parametrize("name,universe", [
        ("NIFTY_50", NIFTY_50),
        ("BANK_NIFTY", BANK_NIFTY),
        ("NIFTY_NEXT_50", NIFTY_NEXT_50),
        ("NIFTY_MIDCAP_100", NIFTY_MIDCAP_100),
        ("NIFTY_SMALLCAP_100", NIFTY_SMALLCAP_100),
        ("FNO_STOCKS", FNO_STOCKS),
        ("BSE_SENSEX", BSE_SENSEX),
    ])
    def test_all_strings(self, name, universe):
        for ticker in universe:
            assert isinstance(ticker, str), f"{name}: {ticker} is not a string"

    @pytest.mark.parametrize("name,universe", [
        ("NIFTY_50", NIFTY_50),
        ("BANK_NIFTY", BANK_NIFTY),
        ("NIFTY_NEXT_50", NIFTY_NEXT_50),
        ("FNO_STOCKS", FNO_STOCKS),
    ])
    def test_no_empty_strings(self, name, universe):
        for ticker in universe:
            assert len(ticker) > 0, f"{name}: contains empty string"

    @pytest.mark.parametrize("name,universe,expected_min", [
        ("NIFTY_50", NIFTY_50, 45),
        ("BANK_NIFTY", BANK_NIFTY, 15),
        ("NIFTY_NEXT_50", NIFTY_NEXT_50, 40),
        ("NIFTY_MIDCAP_100", NIFTY_MIDCAP_100, 50),
        ("NIFTY_SMALLCAP_100", NIFTY_SMALLCAP_100, 40),
        ("FNO_STOCKS", FNO_STOCKS, 50),
        ("BSE_SENSEX", BSE_SENSEX, 25),
        ("NIFTY_IT", NIFTY_IT, 8),
    ])
    def test_expected_sizes(self, name, universe, expected_min):
        assert len(universe) >= expected_min, \
            f"{name}: expected >= {expected_min}, got {len(universe)}"


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSES Map
# ══════════════════════════════════════════════════════════════════════════════


class TestUniversesMap:
    def test_not_empty(self):
        assert len(UNIVERSES) > 0

    def test_all_values_are_lists(self):
        for name, universe in UNIVERSES.items():
            assert isinstance(universe, list), f"{name}: not a list"

    def test_no_empty_universes(self):
        for name, universe in UNIVERSES.items():
            assert len(universe) > 0, f"{name}: empty universe"

    def test_expected_keys_exist(self):
        expected_keys = [
            "NIFTY 50", "BANK NIFTY", "NIFTY NEXT 50",
            "NIFTY MIDCAP 100", "NIFTY SMALLCAP 100",
            "FnO STOCKS", "CASH MARKET",
            "BSE SENSEX", "ALL (Combined)",
        ]
        for key in expected_keys:
            assert key in UNIVERSES, f"Missing key: {key}"

    def test_nifty50_in_map(self):
        assert UNIVERSES["NIFTY 50"] is NIFTY_50

    def test_all_combined_includes_nifty50(self):
        all_stocks = set(UNIVERSES["ALL (Combined)"])
        for ticker in NIFTY_50:
            assert ticker in all_stocks, f"NIFTY_50 stock {ticker} missing from ALL"

    def test_all_combined_includes_banknifty(self):
        all_stocks = set(UNIVERSES["ALL (Combined)"])
        for ticker in BANK_NIFTY:
            assert ticker in all_stocks, f"BANK_NIFTY stock {ticker} missing from ALL"

    def test_cash_market_is_superset(self):
        """CASH MARKET should be a superset of NIFTY_50 + BANK_NIFTY."""
        cash = set(CASH_MARKET)
        for ticker in NIFTY_50:
            assert ticker in cash, f"NIFTY_50 stock {ticker} missing from CASH MARKET"
        for ticker in BANK_NIFTY:
            assert ticker in cash, f"BANK_NIFTY stock {ticker} missing from CASH MARKET"


# ══════════════════════════════════════════════════════════════════════════════
# Key Stock Presence
# ══════════════════════════════════════════════════════════════════════════════


class TestKeyStocks:
    """Verify well-known stocks appear in the right universes."""

    def test_reliance_in_nifty50(self):
        assert "RELIANCE" in NIFTY_50

    def test_wipro_in_nifty50(self):
        assert "WIPRO" in NIFTY_50

    def test_infy_in_nifty50(self):
        assert "INFY" in NIFTY_50

    def test_hdfcbank_in_banknifty(self):
        assert "HDFCBANK" in BANK_NIFTY

    def test_sbin_in_banknifty(self):
        assert "SBIN" in BANK_NIFTY

    def test_tcs_in_it(self):
        assert "TCS" in NIFTY_IT

    def test_sunpharma_in_pharma(self):
        assert "SUNPHARMA" in NIFTY_PHARMA

    def test_maruti_in_auto(self):
        assert "MARUTI" in NIFTY_AUTO

    def test_tatasteel_in_metal(self):
        assert "TATASTEEL" in NIFTY_METAL

    def test_dlf_in_realty(self):
        assert "DLF" in NIFTY_REALTY


# ══════════════════════════════════════════════════════════════════════════════
# Suspended / delisted members — annotated so scans skip re-fetching them
# ══════════════════════════════════════════════════════════════════════════════


class TestDeadMembers:
    """GSPL/TATAMETALI stay in the lists (membership intact) but strip out."""

    def test_annotated_names_are_real_list_members(self):
        assert "GSPL" in FNO_STOCKS
        assert "GSPL" in NIFTY_NEXT_50
        assert "GSPL" in NIFTY_MIDCAP_100
        assert "TATAMETALI" in NIFTY_MIDCAP_100

    def test_headline_universes_are_unaffected(self):
        """NIFTY 50 and BANK NIFTY contain no suspended/delisted names."""
        assert not (set(NIFTY_50) & set(SUSPENDED_OR_DELISTED))
        assert not (set(BANK_NIFTY) & set(SUSPENDED_OR_DELISTED))

    def test_strip_splits_dead_from_active_preserving_order(self):
        tickers = ["RELIANCE", "GSPL", "TCS", "TATAMETALI", "INFY"]
        active, dead = strip_dead_members(tickers)
        assert active == ["RELIANCE", "TCS", "INFY"]
        assert dead == ["GSPL", "TATAMETALI"]

    def test_strip_with_no_dead_is_identity(self):
        tickers = ["RELIANCE", "TCS"]
        active, dead = strip_dead_members(tickers)
        assert active == tickers
        assert dead == []

    def test_dead_member_reason(self):
        assert "suspended" in (dead_member_reason("GSPL") or "")
        assert "delisted" in (dead_member_reason("TATAMETALI") or "")
        assert dead_member_reason("RELIANCE") is None
