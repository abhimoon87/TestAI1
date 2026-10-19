"""Tests for macro_data.detect_market_regime — pure regime classification.

The regime decision is pure arithmetic over dataclasses, so it is tested
directly (no network) — this is the logic that steers score thresholds.
"""

from scanner.api.macro_data import (
    EcondbData,
    EconPulseData,
    FredData,
    YahooMacroData,
    detect_market_regime,
)


class TestDetectMarketRegime:
    def test_no_data_is_neutral(self):
        r = detect_market_regime()
        assert r.regime == "neutral"
        assert r.confidence == 0.5
        assert r.recommended_score_threshold == 50

    def test_recession_on_high_unemployment_and_negative_gdp(self):
        fred = FredData(unemployment_rate=7.5, gdp_growth=-2.0)  # 2 + 3
        r = detect_market_regime(fred=fred)
        assert r.regime == "recession"
        assert r.recommended_score_threshold == 65
        assert r.active_categories  # recession narrows to core categories

    def test_deeply_inverted_curve_contributes_to_recession(self):
        fred = FredData(yield_curve_spread=-0.8, unemployment_rate=7.0)  # 3 + 2
        r = detect_market_regime(fred=fred)
        assert r.regime == "recession"

    def test_yahoo_yield_spread_used_when_fred_missing(self):
        # 10y - 2y = -0.9 → deeply inverted (+3); unemployment +2
        yahoo = YahooMacroData(us_10y_yield=3.0, us_2y_yield=3.9)
        r = detect_market_regime(fred=FredData(unemployment_rate=7.0), yahoo=yahoo)
        assert r.regime == "recession"

    def test_risk_off_on_high_rates_and_vix(self):
        fred = FredData(fed_funds_rate=5.5)  # +2
        yahoo = YahooMacroData(vix=35.0)  # +2
        r = detect_market_regime(fred=fred, yahoo=yahoo)
        assert r.regime == "risk_off"
        assert r.recommended_score_threshold == 55

    def test_risk_on_on_low_rates_vix_and_strong_india_gdp(self):
        fred = FredData(fed_funds_rate=1.0)  # +2
        yahoo = YahooMacroData(vix=12.0)  # +1
        econdb = EcondbData(india_gdp_growth=7.5)  # +1
        r = detect_market_regime(fred=fred, yahoo=yahoo, econdb=econdb)
        assert r.regime == "risk_on"
        assert r.recommended_score_threshold == 45

    def test_oil_falls_back_to_econpulse(self):
        # crude_oil unset on econdb → uses econpulse.oil_wti; >100 is risk_off
        r = detect_market_regime(
            econpulse=EconPulseData(oil_wti=110.0),
            fred=FredData(fed_funds_rate=5.5, unemployment_rate=7.0),  # off2 + rec2
            yahoo=YahooMacroData(vix=31.0),  # off2
        )
        assert r.regime == "risk_off"
