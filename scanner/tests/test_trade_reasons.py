"""Tests for scanner.trade_reasons.build_trade_reasons edge cases."""

import math

from scanner.trade_reasons import build_trade_reasons


def test_empty_row_returns_empty():
    assert build_trade_reasons({}) == []


def test_non_dict_row_returns_empty():
    assert build_trade_reasons(None) == []
    assert build_trade_reasons("junk") == []


def test_string_adx_and_rsi_do_not_raise():
    row = {"adx_val": "strong", "rsi_val": "high", "trend": "up",
           "momentum": "fast"}
    assert build_trade_reasons(row) == []


def test_nan_inputs_do_not_raise():
    row = {"adx_val": float("nan"), "rsi_val": float("nan"),
           "trend": float("nan"), "momentum": float("nan"),
           "macd": float("nan"), "volume": float("nan"),
           "rel_str": float("nan"), "fundamentals": float("nan"),
           "volatility": float("nan")}
    assert build_trade_reasons(row) == []


def test_string_poc_does_not_raise():
    row = {"above_poc": True, "vp_poc": "high"}
    assert build_trade_reasons(row) == []


def test_max_reasons_edges():
    row = {"entry_signal": True, "crossover_bars_ago": 1,
           "close_above_both_ma": True, "momentum": 12,
           "volume": 9, "rel_str": 9, "fundamentals": 15}
    assert build_trade_reasons(row, max_reasons=0) == []
    assert build_trade_reasons(row, max_reasons=-3) == []
    assert len(build_trade_reasons(row, max_reasons=2)) == 2
    assert build_trade_reasons(row, max_reasons="bad") != []


def test_overbought_risk_flag():
    row = {"rsi_val": 80}
    reasons = build_trade_reasons(row)
    assert any(r.startswith("Risk: RSI") for r in reasons)


def test_fii_string_net_does_not_raise():
    row = {"_fii_is_buying": True, "_fii_net": "lots"}
    reasons = build_trade_reasons(row)
    assert reasons == ["FII net buying signal"]


def test_math_inf_trend_formats():
    row = {"trend": math.inf}
    reasons = build_trade_reasons(row)
    assert any("trend score" in r for r in reasons)
