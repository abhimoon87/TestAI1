"""Offline tests for the V2 engine (V4 spec): weekly-close arming, the
5-rule confirmation gate, expiry in weeks, weekly-sell + risk exits, VP POC
bins, seeds, commission, and the archive-backed load path."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scanner.backend import backtest_v2
from scanner.backend.backtest_models import Position
from scanner.backend.backtest_position import _close_position
from scanner.backend.backtest_v2 import Pending, V2Engine
from scanner.shared import db

W_BULL = 65  # weekly bar index of the crafted HMA cross-above
W_BEAR = 70  # weekly bar index of the crafted cross-below
# _rise_frame starts on a Monday, so bar i falls on weekday i % 5
# (4 = Friday = the week's last session = the spec evaluation point).
FRI = 284


# ── Helpers ────────────────────────────────────────────────────────────────


def _rise_frame(n=420, flat=200, base=100.0, end=200.0):
    """Flat close then a steady ramp; deterministic green rising bars."""
    dates = pd.bdate_range("2023-01-02", periods=n)  # Monday
    flat = min(flat, n)
    close = np.concatenate([np.full(flat, base), np.linspace(base, end, n - flat)])
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )


def _stock(df, settings=None, ticker="TEST"):
    st = backtest_v2.precompute_stock(ticker, df, {**V2Engine(settings).settings})
    assert st is not None and st.weekly_df is not None, "fixture needs >=60 weeks"
    n_w = len(st.weekly_df)
    assert n_w > W_BULL + 1, f"fixture too short: {n_w} weekly bars"
    return st


def _craft_weekly(stock, bull_week=W_BULL, bear_week=None):
    """Replace weekly trigger MAs with a scripted cross at bull_week (and
    optionally a cross-below at bear_week)."""
    w_idx = stock.weekly_df.index
    n_w = len(w_idx)
    hma = np.where(np.arange(n_w) < bull_week, 100.0, 120.0)
    if bear_week is not None:
        hma = np.where(np.arange(n_w) >= bear_week, 100.0, hma)
    stock.w_hma = pd.Series(hma, index=w_idx)
    stock.w_ema50 = pd.Series(np.full(n_w, 110.0), index=w_idx)


def _attach(engine, stock):
    """Wire a precomputed stock into an engine the way run()/load_data do."""
    engine.stocks = [stock]
    engine._stock_map = {stock.ticker: stock}
    engine.nifty_df = None
    per = stock.df.index.to_period("W")
    engine._rolls[stock.ticker] = np.flatnonzero(np.asarray(per[1:] != per[:-1])) + 1
    engine._periods[stock.ticker] = per
    engine._pos_size_pct = engine.settings["position_size_pct"]
    engine._score_threshold = engine.settings["score_threshold"]
    engine._stop_pct = engine.settings["stop_loss_pct"]
    engine._target_pct = engine.settings["target_pct"]
    engine._rotation_enabled = engine.settings.get("sector_rotation_enabled", False)
    engine._rotation_boost = engine.settings.get("sector_boost_weight", 0.5)
    engine._sim_end = None
    engine._regime_close = engine._regime_ema = None
    engine._cash = engine.settings["initial_capital"]
    engine._max_pos = engine.settings["max_positions"]
    engine._max_per_sector = engine.settings.get("max_positions_per_sector", 999)
    engine._signals_taken = 0
    engine._new_positions = set()


def _week_end_bar(eng, w):
    """Last daily bar of week w (its Friday)."""
    return int(eng._rolls["TEST"][w]) - 1


def _week_window(eng, stock, w):
    start = eng._week_start("TEST", w)
    end = _week_end_bar(eng, w)
    return stock.df.iloc[start : end + 1]


def _position(ticker="TEST", entry=100.0, stop=92.0, tp=124.0, peak=None):
    return Position(
        ticker=ticker,
        entry_date=datetime(2024, 6, 3),
        entry_price=entry,
        stop_loss=stop,
        target_price=tp,
        shares=10,
        peak_price=peak if peak is not None else entry,
        sector="Other",
    )


@pytest.fixture
def score_stub(monkeypatch):
    """Constant score/sideways so strategy logic is tested in isolation."""
    state = {"total": 80.0, "is_sideways": False}

    def _stub(*_a, **_k):
        return dict(state)

    monkeypatch.setattr(backtest_v2, "compute_score_at_bar", _stub)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Archive (db) roundtrip
# ══════════════════════════════════════════════════════════════════════════════


class TestArchive:
    def test_put_range_get_roundtrip(self):
        df = _rise_frame(n=10)
        assert db.archive_put("TEST", df) == 10
        assert db.archive_range("TEST") == (
            df.index[0].strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d"),
        )
        got = db.archive_get("TEST", start=df.index[2].strftime("%Y-%m-%d"))
        assert len(got) == 8
        assert float(got["close"].iloc[0]) == pytest.approx(float(df["close"].iloc[2]))

    def test_upsert_overwrites_and_isolates_symbols(self):
        df = _rise_frame(n=5)
        db.archive_put("A", df)
        db.archive_put("B", df)
        df2 = df.copy()
        df2["close"] = 999.0
        db.archive_put("A", df2)
        assert db.archive_range("B") is not None
        got = db.archive_get("A")
        assert float(got["close"].iloc[0]) == pytest.approx(999.0)
        assert float(db.archive_get("B")["close"].iloc[0]) == pytest.approx(
            float(df["close"].iloc[0])
        )


# ══════════════════════════════════════════════════════════════════════════════
# Week-end helpers + cross detection
# ══════════════════════════════════════════════════════════════════════════════


class TestWeekHelpers:
    def _mk(self, n):
        days = pd.bdate_range("2024-01-08", periods=n)  # Monday start
        df = pd.DataFrame({"close": [1.0] * n}, index=days)
        stock = SimpleNamespace(df=df, ticker="TEST", weekly_df=None)
        eng = V2Engine()
        eng._stock_map = {"TEST": stock}
        per = days.to_period("W")
        eng._rolls["TEST"] = np.flatnonzero(np.asarray(per[1:] != per[:-1])) + 1
        eng._periods["TEST"] = per
        return eng, stock

    def test_week_id_and_start(self):
        eng, _ = self._mk(15)
        assert eng._week_id("TEST", 0) == 0
        assert eng._week_id("TEST", 4) == 0  # Friday of week 0
        assert eng._week_id("TEST", 5) == 1  # Monday of week 1
        assert eng._week_start("TEST", 0) == 0
        assert eng._week_start("TEST", 1) == 5

    def test_week_ended_only_on_friday(self):
        eng, _ = self._mk(14)  # last bar idx 13 = Thursday
        assert not eng._week_ended("TEST", 3)  # Thursday, Friday follows
        assert eng._week_ended("TEST", 4)  # Friday
        assert not eng._week_ended("TEST", 5)  # Monday
        assert not eng._week_ended("TEST", 13)  # history ends Thursday

    def test_last_bar_counts_when_friday_or_sunday_label(self):
        eng, _ = self._mk(15)  # last bar idx 14 = Friday
        assert eng._week_ended("TEST", 14)
        # weekly-mode bars carry Sunday labels and always count
        days = pd.DatetimeIndex(
            [pd.Timestamp("2024-01-07"), pd.Timestamp("2024-01-14")]
        )
        df = pd.DataFrame({"close": [1.0, 1.0]}, index=days)
        eng2 = V2Engine()
        eng2._stock_map = {"W": SimpleNamespace(df=df, ticker="W", weekly_df=None)}
        per = days.to_period("W")
        eng2._rolls["W"] = np.flatnonzero(np.asarray(per[1:] != per[:-1])) + 1
        eng2._periods["W"] = per
        assert eng2._week_ended("W", 1)  # last bar, Sunday label

    def test_weekly_cross_bull_bear_and_bounds(self):
        w = pd.Series([100.0, 100.0, 120.0, 100.0, 90.0])
        e = pd.Series([110.0] * 5)
        stock = SimpleNamespace(w_hma=w, w_ema50=e)
        eng = V2Engine()
        assert eng._weekly_cross(stock, 2) == (True, False)  # 100<=110 -> 120>110
        assert eng._weekly_cross(stock, 3) == (False, True)  # 120>=110 -> 100<110
        assert eng._weekly_cross(stock, 1) == (False, False)
        assert eng._weekly_cross(stock, 0) == (False, False)  # needs w_idx >= 1
        assert eng._weekly_cross(stock, 99) == (False, False)  # out of bounds
        stock.w_hma = pd.Series([np.nan, np.nan, 120.0, 100.0, 90.0])
        assert eng._weekly_cross(stock, 2) == (False, False)


# ══════════════════════════════════════════════════════════════════════════════
# VP POC port (Pine vpBins proportional-overlap math)
# ══════════════════════════════════════════════════════════════════════════════


class TestVpPoc:
    def test_hand_computed_bins(self):
        df = pd.DataFrame(
            {
                "open": [100.5] * 3,
                "high": [101.0, 110.0, 105.0],
                "low": [100.0, 109.0, 104.0],
                "close": [100.5] * 3,
                "volume": [10.0, 100.0, 5.0],
            },
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        stock = SimpleNamespace(df=df, ticker="TEST")
        eng = V2Engine({"vp_weekly_lookback": 12, "vp_rows": 10})
        eng._rolls["TEST"] = np.array([], dtype=int)  # span fallback = bar_idx+1
        poc = eng._vp_poc(stock, 2)  # window bars 0..2, hh=110 ll=100
        # Volume concentrates in bin 9 ([109,110]) -> unique max, POC = 109.5.
        assert poc == pytest.approx(109.5, abs=1e-9)

    def test_flat_window_returns_a_level(self):
        df = _rise_frame(n=10)
        stock = SimpleNamespace(df=df, ticker="TEST")
        eng = V2Engine()
        eng._rolls["TEST"] = np.array([], dtype=int)
        poc = eng._vp_poc(stock, 9)
        assert poc is not None and 99.0 <= poc <= 101.0


# ══════════════════════════════════════════════════════════════════════════════
# Arming at the weekly close / confirmation expiry (scorer stubbed)
# ══════════════════════════════════════════════════════════════════════════════


class TestArming:
    def _arm(self, eng, stock, bar_idx):
        day = stock.df.index[bar_idx]
        eng._advance_pending(stock, bar_idx, day, eng._regime_ok(day))

    def test_arm_on_week_end_of_cross_week(self, score_stub):
        df = _rise_frame()
        stock = _stock(df)
        _craft_weekly(stock)
        eng = V2Engine()
        _attach(eng, stock)
        arm_bar = _week_end_bar(eng, W_BULL)
        assert stock.df.index[arm_bar].weekday() == 4  # Friday = weekly close
        self._arm(eng, stock, arm_bar)
        pend = eng._pending["TEST"]
        assert pend.bar_idx == arm_bar
        assert pend.w_idx == W_BULL
        wk = _week_window(eng, stock, W_BULL)
        assert pend.level == pytest.approx(float(wk["high"].max()))
        assert pend.low == pytest.approx(float(wk["low"].min()))
        assert eng._signals_generated == 1

    def test_no_arming_midweek(self, score_stub):
        df = _rise_frame()
        stock = _stock(df)
        _craft_weekly(stock)
        eng = V2Engine()
        _attach(eng, stock)
        mid = _week_end_bar(eng, W_BULL) - 2  # Wednesday of the cross week
        assert stock.df.index[mid].weekday() == 2
        self._arm(eng, stock, mid)
        assert "TEST" not in eng._pending

    def test_sideways_blocks_arming(self, score_stub):
        score_stub["is_sideways"] = True
        df = _rise_frame()
        stock = _stock(df)
        _craft_weekly(stock)
        eng = V2Engine()
        _attach(eng, stock)
        self._arm(eng, stock, _week_end_bar(eng, W_BULL))
        assert "TEST" not in eng._pending
        assert eng._signals_blocked == 1

    def test_low_score_blocks_arming(self, score_stub):
        score_stub["total"] = 49.0  # default threshold is 50
        df = _rise_frame()
        stock = _stock(df)
        _craft_weekly(stock)
        eng = V2Engine()
        _attach(eng, stock)
        self._arm(eng, stock, _week_end_bar(eng, W_BULL))
        assert "TEST" not in eng._pending
        assert eng._signals_blocked == 1

    def test_regime_blocks_arming(self, score_stub):
        df = _rise_frame()
        stock = _stock(df)
        _craft_weekly(stock)
        eng = V2Engine()
        _attach(eng, stock)
        eng._regime_close = pd.Series([100.0], index=[stock.df.index[0]])
        eng._regime_ema = pd.Series([200.0], index=[stock.df.index[0]])
        self._arm(eng, stock, _week_end_bar(eng, W_BULL))
        assert "TEST" not in eng._pending
        assert eng._signals_blocked == 1

    def test_held_position_blocks_arming(self, score_stub):
        df = _rise_frame()
        stock = _stock(df)
        _craft_weekly(stock)
        eng = V2Engine()
        _attach(eng, stock)
        eng.positions = [_position(ticker="TEST")]
        self._arm(eng, stock, _week_end_bar(eng, W_BULL))
        assert "TEST" not in eng._pending
        assert eng._signals_blocked == 1

    def test_arm_ignores_close_vs_mas(self, score_stub):
        # Closes stay below the crafted MAs (max 110) — arming no longer
        # checks close > max(MAs); that gate moved to the confirmation.
        df = _rise_frame(flat=200, base=100.0, end=110.0)
        stock = _stock(df)
        _craft_weekly(stock)
        eng = V2Engine()
        _attach(eng, stock)
        self._arm(eng, stock, _week_end_bar(eng, W_BULL))
        assert "TEST" in eng._pending

    def test_bear_cross_clears_pending(self, score_stub):
        df = _rise_frame()
        stock = _stock(df)
        _craft_weekly(stock, bull_week=W_BULL, bear_week=W_BEAR)
        eng = V2Engine()
        _attach(eng, stock)
        self._arm(eng, stock, _week_end_bar(eng, W_BULL))
        assert "TEST" in eng._pending
        self._arm(eng, stock, _week_end_bar(eng, W_BEAR))
        assert "TEST" not in eng._pending

    def test_pending_expires_after_confirm_weeks(self, score_stub):
        df = _rise_frame()
        stock = _stock(df)
        _craft_weekly(stock)
        eng = V2Engine({"confirm_weeks": 2})
        _attach(eng, stock)
        arm_bar = _week_end_bar(eng, W_BULL)
        eng._pending["TEST"] = Pending(arm_bar, W_BULL, 999.0, 1.0)
        for w in (W_BULL + 1, W_BULL + 2):
            b = _week_end_bar(eng, w)
            self._arm(eng, stock, b)
        assert "TEST" in eng._pending  # within window
        b = _week_end_bar(eng, W_BULL + 3)
        self._arm(eng, stock, b)
        assert "TEST" not in eng._pending  # expired (n_w > confirm_weeks)


# ══════════════════════════════════════════════════════════════════════════════
# Entry gates (planted pending, evaluated at the weekly close)
# ══════════════════════════════════════════════════════════════════════════════


class TestEntryGates:
    def _setup(self, score_stub, **overrides):
        df = _rise_frame()
        stock = _stock(df)
        eng = V2Engine({"score_threshold": 0, **overrides})
        _attach(eng, stock)
        # No scripted weekly cross -> _advance_pending never arms/clears.
        _craft_weekly(stock, bull_week=10_000)
        assert stock.df.index[FRI].weekday() == 4
        return eng, stock

    def _plant(self, eng, stock, bar_idx, level=None, w_offset=1):
        w = eng._week_id("TEST", bar_idx)
        if level is None:
            level = float(stock.df["close"].iloc[bar_idx - 1]) + 0.1
        eng._pending["TEST"] = Pending(
            bar_idx,
            w - w_offset,
            level,
            float(stock.df["low"].iloc[bar_idx - 1]),
        )

    def test_all_gates_green_opens_next_open(self, score_stub):
        eng, stock = self._setup(score_stub)
        self._plant(eng, stock, FRI)
        eng._try_entries(stock.df.index[FRI])
        assert len(eng.positions) == 1
        pos = eng.positions[0]
        assert pos.entry_price == pytest.approx(float(stock.df["open"].iloc[FRI + 1]))
        assert pos.entry_date == stock.df.index[FRI + 1]
        assert "TEST" not in eng._pending  # consumed

    def test_midweek_bar_never_enters(self, score_stub):
        eng, stock = self._setup(score_stub)
        wed = FRI - 2
        assert stock.df.index[wed].weekday() == 2
        self._plant(eng, stock, wed)
        eng._try_entries(stock.df.index[wed])
        assert not eng.positions
        assert "TEST" in eng._pending

    def test_red_week_blocks(self, score_stub):
        eng, stock = self._setup(score_stub)
        self._plant(eng, stock, FRI)
        ws = eng._week_start("TEST", eng._week_id("TEST", FRI))
        stock.df.iloc[ws, stock.df.columns.get_loc("open")] = (
            float(stock.df["close"].iloc[FRI]) + 1.0
        )  # week opens above its close -> red
        eng._try_entries(stock.df.index[FRI])
        assert not eng.positions
        assert "TEST" in eng._pending

    def test_close_below_signal_week_high_blocks(self, score_stub):
        eng, stock = self._setup(score_stub)
        eng._pending["TEST"] = Pending(
            FRI, eng._week_id("TEST", FRI) - 1, 1e9, 1.0
        )  # impossible level
        eng._try_entries(stock.df.index[FRI])
        assert not eng.positions

    def test_close_below_weekly_mas_blocks(self, score_stub):
        eng, stock = self._setup(score_stub)
        self._plant(eng, stock, FRI)
        n_w = len(stock.weekly_df)
        stock.w_hma = pd.Series(np.full(n_w, 1000.0), index=stock.weekly_df.index)
        stock.w_ema50 = pd.Series(np.full(n_w, 1100.0), index=stock.weekly_df.index)
        eng._try_entries(stock.df.index[FRI])
        assert not eng.positions
        assert "TEST" in eng._pending

    def test_close_below_poc_blocks(self, score_stub):
        eng, stock = self._setup(score_stub)
        self._plant(eng, stock, FRI)
        # Heavy volume concentrated in a narrow band at the window high ->
        # POC lands above the entry close.
        stock.df.iloc[275, stock.df.columns.get_loc("high")] = 200.0
        stock.df.iloc[275, stock.df.columns.get_loc("low")] = 199.5
        stock.df.iloc[275, stock.df.columns.get_loc("volume")] = 1e9
        assert eng._vp_poc(stock, FRI) > float(stock.df["close"].iloc[FRI])
        eng._try_entries(stock.df.index[FRI])
        assert not eng.positions

    def test_low_score_blocks(self, score_stub):
        score_stub["total"] = 0.0
        eng, stock = self._setup(score_stub, score_threshold=45)
        self._plant(eng, stock, FRI)
        eng._try_entries(stock.df.index[FRI])
        assert not eng.positions

    def test_expiry_window_rejects_late_bar(self, score_stub):
        eng, stock = self._setup(score_stub, confirm_weeks=2)
        self._plant(eng, stock, FRI, w_offset=3)  # n_w = 3 > 2
        eng._try_entries(stock.df.index[FRI])
        assert not eng.positions


# ══════════════════════════════════════════════════════════════════════════════
# Stop seeds at fill (chandelier hybrid / rbracket target)
# ══════════════════════════════════════════════════════════════════════════════


class TestStopSeeds:
    def _open_at(self, mode, pend_low):
        df = _rise_frame()
        stock = _stock(df)
        _craft_weekly(stock, bull_week=10_000)  # no weekly cross interference
        eng = V2Engine({"score_threshold": 0, "exit_mode": mode})
        _attach(eng, stock)
        bar_idx = FRI
        w = eng._week_id("TEST", bar_idx)
        eng._pending["TEST"] = Pending(
            bar_idx, w - 1, float(df["close"].iloc[bar_idx - 1]) + 0.1, pend_low
        )
        eng._w_atr["TEST"] = pd.Series(
            [5.0], index=pd.DatetimeIndex([df.index[bar_idx]])
        )
        eng._try_entries(df.index[bar_idx])
        assert len(eng.positions) == 1
        return eng, stock, eng.positions[0], float(df["open"].iloc[bar_idx + 1])

    def test_chandelier_structure_seed_used_when_within_cap(self):
        df_probe = _rise_frame()
        entry = float(df_probe["open"].iloc[FRI + 1])
        _, _, pos, entry_px = self._open_at("chandelier", pend_low=entry - 9.5)
        # struct = pendLow - 0.3*5 = entry - 11 <= cap 2.0*2.0*5=20
        assert pos.stop_loss == pytest.approx(entry_px - 11.0)

    def test_chandelier_fallback_when_structure_too_far(self):
        _, _, pos, entry_px = self._open_at("chandelier", pend_low=1.0)
        assert pos.stop_loss == pytest.approx(entry_px - 2.0 * 5.0)

    def test_rbracket_target_seeded(self):
        _, _, pos, entry_px = self._open_at("rbracket", pend_low=99.0)
        r = entry_px * 8.0 / 100.0
        assert pos.target_price == pytest.approx(entry_px + r * 3.0)


# ══════════════════════════════════════════════════════════════════════════════
# Risk exits
# ══════════════════════════════════════════════════════════════════════════════


class TestRiskExits:
    def _bar(self, day, open_, high, low):
        return pd.Series(
            {"open": open_, "high": high, "low": low, "close": (open_ + high) / 2},
            name=pd.Timestamp(day),
        )

    def test_signals_mode_has_no_risk_exits(self):
        eng = V2Engine({"exit_mode": "signals"})
        pos = _position()
        assert eng._risk_exit(pos, None, self._bar("2024-06-04", 90, 95, 85)) is None

    def test_rbracket_stop(self):
        eng = V2Engine({"exit_mode": "rbracket"})
        pos = _position(entry=100.0, stop=92.0, peak=100.0)
        res = eng._risk_exit(pos, None, self._bar("2024-06-04", 93, 95, 91))
        assert res.exit_reason == "RBRACKET_STOP"
        assert res.exit_price == pytest.approx(92.0)

    def test_rbracket_stop_gap_down_fills_at_open(self):
        eng = V2Engine({"exit_mode": "rbracket"})
        pos = _position(entry=100.0, stop=92.0, peak=100.0)
        res = eng._risk_exit(pos, None, self._bar("2024-06-04", 88, 90, 85))
        assert res.exit_price == pytest.approx(88.0)

    def test_rbracket_be_roll_tracks_peak(self):
        eng = V2Engine({"exit_mode": "rbracket"})
        pos = _position(entry=100.0, stop=92.0, peak=112.0)
        res = eng._risk_exit(pos, None, self._bar("2024-06-04", 105, 115, 102.5))
        # stop rolled to peak - 1R = 104; open above stop -> fill at stop
        assert res.exit_reason == "RBRACKET_STOP"
        assert res.exit_price == pytest.approx(104.0)

    def test_rbracket_target_and_gap_up(self):
        eng = V2Engine({"exit_mode": "rbracket"})
        pos = _position(entry=100.0, stop=92.0, tp=124.0, peak=110.0)
        res = eng._risk_exit(pos, None, self._bar("2024-06-04", 110, 125, 105))
        assert res.exit_reason == "TARGET"
        assert res.exit_price == pytest.approx(124.0)
        pos2 = _position(entry=100.0, stop=92.0, tp=124.0, peak=110.0)
        res2 = eng._risk_exit(pos2, None, self._bar("2024-06-05", 130, 135, 128))
        assert res2.exit_price == pytest.approx(130.0)

    def test_rbracket_updates_peak_without_exit(self):
        eng = V2Engine({"exit_mode": "rbracket"})
        pos = _position(entry=100.0, stop=92.0, peak=100.0)
        assert eng._risk_exit(pos, None, self._bar("2024-06-04", 101, 110, 99)) is None
        assert pos.peak_price == pytest.approx(110.0)

    def test_chandelier_ratchet_then_stop(self):
        eng = V2Engine({"exit_mode": "chandelier", "atr_mult": 2.0})
        stock = SimpleNamespace(ticker="TEST")
        day = pd.Timestamp("2024-06-04")
        eng._w_atr["TEST"] = pd.Series([5.0], index=pd.DatetimeIndex([day]))
        pos = _position(entry=100.0, stop=95.0, peak=110.0)
        # Bar 1: high 112 -> peak 112, stop ratchets to 112-10=102, no exit.
        assert eng._risk_exit(pos, stock, self._bar(day, 103, 112, 101)) is None
        assert pos.stop_loss == pytest.approx(102.0)
        # Bar 2: low pierces the ratcheted stop -> CHAND_STOP.
        res = eng._risk_exit(pos, stock, self._bar("2024-06-05", 103, 105, 101.5))
        assert res.exit_reason == "CHAND_STOP"
        assert res.exit_price == pytest.approx(102.0)

    def test_chandelier_stop_gap_down_fills_at_open(self):
        eng = V2Engine({"exit_mode": "chandelier", "atr_mult": 2.0})
        stock = SimpleNamespace(ticker="TEST")
        day = pd.Timestamp("2024-06-04")
        eng._w_atr["TEST"] = pd.Series([5.0], index=pd.DatetimeIndex([day]))
        pos = _position(entry=100.0, stop=102.0, peak=112.0)
        res = eng._risk_exit(pos, stock, self._bar(day, 95, 100, 90))
        assert res.exit_reason == "CHAND_STOP"
        assert res.exit_price == pytest.approx(95.0)


# ══════════════════════════════════════════════════════════════════════════════
# Weekly sell exit + re-arm after a non-signal close
# ══════════════════════════════════════════════════════════════════════════════


class TestWeeklySellAndReArm:
    def _held_setup(self, score_stub, bear_week=W_BEAR, mode="signals", **overrides):
        df = _rise_frame()
        stock = _stock(df)
        _craft_weekly(stock, bull_week=W_BULL, bear_week=bear_week)
        eng = V2Engine({"score_threshold": 0, "exit_mode": mode, **overrides})
        _attach(eng, stock)
        eng.positions = [_position(ticker="TEST")]
        return eng, stock

    def test_bear_cross_schedules_next_open_exit(self, score_stub):
        eng, stock = self._held_setup(score_stub)
        sell_bar = _week_end_bar(eng, W_BEAR)  # Friday closes the bear week
        day = stock.df.index[sell_bar]
        eng._check_exits(day)
        assert not eng.trades  # scheduled, not closed same bar
        assert "TEST" in eng._exit_pending
        # Next bar fills at the open (= next week's first session).
        nxt = stock.df.index[sell_bar + 1]
        eng._check_exits(nxt)
        assert len(eng.trades) == 1
        trade = eng.trades[0]
        assert trade.exit_reason == "WEEKLY_SELL"
        assert trade.exit_price == pytest.approx(
            float(stock.df["open"].iloc[sell_bar + 1])
        )
        assert not eng.positions
        assert "TEST" not in eng._exit_pending

    def test_stop_close_rearms_confirmation_gate(self, score_stub):
        eng, stock = self._held_setup(score_stub, bear_week=10_000, mode="chandelier")
        # No weekly bear; chandelier stop pierced intrabar.
        bar_idx = 350
        day = stock.df.index[bar_idx]
        pos = eng.positions[0]
        lo = float(stock.df["low"].iloc[bar_idx])
        hi = float(stock.df["high"].iloc[bar_idx])
        pos.stop_loss = (lo + hi) / 2.0  # mid-bar: low <= stop < high
        eng._check_exits(day)
        assert len(eng.trades) == 1
        assert eng.trades[0].exit_reason == "CHAND_STOP"
        # Weekly still bull + rising, regime/score OK -> re-armed with a
        # week-indexed pending (gates start at the next weekly close).
        assert "TEST" in eng._pending
        assert eng._pending["TEST"].bar_idx == bar_idx
        assert eng._pending["TEST"].w_idx == eng._week_id("TEST", bar_idx) - 1

    def test_weekly_sell_does_not_rearm(self, score_stub):
        eng, stock = self._held_setup(score_stub)
        sell_bar = _week_end_bar(eng, W_BEAR)
        eng._check_exits(stock.df.index[sell_bar])
        eng._check_exits(stock.df.index[sell_bar + 1])
        assert eng.trades and eng.trades[0].exit_reason == "WEEKLY_SELL"
        assert "TEST" not in eng._pending


# ══════════════════════════════════════════════════════════════════════════════
# Commission math
# ══════════════════════════════════════════════════════════════════════════════


class TestCommission:
    def test_close_position_per_side_commission(self):
        pos = _position(entry=100.0)
        t = _close_position(pos, 110.0, datetime(2024, 6, 10), "X", commission_pct=0.1)
        # gross 10*10=100; commission 0.001*(110+100)*10=2.1
        assert t.pnl == pytest.approx(97.9)
        assert t.pnl_pct == pytest.approx(9.79)

    def test_zero_commission_keeps_gross(self):
        pos = _position(entry=100.0)
        t = _close_position(pos, 110.0, datetime(2024, 6, 10), "X")
        assert t.pnl == pytest.approx(100.0)
        assert t.pnl_pct == pytest.approx(10.0)

    def test_engine_close_uses_settings_commission(self):
        eng = V2Engine({"commission_pct": 0.1})
        pos = _position(entry=100.0)
        t = eng._close(pos, 110.0, datetime(2024, 6, 10), "X")
        assert t.pnl == pytest.approx(97.9)


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end: full run() over a scripted cross (real scorer)
# ══════════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    def _run(self, bear_week=None, **overrides):
        df = _rise_frame()
        stock = _stock(df)
        _craft_weekly(stock, bull_week=W_BULL, bear_week=bear_week)
        eng = V2Engine(
            {
                "score_threshold": -1000,  # accept any score; gates still apply
                "exit_mode": "signals",
                "initial_capital": 1_000_000,
                **overrides,
            }
        )
        _attach(eng, stock)
        metrics = eng.run()
        return eng, metrics

    def test_enters_after_cross_and_closes_at_end(self):
        eng, metrics = self._run()
        assert metrics["total_trades"] == 1
        trade = eng.trades[0]
        assert trade.exit_reason == "END_OF_DATA"
        # Entry is the next open after the first all-gates-green weekly close.
        entry_bar = eng._stock_map["TEST"].df.index.get_loc(trade.entry_date)
        assert trade.entry_price == pytest.approx(
            float(eng._stock_map["TEST"].df["open"].iloc[entry_bar])
        )
        assert entry_bar > _week_end_bar(eng, W_BULL)  # week after the cross
        assert 0 < trade.pnl_pct < 100  # uptrend + commission applied

    def test_weekly_bear_exits_at_next_open(self):
        eng, metrics = self._run(bear_week=W_BEAR)
        assert metrics["total_trades"] == 1
        trade = eng.trades[0]
        assert trade.exit_reason == "WEEKLY_SELL"
        stock = eng._stock_map["TEST"]
        sell_bar = _week_end_bar(eng, W_BEAR)
        assert trade.exit_date == stock.df.index[sell_bar + 1]
        assert trade.exit_price == pytest.approx(
            float(stock.df["open"].iloc[sell_bar + 1])
        )

    def test_score_gate_blocks_all_entries(self):
        eng, metrics = self._run(score_threshold=101)
        assert metrics["total_trades"] == 0
        assert eng._signals_generated == 1  # cross fired
        assert eng._signals_blocked == 1  # score below threshold at arm time
        assert "TEST" not in eng._pending  # never armed


# ══════════════════════════════════════════════════════════════════════════════
# load_data offline path (archive hit — no network)
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadData:
    def test_archive_hit_avoids_fetch(self):
        days = pd.bdate_range(
            end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=900
        )
        for sym in ("TEST", "^NSEI"):
            df = _rise_frame(n=900)
            df.index = days
            db.archive_put(sym, df)
        eng = V2Engine({"score_threshold": 0})
        with (
            patch.object(
                backtest_v2,
                "fetch_batch_yfinance",
                side_effect=AssertionError("network fetch"),
            ),
            patch.object(
                backtest_v2,
                "fetch_index_data",
                side_effect=AssertionError("network fetch"),
            ),
        ):
            eng.load_data(["TEST"], period="3y")
        assert len(eng.stocks) == 1
        assert eng.nifty_df is not None
        assert "TEST" in eng._rolls and len(eng._rolls["TEST"]) > 0
        assert "TEST" in eng._periods
        assert "TEST" in eng._w_atr

    def test_stale_archive_triggers_fetch_path(self):
        days = pd.bdate_range(
            end=pd.Timestamp.now().normalize() - pd.Timedelta(days=400), periods=500
        )
        df = _rise_frame(n=500)
        df.index = days
        db.archive_put("TEST", df)  # older than the 7-day freshness window
        eng = V2Engine({"score_threshold": 0})
        days = pd.bdate_range(
            end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=500
        )
        fetched = _rise_frame(n=500)
        fetched.index = days
        with (
            patch.object(
                backtest_v2, "fetch_batch_yfinance", return_value={"TEST": fetched}
            ) as fetch_mock,
            patch.object(backtest_v2, "fetch_index_data", return_value=None),
        ):
            eng.load_data(["TEST"], period="1y")
        assert fetch_mock.called
        assert len(eng.stocks) == 1
        # Fresh data upserted into the archive.
        rng = db.archive_range("TEST")
        assert rng[1] >= (
            pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
        ).strftime("%Y-%m-%d")


# ══════════════════════════════════════════════════════════════════════════════
# Weekly timeframe mode (Pine on a 1W chart)
# ══════════════════════════════════════════════════════════════════════════════


class TestWeeklyMode:
    def test_load_resamples_to_weekly_and_uses_max_fetch(self):
        days = pd.bdate_range(
            end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=2000
        )
        daily = _rise_frame(n=2000)
        daily.index = days
        eng = V2Engine({"score_threshold": 0, "timeframe": "weekly"})
        with (
            patch.object(
                backtest_v2, "fetch_batch_yfinance", return_value={"TEST": daily}
            ) as fetch_mock,
            patch.object(backtest_v2, "fetch_index_data", return_value=daily),
        ):
            eng.load_data(["TEST"], period="24y")
        assert fetch_mock.call_args.kwargs["period"] == "max"
        assert len(eng.stocks) == 1
        wdf = eng.stocks[0].df
        assert 395 <= len(wdf) <= 405  # 2000 daily bars -> ~400 weekly
        assert wdf.index.to_period("W").nunique() == len(wdf)  # one bar per week
        assert eng.nifty_df is not None and len(eng.nifty_df) == len(wdf)

    def test_cross_arms_on_the_weekly_bar_itself(self, score_stub):
        daily = _rise_frame(n=1500, flat=200)
        wdf = backtest_v2.to_weekly(daily)
        st = backtest_v2.precompute_stock("TEST", wdf, V2Engine().settings)
        assert st is not None
        cross_bar = 150
        n_w = len(st.weekly_df)
        hma = np.where(np.arange(n_w) < cross_bar, 100.0, 120.0)
        st.w_hma = pd.Series(hma, index=st.weekly_df.index)
        st.w_ema50 = pd.Series(np.full(n_w, 110.0), index=st.weekly_df.index)
        eng = V2Engine({"score_threshold": 0, "timeframe": "weekly"})
        _attach(eng, st)
        day = st.df.index[cross_bar]
        eng._advance_pending(st, cross_bar, day, True)
        assert eng._pending["TEST"].bar_idx == cross_bar  # armed same bar
        assert eng._pending["TEST"].w_idx == cross_bar
        assert eng._signals_generated == 1


# ══════════════════════════════════════════════════════════════════════════════
# TV-parity knobs: warmup, sl_pct wiring, risk sizing
# ══════════════════════════════════════════════════════════════════════════════


class TestParityKnobs:
    def test_stop_loss_pct_follows_sl_pct(self):
        assert V2Engine({"sl_pct": 10}).settings["stop_loss_pct"] == 10
        assert V2Engine().settings["stop_loss_pct"] == 8  # sl_pct default

    def test_warmup_bars_configurable(self):
        df = _rise_frame()
        stock = _stock(df)
        eng = V2Engine()
        _attach(eng, stock)
        assert len(eng._simulation_window()) == len(df) - 260
        eng2 = V2Engine({"warmup_bars": 300})
        _attach(eng2, stock)
        assert len(eng2._simulation_window()) == len(df) - 300

    def test_risk_pct_sizes_by_sl_pct_distance(self, score_stub):
        df = _rise_frame()
        stock = _stock(df)
        _craft_weekly(stock, bull_week=10_000)
        eng = V2Engine({"score_threshold": 0, "risk_pct": 8.5, "sl_pct": 10})
        _attach(eng, stock)
        equity = eng._cash
        bar_idx = FRI
        w = eng._week_id("TEST", bar_idx)
        eng._pending["TEST"] = Pending(
            bar_idx, w - 1, float(df["close"].iloc[bar_idx - 1]) + 0.1, 1.0
        )
        eng._try_entries(df.index[bar_idx])
        entry = float(df["open"].iloc[bar_idx + 1])
        dist = entry * 0.10  # slPct stop distance, Pine qty formula
        expected = min(int(equity * 0.085 / dist), int(equity / entry))
        assert eng.positions[0].shares == expected


# ══════════════════════════════════════════════════════════════════════════════
# Live status (TRIGGERED / ARMED as of the latest bar)
# ══════════════════════════════════════════════════════════════════════════════


class TestLiveStatus:
    def _mk(self, score_stub, offset, bear_offset=None, **overrides):
        """Weekly-mode stock; bull cross `offset` bars before the end."""
        daily = _rise_frame(n=1500, flat=200, end=1000.0)
        wdf = backtest_v2.to_weekly(daily)
        st = backtest_v2.precompute_stock("TEST", wdf, V2Engine().settings)
        assert st is not None
        n_w = len(st.weekly_df)
        _craft_weekly(
            st,
            bull_week=n_w - offset,
            bear_week=(n_w - bear_offset) if bear_offset else None,
        )
        eng = V2Engine({"score_threshold": 0, "timeframe": "weekly", **overrides})
        _attach(eng, st)
        return eng, st, n_w

    def test_triggered_when_gates_green_on_last_bar(self, score_stub):
        eng, st, _ = self._mk(score_stub, offset=2)
        rows = eng.run_status()
        assert len(rows) == 1
        r = rows[0]
        assert r["state"] == "TRIGGERED"
        assert r["n"] == 1  # cross at n_w-2, last bar n_w-1
        assert r["score"] == pytest.approx(80.0)

    def test_armed_on_cross_bar_itself(self, score_stub):
        eng, st, _ = self._mk(score_stub, offset=1)
        rows = eng.run_status()
        assert len(rows) == 1
        assert rows[0]["state"] == "ARMED"
        assert rows[0]["n"] == 0
        assert rows[0]["score"] is None

    def test_bear_cross_cancels_status(self, score_stub):
        eng, st, _ = self._mk(score_stub, offset=4, bear_offset=1)
        rows = eng.run_status()
        assert rows == []


# ══════════════════════════════════════════════════════════════════════════════
# High-break entry trigger (buy-stop at the signal-week high)
# ══════════════════════════════════════════════════════════════════════════════


class TestHighBreak:
    @pytest.fixture(autouse=True)
    def _stub_poc(self, monkeypatch):
        # Crafted last-bar OHLC drags the real POC above the crafted close;
        # POC gating is orthogonal to what these tests verify.
        monkeypatch.setattr(V2Engine, "_vp_poc", lambda self, s, b: 1.0)

    def _mk(self, score_stub, open_off, close_off, high_off):
        """Weekly stock with cross at n_w-2; last bar OHLC offset from level."""
        daily = _rise_frame(n=1500, flat=200, end=1000.0)
        wdf = backtest_v2.to_weekly(daily)
        st = backtest_v2.precompute_stock("TEST", wdf, V2Engine().settings)
        assert st is not None
        n_w = len(st.df)
        _craft_weekly(st, bull_week=n_w - 2)
        level = float(st.df["high"].iloc[n_w - 2])
        o, c, h = level + open_off, level + close_off, level + high_off
        for col, val in (
            ("open", o),
            ("close", c),
            ("high", h),
            ("low", min(o, c) - 0.5),
        ):
            st.df.iloc[-1, st.df.columns.get_loc(col)] = val
        return st, level

    def _eng(self, st, **overrides):
        eng = V2Engine({"score_threshold": 0, "timeframe": "weekly", **overrides})
        _attach(eng, st)
        # bar n_w-2 arms the cross; last bar evaluates the gates
        for i in (len(st.df) - 2, len(st.df) - 1):
            eng._try_entries(st.df.index[i])
        return eng

    def test_highbreak_opens_last_bar_close_mode_blocked(self, score_stub):
        # green close BELOW the level, high pierces it
        st, level = self._mk(score_stub, open_off=-1.0, close_off=-0.5, high_off=0.5)
        eng_close = self._eng(st)
        assert eng_close.positions == []
        eng_hb = self._eng(st, entry_trigger="highbreak")
        assert len(eng_hb.positions) == 1
        assert eng_hb.positions[0].entry_price == pytest.approx(level)

    def test_highbreak_gap_open_fills_at_open(self, score_stub):
        st, level = self._mk(score_stub, open_off=2.0, close_off=3.0, high_off=4.0)
        eng = self._eng(st, entry_trigger="highbreak")
        assert len(eng.positions) == 1
        assert eng.positions[0].entry_price == pytest.approx(level + 2.0)

    def test_highbreak_keeps_green_candle_gate(self, score_stub):
        # high above level but bar closes red -> still blocked
        st, _ = self._mk(score_stub, open_off=1.0, close_off=0.5, high_off=2.0)
        eng = self._eng(st, entry_trigger="highbreak")
        assert eng.positions == []

    def test_default_entry_trigger_is_close(self):
        assert V2Engine().settings["entry_trigger"] == "close"


class TestHighBreakIntraWeek:
    """Daily mode: confirm week = 5 sessions; the stop fills at the FIRST
    session whose high crosses the signal-week level, not at the Friday
    evaluation bar."""

    @pytest.fixture(autouse=True)
    def _stub_poc(self, monkeypatch):
        monkeypatch.setattr(V2Engine, "_vp_poc", lambda self, s, b: 1.0)

    def test_fills_at_first_session_above_level(self, score_stub):
        daily = _rise_frame(n=1000, flat=100, end=1000.0)  # Monday start
        st = backtest_v2.precompute_stock("TEST", daily, V2Engine().settings)
        assert st is not None and st.weekly_df is not None
        n_w = len(st.weekly_df)
        assert n_w == 200
        _craft_weekly(st, bull_week=n_w - 2)
        # bar i weekday = i % 5 -> signal week 990..994 (Fri), final 995..999
        level = float(st.df["high"].iloc[990:995].max())
        # Mon/Tue stay under level, Wed pierces it (the stop hit), Thu/Fri on
        for i, (o, c, h, lo) in {
            995: (-10, -9, -1, -11),
            996: (-9, -8, -0.5, -10),
            997: (-2, 1, 3, -3),  # Wednesday: breaks the level
            998: (1, 2, 3, 0),
            999: (2, 4, 5, 1),
        }.items():
            for col, off in (("open", o), ("close", c), ("high", h), ("low", lo)):
                st.df.iloc[i, st.df.columns.get_loc(col)] = level + off
        eng = V2Engine(
            {"score_threshold": 0, "timeframe": "daily", "entry_trigger": "highbreak"}
        )
        _attach(eng, st)
        for i in range(990, 1000):
            eng._try_entries(st.df.index[i])
        assert len(eng.positions) == 1
        pos = eng.positions[0]
        assert pos.entry_date == st.df.index[997]  # Wednesday, not Friday
        assert pos.entry_price == pytest.approx(level)  # Wed open below level


# ══════════════════════════════════════════════════════════════════════════════
# Live scan (scan_latest): latest bullish crosses + entry readiness
# ══════════════════════════════════════════════════════════════════════════════


class TestWeeklyShortHistory:
    def test_precompute_accepts_short_weekly_frame(self):
        """Regression: weekly mode fed <260 WEEKLY bars into the daily
        WARMUP_BARS gate, so `--timeframe weekly --years 3` loaded 0/84."""
        daily = _rise_frame(n=800, flat=100, end=500.0)  # ~160 weeks
        wdf = backtest_v2.to_weekly(daily)
        assert wdf is not None and len(wdf) < 260
        st = backtest_v2.precompute_stock(
            "SHORT", wdf, V2Engine({"timeframe": "weekly"}).settings
        )
        assert st is not None
        assert st.weekly_df is not None


class TestScanLatest:
    def _mk(self, score_stub, offset, bear_offset=None, **overrides):
        """Weekly-mode stock; bull cross `offset` bars before the end.

        Uses unique frame content (flat=185, end=1007) — ``to_weekly``
        caches by close-content and returns a SHARED object, and the
        TestHighBreak* classes mutate ``st.df`` in place, which would
        poison any later test reusing the same fixture content.
        """
        daily = _rise_frame(n=1500, flat=185, end=1007.0)
        wdf = backtest_v2.to_weekly(daily)
        st = backtest_v2.precompute_stock("TEST", wdf, V2Engine().settings)
        assert st is not None
        n_w = len(st.weekly_df)
        _craft_weekly(
            st,
            bull_week=n_w - offset,
            bear_week=(n_w - bear_offset) if bear_offset else None,
        )
        eng = V2Engine({"score_threshold": 0, "timeframe": "weekly", **overrides})
        _attach(eng, st)
        return eng, st, n_w

    def test_triggered_when_all_gates_green(self, score_stub):
        eng, st, _ = self._mk(score_stub, offset=2)
        rows = eng.scan_latest()
        assert len(rows) == 1
        r = rows[0]
        assert r["state"] == "TRIGGERED"
        assert r["n"] == 1
        assert r["score"] == pytest.approx(80.0)
        assert r["missing"] == []
        assert r["level"] is not None
        assert eng.scan_meta["scanned"] == 1
        assert eng.scan_meta["crosses_in_window"] >= 1

    def test_armed_on_cross_bar_names_confirm(self, score_stub):
        eng, st, _ = self._mk(score_stub, offset=1)
        rows = eng.scan_latest()
        assert len(rows) == 1
        r = rows[0]
        assert r["state"] == "ARMED"
        assert r["n"] == 0
        assert r["score"] == pytest.approx(80.0)  # arming-week score
        assert "n=0" in r["note"]

    def test_armed_names_missing_green_gate(self, score_stub):
        eng, st, n_w = self._mk(score_stub, offset=3)
        st.df = st.df.copy()  # mutate our copy; to_weekly cache stays clean
        cols = st.df.columns
        # Block an EARLIER qualification: the rising fixture passes the
        # default close trigger at n=1 (week n_w-2), which would be
        # TRIGGERED. Kill both triggers: close AND high below the level.
        signal_high = float(st.df["high"].iloc[n_w - 3])
        st.df.iloc[n_w - 2, cols.get_loc("high")] = signal_high - 5.0
        st.df.iloc[n_w - 2, cols.get_loc("close")] = signal_high - 10.0
        # Make the last weekly bar red: green-candle gate fails.
        last = len(st.df) - 1
        st.df.iloc[last, cols.get_loc("close")] = (
            float(st.df["open"].iloc[last]) - 1.0
        )
        rows = eng.scan_latest()
        assert len(rows) == 1
        r = rows[0]
        assert r["state"] == "ARMED"
        assert r["n"] == 2
        assert "green" in r["missing"]
        assert "green week" in r["note"]

    def test_bear_cancel_shows_recent_expired(self, score_stub):
        eng, st, n_w = self._mk(score_stub, offset=4, bear_offset=1)
        st.df = st.df.copy()
        cols = st.df.columns
        # Keep weeks n_w-3..n_w-2 from qualifying (default close trigger AND
        # highbreak both fail: close and high below the signal level) before
        # the bear at n_w-1 cancels the pending.
        signal_high = float(st.df["high"].iloc[n_w - 4])
        for w in (n_w - 3, n_w - 2):
            st.df.iloc[w, cols.get_loc("high")] = signal_high - 5.0
            st.df.iloc[w, cols.get_loc("close")] = signal_high - 10.0
        rows = eng.scan_latest()
        assert len(rows) == 1
        r = rows[0]
        assert r["state"] == "EXPIRED"
        assert r["note"] == "bearish cross"

    def test_earlier_break_reports_qualified_not_armed(self, score_stub):
        """Backtest parity: a signal-high break that already qualified inside
        the window shows TRIGGERED + qualified <date> — not ARMED — even
        when the latest week fails every gate (retrace)."""
        eng, st, n_w = self._mk(score_stub, offset=10)
        st.df = st.df.copy()
        # Rising fixture qualifies at n=1 (week n_w-9); break the latest
        # week so a last-bar-only scan would say ARMED.
        last = len(st.df) - 1
        st.df.iloc[last, st.df.columns.get_loc("close")] = (
            float(st.df["open"].iloc[last]) - 1.0
        )
        rows = eng.scan_latest()
        assert len(rows) == 1
        r = rows[0]
        assert r["state"] == "TRIGGERED"
        assert r["n"] == 9  # current weeks since cross (scan-date window open)
        assert r["note"] == f"qualified {st.df.index[n_w - 9]:%Y-%m-%d}"
        assert r["missing"] == []
        assert r["score"] == pytest.approx(80.0)

    def test_qualified_survives_later_bear(self, score_stub):
        """Entry already happened at n=1 — a later bearish cross must not
        reclassify it as EXPIRED."""
        eng, st, n_w = self._mk(score_stub, offset=6, bear_offset=1)
        rows = eng.scan_latest()
        assert len(rows) == 1
        r = rows[0]
        assert r["state"] == "TRIGGERED"
        assert r["note"] == f"qualified {st.df.index[n_w - 5]:%Y-%m-%d}"
        assert r["n"] == 5  # cross still within confirm weeks of scan date

    def test_stale_qualification_never_generates_long(self, score_stub):
        """A Long is generated only to the date the scan is run: a
        qualification whose confirm window closed before the scan date
        must not show as TRIGGERED (it falls back to EXPIRED/hidden)."""
        eng, st, _ = self._mk(score_stub, offset=20)  # cross 19w before end
        # Rising fixture qualifies at n=1 — but the window times out at
        # n>15, ~3 weeks before the scan date.
        rows = eng.scan_latest()
        assert len(rows) == 1
        r = rows[0]
        assert r["state"] == "EXPIRED"  # NOT TRIGGERED despite qualifying
        assert r["note"] == "timeout"

    def test_old_expiry_not_reported(self, score_stub):
        eng, st, _ = self._mk(score_stub, offset=4, bear_offset=8)
        assert eng.scan_latest() == []

    def test_no_rows_without_cross(self, score_stub):
        daily = _rise_frame(n=1500, flat=185, end=1007.0)
        wdf = backtest_v2.to_weekly(daily)
        st = backtest_v2.precompute_stock("TEST", wdf, V2Engine().settings)
        assert st is not None  # no crafted cross — real MAs, cross ages out
        eng = V2Engine({"score_threshold": 0, "timeframe": "weekly"})
        _attach(eng, st)
        assert eng.scan_latest() == []

    def test_daily_midweek_run_evals_last_friday(self, score_stub):
        # Last bar is a Wednesday (bar 997, 997 % 5 == 2): gates must be
        # evaluated at the previous Friday (bar 994, week n_w-2).
        daily = _rise_frame(n=998, flat=100, end=1000.0)
        st = backtest_v2.precompute_stock("TEST", daily, V2Engine().settings)
        assert st is not None and st.weekly_df is not None
        n_w = len(st.weekly_df)
        _craft_weekly(st, bull_week=n_w - 3)
        assert st.df.index[-1].weekday() == 2  # Wednesday
        eng = V2Engine({"score_threshold": 0, "timeframe": "daily"})
        _attach(eng, st)
        rows = eng.scan_latest()
        assert len(rows) == 1
        r = rows[0]
        assert r["state"] == "TRIGGERED"
        assert r["n"] == 1  # cross at week n_w-3, eval week n_w-2

    def test_scan_html_smoke(self, score_stub, tmp_path):
        eng, st, _ = self._mk(score_stub, offset=2)
        rows = eng.scan_latest()
        out = tmp_path / "scan.html"
        from scanner.backend.backtest_report import generate_scan_html

        generate_scan_html(
            rows, eng.settings, eng.scan_meta, str(out)
        )
        html = out.read_text(encoding="utf-8")
        assert "TEST" in html
        assert "Entry now" in html
        assert "all gates OK" in html
