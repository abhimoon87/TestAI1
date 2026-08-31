"""Tests for the backtest engine — SectorTracker, Position, _close_position, update_position, _compute_metrics."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from scanner.backtest import (
    Position,
    TradeResult,
    SectorTracker,
    StockData,
    BacktestEngine,
    _close_position,
    update_position,
    get_sector,
    DEFAULT_SETTINGS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(ticker="RELIANCE", sector="OilGas", pnl=10.0, days=5,
                entry_price=100.0, entry_score=60.0,
                entry_date=None, exit_date=None):
    ed = entry_date or datetime(2025, 1, 1)
    xd = exit_date or (ed + timedelta(days=days))
    return TradeResult(
        ticker=ticker,
        entry_date=ed,
        entry_price=entry_price,
        entry_score=entry_score,
        stop_loss=entry_price * 0.95,
        target_price=entry_price * 1.20,
        exit_date=xd,
        exit_price=entry_price * (1 + pnl / 100),
        exit_reason="TARGET",
        shares=10,
        pnl=(entry_price * pnl / 100) * 10,
        pnl_pct=pnl,
        days_held=days,
        investment=entry_price * 10,
        sector=sector,
    )


def _make_pos(ticker="RELIANCE", entry_price=100.0, shares=10,
              stop_loss=95.0, target_price=120.0, entry_date=None):
    return Position(
        ticker=ticker,
        entry_date=entry_date or datetime(2025, 1, 1),
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        shares=shares,
        entry_score=60.0,
        sector="OilGas",
    )


# ---------------------------------------------------------------------------
# get_sector
# ---------------------------------------------------------------------------

class TestGetSector:
    def test_known_ticker(self):
        assert get_sector("RELIANCE") == "OilGas"
        assert get_sector("TCS") == "IT"
        assert get_sector("SBIN") == "Banking"

    def test_unknown_ticker(self):
        assert get_sector("FAKE_TICKER") == "Other"


# ---------------------------------------------------------------------------
# SectorTracker
# ---------------------------------------------------------------------------

class TestSectorTracker:
    def test_empty_sector_returns_zero_momentum(self):
        tracker = SectorTracker()
        assert tracker.get_sector_momentum("Nonexistent") == 0.0

    def test_momentum_positive_with_winners(self):
        tracker = SectorTracker(lookback=5)
        for i in range(6):
            tracker.record_trade(_make_trade(pnl=10.0, sector="IT"))
        mom = tracker.get_sector_momentum("IT")
        assert mom > 0

    def test_momentum_negative_with_losers(self):
        tracker = SectorTracker(lookback=5)
        for i in range(6):
            tracker.record_trade(_make_trade(pnl=-10.0, sector="Banking"))
        mom = tracker.get_sector_momentum("Banking")
        assert mom < 0

    def test_sector_blocked_after_5_trades_below_threshold(self):
        tracker = SectorTracker(lookback=8, block_threshold=-5.0)
        for i in range(6):
            tracker.record_trade(_make_trade(pnl=-8.0, sector="Pharma"))
        assert tracker.is_sector_blocked("Pharma") is True

    def test_sector_not_blocked_with_few_trades(self):
        tracker = SectorTracker(lookback=8, block_threshold=-5.0)
        for i in range(4):
            tracker.record_trade(_make_trade(pnl=-8.0, sector="Pharma"))
        assert tracker.is_sector_blocked("Pharma") is False

    def test_sector_not_blocked_above_threshold(self):
        tracker = SectorTracker(lookback=8, block_threshold=-5.0)
        for i in range(6):
            tracker.record_trade(_make_trade(pnl=-2.0, sector="Pharma"))
        assert tracker.is_sector_blocked("Pharma") is False

    def test_top_sector_requires_min_trades(self):
        tracker = SectorTracker()
        tracker.record_trade(_make_trade(pnl=10.0, sector="IT"))
        tracker.record_trade(_make_trade(pnl=10.0, sector="IT"))
        assert tracker.is_top_sector("IT", top_n=3) is False  # only 2 trades

    def test_top_sector_with_enough_trades(self):
        tracker = SectorTracker()
        for _ in range(5):
            tracker.record_trade(_make_trade(pnl=10.0, sector="IT"))
        for _ in range(3):
            tracker.record_trade(_make_trade(pnl=-5.0, sector="Banking"))
        for _ in range(4):
            tracker.record_trade(_make_trade(pnl=2.0, sector="Pharma"))
        for _ in range(3):
            tracker.record_trade(_make_trade(pnl=1.0, sector="Auto"))
        assert tracker.is_top_sector("IT", top_n=3) is True
        assert tracker.is_top_sector("Banking", top_n=3) is False

    def test_get_sector_summary(self):
        tracker = SectorTracker()
        for _ in range(3):
            tracker.record_trade(_make_trade(pnl=10.0, sector="IT"))
        summary = tracker.get_sector_summary()
        assert "IT" in summary
        assert summary["IT"]["trades"] == 3
        assert summary["IT"]["wins"] == 3

    def test_get_all_momenta(self):
        tracker = SectorTracker()
        tracker.record_trade(_make_trade(pnl=5.0, sector="IT"))
        tracker.record_trade(_make_trade(pnl=-3.0, sector="Banking"))
        mom = tracker.get_all_momenta()
        assert "IT" in mom
        assert "Banking" in mom

    def test_log_decision(self):
        tracker = SectorTracker()
        tracker.log_decision("RELIANCE", "Energy", "BOOSTED", 5.0, 70.0)
        assert len(tracker.decisions) == 1
        assert tracker.decisions[0]["action"] == "BOOSTED"


# ---------------------------------------------------------------------------
# _close_position
# ---------------------------------------------------------------------------

class TestClosePosition:
    def test_basic_close(self):
        pos = _make_pos(entry_price=100.0, stop_loss=95.0)
        result = _close_position(pos, 110.0, datetime(2025, 1, 10), "TARGET")
        assert result.exit_price == 110.0
        assert result.exit_reason == "TARGET"
        assert result.pnl == pytest.approx(100.0, abs=0.1)  # (110-100)*10
        assert result.pnl_pct == pytest.approx(10.0, abs=0.1)
        assert result.days_held == 9

    def test_stop_loss_preserved_for_all_reasons(self):
        pos = _make_pos(entry_price=100.0, stop_loss=95.0)
        for reason in ["STOP_LOSS", "TRAILING_STOP", "END_OF_DATA", "TARGET"]:
            pos2 = _make_pos(entry_price=100.0, stop_loss=95.0)
            result = _close_position(pos2, 105.0, datetime(2025, 1, 10), reason)
            assert result.stop_loss == 95.0, f"stop_loss lost for {reason}"

    def test_pnl_calculation(self):
        pos = _make_pos(entry_price=200.0, shares=5, stop_loss=190.0)
        result = _close_position(pos, 220.0, datetime(2025, 1, 10), "TARGET")
        assert result.pnl == pytest.approx(100.0, abs=0.1)  # (220-200)*5
        assert result.pnl_pct == pytest.approx(10.0, abs=0.1)


# ---------------------------------------------------------------------------
# update_position
# ---------------------------------------------------------------------------

class TestUpdatePosition:
    def test_stop_loss_triggered(self):
        settings = {**DEFAULT_SETTINGS}
        pos = _make_pos(entry_price=100.0, stop_loss=95.0)
        bar = pd.Series({"open": 94.0, "high": 96.0, "low": 93.0, "close": 94.5},
                        name=datetime(2025, 1, 5))
        result = update_position(pos, bar, 0, settings)
        assert result is not None
        assert result.exit_reason == "STOP_LOSS"
        assert result.exit_price == 94.0  # gap down → filled at open

    def test_stop_loss_gap_down(self):
        settings = {**DEFAULT_SETTINGS}
        pos = _make_pos(entry_price=100.0, stop_loss=95.0)
        bar = pd.Series({"open": 90.0, "high": 91.0, "low": 89.0, "close": 90.5},
                        name=datetime(2025, 1, 5))
        result = update_position(pos, bar, 0, settings)
        assert result is not None
        assert result.exit_price == 90.0  # gap below stop → filled at open

    def test_target_hit_activates_trailing(self):
        settings = {**DEFAULT_SETTINGS, "trail_pct": 2.0}
        pos = _make_pos(entry_price=100.0, stop_loss=95.0, target_price=120.0)
        # Bar hits target but close == target_price (no further upside to update trail)
        bar = pd.Series({"open": 119.0, "high": 121.0, "low": 118.0, "close": 120.0},
                        name=datetime(2025, 1, 5))
        result = update_position(pos, bar, 0, settings)
        # Target is hit, trailing stop activates, but low (118) > trail_stop (117.6)
        assert result is None
        assert pos.hit_target is True
        assert pos.trail_stop > 0

    def test_trailing_stop_triggered(self):
        settings = {**DEFAULT_SETTINGS, "trail_pct": 2.0}
        pos = _make_pos(entry_price=100.0, stop_loss=95.0, target_price=120.0)
        pos.hit_target = True
        pos.peak_price = 130.0
        pos.trail_stop = 130.0 * 0.98  # 127.4
        bar = pd.Series({"open": 128.0, "high": 129.0, "low": 126.0, "close": 127.0},
                        name=datetime(2025, 1, 10))
        result = update_position(pos, bar, 0, settings)
        assert result is not None
        assert result.exit_reason == "TRAILING_STOP"

    def test_trailing_stop_not_mislabeled_as_stop_loss(self):
        """After target hit, trailing stop should be labeled TRAILING_STOP, not STOP_LOSS."""
        settings = {**DEFAULT_SETTINGS, "trail_pct": 2.0}
        pos = _make_pos(entry_price=100.0, stop_loss=95.0, target_price=120.0)
        # Simulate: target was hit on a previous bar, stop_loss was ratcheted up
        pos.hit_target = True
        pos.peak_price = 125.0
        pos.trail_stop = 125.0 * 0.98  # 122.5
        pos.stop_loss = max(95.0, pos.trail_stop)  # ratcheted to 122.5
        # Bar where low pierces the trail level
        bar = pd.Series({"open": 123.0, "high": 124.0, "low": 121.0, "close": 122.0},
                        name=datetime(2025, 1, 10))
        result = update_position(pos, bar, 0, settings)
        assert result is not None
        assert result.exit_reason == "TRAILING_STOP", \
            f"Expected TRAILING_STOP but got {result.exit_reason}"

    def test_no_exit_on_normal_bar(self):
        settings = {**DEFAULT_SETTINGS}
        pos = _make_pos(entry_price=100.0, stop_loss=95.0, target_price=120.0)
        bar = pd.Series({"open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0},
                        name=datetime(2025, 1, 5))
        result = update_position(pos, bar, 0, settings)
        assert result is None


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_empty_trades(self):
        engine = BacktestEngine()
        engine.trades = []
        engine.equity_curve = []
        metrics = engine._compute_metrics(1_000_000, 0, 0)
        assert metrics["total_trades"] == 0

    def test_basic_metrics(self):
        engine = BacktestEngine()
        engine.trades = [
            _make_trade(pnl=10.0, days=5, entry_price=100.0),
            _make_trade(pnl=-5.0, days=3, entry_price=100.0),
            _make_trade(pnl=15.0, days=7, entry_price=100.0),
        ]
        engine.equity_curve = [
            (datetime(2025, 1, 1), 1_000_000),
            (datetime(2025, 1, 6), 1_010_000),
            (datetime(2025, 1, 9), 1_005_000),
            (datetime(2025, 1, 16), 1_020_000),
        ]
        metrics = engine._compute_metrics(1_000_000, 10, 3)
        assert metrics["total_trades"] == 3
        assert metrics["win_rate"] == pytest.approx(200 / 3, abs=0.1)
        assert metrics["total_pnl"] > 0
        assert metrics["best_trade"].pnl_pct == 15.0
        assert metrics["worst_trade"].pnl_pct == -5.0
        assert metrics["max_drawdown_pct"] >= 0


class TestNegativeScenarios:
    def test_losing_trade_pnl_negative(self):
        t = _make_trade(pnl=-10.0)
        assert t.pnl < 0
        assert t.pnl_pct == -10.0

    def test_no_cross_stays_open(self):
        """Fresh engine has no open positions."""
        engine = BacktestEngine()
        assert len(engine.positions) == 0

    def test_sector_tracker_no_trades(self):
        tracker = SectorTracker(lookback=5)
        assert tracker.is_top_sector("IT") is False
        assert tracker.is_sector_blocked("Banking") is False

    def test_compute_metrics_all_losing(self):
        engine = BacktestEngine()
        engine.trades = [
            _make_trade(pnl=-5.0),
            _make_trade(pnl=-3.0),
            _make_trade(pnl=-8.0),
        ]
        engine.equity_curve = [
            (datetime(2025, 1, 1), 1_000_000),
            (datetime(2025, 1, 20), 984_000),
        ]
        metrics = engine._compute_metrics(1_000_000, 10, 3)
        assert metrics["win_rate"] == 0.0
        assert metrics["total_pnl"] < 0
        assert metrics["worst_trade"].pnl_pct == -8.0
