"""Detail panel — Institutional Positioning card (per-stock FII%, DII%, Promoters)."""

import datetime
from types import SimpleNamespace
from unittest.mock import patch

import flet as ft

from scanner.tests.ui.conftest import make_app as _make_app

_SHP = {
    "quarter": "Jun 2026",
    "series": {
        "promoters": {"latest": 50.48, "total": 0.21, "recent": 0.48},
        "foreign_institutions": {"latest": 17.19, "total": -5.41, "recent": -1.48},
        "domestic_institutions": {"latest": 21.1, "total": 5.11, "recent": 0.64},
    },
}
# 10 rows → total window = recent window = 5 vs 5, hand-computable:
# fii now -5000 vs then -10000 → +50.0%;  dii now +5000 vs then +2500 → +100.0%.
_FLOW = [
    {
        "date": f"2026-09-{i + 1:02d}",
        "fii": -1000.0 if i < 5 else -2000.0,
        "dii": 1000.0 if i < 5 else 500.0,
    }
    for i in range(10)
]
_FLOW_ZERO = [{"date": "d", "fii": 0.0, "dii": 0.0} for _ in range(10)]


def _row(ticker="RELIANCE", promoter=51.8):
    row = {
        "ticker": ticker,
        "total": 62.0,
        "close": 2500.0,
        "combined_rating": "GOOD",
        "entry_signal": False,
        "trend_dir": "Bull",
        "trend": 10,
        "momentum": 10,
        "rsi": 55,
        "macd": 5,
        "volume": 8,
        "rel_str": 7,
        "fundamentals": 12,
        "pc1m": 2.5,
        "adx_val": 25,
        "is_sideways": False,
        "ma_bullish": True,
    }
    if promoter is not None:
        row["_promoter_holding"] = promoter
    return row


def _texts(control, out=None):
    """Every ft.Text value under a control tree."""
    if out is None:
        out = []
    if isinstance(control, ft.Text):
        out.append(str(control.value))
    for attr in ("controls", "content"):
        child = getattr(control, attr, None)
        if isinstance(child, list):
            for x in child:
                _texts(x, out)
        elif child is not None and not isinstance(child, (str, int, float, bool)):
            try:
                _texts(child, out)
            except (AttributeError, TypeError):
                pass
    return out


def _open(
    app,
    row,
    flow=_FLOW,
    shp=_SHP,
    start_recorder=None,
    promoter_series=None,
):
    """Render the detail panel with all institutional sources patched."""
    app.all_results = [row]
    if shp:
        row["_shareholding"] = shp
    if start_recorder is None:
        started = []
        app._start_institutional_load = lambda t: started.append(t)
    else:
        started = start_recorder
    series = promoter_series or [
        ["2026-01-01", 51.8],
        [datetime.date.today().isoformat(), 52.0],
    ]
    with (
        patch(
            "scanner.api.indian_market.fetch_fii_dii_history",
            side_effect=lambda limit=60, cache_only=False: flow,
        ),
        patch(
            "scanner.api.indian_fundamentals.record_promoter_snapshot",
            return_value=series,
        ),
    ):
        app._show_stock_detail(row["ticker"])
    return started


class TestInstitutionalCardRender:
    def test_card_is_own_section_below_main_row(self):
        app = _make_app()
        _open(app, _row())
        panel = app.table_column.controls[0]
        outer = panel.content
        assert len(outer.controls) == 3  # header, main row, card section
        main_row = outer.controls[1]
        assert any("Why this trade?" in _texts(c) for c in main_row.controls)
        assert "Institutional Positioning" in _texts(outer.controls[2])
        assert "Institutional Positioning" not in _texts(main_row)

    def test_all_tiles_and_labels_present(self):
        app = _make_app()
        started = _open(app, _row())
        texts = _texts(app.table_column)
        assert "Institutional Positioning" in texts
        for label in ("FII", "DII", "Promoters"):
            assert label in texts
        # Market-wide sources never render — same numbers on every stock.
        assert "FPI" not in texts
        assert "Smart Money" not in texts
        assert started == []  # every source on the row → no background load

    def test_shareholding_tiles_and_sublines(self):
        app = _make_app()
        _open(app, _row())
        texts = _texts(app.table_column)
        assert "17.2%" in texts  # FII latest, 1dp
        assert "21.1%" in texts  # DII latest
        assert "50.5%" in texts  # Promoters latest
        assert "▼" in texts
        assert "▲" in texts
        assert "total -5.41% · recent -1.48%" in texts  # FII pp deltas
        assert "total +5.11% · recent +0.64%" in texts  # DII
        assert "total +0.21% · recent +0.48%" in texts  # Promoters

    def test_up_arrow_when_shareholding_increased(self):
        app = _make_app()
        shp = {
            "quarter": "Jun 2026",
            "series": {
                "promoters": _SHP["series"]["promoters"],
                "foreign_institutions": {
                    "latest": 18.5,
                    "total": 12.4,
                    "recent": 3.1,
                },
                "domestic_institutions": _SHP["series"]["domestic_institutions"],
            },
        }
        _open(app, _row(), shp=shp)
        texts = _texts(app.table_column)
        assert "▲" in texts
        assert "total +12.4% · recent +3.10%" in texts

    def test_flat_positions_say_no_change(self):
        app = _make_app()
        flat = {
            "quarter": "Jun 2026",
            "series": {
                k: {"latest": 20.0, "total": 0.0, "recent": 0.0}
                for k in (
                    "promoters",
                    "foreign_institutions",
                    "domestic_institutions",
                )
            },
        }
        _open(app, _row(), flow=_FLOW_ZERO, shp=flat)
        texts = _texts(app.table_column)
        assert texts.count("No change") == 3
        assert "▼" not in texts and "▲" not in texts

    def test_missing_data_shows_loading_and_requests_fetch(self):
        app = _make_app()
        started = _open(app, _row(promoter=None), flow=None, shp=None)
        texts = _texts(app.table_column)
        assert texts.count("loading…") == 3
        assert started == ["RELIANCE"]

    def test_screener_miss_falls_back_to_cash_flow_tiles(self):
        app = _make_app()
        started = _open(app, _row(promoter=51.8), shp=None)
        texts = _texts(app.table_column)
        # FII/DII fall back to ₹Cr flow values from the history window.
        assert "₹-1,000 Cr" in texts  # newest flow row, fii
        assert "₹+1,000 Cr" in texts  # dii
        assert "total +50.0% · recent +50.0%" in texts
        # Promoters fall back to the yfinance snapshot series.
        assert "51.8%" in texts
        assert "total +0.20% · recent +0.20%" in texts
        assert started == ["RELIANCE"]  # shareholding still missing → load

    def test_dii_falls_back_to_flow_when_domestic_row_missing(self):
        app = _make_app()
        shp = {
            "quarter": "Jun 2026",
            "series": {
                "promoters": _SHP["series"]["promoters"],
                "foreign_institutions": _SHP["series"]["foreign_institutions"],
                # domestic_institutions row missing (screener dropped it)
            },
        }
        _open(app, _row(), shp=shp)
        texts = _texts(app.table_column)
        assert "17.2%" in texts  # FII still from shareholding
        assert "₹+1,000 Cr" in texts  # DII falls back to newest flow row
        assert "total +100.0% · recent +100.0%" in texts  # dii window sums

    def test_marker_only_row_renders_nse_net_tiles(self):
        """The FII/DII filter passes marker rows — their tiles must render too.

        Screener shareholding AND moneycontrol flow both missing: the tile
        falls back to the scan-time NSE net stamped on the row (the same
        markers ``_has_fii_dii`` keys off), instead of showing n/a.
        """
        app = _make_app()
        row = _row()
        row["_fii_is_buying"] = True
        row["_dii_is_buying"] = False
        row["_fii_net"] = 2838.0
        row["_dii_net"] = -1500.0
        started = _open(app, row, flow=None, shp=None)
        texts = _texts(app.table_column)
        assert "₹+2,838 Cr" in texts  # FII tile from the row marker
        assert "₹-1,500 Cr" in texts  # DII marker (selling) renders too
        assert started == ["RELIANCE"]  # shareholding still missing → load


class TestPerTickerSlots:
    """Fresh opens and late thread finishes must not clobber other tickers."""

    def test_fresh_open_does_not_clear_other_tickers_slots(self):
        app = _make_app()
        _open(app, _row("AAA", promoter=None), flow=None, shp=None)
        assert isinstance(app._inst_attempted, set)
        assert isinstance(app._inst_done, set)
        app._inst_attempted.add("AAA")  # AAA's load still in flight
        app._inst_done.add("AAA")
        _open(app, _row("BBB", promoter=None), flow=None, shp=None)
        assert "AAA" in app._inst_attempted  # single-slot code reset these
        assert "AAA" in app._inst_done
        assert "BBB" not in app._inst_done

    def test_late_finish_of_previous_ticker_keeps_current_settled(self):
        app = _make_app()
        _open(app, _row("AAA", promoter=None), flow=None, shp=None)
        _open(app, _row("BBB", promoter=None), flow=None, shp=None)
        # BBB's thread settles it, then AAA's slow thread finishes late —
        # BBB must stay done (old code's single slot would show "loading…").
        app._inst_done.add("BBB")
        app._inst_done.add("AAA")
        _open(app, _row("BBB", promoter=None), flow=None, shp=None)
        texts = _texts(app.table_column)
        assert texts.count("n/a") == 3  # settled missing state, all tiles
        assert "loading…" not in texts


class TestBackgroundLoad:
    def _run_thread(self, app, row, shp_return=None, fund=None):
        renders = []
        app.all_results = [row]
        app._detail_ticker = row["ticker"]
        app._show_stock_detail = lambda t: renders.append(t)
        with (
            patch(
                "scanner.api.indian_market.fetch_fii_dii_history",
                return_value=_FLOW,
            ),
            patch(
                "scanner.api.indian_fundamentals.fetch_shareholding_pattern",
                return_value=shp_return,
            ),
            patch(
                "scanner.api.indian_fundamentals.fetch_trendlyne_fundamentals",
                return_value=fund,
            ) as trendlyne,
        ):
            app._start_institutional_load(row["ticker"])
            app._inst_thread.join(timeout=5)
        return renders, trendlyne

    def test_thread_falls_back_to_promoter_snapshot_and_rerenders_once(self):
        app = _make_app()
        row = _row(promoter=None)
        fund = SimpleNamespace(promoter_holding=51.9)
        renders, _ = self._run_thread(app, row, shp_return=None, fund=fund)
        assert row["_promoter_holding"] == 51.9
        assert row.get("_shareholding") is None
        assert row["ticker"] in app._inst_done
        assert renders == [row["ticker"]]

    def test_thread_stores_shareholding_and_skips_trendlyne(self):
        app = _make_app()
        row = _row(promoter=None)
        renders, trendlyne = self._run_thread(app, row, shp_return=_SHP)
        assert row["_shareholding"] == _SHP
        trendlyne.assert_not_called()
        assert renders == [row["ticker"]]

    def test_attempted_guards_against_second_thread(self):
        app = _make_app()
        row = _row(promoter=None)
        renders, _ = self._run_thread(app, row, shp_return=None, fund=None)
        assert renders == [row["ticker"]]
        # Second call for the same ticker is a no-op.
        app._start_institutional_load(row["ticker"])
        assert renders == [row["ticker"]]
