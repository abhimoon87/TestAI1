"""Regression tests for the post-scan UI pipeline.

Guards against the empty-grid bug: ``_row_pool`` / ``_row_cells`` were
type-annotated but never initialised in ``ScannerApp.__init__``, so
``_create_row_controls`` raised ``AttributeError`` on every row, the
per-row ``except`` swallowed it, and the grid rendered with only a header.

Covers:
- ``_render_current_page`` full-rebuild path through ``_create_row_controls``
- ``_scan_complete`` pool clearing + final render
- ``_final_sync`` result merge (result.results vs streaming fallback)
- ``_on_page_size_change`` / ``_load_all_pages``
"""

from __future__ import annotations

import flet as ft
import pytest

from scanner.tests.ui.test_view_mixins import _make_app


@pytest.fixture(autouse=True)
def _isolate_settings_write(monkeypatch):
    """_on_page_size_change / _load_all_pages call _save_ui_prefs — never
    write the real settings.json from unit tests. _scan_complete also
    persists rows — never write the real last_results.json either."""
    import scanner.backend.settings_store as store_mod

    monkeypatch.setattr(store_mod, "save_settings", lambda s: None)
    monkeypatch.setattr(store_mod, "save_results", lambda rows: None)


def _result(i: int, score: float = 60.0) -> dict:
    return {
        "ticker": f"STK{i:03d}",
        "total": score,
        "close": 100.0 + i,
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


class TestRowPoolInitialisation:
    """The root-cause regression: pool must exist before first row build."""

    def test_init_creates_pool_dicts(self):
        app = _make_app()
        # __init__ path (via _make_app which calls the real initialisers)
        assert hasattr(app, "_row_pool")
        assert hasattr(app, "_row_cells")
        assert app._row_pool == {}
        assert app._row_cells == {}

    def test_create_row_registers_in_pool(self):
        app = _make_app()
        c = app.theme_colors
        row = app._create_row_controls(_result(1), 1, c, c["card"], 50)
        assert isinstance(row, ft.Container)
        assert "STK001" in app._row_pool
        assert app._row_pool["STK001"] is row
        assert "STK001" in app._row_cells
        assert len(app._row_cells["STK001"]) == 18  # RESULT_COLS has 18 cols
        assert row._pool_ticker == "STK001"

    def test_create_row_defensive_when_pool_missing(self):
        """__new__-constructed app (tests bypassing __init__) must not crash."""
        app = _make_app()
        del app._row_pool
        del app._row_cells
        c = app.theme_colors
        row = app._create_row_controls(_result(2), 1, c, c["card"], 50)
        assert isinstance(row, ft.Container)
        assert hasattr(app, "_row_pool") and "STK002" in app._row_pool

    def test_update_row_tolerates_missing_pool(self):
        app = _make_app()
        app._row_pool = {}
        app._row_cells = {}
        assert app._update_row("NOPE", 1, app.theme_colors, 50, 1) is False


class TestFullRebuildRendersRows:
    """The exact path that produced the empty grid."""

    def test_render_current_page_populates_table(self):
        app = _make_app()
        results = [_result(i, score=50 + (i % 40)) for i in range(150)]
        app.all_results = list(results)
        app.filtered_results = list(results)
        app.active_view = "dashboard"
        app.scanning = False
        app.sort_col = None
        app.page_size = 200  # default 100 would only render one page

        app._render_current_page()

        ctrls = app.table_column.controls
        # header + data rows (no empty-state, no placeholder)
        assert len(ctrls) == 1 + 150, f"expected header + 150 rows, got {len(ctrls)}"
        # every data row registered in the pool
        assert len(app._row_pool) == 150
        assert len(app._row_cells) == 150

    def test_rows_have_pool_ticker_and_are_visible(self):
        app = _make_app()
        app.all_results = [_result(i) for i in range(5)]
        app.filtered_results = list(app.all_results)
        app.active_view = "dashboard"
        app.scanning = False
        app.sort_col = None

        app._render_current_page()

        data_rows = [
            c
            for c in app.table_column.controls
            if isinstance(c, ft.Container) and getattr(c, "_pool_ticker", None)
        ]
        assert len(data_rows) == 5
        for row in data_rows:
            assert row.opacity != 0

    def test_summary_and_count_label_populated(self):
        app = _make_app()
        app.all_results = [_result(i) for i in range(10)]
        app.filtered_results = list(app.all_results)
        app.active_view = "dashboard"
        app.scanning = False

        app._render_current_page()

        assert "no scan yet" not in (app.result_count_label.value or "")
        assert "10 scanned" in app.result_count_label.value

    def test_second_render_takes_stream_path_without_error(self):
        app = _make_app()
        app.all_results = [_result(i) for i in range(10)]
        app.filtered_results = list(app.all_results)
        app.active_view = "dashboard"
        app.scanning = False
        app.sort_col = None

        app._render_current_page()  # full rebuild, fills pool
        first_count = len(app.table_column.controls)
        app._render_current_page()  # should hit streaming fast-path
        assert len(app.table_column.controls) >= first_count - 1  # header stays

    def test_scan_complete_clears_pool_and_rebuilds(self):
        app = _make_app()
        # simulate streaming state
        app.all_results = [_result(i) for i in range(8)]
        app.filtered_results = list(app.all_results)
        app.results = app.all_results
        app.active_view = "dashboard"
        app.scanning = True
        app._scan_cancelled = False
        app._row_pool = {"STK000": ft.Container()}
        app._row_cells = {"STK000": []}

        app._scan_complete()

        assert app.scanning is False
        # pool was cleared then repopulated by the full rebuild
        assert len(app._row_pool) == 8
        ctrls = app.table_column.controls
        assert len(ctrls) == 1 + 8

    def test_scan_complete_persists_results(self, monkeypatch):
        """_scan_complete writes all_results through save_results."""
        import scanner.backend.settings_store as store_mod

        saved = []
        monkeypatch.setattr(store_mod, "save_results", saved.append)

        app = _make_app()
        app.all_results = [_result(i) for i in range(3)]
        app.filtered_results = list(app.all_results)
        app.results = app.all_results
        app.active_view = "dashboard"
        app.scanning = True
        app._scan_cancelled = False

        app._scan_complete()

        assert len(saved) == 1
        assert [r["ticker"] for r in saved[0]] == ["STK000", "STK001", "STK002"]


class TestFinalSync:
    """_final_sync merge semantics: engine results preferred, streaming fallback."""

    def _make_engine_result(self, results, cancelled=False, error=None):
        class R:
            pass

        r = R()
        r.results = results
        r.cancelled = cancelled
        r.error = error
        r.warnings = []
        return r

    def test_uses_engine_results_when_present(self):
        app = _make_app()
        engine_rows = [_result(i) for i in range(20)]
        app.all_results = [_result(i) for i in range(5)]  # stale streaming

        # Replicate _final_sync body (closure captures result)
        result = self._make_engine_result(engine_rows)
        final_results = result.results
        if not final_results and app.all_results:
            final_results = app.all_results
        with app._results_lock:
            app.results = final_results
            app.all_results = list(final_results)
            app.filtered_results = [
                r for r in final_results if app._row_matches_filters(r)
            ]

        assert len(app.all_results) == 20
        assert app.results is not app.all_results  # copied

    def test_falls_back_to_streaming_when_engine_empty(self):
        app = _make_app()
        streaming = [_result(i) for i in range(12)]
        app.all_results = list(streaming)

        result = self._make_engine_result([])  # engine lost the results
        final_results = result.results
        if not final_results and app.all_results:
            final_results = app.all_results
        with app._results_lock:
            app.results = final_results
            app.all_results = list(final_results)

        assert len(app.all_results) == 12

    def test_on_stream_batch_populates_all_results(self):
        app = _make_app()
        app.all_results = []
        app.filtered_results = []
        app.results = []
        app.active_view = "dashboard"
        app.scanning = True

        batch = [_result(i) for i in range(30)]
        app._on_stream_batch(batch)

        assert len(app.all_results) == 30
        assert app.results is app.all_results

    def test_on_stream_batch_upserts_by_ticker(self):
        app = _make_app()
        app.all_results = []
        app.filtered_results = []
        app.results = []

        app._on_stream_batch([_result(1, score=55)])
        app._on_stream_batch([_result(1, score=77)])  # update, not duplicate

        assert len(app.all_results) == 1
        assert app.all_results[0]["total"] == 77


class TestPageSize:
    def test_on_page_size_change_from_event_data(self):
        app = _make_app()
        app.page_size = 100

        class E:
            data = "500"
            control = None

        app._on_page_size_change(E())
        assert app.page_size == 500
        assert app.current_page == 0

    def test_on_page_size_change_from_control_value(self):
        app = _make_app()
        app.page_size = 100
        app.page_size_dd.value = "200"

        class Ctl:
            value = "200"

        class E:
            data = None
            control = Ctl()

        app._on_page_size_change(E())
        assert app.page_size == 200

    def test_on_page_size_change_garbage_falls_back(self):
        app = _make_app()
        app.page_size = 100
        app.page_size_dd.value = "not-a-number"

        class E:
            data = "???"
            control = None

        app._on_page_size_change(E())
        assert app.page_size == 100  # unchanged fallback

    def test_on_page_size_change_clamped_to_500(self):
        app = _make_app()

        class E:
            data = "9999"
            control = None

        app._on_page_size_change(E())
        assert app.page_size == 500

    def test_load_all_pages_sets_page_size(self):
        app = _make_app()
        app.all_results = [_result(i) for i in range(350)]
        app.filtered_results = list(app.all_results)
        app.active_view = "dashboard"
        app.scanning = False

        app._load_all_pages()
        assert app.page_size == 350

    def test_load_all_pages_empty(self):
        app = _make_app()
        app.all_results = []
        app.filtered_results = []
        app.active_view = "dashboard"

        app._load_all_pages()
        assert app.page_size == 500
