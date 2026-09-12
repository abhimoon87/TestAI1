"""Offline tests for the view-builder mixins in scanner.app.

The mixins (``LayoutViewMixin`` / ``ResultsViewMixin`` / ``SettingsViewMixin``)
build plain Flet controls, which need no running app or display to construct.
Following the pattern from ``test_app_cache_ui.py``, we create an
un-initialized ``ScannerApp`` via ``__new__``, attach only the state each
method reads (theme colors, settings, a stub page), and inspect the returned
control trees — so the full dashboard/settings build runs without a window.
"""

import flet as ft

import scanner.app as app_mod
import scanner.settings_store as store_mod
from scanner.app import ScannerApp
from scanner.themes import THEMES
from scanner.ui_kit import _score_of
from scanner.views_results import ResultsViewMixin


class _FakePage:
    """Stub flet.Page recording control adds/updates and dialog open/close."""

    def __init__(self):
        self.controls = []
        self.window = type("W", (), {})()
        self.update_calls = 0
        self.opened = []
        self.closed = []

    def add(self, control):
        self.controls.append(control)

    def update(self):
        self.update_calls += 1

    def open(self, control):
        self.opened.append(control)

    def close(self, control):
        self.closed.append(control)


def _make_app():
    """Un-initialized ScannerApp with the state view builders read."""
    import threading
    app = ScannerApp.__new__(ScannerApp)
    app.page = _FakePage()
    app.current_theme = "dark"
    app.theme_colors = THEMES["dark"]
    app.settings = dict(app_mod.DEFAULT_SETTINGS)
    app.active_view = "dashboard"
    app.scanning = False
    app._scan_lock = threading.Lock()
    app._scan_epoch = 0
    app.logged = []
    app._log = app.logged.append
    return app


def _cell(row, idx):
    """The ft.Text inside results-grid cell ``idx`` of a data row."""
    cell = row.content.controls[idx]
    return cell.content


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard build — exercises every LayoutViewMixin builder
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildDashboard:
    def test_build_ui_creates_all_controls(self):
        """_build_ui assembles rail + sidebar + main + right panel without error."""
        app = _make_app()
        app._build_ui()

        # The page got exactly one top-level layout row
        assert len(app.page.controls) == 1
        assert len(app.page.controls[0].controls) == 4  # rail/sidebar/main/right

        # Sidebar controls the rest of the app reads after build
        for attr in ("universe_dd", "timeframe_dd", "period_dd", "trend_filter_dd",
                     "rating_filter_dd", "threshold_slider", "action_btn",
                     "progress_bar", "progress_label", "cache_status_lbl"):
            assert hasattr(app, attr), f"missing sidebar control {attr}"

        # Main-area controls
        for attr in ("search_entry", "html_btn", "csv_btn", "clear_btn",
                     "table_column", "main_area_box", "dashboard_content",
                     "summary_cards", "chart_bars", "pagination_bar"):
            assert hasattr(app, attr), f"missing main-area control {attr}"

        # Right panel + rail
        for attr in ("status_label", "topicks_column", "log_column",
                     "log_view", "_rail_pills"):
            assert hasattr(app, attr), f"missing panel control {attr}"
        # Dashboard pill is visible; settings page builds from it
        assert app._rail_pills["dashboard"].visible is True

    def test_build_summary_row_creates_eight_cards(self):
        """Summary stat cards are keyed and labelled for later updates."""
        app = _make_app()
        app.summary_cards = {}
        row = app._build_summary_row()
        assert len(row.controls) == 8
        assert set(app.summary_cards) == {
            "total", "passed", "entry", "avg", "high", "bull", "bear", "dead_skip",
        }


class TestBuildSettingsView:
    def test_build_settings_view_makes_one_input_per_spec_field(self):
        """Every declared setting gets an input, keyed for the save handler."""
        app = _make_app()
        view = app._build_settings_view()

        spec_fields = sum(len(fields) for _, fields in app._SETTINGS_SPEC)
        assert view is not None
        assert len(app._settings_inputs) == spec_fields > 0
        # The stale-member age input exists in the spec
        assert app._settings_inputs["stale_member_max_age_days"]._settings_kind == "float"
        assert app._settings_inputs["stale_member_max_age_days"]._settings_lo == 7
        assert app._settings_inputs["stale_member_max_age_days"]._settings_hi == 730
        # spot-check representative keys are present and typed
        assert app._settings_inputs["fast_ma_len"]._settings_kind == "int"
        assert app._settings_inputs["fast_ma_type"]._settings_kind == "ma_type"
        assert app._settings_inputs["theme"]._settings_kind == "theme"
        assert hasattr(app, "_settings_error")

    def test_stale_age_input_reflects_saved_value(self):
        """After a restart the settings page pre-fills the persisted value."""
        app = _make_app()
        app.settings = {**app_mod.DEFAULT_SETTINGS, "stale_member_max_age_days": 60.0}

        app._build_settings_view()

        assert app._settings_inputs["stale_member_max_age_days"].value == "60.0"

    def test_settings_build_creates_stale_audit_controls(self):
        """Settings page ships the audit button + Apply fixes button + status."""
        app = _make_app()

        view = app._build_settings_view()

        assert hasattr(app, "stale_audit_lbl")
        assert app.stale_audit_lbl.value == ""
        assert hasattr(app, "stale_fix_btn")
        assert app.stale_fix_btn.disabled is True  # only after an audit enables it
        assert view is not None

    def test_audit_report_core_runs_and_formats(self, monkeypatch):
        """_audit_report drives audit_stale_members and formats the table."""
        import scanner.audit_stale_members as audit_mod
        fake_res = {
            "unannotated_stale": [("XXYYY", "2024-02-05")],
            "annotated_stale": [("GSPL", "2026-05-11")],
            "annotated_fresh": [],
            "missing": [],
            "neg_cache_skipped": [],
            "period": "3y", "max_age_days": 45.0,
            "tickers": 51, "fetched": 51, "stale_total": 2,
            "membership": {}, "annotated": [],
        }
        monkeypatch.setattr(audit_mod, "audit_stale_members", lambda t: fake_res)
        app = _make_app()

        res, report = app._audit_report("NIFTY 50")

        assert res is fake_res
        assert "STALE — NOT ANNOTATED" in report
        assert "XXYYY" in report

    def test_audit_verdict_reports_fix_ready(self):
        """Verdict flags 'fixes ready' exactly when apply_fixes has work."""
        app = _make_app()
        res = {
            "unannotated_stale": [], "annotated_stale": [], "missing": [],
            "annotated_fresh": [], "rename_suggestions": {},
        }
        assert "fixes ready" not in app._audit_verdict(res)

        res["rename_suggestions"] = {"ASTER": "ASTERDM"}
        verdict = app._audit_verdict(res)
        assert "fixes ready" in verdict
        assert "1 rename(s) suggested" in verdict

        res["rename_suggestions"] = {}
        res["unannotated_stale"] = [("STALECO", "2025-11-10")]
        assert "fixes ready" in app._audit_verdict(res)

    def test_apply_fixes_core_runs_and_formats(self, monkeypatch):
        """_apply_fixes_core drives apply_fixes and formats the summary."""
        import scanner.audit_stale_members as audit_mod
        fake_summary = {
            "changed": True, "renamed": [("ASTER", "ASTERDM", 2)],
            "annotated_added": [("STALECO", "2025-11-10")],
            "annotated_removed": [], "not_found": [],
            "backup": "/x/universes.py.bak",
        }
        monkeypatch.setattr(audit_mod, "apply_fixes", lambda res: fake_summary)
        app = _make_app()

        summary, text = app._apply_fixes_core({"rename_suggestions": {}})

        assert summary is fake_summary
        assert "ASTER -> ASTERDM" in text
        assert "STALECO to SUSPENDED_OR_DELISTED" in text

    def _audit_res(self, **over):
        res = {
            "unannotated_stale": [], "annotated_stale": [], "missing": [],
            "annotated_fresh": [], "rename_suggestions": {},
        }
        res.update(over)
        return res

    def test_apply_stale_fixes_opens_confirmation_dialog(self):
        """Clicking Apply fixes opens a modal dialog listing what will change."""
        app = _make_app()
        app._safe_update = lambda fn: fn()
        app._last_audit_res = self._audit_res(
            annotated_fresh=["RESUMEDCO"],
            rename_suggestions={"ASTER": "ASTERDM"},
        )

        app._apply_stale_fixes()

        assert len(app.page.opened) == 1
        dlg = app.page.opened[0]
        assert dlg.modal is True
        assert "1 rename(s), 1 annotation removal(s)" in dlg.content.value
        assert [a.content.value for a in dlg.actions] == ["Cancel", "Apply fixes"]

    def test_apply_stale_fixes_noop_without_fixable_result(self):
        """No dialog when there is no audit result or nothing to apply."""
        app = _make_app()
        app._safe_update = lambda fn: fn()

        app._apply_stale_fixes()  # no _last_audit_res yet
        assert app.page.opened == []

        app._last_audit_res = self._audit_res()  # empty findings
        app._apply_stale_fixes()
        assert app.page.opened == []

    def test_cancel_closes_dialog_without_applying(self):
        """Cancel closes the dialog and never touches universes.py."""
        app = _make_app()
        app._safe_update = lambda fn: fn()
        app._last_audit_res = self._audit_res(
            rename_suggestions={"ASTER": "ASTERDM"})
        app._apply_stale_fixes()
        dlg = app.page.opened[0]

        app._close_dialog(dlg)

        assert app.page.closed == [dlg]
        assert app.page.opened == [dlg]  # no apply ran

    def test_run_apply_fixes_applies_logs_and_triggers_reauth(self, monkeypatch):
        """The threaded runner applies, logs the summary, and re-audits."""
        import time

        import scanner.audit_stale_members as audit_mod
        app = _make_app()
        app._safe_update = lambda fn: fn()
        app._last_audit_res = self._audit_res(
            unannotated_stale=[("STALECO", "2025-11-10")])
        fake_summary = {
            "changed": True, "renamed": [], "not_found": [],
            "annotated_added": [("STALECO", "2025-11-10")],
            "annotated_removed": [], "backup": "/x/universes.py.bak",
        }
        monkeypatch.setattr(audit_mod, "apply_fixes", lambda res: fake_summary)
        monkeypatch.setattr(app_mod.ScannerApp, "_reload_universes",
                            lambda self: self.logged.append("RELOAD"))
        monkeypatch.setattr(app_mod.ScannerApp, "_run_stale_audit",
                            lambda self, e=None: self.logged.append("REAUDIT"))

        app._run_apply_fixes()

        deadline = time.time() + 5
        while getattr(app, "stale_fix_running", False) and time.time() < deadline:
            time.sleep(0.01)
        log = " ".join(app.logged)
        assert "Fix done" in log
        assert "STALECO" in log
        assert "RELOAD" in app.logged and "REAUDIT" in app.logged
        assert app._last_audit_res is None

    def test_run_apply_fixes_keeps_newer_reauth_result(self, monkeypatch):
        """A re-audit result stored during the fix survives the finally."""
        import time

        import scanner.audit_stale_members as audit_mod
        app = _make_app()
        app._safe_update = lambda fn: fn()
        fresh = {"fresh": True}
        app._last_audit_res = self._audit_res(
            unannotated_stale=[("STALECO", "2025-11-10")])
        monkeypatch.setattr(audit_mod, "apply_fixes", lambda res: {
            "changed": True, "renamed": [], "not_found": [],
            "annotated_added": [("STALECO", "2025-11-10")],
            "annotated_removed": [], "backup": "/x/universes.py.bak",
        })
        monkeypatch.setattr(app_mod.ScannerApp, "_reload_universes",
                            lambda self: None)
        monkeypatch.setattr(
            app_mod.ScannerApp, "_run_stale_audit",
            lambda self, e=None: setattr(self, "_last_audit_res", fresh))

        app._run_apply_fixes()

        deadline = time.time() + 5
        while getattr(app, "stale_fix_running", False) and time.time() < deadline:
            time.sleep(0.01)
        assert app._last_audit_res is fresh  # NOT clobbered to None

    def test_failed_audit_clears_result_and_disables_apply(self, monkeypatch):
        """A failed audit never leaves a stale fixable result actionable."""
        import time
        app = _make_app()
        app._safe_update = lambda fn: fn()
        app._build_settings_view()  # creates stale_fix_btn
        app._last_audit_res = self._audit_res(
            rename_suggestions={"ASTER": "ASTERDM"})
        app.stale_fix_btn.disabled = False  # as if a prior good audit ran

        def boom(universe):
            raise RuntimeError("network down")
        monkeypatch.setattr(app, "_audit_report", boom)

        app._run_stale_audit()

        deadline = time.time() + 5
        while getattr(app, "stale_audit_running", False) and time.time() < deadline:
            time.sleep(0.01)
        assert app._last_audit_res is None
        assert app.stale_fix_btn.disabled is True


# ══════════════════════════════════════════════════════════════════════════════
# Results mixin — sorting keys and data-row formatting
# ══════════════════════════════════════════════════════════════════════════════


def _row(**over):
    base = {
        "ticker": "TCS", "total": 45.0, "combined_rating": "POOR",
        "entry_signal": False, "close": 120.0, "trend": 8.0,
        "momentum": 9.0, "rsi": 5.0, "macd": 4.0, "volume": 6.0,
        "rel_str": 4.0, "fundamentals": 0.0, "pc1m": -2.5,
        "trend_dir": "Bear", "adx_val": 18.0, "is_sideways": True,
        "ma_crossed_above": False, "ma_bullish": False,
        "crossover_bars_ago": -1, "crossover_count": 0,
    }
    base.update(over)
    return base


class TestGetSortKey:
    def test_sorts_by_score_descending(self):
        app = _make_app()
        rows = [_row(ticker="A", total=30), _row(ticker="B", total=80)]
        ordered = sorted(rows, key=app._get_sort_key(2), reverse=True)
        assert ordered[0]["ticker"] == "B"

    def test_sorts_by_rating_order(self):
        app = _make_app()
        key = app._get_sort_key(3)
        rows = [_row(ticker="A", combined_rating="POOR"),
                _row(ticker="B", combined_rating="EXCELLENT"),
                _row(ticker="C", combined_rating="MODERATE")]
        by_rating = sorted(rows, key=key, reverse=True)
        assert [r["ticker"] for r in by_rating] == ["B", "C", "A"]

    def test_entry_rows_sort_above_non_entry(self):
        app = _make_app()
        key = app._get_sort_key(4)
        rows = [_row(ticker="A", entry_signal=False), _row(ticker="B", entry_signal=True)]
        ordered = sorted(rows, key=key, reverse=True)
        assert ordered[0]["ticker"] == "B"

    def test_ma_column_prefers_fresh_crossover(self):
        """MA sort: fresh crossover > bullish > bearish."""
        app = _make_app()
        key = app._get_sort_key(6)
        rows = [
            _row(ticker="cross", ma_crossed_above=True),
            _row(ticker="bull", ma_bullish=True),
            _row(ticker="bear"),
        ]
        ordered = sorted(rows, key=key, reverse=True)
        assert [r["ticker"] for r in ordered] == ["cross", "bull", "bear"]


class TestMaText:
    def test_ma_text_labels(self):
        app = _make_app()
        assert app._ma_text(_row(ma_crossed_above=True, crossover_bars_ago=3,
                                 crossover_count=1)) == "^ X3"
        assert app._ma_text(_row(ma_crossed_above=True, crossover_bars_ago=3,
                                 crossover_count=2)) == "^ X3(2)"
        assert app._ma_text(_row(ma_bullish=True)) == "^ Bull"
        assert app._ma_text(_row()) == "v Bear"

    def test_ma_color_uses_theme(self):
        app = _make_app()
        c = app.theme_colors
        assert app._ma_color(_row(ma_crossed_above=True)) == c["green"]
        assert app._ma_color(_row(ma_bullish=True)) == c["lime"]
        assert app._ma_color(_row()) == c["red"]


class TestDataRowColoring:
    def test_score_coloring_and_threshold_highlight(self):
        app = _make_app()
        c = app.theme_colors
        above = app._make_data_row(_row(ticker="HIGH", total=72), 1, c, c["card"], 50)
        below = app._make_data_row(_row(ticker="LOW", total=20), 2, c, c["main_bg"], 50)

        # Ticker cell: green when above threshold, theme text otherwise
        assert _cell(above, 1).value == "HIGH"
        assert _cell(above, 1).color == c["green"]
        assert _cell(below, 1).color == c["text"]

        # Score cell: 70+ -> green
        assert _cell(above, 2).value == "72"
        assert _cell(above, 2).color == c["green"]
        # 20 -> red
        assert _cell(below, 2).value == "20"
        assert _cell(below, 2).color == c["red"]

    def test_rating_and_entry_cells(self):
        app = _make_app()
        c = app.theme_colors
        row = app._make_data_row(
            _row(ticker="HDFC", combined_rating="EXCELLENT", entry_signal=True,
                 is_sideways=False, trend_dir="Bull", pc1m=3.4),
            1, c, c["card"], 50,
        )
        assert _cell(row, 3).value == "EXCELLENT"
        assert _cell(row, 3).color == c["green"]
        assert _cell(row, 4).value == "YES"
        assert _cell(row, 4).color == c["green"]
        assert _cell(row, 14).value == "+3.4%"
        assert _cell(row, 17).value == "OK"

    def test_bearish_row_shows_down_arrow(self):
        app = _make_app()
        c = app.theme_colors
        row = app._make_data_row(_row(ticker="SBI"), 1, c, c["card"], 50)
        assert _cell(row, 15).value == "v Bear"


# ══════════════════════════════════════════════════════════════════════════════
# Layout mixin — top-pick cards and summary updates
# ══════════════════════════════════════════════════════════════════════════════


class TestTopPicks:
    def test_empty_state_message(self):
        app = _make_app()
        app.topicks_column = ft.Column(spacing=0)
        app._render_topicks([])
        msg = app.topicks_column.controls[0].content.value
        assert "Run a scan" in msg

    def test_leader_cards_show_rank_and_score(self):
        app = _make_app()
        app.topicks_column = ft.Column(spacing=0)
        tops = [_row(ticker=f"T{i}", total=90 - i) for i in range(6)]
        app._render_topicks(tops)
        assert len(app.topicks_column.controls) == 5  # capped at top 5
        row = app.topicks_column.controls[0].content  # ft.Row of the #1 card
        rank = row.controls[0].content.value
        ticker = row.controls[1].controls[0].value
        score = row.controls[2].value
        assert rank == "1" and ticker == "T0" and score == "90"


def test_summary_cards_update_values():
    app = _make_app()
    app.summary_cards = {k: ft.Text("—") for k in
                         ("total", "passed", "entry", "avg", "high",
                          "bull", "bear", "dead_skip")}
    app._update_summary([
        _row(ticker="A", total=70, entry_signal=True, trend_dir="Bull"),
        _row(ticker="B", total=40, trend_dir="Bull"),
        _row(ticker="C", total=30, trend_dir="Bear"),
    ])
    assert app.summary_cards["total"].value == "3"
    assert app.summary_cards["passed"].value == "1"  # 70 >= threshold 50
    assert app.summary_cards["entry"].value == "1"
    assert app.summary_cards["bull"].value == "2"
    assert app.summary_cards["bear"].value == "1"
    assert app.summary_cards["high"].value == "70"


def test_score_of_tolerates_missing_total():
    assert _score_of({}) == 0
    assert _score_of({"total": None}) == 0
    assert _score_of({"total": 55}) == 55


class TestHeroWarningRendering:
    """The results hero renders ScanResult.warnings amber under the summary."""

    class _FakeLabel:
        def __init__(self):
            self.value = self.color = self.size = None

    def _hero_ns(self, warnings):
        ns = type("NS", (), {})()
        ns.hero_sub = self._FakeLabel()
        ns.theme_colors = dict(THEMES["dark"])
        ns.scanning = False
        ns.settings = {"min_score": 50.0}
        ns.last_warnings = list(warnings)
        return ns

    def test_stale_member_warning_renders_amber(self):
        stale = "1 universe member(s) have stale data (last bar > 45d old): " \
                "GSPL (2026-05-11) — suspended/delisted?"
        ns = self._hero_ns([stale])

        ResultsViewMixin._update_hero_status(
            ns, [{"total": 55.0, "entry_signal": True, "trend_dir": "Bull"}]
        )

        assert "⚠" in ns.hero_sub.value
        assert "GSPL (2026-05-11)" in ns.hero_sub.value
        # warning text forces the amber small-font style under the summary line
        assert ns.hero_sub.color == ns.theme_colors["orange"]
        assert ns.hero_sub.size == 11

    def test_clean_scan_uses_hero_style(self):
        ns = self._hero_ns([])

        ResultsViewMixin._update_hero_status(
            ns, [{"total": 55.0, "entry_signal": True}]
        )

        assert "⚠" not in ns.hero_sub.value
        # The status line sits on the gradient hero, so it uses the tinted
        # hero_sub token (not the flat text_dim used elsewhere).
        assert ns.hero_sub.color == ns.theme_colors["hero_sub"]
        assert ns.hero_sub.size == 12


# ══════════════════════════════════════════════════════════════════════════════
# Sentiment badge + table-view persistence
# ══════════════════════════════════════════════════════════════════════════════


class TestSentimentBadge:
    def test_badge_shown_next_to_ticker_when_articles_exist(self):
        """Rows with enrichment data get an arrow+count pill; ticker stays first."""
        app = _make_app()
        c = app.theme_colors
        row = app._make_data_row(
            _row(ticker="TCS", _article_count=4, _sentiment_score=0.6),
            1, c, c["card"], 50,
        )
        content = row.content.controls[1].content
        assert isinstance(content, ft.Row)
        assert content.controls[0].value == "TCS"  # ticker text is first
        badge = content.controls[1]
        assert badge.content.controls[1].value == "4"
        assert badge.content.controls[0].value == "↑"  # positive tone
        assert "positive" in badge.tooltip

    def test_no_badge_without_article_count(self):
        """Keyless rows keep a plain ticker Text (existing layout preserved)."""
        app = _make_app()
        c = app.theme_colors
        row = app._make_data_row(_row(ticker="TCS"), 1, c, c["card"], 50)
        assert _cell(row, 1).value == "TCS"

    def test_badge_negative_tone(self):
        """Negative sentiment scores render a down arrow + red count."""
        app = _make_app()
        c = app.theme_colors
        row = app._make_data_row(
            _row(ticker="XYZ", _article_count=2, _sentiment_score=-0.8),
            1, c, c["card"], 50,
        )
        content = row.content.controls[1].content
        badge = content.controls[1]
        assert badge.content.controls[0].value == "↓"
        assert "negative" in badge.tooltip

    def test_news_scanner_finds_ticker_behind_badge(self):
        """_row_ticker_text resolves the ticker even when a badge Row wraps it."""
        app = _make_app()
        c = app.theme_colors
        row = app._make_data_row(
            _row(ticker="HDFC", _article_count=3, _sentiment_score=0.0),
            1, c, c["card"], 50,
        )
        ticker_text = app._row_ticker_text(row.content.controls[1])
        assert ticker_text.value == "HDFC"
        assert app._row_ticker_text(row.content.controls[0]) is None  # rank cell


class TestUiPrefsPersistence:
    def _app_with_pref_controls(self):
        app = _make_app()
        app.page_size_options = ["50", "100", "200", "500"]
        app.rating_filter_dd = type("DD", (), {"value": "All"})()
        app.page_size_dd = type("DD", (), {"value": "100"})()
        return app

    def test_save_ui_prefs_writes_sort_size_and_rating(self, monkeypatch):
        """Sort / page size / rating filter persist to settings.json."""
        app = self._app_with_pref_controls()
        app.sort_col = 18      # the 1M sparkline column
        app.sort_reverse = True
        app.page_size = 200
        app.rating_filter_dd.value = "Good"
        written = {}
        monkeypatch.setattr(store_mod, "save_settings", lambda s: written.update(s))

        app._save_ui_prefs()

        assert written["ui_sort_col"] == 18
        assert written["ui_sort_reverse"] is True
        assert written["ui_page_size"] == 200
        assert written["ui_rating_filter"] == "GOOD"

    def test_load_ui_prefs_restores_saved_view(self):
        """After a restart the saved sort/size/filter are re-applied."""
        app = self._app_with_pref_controls()
        app.settings = {**app_mod.DEFAULT_SETTINGS,
                        "ui_sort_col": 18, "ui_sort_reverse": True,
                        "ui_page_size": 200, "ui_rating_filter": "GOOD"}

        app._load_ui_prefs()

        assert app.sort_col == 18
        assert app.sort_reverse is True
        assert app.page_size == 200
        assert app.page_size_dd.value == "200"
        assert app.rating_filter_dd.value == "Good"

    def test_load_ui_prefs_ignores_out_of_range_sort_col(self):
        """A stale sort column (UI changed between versions) is not applied."""
        app = self._app_with_pref_controls()
        app.sort_col = None  # normally set by ScannerApp.__init__
        app.settings = {**app_mod.DEFAULT_SETTINGS, "ui_sort_col": 999}

        app._load_ui_prefs()

        assert app.sort_col is None


# ══════════════════════════════════════════════════════════════════════════════
# Error paths — deferred UI updates must not NameError on exception variables
# ══════════════════════════════════════════════════════════════════════════════
# The scan runner and the HTML exporter used to build their error log as
# ``lambda: self._log(f"...{e}")`` inside an ``except`` block.  Python clears
# the exception target when the block exits, so a UI update that runs after
# the handler returns (as _safe_update does when marshalled to the UI thread)
# would NameError instead of logging.  Both now bind the message to a plain
# local first; these tests force each failure and run the deferred callbacks
# AFTER the except scope is gone, so the old pattern would fail them.


class TestErrorPaths:
    def _deferred_app(self):
        app = _make_app()
        app._deferred = []
        app._safe_update = app._deferred.append  # run later, off the except scope
        return app

    @staticmethod
    def _flush(app):
        for fn in list(app._deferred):
            fn()

    def test_scan_failure_logs_error_with_deferred_update(self, monkeypatch):
        """A crashing scan logs \"ERROR: …\" even when the UI update is deferred."""
        import scanner.scanner_engine as se_mod
        app = self._deferred_app()
        app.universe_dd = type("DD", (), {"value": "NIFTY 50"})()
        app._scan_complete = lambda: None

        class _BoomEngine:
            def cancel(self):
                pass

            def set_progress_callback(self, *a):
                pass

            def set_log_callback(self, *a):
                pass

            def scan_stream(self, **k):
                raise RuntimeError("boom")

        monkeypatch.setattr(se_mod, "ScannerEngine", _BoomEngine)

        app._run_scan()
        self._flush(app)  # runs after the except block has exited

        assert any("ERROR: boom" in m for m in app.logged)

    def test_html_export_failure_logs_error_with_deferred_update(self, monkeypatch):
        """A crashing export logs the failure even with a deferred UI update."""
        import time
        app = self._deferred_app()
        app.results = [{"ticker": "TCS"}]
        app.universe_dd = type("DD", (), {"value": "NIFTY 50"})()

        def _boom(*a, **k):
            raise RuntimeError("export boom")
        monkeypatch.setattr(app_mod, "generate_html_report", _boom)

        app._export_html()

        # The export runs in a worker thread — wait for its except handler
        # to enqueue the deferred log before flushing.
        deadline = time.time() + 5
        while not app._deferred and time.time() < deadline:
            time.sleep(0.01)
        self._flush(app)

        assert any("HTML export failed: export boom" in m for m in app.logged)

    def test_backtest_failure_reports_error_with_deferred_update(self, monkeypatch):
        """A crashing walk-forward job reports its error after the except exits."""
        import scanner.walkforward as wf_mod
        app = self._deferred_app()
        app._wf_overrides = dict
        app.wf_state = {"gates": {}}  # normally seeded by _run_wf before threading

        def _boom(**k):
            raise RuntimeError("wf boom")
        monkeypatch.setattr(wf_mod, "run_walkforward", _boom)

        app._wf_job(years=3, stop=5, target=10, trail=2, min_adx=20,
                    ma_set="current")
        self._flush(app)  # runs _wf_done after the except scope is gone

        assert app.wf_state["error"] == "wf boom"
        assert any("Walk-forward failed: wf boom" in m for m in app.logged)

    def test_prune_price_cache_failure_is_logged(self, monkeypatch):
        """A failing prune logs the error (direct log path, no deferral)."""
        import scanner.data_providers as dp_mod
        app = self._deferred_app()

        def _boom(*a, **k):
            raise RuntimeError("prune boom")
        monkeypatch.setattr(dp_mod, "prune_stale_cache", _boom)

        app._prune_price_cache()

        assert any("Could not prune price cache: prune boom" in m for m in app.logged)

    def test_clear_negative_cache_failure_is_logged(self, monkeypatch):
        """A failing dead-symbol-cache clear logs the error."""
        import scanner.data_fetcher as df_mod
        app = self._deferred_app()
        monkeypatch.setattr(df_mod, "_negative_cache_load", dict)

        def _boom(*a, **k):
            raise RuntimeError("neg boom")
        monkeypatch.setattr(df_mod, "_negative_cache_update", _boom)

        app._clear_negative_cache()

        assert any("Could not clear dead-symbol cache: neg boom" in m for m in app.logged)

    def test_clear_enrichment_cache_failure_is_logged(self, monkeypatch):
        """A failing enrichment-cache clear logs the error."""
        import scanner.data_fetcher as df_mod
        app = self._deferred_app()

        def _boom(*a, **k):
            raise RuntimeError("enrich boom")
        monkeypatch.setattr(df_mod, "enrichment_cache_clear", _boom)

        app._clear_enrichment_cache()

        assert any("Could not clear enrichment cache: enrich boom" in m for m in app.logged)


# ══════════════════════════════════════════════════════════════════════════════
# Ticker click → inline news panel (loading placeholder / replace / collapse)
# ══════════════════════════════════════════════════════════════════════════════
# Regression guard: the "Loading news & sentiment…" placeholder used to carry
# the same ``_news_ticker`` marker as a real panel, so when a fetch finished
# ``_show_news`` mistook it for an already-open panel and collapsed it — the
# news (or the "No recent news found." message) never appeared.  Placeholders
# are flagged ``_news_loading`` and must be replaced, never collapsed.


class TestTickerNewsToggle:
    class _NoopThread:
        """Captures the fetch worker without running it (no network in tests)."""

        def __init__(self, *a, **k):
            self.target = k.get("target")

        def start(self):
            pass

    @staticmethod
    def _app_with_row(monkeypatch):
        import scanner.views_results as vr_mod
        monkeypatch.setattr(vr_mod.threading, "Thread", TestTickerNewsToggle._NoopThread)
        app = _make_app()
        app.table_column = ft.Column(spacing=0)
        c = app.theme_colors
        row = app._make_data_row(_row(ticker="TCS"), 1, c, c["card"], 50)
        app.table_column.controls.append(row)
        return app

    @staticmethod
    def _frames(app):
        return [x for x in app.table_column.controls if hasattr(x, "_news_ticker")]

    @staticmethod
    def _body_text(frame):
        """Concatenate the text values inside a news/loading frame."""
        out = []
        stack = [frame]
        seen = set()
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, ft.Text):
                out.append(node.value or "")
            children = getattr(node, "content", None)
            if isinstance(children, (list, tuple)):
                stack.extend(children)
            elif children is not None:
                stack.append(children)
            stack.extend(getattr(node, "controls", None) or [])
        return " ".join(out)

    def test_click_inserts_loading_placeholder(self, monkeypatch):
        """First click shows the "fetching…" placeholder under the row."""
        app = self._app_with_row(monkeypatch)

        app._toggle_stock_news("TCS")

        frames = self._frames(app)
        assert len(frames) == 1
        assert getattr(frames[0], "_news_loading", False) is True
        assert "Loading news & sentiment" in self._body_text(frames[0])

    def test_duplicate_clicks_while_loading_are_ignored(self, monkeypatch):
        """Clicking again while a fetch is in flight adds no second placeholder."""
        app = self._app_with_row(monkeypatch)

        app._toggle_stock_news("TCS")
        app._toggle_stock_news("TCS")

        assert len(self._frames(app)) == 1

    def test_finished_fetch_replaces_loading_with_news_panel(self, monkeypatch):
        """The completed fetch swaps the placeholder for the real news panel
        (previously it collapsed the placeholder and showed nothing)."""
        app = self._app_with_row(monkeypatch)
        app._toggle_stock_news("TCS")

        app._show_news("TCS", [{"title": "TCS beats estimates", "summary": "",
                                 "date": "2026-08-01", "provider": "Reuters",
                                 "sentiment": "Good"}])

        frames = self._frames(app)
        assert len(frames) == 1
        assert getattr(frames[0], "_news_loading", False) is False
        body = self._body_text(frames[0])
        assert "TCS beats estimates" in body
        assert "1 Good" in body and "0 Bad" in body
        assert "Loading news" not in body

    def test_empty_fetch_shows_no_news_message(self, monkeypatch):
        """A fetch that returns nothing shows the empty message (not nothing)."""
        app = self._app_with_row(monkeypatch)
        app._toggle_stock_news("TCS")

        app._show_news("TCS", [])

        frames = self._frames(app)
        assert len(frames) == 1
        assert "No recent news found" in self._body_text(frames[0])
        assert getattr(frames[0], "_news_loading", False) is False

    def test_second_click_on_open_panel_collapses_it(self, monkeypatch):
        """Clicking a ticker whose panel is open closes it again (no re-fetch)."""
        app = self._app_with_row(monkeypatch)
        app._show_news("TCS", [{"title": "T", "summary": "", "date": "",
                                 "provider": "", "sentiment": "Neutral"}])
        assert len(self._frames(app)) == 1

        app._toggle_stock_news("TCS")

        assert self._frames(app) == []


# ══════════════════════════════════════════════════════════════════════════════
# Scan-time news prefetch (top-50 attach) + instant click-to-read rows
# ══════════════════════════════════════════════════════════════════════════════


class TestNewsPrefetch:
    """app._prefetch_row_news attaches prefetched stories to the top rows."""

    @staticmethod
    def _app(rows):
        import threading
        app = _make_app()
        app._results_lock = threading.Lock()
        app.all_results = [dict(r) for r in rows]
        app.results = list(app.all_results)
        app._scan_epoch = 1
        app._scan_cancelled = False
        app._deferred = []
        app._safe_update = app._deferred.append  # record UI ops (never run them)
        return app

    @staticmethod
    def _sample_items(n=2):
        return [{"title": f"Story {i}", "summary": "", "date": "2026-08-01",
                 "provider": "Reuters", "sentiment": "Good"} for i in range(n)]

    def test_attaches_news_and_badge_fields_to_top_rows(self, monkeypatch):
        rows = [_row(ticker="A", total=90), _row(ticker="B", total=80),
                _row(ticker="C", total=20)]
        app = self._app(rows)
        def _mock_fetch(tickers, **_k):
            return {t: ([] if t == "C" else self._sample_items())
                    for t in tickers}
        monkeypatch.setattr(app_mod, "fetch_news_batch", _mock_fetch)

        app._prefetch_row_news()

        by_t = {r["ticker"]: r for r in app.all_results}
        assert len(by_t["A"]["_news_items"]) == 2
        assert by_t["A"]["_article_count"] == 2
        assert by_t["A"]["_sentiment_score"] == 1.0  # all Good
        assert len(by_t["B"]["_news_items"]) == 2
        assert by_t["C"]["_news_items"] == []  # attached even when empty
        assert "_article_count" not in by_t["C"]
        # The flush that repaints badges was scheduled on the UI thread.
        assert any(fn == app._flush_news_badges for fn in app._deferred)
        assert any("News prefetched for 3/3 top stocks" in m for m in app.logged)

    def test_keeps_existing_enrichment_counts(self, monkeypatch):
        row = _row(ticker="A", total=90, _article_count=7, _sentiment_score=0.5)
        app = self._app([row])
        monkeypatch.setattr(app_mod, "fetch_news_batch",
                            lambda tickers, **_k: {"A": self._sample_items()})

        app._prefetch_row_news()

        got = app.all_results[0]
        assert got["_article_count"] == 7      # provider enrichment wins
        assert got["_sentiment_score"] == 0.5
        assert len(got["_news_items"]) == 2    # stories still attached

    def test_skips_rows_that_already_have_stories(self, monkeypatch):
        row = _row(ticker="A", total=90)
        row["_news_items"] = []
        app = self._app([row])
        called = []
        monkeypatch.setattr(app_mod, "fetch_news_batch",
                            lambda *a, **k: called.append(1) or {})

        app._prefetch_row_news()

        assert called == []  # nothing left to prefetch — no fetch started

    def test_attach_dropped_when_newer_scan_started(self, monkeypatch):
        app = self._app([_row(ticker="A", total=90)])

        def _bump_epoch(tickers, **_k):
            app._scan_epoch = 2  # a new scan took over while we were fetching
            return {"A": self._sample_items()}
        monkeypatch.setattr(app_mod, "fetch_news_batch", _bump_epoch)

        app._prefetch_row_news()

        assert "_news_items" not in app.all_results[0]

    def test_attach_skipped_when_scan_cancelled(self, monkeypatch):
        app = self._app([_row(ticker="A", total=90)])
        app._scan_cancelled = True
        monkeypatch.setattr(app_mod, "fetch_news_batch",
                            lambda tickers, **_k: {"A": self._sample_items()})

        app._prefetch_row_news()

        assert "_news_items" not in app.all_results[0]


class TestPrefetchedRowClick:
    """Rows carrying prefetched _news_items open instantly, no fetch thread."""

    class _BoomThread:
        def __init__(self, *a, **k):
            raise AssertionError("prefetched rows must not start a fetch thread")

    @staticmethod
    def _app_with_prefetch(monkeypatch, items, **stats):
        app = _make_app()
        row = _row(ticker="TCS", **stats)
        row["_news_items"] = items
        app.all_results = [row]
        app.table_column = ft.Column(spacing=0)
        c = app.theme_colors
        app.table_column.controls.append(
            app._make_data_row(dict(row), 1, c, c["card"], 50)
        )
        import scanner.views_results as vr_mod
        monkeypatch.setattr(vr_mod.threading, "Thread",
                            TestPrefetchedRowClick._BoomThread)
        return app

    def _open_panel(self, monkeypatch, items, **stats):
        app = self._app_with_prefetch(monkeypatch, items, **stats)
        app._toggle_stock_news("TCS")
        frames = [x for x in app.table_column.controls
                  if hasattr(x, "_news_ticker")]
        assert len(frames) == 1
        return frames[0]

    def test_stories_render_instantly(self, monkeypatch):
        frame = self._open_panel(
            monkeypatch,
            [{"title": "TCS profit beat", "summary": "", "date": "2026-08-01",
              "provider": "Reuters", "sentiment": "Good"}],
        )
        body = TestTickerNewsToggle._body_text(frame)
        assert "TCS profit beat" in body
        assert "1 Good" in body and "0 Bad" in body
        assert "Loading news" not in body

    def test_stats_chips_in_panel_header(self, monkeypatch):
        frame = self._open_panel(
            monkeypatch,
            [{"title": "T", "summary": "", "date": "2026-08-01",
              "provider": "", "sentiment": "Neutral"}],
            close=2450.0, rsi=62.0, pc1m=4.2,
        )
        body = TestTickerNewsToggle._body_text(frame)
        assert "2,450" in body
        assert "RSI 62" in body
        assert "1M +4.2%" in body

    def test_no_stories_shows_empty_message_instantly(self, monkeypatch):
        frame = self._open_panel(monkeypatch, [])
        assert "No recent news found" in TestTickerNewsToggle._body_text(frame)
        assert getattr(frame, "_news_loading", False) is False
