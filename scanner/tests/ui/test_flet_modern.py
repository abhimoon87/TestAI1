"""Modern Flet chrome: fuzzy palette, view-stack switching, toasts, sidebar.

Covers the toolkit-free palette matcher plus the app wiring, driven through
the same ``_make_app`` mock harness as the view tests (``_safe_update``
stubbed to run inline, as the existing error-path tests do).
"""

import json
import threading

import scanner.backend.settings_store as store
from scanner.tests.ui.test_view_mixins import _make_app
from scanner.ui.ui_kit import (
    PaletteAction,
    filter_actions,
    fuzzy_score,
    parse_watchlist_text,
)


def _ready_app():
    """Mock app with a built UI and the runtime state view switches read."""
    app = _make_app()
    app.results = []
    app.all_results = []
    app.filter_text = ""
    app.page_size = 100
    app.current_page = 0
    app.sort_col = None
    app.sort_reverse = False
    # Production uses a re-entrant lock: _render_current_page holds it while
    # _visible_results re-acquires it. A plain Lock would deadlock here.
    app._results_lock = threading.RLock()
    app._safe_update = lambda fn: fn()
    app._build_ui()
    return app


def _visible_views(app):
    return [v for v in (app.dashboard_view, app.settings_view)
            if v.visible]


# ── Fuzzy matcher (pure, no page needed) ──────────────────────────────

def test_fuzzy_exact_and_prefix_rank_first():
    actions = [
        PaletteAction("go-dashboard", "Go to Dashboard", "Ctrl+1"),
        PaletteAction("run-stop", "Run scan", "Ctrl+R"),
        PaletteAction("export-html", "Export HTML report", "Ctrl+E"),
    ]
    assert filter_actions("go-dashboard", actions)[0].action_id == "go-dashboard"
    assert filter_actions("dash", actions)[0].action_id == "go-dashboard"


def test_fuzzy_no_match_returns_empty():
    actions = [PaletteAction("run-stop", "Run scan", "Ctrl+R")]
    assert filter_actions("zzzqqq", actions) == []


def test_fuzzy_empty_query_returns_all():
    actions = [PaletteAction("a", "A", ""), PaletteAction("b", "B", "")]
    assert len(filter_actions("", actions)) == 2


def test_fuzzy_score_none_on_missing_letter():
    assert fuzzy_score("zxq", "dashboard") is None
    assert fuzzy_score("dash", "dashboard") is not None
    assert fuzzy_score("DASH", "dashboard") == fuzzy_score("dash", "dashboard")


# ── View-stack switching (exactly one pane visible) ───────────────────

def test_build_shows_only_dashboard():
    app = _ready_app()
    assert [type(v).__name__ for v in _visible_views(app)] == ["Column"]
    assert app.dashboard_view.visible is True
    assert app.settings_view.visible is False


def test_show_view_toggles_exactly_one_pane():
    app = _ready_app()
    app._show_settings()
    assert app.dashboard_view.visible is False
    assert app.settings_view.visible is True
    app._show_view("dashboard")
    assert app.dashboard_view.visible is True
    assert app.settings_view.visible is False
    assert app._rail_pills["dashboard"].visible is True


# ── Palette dialog ────────────────────────────────────────────────────

def test_palette_opens_with_all_actions():
    app = _ready_app()
    app._open_palette()
    assert len(app.page.shown) == 1
    import flet as ft
    assert isinstance(app.page.shown[0], ft.AlertDialog)
    # List is capped at 8 of the 12 registered actions
    assert len(app._palette_state["shown"]) == 8
    assert len(app._palette_actions()) == 12


def test_palette_filters_on_typing():
    app = _ready_app()
    app._open_palette()
    app._palette_query.value = "scan"
    app._palette_query.on_change(None)
    shown = app._palette_state["shown"]
    assert shown and all(
        "scan" in (a.title + a.action_id).lower() for a in shown)
    assert shown[0].action_id == "run-stop"


def test_palette_action_runs_and_closes():
    app = _ready_app()
    app._open_palette()
    app._palette_query.value = "settings"
    app._palette_query.on_change(None)
    top = app._palette_state["shown"][0]
    assert top.action_id == "go-settings"
    app._run_palette_action(top)
    assert app.active_view == "settings"
    assert app.settings_view.visible is True
    assert app.page.popped == 1


def test_palette_actions_cover_run_stop():
    app = _ready_app()
    ids = [a.action_id for a in app._palette_actions()]
    assert "run-stop" in ids
    run = [a for a in app._palette_actions() if a.action_id == "run-stop"][0]
    assert run.title == "Run scan"  # idle state


# ── Toasts ────────────────────────────────────────────────────────────

def test_toast_sets_snackbar_and_updates():
    import flet as ft
    app = _ready_app()
    before = app.page.update_calls
    app._toast("Scan complete — 5 results", "success")
    assert len(app.page.shown) == 1
    bar = app.page.shown[0]
    assert isinstance(bar, ft.SnackBar)
    assert bar.content.value == "Scan complete — 5 results"
    assert bar.open is True
    assert app.page.update_calls > before


def test_toast_error_uses_white_text():
    app = _ready_app()
    app._toast("boom", "error")
    assert app.page.shown[0].content.color == "#ffffff"


# ── Sidebar collapse + keyboard ───────────────────────────────────────

def test_sidebar_toggle_flips_visibility():
    app = _ready_app()
    assert app.sidebar.visible is not False
    app._toggle_sidebar()
    assert app.sidebar.visible is False
    app._toggle_sidebar()
    assert app.sidebar.visible is True


def test_collapse_survives_ui_rebuild():
    app = _ready_app()
    app._toggle_sidebar()
    assert app.sidebar.visible is False
    app._build_ui()  # theme-switch-style rebuild
    assert app.sidebar.visible is False


def test_keyboard_routes_shortcuts():
    app = _ready_app()

    class _Key:
        def __init__(self, key, ctrl=False):
            self.key = key
            self.ctrl = ctrl

    app._on_keyboard_event(_Key("2", ctrl=True))
    assert app.active_view == "settings"
    app._on_keyboard_event(_Key("1", ctrl=True))
    assert app.active_view == "dashboard"
    # Dark-only UI: Ctrl+T is unbound and changes nothing.
    app._on_keyboard_event(_Key("T", ctrl=True))
    assert app.current_theme == "dark"
    app._on_keyboard_event(_Key("/", ctrl=False))
    assert app.active_view == "dashboard"  # stray keys never crash
    app._on_keyboard_event(_Key("F1", ctrl=False))


def test_keyboard_run_stop_invokes_action(monkeypatch):
    app = _ready_app()
    calls = []
    monkeypatch.setattr(app, "_on_action_click", lambda *a, **k: calls.append(1))

    class _Key:
        def __init__(self, key, ctrl=False):
            self.key = key
            self.ctrl = ctrl

    app._on_keyboard_event(_Key("R", ctrl=True))
    assert calls == [1]


def test_scroll_to_top_without_loop_warns_nothing():
    """scroll_to() is a coroutine: without a running loop it must close the
    coroutine instead of dropping it (no RuntimeWarning), and never raise."""
    import warnings
    app = _ready_app()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        app._scroll_to_top()  # main_scroll built; no event loop here
    # Missing scroll area is also fine (early-init guards).
    app.main_scroll = None
    app._scroll_to_top()


def test_dark_is_the_only_theme():
    from scanner.shared.themes import THEMES
    assert set(THEMES) == {"dark"}


def test_saved_light_theme_self_heals_to_dark(tmp_path, monkeypatch):
    cfg = tmp_path / "settings.json"
    cfg.write_text(json.dumps({"theme": "light"}), encoding="utf-8")
    monkeypatch.setattr(store, "SETTINGS_FILE", str(cfg))
    assert store.load_settings()["theme"] == "dark"


def test_stream_batch_multi_removal_keeps_grid_updating():
    """Several tickers dropping out of the filter in ONE batch must remove
    exactly those rows — the old mid-loop pop() shifted indices, deleting
    the wrong rows or raising IndexError (which the engine swallows, so the
    grid silently never updated)."""
    app = _ready_app()
    rendered = []
    app._render_current_page = lambda: rendered.append(1)
    app.filter_text = "A"  # only AAA matches; BBB/CCC/DDD drop out
    rows = [{"ticker": t, "total": 60.0} for t in ("AAA", "BBB", "CCC", "DDD")]
    app.all_results = [dict(r) for r in rows]
    app.filtered_results = [dict(r) for r in rows]
    batch = [{"ticker": t, "total": 61.0} for t in ("AAA", "BBB", "CCC", "DDD")]

    app._on_stream_batch(batch)  # must not raise

    assert [r["ticker"] for r in app.all_results] == ["AAA", "BBB", "CCC", "DDD"]
    assert [r["ticker"] for r in app.filtered_results] == ["AAA"]
    assert app.filtered_results[0]["total"] == 61.0  # winner replaced in place
    assert rendered  # grid refresh was still requested


# ── Watchlist parsing (pure) ──────────────────────────────────────────

def test_parse_plain_and_csv_lists():
    text = "Ticker,Price\nRELIANCE,2500\ntcs\nINFY.NS\n\n"
    assert parse_watchlist_text(text) == ["RELIANCE", "TCS", "INFY"]


def test_parse_dedupes_and_skips_garbage():
    text = "RELIANCE\nRELIANCE\n!!!\n X \nM&M\nHDFC-BANK\n"
    # Single letters pass (permissive on purpose — the engine simply finds
    # no data for unknown symbols); only junk is dropped.
    assert parse_watchlist_text(text) == ["RELIANCE", "X", "M&M", "HDFC-BANK"]


def test_parse_empty_and_limit():
    assert parse_watchlist_text("") == []
    assert parse_watchlist_text("AAA\nBBB\nCCC", limit=2) == ["AAA", "BBB"]


# ── Copy ticker + watchlist import (1.0 client actions) ───────────────

def test_news_stats_row_ends_with_copy_action():
    import flet as ft
    from flet.controls.context import _context_page

    class _FakeServices:
        def __init__(self):
            self.registered = []

        def register_service(self, svc):
            self.registered.append(svc)

    app = _ready_app()
    # Client actions resolve per-page services through Flet's page context
    # (as production click handlers provide it); the stub services registry
    # stands in for the real page's.
    app.page._services = _FakeServices()
    token = _context_page.set(app.page)
    try:
        row = app._news_stats_row(
            {"ticker": "TCS", "close": 2450.0, "rsi": 62.0, "pc1m": 4.2})
    finally:
        _context_page.reset(token)
    assert isinstance(row, ft.Row)
    btn = row.controls[-1]
    assert isinstance(btn, ft.IconButton)
    assert btn.tooltip == "Copy TCS"
    assert isinstance(btn.action, ft.CopyToClipboard)
    assert btn.action.data == "TCS"


def test_news_stats_row_omits_copy_without_page_context():
    import flet as ft
    # Background flushes have no page context — chips render, button omitted.
    app = _ready_app()
    row = app._news_stats_row(
        {"ticker": "TCS", "close": 2450.0, "rsi": 62.0, "pc1m": 4.2})
    assert isinstance(row, ft.Row)
    assert len(row.controls) == 3  # chips only, no copy button
    assert not any(isinstance(c, ft.IconButton) for c in row.controls)



def test_watchlist_import_registers_universe(tmp_path):
    from types import SimpleNamespace

    import flet as ft

    from scanner.shared.universes import UNIVERSES
    app = _ready_app()
    assert isinstance(app.file_picker, ft.FilePicker)
    # Services register on page.services — never the overlay/control tree
    # (a Service there makes the client show "Unknown control").
    assert app.file_picker in app.page.services
    csv_file = tmp_path / "list.csv"
    csv_file.write_text("Ticker,Price\nRELIANCE,2500\ntcs\nINFY.NS\n",
                        encoding="utf-8")
    event = SimpleNamespace(files=[SimpleNamespace(path=str(csv_file))])
    try:
        app._on_watchlist_picked(event)
        assert UNIVERSES["WATCHLIST (3)"] == ["RELIANCE", "TCS", "INFY"]
        keys = [getattr(o, "key", None) for o in app.universe_dd.options]
        assert "WATCHLIST (3)" in keys
        assert app.universe_dd.value == "WATCHLIST (3)"
    finally:
        UNIVERSES.pop("WATCHLIST (3)", None)


def test_watchlist_import_rejects_empty_file(tmp_path):
    from types import SimpleNamespace
    app = _ready_app()
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("Ticker\n", encoding="utf-8")
    before = len(app.page.shown)
    app._on_watchlist_picked(
        SimpleNamespace(files=[SimpleNamespace(path=str(csv_file))]))
    assert len(app.page.shown) == before + 1  # error toast shown
    assert "WATCHLIST (" not in str(getattr(app.universe_dd, "value", ""))
