"""Scan-history browser (F5): palette action → dialog → restore into grid.

Harness lives in ``tests/ui/conftest.py`` (``make_app`` / ``FakePage``).
"""

import flet as ft

from scanner.backend import settings_store

from .conftest import make_app as _make_app


def test_palette_includes_open_scan_history():
    app = _make_app()
    ids = [a.action_id for a in app._palette_actions()]
    assert "open-scan-history" in ids


def test_open_scan_history_shows_dialog_newest_first(monkeypatch):
    monkeypatch.setattr(
        settings_store,
        "list_scans",
        lambda limit=50: [
            {"id": 7, "created_at": "2026-11-10T09:15:33", "rows": 42},
            {"id": 3, "created_at": "2026-11-01T18:00:01", "rows": 10},
        ],
    )
    app = _make_app()

    app._open_scan_history()

    assert len(app.page.shown) == 1
    dlg = app.page.shown[0]
    assert isinstance(dlg, ft.AlertDialog)
    dd = dlg.content
    assert [o.key for o in dd.options] == ["7", "3"]  # newest first
    assert dd.value == "7"  # newest preselected
    assert "42 rows" in dd.options[0].text


def test_open_scan_history_empty_history_toasts(monkeypatch):
    monkeypatch.setattr(settings_store, "list_scans", lambda limit=50: [])
    app = _make_app()

    app._open_scan_history()

    assert app.page.shown  # snackbar shown…
    assert not isinstance(app.page.shown[0], ft.AlertDialog)  # …not a dialog


def test_open_chosen_scan_restores_rows(monkeypatch):
    seen = {}

    def _load(scan_id=None):
        seen["scan_id"] = scan_id
        return [{"ticker": "OLD", "total": 55.0}]

    monkeypatch.setattr(settings_store, "load_results", _load)
    app = _make_app()
    views = []
    monkeypatch.setattr(app, "_show_view", lambda name: views.append(name))

    app._open_chosen_scan(ft.AlertDialog(), "7")

    assert seen["scan_id"] == 7  # key coerced back to int
    assert views == ["dashboard"]
    assert app.results == [{"ticker": "OLD", "total": 55.0}]
    assert app.html_btn.disabled is False
    assert app.csv_btn.disabled is False
    assert app.page.popped == 1  # dialog closed


def test_open_chosen_scan_empty_rows_toasts_without_switching(monkeypatch):
    monkeypatch.setattr(settings_store, "load_results", lambda scan_id=None: [])
    app = _make_app()
    views = []
    monkeypatch.setattr(app, "_show_view", lambda name: views.append(name))

    app._open_chosen_scan(ft.AlertDialog(), "7")

    assert views == []  # no view switch on a failed load
    assert app.page.shown  # "no rows" toast
