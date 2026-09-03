"""Unit tests for the GUI enrichment-cache status/clear handlers in scanner.app.

The handlers are exercised without a display by creating an un-initialized
ScannerApp (``__new__``) and attaching recorder stand-ins for the Flet label,
button and page, so no window or event loop is required.
"""

import scanner.data_fetcher as data_fetcher
from scanner.app import ScannerApp


class _FakeLabel:
    """Recorder stand-in for a Flet Text control (uses ``.value``)."""

    def __init__(self):
        self.value = None


class _FakeButton:
    """Recorder stand-in for a Flet button (uses ``.visible``)."""

    def __init__(self):
        self.visible = True


class _FakePage:
    """Recorder stand-in for a Flet Page (tracks ``update()`` calls)."""

    def __init__(self):
        self.update_calls = 0

    def update(self):
        self.update_calls += 1


def _make_app():
    """Un-initialized ScannerApp with recorder widgets + a _log hook."""
    app = ScannerApp.__new__(ScannerApp)
    app.enrich_cache_status_lbl = _FakeLabel()
    app.enrich_cache_clear_btn = _FakeButton()
    app.page = _FakePage()
    app.logged = []
    app._log = app.logged.append
    return app


def _populated_text(n):
    return (
        f"Enrichment cache: {n} "
        f"(auto-resets ~{data_fetcher.ENRICHMENT_CACHE_TTL_HOURS}h)"
    )


def test_refresh_populated_shows_count_and_reveals_clear(monkeypatch):
    monkeypatch.setattr(data_fetcher, "enrichment_cache_size", lambda: 3)
    app = _make_app()

    app._refresh_enrich_cache_ui()

    assert app.enrich_cache_status_lbl.value == _populated_text(3)
    assert app.enrich_cache_clear_btn.visible is True


def test_refresh_empty_hides_clear_button(monkeypatch):
    monkeypatch.setattr(data_fetcher, "enrichment_cache_size", lambda: 0)
    app = _make_app()

    app._refresh_enrich_cache_ui()

    assert app.enrich_cache_status_lbl.value == "Enrichment cache: empty"
    assert app.enrich_cache_clear_btn.visible is False


def test_refresh_before_sidebar_built_is_noop(monkeypatch):
    """Early-startup call (no widgets yet) must not raise or touch data_fetcher."""

    def boom():
        raise AssertionError("data_fetcher must not be touched before sidebar exists")

    monkeypatch.setattr(data_fetcher, "enrichment_cache_size", boom)

    class _NoSidebarYet:
        pass

    ScannerApp._refresh_enrich_cache_ui(_NoSidebarYet())  # guard returns first


def test_refresh_falls_back_to_empty_when_cache_unreadable(monkeypatch):
    def boom():
        raise RuntimeError("cache corrupt")

    monkeypatch.setattr(data_fetcher, "enrichment_cache_size", boom)
    app = _make_app()

    app._refresh_enrich_cache_ui()

    assert app.enrich_cache_status_lbl.value == "Enrichment cache: empty"
    assert app.enrich_cache_clear_btn.visible is False


def test_clear_wipes_real_cache_and_refreshes(tmp_path, monkeypatch):
    """End-to-end: seed the real (tmp-isolated) cache, clear via the handler."""
    monkeypatch.setattr(
        data_fetcher, "_ENRICHMENT_CACHE_PATH",
        str(tmp_path / "enrichment_cache.json"),
    )
    monkeypatch.setattr(data_fetcher, "_enrichment_cache", None)
    data_fetcher._enrichment_cache_put("RELIANCE", {"sentiment": {"score": 0.8}}, {"pe": 21.0})
    data_fetcher._enrichment_cache_put("TCS", {"social": {"hits": 5}}, None)
    assert data_fetcher.enrichment_cache_size() == 2

    app = _make_app()
    app._refresh_enrich_cache_ui()
    assert app.enrich_cache_status_lbl.value == _populated_text(2)
    assert app.enrich_cache_clear_btn.visible is True

    app._clear_enrichment_cache()

    assert data_fetcher.enrichment_cache_size() == 0
    assert app.logged == ["Cleared enrichment cache — next scan will re-fetch phase-2 data"]
    # UI refreshed to the empty state and flushed to the page
    assert app.enrich_cache_status_lbl.value == "Enrichment cache: empty"
    assert app.enrich_cache_clear_btn.visible is False
    assert app.page.update_calls >= 1


def test_clear_error_is_logged_and_ui_still_refreshes(monkeypatch):
    def boom():
        raise OSError("disk full")

    monkeypatch.setattr(data_fetcher, "enrichment_cache_clear", boom)
    monkeypatch.setattr(data_fetcher, "enrichment_cache_size", lambda: 5)
    app = _make_app()

    app._clear_enrichment_cache()  # must not raise

    assert len(app.logged) == 1
    assert app.logged[0].startswith("Could not clear enrichment cache: disk full")
    # Refresh still ran with the (patched) populated cache
    assert app.enrich_cache_status_lbl.value == _populated_text(5)
    assert app.enrich_cache_clear_btn.visible is True
