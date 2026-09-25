"""Shared Flet test harness for the UI test package.

Un-initialized ScannerApp shells + recorder stand-ins so view builders run
without a window. Import from here — do not re-define in each test module.
"""

import asyncio
import concurrent.futures
import threading

import flet as ft

import scanner.ui.app as app_mod
from scanner.shared.themes import THEMES
from scanner.ui.app import ScannerApp


class FakeLabel:
    """Recorder stand-in for a Flet Text control (uses ``.value``)."""

    def __init__(self):
        self.value = None


class FakeButton:
    """Recorder stand-in for a Flet button (uses ``.visible`` / ``.disabled``)."""

    def __init__(self, disabled=False, visible=True):
        self.visible = visible
        self.disabled = disabled
        self.bgcolor = None


class FakePage:
    """Stub flet.Page recording adds/updates and dialog show/pop."""

    def __init__(self):
        self.controls = []
        self.window = type("W", (), {})()
        self.update_calls = 0
        self.shown = []
        self.popped = 0
        self.services = []

    def add(self, control):
        self.controls.append(control)

    def update(self):
        self.update_calls += 1

    def show_dialog(self, control):
        self.shown.append(control)

    def pop_dialog(self):
        self.popped += 1

    def run_task(self, handler):
        """Run async coroutines synchronously in the test harness."""
        coro = handler()
        if asyncio.iscoroutine(coro):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(asyncio.run, coro).result(timeout=2)
            else:
                asyncio.run(coro)


def make_app():
    """Un-initialized ScannerApp with the state view builders read."""
    app = ScannerApp.__new__(ScannerApp)
    app.page = FakePage()
    app.current_theme = "dark"
    app.theme_colors = THEMES["dark"]
    app.settings = dict(app_mod.DEFAULT_SETTINGS)
    app.active_view = "dashboard"
    app.scanning = False
    app._scan_lock = threading.Lock()
    app._results_lock = threading.RLock()
    app._state_lock = threading.Lock()
    app._scan_epoch = 0
    app.logged = []
    app._log = app.logged.append

    # Result / view state normally set in ScannerApp.__init__
    app.results = []
    app.all_results = []
    app.filtered_results = []
    app._row_pool = {}
    app._row_cells = {}
    app.filter_text = ""
    app.sort_col = None
    app.sort_reverse = False
    app.page_size = 100
    app.current_page = 0
    app._scan_cancelled = False
    app._last_warnings = []
    app._last_stream_render = 0.0

    # Dropdown / pagination stubs
    app.rating_filter_dd = type("DD", (), {"value": "All"})()
    app.page_size_dd = type("DD", (), {"value": "100"})()
    app.page_size_options = ["50", "100", "200", "500"]

    # Lightweight controls read by _render_current_page / _scan_complete
    app.table_column = ft.Column(spacing=0)
    app.header_holder = ft.Column(spacing=0)
    app.empty_label = ft.Container(visible=False)
    app.pagination_bar = ft.Container(visible=False)
    app.pagination_row = ft.Row(visible=False)
    app.page_label = ft.Text("Page 1 / 1")
    app.result_count_label = ft.Text("no scan yet")
    app.summary_cards = {
        k: ft.Text("—")
        for k in (
            "avg",
            "high",
            "bull",
            "bear",
        )
    }
    app.hero_sub = ft.Text("")
    app.topicks_column = ft.Column(spacing=0)
    app.chart_holder = ft.Container(visible=False)
    app.chart_bars = ft.Row(spacing=2)
    app.chart_sub = ft.Text("")
    app.progress_label = ft.Text("Ready")
    app.status_label = ft.Text("Status: Ready")
    app.progress_bar = ft.ProgressBar(value=0)
    app.action_btn = FakeButton(disabled=False)
    app.action_btn_label = ft.Text("RUN")
    app.html_btn = FakeButton(disabled=True)
    app.csv_btn = FakeButton(disabled=True)
    app.clear_btn = FakeButton(disabled=True)

    # Cache UI stubs
    app.cache_status_lbl = FakeLabel()
    app.cache_clear_btn = FakeButton()
    app.enrich_cache_status_lbl = FakeLabel()
    app.enrich_cache_clear_btn = FakeButton()
    app.price_cache_status_lbl = FakeLabel()
    app.price_cache_prune_btn = FakeButton()

    # Log panel
    app.log_column = type("C", (), {"controls": []})()

    # Run deferred UI work inline (no Flet event loop in unit tests)
    app._safe_update = lambda fn: fn()
    # Don't spin real background work from _scan_complete in tests
    app._warm_market = lambda: None
    app._news_prefetcher = type("P", (), {"start": lambda self, *a, **k: None})()
    return app


# Back-compat aliases used by existing test modules
_FakePage = FakePage
_FakeLabel = FakeLabel
_FakeButton = FakeButton
_make_app = make_app
