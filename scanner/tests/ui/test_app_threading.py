"""Regression tests for ScannerApp UI-thread marshalling.

Flet only patches the client reliably when controls are mutated on the
thread that owns the page's asyncio event loop.  ``_safe_update`` /
``_run_on_ui_thread`` must therefore move worker-thread UI work onto that
loop (``call_soon_threadsafe``) instead of running it in the caller's
thread — otherwise background scan results stay stuck on the placeholder
until the next user interaction forces a redraw.

These tests drive the real asyncio loop machinery (no mocks of the loop),
so a regression that runs the work on the worker thread fails outright.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

from scanner.ui.app import _UI_LOCK_FACTORY, ScannerApp


def _make_app(loop):
    """A ScannerApp shell with a fake page bound to a real asyncio loop."""
    app = object.__new__(ScannerApp)
    app._ui_lock = _UI_LOCK_FACTORY()  # production lock semantics
    calls = {"fn": [], "updates": []}

    class FakePage:
        def __init__(self):
            self.session = SimpleNamespace(connection=SimpleNamespace(loop=loop))

        def update(self):
            calls["updates"].append(threading.get_ident())

    app.page = FakePage()
    return app, calls


def _wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestSafeUpdateThreadMarshalling:
    def test_worker_thread_work_runs_on_loop_thread(self):
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        try:
            _wait_until(loop.is_running)
            app, calls = _make_app(loop)
            loop_tid = loop._thread_id

            app._safe_update(lambda: calls["fn"].append(threading.get_ident()))

            # The callback must have been marshalled onto the loop thread —
            # never run (or half-run) on the calling worker thread.
            assert _wait_until(lambda: calls["fn"] and calls["updates"])
            assert calls["fn"] == [loop_tid]
            assert calls["updates"] == [loop_tid]
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5)

    def test_safe_update_returns_without_waiting_for_loop(self):
        """Worker threads must not block: _safe_update queues and returns."""
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        try:
            _wait_until(loop.is_running)
            app, calls = _make_app(loop)

            # A slow fn must NOT run on the calling thread (that would make
            # _safe_update block for its full duration).
            def slow_fn():
                time.sleep(0.3)
                calls["fn"].append(threading.get_ident())

            start = time.time()
            app._safe_update(slow_fn)
            assert time.time() - start < 0.1  # queued, not executed inline
            assert calls["fn"] == []  # not run yet on the caller thread
            assert _wait_until(lambda: calls["fn"])  # ...but runs on the loop
            assert calls["fn"] == [loop._thread_id]
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5)

    def test_no_loop_falls_back_to_inline(self):
        """Unit-test / headless paths (no live page loop) run inline."""
        app, calls = _make_app(None)
        app.page.session = SimpleNamespace(connection=SimpleNamespace(loop=None))
        app._safe_update(lambda: calls["fn"].append(threading.get_ident()))
        assert calls["fn"] == [threading.get_ident()]
        assert calls["updates"] == [threading.get_ident()]

    def test_loop_thread_calls_run_inline(self):
        """Event handlers already on the loop thread are not re-queued."""
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        try:
            _wait_until(loop.is_running)
            app, calls = _make_app(loop)
            done = {"ran": False}

            def on_loop_thread():
                app._safe_update(lambda: calls["fn"].append(threading.get_ident()))
                done["ran"] = True

            loop.call_soon_threadsafe(on_loop_thread)
            assert _wait_until(lambda: done["ran"])
            assert _wait_until(lambda: calls["fn"])
            assert calls["fn"] == [loop._thread_id]
            assert calls["updates"] == [loop._thread_id]
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5)
class TestNestedSafeUpdateReentrancy:
    """``_safe_update`` wrappers must be able to re-enter ``_safe_update``.

    ``_safe_update`` runs the wrapped fn while holding ``_ui_lock``, and its
    fns do call back into it — ``_scan_complete`` presents a SnackBar through
    ``_toast``, which is itself a ``_safe_update`` call.  On the page's UI
    thread ``_run_on_ui_thread`` executes that nested call inline, so a
    non-reentrant lock self-deadlocks the loop thread and the app freezes
    right after a scan finishes (the engine's ``scan_stream ok`` line is the
    last thing ever logged).
    """

    def test_ui_lock_is_reentrant(self):
        """A second acquisition on the same thread must not block."""
        lock = _UI_LOCK_FACTORY()
        assert lock.acquire(timeout=1)
        try:
            # A plain ``threading.Lock`` returns False here (times out);
            # the re-entrant factory must succeed immediately.
            assert lock.acquire(timeout=1)
            lock.release()
        finally:
            lock.release()

    def test_nested_safe_update_completes(self):
        """A nested _safe_update inside a wrapped fn must not deadlock."""
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        try:
            _wait_until(loop.is_running)
            app, calls = _make_app(loop)
            done = {"outer": False, "inner": False}

            def inner():
                calls["fn"].append("inner")
                done["inner"] = True

            def outer():
                # Mirrors _scan_complete -> _toast -> _safe_update: re-enter
                # while the outer _apply still holds _ui_lock.
                calls["fn"].append("outer")
                app._safe_update(inner)
                done["outer"] = True

            app._safe_update(outer)  # worker-thread call, marshalled to loop

            assert _wait_until(lambda: done["outer"] and done["inner"]), (
                "nested _safe_update deadlocked the UI loop thread"
            )
            assert calls["fn"] == ["outer", "inner"]
            # Both the inner and outer frames ran on the loop thread.
            assert calls["updates"]
            assert all(tid == loop._thread_id for tid in calls["updates"])
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5)

