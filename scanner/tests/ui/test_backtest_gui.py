"""Backtest-from-GUI (F4): subprocess run, streamed log, report opened.

Harness lives in ``tests/ui/conftest.py`` (``make_app`` / ``wait_until``).
"""

import subprocess

import scanner.ui.app as app_mod

from .conftest import make_app as _make_app
from .conftest import wait_until


def _fake_popen(monkeypatch, html_path, lines, rc=0, write_html=True):
    """Replace subprocess.Popen with a stub that 'runs' the backtest."""
    calls = {}

    def fake(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        if write_html:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write("<html>report</html>")

        class Proc:
            stdout = iter(lines)

            def wait(self):
                return rc

        return Proc()

    monkeypatch.setattr(subprocess, "Popen", fake)
    return calls


def test_palette_includes_run_backtest():
    app = _make_app()
    ids = [a.action_id for a in app._palette_actions()]
    assert "run-backtest" in ids


def test_backtest_success_opens_report(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "REPORTS_DIR", str(tmp_path))
    opened = []
    monkeypatch.setattr(app_mod.webbrowser, "open", lambda url: opened.append(url))
    html_path = str(tmp_path / "backtest_report.html")
    calls = _fake_popen(monkeypatch, html_path, ["line one", "line two"])
    app = _make_app()

    app._run_backtest()

    assert wait_until(lambda: not app.backtest_running)
    cmd = calls["cmd"]
    assert cmd[1:3] == ["-m", "scanner.backend.backtest"]
    assert html_path in cmd
    assert str(tmp_path / "backtest_trades.csv") in cmd
    assert calls["kwargs"]["cwd"] == app_mod._PROJECT_ROOT
    assert any("line one" in m for m in app.logged)  # stdout streamed to log
    assert opened and opened[0].startswith("file://")
    assert any("Backtest done" in m for m in app.logged)


def test_backtest_no_trades_never_opens_stale_report(monkeypatch, tmp_path):
    """A 0-trade run writes no HTML — the pre-deleted stale file must stay gone."""
    monkeypatch.setattr(app_mod, "REPORTS_DIR", str(tmp_path))
    opened = []
    monkeypatch.setattr(app_mod.webbrowser, "open", lambda url: opened.append(url))
    html_path = tmp_path / "backtest_report.html"
    html_path.write_text("<stale>", encoding="utf-8")
    _fake_popen(monkeypatch, str(html_path), ["no trades"], write_html=False)
    app = _make_app()

    app._run_backtest()

    assert wait_until(lambda: not app.backtest_running)
    assert opened == []
    assert not html_path.exists()
    assert any("no trades" in m for m in app.logged)


def test_backtest_nonzero_exit_reports_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "REPORTS_DIR", str(tmp_path))
    opened = []
    monkeypatch.setattr(app_mod.webbrowser, "open", lambda url: opened.append(url))
    html_path = str(tmp_path / "backtest_report.html")
    _fake_popen(monkeypatch, html_path, ["boom"], rc=3, write_html=False)
    app = _make_app()

    app._run_backtest()

    assert wait_until(lambda: not app.backtest_running)
    assert opened == []
    assert any("exit 3" in m for m in app.logged)


def test_backtest_setup_failure_resets_running_flag(monkeypatch, tmp_path):
    """A failure before the subprocess starts must not wedge the command."""
    monkeypatch.setattr(app_mod, "REPORTS_DIR", str(tmp_path))
    app = _make_app()

    def boom(*a, **kw):
        raise OSError("denied")

    monkeypatch.setattr(app_mod.os, "makedirs", boom)

    app._run_backtest()

    assert wait_until(lambda: not app.backtest_running)
    assert any("Backtest failed" in m for m in app.logged)


def test_backtest_reentrancy_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "REPORTS_DIR", str(tmp_path))
    started = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: started.append(1) or None
    )
    app = _make_app()
    app.backtest_running = True

    app._run_backtest()

    assert started == []  # second run never launches
    assert app.backtest_running is True  # flag not clobbered by the first caller
