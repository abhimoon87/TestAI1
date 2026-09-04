"""Walk-forward backtest page -- ``BacktestViewMixin`` for ``scanner.app.ScannerApp``.

Runs the full/train/test walk-forward protocol (see ``scanner.walkforward``)
against the *current* indicator settings, so a parameter set can be validated
out-of-sample before it is saved.  The engine simulation mirrors the scanner's
MA types/lengths, crossover lookback and ADX entry gate via ``ma_overrides``;
only the risk parameters (stop / target / trail) and gate toggles are edited
on this page.

The heavy lifting runs on a daemon thread and results are pushed back with
``_safe_update``, mirroring ``_run_scan``.  All methods are mixins that run
against the ``ScannerApp`` instance and rely on ``self.settings``,
``self.theme_colors``, ``self.active_view``, ``self.main_area_box`` and
``self._log``.
"""

import threading

import flet as ft

from .ui_kit import (
    _card_shadow,
    _glass_bg,
    _glass_border,
    _padding_only,
)

# Scanner keys mirrored into the backtest engine so the simulation validates
# the settings the user is about to save (engine defaults cover the rest).
_ENGINE_KEYS = (
    "fast_ma_type", "fast_ma_len", "slow_ma_type", "slow_ma_len",
    "crossover_lookback", "min_adx_entry", "adx_len", "adx_threshold",
    "chop_len", "chop_threshold", "slope_ma_type", "slope_ma_len",
    "slope_lookback", "flat_threshold", "rs_length", "vol_ma_len", "atr_len",
    "vp_lookback", "sideways_strong_move_pct", "volume_participation_len",
)

# The MA set the 2024-26 out-of-sample findings were measured on.  Reference vs
# current is exposed as a per-run toggle because the saved loose 20x40 filter
# screens broadly but backtests far worse than 40x50.
_MA_REFERENCE = {"fast_ma_type": "HMA", "fast_ma_len": 40,
                 "slow_ma_type": "EMA", "slow_ma_len": 50,
                 "crossover_lookback": 20}

# Config presets (risk + ADX gate + which MA set to run them on).  These are
# documented OOS results for NIFTY 50 on the reference MA set, NOT guarantees:
# the same configs were negative on BANK NIFTY and on the loose 20x40 filter.
_WF_PRESETS = {
    "Custom": None,
    "S5/T8 + ADX20 (OOS best)": {"stop": 5, "target": 8, "trail": 2,
                                  "min_adx": 20, "ma": "reference"},
    "S2/T10 + ADX20": {"stop": 2, "target": 10, "trail": 2,
                        "min_adx": 20, "ma": "reference"},
    "S3/T12 + ADX20": {"stop": 3, "target": 12, "trail": 2,
                        "min_adx": 20, "ma": "reference"},
    "Default 2/20 + ADX20": {"stop": 2, "target": 20, "trail": 2,
                              "min_adx": 20, "ma": "reference"},
}
_MA_SET_OPTIONS = [("current", "Current scanner settings"),
                   ("reference", "Reference 40x50 (OOS findings)")]


class BacktestViewMixin:
    def _wf_overrides(self) -> dict:
        """Engine-usable copy of the current scanner settings."""
        return {k: v for k, v in self.settings.items() if k in _ENGINE_KEYS}

    # -- small themed controls -------------------------------------------
    def _wf_field(self, label, value):
        c = self.theme_colors
        ctrl = ft.TextField(
            value=str(value), width=110, height=40, text_size=13,
            bgcolor=c["option_bg"], color=c["text"], border_color=c["border"],
            border_width=1, border_radius=8, focused_border_color=c["purple"],
        )
        return ctrl, ft.Row([
            ft.Text(label, size=12, color=c["text"], expand=True),
            ctrl,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def _wf_card(self, title, body_controls, accent="cyan"):
        c = self.theme_colors
        return ft.Container(
            content=ft.Column([
                ft.Text(title, size=12, weight=ft.FontWeight.BOLD, color=c.get(accent, c["cyan"])),
                ft.Divider(height=1, color=c["border"]),
                *body_controls,
            ], spacing=8),
            bgcolor=_glass_bg(), border=_glass_border(),
            border_radius=14, shadow=_card_shadow(), padding=14,
        )

    def _wf_metric_line(self, label, m):
        if not m:
            return ft.Text(f"{label:<16} no trades", size=11, font_family="monospace",
                           color=self.theme_colors["text_dim"], selectable=True)
        return ft.Text(
            f"{label:<16} {m['total_trades']:>3} trades   "
            f"{m['total_return_pct']:+7.2f}%   WR {m['win_rate']:5.1f}%   "
            f"PF {m['profit_factor']:5.2f}   DD {m['max_drawdown_pct']:4.1f}%",
            size=11, font_family="monospace", color=self.theme_colors["text"], selectable=True,
        )

    # -- the page ---------------------------------------------------------
    def _build_backtest_view(self):
        c = self.theme_colors
        s = self.settings
        state = getattr(self, "wf_state", {"status": "Not run yet.", "result": None, "error": None})
        busy = bool(getattr(self, "_wf_busy", False))

        params = state.get("params") or {}
        self._wf_stop, r1 = self._wf_field("Stop loss (%)", params.get("stop", 2.0))
        self._wf_target, r2 = self._wf_field("Target (%)", params.get("target", 20.0))
        self._wf_trail, r3 = self._wf_field("Trail (%)", params.get("trail", 2.0))
        self._wf_minadx, r4 = self._wf_field("Min ADX gate", s.get("min_adx_entry", 20.0))

        yrs = params.get("years", 3)
        self._wf_years = ft.Dropdown(
            options=[ft.dropdown.Option(str(y)) for y in (3, 5)],
            value=str(yrs), width=110, height=40, text_size=13,
            bgcolor=c["option_bg"], color=c["text"], border_color=c["border"],
            border_width=1, border_radius=8, focused_border_color=c["purple"],
        )
        r5 = ft.Row([
            ft.Text("History (years)", size=12, color=c["text"], expand=True),
            self._wf_years,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)

        self._wf_preset = ft.Dropdown(
            options=[ft.dropdown.Option(n) for n in _WF_PRESETS],
            value=params.get("preset", "Custom"), width=250, height=40, text_size=13,
            bgcolor=c["option_bg"], color=c["text"], border_color=c["border"],
            border_width=1, border_radius=8, focused_border_color=c["purple"],
            on_select=self._wf_apply_preset,
        )
        r6 = ft.Row([
            ft.Text("Config preset", size=12, color=c["text"], expand=True),
            self._wf_preset,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)

        self._wf_ma_set = ft.Dropdown(
            options=[ft.dropdown.Option(key, text)
                     for key, text in _MA_SET_OPTIONS],
            value=params.get("ma_set", "current"), width=250, height=40, text_size=13,
            bgcolor=c["option_bg"], color=c["text"], border_color=c["border"],
            border_width=1, border_radius=8, focused_border_color=c["purple"],
        )
        r7 = ft.Row([
            ft.Text("MA set for this run", size=12, color=c["text"], expand=True),
            self._wf_ma_set,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)

        gates = state.get("gates", {})
        self._wf_regime = ft.Checkbox(label="Index regime (NIFTY > EMA50)", value=bool(gates.get("regime")),
                                      check_color=c["purple"], fill_color=c["card"])
        self._wf_rotation = ft.Checkbox(label="Sector rotation", value=bool(gates.get("rotation")),
                                        check_color=c["purple"], fill_color=c["card"])
        self._wf_nothu = ft.Checkbox(label="Skip Thursday entries", value=bool(gates.get("no_thursday")),
                                     check_color=c["purple"], fill_color=c["card"])

        ma = s
        ma_set_label = "current settings"
        if params.get("ma_set") == "reference":
            ma = {**s, **_MA_REFERENCE}
            ma_set_label = "reference 40x50 (OOS findings)"
        n_stocks = (state.get("result") or {}).get("n_stocks", "…")
        ma_desc = (f"Indicators: HMA({ma.get('fast_ma_len', 40)}) x EMA({ma.get('slow_ma_len', 50)}), "
                   f"crossover lookback {ma.get('crossover_lookback', 20)}, ADX gate "
                   f"{ma.get('min_adx_entry', 20)}. MA set: {ma_set_label}. "
                   f"Universe: NIFTY 50 ({n_stocks} stocks).")

        params_card = self._wf_card("Risk parameters", [r1, r2, r3, r4, r5, r6, r7,
                                     ft.Text("The saved loose 20x40 filter screens broadly but backtests far "
                                             "worse than 40x50 (S5/T8 full-window +1.7% -> -16.9%) — pick the "
                                             "reference MA set to reproduce the OOS findings.",
                                             size=10, color=c["text_dim"])])
        gates_card = self._wf_card("Entry gates (tested on top of the ADX gate)",
                                   [self._wf_regime, self._wf_rotation, self._wf_nothu,
                                    ft.Text("Only the ADX>=20 gate has held up out-of-sample so far — "
                                            "stacking regime/rotation did NOT help in the 2024-26 tests.",
                                            size=10, color=c["text_dim"])], accent="orange")
        settings_note = ft.Text(ma_desc, size=10, color=c["text_dim"])

        self._wf_status = ft.Text(state.get("status", ""), size=12,
                                  color=c["green"] if not state.get("error") else c["red"])
        self._wf_btn = ft.ElevatedButton(
            content=ft.Text("Run Walk-forward" if not busy else "Running…", size=13),
            bgcolor=c["green"], color="#052e16", disabled=busy,
            on_click=None if busy else lambda e: self._run_wf(),
        )
        self._wf_btn_label = ft.Text(
            "Splits the window in half: TRAIN picks the best ADX threshold, TEST measures it "
            "out-of-sample. ~30-90 s on 3y NIFTY 50. FULL-window numbers overstate edge.",
            size=10, color=c["text_dim"], selectable=True)

        results_body = self._wf_result_controls(state)
        results_card = self._wf_card(
            "Walk-forward results",
            results_body if results_body else [ft.Text("—", size=11, color=c["text_dim"])],
            accent="green",
        )

        header = ft.Container(
            content=ft.Column([
                ft.Text("Backtest — walk-forward validation", size=18,
                        weight=ft.FontWeight.BOLD, color=c["text"]),
                ft.Text("Validate settings out-of-sample before saving. "
                        "Only configurations with a positive TEST column have real edge.",
                        size=11, color=c["text_dim"]),
            ], spacing=2),
            padding=_padding_only(left=4, top=12, bottom=4),
        )
        body = ft.Column(
            controls=[params_card, gates_card, settings_note,
                      ft.Row([self._wf_btn, self._wf_status], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                      self._wf_btn_label, results_card],
            spacing=10, scroll=ft.ScrollMode.AUTO, expand=True,
        )
        footer = ft.Row([
            ft.Container(expand=True),
            ft.TextButton(content=ft.Text("Back to dashboard", size=13),
                          on_click=lambda e: self._show_view("dashboard")),
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)
        return ft.Column(
            controls=[header, body,
                      ft.Container(content=footer, padding=_padding_only(top=8, bottom=14, right=6))],
            spacing=0, expand=True,
        )

    def _wf_result_controls(self, state):
        """Text rows describing a completed walk-forward result dict."""
        c = self.theme_colors
        res = state.get("result") or {}
        err = state.get("error")
        if err:
            return [ft.Text(f"Error: {err}", size=11, color=c["red"], selectable=True)]
        if not res:
            if state.get("status", "").startswith("Running"):
                return [ft.Text("Running… results appear here when done.", size=11, color=c["text_dim"])]
            return []
        lines = []
        p = res.get("params", {})
        lines.append(ft.Text(
            f"Split {res.get('split_date')}  |  risk S{p.get('stop', 0):g}/T{p.get('target', 0):g}  |  "
            f"{res.get('n_stocks', 0)} stocks", size=11, font_family="monospace",
            color=c["text_dim"], selectable=True))
        lines.append(self._wf_metric_line("FULL (in-sample)", res.get("full")))
        for adx, m in sorted((res.get("train_by_adx") or {}).items()):
            lines.append(self._wf_metric_line(f"TRAIN adx>={adx:g}", m))
        chosen = res.get("chosen_adx")
        lines.append(ft.Text(f"-> TRAIN picks min ADX = {chosen:g}",
                             size=11, font_family="monospace", color=c["orange"], selectable=True))
        lines.append(self._wf_metric_line("TEST no gate", res.get("test_no_gate")))
        lines.append(self._wf_metric_line(f"TEST adx>={chosen:g} (OOS)", res.get("test_chosen")))
        verdict = _wf_verdict(res)
        if verdict:
            color = c["green"] if "positive" in verdict else c["text"]
            lines.append(ft.Text(verdict, size=11, weight=ft.FontWeight.BOLD,
                                 color=color, selectable=True))
        return lines

    # -- run flow ---------------------------------------------------------
    def _run_wf(self, e=None):
        try:
            stop = float(self._wf_stop.value or 0)
            target = float(self._wf_target.value or 0)
            trail = float(self._wf_trail.value or 0)
            min_adx = float(self._wf_minadx.value or 0)
            years = int(self._wf_years.value or 3)
            if stop <= 0 or target <= 0 or trail < 0 or min_adx < 0:
                raise ValueError("risk values must be positive")
        except ValueError:
            self.wf_state = {"status": "Invalid numbers in the risk fields.",
                             "result": None, "error": "invalid input"}
            self._wf_status.value = self.wf_state["status"]
            self._wf_status.color = self.theme_colors["red"]
            self.page.update()
            return
        ma_set = self._wf_ma_set.value if hasattr(self, "_wf_ma_set") else "current"
        self.wf_state = {
            "status": "Running walk-forward (NIFTY 50)… window stays open, results appear here.",
            "result": None, "error": None,
            "params": {"stop": stop, "target": target, "trail": trail,
                       "years": years, "min_adx": min_adx,
                       "preset": self._wf_preset.value, "ma_set": ma_set},
            "gates": {"regime": self._wf_regime.value, "rotation": self._wf_rotation.value,
                      "no_thursday": self._wf_nothu.value},
        }
        self._wf_busy = True
        self.main_area_box.content = self._build_backtest_view()
        threading.Thread(target=self._wf_job,
                         kwargs={"years": years, "stop": stop, "target": target,
                                 "trail": trail, "min_adx": min_adx, "ma_set": ma_set},
                         daemon=True).start()
        self.page.update()

    def _wf_apply_preset(self, e=None):
        """Fill the risk fields (and MA set) from the selected preset."""
        p = _WF_PRESETS.get(self._wf_preset.value)
        if not p:
            return
        self._wf_stop.value = str(p["stop"])
        self._wf_target.value = str(p["target"])
        self._wf_trail.value = str(p["trail"])
        self._wf_minadx.value = str(p["min_adx"])
        if p.get("ma") == "reference":
            self._wf_ma_set.value = "reference"
        self.page.update()

    def _wf_job(self, years, stop, target, trail, min_adx, ma_set="current"):
        try:
            import contextlib
            import io

            from .walkforward import run_walkforward
            overrides = self._wf_overrides()
            if ma_set == "reference":
                overrides = {**overrides, **_MA_REFERENCE}
            # The engine prints its banners to stdout; swallow them so the app
            # console stays clean (results come back via the return dict).
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_walkforward(
                    period=f"{years}y", stop=stop, target=target, trail=trail,
                    min_adx=min_adx, verbose=False,
                    regime=self.wf_state.get("gates", {}).get("regime", False),
                    rotation=self.wf_state.get("gates", {}).get("rotation", False),
                    no_thursday=self.wf_state.get("gates", {}).get("no_thursday", False),
                    ma_overrides=overrides,
                )
            self._safe_update(lambda: self._wf_done(res))
        except Exception as ex:  # pragma: no cover - defensive
            msg = str(ex)
            self._safe_update(lambda: self._wf_done({"error": msg}))

    def _wf_done(self, res):
        self._wf_busy = False
        if isinstance(res, dict) and res.get("error"):
            self.wf_state = {"status": f"Walk-forward failed: {res['error']}",
                             "result": None, "error": res["error"]}
        else:
            tc = (res or {}).get("test_chosen")
            self.wf_state = {"status": "Walk-forward complete." if tc else "Walk-forward finished (no trades).",
                             "result": res, "error": None}
        if self.active_view == "backtest" and self.main_area_box is not None:
            self.main_area_box.content = self._build_backtest_view()
        self._log(self.wf_state["status"])
        self.page.update()


def _wf_verdict(res: dict) -> str:
    """One-line read of the out-of-sample numbers ('' when nothing to judge)."""
    tc, tg = res.get("test_chosen"), res.get("test_no_gate")
    chosen = res.get("chosen_adx")
    if not tc or not tg or tc.get("total_trades", 0) == 0:
        return ""
    delta = tc["total_return_pct"] - tg["total_return_pct"]
    gate = f"ADX>={chosen:g}" if chosen else "no gate"
    pos = tc["total_return_pct"] >= 0
    s = (f"Out-of-sample with {gate}: {tc['total_return_pct']:+.2f}% "
         f"({delta:+.2f} pts vs no gate). ")
    if pos:
        s += "Positive out-of-sample — worth testing further."
    elif tg.get("total_return_pct", 0) >= 0:
        s += "No-gate was already positive; the gate only made it worse here."
    else:
        s += "Still negative — do not save these risk settings."
    return s
