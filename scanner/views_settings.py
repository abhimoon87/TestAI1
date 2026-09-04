"""Scanner settings page — ``SettingsViewMixin`` for ``scanner.app.ScannerApp``.

Builds the parameter-editing page (grouped cards over a declarative spec)
and the input controls it is made of. ``_save_settings_page`` persists via
``scanner.app.save_settings``; it imports the module lazily at call time to
avoid a circular import, since ``scanner.app`` imports this mixin.

The methods are mixins: they run against the ``ScannerApp`` instance (via
the MRO) and rely on ``self.settings``, ``self.theme_colors`` and the
``_settings_inputs`` registry the builder creates.
"""

import flet as ft

from .ui_kit import (
    _card_shadow,
    _glass_bg,
    _glass_border,
    _padding_only,
)


class SettingsViewMixin:
    _MA_TYPES = ["HMA", "EMA", "SMA", "KAMA", "VWMA"]

    # (key, label, kind, min, max) — kind in {"int", "float", "ma_type", "theme"}.
    # Bounds are intentionally lenient: they only reject values the engine
    # itself cannot tolerate, so a stored setting can never brick Save.
    _SETTINGS_SPEC = [
        ("Signal — HMA×EMA crossover", [
            ("fast_ma_type", "Fast MA type", "ma_type", None, None),
            ("fast_ma_len", "Fast MA length", "int", 2, 500),
            ("slow_ma_type", "Slow MA type", "ma_type", None, None),
            ("slow_ma_len", "Slow MA length", "int", 2, 500),
            ("crossover_lookback", "Crossover lookback (bars)", "int", 1, 200),
        ]),
        ("Momentum & volume", [
            ("rsi_len", "RSI length", "int", 2, 200),
            ("vol_ma_len", "Volume MA length", "int", 2, 300),
            ("volume_participation_len", "Volume participation window (bars)", "int", 1, 50),
            ("rs_length", "Relative-strength length", "int", 2, 200),
        ]),
        ("Trend strength & slope", [
            ("adx_len", "ADX length", "int", 2, 200),
            ("adx_threshold", "ADX threshold", "float", 0, 100),
            ("min_adx_entry", "Min ADX for entry signal", "float", 0, 100),
            ("slope_ma_type", "Slope MA type", "ma_type", None, None),
            ("slope_ma_len", "Slope MA length", "int", 2, 500),
            ("slope_lookback", "Slope lookback (bars)", "int", 1, 200),
            ("flat_threshold", "Flat-slope threshold", "float", 0, 100),
        ]),
        ("Range / chop & ATR", [
            ("atr_len", "ATR length", "int", 2, 200),
            ("chop_len", "Chop length", "int", 2, 200),
            ("chop_threshold", "Chop threshold", "float", 0, 100),
            ("sideways_strong_move_pct", "Strong-move sideways guard (% 1M move)", "float", 0, 50),
        ]),
        ("Volume profile", [
            ("vp_lookback", "VP lookback (bars)", "int", 2, 1000),
            ("vp_rows", "VP rows", "int", 2, 200),
        ]),
        ("Output, cache & theme", [
            ("min_score", "Min score", "float", 0, 100),
            ("negative_cache_ttl_hours", "Dead-cache TTL (hours)", "float", 0, 1440),
            ("stale_member_max_age_days", "Stale-member age (days)", "float", 7, 730),
            ("theme", "Theme", "theme", None, None),
        ]),
    ]

    def _settings_input(self, key, label, kind, lo, hi):
        c = self.theme_colors
        if kind == "ma_type":
            opts = list(self._MA_TYPES)
            cur = str(self.settings.get(key, opts[0]))
            ctrl = ft.Dropdown(
                options=[ft.dropdown.Option(v) for v in opts],
                value=cur if cur in opts else opts[0],
                width=150, height=40, text_size=13,
                bgcolor=c["option_bg"], color=c["text"],
                border_color=c["border"], border_width=1, border_radius=8,
                focused_border_color=c["purple"],
            )
        elif kind == "theme":
            opts = ["dark", "light"]
            cur = str(self.settings.get(key, "dark"))
            ctrl = ft.Dropdown(
                options=[ft.dropdown.Option(v) for v in opts],
                value=cur if cur in opts else "dark",
                width=150, height=40, text_size=13,
                bgcolor=c["option_bg"], color=c["text"],
                border_color=c["border"], border_width=1, border_radius=8,
                focused_border_color=c["purple"],
            )
        else:
            ctrl = ft.TextField(
                value=str(self.settings.get(key, "")), width=150, height=40, text_size=13,
                bgcolor=c["card"], color=c["text"],
                border_color=c["border"], border_width=1, border_radius=8,
                content_padding=_padding_only(left=10, right=8, top=6, bottom=6),
            )
        ctrl._settings_key = key
        ctrl._settings_kind = kind
        ctrl._settings_lo = lo
        ctrl._settings_hi = hi
        self._settings_inputs[key] = ctrl
        return ft.Row([
            ft.Text(label, size=12, color=c["text"], expand=True),
            ctrl,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def _build_settings_view(self):
        c = self.theme_colors
        self._settings_inputs = {}
        cards = []
        for title, fields in self._SETTINGS_SPEC:
            rows = [self._settings_input(key, label, kind, lo, hi)
                    for key, label, kind, lo, hi in fields]
            cards.append(
                ft.Container(
                    content=ft.Column([
                        ft.Text(title, size=12, weight=ft.FontWeight.BOLD, color=c["cyan"]),
                        ft.Divider(height=1, color=c["border"]),
                        *rows,
                    ], spacing=8),
                    bgcolor=_glass_bg(),
                    border=_glass_border(),
                    border_radius=14,
                    shadow=_card_shadow(),
                    padding=14,
                )
            )
        self._settings_error = ft.Text("", size=11, color=c["red"])
        header = ft.Container(
            content=ft.Column([
                ft.Text("⚙  Scanner Settings", size=18, weight=ft.FontWeight.BOLD, color=c["text"]),
                ft.Text("Indicator parameters consumed by the HMA×EMA engine — applied on Save.",
                        size=11, color=c["text_dim"]),
            ], spacing=2),
            padding=_padding_only(left=4, top=12, bottom=4),
        )
        body = ft.Column(
            controls=[*cards, self._settings_error],
            spacing=10, scroll=ft.ScrollMode.AUTO, expand=True,
        )
        self.stale_audit_lbl = ft.Text("", size=10, color=c["text_dim"])
        self.stale_fix_btn = ft.OutlinedButton(
            content=ft.Text("Apply fixes", size=12),
            on_click=self._apply_stale_fixes,
            disabled=True,  # enabled after an audit finds something fixable
        )
        audit_row = ft.Row([
            ft.Text("Maintenance", size=10, weight=ft.FontWeight.BOLD, color=c["text_faint"]),
            ft.Container(expand=True),
            ft.OutlinedButton(content=ft.Text("Check stale members", size=12),
                              on_click=self._run_stale_audit),
            self.stale_fix_btn,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)
        footer = ft.Row([
            ft.Container(expand=True),
            ft.TextButton(content=ft.Text("Cancel", size=13),
                          on_click=lambda e: self._show_view("dashboard")),
            ft.ElevatedButton(content=ft.Text("Save Settings", size=13),
                               bgcolor=c["green"], color="#052e16",
                               on_click=lambda e: self._save_settings_page()),
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)
        return ft.Column(
            controls=[
                header, body,
                ft.Container(content=audit_row, padding=_padding_only(left=6, right=6, top=8)),
                ft.Container(content=self.stale_audit_lbl, padding=_padding_only(left=6, right=6, bottom=4)),
                ft.Container(content=footer, padding=_padding_only(top=4, bottom=14, right=6)),
            ],
            spacing=0, expand=True,
        )

    def _save_settings_page(self, e=None):
        bad = []
        for key, ctrl in self._settings_inputs.items():
            kind = ctrl._settings_kind
            if kind in ("ma_type", "theme"):
                val = ctrl.value
                ok_opts = self._MA_TYPES if kind == "ma_type" else ["dark", "light"]
                if val not in ok_opts:
                    bad.append(key)
                    continue
            else:
                raw = (ctrl.value or "").strip()
                try:
                    val = int(float(raw)) if kind == "int" else float(raw)
                    lo, hi = ctrl._settings_lo, ctrl._settings_hi
                    if (lo is not None and val < lo) or (hi is not None and val > hi):
                        raise ValueError(key)
                except (ValueError, TypeError):
                    bad.append(key)
                    continue
            self.settings[key] = val
        if bad:
            self._settings_error.value = f"Invalid value: {', '.join(bad)}"
            self.page.update()
            return
        new_theme = self.settings.get("theme", self.current_theme)
        # Local import: scanner.app imports this mixin at module load, so the
        # module-level helper is only resolvable once app.py has finished.
        from .app import save_settings
        save_settings(self.settings)
        self._apply_cache_settings()
        self._log("Settings saved")
        if new_theme != self.current_theme:
            self._switch_theme(to=new_theme)
        else:
            self._load_settings_to_ui()
            self._show_view("dashboard")
