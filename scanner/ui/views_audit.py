"""Stale-member audit flow — ``AuditViewMixin`` for ``scanner.app.ScannerApp``.

Owns the stale-member audit report/verdict, the threaded audit run, and the
apply-fixes confirmation dialog + runner that edits ``universes.py``.

The methods are mixins: they run against the ``ScannerApp`` instance (via
the MRO) and rely on state/controls created elsewhere (``self._log``,
``self._toast``, ``self._safe_update``, ``self._close_dialog``, …).
"""

import threading

import flet as ft

from .ui_kit import RADIUS_LG, RADIUS_MD


class AuditViewMixin:
    def _audit_report(self, universe: str):
        """Sync core of the stale-member audit button (threaded in the GUI)."""
        from ..backend.audit_stale_members import audit_stale_members, format_report
        from ..shared.universes import UNIVERSES

        tickers = list(UNIVERSES.get(universe, []))
        res = audit_stale_members(tickers)
        return res, format_report(res)

    def _audit_fixable(self, res: dict) -> bool:
        """True when the audit result has anything apply_fixes could change."""
        return bool(
            res.get("rename_suggestions")
            or res.get("unannotated_stale")
            or res.get("annotated_fresh")
        )

    def _audit_verdict(self, res: dict) -> str:
        """One-line audit summary shown in the log + status label."""
        verdict = (
            f"{len(res['unannotated_stale'])} new candidate(s), "
            f"{len(res['annotated_stale'])} still annotated, "
            f"{len(res['missing'])} missing"
        )
        renames = res.get("rename_suggestions", {})
        if renames:
            verdict += f", {len(renames)} rename(s) suggested"
        if self._audit_fixable(res):
            verdict += " — fixes ready"
        return verdict

    def _run_stale_audit(self, e=None):
        lock = getattr(self, "_state_lock", None)
        if lock is not None:
            with lock:
                if getattr(self, "stale_audit_running", False):
                    return
                self.stale_audit_running = True
        elif getattr(self, "stale_audit_running", False):
            return
        else:
            self.stale_audit_running = True
        universe = None
        dd = getattr(self, "universe_dd", None)
        if dd is not None and getattr(dd, "value", None):
            universe = str(dd.value)
        universe = universe or "ALL (Combined)"

        def _bg():
            try:
                self._log(f"Stale-member audit: {universe} …")
                res, report = self._audit_report(universe)
                verdict = self._audit_verdict(res)
                lock = getattr(self, "_state_lock", None)
                if lock is not None:
                    with lock:
                        self._last_audit_res = res
                else:
                    self._last_audit_res = res
                self._log("\n" + report)
                self._safe_update(lambda: self._log(f"Audit done — {verdict}"))
                self._safe_update(
                    lambda: self._toast(f"Audit done — {verdict}", "info")
                )
                if hasattr(self, "stale_audit_lbl"):
                    self._safe_update(
                        lambda: setattr(
                            self.stale_audit_lbl, "value", f"Audit done — {verdict}"
                        )
                    )
                if hasattr(self, "stale_fix_btn"):
                    self._safe_update(
                        lambda: setattr(
                            self.stale_fix_btn, "disabled", not self._audit_fixable(res)
                        )
                    )
            except Exception as ex:
                msg = f"Stale-member audit failed: {ex}"
                self._safe_update(lambda: self._log(msg))
                # Never leave a stale (previously fixable) result actionable
                # after a failed audit — force a fresh audit before applying.
                lock = getattr(self, "_state_lock", None)
                if lock is not None:
                    with lock:
                        self._last_audit_res = None
                else:
                    self._last_audit_res = None
                if hasattr(self, "stale_fix_btn"):
                    self._safe_update(
                        lambda: setattr(self.stale_fix_btn, "disabled", True)
                    )
            finally:
                lock = getattr(self, "_state_lock", None)
                if lock is not None:
                    with lock:
                        self.stale_audit_running = False
                else:
                    self.stale_audit_running = False
                self._safe_update(lambda: self.page.update())

        threading.Thread(target=_bg, daemon=True).start()

    def _apply_fixes_core(self, res: dict):
        """Sync core of the Apply fixes button (threaded in the GUI)."""
        from ..backend.audit_stale_members import apply_fixes, format_fix_summary

        summary = apply_fixes(res)
        return summary, format_fix_summary(summary)

    def _apply_stale_fixes(self, e=None):
        """Ask for confirmation, then apply audit fixes to universes.py.

        The dialog lists exactly what will change (renames / annotation adds /
        removals); its Apply button runs the threaded fix, which keeps a .bak
        and re-audits afterwards to confirm the file is clean.
        """
        if getattr(self, "stale_fix_running", False):
            return
        lock = getattr(self, "_state_lock", None)
        if lock is not None:
            with lock:
                res = getattr(self, "_last_audit_res", None)
        else:
            res = getattr(self, "_last_audit_res", None)
        if not res or not self._audit_fixable(res):
            return
        parts = []
        renames = len(res.get("rename_suggestions", {}))
        if renames:
            parts.append(f"{renames} rename(s)")
        adds = len(res.get("unannotated_stale", []))
        if adds:
            parts.append(f"{adds} annotation add(s)")
        removes = len(res.get("annotated_fresh", []))
        if removes:
            parts.append(f"{removes} annotation removal(s)")
        c = self.theme_colors
        dlg = ft.AlertDialog(
            modal=True,
            bgcolor=c["panel_bg"],
            shape=ft.RoundedRectangleBorder(radius=RADIUS_LG),
            actions_alignment=ft.MainAxisAlignment.END,
            title=ft.Text("Apply audit fixes?"),
            content=ft.Text(
                f"This edits scanner/shared/universes.py ({', '.join(parts)}). "
                "A .bak backup is kept and the edit is syntax-checked. "
                "The app re-audits afterwards; restart it for scans to "
                "pick up the new universe lists.",
                size=13,
            ),
            actions=[
                ft.TextButton(
                    content=ft.Text("Cancel"),
                    on_click=lambda _: self._close_dialog(dlg),
                ),
                ft.Button(
                    content=ft.Text("Apply fixes"),
                    bgcolor=c["green"],
                    color=c["on_accent"],
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=RADIUS_MD)
                    ),
                    on_click=lambda _: self._run_apply_fixes(dlg),
                ),
            ],
        )
        self._safe_update(lambda: self.page.show_dialog(dlg))

    def _reload_universes(self):
        """Re-read universes.py after a fix so a re-audit sees the edits."""
        try:
            import importlib

            import scanner.shared.universes as univ

            importlib.reload(univ)
        except Exception as ex:
            self._log(f"Could not reload universes after fix: {ex}")

    def _run_apply_fixes(self, dlg=None):
        """Threaded fix runner: apply_fixes, log the summary, re-audit."""
        if dlg is not None:
            self._close_dialog(dlg)
        res = getattr(self, "_last_audit_res", None)
        if not res or getattr(self, "stale_fix_running", False):
            return
        self.stale_fix_running = True

        def _bg():
            try:
                self._log("Applying audit fixes to scanner/shared/universes.py …")
                summary, text = self._apply_fixes_core(res)
                self._log("\n" + text)
                if summary.get("changed"):
                    verdict = (
                        f"{len(summary['renamed'])} rename(s), "
                        f"{len(summary['annotated_added'])} annotation(s) added, "
                        f"{len(summary['annotated_removed'])} removed — "
                        "universes.py updated (backup kept; restart the app "
                        "for scans to pick up the new symbols)"
                    )
                else:
                    verdict = "Nothing to apply (annotation list and symbols current)"
                self._safe_update(lambda: self._log(f"Fix done — {verdict}"))
                if hasattr(self, "stale_audit_lbl"):
                    self._safe_update(
                        lambda: setattr(
                            self.stale_audit_lbl, "value", f"Fix done — {verdict}"
                        )
                    )
                if hasattr(self, "stale_fix_btn"):
                    self._safe_update(
                        lambda: setattr(self.stale_fix_btn, "disabled", True)
                    )
                # Confirm the file is clean: reload the edited module and
                # re-run the same audit (cache-warm, fast).
                if summary.get("changed"):
                    self._reload_universes()
                    self._safe_update(
                        lambda: self._log(
                            "Re-auditing to confirm universes.py is clean …"
                        )
                    )
                    self._run_stale_audit()
            except Exception as ex:
                msg = f"Applying audit fixes failed: {ex}"
                self._safe_update(lambda: self._log(msg))
            finally:
                # Clear only the result we applied — the re-audit started above
                # may already have stored a newer result that must survive.
                # The check-and-clear is atomic under the lock; without it a
                # racing audit could store a fresh result between our read and
                # write and we would silently discard it.
                lock = getattr(self, "_state_lock", None)
                if lock is not None:
                    with lock:
                        if self._last_audit_res is res:
                            self._last_audit_res = None
                elif self._last_audit_res is res:
                    self._last_audit_res = None
                self.stale_fix_running = False
                self._safe_update(lambda: self.page.update())

        threading.Thread(target=_bg, daemon=True).start()
