"""Responsive tier engine: width tiers, column hiding, pane reflow.

Pure tier helpers (``width_tier`` / ``hidden_cols``) plus the ScannerApp
reflow plumbing — auto pane collapse, deferred reflow during a streaming
scan, and header/row cell filtering that must stay aligned per tier.
"""

import flet as ft

import scanner.backend.settings_store as store_mod
import scanner.ui.app as app_mod
from scanner.shared.constants import RESULT_COLS
from scanner.tests.ui.conftest import make_app
from scanner.ui.ui_kit import (
    COL_HIDE_ORDER,
    OPTIONAL_COLS,
    ROW_H,
    TIER_COMPACT,
    TIER_STANDARD,
    TIER_WIDE,
    hidden_cols,
    width_tier,
)
from scanner.ui.views_layout import SIDE_W

_ROW = {"ticker": "TCS", "close": 100}


# ── Pure tier helpers ───────────────────────────────────────────────


def test_width_tier_boundaries():
    assert width_tier(1280) == TIER_COMPACT  # window floor
    assert width_tier(1399) == TIER_COMPACT
    assert width_tier(1400) == TIER_STANDARD
    assert width_tier(1599) == TIER_STANDARD
    assert width_tier(1600) == TIER_WIDE
    assert width_tier(1920) == TIER_WIDE
    assert width_tier(None) == TIER_WIDE
    assert width_tier("junk") == TIER_WIDE


def test_hidden_cols_are_tail_only_and_nested():
    compact, standard = hidden_cols(TIER_COMPACT), hidden_cols(TIER_STANDARD)
    assert hidden_cols(TIER_WIDE) == frozenset()
    assert len(standard) == 3 and len(compact) == 6
    assert standard < compact <= frozenset(COL_HIDE_ORDER)
    # Positional row specials (indexes 0/1/3/4/6: rail, ticker, rating,
    # entry, MA) must never shift — only tail columns may be hidden.
    assert min(RESULT_COLS.index(name) for name in compact) >= 9
    assert compact <= frozenset(RESULT_COLS)


# ── Header / cell filtering follows the tier ────────────────────────


def test_specs_and_header_follow_tier():
    app = make_app()
    c = app.theme_colors
    assert len(app._row_specs(_ROW, 1, c, 50)) == 18
    assert len(app._make_header_row(c).content.controls) == 19

    app.width_tier = TIER_STANDARD
    assert len(app._row_specs(_ROW, 1, c, 50)) == 15
    assert len(app._make_header_row(c).content.controls) == 16

    app.width_tier = TIER_COMPACT
    assert len(app._row_specs(_ROW, 1, c, 50)) == 12
    assert len(app._make_header_row(c).content.controls) == 13


def test_created_row_cells_match_filtered_header():
    app = make_app()
    c = app.theme_colors
    app.width_tier = TIER_COMPACT
    row = app._create_row_controls(_ROW, 1, c, c["card"], 50)
    assert len(app._row_cells["TCS"]) == 12  # filtered spec pairs
    assert len(row.content.controls) == 13  # + sparkline cell


def test_report_parity_columns_in_row_specs():
    """POC / Both MA / RSI Val / Volatility mirror the HTML report columns."""
    app = make_app()
    c = app.theme_colors
    row = {
        **_ROW,
        "above_poc": True,
        "close_above_both_ma": False,
        "rsi_val": 55.0,
        "volat_stat": "High",
    }
    specs = app._row_specs(row, 1, c, 50)
    assert RESULT_COLS[7:9] == ["POC", "Both MA"]
    assert RESULT_COLS[12] == "RSI Val"
    assert RESULT_COLS[16] == "Volatility"
    assert specs[7] == ("Above", c["green"])
    assert specs[8] == ("NO", c["text_dim"])
    assert specs[12] == ("55.0", c["green"])  # 40-70 band
    assert specs[16] == ("High", c["text"])


def test_fund_cell_shows_na_when_zero():
    """F/20 = 0 (no data) renders N/A; a real score stays numeric."""
    app = make_app()
    c = app.theme_colors
    specs = app._row_specs({"ticker": "X", "fundamentals": 0}, 1, c, 50)
    assert specs[11] == ("N/A", c["text_dim"])
    specs = app._row_specs({"ticker": "X", "fundamentals": 13.0}, 1, c, 50)
    assert specs[11] == ("13", c.get("fund", c["yellow"]))


# ── Pane reflow plumbing ────────────────────────────────────────────


def test_apply_tier_auto_collapses_and_restores_panes():
    app = make_app()
    app.sidebar = ft.Container(width=SIDE_W)
    app.right_panel = ft.Container(visible=True)

    app._apply_width_tier(TIER_COMPACT)
    assert app.width_tier == TIER_COMPACT
    assert app.sidebar.width == 0
    assert app.right_panel.visible is False
    assert app._sidebar_collapsed is True
    assert app._auto_side is True
    assert app.header_holder.controls == []

    app._apply_width_tier(TIER_WIDE)
    assert app.sidebar.width == SIDE_W
    assert app.right_panel.visible is True
    assert app._sidebar_collapsed is False


def test_compact_keeps_pinned_right_panel():
    app = make_app()
    app.right_panel = ft.Container(visible=True)
    app._right_pinned = True

    app._apply_width_tier(TIER_COMPACT)
    assert app.right_panel.visible is True


def test_manual_sidebar_collapse_survives_tier_cycle():
    app = make_app()
    app.sidebar = ft.Container(width=0)
    app._sidebar_collapsed = True

    app._apply_width_tier(TIER_COMPACT)
    app._apply_width_tier(TIER_WIDE)
    assert app.sidebar.width == 0  # manual state wins over auto-restore
    assert app._auto_side is False


# ── Resize handler: tier-change only, deferred under a live scan ────


def test_resize_same_tier_is_noop():
    app = make_app()
    app.page.width = 1600
    app._on_page_resize()
    assert app.width_tier == TIER_WIDE
    assert app._pending_tier is None


def test_resize_defers_while_scanning_then_drains():
    app = make_app()
    app.page.width = 1300
    app.scanning = True

    app._on_page_resize()
    assert app.width_tier == TIER_WIDE  # no reflow mid-scan
    assert app._pending_tier == TIER_COMPACT

    app.scanning = False
    app._apply_pending_width_tier()
    assert app.width_tier == TIER_COMPACT
    assert app._pending_tier is None

    # Second drain with nothing pending is a no-op.
    app._pending_tier = None
    app._apply_pending_width_tier()
    assert app.width_tier == TIER_COMPACT


def test_resize_returning_to_original_tier_drops_stale_pending():
    """A narrow-then-widen round trip mid-scan must not queue a stale tier.

    Regression: the same-tier early return used to skip the queue update, so
    the queued 'compact' survived and was applied at teardown — reflowing a
    window that was actually wide again into the compact layout.
    """
    app = make_app()
    app.page.width = 1600
    app.width_tier = TIER_WIDE
    app.scanning = True

    # Narrow mid-scan -> queued.
    app.page.width = 1300
    app._on_page_resize()
    assert app._pending_tier == TIER_COMPACT

    # Widen back mid-scan. Tier now matches the current tier, so the handler
    # takes the early-return path — which must clear the stale queue.
    app.page.width = 1600
    app._on_page_resize()
    assert app._pending_tier is None

    # Teardown must leave the app in the tier the window is actually in.
    app.scanning = False
    app._apply_pending_width_tier()
    assert app.width_tier == TIER_WIDE


def test_resize_to_queued_tier_replaces_pending():
    """A second queued tier supersedes the first (queue holds the latest)."""
    app = make_app()
    app.page.width = 1600
    app.width_tier = TIER_WIDE
    app.scanning = True

    app.page.width = 1300
    app._on_page_resize()
    assert app._pending_tier == TIER_COMPACT

    # 1450 is still a change away from the current (wide) tier -> re-queue.
    app.page.width = 1450
    app._on_page_resize()
    assert app._pending_tier == TIER_STANDARD

    app.scanning = False
    app._apply_pending_width_tier()
    assert app.width_tier == TIER_STANDARD
    assert app._pending_tier is None


# ── Columns menu + Density menu (grid view toggles) ─────────────────


def test_optional_cols_menu_is_tail_only():
    """The Columns menu never offers a positional row special."""
    assert OPTIONAL_COLS == tuple(RESULT_COLS[9:])
    assert {"1M", "Chop", "Volatility"} <= set(OPTIONAL_COLS)
    assert not {"#", "Ticker", "ENTRY", "Rating", "MA"} & set(OPTIONAL_COLS)


def test_user_hidden_columns_unite_with_tier_hides():
    app = make_app()
    c = app.theme_colors
    app.user_hidden_cols = {"Chop"}

    assert app._hidden_cols() == frozenset({"Chop"})
    assert len(app._make_header_row(c).content.controls) == 18

    app.width_tier = TIER_COMPACT  # tier drops six more on top
    assert app._hidden_cols() == frozenset(COL_HIDE_ORDER) | {"Chop"}


def test_toggle_user_col_guards_and_invalidates_pool():
    app = make_app()
    c = app.theme_colors
    app._debounce = lambda *a, **k: None
    app._row_pool = {"STALE": object()}

    app._toggle_user_col("Ticker")  # positional special → refused
    assert app._user_hidden() == set() and app._row_pool

    app.width_tier = TIER_COMPACT
    app._toggle_user_col("Chop")  # tier already hides it → refused
    assert app._user_hidden() == set() and app._row_pool

    app.width_tier = TIER_WIDE
    app._toggle_user_col("Chop")  # hides it and drops pooled rows
    assert app._user_hidden() == {"Chop"} and not app._row_pool
    assert len(app._make_header_row(c).content.controls) == 18

    app._toggle_user_col("Chop")  # toggle back restores the column
    assert app._user_hidden() == set()
    assert len(app._make_header_row(c).content.controls) == 19


def test_density_switch_changes_row_and_header_height():
    app = make_app()
    c = app.theme_colors
    app._debounce = lambda *a, **k: None

    assert app._row_density() == "comfort"
    assert app._row_h() == ROW_H["comfort"] == 34
    app.row_density = "compact"
    assert app._row_h() == 27
    assert app._make_header_row(c).height == 27

    app.row_density = "sunset"  # unknown preset falls back to Comfort
    assert app._row_density() == "comfort" and app._row_h() == 34

    app.row_density = "comfort"
    app._row_pool = {"STALE": object()}
    app._set_density("comfort")  # no-op leaves pooled rows alone
    assert app._row_pool

    app._set_density("compact")  # height is baked into pooled rows
    assert app.row_density == "compact" and not app._row_pool


def test_hiding_1m_drops_sparkline_so_cells_align_with_header():
    """Column 18 draws its own cell — hiding it must drop both sides."""
    app = make_app()
    c = app.theme_colors
    app.user_hidden_cols = {"1M"}

    header = app._make_header_row(c)
    row = app._create_row_controls(
        {**_ROW, "px_tail": [1.0, 2.0, 3.0]}, 1, c, c["card"], 50
    )

    assert len(header.content.controls) == 18
    assert len(row.content.controls) == 18
    assert len(app._row_cells["TCS"]) == 18  # spec pairs, sparkline excluded


def test_menu_checkmarks_follow_state_in_place():
    """Checks are stamped onto the *existing* items: rebuilding the items
    list reassigns control ids, and the open menu then dispatches taps
    against the stale ids (handler never fires)."""
    app = make_app()
    app._debounce = lambda *a, **k: None
    app._build_ui()
    items = app.density_menu.items

    app._set_density("compact")

    assert app.density_menu.items is items  # never replaced
    assert [i.checked for i in items] == [False, True]  # Comfort off

    app.user_hidden_cols = {"Chop"}
    app._sync_menu_checks()
    chop = next(i for i in app.columns_menu.items if i.content.value == "Chop")
    assert chop.checked is False


def test_density_and_hidden_cols_round_trip_through_prefs(monkeypatch):
    app = make_app()
    app._debounce = lambda key, delay, fn: fn()  # run the save inline
    app.row_density = "compact"
    app.user_hidden_cols = {"Chop", "ADX"}
    written = {}
    monkeypatch.setattr(store_mod, "save_settings", lambda s: written.update(s))

    app._save_ui_prefs()

    assert written["ui_density"] == "compact"
    assert written["ui_hidden_cols"] == "ADX,Chop"

    app.settings = {
        **app_mod.DEFAULT_SETTINGS,
        "ui_density": "compact",
        "ui_hidden_cols": "Chop,NotAColumn",
    }
    app._load_ui_prefs()
    assert app.row_density == "compact"
    assert app.user_hidden_cols == {"Chop"}  # unknown name dropped

    app.settings = {**app_mod.DEFAULT_SETTINGS, "ui_density": "sunset"}
    app._load_ui_prefs()
    assert app.row_density == "comfort"
