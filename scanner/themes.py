"""
Theme definitions for the HMAxEMA Scanner GUI — Aurora v3 (Flet edition).

Palette inspired by premium trading terminals (linear.app / vercel dark).
Matches scanner_report HTML CSS variables in both variants.

Flet uses ft.Theme and ft.ColorScheme for theming. This module provides
color dicts consumed by the Flet UI code.
"""

from .settings_store import save_settings

THEMES = {
    "dark": {
        # Base surfaces — deep slate / charcoal
        "root_bg": "#121212",
        "rail_bg": "#0b0b0d",
        "side_bg": "#161618",
        "main_bg": "#121212",
        "panel_bg": "#1a1a20",
        # Cards / rows
        "card": "#1c1c22",
        "card2": "#24242c",
        "card_hover": "#26262e",
        "border": "#2b2b34",
        "border_light": "#3a3a46",
        "row_alt": "#151518",
        "row_hover": "#1c1c22",
        # Text
        "text": "#e9eaf0",
        "text_dim": "#8e93a8",
        "text_faint": "#5c6178",
        # Accents — neon blue family + supporting hues
        "purple": "#38bdf8",
        "purple_hover": "#7dd3fc",
        "purple_muted": "#0c2b3d",
        "pink": "#a78bfa",
        "cyan": "#22d3ee",
        "cyan_dim": "#0e7490",
        "green": "#34d399",
        "lime": "#a3e635",
        "orange": "#fb923c",
        "red": "#f87171",
        "red_hover": "#ef4444",
        "blue": "#60a5fa",
        "yellow": "#facc15",
        "neon": "#38bdf8",
        # Controls
        "option_bg": "#202027",
        "option_btn": "#2b2b34",
        "option_drop": "#1c1c22",
        "entry_bg": "#1c1c22",
        "entry_border": "#2b2b34",
        "entry_focus": "#38bdf8",
        "progress_bg": "#26262e",
        "progress_fg": "#38bdf8",
        "nav_active": "#202027",
        "chip_good": "#0b2f2e",
        "chip_bad": "#3a1519",
        "chip_neutral": "#23232b",
        # Hero gradient — neon blue into violet, dissolving to charcoal
        "hero_grad": ["#0ea5e9", "#6366f1", "#3b2d8f", "#17171c"],
        "shadow": "#00000066",
        # Flet theme mode
        "flet_mode": "dark",
    },
    "light": {
        "root_bg": "#f0f5f1",
        "rail_bg": "#e6efe8",
        "side_bg": "#f7fbf8",
        "main_bg": "#f0f5f1",
        "panel_bg": "#ffffff",
        "card": "#ffffff",
        "card2": "#eef5f0",
        "card_hover": "#e6efe8",
        "border": "#d1e3d6",
        "border_light": "#e6efe8",
        "row_alt": "#f7fbf8",
        "row_hover": "#eef5f0",
        "text": "#0f2318",
        "text_dim": "#5a7a65",
        "text_faint": "#8aa89a",
        "purple": "#047857",
        "purple_hover": "#065f46",
        "purple_muted": "#d1fae5",
        "pink": "#65a30d",
        "cyan": "#0e7490",
        "cyan_dim": "#0c6580",
        "green": "#059669",
        "lime": "#65a30d",
        "orange": "#d97706",
        "red": "#dc2626",
        "red_hover": "#b91c1c",
        "blue": "#0284c7",
        "yellow": "#ca8a04",
        "neon": "#0284c7",
        "option_bg": "#ffffff",
        "option_btn": "#d1e3d6",
        "option_drop": "#ffffff",
        "entry_bg": "#ffffff",
        "entry_border": "#c1d9c8",
        "entry_focus": "#047857",
        "progress_bg": "#dbeee3",
        "progress_fg": "#047857",
        "nav_active": "#d9f2e4",
        "chip_good": "#d7f2e2",
        "chip_bad": "#fbdfdf",
        "chip_neutral": "#eef5f0",
        "hero_grad": ["#10b981", "#0ea5a5", "#0284c7", "#075985"],
        "shadow": "#00000014",
        "flet_mode": "light",
    },
}


def apply_theme(page, theme_name: str):
    """Apply a Flet theme to the page."""
    c = THEMES[theme_name]
    page.theme_mode = "dark" if theme_name == "dark" else "light"
    page.bgcolor = c["main_bg"]
    page.theme = page.theme
