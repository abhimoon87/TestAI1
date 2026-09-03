"""
Theme definitions for the HMAxEMA Scanner GUI — Aurora v3.

Palette inspired by premium trading terminals (linear.app / vercel dark).
Matches scanner_report HTML CSS variables in both variants.
"""

import customtkinter as ctk

from .settings_store import save_settings

THEMES = {
    "dark": {
        "ctk_mode": "dark",
        # Base surfaces — deep forest with subtle blue undertone for depth
        "root_bg": "#080f0c",
        "rail_bg": "#060d09",
        "side_bg": "#0b1a13",
        "main_bg": "#080f0c",
        "panel_bg": "#0e241b",
        # Cards / rows — layered with 1px border for glass effect
        "card": "#0f271c",
        "card2": "#143323",
        "card_hover": "#173a28",
        "border": "#1e4a2f",
        "border_light": "#244a32",
        "row_alt": "#0c2016",
        "row_hover": "#143323",
        # Text — warm off-white for readability, muted sage for secondary
        "text": "#dff0e2",
        "text_dim": "#6b9a7a",
        "text_faint": "#4a6b54",
        # Accents — neon mint primary, amber secondary
        "purple": "#00e67a",
        "purple_hover": "#00ffa0",
        "purple_muted": "#0a3d24",
        "pink": "#c8ff00",
        "cyan": "#22d3c4",
        "cyan_dim": "#0e8f86",
        "green": "#00e67a",
        "lime": "#c8ff00",
        "orange": "#ff9f1c",
        "red": "#ff4d4d",
        "red_hover": "#d62b2b",
        "blue": "#3b9eff",
        "yellow": "#ffd23f",
        # Controls
        "option_bg": "#143323",
        "option_btn": "#1e4a2f",
        "option_drop": "#0f271c",
        "entry_bg": "#0f271c",
        "entry_border": "#1e4a2f",
        "entry_focus": "#00e67a",
        "progress_bg": "#143323",
        "progress_fg": "#00e67a",
        "nav_active": "#143323",
        "chip_good": "#0d3320",
        "chip_bad": "#3d1a1a",
        "chip_neutral": "#1a2e1f",
        # Hero gradient — mint → teal → deep forest
        "hero_grad": ["#00e67a", "#14c9b0", "#0a8a7a", "#062419"],
        # Shadows
        "shadow": "#00000066",
    },
    "light": {
        "ctk_mode": "light",
        # Base surfaces — warm paper with sage tint
        "root_bg": "#f0f5f1",
        "rail_bg": "#e6efe8",
        "side_bg": "#f7fbf8",
        "main_bg": "#f0f5f1",
        "panel_bg": "#ffffff",
        # Cards / rows
        "card": "#ffffff",
        "card2": "#eef5f0",
        "card_hover": "#e6efe8",
        "border": "#d1e3d6",
        "border_light": "#e6efe8",
        "row_alt": "#f7fbf8",
        "row_hover": "#eef5f0",
        # Text
        "text": "#0f2318",
        "text_dim": "#5a7a65",
        "text_faint": "#8aa89a",
        # Accents — deeper for light contrast
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
        # Controls
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
    },
}


def apply_theme(app, theme_name: str):
    """Apply a theme to the entire application."""
    ctk.set_appearance_mode(THEMES[theme_name]["ctk_mode"])
    app.current_theme = theme_name
    app.theme_colors = THEMES[theme_name]
    app.settings["theme"] = theme_name
    save_settings(app.settings)
