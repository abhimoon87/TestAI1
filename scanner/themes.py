"""
Theme definitions for the HMAxEMA Scanner GUI.

Palette matches the scanner_report HTML CSS variables (--bg/--surface/
--green/--lime/...) in both dark and light variants.
"""

import customtkinter as ctk

from .settings_store import save_settings

THEMES = {
    "dark": {
        "ctk_mode": "dark",
        # Base surfaces (report CSS vars: --bg/--surface/--surface2/--border)
        "root_bg": "#0a1a10",
        "rail_bg": "#071309",
        "side_bg": "#0c1e13",
        "main_bg": "#0a1a10",
        "panel_bg": "#0f2a1a",
        # Cards / rows
        "card": "#0f2a1a",
        "card2": "#153520",
        "border": "#1a4a2a",
        "row_alt": "#0d2114",
        # Text (--text / --text-dim)
        "text": "#c8d8c0",
        "text_dim": "#6a8a6a",
        # Accents (--green --lime --orange --red --blue --cyan)
        "purple": "#00ff88", "purple_hover": "#33ffaa",   # primary accent
        "pink": "#aaff00",                                 # secondary accent
        "cyan": "#00ddcc",
        "green": "#00ff88", "lime": "#aaff00",
        "orange": "#ffaa00", "red": "#ff4444",
        "blue": "#00aaff",
        # Controls
        "option_bg": "#153520", "option_btn": "#1a4a2a", "option_drop": "#0f2a1a",
        "entry_bg": "#153520", "entry_border": "#1a4a2a",
        "progress_bg": "#153520", "progress_fg": "#00ff88",
        "nav_active": "#0f3320",
        "chip_good": "#0b3a20", "chip_bad": "#3a1414",
        # Hero gradient (green → cyan → deep teal)
        "hero_grad": ["#00ff88", "#00ddcc", "#0088aa", "#06251a"],
    },
    "light": {
        "ctk_mode": "light",
        # Base surfaces
        "root_bg": "#eef6f0",
        "rail_bg": "#e2eee6",
        "side_bg": "#f4faf6",
        "main_bg": "#fbfdfb",
        "panel_bg": "#ffffff",
        # Cards / rows
        "card": "#ffffff",
        "card2": "#e8f4ec",
        "border": "#cfe4d7",
        "row_alt": "#f1f9f4",
        # Text
        "text": "#12281c",
        "text_dim": "#55705f",
        # Accents (readable daylight variants of report hues)
        "purple": "#047857", "purple_hover": "#036c4e",
        "pink": "#65a30d",
        "cyan": "#0e7490",
        "green": "#059669", "lime": "#65a30d",
        "orange": "#d97706", "red": "#dc2626",
        "blue": "#0284c7",
        # Controls
        "option_bg": "#ffffff", "option_btn": "#cfe4d7", "option_drop": "#ffffff",
        "entry_bg": "#ffffff", "entry_border": "#b7d8c5",
        "progress_bg": "#dbeee3", "progress_fg": "#047857",
        "nav_active": "#d9f2e4",
        "chip_good": "#d7f2e2", "chip_bad": "#fbdfdf",
        "hero_grad": ["#10b981", "#0ea5a5", "#0284c7", "#075985"],
    },
}


def apply_theme(app, theme_name: str):
    """Apply a theme to the entire application."""
    ctk.set_appearance_mode(THEMES[theme_name]["ctk_mode"])
    app.current_theme = theme_name
    app.theme_colors = THEMES[theme_name]
    app.settings["theme"] = theme_name
    save_settings(app.settings)
