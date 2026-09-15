"""
Theme definitions for the HMAxEMA Scanner GUI — Aurora v3 (Flet edition).

Palette inspired by premium trading terminals (linear.app / vercel dark).
Matches scanner_report HTML CSS variables in both variants.

Flet uses ft.Theme and ft.ColorScheme for theming. This module provides
color dicts consumed by the Flet UI code.
"""

THEMES = {
    "dark": {
        # Base surfaces — deep slate / charcoal with blue undertone
        "root_bg": "#0f1117",
        "rail_bg": "#090a0e",
        "side_bg": "#14151c",
        "main_bg": "#0f1117",
        "panel_bg": "#181920",
        # Cards / rows — slightly elevated surfaces
        "card": "#1a1b24",
        "card2": "#22232e",
        "card_hover": "#252633",
        "border": "#2a2b38",
        "border_light": "#38394a",
        "row_alt": "#13141b",
        "row_hover": "#1e1f2a",
        # Text
        "text": "#e8eaf2",
        "text_dim": "#8c92b0",
        "text_faint": "#585d78",
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
        "macd": "#aa88ff",
        "fund": "#ffe600",
        "hero_value": "#ffffff",
        # Controls
        "option_bg": "#1e1f2a",
        "option_btn": "#2a2b38",
        "option_drop": "#1a1b24",
        "entry_bg": "#1a1b24",
        "entry_border": "#2a2b38",
        "entry_focus": "#38bdf8",
        "progress_bg": "#22232e",
        "progress_fg": "#38bdf8",
        "nav_active": "#1e1f2a",
        "chip_good": "#0b2f2e",
        "chip_bad": "#3a1519",
        "chip_neutral": "#1e1f2a",
        # Hero gradient — neon blue into violet, dissolving to charcoal
        "hero_grad": ["#0ea5e9", "#6366f1", "#3b2d8f", "#17171c"],
        "hero_title": "#ffffff",
        "hero_sub": "#d8e8ff",
        # Soft top-left radial wash over the main area
        "bg_radial": "#1b2f4d",
        # Text sitting on an accent (green) button
        "on_accent": "#052e16",
        # Right-panel profile avatar
        "avatar_bg": "#12331f",
        "avatar_text": "#8dffc4",
        "avatar_border": "#22d3ee",
        "shadow": "#00000066",
        # Flet theme mode
        "flet_mode": "dark",
    },
    "light": {
        "root_bg": "#edf2f7",
        "rail_bg": "#e2eaf5",
        "side_bg": "#f4f7fb",
        "main_bg": "#edf2f7",
        "panel_bg": "#ffffff",
        "card": "#ffffff",
        "card2": "#eaf1f8",
        "card_hover": "#e2eaf5",
        "border": "#c8d6e5",
        "border_light": "#dfe8f2",
        "row_alt": "#f4f7fb",
        "row_hover": "#eaf1f8",
        "text": "#0f1b2d",
        "text_dim": "#5a6f8a",
        "text_faint": "#8da0b8",
        "purple": "#047857",
        "purple_hover": "#065f46",
        "purple_muted": "#d1fae5",
        "pink": "#9333ea",
        "cyan": "#0e7490",
        "cyan_dim": "#0c6580",
        "green": "#059669",
        "lime": "#4d7c0f",
        "orange": "#d97706",
        "red": "#dc2626",
        "red_hover": "#b91c1c",
        "blue": "#0284c7",
        "yellow": "#ca8a04",
        "neon": "#0284c7",
        "macd": "#7c3aed",
        "fund": "#92400e",
        "hero_value": "#0f1b2d",
        "option_bg": "#ffffff",
        "option_btn": "#c8d6e5",
        "option_drop": "#ffffff",
        "entry_bg": "#ffffff",
        "entry_border": "#c8d6e5",
        "entry_focus": "#047857",
        "progress_bg": "#d5e3f0",
        "progress_fg": "#047857",
        "nav_active": "#d1f2e4",
        "chip_good": "#d7f2e2",
        "chip_bad": "#fbdfdf",
        "chip_neutral": "#eaf1f8",
        # Hero gradient — deep emerald/teal (kept rich so white hero text
        # stays readable over the whole band, even in the light theme)
        "hero_grad": ["#0f9d7b", "#0d9488", "#0e7490", "#134e4a"],
        "hero_title": "#ffffff",
        "hero_sub": "#d3f9ee",
        # Soft top-left radial wash over the main area
        "bg_radial": "#dcefe3",
        # Text sitting on an accent (green) button
        "on_accent": "#ffffff",
        # Right-panel profile avatar
        "avatar_bg": "#d7f2e2",
        "avatar_text": "#047857",
        "avatar_border": "#0e7490",
        "shadow": "#00000014",
        "flet_mode": "light",
    },
}

