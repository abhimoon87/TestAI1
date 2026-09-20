"""
Dark theme definition for the HMAxEMA Scanner GUI — Aurora v3.

Palette inspired by premium trading terminals (linear.app / vercel dark).
Matches the scanner_report HTML CSS variables.

``THEMES`` keeps its mapping shape (name -> color dict) so every consumer
keeps working unchanged; "dark" is the only variant.
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
    },
}

