"""
Dark theme definition for the HMAxEMA Scanner GUI — Emerald.

Palette inspired by deep green / teal trading terminals.
Matches the scanner_report HTML CSS variables.

``THEMES`` keeps its mapping shape (name -> color dict) so every consumer
keeps working unchanged; "dark" is the only variant.
"""

THEMES = {
    "dark": {
        # Base surfaces — deep black-green
        "root_bg": "#0a0f0c",
        "rail_bg": "#070a08",
        "side_bg": "#0d1512",
        "main_bg": "#0a0f0c",
        "panel_bg": "#111a16",
        # Cards / rows — slightly elevated surfaces
        "card": "#152019",
        "card2": "#1a2a22",
        "card_hover": "#1f3328",
        "border": "#1f3328",
        "border_light": "#2a4a3a",
        "row_alt": "#0d1512",
        "row_hover": "#1a2a22",
        # Text
        "text": "#e8f5ee",
        "text_dim": "#8ca89a",
        "text_faint": "#5a7a6a",
        # Accents — emerald / teal family + supporting hues
        "purple": "#10b981",
        "purple_hover": "#34d399",
        "purple_muted": "#052e16",
        "pink": "#a78bfa",
        "cyan": "#14b8a6",
        "cyan_dim": "#0f766e",
        "green": "#10b981",
        "lime": "#a3e635",
        "orange": "#fb923c",
        "red": "#f87171",
        "red_hover": "#ef4444",
        "blue": "#2dd4bf",
        "yellow": "#facc15",
        "neon": "#10b981",
        "macd": "#2dd4bf",
        "fund": "#ffe600",
        "hero_value": "#ffffff",
        # Controls
        "option_bg": "#1a2a22",
        "option_btn": "#1f3328",
        "option_drop": "#152019",
        "entry_bg": "#152019",
        "entry_border": "#1f3328",
        "entry_focus": "#10b981",
        "progress_bg": "#1a2a22",
        "progress_fg": "#10b981",
        "nav_active": "#1a2a22",
        "chip_good": "#0b2f2e",
        "chip_bad": "#3a1519",
        "chip_neutral": "#1a2a22",
        # Hero gradient — emerald into teal, dissolving to black-green
        "hero_grad": ["#059669", "#10b981", "#0d9488", "#0a0f0c"],
        "hero_title": "#ffffff",
        "hero_sub": "#d1fae5",
        # Soft top-left radial wash over the main area
        "bg_radial": "#0d3328",
        # Text sitting on an accent (green) button
        "on_accent": "#052e16",
        # Right-panel profile avatar
        "avatar_bg": "#052e16",
        "avatar_text": "#6ee7b7",
        "avatar_border": "#10b981",
        "shadow": "#00000066",
    },
}
