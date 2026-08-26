"""
Settings persistence for the HMAxEMA Scanner GUI.

Handles loading/saving user settings to settings.json and defines the
defaults that mirror the Pine Script indicator inputs.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

SCANNER_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(SCANNER_DIR, "settings.json")

# ── Default Settings (mirrors Pine Script indicator) ─────────────────────────
DEFAULT_SETTINGS = {
    # Moving Averages
    "fast_ma_type": "HMA",
    "fast_ma_len": 40,
    "slow_ma_type": "EMA",
    "slow_ma_len": 50,
    # Technical Analysis
    "rsi_len": 14,
    "rs_length": 14,
    "vol_ma_len": 20,
    "atr_len": 14,
    # Relative Strength
    "index_symbol": "NSEI",
    # Volume Profile
    "vp_lookback": 200,
    "vp_rows": 30,
    "vp_width": 40,
    # Sideways Filter
    "adx_len": 14,
    "adx_threshold": 20.0,
    "chop_len": 14,
    "chop_threshold": 61.8,
    "slope_ma_type": "EMA",
    "slope_ma_len": 50,
    "slope_lookback": 10,
    "flat_threshold": 0.5,
    # Step Channel
    "sc_pivot_len": 3,
    "sc_bands_mult": 0.6,
    # MA Crossover
    "crossover_lookback": 20,
    # Scanner
    "min_score": 50.0,
    "data_period": "1y",
    "timeframe": "D",
    "trend_filter": "All",
    # UI
    "theme": "dark",
}


def load_settings() -> dict:
    """Load settings from JSON file, falling back to defaults."""
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            settings.update(saved)
        except Exception as e:
            logger.debug("Failed to load settings: %s", e)
    return settings


def save_settings(settings: dict):
    """Save settings to JSON file."""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.debug("Failed to save settings: %s", e)
