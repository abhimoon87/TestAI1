"""
Settings persistence for the HMAxEMA Scanner GUI.

Handles loading/saving user settings to settings.json and defines the
defaults that mirror the Pine Script indicator inputs.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import TypedDict

from ..shared.constants import RESULT_COLS

logger = logging.getLogger(__name__)


class ScannerSettings(TypedDict, total=False):
    """Canonical key list for the settings dict shared by GUI, CLI and engine.

    The runtime representation stays a plain ``dict`` (zero-cost adoption —
    every existing ``settings.get(...)`` call site keeps working), while the
    TypedDict gives static checkers and IDEs autocomplete + typo detection.
    ``DEFAULT_SETTINGS`` below is the single source of truth for values;
    ``_sanitize_settings`` is the single source of truth for validation.
    """

    # Moving Averages
    fast_ma_type: str  # HMA | EMA | SMA | KAMA | VWMA
    fast_ma_len: int
    slow_ma_type: str  # HMA | EMA | SMA | KAMA | VWMA
    slow_ma_len: int
    # Technical Analysis
    rsi_len: int
    rs_length: int
    vol_ma_len: int
    atr_len: int
    # Relative Strength
    index_symbol: str
    # Volume Profile
    vp_lookback: int
    vp_rows: int
    vp_width: int
    # Sideways Filter
    adx_len: int
    adx_threshold: float
    chop_len: int
    chop_threshold: float
    slope_ma_type: str
    slope_ma_len: int
    slope_lookback: int
    flat_threshold: float
    sideways_strong_move_pct: float
    volume_participation_len: int
    # Step Channel
    sc_pivot_len: int
    sc_bands_mult: float
    # MA Crossover
    crossover_lookback: int
    # Entry gate (mirrors backtest engine): require ADX >= this for entry signal
    min_adx_entry: float
    # Scanner
    min_score: float  # 0-100
    data_period: str  # 6mo | 1y | 2y
    timeframe: str  # D | W | M
    trend_filter: str  # All | Bullish Only | Bearish Only
    # Dead-symbol cache
    negative_cache_ttl_hours: int
    # Scan hygiene: warn when a universe member's data is this old (days)
    stale_member_max_age_days: float
    # UI
    theme: str
    reduce_motion: bool  # suppress decorative transitions/animations
    # Provider toggles
    use_market_sentiment: bool
    use_social_sentiment: bool
    use_indian_market: bool
    use_indian_fundamentals: bool
    use_insider_data: bool
    use_macro_data: bool
    # Entry mode: "classic" (original), "high_probability" (optimized), "custom"
    entry_mode: str
    hp_counter_signal_penalty: bool
    hp_freshness_max_bars: int
    hp_volume_confirmation: bool


SCANNER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(SCANNER_DIR, "settings.json")
API_CONFIG_FILE = os.path.join(SCANNER_DIR, "api_config.json")

# ── Default Settings (mirrors Pine Script indicator) ─────────────────────────
DEFAULT_SETTINGS: ScannerSettings = {
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
    "sideways_strong_move_pct": 5.0,
    "volume_participation_len": 5,
    # Step Channel
    "sc_pivot_len": 3,
    "sc_bands_mult": 0.6,
    # MA Crossover
    "crossover_lookback": 20,
    # Entry gate (mirrors backtest engine): require ADX >= this for entry signal
    "min_adx_entry": 20.0,
    # Scanner
    "min_score": 50.0,
    "data_period": "1y",
    "timeframe": "D",
    "trend_filter": "All",
    # Dead-symbol cache
    "negative_cache_ttl_hours": 24,
    # Scan hygiene: warn when a universe member's data is this old (days)
    "stale_member_max_age_days": 45.0,
    # UI
    "theme": "dark",
    "reduce_motion": False,
    # Provider toggles
    "use_market_sentiment": True,
    "use_social_sentiment": True,
    "use_indian_market": True,
    "use_indian_fundamentals": True,
    "use_insider_data": True,
    "use_macro_data": True,
    # Entry mode: "classic" (original), "high_probability" (optimized), "custom"
    "entry_mode": "classic",
    "hp_counter_signal_penalty": True,
    "hp_freshness_max_bars": 2,
    "hp_volume_confirmation": True,
}


# ── API Key Management ───────────────────────────────────────────────────────

# All known API keys with descriptions and free tier info
API_KEY_REGISTRY = {
    # Finance providers
    "FINNHUB_API_KEY": {
        "description": "Finnhub — Institutional-grade fundamental data",
        "free_tier": "60 calls/min",
        "url": "https://finnhub.io/register",
        "category": "finance",
    },
    "ALPHA_VANTAGE_API_KEY": {
        "description": "Alpha Vantage — Technical indicators & fundamentals",
        "free_tier": "25 calls/day",
        "url": "https://www.alphavantage.co/support/#api-key",
        "category": "finance",
    },
    "TWELVE_DATA_API_KEY": {
        "description": "Twelve Data — Stock market data (real-time & historical)",
        "free_tier": "800 calls/day",
        "url": "https://twelvedata.com/register",
        "category": "finance",
    },
    "MARKETAUX_API_KEY": {
        "description": "MarketAux — Live stock market news with ticker tags",
        "free_tier": "100 calls/day",
        "url": "https://marketaux.com/register",
        "category": "news",
    },
    "NEWS_API_KEY": {
        "description": "NewsAPI — 80k+ news sources worldwide",
        "free_tier": "100 calls/day",
        "url": "https://newsapi.org/register",
        "category": "news",
    },
    "GNEWS_API_KEY": {
        "description": "GNews — News search API",
        "free_tier": "100 calls/day",
        "url": "https://gnews.io/register",
        "category": "news",
    },
    # NLP/Sentiment
    "MEANINGCLOUD_API_KEY": {
        "description": "MeaningCloud — Multilingual sentiment analysis",
        "free_tier": "500 calls/day",
        "url": "https://www.meaningcloud.com/developer/login",
        "category": "nlp",
    },
    "NLPCLOUD_API_KEY": {
        "description": "NLP Cloud — NER, sentiment, classification",
        "free_tier": "500 calls/day",
        "url": "https://nlpcloud.com/register",
        "category": "nlp",
    },
    "HF_API_KEY": {
        "description": "Hugging Face — Open-source sentiment models",
        "free_tier": "300 calls/day",
        "url": "https://huggingface.co/settings/tokens",
        "category": "nlp",
    },
    "GROQ_API_KEY": {
        "description": "Groq — Fast LLM inference for analysis",
        "free_tier": "14,400 req/day",
        "url": "https://console.groq.com/keys",
        "category": "nlp",
    },
    "TWITTER_API_KEY": {
        "description": "GetXAPI — Twitter/X posts for social sentiment",
        "free_tier": "Paid (trial available)",
        "url": "https://getxapi.com/",
        "category": "nlp",
    },
    # Premium finance
    "EOD_API_KEY": {
        "description": "EOD Historical Data — 150+ exchanges, fundamentals",
        "free_tier": "20 calls/day",
        "url": "https://eodhistoricaldata.com/register",
        "category": "finance",
    },
    "FMP_API_KEY": {
        "description": "Financial Modeling Prep — Financial statements & ratios",
        "free_tier": "250 calls/day",
        "url": "https://financialmodelingprep.com/register",
        "category": "finance",
    },
    "IEX_API_KEY": {
        "description": "IEX Cloud — Real-time US + India market data",
        "free_tier": "50,000 calls/month",
        "url": "https://iexcloud.io/register",
        "category": "finance",
    },
    "POLYGON_API_KEY": {
        "description": "Polygon — Historical stock market data",
        "free_tier": "5 req/min",
        "url": "https://polygon.io/register",
        "category": "finance",
    },
    "STOCKDATA_API_KEY": {
        "description": "StockData — Real-time news & sentiment API",
        "free_tier": "500 calls/month",
        "url": "https://stockdata.org/register",
        "category": "finance",
    },
    "STYVIO_API_KEY": {
        "description": "Styvio — Stock sentiment scores",
        "free_tier": "Free tier available",
        "url": "https://styvio.com/register",
        "category": "finance",
    },
    # Insider data
    "ALETHEIA_API_KEY": {
        "description": "Aletheia — Insider trading data",
        "free_tier": "100 calls/day",
        "url": "https://aletheia.com/register",
        "category": "insider",
    },
    "CONGRESS_API_KEY": {
        "description": "CongressInvests — Congressional stock trades",
        "free_tier": "100 calls/day",
        "url": "https://congressinvests.com/register",
        "category": "insider",
    },
    # Macro
    "FRED_API_KEY": {
        "description": "FRED — Federal Reserve economic data",
        "free_tier": "120 calls/min",
        "url": "https://fred.stlouisfed.org/docs/api/api_key.html",
        "category": "macro",
    },
    "ECONPULSE_API_KEY": {
        "description": "EconPulse — Live economic data",
        "free_tier": "100 calls/day",
        "url": "https://econpulse.com/register",
        "category": "macro",
    },
    "ECONDB_API_KEY": {
        "description": "Econdb — Global macroeconomic data",
        "free_tier": "Free tier available",
        "url": "https://www.econdb.com/register",
        "category": "macro",
    },
    # Environment/ESG
    "CARBON_INTERFACE_API_KEY": {
        "description": "Carbon Interface — CO2 emissions estimates",
        "free_tier": "100 calls/month",
        "url": "https://www.carboninterface.com/register",
        "category": "esg",
    },
    "CLIMATIQ_API_KEY": {
        "description": "Climatiq — Environmental footprint calculation",
        "free_tier": "1,000 calls/month",
        "url": "https://www.climatiq.io/register",
        "category": "esg",
    },
    # Shariah
    "HALAL_API_KEY": {
        "description": "Halal Terminal — Shariah-compliant stock screening",
        "free_tier": "Free tier available",
        "url": "https://halalterminal.com/register",
        "category": "shariah",
    },
    # Time series
    "TIMEDOOR_API_KEY": {
        "description": "Time Door — Time series anomaly detection",
        "free_tier": "Free tier available",
        "url": "https://timedoor.com/register",
        "category": "ml",
    },
}


def load_api_config() -> dict:
    """Load API keys from config file and environment variables."""
    config = {}

    # Load from config file
    if os.path.exists(API_CONFIG_FILE):
        try:
            with open(API_CONFIG_FILE) as f:
                config = json.load(f)
        except Exception as e:
            logger.info("Failed to load API config: %s", e)

    # Environment variables override config file
    for key in API_KEY_REGISTRY:
        env_val = os.environ.get(key)
        if env_val:
            config[key] = env_val

    return config


def get_api_key(key_name: str, config: dict | None = None) -> str | None:
    """
    Get an API key by name. Checks config dict, then environment variable.

    Args:
        key_name: API key name (e.g., "FINNHUB_API_KEY")
        config: Optional pre-loaded config dict

    Returns:
        API key string or None
    """
    if config and key_name in config:
        return config[key_name]
    return os.environ.get(key_name)


def _sanitize_settings(saved: dict) -> dict:
    """Validate and coerce loaded settings against DEFAULT_SETTINGS.

    Unknown keys are dropped (except ``ui_*`` table-view prefs, which are
    validated separately); known keys are coerced to the default's type and
    checked against small enum sets. Out-of-range numerics fall back to the
    default so a corrupt file can never brick Save or the engine.
    """
    cleaned: dict = {}
    for key, val in saved.items():
        if key not in DEFAULT_SETTINGS:
            if (
                key == "ui_sort_col"
                and isinstance(val, int)
                and 0 <= val < len(RESULT_COLS)
            ):
                cleaned[key] = val
            elif key == "ui_sort_reverse" and isinstance(val, bool):
                cleaned[key] = val
            elif key == "ui_page_size" and isinstance(val, int) and 0 < val <= 500:
                cleaned[key] = val
            elif key == "ui_rating_filter" and val in (
                "ALL",
                "EXCELLENT",
                "GOOD",
                "MODERATE",
                "POOR",
            ):
                cleaned[key] = val
            elif key == "universe" and isinstance(val, str) and val:
                cleaned[key] = val
            else:
                logger.debug("Dropping unknown setting %r", key)
            continue
        default = DEFAULT_SETTINGS[key]
        try:
            if isinstance(default, bool):
                if isinstance(val, bool):
                    cleaned[key] = val
                elif isinstance(val, str) and val.lower() in ("true", "false"):
                    cleaned[key] = val.lower() == "true"
                else:
                    raise ValueError(key)
            elif isinstance(default, int) and not isinstance(default, bool):
                ival = int(float(val))
                if key == "ui_sort_col":
                    raise ValueError(key)  # not a default key; guarded above
                cleaned[key] = ival
            elif isinstance(default, float):
                fval = float(val)
                if key == "min_score" and not 0 <= fval <= 100:
                    raise ValueError(key)
                if (
                    key in ("negative_cache_ttl_hours", "stale_member_max_age_days")
                    and fval <= 0
                ):
                    raise ValueError(key)
                cleaned[key] = fval
            elif isinstance(default, str):
                sval = str(val)
                enums = {
                    "theme": ("dark",),
                    "entry_mode": ("classic", "high_probability", "custom"),
                    "fast_ma_type": ("HMA", "EMA", "SMA", "KAMA", "VWMA"),
                    "slow_ma_type": ("HMA", "EMA", "SMA", "KAMA", "VWMA"),
                    "slope_ma_type": ("HMA", "EMA", "SMA", "KAMA", "VWMA"),
                    "data_period": ("6mo", "1y", "2y"),
                    "timeframe": ("D", "W", "M"),
                    "trend_filter": ("All", "Bullish Only", "Bearish Only"),
                }
                if key in enums and sval not in enums[key]:
                    raise ValueError(key)
                cleaned[key] = sval
            else:
                cleaned[key] = val
        except (ValueError, TypeError):
            logger.debug("Dropping invalid setting %r=%r", key, val)
    return cleaned


def load_settings() -> ScannerSettings:
    """Load settings from JSON file, falling back to defaults."""
    settings: ScannerSettings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                settings.update(_sanitize_settings(saved))
            else:
                logger.warning("Settings file is not a JSON object; using defaults")
        except Exception as e:
            logger.warning("Failed to load settings: %s", e)
    return settings


def save_settings(settings: ScannerSettings):
    """Save settings to JSON file atomically."""
    try:
        dir_name = os.path.dirname(SETTINGS_FILE) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(settings, f, indent=2)
            os.replace(tmp_path, SETTINGS_FILE)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.debug("Failed to clean up temp settings file", exc_info=True)
            raise
    except Exception as e:
        logger.warning("Failed to save settings: %s", e)
        raise
