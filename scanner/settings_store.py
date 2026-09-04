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
API_CONFIG_FILE = os.path.join(SCANNER_DIR, "api_config.json")

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
    # Provider toggles
    "use_market_sentiment": True,
    "use_social_sentiment": True,
    "use_indian_market": True,
    "use_indian_fundamentals": True,
    "use_insider_data": True,
    "use_macro_data": True,
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
            with open(API_CONFIG_FILE, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.debug("Failed to load API config: %s", e)

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
