"""
API Key Setup Script for HMAxEMA Scanner
Configure all provider API keys for enhanced data sources.
"""

import logging
import os
import sys
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Config file location
CONFIG_FILE = Path(__file__).parent / "api_config.json"


def print_header():
    logger.info("=" * 60)
    logger.info("  HMAxEMA Scanner — API Key Setup")
    logger.info("=" * 60)
    logger.info("")


def print_section(title):
    logger.info("\n%s", "─" * 60)
    logger.info("  %s", title)
    logger.info("%s\n", "─" * 60)


def load_config():
    """Load existing config."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.debug("Failed to load API config: %s", e)
    return {}


def save_config(config):
    """Save config to file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("  [OK] Config saved to %s", CONFIG_FILE)
    except Exception as e:
        logger.error("  [ERROR] Could not save config: %s", e)


def set_env_variable(key, value):
    """Set environment variable for current session."""
    os.environ[key] = value


def setup_key(name, description, free_tier, url, config):
    """Setup a single API key interactively."""
    key = input(f"  Paste your {name} (or press Enter to skip): ").strip()
    if key:
        set_env_variable(name, key)
        config[name] = key
        logger.info("  [OK] %s set", name)
        return key
    return None


def get_current_status():
    """Check current API key status."""
    config = load_config()
    status = {}
    for key_name in [
        "FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEY", "TWELVE_DATA_API_KEY",
        "MARKETAUX_API_KEY", "NEWS_API_KEY", "GNEWS_API_KEY",
        "MEANINGCLOUD_API_KEY", "NLPCLOUD_API_KEY", "HF_API_KEY", "GROQ_API_KEY",
        "EOD_API_KEY", "FMP_API_KEY", "IEX_API_KEY", "POLYGON_API_KEY",
        "STOCKDATA_API_KEY", "STYVIO_API_KEY",
        "ALETHEIA_API_KEY", "CONGRESS_API_KEY",
        "FRED_API_KEY", "ECONPULSE_API_KEY",
        "CARBON_INTERFACE_API_KEY", "CLIMATIQ_API_KEY",
        "HALAL_API_KEY", "TIMEDOOR_API_KEY",
    ]:
        val = config.get(key_name, "") or os.environ.get(key_name, "")
        status[key_name] = val
    return status


def setup_finance_keys(config):
    """Setup finance provider API keys."""
    print_section("FINANCE PROVIDERS")

    logger.info("  1. Finnhub — Institutional-grade fundamental data")
    logger.info("     Free tier: 60 calls/min | https://finnhub.io/register")
    setup_key("FINNHUB_API_KEY", "", "", "", config)

    logger.info("\n  2. Alpha Vantage — Technical indicators & fundamentals")
    logger.info("     Free tier: 25 calls/day | https://www.alphavantage.co/support/#api-key")
    setup_key("ALPHA_VANTAGE_API_KEY", "", "", "", config)

    logger.info("\n  3. Twelve Data — Stock market data (real-time & historical)")
    logger.info("     Free tier: 800 calls/day | https://twelvedata.com/register")
    setup_key("TWELVE_DATA_API_KEY", "", "", "", config)

    logger.info("\n  4. EOD Historical Data — 150+ exchanges, fundamentals")
    logger.info("     Free tier: 20 calls/day | https://eodhistoricaldata.com/register")
    setup_key("EOD_API_KEY", "", "", "", config)

    logger.info("\n  5. Financial Modeling Prep — Financial statements & ratios")
    logger.info("     Free tier: 250 calls/day | https://financialmodelingprep.com/register")
    setup_key("FMP_API_KEY", "", "", "", config)

    logger.info("\n  6. IEX Cloud — Real-time US + India market data")
    logger.info("     Free tier: 50,000 calls/month | https://iexcloud.io/register")
    setup_key("IEX_API_KEY", "", "", "", config)

    logger.info("\n  7. Polygon — Historical stock market data")
    logger.info("     Free tier: 5 req/min | https://polygon.io/register")
    setup_key("POLYGON_API_KEY", "", "", "", config)

    logger.info("\n  8. StockData — Real-time news & sentiment API")
    logger.info("     Free tier: 500 calls/month | https://stockdata.org/register")
    setup_key("STOCKDATA_API_KEY", "", "", "", config)

    logger.info("\n  9. Styvio — Stock sentiment scores")
    logger.info("     Free tier available | https://styvio.com/register")
    setup_key("STYVIO_API_KEY", "", "", "", config)


def setup_news_keys(config):
    """Setup news provider API keys."""
    print_section("NEWS PROVIDERS")

    logger.info("  1. MarketAux — Live stock market news with ticker tags")
    logger.info("     Free tier: 100 calls/day | https://marketaux.com/register")
    setup_key("MARKETAUX_API_KEY", "", "", "", config)

    logger.info("\n  2. NewsAPI — 80k+ news sources worldwide")
    logger.info("     Free tier: 100 calls/day | https://newsapi.org/register")
    setup_key("NEWS_API_KEY", "", "", "", config)

    logger.info("\n  3. GNews — News search API")
    logger.info("     Free tier: 100 calls/day | https://gnews.io/register")
    setup_key("GNEWS_API_KEY", "", "", "", config)


def setup_nlp_keys(config):
    """Setup NLP/Sentiment provider API keys."""
    print_section("NLP / SENTIMENT PROVIDERS")

    logger.info("  1. MeaningCloud — Multilingual sentiment analysis")
    logger.info("     Free tier: 500 calls/day | https://www.meaningcloud.com/developer/login")
    setup_key("MEANINGCLOUD_API_KEY", "", "", "", config)

    logger.info("\n  2. NLP Cloud — NER, sentiment, classification")
    logger.info("     Free tier: 500 calls/day | https://nlpcloud.com/register")
    setup_key("NLPCLOUD_API_KEY", "", "", "", config)

    logger.info("\n  3. Hugging Face — Open-source sentiment models")
    logger.info("     Free tier: 300 calls/day | https://huggingface.co/settings/tokens")
    setup_key("HF_API_KEY", "", "", "", config)

    logger.info("\n  4. Groq — Fast LLM inference for analysis")
    logger.info("     Free tier: 14,400 req/day | https://console.groq.com/keys")
    setup_key("GROQ_API_KEY", "", "", "", config)


def setup_insider_keys(config):
    """Setup insider data provider API keys."""
    print_section("INSIDER DATA PROVIDERS")

    logger.info("  1. Aletheia — Insider trading data")
    logger.info("     Free tier: 100 calls/day | https://aletheia.com/register")
    setup_key("ALETHEIA_API_KEY", "", "", "", config)

    logger.info("\n  2. CongressInvests — Congressional stock trades")
    logger.info("     Free tier: 100 calls/day | https://congressinvests.com/register")
    setup_key("CONGRESS_API_KEY", "", "", "", config)


def setup_macro_keys(config):
    """Setup macro data provider API keys."""
    print_section("MACRO DATA PROVIDERS")

    logger.info("  1. FRED — Federal Reserve economic data")
    logger.info("     Free tier: 120 calls/min | https://fred.stlouisfed.org/docs/api/api_key.html")
    setup_key("FRED_API_KEY", "", "", "", config)

    logger.info("\n  2. EconPulse — Live economic data")
    logger.info("     Free tier: 100 calls/day | https://econpulse.com/register")
    setup_key("ECONPULSE_API_KEY", "", "", "", config)


def setup_esg_keys(config):
    """Setup ESG/Environment provider API keys."""
    print_section("ESG / ENVIRONMENT PROVIDERS")

    logger.info("  1. Carbon Interface — CO2 emissions estimates")
    logger.info("     Free tier: 100 calls/month | https://www.carboninterface.com/register")
    setup_key("CARBON_INTERFACE_API_KEY", "", "", "", config)

    logger.info("\n  2. Climatiq — Environmental footprint calculation")
    logger.info("     Free tier: 1,000 calls/month | https://www.climatiq.io/register")
    setup_key("CLIMATIQ_API_KEY", "", "", "", config)


def setup_shariah_keys(config):
    """Setup Shariah screening provider API keys."""
    print_section("SHARIAH SCREENING PROVIDERS")

    logger.info("  1. Halal Terminal — Shariah-compliant stock screening")
    logger.info("     Free tier available | https://halalterminal.com/register")
    setup_key("HALAL_API_KEY", "", "", "", config)


def test_providers():
    """Test if the providers work with current keys."""
    print_section("TESTING PROVIDERS")

    try:
        from scanner.data_providers import DataProvider

        provider = DataProvider()
        test_ticker = "RELIANCE"

        logger.info("  Testing fundamentals for %s...", test_ticker)
        fund = provider.fetch_fundamentals(test_ticker)

        if fund:
            logger.info("  [OK] Provider: %s", provider.last_provider)
            logger.info("       P/E Ratio: %s", fund.get('pe_ratio', 'N/A'))
            logger.info("       EPS Growth: %s", fund.get('eps_growth', 'N/A'))
            logger.info("       Revenue Growth: %s", fund.get('rev_growth', 'N/A'))
            return True
        else:
            logger.warning("  [WARN] No data fetched (using fallback)")
            return False

    except Exception as e:
        logger.error("  [ERROR] Test failed: %s", e)
        return False


def print_permanent_instructions():
    """Print instructions for permanent setup."""
    print_section("PERMANENT SETUP (Optional)")
    logger.info("  To make keys permanent, add them to your system:")
    logger.info("")
    logger.info("  Windows (PowerShell):")
    logger.info('    $env:FINNHUB_API_KEY="your_key"')
    logger.info('    $env:ALPHA_VANTAGE_API_KEY="your_key"')
    logger.info("")
    logger.info("  Windows (System Environment Variables):")
    logger.info("    1. Search 'Environment Variables' in Start Menu")
    logger.info("    2. Click 'Environment Variables'")
    logger.info("    3. Add new User variables")
    logger.info("")
    logger.info("  macOS/Linux:")
    logger.info('    export FINNHUB_API_KEY="your_key"')
    logger.info('    export ALPHA_VANTAGE_API_KEY="your_key"')
    logger.info("    # Add to ~/.bashrc or ~/.zshrc for persistence")


def main():
    """Main setup function."""
    print_header()

    # Show current status
    status = get_current_status()
    logger.info("  Current API Key Status:")
    for key_name, val in status.items():
        logger.info("    %-30s: %s", key_name, "SET" if val else "NOT SET")

    # Setup keys
    config = load_config()

    setup_finance_keys(config)
    setup_news_keys(config)
    setup_nlp_keys(config)
    setup_insider_keys(config)
    setup_macro_keys(config)
    setup_esg_keys(config)
    setup_shariah_keys(config)

    # Save config
    if config:
        save_config(config)

    # Test providers
    test_providers()

    # Print permanent setup instructions
    print_permanent_instructions()

    print_section("SETUP COMPLETE")
    logger.info("  API keys are configured for this session.")
    logger.info("  Restart the scanner to use the new providers.")
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n  [CANCELLED] Setup cancelled.")
        sys.exit(0)
