"""
API Key Setup Script for HMAxEMA Scanner
Helps you configure Finnhub and Alpha Vantage API keys for enhanced data.
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

def get_current_status():
    """Check current API key status."""
    config = load_config()
    
    finnhub_key = config.get("FINNHUB_API_KEY", "") or os.environ.get("FINNHUB_API_KEY", "")
    av_key = config.get("ALPHA_VANTAGE_API_KEY", "") or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    
    return {
        "FINNHUB_API_KEY": finnhub_key,
        "ALPHA_VANTAGE_API_KEY": av_key
    }

def setup_finnhub():
    """Setup Finnhub API key."""
    print_section("FINNHUB SETUP")
    logger.info("  Finnhub provides institutional-grade fundamental data.")
    logger.info("  Free tier: 60 API calls per minute")
    logger.info("")
    logger.info("  Steps to get your API key:")
    logger.info("  1. Go to: https://finnhub.io/register")
    logger.info("  2. Sign up with your email")
    logger.info("  3. Verify your email")
    logger.info("  4. Go to Dashboard → API Keys")
    logger.info("  5. Copy your API key")
    logger.info("")
    
    key = input("  Paste your Finnhub API key (or press Enter to skip): ").strip()
    
    if key:
        set_env_variable("FINNHUB_API_KEY", key)
        logger.info("  [OK] FINNHUB_API_KEY set for this session")
        return key
    else:
        logger.info("  [SKIP] Finnhub setup skipped")
        return None

def setup_alpha_vantage():
    """Setup Alpha Vantage API key."""
    print_section("ALPHA VANTAGE SETUP")
    logger.info("  Alpha Vantage provides technical indicators and fundamentals.")
    logger.info("  Free tier: 25 API calls per day")
    logger.info("")
    logger.info("  Steps to get your API key:")
    logger.info("  1. Go to: https://www.alphavantage.co/support/#api-key")
    logger.info("  2. Fill in the form (name, email, usage)")
    logger.info("  3. Click 'Get Free API Key'")
    logger.info("  4. Check your email for the key")
    logger.info("")
    
    key = input("  Paste your Alpha Vantage API key (or press Enter to skip): ").strip()
    
    if key:
        set_env_variable("ALPHA_VANTAGE_API_KEY", key)
        logger.info("  [OK] ALPHA_VANTAGE_API_KEY set for this session")
        return key
    else:
        logger.info("  [SKIP] Alpha Vantage setup skipped")
        return None

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
    logger.info("    FINNHUB_API_KEY:       %s", 'SET' if status['FINNHUB_API_KEY'] else 'NOT SET')
    logger.info("    ALPHA_VANTAGE_API_KEY: %s", 'SET' if status['ALPHA_VANTAGE_API_KEY'] else 'NOT SET')
    
    # Setup keys
    config = load_config()
    
    finnhub_key = setup_finnhub()
    if finnhub_key:
        config["FINNHUB_API_KEY"] = finnhub_key
    
    av_key = setup_alpha_vantage()
    if av_key:
        config["ALPHA_VANTAGE_API_KEY"] = av_key
    
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
