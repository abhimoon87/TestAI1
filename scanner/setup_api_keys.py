"""
API Key Setup Script for HMAxEMA Scanner
Helps you configure Finnhub and Alpha Vantage API keys for enhanced data.
"""

import os
import sys
import json
from pathlib import Path

# Config file location
CONFIG_FILE = Path(__file__).parent / "api_config.json"

def print_header():
    print("=" * 60)
    print("  HMAxEMA Scanner — API Key Setup")
    print("=" * 60)
    print()

def print_section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")

def load_config():
    """Load existing config."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(config):
    """Save config to file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  [OK] Config saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"  [ERROR] Could not save config: {e}")

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
    print("  Finnhub provides institutional-grade fundamental data.")
    print("  Free tier: 60 API calls per minute")
    print()
    print("  Steps to get your API key:")
    print("  1. Go to: https://finnhub.io/register")
    print("  2. Sign up with your email")
    print("  3. Verify your email")
    print("  4. Go to Dashboard → API Keys")
    print("  5. Copy your API key")
    print()
    
    key = input("  Paste your Finnhub API key (or press Enter to skip): ").strip()
    
    if key:
        set_env_variable("FINNHUB_API_KEY", key)
        print(f"  [OK] FINNHUB_API_KEY set for this session")
        return key
    else:
        print("  [SKIP] Finnhub setup skipped")
        return None

def setup_alpha_vantage():
    """Setup Alpha Vantage API key."""
    print_section("ALPHA VANTAGE SETUP")
    print("  Alpha Vantage provides technical indicators and fundamentals.")
    print("  Free tier: 25 API calls per day")
    print()
    print("  Steps to get your API key:")
    print("  1. Go to: https://www.alphavantage.co/support/#api-key")
    print("  2. Fill in the form (name, email, usage)")
    print("  3. Click 'Get Free API Key'")
    print("  4. Check your email for the key")
    print()
    
    key = input("  Paste your Alpha Vantage API key (or press Enter to skip): ").strip()
    
    if key:
        set_env_variable("ALPHA_VANTAGE_API_KEY", key)
        print(f"  [OK] ALPHA_VANTAGE_API_KEY set for this session")
        return key
    else:
        print("  [SKIP] Alpha Vantage setup skipped")
        return None

def test_providers():
    """Test if the providers work with current keys."""
    print_section("TESTING PROVIDERS")
    
    try:
        from scanner.data_providers import DataProvider
        
        provider = DataProvider()
        test_ticker = "RELIANCE"
        
        print(f"  Testing fundamentals for {test_ticker}...")
        fund = provider.fetch_fundamentals(test_ticker)
        
        if fund:
            print(f"  [OK] Provider: {provider.last_provider}")
            print(f"       P/E Ratio: {fund.get('pe_ratio', 'N/A')}")
            print(f"       EPS Growth: {fund.get('eps_growth', 'N/A')}")
            print(f"       Revenue Growth: {fund.get('rev_growth', 'N/A')}")
            return True
        else:
            print(f"  [WARN] No data fetched (using fallback)")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Test failed: {e}")
        return False

def print_permanent_instructions():
    """Print instructions for permanent setup."""
    print_section("PERMANENT SETUP (Optional)")
    print("  To make keys permanent, add them to your system:")
    print()
    print("  Windows (PowerShell):")
    print("    $env:FINNHUB_API_KEY=\"your_key\"")
    print("    $env:ALPHA_VANTAGE_API_KEY=\"your_key\"")
    print()
    print("  Windows (System Environment Variables):")
    print("    1. Search 'Environment Variables' in Start Menu")
    print("    2. Click 'Environment Variables'")
    print("    3. Add new User variables")
    print()
    print("  macOS/Linux:")
    print("    export FINNHUB_API_KEY=\"your_key\"")
    print("    export ALPHA_VANTAGE_API_KEY=\"your_key\"")
    print("    # Add to ~/.bashrc or ~/.zshrc for persistence")

def main():
    """Main setup function."""
    print_header()
    
    # Show current status
    status = get_current_status()
    print("  Current API Key Status:")
    print(f"    FINNHUB_API_KEY:       {'SET' if status['FINNHUB_API_KEY'] else 'NOT SET'}")
    print(f"    ALPHA_VANTAGE_API_KEY: {'SET' if status['ALPHA_VANTAGE_API_KEY'] else 'NOT SET'}")
    
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
    print("  API keys are configured for this session.")
    print("  Restart the scanner to use the new providers.")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  [CANCELLED] Setup cancelled.")
        sys.exit(0)
