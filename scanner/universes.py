"""
Stock universe definitions for Indian market scanner.
All tickers are NSE symbols (suffix .NS added at fetch time).
"""

# ── NIFTY 50 ────────────────────────────────────────────────────────────────
NIFTY_50 = [
    "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL", "BRITANNIA",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC",
    "JSWSTEEL", "KOTAKBANK", "LTTS", "LT", "M&M",
    "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
    "RELIANCE", "SBICARD", "SBILIFE", "SBIN", "SUNPHARMA", "TATACONSUM",
    "TATAPOWER", "TATASTEEL", "TECHM", "TITAN", "TRENT",
    "ULTRACEMCO", "WIPRO", "HDFCAMC", "DIVISLAB", "SHRIRAMFIN"
]

# ── BANK NIFTY ──────────────────────────────────────────────────────────────
BANK_NIFTY = [
    "AUBANK", "AXISBANK", "BANDHANBNK", "FEDERALBNK", "HDFCBANK",
    "ICICIBANK", "IDFCFIRSTB", "INDUSINDBK", "KOTAKBANK", "PNB",
    "SBIN", "BANKBARODA", "BANKINDIA", "CANBK", "CENTRALBK",
    "IOB", "MAHABANK", "UNIONBANK", "UCOBANK", "INDIANB"
]

# ── NIFTY NEXT 50 ──────────────────────────────────────────────────────────
NIFTY_NEXT_50 = [
    "ABB", "ACC", "ADANIENT", "ADANIGREEN", "AMBUJACEM",
    "ASHOKLEY", "ASTRAL", "ATUL", "AVALONLABS", "BALKRISIND",
    "BIOCON", "BOSCHLTD", "CANFINHOME", "CHAMBLFERT", "COFORGE",
    "CONCOR", "COROMANDEL", "CROMPTON", "CUMMINSIND", "DALBHARAT",
    "DEEPAKNTR", "DIXON", "EMAMILTD", "ESCORTS", "GLENMARK",
    "GODREJCP", "GODREJPROP", "GSPL", "HAL", "HONAUT",
    "IDEA", "INDHOTEL", "IRCTC", "JUBLFOOD", "KPITTECH",
    "LALPATHLAB", "LICHSGFIN", "LUPIN", "MANAPPURAM", "MARICO",
    "MFSL", "MGL", "MOTHERSON", "MPHASIS", "MUTHOOTFIN",
    "NAM-INDIA", "OBEROIRLTY", "PERSISTENT", "PETRONET", "PFIZER"
]

# ── NIFTY MIDCAP 100 ──────────────────────────────────────────────────────
NIFTY_MIDCAP_100 = [
    "ABCAPITAL", "ABFRL", "ALKEM", "ASHOKLEY", "ASTRAL",
    "BALKRISIND", "BATAINDIA", "BEL", "BIOCON", "BOSCHLTD",
    "CANFINHOME", "CHAMBLFERT", "COFORGE", "CONCOR", "COROMANDEL",
    "CROMPTON", "CUMMINSIND", "DALBHARAT", "DEEPAKNTR", "DIXON",
    "EMAMILTD", "ESCORTS", "GLENMARK", "GODREJCP", "GODREJPROP",
    "GSPL", "HAL", "HONAUT", "INDHOTEL", "IRCTC",
    "JUBLFOOD", "KPITTECH", "LALPATHLAB", "LICHSGFIN", "LUPIN",
    "MANAPPURAM", "MARICO", "MFSL", "MGL", "MOTHERSON",
    "MPHASIS", "MUTHOOTFIN", "OBEROIRLTY", "PERSISTENT", "PETRONET",
    "PIIND", "PRESTIGE", "PVRINOX", "RAJESHEXPO", "RAMCOCEM",
    "RBLBANK", "RECLTD", "SAIL", "SOLARINDS", "SONACOMS",
    "SUNDARMFIN", "SUPREMEIND", "TATACHEM", "TORNTPHARM", "TATAPOWER",
    "VOLTAS", "ZEEL", "ZYDUSLIFE", "FEDERALBNK", "IDFCFIRSTB",
    "INDIACEM", "INDUSTOWER", "JSL", "NATIONALUM", "NMDC",
    "OFSS", "PNCINFRA", "POLYCAB", "TATACOMM",
    "TATAMETALI", "THERMAX", "TITAN", "TRENT", "TVSMOTOR",
    "UBL", "UNIONBANK", "VSTIND", "WHIRLPOOL", "ZENSARTECH"
]

# ── NIFTY SMALLCAP 100 ─────────────────────────────────────────────────────
NIFTY_SMALLCAP_100 = [
    "AFFLE", "ALKYLAMINE", "ANGELONE", "APTUS", "ASTER",
    "BIRLASOFT", "BLUESTARCO", "BSOFT", "CAMPUS", "CDSL",
    "CENTURYPLY", "CLEAN", "CAMS", "CYIENT", "DATAPATTNS",
    "EASEMYTRIP", "ELECON", "ELGIEQUIP", "GALAXYSURF",
    "GRINDWELL", "HAPPSTMNDS", "IIFL",
    "INDIAMART", "JYOTHYLAB", "KAJARIACER",
    "KPITTECH", "KRBL", "LATENTVIEW", "LEMONTREE",
    "METROPOLIS", "MOTILALOFS", "NATCOPHARM",
    "NEWGEN", "PHOENIXLTD", "PIIND",
    "PRINCEPIPE", "RATNAMANI", "RELAXO", "RITES",
    "SANOFI", "SCHAEFFLER", "SOBHA",
    "SONATSOFTW", "SRF", "SUDARSCHEM",
    "SUPRAJIT", "TATVA", "TRIDENT",
    "UTIAMC", "VAIBHAVGBL", "WELCORP", "WELSPUNIND",
    "WESTLIFE", "WPIL", "ZFCVINDIA", "ZYDUSWELL"
]

# ── FnO STOCKS (Futures & Options enabled) ──────────────────────────────────
FNO_STOCKS = [
    # Nifty components + high liquidity F&O stocks
    "ADANIPORTS", "ADANIENT", "ADANIGREEN", "AMBUJACEM", "APOLLOHOSP",
    "ASHOKLEY", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE",
    "BAJAJFINSV", "BALRAMCHIN", "BALKRISIND", "BANDHANBNK", "BHEL",
    "BPCL", "BRITANNIA", "CANBK", "CHAMBLFERT", "CIPLA",
    "COALINDIA", "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON",
    "CUMMINSIND", "DALBHARAT", "DEEPAKNTR", "DIVISLAB", "DIXON",
    "EICHERMOT", "ESCORTS", "FEDERALBNK", "GAIL", "GLENMARK",
    "GODREJCP", "GODREJPROP", "GRASIM", "GSPL", "HAL",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDCOPPER", "HINDUNILVR", "ICICIBANK", "IDEA", "INDIACEM",
    "INDUSINDBK", "INFY", "IOC", "IRCTC", "ITC",
    "JINDALSTEL", "JSWENERGY", "JSWSTEEL", "KOTAKBANK", "LICHSGFIN",
    "LT", "LUPIN", "M&M", "MANAPPURAM", "MARICO",
    "MARUTI", "MFSL", "MGL", "MOTHERSON", "MPHASIS",
    "MUTHOOTFIN", "NATIONALUM", "NTPC", "ONGC", "PERSISTENT",
    "PETRONET", "PFC", "PIDILITIND", "PNB", "POLYCAB",
    "POWERGRID", "PVRINOX", "RECLTD", "RELIANCE", "SAIL",
    "SBICARD", "SBILIFE", "SBIN", "SHRIRAMFIN", "SONACOMS",
    "SRF", "SUNPHARMA", "TATACHEM", "TATACOMM", "TATACONSUM",
    "TATAELXSI", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS",
    "TECHM", "TORNTPHARM", "TRENT", "TVSMOTOR", "UBL",
    "ULTRACEMCO", "VEDL", "VOLTAS", "WIPRO", "ZEEL"
]

# ── CASH MARKET (High liquidity NSE stocks) ────────────────────────────────
CASH_MARKET = list(set(
    NIFTY_50 + BANK_NIFTY + NIFTY_NEXT_50 + FNO_STOCKS
))

# ── BSE SENSEX 30 ──────────────────────────────────────────────────────────
BSE_SENSEX = [
    "AXISBANK", "ASIANPAINT", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
    "BPCL", "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA",
    "DRREDDY", "EICHERMOT", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDUNILVR", "ICICIBANK", "INFY", "ITC",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND",
    "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "TATASTEEL", "TECHM",
    "TITAN", "ULTRACEMCO", "WIPRO"
]

# ── BSE MIDCAP ──────────────────────────────────────────────────────────────
BSE_MIDCAP = [
    "ABCAPITAL", "ABFRL", "ALKEM", "ASHOKLEY", "ASTRAL",
    "BALKRISIND", "BEL", "BIOCON", "BOSCHLTD", "CANFINHOME",
    "COFORGE", "CONCOR", "CROMPTON", "CUMMINSIND", "DALBHARAT",
    "DEEPAKNTR", "DIXON", "ESCORTS", "GLENMARK", "GODREJCP",
    "GODREJPROP", "HAL", "HONAUT", "INDHOTEL", "IRCTC",
    "JUBLFOOD", "LALPATHLAB", "LUPIN", "MARICO", "MOTHERSON",
    "PERSISTENT", "PETRONET", "PIIND", "PRESTIGE", "RAMCOCEM",
    "RECLTD", "SOLARINDS", "TATACHEM", "TORNTPHARM", "TATAPOWER",
    "VOLTAS", "ZEEL", "ZYDUSLIFE", "TVSMOTOR", "UBL"
]

# ── BSE SMALLCAP ────────────────────────────────────────────────────────────
BSE_SMALLCAP = [
    "AFFLE", "ANGELONE", "APTUS", "ASTER", "BIRLASOFT",
    "BLUESTARCO", "BSOFT", "CDSL", "CYIENT", "DATAPATTNS",
    "ELECON", "ELGIEQUIP", "GRINDWELL", "HAPPSTMNDS", "INDIAMART",
    "KPITTECH", "KRBL", "LATENTVIEW", "LEMONTREE", "METROPOLIS",
    "MOTILALOFS", "NATCOPHARM", "NEWGEN", "PHOENIXLTD",
    "PRINCEPIPE", "RATNAMANI", "RELAXO", "RITES", "SANOFI",
    "SCHAEFFLER", "SOBHA", "SONATSOFTW", "SRF", "TATVA",
    "UTIAMC", "VAIBHAVGBL", "WELCORP", "WELSPUNIND"
]

# ── NIFTY IT ────────────────────────────────────────────────────────────────
NIFTY_IT = [
    "COFORGE", "HCLTECH", "INFY", "LTTS", "MPHASIS",
    "PERSISTENT", "TCS", "TECHM", "WIPRO", "ZENSARTECH"
]

# ── NIFTY PHARMA ────────────────────────────────────────────────────────────
NIFTY_PHARMA = [
    "ALKEM", "AUROPHARMA", "BIOCON", "CIPLA", "DRREDDY",
    "GLENMARK", "LUPIN", "SUNPHARMA", "TORNTPHARM", "ZYDUSLIFE"
]

# ── NIFTY AUTO ──────────────────────────────────────────────────────────────
NIFTY_AUTO = [
    "ASHOKLEY", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "M&M",
    "MARUTI", "MOTHERSON", "TATAPOWER", "TVSMOTOR", "BHARATFORG"
]

# ── NIFTY METAL ────────────────────────────────────────────────────────────
NIFTY_METAL = [
    "HINDALCO", "JSWSTEEL", "NATIONALUM", "NMDC", "SAIL",
    "TATASTEEL", "VEDL", "WELCORP", "COALINDIA"
]

# ── NIFTY REALTY ────────────────────────────────────────────────────────────
NIFTY_REALTY = [
    "BRIGADE", "DLF", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD",
    "PRESTIGE", "SOBHA", "LODHA"
]

# ── NIFTY ENERGY ────────────────────────────────────────────────────────────
NIFTY_ENERGY = [
    "BPCL", "GAIL", "HINDPETRO", "IOC", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "TATAPOWER"
]

# ── NIFTY FINANCIAL SERVICES ────────────────────────────────────────────────
NIFTY_FINANCIAL = [
    "BAJFINANCE", "BAJAJFINSV", "HDFCBANK", "HDFCLIFE", "ICICIBANK",
    "ICICIPRULI", "KOTAKBANK", "LICHSGFIN", "SBICARD", "SBILIFE",
    "SBIN", "SHRIRAMFIN"
]

# ── BROADER MARKET ─────────────────────────────────────────────────────────
NIFTY_BROAD = list(set(
    NIFTY_50 + BANK_NIFTY + NIFTY_NEXT_50 + NIFTY_MIDCAP_100 + NIFTY_SMALLCAP_100
))

# ── Dynamic Full-Market Placeholders (live fetch via symbol_fetcher) ─────
# These are resolved lazily in get_universe() to avoid import-time network calls.
# Initialized with NIFTY_BROAD so len>0 for tests/UI before live fetch replaces them.
_NSE_ALL_PLACEHOLDER: list = list(NIFTY_BROAD)  # ~2,200 NSE mainboard (live → 2,567)
_BSE_ALL_PLACEHOLDER: list = list(NIFTY_BROAD)  # ~4,500 BSE active (live → 2,567/4,500)
_FULL_MARKET_PLACEHOLDER: list = list(NIFTY_BROAD)  # ~5,900 unique NSE+BSE (live → 3,136)

# ── Universe Map ────────────────────────────────────────────────────────────
UNIVERSES = {
    # NSE Indexes
    "NIFTY 50":           NIFTY_50,
    "BANK NIFTY":         BANK_NIFTY,
    "NIFTY NEXT 50":      NIFTY_NEXT_50,
    "NIFTY MIDCAP 100":   NIFTY_MIDCAP_100,
    "NIFTY SMALLCAP 100": NIFTY_SMALLCAP_100,

    # Market Segment
    "FnO STOCKS":         FNO_STOCKS,
    "CASH MARKET":        CASH_MARKET,

    # BSE Indexes
    "BSE SENSEX":         BSE_SENSEX,
    "BSE MIDCAP":         BSE_MIDCAP,
    "BSE SMALLCAP":       BSE_SMALLCAP,

    # Sector Indexes
    "NIFTY IT":           NIFTY_IT,
    "NIFTY PHARMA":       NIFTY_PHARMA,
    "NIFTY AUTO":         NIFTY_AUTO,
    "NIFTY METAL":        NIFTY_METAL,
    "NIFTY REALTY":       NIFTY_REALTY,
    "NIFTY ENERGY":       NIFTY_ENERGY,
    "NIFTY FINANCIAL":    NIFTY_FINANCIAL,

    # Combined (static broad)
    "ALL (Combined)":     NIFTY_BROAD,

    # ── Full Market — Live (chunked fetch, 5,900 unique) ──────────────────
    "NSE ALL (Live ~2,200)": _NSE_ALL_PLACEHOLDER,
    "BSE ALL (Live ~4,500)": _BSE_ALL_PLACEHOLDER,
    "FULL MARKET (NSE+BSE ~5,900)": _FULL_MARKET_PLACEHOLDER,
}


def get_universe(name: str) -> list:
    """Get universe tickers by name, case-insensitive.

    For static universes, returns the pre-built list.
    For live full-market universes (NSE ALL / BSE ALL / FULL MARKET),
    fetches via symbol_fetcher with 4h cache and falls back to static
    NIFTY_BROAD if live fetch fails (so scan never breaks).
    """
    # Dynamic full-market — live fetch (cached 4h)
    low = name.strip().lower()
    if low in ("nse all (live ~2,200)", "nse all", "nse all (live)"):
        try:
            from .symbol_fetcher import fetch_nse_mainboard

            live = fetch_nse_mainboard()
            if live and len(live) > 500:
                # Update placeholder so UNIVERSES reflects live count in UI
                UNIVERSES["NSE ALL (Live ~2,200)"] = live
                return live
        except Exception:
            pass
        return UNIVERSES.get("NSE ALL (Live ~2,200)") or NIFTY_BROAD

    if low in ("bse all (live ~4,500)", "bse all", "bse all (live)"):
        try:
            from .symbol_fetcher import fetch_bse_all_live

            live = fetch_bse_all_live()
            if live and len(live) > 500:
                UNIVERSES["BSE ALL (Live ~4,500)"] = live
                return live
        except Exception:
            pass
        return UNIVERSES.get("BSE ALL (Live ~4,500)") or NIFTY_BROAD

    if low in ("full market (nse+bse ~5,900)", "full market", "full market (live)", "all market"):
        try:
            from .symbol_fetcher import fetch_all_market_symbols

            live = fetch_all_market_symbols()
            if live and len(live) > 500:
                UNIVERSES["FULL MARKET (NSE+BSE ~5,900)"] = live
                return live
        except Exception:
            pass
        return UNIVERSES.get("FULL MARKET (NSE+BSE ~5,900)") or NIFTY_BROAD

    for key, value in UNIVERSES.items():
        if key.lower() == low:
            # If placeholder is still empty (e.g. NSE ALL before first fetch), fetch now
            if not value and "live" in low:
                # Trigger live fetch via recursion (will hit the branches above)
                return get_universe(key)
            return value
    return []


# ── Sector Mapping ──────────────────────────────────────────────────────────

SECTOR_MAP = {
    # Banking & Financial
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "KOTAKBANK": "Banking",
    "AXISBANK": "Banking", "INDUSINDBK": "Banking", "SBIN": "Banking",
    "PNB": "Banking", "BANKBARODA": "Banking", "FEDERALBNK": "Banking",
    "BANDHANBNK": "Banking", "AUBANK": "Banking", "IDFCFIRSTB": "Banking",
    "CANBK": "Banking", "UNIONBANK": "Banking", "CENTRALBK": "Banking",
    "IOB": "Banking", "MAHABANK": "Banking", "UCOBANK": "Banking", "INDIANB": "Banking",
    "BAJFINANCE": "Finance", "BAJAJFINSV": "Finance", "SBICARD": "Finance",
    "SBILIFE": "Finance", "HDFCLIFE": "Finance", "HDFCAMC": "Finance",
    "LICHSGFIN": "Finance", "MUTHOOTFIN": "Finance", "MANAPPURAM": "Finance",
    "MFSL": "Finance", "SHRIRAMFIN": "Finance", "PFC": "Finance",
    "RECLTD": "Finance", "CHOLAFIN": "Finance",
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "TECHM": "IT", "LTTS": "IT", "PERSISTENT": "IT", "MPHASIS": "IT",
    "COFORGE": "IT", "KPITTECH": "IT", "TATAELXSI": "IT",
    "ZENSARTECH": "IT", "BSOFT": "IT", "BIRLASOFT": "IT",
    # Pharma
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "TORNTPHARM": "Pharma", "LUPIN": "Pharma",
    "GLENMARK": "Pharma", "ALCHEMIST": "Pharma", "ZYDUSLIFE": "Pharma",
    "BIOCON": "Pharma", "AUROPHARMA": "Pharma", "NATCOPHARM": "Pharma",
    # Auto
    "MARUTI": "Auto", "M&M": "Auto", "TATAMOTORS": "Auto",
    "HEROMOTOCO": "Auto", "BAJAJ-AUTO": "Auto", "EICHERMOT": "Auto",
    "ASHOKLEY": "Auto", "TVSMOTOR": "Auto", "MOTHERSON": "Auto",
    "ESCORTS": "Auto", "BALKRISIND": "Auto", "BHARATFORG": "Auto",
    # Metals & Mining
    "TATASTEEL": "Metals", "HINDALCO": "Metals", "JSWSTEEL": "Metals",
    "VEDL": "Metals", "SAIL": "Metals", "NMDC": "Metals",
    "NATIONALUM": "Metals", "JINDALSTEL": "Metals", "JSL": "Metals",
    # Oil & Gas
    "RELIANCE": "OilGas", "ONGC": "OilGas", "BPCL": "OilGas",
    "IOC": "OilGas", "GAIL": "OilGas", "PETRONET": "OilGas",
    "TATACOMM": "OilGas", "HINDPETRO": "OilGas",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "BRITANNIA": "FMCG",
    "NESTLEIND": "FMCG", "TATACONSUM": "FMCG", "MARICO": "FMCG",
    "DABUR": "FMCG", "GODREJCP": "FMCG", "COLPAL": "FMCG",
    "UBL": "FMCG", "RADICO": "FMCG", "BALRAMCHIN": "FMCG",
    # Power & Infrastructure
    "NTPC": "Power", "POWERGRID": "Power", "TATAPOWER": "Power",
    "ADANIGREEN": "Power", "JSWENERGY": "Power",
    # Real Estate
    "GODREJPROP": "Realty", "OBEROIRLTY": "Realty", "PRESTIGE": "Realty",
    "PHOENIXLTD": "Realty", "DLF": "Realty", "BRIGADE": "Realty",
    "SOBHA": "Realty", "LODHA": "Realty",
    # Cement & Materials
    "ULTRACEMCO": "Cement", "GRASIM": "Cement", "AMBUJACEM": "Cement",
    "ACC": "Cement", "DALBHARAT": "Cement", "RAMCOCEM": "Cement",
    # Chemicals
    "TATACHEM": "Chemicals", "COROMANDEL": "Chemicals",
    "PIDILITIND": "Chemicals", "SRF": "Chemicals",
    "CHAMBLFERT": "Chemicals", "GSPL": "Chemicals",
    # Consumer
    "TITAN": "Consumer", "TRENT": "Consumer", "VOLTAS": "Consumer",
    "HAVELLS": "Consumer", "POLYCAB": "Consumer", "DIXON": "Consumer",
    # Telecom
    "BHARTIARTL": "Telecom", "IDEA": "Telecom", "INDUSTOWER": "Telecom",
    # Ports & Logistics
    "ADANIPORTS": "Infra", "CONCOR": "Infra", "DELHIVERY": "Infra",
    "PNCINFRA": "Infra",
    # Defence & Industrials
    "HAL": "Defence", "BEL": "Defence", "COCHINSHIP": "Defence",
    # Miscellaneous
    "IRCTC": "Misc", "PVRINOX": "Misc", "ZOMATO": "Misc",
    "NYKAA": "Misc", "PAYTM": "Misc",
    "LALPATHLAB": "Misc", "METROPOLIS": "Misc",
    "SONACOMS": "Misc", "CROMPTON": "Misc",
    # Other
    "BHEL": "Other", "HINDCOPPER": "Other", "ZEEL": "Other",
}


def get_sector(ticker: str) -> str:
    """Get the sector for a stock ticker."""
    return SECTOR_MAP.get(ticker, "Other")


SECTOR_COLORS = {
    "Banking": "#3b82f6", "Finance": "#8b5cf6", "IT": "#22c55e",
    "Pharma": "#ec4899", "Auto": "#f97316", "Metals": "#94a3b8",
    "OilGas": "#64748b", "FMCG": "#eab308", "Power": "#06b6d4",
    "Realty": "#a855f7", "Cement": "#78716c", "Chemicals": "#14b8a6",
    "Consumer": "#f43f5e", "Telecom": "#6366f1", "Infra": "#84cc16",
    "Defence": "#0ea5e9", "Misc": "#6b7280", "Other": "#6b7280",
}
