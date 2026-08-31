"""
Scanner Engine — Pure Python headless scanning logic.
Extracted from app.py for testability and reusability.
"""

import logging
import threading
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any

from .data_fetcher import fetch_index_data, fetch_batch_yfinance, fetch_fundamentals
from .scoring import compute_scores, check_filter, get_direction
from .universes import UNIVERSES, get_universe
from .settings_store import load_api_config, get_api_key

logger = logging.getLogger(__name__)


class ScanProgress:
    """Progress tracking for scan operations."""
    def __init__(self):
        self.value = 0.0
        self.text = ""
        self.cancelled = False


class ScanResult:
    """Container for scan results."""
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.filtered_out = 0
        self.direction_counts = {"Bull": 0, "Bear": 0}
        self.cancelled = False
        self.error: Optional[str] = None


class ScannerEngine:
    """
    Headless scanner engine for HMA/EMA multi-score swing strategy.
    
    Usage:
        engine = ScannerEngine()
        progress_cb = lambda p, t: print(f"{p:.0%}: {t}")
        log_cb = lambda msg: print(msg)
        result = engine.scan(universe="NIFTY 50", settings=settings, 
                            progress_callback=progress_cb, log_callback=log_cb)
    """
    
    def __init__(self):
        self._cancel_event = threading.Event()
        self._progress_callback: Optional[Callable[[float, str], None]] = None
        self._log_callback: Optional[Callable[[str], None]] = None
    
    def cancel(self):
        """Signal the scan to cancel."""
        self._cancel_event.set()
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates: callback(progress: 0.0-1.0, text: str)"""
        self._progress_callback = callback
    
    def set_log_callback(self, callback: Callable[[str], None]):
        """Set callback for log messages: callback(message: str)"""
        self._log_callback = callback
    
    def _progress(self, value: float, text: str = ""):
        if self._progress_callback:
            self._progress_callback(value, text)
    
    def _log(self, msg: str):
        if self._log_callback:
            self._log_callback(msg)

    def _enrich_with_providers(self, ticker: str, settings: dict) -> dict:
        """
        Enrich settings with data from provider modules.
        Calls sentiment, social, indian_market, indian_fundamentals, insider
        providers and populates the settings dict keys that scoring.py expects.
        """
        enriched = settings.copy()

        # Load API keys
        api_config = load_api_config()

        # ── Market Sentiment (Category 11) ──────────────────────────────
        if settings.get("use_market_sentiment", True):
            try:
                from .market_sentiment import fetch_sentiment
                sent = fetch_sentiment(
                    ticker,
                    marketaux_key=get_api_key("MARKETAUX_API_KEY", api_config),
                    newsapi_key=get_api_key("NEWS_API_KEY", api_config),
                    gnews_key=get_api_key("GNEWS_API_KEY", api_config),
                )
                enriched["_sentiment_score"] = sent.get("sentiment_score", 0.0)
                enriched["_article_count"] = sent.get("article_count", 0)
                enriched["_sentiment_source"] = sent.get("source", "none")
            except Exception as e:
                logger.debug("Sentiment fetch failed for %s: %s", ticker, e)

        # ── Social Sentiment (Category 12) ──────────────────────────────
        if settings.get("use_social_sentiment", True):
            try:
                from .social_sentiment import fetch_social_sentiment
                social = fetch_social_sentiment(
                    ticker,
                    twitter_api_key=get_api_key("HF_API_KEY", api_config),
                )
                enriched["_social_score"] = social.get("social_score", 0.0)
                enriched["_mention_count"] = social.get("mention_count", 0)
                enriched["_social_source"] = social.get("source", "none")
            except Exception as e:
                logger.debug("Social sentiment fetch failed for %s: %s", ticker, e)

        # ── Indian Market Data (Categories 13, 14, 15) ──────────────────
        if settings.get("use_indian_market", True):
            try:
                from .indian_market import fetch_indian_market_data
                india = fetch_indian_market_data(ticker)

                delivery = india.get("delivery")
                if delivery:
                    enriched["_delivery_pct"] = delivery.delivery_pct
                    enriched["_delivery_change_pct"] = delivery.delivery_change_pct
                    enriched["_delivery_source"] = "nse"

                fii_dii = india.get("fii_dii")
                if fii_dii:
                    enriched["_fii_is_buying"] = fii_dii.fii_is_buying
                    enriched["_dii_is_buying"] = fii_dii.dii_is_buying
                    enriched["_fii_net"] = fii_dii.fii_net
                    enriched["_dii_net"] = fii_dii.dii_net
                    enriched["_institutional_source"] = "nse"

                week52 = india.get("week52")
                if week52:
                    enriched["_52w_position"] = week52.position_in_range
                    enriched["_52w_pct_from_high"] = week52.pct_from_52w_high
                    enriched["_52w_source"] = "nse"
            except Exception as e:
                logger.debug("Indian market fetch failed for %s: %s", ticker, e)

        # ── Indian Fundamentals (Category 16) ───────────────────────────
        if settings.get("use_indian_fundamentals", True):
            try:
                from .indian_fundamentals import fetch_indian_fundamentals
                indian_fund = fetch_indian_fundamentals(ticker)

                screener = indian_fund.get("screener")
                if screener:
                    if screener.industry_pe and screener.stock_pe:
                        enriched["_pe_relative_to_industry"] = screener.stock_pe / screener.industry_pe
                    enriched["_is_quality_stock"] = screener.is_quality
                    enriched["_valuation_source"] = indian_fund.get("source", "none")
            except Exception as e:
                logger.debug("Indian fundamentals fetch failed for %s: %s", ticker, e)

        # ── Insider Data (adjustment to Fundamentals) ───────────────────
        if settings.get("use_insider_data", True):
            try:
                from .insider_data import fetch_insider_data
                insider = fetch_insider_data(ticker)
                enriched["_insider_score"] = insider.get("insider_score", 0.0)
                enriched["_insider_source"] = insider.get("source", "none")
            except Exception as e:
                logger.debug("Insider data fetch failed for %s: %s", ticker, e)

        # ── Macro Data ──────────────────────────────────────────────────
        if settings.get("use_macro_data", True):
            try:
                from .macro_data import fetch_macro_data
                macro = fetch_macro_data(
                    fred_key=get_api_key("FRED_API_KEY", api_config),
                    econpulse_key=get_api_key("ECONPULSE_API_KEY", api_config),
                    econdb_key=get_api_key("ECONDB_API_KEY", api_config),
                )
                regime = macro.get("regime")
                if regime:
                    enriched["_macro_regime"] = regime.regime
                    enriched["_macro_confidence"] = regime.confidence
                    enriched["_macro_signals"] = regime.signals

                # Forex data (Category 18)
                forex = macro.get("forex")
                if forex:
                    enriched["_inr_change_1d"] = forex.change_1d
                    enriched["_inr_change_1w"] = forex.change_1w
                    enriched["_forex_source"] = "frankfurter"

                # Crypto sentiment (for market regime context)
                crypto = macro.get("crypto")
                if crypto:
                    enriched["_btc_fear_greed"] = crypto.fear_greed_index
                    enriched["_btc_fear_greed_label"] = crypto.fear_greed_label
            except Exception as e:
                logger.debug("Macro data fetch failed: %s", e)

        # ── Free APIs (Categories 17, 19, 20) ──────────────────────────
        if settings.get("use_macro_data", True):
            try:
                from .free_apis import (
                    fetch_mandi_prices,
                    fetch_crypto_sentiment,
                    fetch_wallstreetbets_sentiment,
                )

                # Commodity prices (Category 17)
                mandi = fetch_mandi_prices()
                if mandi:
                    # Simple: if any commodity prices available, mark as "neutral"
                    enriched["_commodity_trend"] = "neutral"
                    enriched["_commodity_source"] = "mandi"
            except Exception as e:
                logger.debug("Free APIs fetch failed: %s", e)

        # ── Premium Finance (Category 20 - Shariah) ────────────────────
        try:
            from .premium_finance import fetch_shariah_data
            shariah = fetch_shariah_data(
                ticker,
                api_key=get_api_key("HALAL_API_KEY", api_config),
            )
            if shariah:
                enriched["_is_shariah_compliant"] = shariah.is_shariah_compliant
                enriched["_shariah_source"] = "halal_terminal"
        except Exception as e:
            logger.debug("Shariah data fetch failed for %s: %s", ticker, e)

        return enriched
    
    def scan(
        self,
        universe: str,
        settings: Dict[str, Any],
        period: str = "1y",
        timeframe: str = "D",
        trend_filter: str = "All",
        index_symbol: str = "NSEI",
    ) -> ScanResult:
        """
        Run a full scan.
        
        Args:
            universe: Universe name (key in UNIVERSES or dynamic universe)
            settings: Scanner settings dict
            period: Data period (e.g., "1y", "3y")
            timeframe: "D" (daily), "W" (weekly), "M" (monthly)
            trend_filter: "All", "Bullish Only", "Bearish Only"
            index_symbol: Index symbol for relative strength (e.g., "NSEI")
        
        Returns:
            ScanResult with results list and metadata
        """
        self._cancel_event.clear()
        result = ScanResult()
        
        try:
            # Resolve universe
            try:
                tickers = get_universe(universe)
            except KeyError:
                tickers = UNIVERSES.get(universe, [])
            
            tf_names = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
            
            self._log("\n" + "=" * 50)
            self._log(f"START SCAN | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._log("=" * 50)
            self._log(f"Starting scan: {universe} ({len(tickers)} stocks)")
            self._log(f"Timeframe: {tf_names.get(timeframe, timeframe)} | Period: {period} | Filter: {trend_filter}")
            self._log(f"FastMA={settings.get('fast_ma_type','HMA')}{settings.get('fast_ma_len',40)} "
                       f"SlowMA={settings.get('slow_ma_type','EMA')}{settings.get('slow_ma_len',50)} "
                       f"RSI={settings.get('rsi_len',14)} Threshold={settings.get('min_score',50)}")
            
            # Fetch NIFTY index
            self._progress(0.0, "Fetching NIFTY 50 index...")
            index_df = fetch_index_data(f"^{index_symbol}", period=period)
            if index_df is not None:
                self._log(f"{index_symbol} index loaded ({len(index_df)} bars)")
            else:
                self._log(f"Warning: {index_symbol} index unavailable, using proxy for RS")
            
            # Batch download all stocks
            self._progress(0.05, f"Batch downloading {len(tickers)} stocks...")
            self._log(f"Batch downloading {len(tickers)} stocks via yfinance...")
            batch_data = fetch_batch_yfinance(tickers, period=period, timeframe=timeframe)
            self._log(f"Batch download complete: {len(batch_data)}/{len(tickers)} stocks fetched")
            
            # 3-Model Pipeline
            results = []
            total = len(batch_data)
            filtered_out = 0
            direction_counts = {"Bull": 0, "Bear": 0}
            
            for i, (ticker, df) in enumerate(batch_data.items(), 1):
                if self._cancel_event.is_set():
                    self._log("\n\u23f9  Scan cancelled by user")
                    result.cancelled = True
                    break
                
                progress = 0.1 + (i / total * 0.9) if total > 0 else 0.5
                self._progress(progress, f"[{i}/{total}] {ticker}")
                
                try:
                    if df is None or df.empty:
                        continue
                    
                    # MODEL 1: Stock Filter
                    filter_result = check_filter(
                        df,
                        fast_ma_type=settings.get("fast_ma_type", "HMA"),
                        fast_ma_len=settings.get("fast_ma_len", 40),
                        slow_ma_type=settings.get("slow_ma_type", "EMA"),
                        slow_ma_len=settings.get("slow_ma_len", 50),
                        crossover_lookback=settings.get("crossover_lookback", 20),
                    )
                    if filter_result is None:
                        filtered_out += 1
                        continue
                    
                    # MODEL 2: Bullish / Bearish
                    direction = get_direction(filter_result)
                    
                    trend_filter_val = trend_filter
                    if trend_filter_val == "Bullish Only" and direction != "Bull":
                        filtered_out += 1
                        continue
                    elif trend_filter_val == "Bearish Only" and direction != "Bear":
                        filtered_out += 1
                        continue
                    
                    direction_counts[direction] = direction_counts.get(direction, 0) + 1
                    
                    # MODEL 3: Techno-Fundamental Scoring
                    if not hasattr(df, '_fundamentals') or df._fundamentals is None:
                        try:
                            fund = fetch_fundamentals(ticker)
                            if fund is not None:
                                object.__setattr__(df, '_fundamentals', fund)
                        except Exception as e:
                            logger.debug("Fundamentals fetch failed for %s: %s", ticker, e)

                    # ── Enrich settings with provider data ──────────────
                    enriched_settings = self._enrich_with_providers(ticker, settings)

                    scores = compute_scores(
                        df, timeframe=timeframe, index_df=index_df,
                        settings=enriched_settings,
                    )
                    if scores is None:
                        continue
                    
                    scores["ticker"] = ticker
                    scores["trend_dir"] = direction
                    scores["trend_color"] = direction.lower()
                    results.append(scores)
                    
                    if len(results) % 10 == 0 or len(results) <= 5:
                        score_val = scores["total"]
                        tag = "\u2713" if score_val >= settings.get("min_score", 50) else "\u2717"
                        self._log(f"  {'\u2713' if score_val >= settings.get('min_score', 50) else '\u2717'} {ticker}: {score_val:.1f}/100 ({direction})")
                
                except Exception as e:
                    logger.debug("Skipping %s in batch scan: %s", ticker, e)
            
            # Sort and store
            results.sort(key=lambda x: x.get("total", 0) or 0, reverse=True)
            
            passed = len([r for r in results if r["total"] >= settings.get("min_score", 50)])
            
            self._log("\n\u2501" * 25 + " Scan Complete ")
            self._log(f"  Total stocks:  {len(tickers)}")
            self._log(f"  Filtered out:  {filtered_out} (no recent crossover)")
            self._log(f"  Passed filter: {len(results)} ({direction_counts.get('Bull', 0)} Bull, {direction_counts.get('Bear', 0)} Bear)")
            self._log(f"  Scored {settings.get('min_score', 50)}+: {passed}")
            
            result.results = results
            result.filtered_out = filtered_out
            result.direction_counts = direction_counts
            
        except Exception as e:
            result.error = str(e)
            self._log(f"\nERROR: {str(e)}")
        
        return result


# Convenience function for simple usage
def run_scan(
    universe: str,
    settings: Dict[str, Any],
    period: str = "1y",
    timeframe: str = "D",
    trend_filter: str = "All",
    index_symbol: str = "NSEI",
    progress_callback: Optional[Callable[[float, str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> ScanResult:
    """Convenience function to run a scan without managing engine instance."""
    engine = ScannerEngine()
    if progress_callback:
        engine.set_progress_callback(progress_callback)
    if log_callback:
        engine.set_log_callback(log_callback)
    return engine.scan(universe, settings, period, timeframe, trend_filter, index_symbol)