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
                                df._fundamentals = fund
                        except Exception as e:
                            logger.debug("Fundamentals fetch failed for %s: %s", ticker, e)
                    
                    scores = compute_scores(
                        df, timeframe=timeframe, index_df=index_df,
                        settings=settings,
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