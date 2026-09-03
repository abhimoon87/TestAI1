"""
Scanner Engine — Pure Python headless scanning logic.
Extracted from app.py for testability and reusability.
"""

import logging
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

from requests.exceptions import RequestException

from .data_fetcher import (
    fetch_batch_yfinance,
    fetch_batch_yfinance_stream,
    fetch_fundamentals,
    fetch_index_data,
)
from .scoring import check_filter, compute_scores, get_direction
from .settings_store import get_api_key, load_api_config
from .trace import trace
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
        self.results: list[dict[str, Any]] = []
        self.filtered_out = 0
        self.direction_counts = {"Bull": 0, "Bear": 0}
        self.cancelled = False
        self.error: str | None = None


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
        self._progress_callback: Callable[[float, str], None] | None = None
        self._log_callback: Callable[[str], None] | None = None
        self._batch_callback: Callable[[list[dict]], None] | None = None
    
    def cancel(self):
        """Signal the scan to cancel."""
        self._cancel_event.set()
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates: callback(progress: 0.0-1.0, text: str)"""
        self._progress_callback = callback
    
    def set_log_callback(self, callback: Callable[[str], None]):
        """Set callback for log messages: callback(message: str)"""
        self._log_callback = callback

    def set_batch_callback(self, callback: Callable[[list[dict]], None] | None):
        """Set callback for incremental batch results: callback(batch: list[dict])"""
        self._batch_callback = callback
    
    def _progress(self, value: float, text: str = ""):
        if self._progress_callback:
            self._progress_callback(value, text)
    
    def _log(self, msg: str):
        if self._log_callback:
            self._log_callback(msg)

    def _fetch_global_enrichment(self, settings: dict) -> dict:
        """
        Fetch macro/mandi data ONCE (not per-ticker) — parallelized.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        global_data: dict = {}
        api_config = load_api_config()

        def _fetch_macro():
            from .macro_data import fetch_macro_data
            return fetch_macro_data(
                fred_key=get_api_key("FRED_API_KEY", api_config),
                econpulse_key=get_api_key("ECONPULSE_API_KEY", api_config),
                econdb_key=get_api_key("ECONDB_API_KEY", api_config),
            )

        def _fetch_mandi():
            from .free_apis import fetch_mandi_prices
            return fetch_mandi_prices()

        futures = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            if settings.get("use_macro_data", True):
                futures[ex.submit(_fetch_macro)] = "macro"
                futures[ex.submit(_fetch_mandi)] = "mandi"
            for fut in as_completed(futures):
                kind = futures[fut]
                try:
                    res = fut.result()
                    if kind == "macro" and res:
                        regime = res.get("regime")
                        if regime:
                            global_data["_macro_regime"] = regime.regime
                            global_data["_macro_confidence"] = regime.confidence
                            global_data["_macro_signals"] = regime.signals
                        forex = res.get("forex")
                        if forex:
                            global_data["_inr_change_1d"] = forex.change_1d
                            global_data["_inr_change_1w"] = forex.change_1w
                            global_data["_forex_source"] = "frankfurter"
                        crypto = res.get("crypto")
                        if crypto:
                            global_data["_btc_fear_greed"] = crypto.fear_greed_index
                            global_data["_btc_fear_greed_label"] = crypto.fear_greed_label
                    elif kind == "mandi" and res:
                        global_data["_commodity_trend"] = "neutral"
                        global_data["_commodity_source"] = "mandi"
                except (RequestException, ValueError, KeyError) as e:
                    logger.debug("%s fetch failed: %s", kind, e)

        return global_data

    def _enrich_with_providers(self, ticker: str, settings: dict, global_data: dict | None = None) -> dict:
        """
        Enrich settings with data from provider modules.
        Runs all 5 per-ticker providers in parallel via ThreadPoolExecutor.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        enriched = settings.copy()

        # Merge pre-fetched global data (macro/mandi — already fetched once)
        if global_data:
            enriched.update(global_data)

        # Load API keys (cached per scan, not per-ticker)
        api_config = getattr(self, '_api_config', None) or load_api_config()

        # Define per-ticker provider functions
        def _fetch_sentiment():
            from .market_sentiment import fetch_sentiment
            return fetch_sentiment(
                ticker,
                marketaux_key=get_api_key("MARKETAUX_API_KEY", api_config),
                newsapi_key=get_api_key("NEWS_API_KEY", api_config),
                gnews_key=get_api_key("GNEWS_API_KEY", api_config),
            )

        def _fetch_social():
            from .social_sentiment import fetch_social_sentiment
            return fetch_social_sentiment(
                ticker,
                twitter_api_key=get_api_key("HF_API_KEY", api_config),
            )

        def _fetch_indian_market():
            from .indian_market import fetch_indian_market_data
            return fetch_indian_market_data(ticker)

        def _fetch_indian_fundamentals():
            from .indian_fundamentals import fetch_indian_fundamentals
            return fetch_indian_fundamentals(ticker)

        def _fetch_insider():
            from .insider_data import fetch_insider_data
            return fetch_insider_data(ticker)

        # Launch all 5 providers in parallel
        futures = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            if settings.get("use_market_sentiment", True):
                futures[executor.submit(_fetch_sentiment)] = "sentiment"
            if settings.get("use_social_sentiment", True):
                futures[executor.submit(_fetch_social)] = "social"
            if settings.get("use_indian_market", True):
                futures[executor.submit(_fetch_indian_market)] = "india"
            if settings.get("use_indian_fundamentals", True):
                futures[executor.submit(_fetch_indian_fundamentals)] = "india_fund"
            if settings.get("use_insider_data", True):
                futures[executor.submit(_fetch_insider)] = "insider"

            for future in as_completed(futures):
                category = futures[future]
                try:
                    result = future.result()
                    if category == "sentiment":
                        enriched["_sentiment_score"] = result.get("sentiment_score", 0.0)
                        enriched["_article_count"] = result.get("article_count", 0)
                        enriched["_sentiment_source"] = result.get("source", "none")
                    elif category == "social":
                        enriched["_social_score"] = result.get("social_score", 0.0)
                        enriched["_mention_count"] = result.get("mention_count", 0)
                        enriched["_social_source"] = result.get("source", "none")
                    elif category == "india":
                        delivery = result.get("delivery")
                        if delivery:
                            enriched["_delivery_pct"] = delivery.delivery_pct
                            enriched["_delivery_change_pct"] = delivery.delivery_change_pct
                            enriched["_delivery_source"] = "nse"
                        fii_dii = result.get("fii_dii")
                        if fii_dii:
                            enriched["_fii_is_buying"] = fii_dii.fii_is_buying
                            enriched["_dii_is_buying"] = fii_dii.dii_is_buying
                            enriched["_fii_net"] = fii_dii.fii_net
                            enriched["_dii_net"] = fii_dii.dii_net
                            enriched["_institutional_source"] = "nse"
                        week52 = result.get("week52")
                        if week52:
                            enriched["_52w_position"] = week52.position_in_range
                            enriched["_52w_pct_from_high"] = week52.pct_from_52w_high
                            enriched["_52w_source"] = "nse"
                    elif category == "india_fund":
                        screener = result.get("screener")
                        if screener:
                            if screener.industry_pe and screener.stock_pe:
                                enriched["_pe_relative_to_industry"] = screener.stock_pe / screener.industry_pe
                            enriched["_is_quality_stock"] = screener.is_quality
                            enriched["_valuation_source"] = result.get("source", "none")
                    elif category == "insider":
                        enriched["_insider_score"] = result.get("insider_score", 0.0)
                        enriched["_insider_source"] = result.get("source", "none")
                except (RequestException, ValueError, KeyError, AttributeError) as e:
                    logger.debug("Provider %s failed for %s: %s", category, ticker, e)

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
        except (RequestException, ValueError, KeyError, AttributeError) as e:
            logger.debug("Shariah data fetch failed for %s: %s", ticker, e)

        return enriched
    
    @trace(level=logging.INFO, log_args=True)
    def scan(
        self,
        universe: str,
        settings: dict[str, Any],
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
            
            # Batch download all stocks — yfinance chunks, then a per-ticker
            # fallback pass (jugaad-data/nselib) for anything yfinance missed.
            self._progress(0.05, f"Batch downloading {len(tickers)} stocks...")
            self._log(f"Batch downloading {len(tickers)} stocks via yfinance (fallback: jugaad-data/nselib)...")
            batch_data = fetch_batch_yfinance(
                tickers,
                period=period,
                timeframe=timeframe,
                cancel_event=self._cancel_event,
                on_fallback_progress=lambda done, total: self._progress(
                    0.05 + 0.05 * (done / max(total, 1)),
                    f"Fallback fetch {done}/{total} (jugaad/nselib)",
                ),
            )
            if len(batch_data) < len(tickers):
                self._log(
                    f"Batch download complete: {len(batch_data)}/{len(tickers)} stocks fetched "
                    f"({len(tickers) - len(batch_data)} unavailable on all providers)"
                )
            else:
                self._log(f"Batch download complete: {len(batch_data)}/{len(tickers)} stocks fetched")

            # Cache API keys once (not per-ticker)
            self._api_config = load_api_config()

            # Fetch global enrichment data ONCE (macro/mandi — same for all tickers)
            self._log("Fetching global macro/commodity data...")
            global_data = self._fetch_global_enrichment(settings)
            if global_data:
                self._log(f"Global enrichment: {len(global_data)} keys (macro, forex, crypto, commodity)")

            # 3-Model Pipeline — fast mode for 5,900 scan
            # For large universes (>500) we skip per-ticker enrichment and fundamentals
            # on the first pass, score on technicals only, then enrich only the
            # top 200 by score. This cuts 500*~1s enrichment → ~200*1s.
            is_large = len(tickers) > 500
            if is_large:
                self._log(f"Large universe fast mode: {len(tickers)} stocks — technicals first, enrich top 200 only")

            results = []
            total = len(batch_data)
            filtered_out = 0
            direction_counts = {"Bull": 0, "Bear": 0}

            # ── Phase 1: Fast technical scoring (no enrichment) ────────────
            # Use ThreadPool for CPU-bound scoring on large universes
            def _score_one(item):
                ticker, df = item
                try:
                    if df is None or df.empty:
                        return None, "empty"
                    filter_result = check_filter(
                        df,
                        fast_ma_type=settings.get("fast_ma_type", "HMA"),
                        fast_ma_len=settings.get("fast_ma_len", 40),
                        slow_ma_type=settings.get("slow_ma_type", "EMA"),
                        slow_ma_len=settings.get("slow_ma_len", 50),
                        crossover_lookback=settings.get("crossover_lookback", 20),
                    )
                    if filter_result is None:
                        return None, "filtered"
                    direction = get_direction(filter_result)
                    if trend_filter == "Bullish Only" and direction != "Bull":
                        return None, "filtered"
                    if trend_filter == "Bearish Only" and direction != "Bear":
                        return None, "filtered"
                    # Skip fundamentals/enrichment in fast mode phase 1
                    if not is_large and getattr(df, '_fundamentals', None) is None:
                        try:
                            fund = fetch_fundamentals(ticker)
                            if fund is not None:
                                object.__setattr__(df, '_fundamentals', fund)
                        except (RequestException, ValueError, KeyError) as e:
                            logger.debug("Fundamentals fetch failed for %s: %s", ticker, e)
                    if is_large:
                        enriched = {**settings, **global_data, "_skip_vp": True}
                    else:
                        enriched = self._enrich_with_providers(ticker, settings, global_data)
                        enriched["_skip_vp"] = False
                    scores = compute_scores(df, timeframe=timeframe, index_df=index_df, settings=enriched)
                    if scores is None:
                        return None, "no_score"
                    scores["ticker"] = ticker
                    scores["trend_dir"] = direction
                    scores["trend_color"] = direction.lower()
                    # Keep enrichment keys for later if not large
                    if not is_large:
                        for k, v in enriched.items():
                            if k.startswith("_") and k not in scores:
                                scores[k] = v
                    return scores, direction
                except (RequestException, ValueError, KeyError, TypeError) as e:
                    logger.debug("Scoring failed for %s: %s", ticker, e)
                    return None, "error"

            if is_large and total > 200:
                # Parallel scoring for large universes
                from concurrent.futures import ThreadPoolExecutor, as_completed

                max_workers = min(8, (total // 50) + 2)
                self._log(f"Parallel scoring {total} stocks with {max_workers} workers...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_ticker = {executor.submit(_score_one, item): item[0] for item in batch_data.items()}
                    done = 0
                    for future in as_completed(future_to_ticker):
                        if self._cancel_event.is_set():
                            executor.shutdown(wait=False, cancel_futures=True)
                            result.cancelled = True
                            break
                        done += 1
                        if done % 50 == 0 or done <= 5:
                            self._progress(0.1 + (done / total * 0.7), f"Scoring {done}/{total}")
                        scores, direction = future.result()
                        if scores is None:
                            if direction == "filtered":
                                filtered_out += 1
                            continue
                        direction_counts[direction] = direction_counts.get(direction, 0) + 1
                        results.append(scores)
                        if len(results) % 20 == 0 or len(results) <= 5:
                            tag = "\u2713" if scores["total"] >= settings.get("min_score", 50) else "\u2717"
                            self._log(f"  {tag} {scores['ticker']}: {scores['total']:.1f}/100 ({direction})")
            else:
                for i, (ticker, df) in enumerate(batch_data.items(), 1):
                    if self._cancel_event.is_set():
                        self._log("\n\u23f9  Scan cancelled by user")
                        result.cancelled = True
                        break

                    progress = 0.1 + (i / total * 0.9) if total > 0 else 0.5
                    self._progress(progress, f"[{i}/{total}] {ticker}")

                    scores, direction = _score_one((ticker, df))
                    if scores is None:
                        if direction == "filtered":
                            filtered_out += 1
                        continue
                    direction_counts[direction] = direction_counts.get(direction, 0) + 1
                    results.append(scores)

                    if len(results) % 10 == 0 or len(results) <= 5:
                        score_val = scores["total"]
                        tag = "\u2713" if score_val >= settings.get("min_score", 50) else "\u2717"
                        self._log(f"  {tag} {ticker}: {score_val:.1f}/100 ({direction})")

            # ── Phase 2: Enrich top 200 for large universes ───────────────
            if is_large and results:
                results.sort(key=lambda x: x.get("total", 0) or 0, reverse=True)
                top_n = min(200, len(results))
                top = results[:top_n]
                rest = results[top_n:]
                self._log(f"Enriching top {top_n} of {len(results)} with fundamentals/sentiment...")
                from concurrent.futures import ThreadPoolExecutor

                def _enrich_one(r):
                    ticker = r["ticker"]
                    try:
                        enriched = self._enrich_with_providers(ticker, settings, global_data)
                        for k, v in enriched.items():
                            if k.startswith("_") and k not in r:
                                r[k] = v
                        # Also attach fundamentals if missing
                        df = batch_data.get(ticker)
                        if df is not None and getattr(df, '_fundamentals', None) is None:
                            try:
                                fund = fetch_fundamentals(ticker)
                                if fund is not None:
                                    object.__setattr__(df, '_fundamentals', fund)
                            except (RequestException, ValueError, KeyError):
                                pass
                    except (RequestException, ValueError, KeyError, AttributeError) as e:
                        logger.debug("Top enrichment failed for %s: %s", ticker, e)
                    return r

                with ThreadPoolExecutor(max_workers=8) as executor:
                    enriched_top = list(executor.map(_enrich_one, top))
                results = enriched_top + rest

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
            self._log(f"\nERROR: {e!s}")
        
        return result

    @trace(level=logging.INFO, log_args=True)
    def scan_stream(
        self,
        universe: str,
        settings: dict[str, Any],
        period: str = "1y",
        timeframe: str = "D",
        trend_filter: str = "All",
        index_symbol: str = "NSEI",
        on_batch: Callable[[list[dict]], None] | None = None,
    ) -> ScanResult:
        """
        Streaming scan — yields results batch-by-batch to the grid.

        Instead of fetching all ~5900 then scoring, we stream via
        fetch_batch_yfinance_stream: each parallel batch (~200-1000 tickers)
        is scored immediately and on_batch(chunk_results) is called so the
        UI can append to the table without waiting.

        For large universes (>500) we stream technical-only scores per chunk,
        then after all chunks enrich the global top-200 and emit an update
        batch (existing rows are replaced in-place in the UI).
        """
        # Allow callback via setter or direct param
        batch_cb = on_batch or self._batch_callback
        self._cancel_event.clear()
        result = ScanResult()

        try:
            try:
                tickers = get_universe(universe)
            except KeyError:
                tickers = UNIVERSES.get(universe, [])

            tf_names = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
            self._log("\n" + "=" * 50)
            self._log(f"START STREAM SCAN | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._log("=" * 50)
            self._log(f"Starting scan: {universe} ({len(tickers)} stocks)")
            self._log(f"Timeframe: {tf_names.get(timeframe, timeframe)} | Period: {period} | Filter: {trend_filter}")
            self._log(f"FastMA={settings.get('fast_ma_type','HMA')}{settings.get('fast_ma_len',40)} "
                       f"SlowMA={settings.get('slow_ma_type','EMA')}{settings.get('slow_ma_len',50)} "
                       f"RSI={settings.get('rsi_len',14)} Threshold={settings.get('min_score',50)}")

            self._progress(0.0, "Fetching NIFTY 50 index...")
            index_df = fetch_index_data(f"^{index_symbol}", period=period)
            if index_df is not None:
                self._log(f"{index_symbol} index loaded ({len(index_df)} bars)")
            else:
                self._log(f"Warning: {index_symbol} index unavailable, using proxy for RS")

            self._api_config = load_api_config()
            self._log("Fetching global macro/commodity data...")
            global_data = self._fetch_global_enrichment(settings)
            if global_data:
                self._log(f"Global enrichment: {len(global_data)} keys")

            is_large = len(tickers) > 500
            if is_large:
                self._log(f"Large universe streaming mode: {len(tickers)} stocks — per-batch technical scoring, enrich top 200 at end")

            results: list[dict] = []
            batch_data_all: dict = {}
            filtered_out = 0
            direction_counts = {"Bull": 0, "Bear": 0}
            total_tickers = len(tickers)
            fetched_so_far = 0
            batch_idx = 0

            def _score_one(item):
                ticker, df = item
                try:
                    if df is None or df.empty:
                        return None, "empty"
                    filter_result = check_filter(
                        df,
                        fast_ma_type=settings.get("fast_ma_type", "HMA"),
                        fast_ma_len=settings.get("fast_ma_len", 40),
                        slow_ma_type=settings.get("slow_ma_type", "EMA"),
                        slow_ma_len=settings.get("slow_ma_len", 50),
                        crossover_lookback=settings.get("crossover_lookback", 20),
                    )
                    if filter_result is None:
                        return None, "filtered"
                    direction = get_direction(filter_result)
                    if trend_filter == "Bullish Only" and direction != "Bull":
                        return None, "filtered"
                    if trend_filter == "Bearish Only" and direction != "Bear":
                        return None, "filtered"
                    if not is_large and getattr(df, '_fundamentals', None) is None:
                        try:
                            fund = fetch_fundamentals(ticker)
                            if fund is not None:
                                object.__setattr__(df, '_fundamentals', fund)
                        except (RequestException, ValueError, KeyError) as e:
                            logger.debug("Fundamentals fetch failed for %s: %s", ticker, e)
                    if is_large:
                        enriched = {**settings, **global_data, "_skip_vp": True}
                    else:
                        enriched = self._enrich_with_providers(ticker, settings, global_data)
                        enriched["_skip_vp"] = False
                    scores = compute_scores(df, timeframe=timeframe, index_df=index_df, settings=enriched)
                    if scores is None:
                        return None, "no_score"
                    scores["ticker"] = ticker
                    scores["trend_dir"] = direction
                    scores["trend_color"] = direction.lower()
                    if not is_large:
                        for k, v in enriched.items():
                            if k.startswith("_") and k not in scores:
                                scores[k] = v
                    return scores, direction
                except (RequestException, ValueError, KeyError, TypeError) as e:
                    logger.debug("Scoring failed for %s: %s", ticker, e)
                    return None, "error"

            # ── Stream per parallel batch ─────────────────────────────────
            for chunk_data in fetch_batch_yfinance_stream(
                tickers,
                period=period,
                timeframe=timeframe,
                cancel_event=self._cancel_event,
                on_fallback_progress=lambda done, total: self._progress(
                    0.05 + 0.05 * (done / max(total, 1)),
                    f"Fallback fetch {done}/{total} (jugaad/nselib)",
                ),
            ):
                if self._cancel_event.is_set():
                    result.cancelled = True
                    self._log("\n⏹  Scan cancelled by user")
                    break
                if not chunk_data:
                    continue
                batch_idx += 1
                batch_data_all.update(chunk_data)
                fetched_so_far += len(chunk_data)
                self._progress(0.05 + (fetched_so_far / max(total_tickers, 1) * 0.05),
                               f"Batch {batch_idx}: {len(chunk_data)} downloaded ({fetched_so_far}/{total_tickers})")
                self._log(f"Batch {batch_idx} received: {len(chunk_data)} tickers (cumulative {fetched_so_far}) — scoring...")

                chunk_results: list[dict] = []
                # For large universes use parallel scoring per chunk (smaller pool)
                if is_large and len(chunk_data) > 20:
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    max_w = min(4, (len(chunk_data) // 25) + 1)
                    with ThreadPoolExecutor(max_workers=max_w) as ex:
                        futs = {ex.submit(_score_one, item): item[0] for item in chunk_data.items()}
                        for fut in as_completed(futs):
                            if self._cancel_event.is_set():
                                break
                            scores, direction = fut.result()
                            if scores is None:
                                if direction == "filtered":
                                    filtered_out += 1
                                continue
                            direction_counts[direction] = direction_counts.get(direction, 0) + 1
                            chunk_results.append(scores)
                else:
                    for ticker, df in chunk_data.items():
                        if self._cancel_event.is_set():
                            break
                        scores, direction = _score_one((ticker, df))
                        if scores is None:
                            if direction == "filtered":
                                filtered_out += 1
                            continue
                        direction_counts[direction] = direction_counts.get(direction, 0) + 1
                        chunk_results.append(scores)

                if chunk_results:
                    # Keep global results sorted incrementally for top-200 calc later
                    results.extend(chunk_results)
                    # Progress: 0.1 - 0.8 range proportional to fetched
                    prog = 0.1 + (fetched_so_far / max(total_tickers, 1) * 0.7)
                    self._progress(prog, f"Scored {len(results)} passed ({fetched_so_far}/{total_tickers} fetched)")
                    for r in chunk_results[:3]:
                        tag = "✓" if r["total"] >= settings.get("min_score", 50) else "✗"
                        self._log(f"  {tag} {r['ticker']}: {r['total']:.1f}/100 ({r['trend_dir']})")
                    if len(chunk_results) > 3:
                        self._log(f"  ... +{len(chunk_results)-3} more in this batch")
                    if batch_cb:
                        try:
                            batch_cb(chunk_results)
                        except Exception as e:
                            logger.debug("on_batch callback failed: %s", e)

            # ── Phase 2: Enrich top 200 for large universes (update in place) ─
            if is_large and results:
                results.sort(key=lambda x: x.get("total", 0) or 0, reverse=True)
                top_n = min(200, len(results))
                top = results[:top_n]
                rest = results[top_n:]
                self._log(f"Enriching top {top_n} of {len(results)} with fundamentals/sentiment...")
                from concurrent.futures import ThreadPoolExecutor

                def _enrich_one(r):
                    ticker = r["ticker"]
                    try:
                        enriched = self._enrich_with_providers(ticker, settings, global_data)
                        for k, v in enriched.items():
                            if k.startswith("_") and k not in r:
                                r[k] = v
                        df = batch_data_all.get(ticker)
                        if df is not None and getattr(df, '_fundamentals', None) is None:
                            try:
                                fund = fetch_fundamentals(ticker)
                                if fund is not None:
                                    object.__setattr__(df, '_fundamentals', fund)
                            except (RequestException, ValueError, KeyError):
                                pass
                    except (RequestException, ValueError, KeyError, AttributeError) as e:
                        logger.debug("Top enrichment failed for %s: %s", ticker, e)
                    return r

                with ThreadPoolExecutor(max_workers=8) as executor:
                    enriched_top = list(executor.map(_enrich_one, top))
                # Merge: enriched top + rest, will trigger UI update
                results = enriched_top + rest
                if batch_cb and enriched_top:
                    try:
                        # Send enriched top as update batch — UI will replace existing tickers
                        batch_cb(enriched_top)
                        self._log(f"Top {top_n} enrichment complete — grid updated")
                    except Exception as e:
                        logger.debug("on_batch enrichment callback failed: %s", e)

            results.sort(key=lambda x: x.get("total", 0) or 0, reverse=True)
            passed = len([r for r in results if r["total"] >= settings.get("min_score", 50)])
            self._log("\n" + "━" * 25 + " Stream Scan Complete ")
            missing_final = [t for t in tickers if t not in batch_data_all]
            self._log(f"  Total tickers: {total_tickers} | fetched: {len(batch_data_all)}")
            if missing_final:
                self._log(
                    f"  ⚠ {len(missing_final)} tickers unavailable on all providers "
                    "(yfinance, jugaad-data, nselib)"
                )
            self._log(f"  Filtered out: {filtered_out}")
            self._log(f"  Passed filter: {len(results)} ({direction_counts.get('Bull',0)} Bull, {direction_counts.get('Bear',0)} Bear)")
            self._log(f"  Scored {settings.get('min_score',50)}+: {passed}")
            result.results = results
            result.filtered_out = filtered_out
            result.direction_counts = direction_counts

        except Exception as e:
            result.error = str(e)
            self._log(f"\nERROR: {e!s}")

        return result


# Convenience function for simple usage
def run_scan(
    universe: str,
    settings: dict[str, Any],
    period: str = "1y",
    timeframe: str = "D",
    trend_filter: str = "All",
    index_symbol: str = "NSEI",
    progress_callback: Callable[[float, str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> ScanResult:
    """Convenience function to run a scan without managing engine instance."""
    engine = ScannerEngine()
    if progress_callback:
        engine.set_progress_callback(progress_callback)
    if log_callback:
        engine.set_log_callback(log_callback)
    return engine.scan(universe, settings, period, timeframe, trend_filter, index_symbol)