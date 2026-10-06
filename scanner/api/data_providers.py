"""
Multi-source data provider with fallback chain for Indian stock market.

Provider priority for OHLCV:
  1. yfinance — Yahoo Finance, no auth needed
  2. jugaad-data — NSE official API, no auth needed
  3. nselib — NSE library, no auth needed
  4. Market Lens — close+volume only (NaN high/low), last resort

Provider strategy for Fundamentals — fetch ALL free sources, save each
raw result, then compile the four score fields (pe_ratio, eps_growth,
rev_growth, roe; first provider to supply a field wins):
  1. Finnhub — institutional-grade data (free API key)
  2. Alpha Vantage — overview (free API key)
  3. Screener.in — Stock PE + ROE (no key, cached)
  4. Market Lens — NSE profile + quarterly financials (no ROE)
  5. yfinance .info — all fields (429-prone, last full source)
  6. nselib pe_ratio — bulk P/E ratio for all stocks

All providers normalize data to a common DataFrame format:
  columns = [open, high, low, close, volume]
"""

import glob
import hashlib
import io
import json
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from ..shared import db
from ..shared.cache import TTLCache
from .yahoo_symbol import yf_quote_variants

logger = logging.getLogger(__name__)

# Load API keys from config file (re-read on each call to pick up live edits)
_API_KEYS: dict[str, str] = {}
_API_KEYS_MTIME: float = 0.0
_config_file = Path(__file__).parent.parent / "api_config.json"


def _get_api_key(key_name: str) -> str:
    """Resolve an API key: config file first, then os.environ.

    Re-reads api_config.json when its mtime changes so key rotations
    take effect without restarting the process.
    """
    global _API_KEYS, _API_KEYS_MTIME
    try:
        mtime = _config_file.stat().st_mtime if _config_file.exists() else 0.0
    except OSError:
        mtime = 0.0
    if mtime != _API_KEYS_MTIME:
        _API_KEYS_MTIME = mtime
        try:
            if _config_file.exists():
                with open(_config_file) as f:
                    _API_KEYS = json.load(f)
            else:
                _API_KEYS = {}
        except Exception:
            logger.info("API key config parse failed", exc_info=True)
    return _API_KEYS.get(key_name) or os.environ.get(key_name, "")


# ── Cache Directory ────────────────────────────────────────────────────────
_CACHE_WRITE_LOCK = threading.Lock()
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache"
)
CACHE_TTL_HOURS = 4  # Cache expires after 4 hours
# ponytail: hard row cap on price_cache. A full-market scan writes ~2.4k
# parquet rows (~47 KB each, so ~115 MB per 4h generation), and the TTL alone
# left the file at whatever a few concurrent generations peaked at. Keeping
# the newest MAX_ROWS bounds the steady-state size. Upgrade path: make the cap
# a setting, or store parquet on disk with only paths in sqlite, if a smaller
# footprint is ever worth the extra I/O.
PRICE_CACHE_MAX_ROWS = 6000
# Only VACUUM when the db has at least this much reclaimable slack, so the
# routine no-op prune (nothing expired, under the cap) stays free.
VACUUM_SLACK_MB = 8.0

# Fundamentals change quarterly — persist each provider chain's result so
# single-ticker fetches (fallback, detail) and scans share one disk-backed read.
# Two levels share this cache: raw per-provider results under
# make_key(ticker, provider) — saved the moment a provider returns, never
# erased by later failures — and the compiled four-field score dict under
# make_key(ticker), stored only when COMPLETE (a partial compile is retried
# against the still-cached raws next call instead of freezing for 6h).
_FUNDAMENTALS_CACHE: TTLCache = TTLCache(ttl=6 * 3600, namespace="fundamentals")

# The four fields _score_fundamentals_dict consumes (5 pts each, max 20).
_FUND_SCORE_FIELDS = ("pe_ratio", "eps_growth", "rev_growth", "roe")

# Which score fields each provider can fill — lets the chain skip any
# provider that cannot add anything new (and stop once all fields are set).
_PROVIDER_FIELDS: dict[str, tuple[str, ...]] = {
    "finnhub": _FUND_SCORE_FIELDS,
    "alpha_vantage": _FUND_SCORE_FIELDS,
    "screener": ("pe_ratio", "roe"),
    "marketlens": ("pe_ratio", "eps_growth", "rev_growth"),
    "yfinance": _FUND_SCORE_FIELDS,
    "nselib": ("pe_ratio",),
}


from ..shared._index_utils import _normalize_daily_index

_PRUNE_LOCK = threading.Lock()
_last_prune_ts = 0.0
PRUNE_INTERVAL_SECONDS = 3600  # at most one stale-cache sweep per process per hour


def prune_stale_cache(force: bool = False) -> int:
    """Delete expired cache rows (and dormant pre-migration files).

    The db half sweeps ``price_cache`` rows past their expiry; the file half
    removes stale parquet/meta pairs left by the pre-sqlite cache dir.
    Sweeping is rate-limited per process (``PRUNE_INTERVAL_SECONDS``) so scan
    starts stay cheap.

    Returns the number of expired rows + cap-evicted live rows + stale
    (pkl, meta) pairs removed.
    """
    global _last_prune_ts
    now = time.time()
    with _PRUNE_LOCK:
        if not force and now - _last_prune_ts < PRUNE_INTERVAL_SECONDS:
            return 0
        _last_prune_ts = now

    removed = 0
    try:
        conn = db.get_conn()
        removed += conn.execute(
            "DELETE FROM price_cache WHERE expires <= ?", (now,)
        ).rowcount
        # Enforce the row cap on whatever is still live: keep the newest
        # MAX_ROWS by expiry. Without this the file only ever grows to the
        # peak of a few overlapping 4h generations (sqlite does not shrink
        # a file on delete).
        removed += conn.execute(
            "DELETE FROM price_cache WHERE cache_key NOT IN"
            " (SELECT cache_key FROM price_cache ORDER BY expires DESC LIMIT ?)",
            (PRICE_CACHE_MAX_ROWS,),
        ).rowcount
    except Exception as e:
        logger.info("Stale price-cache prune failed: %s", e)

    # Hand the freed pages back to the OS. sqlite never shrinks the file on
    # delete, so a trimmed cache would otherwise leave the file pinned at its
    # high-water mark. A plain VACUUM is the only thing that reclaims here:
    # incremental_vacuum needs auto_vacuum=INCREMENTAL (a schema-level change
    # that cannot be applied to a populated db without a full rebuild anyway)
    # and was measured reclaiming ~1 page per call. Gated on a real amount of
    # slack so the common no-op prune never pays for a rewrite.
    try:
        conn = db.get_conn()
        page = conn.execute("PRAGMA page_size").fetchone()[0]
        free_mb = conn.execute("PRAGMA freelist_count").fetchone()[0] * page / 1e6
        if free_mb >= VACUUM_SLACK_MB:
            conn.execute("VACUUM")
            logger.info("Reclaimed %.1f MB of price-cache slack via VACUUM", free_mb)
    except Exception as e:
        logger.info("price-cache vacuum skipped: %s", e)

    today = date.today().isoformat()
    try:
        for meta in glob.glob(os.path.join(CACHE_DIR, "*.meta")):
            try:
                with open(meta) as f:
                    ts = json.load(f).get("timestamp", "")
            except Exception:
                continue  # unreadable/corrupt meta -- leave the pair alone
            if ts[:10] == today:
                continue
            try:
                parquet = meta[:-5] + ".parquet"
                # Re-check meta timestamp before deleting — a concurrent
                # writer may have refreshed the entry between our first read
                # and now.
                with open(meta) as f:
                    ts2 = json.load(f).get("timestamp", "")
                if ts2[:10] == today:
                    continue  # refreshed by a concurrent writer — skip
                os.remove(parquet)
            except OSError:
                logger.debug("Stale parquet removal failed: %s", parquet, exc_info=True)
            try:
                os.remove(meta[:-5] + ".pkl")  # legacy extension
            except OSError:
                logger.debug(
                    "Stale pkl removal failed: %s", meta[:-5] + ".pkl", exc_info=True
                )
            try:
                os.remove(meta)
            except OSError:
                logger.debug("Stale meta removal failed: %s", meta, exc_info=True)
            removed += 1
    except Exception as e:
        logger.info("Stale-cache prune failed: %s", e)
    if removed:
        logger.info("Pruned %d stale cache entrie(s) from previous days", removed)
    return removed


def cache_health() -> dict:
    """Price-cache census: live vs expired rows in the db.

    Returns ``{price_entries, stale_entries, last_prune}`` where
    ``price_entries`` is the TOTAL row count and ``stale_entries`` is the
    expired-but-not-yet-pruned subset.  ``last_prune`` is an ISO timestamp
    of the last ``prune_stale_cache`` sweep in this process ("" when never
    pruned).
    """
    fresh = stale = 0
    try:
        row = (
            db.get_conn()
            .execute(
                "SELECT COUNT(*) AS n,"
                " COALESCE(SUM(CASE WHEN expires <= ? THEN 1 ELSE 0 END), 0) AS s"
                " FROM price_cache",
                (time.time(),),
            )
            .fetchone()
        )
        fresh = int(row["n"]) - int(row["s"])
        stale = int(row["s"])
    except Exception as e:
        logger.info("cache_health census failed: %s", e)
    with _PRUNE_LOCK:
        last_ts = _last_prune_ts
    last_prune = (
        datetime.fromtimestamp(last_ts).isoformat(timespec="minutes")
        if last_ts > 0
        else ""
    )
    return {
        "price_entries": fresh + stale,
        "stale_entries": stale,
        "last_prune": last_prune,
    }


def _cache_key(ticker: str, period: str, provider: str) -> str:
    """Generate a cache file key.

    Keyed on normalized ticker + period + provider only; freshness is
    enforced by the ``meta`` timestamp against ``CACHE_TTL_HOURS``. (A
    previous day-keyed scheme made entries unreachable after midnight while
    still counting as fresh.)
    """
    norm = str(ticker or "").strip().upper()
    raw = f"{norm}_{period}_{provider}"
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()


def _legacy_cache_key(ticker: str, period: str, provider: str) -> str:
    """Pre-fix day-keyed cache key (kept for reading legacy entries)."""
    today = date.today().isoformat()
    norm = str(ticker or "").strip().upper()
    raw = f"{norm}_{period}_{provider}_{today}"
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()


def _read_cache_pair(
    cache_file: str, meta_file: str, ticker: str
) -> tuple[pd.DataFrame, datetime] | None:
    """(frame, cached timestamp) when the pkl+meta pair exists and is fresh."""
    if not os.path.exists(cache_file) or not os.path.exists(meta_file):
        return None
    try:
        with open(meta_file) as f:
            meta = json.load(f)
        cached_time = datetime.fromisoformat(meta["timestamp"])
        age_hours = (datetime.now() - cached_time).total_seconds() / 3600

        if age_hours > CACHE_TTL_HOURS:
            return None

        return _normalize_daily_index(pd.read_parquet(cache_file)), cached_time
    except Exception as e:
        logger.info("Cache read failed for %s: %s", ticker, e)
        return None


def _read_price_row(key: str, ticker: str) -> pd.DataFrame | None:
    """Return the frame from the db if the BLOB row is still fresh."""
    row = (
        db.get_conn()
        .execute("SELECT payload, expires FROM price_cache WHERE cache_key = ?", (key,))
        .fetchone()
    )
    if row is None or time.time() >= row["expires"]:
        return None
    try:
        return _normalize_daily_index(pd.read_parquet(io.BytesIO(row["payload"])))
    except Exception as e:
        logger.info("Cache read failed for %s: %s", ticker, e)
        return None


def _import_price_pair(key: str, ticker: str) -> pd.DataFrame | None:
    """Fold a pre-migration parquet+meta file pair into the db (once).

    Returns the frame when the pair exists and is fresh (so caches written
    before the sqlite migration keep serving until they age out).
    """
    cache_file = os.path.join(CACHE_DIR, f"{key}.parquet")
    meta_file = os.path.join(CACHE_DIR, f"{key}.meta")
    hit = _read_cache_pair(cache_file, meta_file, ticker)
    if hit is None:
        return None
    df, cached_time = hit
    expires = cached_time.timestamp() + CACHE_TTL_HOURS * 3600
    try:
        _store_price_row(key, df, expires)
    except Exception:
        logger.debug("Cache import failed for %s", ticker, exc_info=True)
    return df


def _encode_price_blob(df: pd.DataFrame) -> bytes:
    """Serialize the frame as parquet bytes (CPU-heavy — keep out of locks)."""
    buf = io.BytesIO()
    try:
        df.to_parquet(buf, index=True)
    except Exception:
        import pyarrow as _pa
        import pyarrow.parquet as _pq

        table = _pa.Table.from_pandas(df.reset_index(drop=False), preserve_index=True)
        _pq.write_table(table, buf)
    return buf.getvalue()


def _upsert_price_row(key: str, payload: bytes, expires: float) -> None:
    db.get_conn().execute(
        "INSERT INTO price_cache (cache_key, payload, expires) VALUES (?, ?, ?)"
        " ON CONFLICT (cache_key)"
        " DO UPDATE SET payload = excluded.payload, expires = excluded.expires",
        (key, payload, expires),
    )


def _store_price_row(key: str, df: pd.DataFrame, expires: float) -> None:
    """Serialize the frame as parquet bytes and upsert the db row."""
    _upsert_price_row(key, _encode_price_blob(df), expires)


def _get_cached(ticker: str, period: str, provider: str) -> pd.DataFrame | None:
    """Retrieve cached data if fresh enough.

    Reads the current (date-independent) key first, then falls back to the
    legacy day-keyed entry so caches written by older versions keep working
    until they age out and are pruned.
    """
    key = _cache_key(ticker, period, provider)
    legacy = _legacy_cache_key(ticker, period, provider)
    for k in (key, legacy):
        hit = _read_price_row(k, ticker)
        if hit is not None:
            return hit
    # db miss → import a legacy file pair (pre-sqlite cache) once
    for k in (key, legacy):
        hit = _import_price_pair(k, ticker)
        if hit is not None:
            return hit
    return None


def _set_cached(ticker: str, period: str, provider: str, df: pd.DataFrame):
    """Store data in the price cache (db BLOB with a CACHE_TTL_HOURS expiry)."""
    key = _cache_key(ticker, period, provider)
    try:
        # Normalize + parquet-encode outside the write lock — 8 chunk
        # threads write here and the encode dwarfs the upsert; only the
        # db write needs serializing.
        payload = _encode_price_blob(_normalize_daily_index(df))
        expires = time.time() + CACHE_TTL_HOURS * 3600
    except Exception as e:
        logger.debug("Cache write failed for %s: %s", ticker, e)
        return
    with _CACHE_WRITE_LOCK:
        try:
            _upsert_price_row(key, payload, expires)
        except Exception as e:
            logger.debug("Cache write failed for %s: %s", ticker, e)


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER: jugaad-data (NSE Official API)
# ══════════════════════════════════════════════════════════════════════════════


def _fetch_jugaad(ticker: str, period: str) -> pd.DataFrame | None:
    """Fetch OHLCV from NSE via jugaad-data. No auth needed."""
    try:
        from datetime import date, timedelta

        from jugaad_data.nse import stock_df

        period_days = {"6mo": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
        days = period_days.get(period, 365)
        end = date.today()
        start = end - timedelta(days=days)

        df = stock_df(symbol=ticker, from_date=start, to_date=end, series="EQ")

        if df is None or df.empty:
            return None

        # Normalize columns
        df = df.rename(
            columns={
                "OPEN": "open",
                "HIGH": "high",
                "LOW": "low",
                "CLOSE": "close",
                "VOLUME": "volume",
                "DATE": "date",
            }
        )

        # Set DATE as index (needed for resampling). jugaad returns rows
        # newest-first — flip to ascending so .iloc[-1] is the latest bar.
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df.index.name = None
            df = df.sort_index()

        # Keep required columns
        cols = ["open", "high", "low", "close", "volume"]
        for c in cols:
            if c not in df.columns:
                return None

        df = df[cols].copy()
        return df.dropna()

    except ImportError:
        return None
    except Exception as e:
        logger.info("jugaad-data failed for %s: %s", ticker, e)
        return None


def _fetch_jugaad_index(ticker: str, period: str) -> pd.DataFrame | None:
    """Fetch index data from NSE via jugaad-data."""
    try:
        from datetime import date, timedelta

        from jugaad_data.nse import index_df

        period_days = {"6mo": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
        days = period_days.get(period, 365)
        end = date.today()
        start = end - timedelta(days=days)

        # Map index tickers to NSE index names
        index_map = {
            "^NSEI": "NIFTY 50",
            "^NSEBANK": "NIFTY BANK",
            "NIFTY 50": "NIFTY 50",
        }
        index_name = index_map.get(ticker, ticker)

        df = index_df(symbol=index_name, from_date=start, to_date=end)

        if df is None or df.empty:
            return None

        # Normalize columns
        col_map = {}
        date_col = None
        for c in df.columns:
            cl = c.lower().strip()
            if "open" in cl:
                col_map[c] = "open"
            elif "high" in cl:
                col_map[c] = "high"
            elif "low" in cl:
                col_map[c] = "low"
            elif "close" in cl:
                col_map[c] = "close"
            elif "volume" in cl or "turnover" in cl:
                col_map[c] = "volume"
            elif "date" in cl:
                # index_df() reports the trading day under HistoricalDate.
                # Keep it so it can become the DatetimeIndex below — without
                # this the frame has a RangeIndex and Relative-Strength
                # date alignment silently breaks (epoch-1970 dates).
                date_col = c

        df = df.rename(columns=col_map)
        cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in cols if c in df.columns]

        if len(available) < 4:
            return None

        # Pull the date column out first, then keep only OHLCV.
        if date_col is not None and date_col not in available:
            df = df[[date_col] + available].copy()
        else:
            df = df[available].copy()

        if date_col is not None and date_col in df.columns:
            dates = pd.to_datetime(df[date_col])
            df = df.drop(columns=[date_col])
            df.index = dates
            df.index.name = None

        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            return None

        # jugaad returns rows newest-first — flip to ascending so .iloc[-1]
        # and date-mask alignment in the scorers use the latest bar.
        df = df.sort_index()

        return df.dropna()

    except ImportError:
        return None
    except Exception as e:
        logger.info("jugaad-data index failed for %s: %s", ticker, e)
        return None


def _fetch_marketlens_index(ticker: str, period: str) -> pd.DataFrame | None:
    """Last-resort index quote from Market Lens (today + derived prev close).

    Only NIFTY 50 / NIFTY BANK exist there. The 2-row frame is enough for
    the hero quote; the RS scorer already rejects frames shorter than
    rs_length + 5 bars, so a scan falls back to its proxy exactly as when
    the index is missing entirely.
    """
    del period  # single-day snapshot — period is irrelevant here
    try:
        from .market_lens import get_indices

        name = {"^NSEI": "NIFTY_50", "^NSEBANK": "BANKNIFTY"}.get(ticker)
        if name is None:
            return None
        rows = get_indices() or []
        row = next((r for r in rows if r.get("ticker") == name), None)
        if not row:
            return None
        value = float(row.get("value"))
        change = float(row.get("change") or 0.0)
        if not value or change <= -100:
            return None
        prev = value / (1 + change / 100.0)
        if not prev:
            return None
        today = pd.Timestamp.now().normalize()
        df = pd.DataFrame(
            {
                "open": [prev, value],
                "high": [value, value],
                "low": [prev, value],
                "close": [prev, value],
                "volume": [0.0, 0.0],
            },
            index=pd.DatetimeIndex([today - pd.Timedelta(days=1), today]),
        )
        df.index.name = None
        return df
    except Exception as e:
        logger.info("Market Lens index failed for %s: %s", ticker, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER: yfinance (Yahoo Finance)
# ══════════════════════════════════════════════════════════════════════════════


def _fetch_yfinance(ticker: str, period: str) -> pd.DataFrame | None:
    """Fetch OHLCV from Yahoo Finance."""
    try:
        import yfinance as yf

        nse_ticker = f"{ticker}.NS"
        stock = yf.Ticker(nse_ticker)
        df = stock.history(period=period, auto_adjust=True)

        if df is None or df.empty:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        return df.dropna()

    except ImportError:
        return None
    except Exception as e:
        logger.info("yfinance failed for %s: %s", ticker, e)
        return None


def _fetch_yfinance_index(ticker: str, period: str) -> pd.DataFrame | None:
    """Fetch index data from Yahoo Finance."""
    try:
        import yfinance as yf

        index = yf.Ticker(ticker)
        df = index.history(period=period, auto_adjust=True)

        if df is None or df.empty:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        return df.dropna()

    except ImportError:
        return None
    except Exception as e:
        logger.info("yfinance index failed for %s: %s", ticker, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER: nselib (NSE Library)
# ══════════════════════════════════════════════════════════════════════════════


def _fetch_nselib(ticker: str, period: str) -> pd.DataFrame | None:
    """Fetch OHLCV from NSE via nselib."""
    try:
        from nselib import capital_market

        period_days = {"6mo": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
        days = period_days.get(period, 365)
        end = date.today()
        start = end - timedelta(days=days)

        from_date = start.strftime("%d-%m-%Y")
        to_date = end.strftime("%d-%m-%Y")

        df = capital_market.price_volume_and_deliverable_position_data(
            symbol=ticker, from_date=from_date, to_date=to_date
        )

        if df is None or df.empty:
            return None

        # Normalize columns
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if "open" in cl:
                col_map[c] = "open"
            elif "high" in cl:
                col_map[c] = "high"
            elif "low" in cl:
                col_map[c] = "low"
            elif "close" in cl or "last" in cl:
                col_map[c] = "close"
            elif "quantity" in cl or "volume" in cl or "traded" in cl:
                col_map[c] = "volume"

        df = df.rename(columns=col_map)
        cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in cols if c in df.columns]

        if len(available) < 4:
            return None

        df = df[available].copy()
        return df.dropna()

    except ImportError:
        return None
    except Exception as e:
        logger.info("nselib failed for %s: %s", ticker, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER: Market Lens (price history — last resort)
# ══════════════════════════════════════════════════════════════════════════════


def _fetch_marketlens_price(ticker: str, period: str) -> pd.DataFrame | None:
    """Daily close+volume from Market Lens — last-resort fallback frame.

    ponytail: close-only source — open=close and high/low stay NaN, so ATR/
    ADX/stochastic read NaN (every consumer isnan-guards them) and the ADX
    entry gate cannot fire on these frames. Reached only after yfinance,
    jugaad-data and nselib have all failed, so it converts a dead ticker
    into a degraded one instead of replacing better OHLC data.
    """
    try:
        from .market_lens import get_price_history

        ml_period = {"6mo": "6M", "1y": "1Y", "2y": "5Y", "3y": "5Y", "5y": "5Y"}.get(
            period, "1Y"
        )
        rows = get_price_history(ticker, ml_period)
        if not rows:
            return None
        dates, closes, volumes = [], [], []
        for r in rows:
            if not isinstance(r, dict):
                continue
            try:
                d = pd.Timestamp(r.get("date"))
                c = float(r.get("price"))
            except (TypeError, ValueError):
                continue
            if pd.isna(d) or pd.isna(c) or c <= 0:
                continue
            try:
                v = float(r.get("volume") or 0)
            except (TypeError, ValueError):
                v = 0.0
            dates.append(d)
            closes.append(c)
            volumes.append(v)
        if not dates:
            return None
        df = pd.DataFrame(
            {
                "open": closes,
                "high": float("nan"),
                "low": float("nan"),
                "close": closes,
                "volume": volumes,
            },
            index=pd.DatetimeIndex(dates),
        ).sort_index()
        df.index.name = None
        return df
    except Exception as e:
        logger.info("Market Lens price history failed for %s: %s", ticker, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL DATA PROVIDERS
# ══════════════════════════════════════════════════════════════════════════════


def _fetch_fundamentals_finnhub(ticker: str) -> dict | None:
    """Fetch fundamentals from Finnhub (free tier, institutional-grade)."""
    try:
        import requests

        # Finnhub uses .NS suffix for NSE stocks
        finnhub_ticker = f"{ticker}.NS"
        api_key = _get_api_key("FINNHUB_API_KEY")

        if not api_key:
            return None

        # Get basic financials
        url = f"https://finnhub.io/api/v1/stock/metric?symbol={finnhub_ticker}&metric=all&token={api_key}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data or "metric" not in data:
            return None

        metric = data["metric"]

        pe_ratio = metric.get("peTTM") or metric.get("peBasicExclExtraTTM")
        roe = metric.get("roeTTM")
        if roe is not None:
            roe = roe * 100 if abs(roe) < 1 else roe

        # Get earnings data for growth
        eps_growth = None
        rev_growth = None

        try:
            earnings_url = f"https://finnhub.io/api/v1/stock/earnings?symbol={finnhub_ticker}&token={api_key}"
            earnings_resp = requests.get(earnings_url, timeout=10)
            earnings_resp.raise_for_status()
            earnings_data = earnings_resp.json()

            if earnings_data and len(earnings_data) >= 2:
                # Compare latest two quarters
                latest = earnings_data[0]
                prev = earnings_data[1]
                if prev.get("eps") and prev["eps"] != 0 and latest.get("eps"):
                    eps_growth = (
                        (latest["eps"] - prev["eps"]) / abs(prev["eps"])
                    ) * 100
                if (
                    prev.get("revenue")
                    and prev["revenue"] != 0
                    and latest.get("revenue")
                ):
                    rev_growth = (
                        (latest["revenue"] - prev["revenue"]) / abs(prev["revenue"])
                    ) * 100
        except Exception as e:
            logger.info("Finnhub earnings fetch failed for %s: %s", ticker, e)

        return {
            "pe_ratio": pe_ratio,
            "eps_growth": eps_growth,
            "rev_growth": rev_growth,
            "roe": roe,
        }

    except Exception as e:
        logger.info("Finnhub fundamentals failed for %s: %s", ticker, e)
        return None


def _fetch_fundamentals_alpha_vantage(ticker: str) -> dict | None:
    """Fetch fundamentals from Alpha Vantage (free API key)."""
    try:
        import requests

        api_key = _get_api_key("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            return None

        # Get overview data
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}.NS&apikey={api_key}"
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if not data or "PERatio" not in data:
            return None

        pe_ratio = float(data.get("PERatio", 0)) or None
        roe = (
            float(data.get("ReturnOnEquityTTM", 0)) * 100
            if data.get("ReturnOnEquityTTM")
            else None
        )
        eps_growth = (
            float(data.get("EPSGrowthTTM", 0)) * 100
            if data.get("EPSGrowthTTM")
            else None
        )
        rev_growth = (
            float(data.get("RevenueGrowthTTM", 0)) * 100
            if data.get("RevenueGrowthTTM")
            else None
        )

        return {
            "pe_ratio": pe_ratio,
            "eps_growth": eps_growth,
            "rev_growth": rev_growth,
            "roe": roe,
        }

    except Exception as e:
        logger.info("Alpha Vantage failed for %s: %s", ticker, e)
        return None


def _is_rate_limited(err: Exception) -> bool:
    """True for Yahoo-style 429 / rate-limit failures (worth retrying)."""
    msg = str(err).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


def _fetch_fundamentals_yfinance(ticker: str) -> dict | None:
    """Fetch fundamentals from yfinance .info (429-tolerant).

    A burst of enrichment workers can trip Yahoo's rate limit; wait out the
    window (2s, then 4s) before retrying instead of caching the miss.
    Non-rate errors fail fast — retrying them only wastes the scan budget.
    """
    try:
        import yfinance as yf

        info = {}
        for attempt in range(3):
            rate_limited = False
            try:
                for yf_ticker in yf_quote_variants(ticker):
                    info = yf.Ticker(yf_ticker).info or {}
                    if info:
                        break
            except Exception as e:
                if not _is_rate_limited(e):
                    logger.info("yfinance fundamentals failed for %s: %s", ticker, e)
                    return None
                rate_limited = True
                info = {}
            if info:
                break
            if not (rate_limited and attempt < 2):
                break  # no data (not a rate limit) — fail fast
            time.sleep(2 * (attempt + 1))  # 2s, 4s

        if not info:
            return None

        pe_ratio = info.get("trailingPE")

        eps_growth = info.get("earningsGrowth")
        if eps_growth is not None:
            eps_growth = eps_growth * 100 if abs(eps_growth) < 1 else eps_growth
        else:
            earnings_q = info.get("earningsQuarterlyGrowth")
            if earnings_q is not None:
                eps_growth = earnings_q * 100 if abs(earnings_q) < 1 else earnings_q

        rev_growth = info.get("revenueGrowth")
        if rev_growth is not None:
            rev_growth = rev_growth * 100 if abs(rev_growth) < 1 else rev_growth

        roe = info.get("returnOnEquity")
        if roe is not None:
            roe = roe * 100 if abs(roe) < 1 else roe

        return {
            "pe_ratio": pe_ratio,
            "eps_growth": eps_growth,
            "rev_growth": rev_growth,
            "roe": roe,
        }

    except Exception as e:
        logger.info("yfinance fundamentals failed for %s: %s", ticker, e)
        return None


def _fetch_fundamentals_nselib(ticker: str) -> dict | None:
    """Fetch P/E ratio from nselib (bulk data, single call for all stocks)."""
    try:
        from nselib import capital_market

        today = date.today()
        to_date = today.strftime("%d-%m-%Y")

        df = capital_market.pe_ratio(trade_date=to_date)

        if df is None or df.empty:
            return None

        sym_col = next(c for c in df.columns if "symbol" in c.lower())
        r = df[df[sym_col].str.strip() == ticker]

        if r.empty:
            return None

        pe_val = r.iloc[0].get("SYMBOLP/E") or r.iloc[0].get("ADJUSTEDP/E")
        if pe_val is not None:
            pe_val = float(pe_val)

        return {
            "pe_ratio": pe_val,
            "eps_growth": None,
            "rev_growth": None,
            "roe": None,
        }

    except Exception as e:
        logger.info("nselib P/E failed for %s: %s", ticker, e)
        return None


def _fetch_fundamentals_marketlens(ticker: str) -> dict | None:
    """Market Lens profile + quarterly financials — P/E + YoY, no ROE."""
    try:
        from .market_lens import get_quarterly, get_stock, yoy_growth

        prof = get_stock(ticker)
        if not prof:
            return None
        try:
            pe = float(prof.get("peRatio"))
        except (TypeError, ValueError):
            pe = None
        quarterly = get_quarterly(ticker)
        fund = {
            "pe_ratio": pe,
            "eps_growth": yoy_growth(quarterly, "eps"),
            "rev_growth": yoy_growth(quarterly, "totalIncome"),
            "roe": None,
        }
        return fund if any(v is not None for v in fund.values()) else None
    except Exception as e:
        logger.info("Market Lens fundamentals failed for %s: %s", ticker, e)
        return None


def _fetch_fundamentals_screener(ticker: str) -> dict | None:
    """Screener.in (keyless, cached) — Stock PE + ROE only."""
    try:
        from .indian_fundamentals import fetch_peer_comparison

        peer = fetch_peer_comparison(ticker)
        if not peer:
            return None
        fund = {
            "pe_ratio": peer.stock_pe,
            "eps_growth": None,
            "rev_growth": None,
            "roe": peer.stock_roe,
        }
        return fund if any(v is not None for v in fund.values()) else None
    except Exception as e:
        logger.info("Screener fundamentals failed for %s: %s", ticker, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROVIDER CLASS
# ══════════════════════════════════════════════════════════════════════════════

from .providers import TIMEOUT as _TIMEOUT
from .providers import call_with_timeout as _call_with_timeout


class DataProvider:
    """
    Multi-source data provider with automatic fallback.

    Usage:
        provider = DataProvider()
        df = provider.fetch_stock("RELIANCE", period="1y")
        fund = provider.fetch_fundamentals("RELIANCE")
    """

    def __init__(self, use_cache: bool = True):
        """
        Args:
            use_cache: Whether to use disk cache for API responses
        """
        self.use_cache = use_cache

        # Track which provider was last used (for UI display)
        # Protected by _meta_lock since multiple threads may call fetch_stock.
        import threading

        self._meta_lock = threading.Lock()
        self.last_provider = None
        self.last_error = None

    def fetch_stock(
        self,
        ticker: str,
        period: str = "1y",
        skip: tuple[str, ...] = (),
        provider_timeout: float | None = None,
    ) -> pd.DataFrame | None:
        """
        Fetch OHLCV data with provider fallback chain.

        Priority:
          1. Cache (if enabled)
          2. yfinance
          3. jugaad-data
          4. nselib
          5. Market Lens (close+volume only — degraded frame)

        Args:
            skip: Provider names to exclude from the chain. Used by the
                  batch-download fallback path, where yfinance just failed
                  at scale (rate limit / outage) and should not be retried
                  per ticker.
            provider_timeout: When set, each provider call is capped at this
                  many seconds. A provider that exceeds the cap is treated
                  like one that returned no data (its thread keeps running
                  in the background as a daemon). Used by the batch fallback
                  so dead symbols fail fast instead of stalling a worker.
        """
        with self._meta_lock:
            self.last_provider = None
            self.last_error = None

        # Check cache first
        if self.use_cache:
            cached = _get_cached(ticker, period, "cache")
            if cached is not None:
                with self._meta_lock:
                    self.last_provider = "cache"
                return cached

        # Provider chain — full OHLC first; Market Lens last because it is
        # close-only and must never shadow a working OHLC source.
        providers = [
            ("yfinance", lambda: _fetch_yfinance(ticker, period)),
            ("jugaad", lambda: _fetch_jugaad(ticker, period)),
            ("nselib", lambda: _fetch_nselib(ticker, period)),
            ("marketlens", lambda: _fetch_marketlens_price(ticker, period)),
        ]

        for name, fetch_fn in providers:
            if name in skip:
                continue
            try:
                if provider_timeout:
                    df = _call_with_timeout(fetch_fn, provider_timeout)
                    if df is _TIMEOUT:
                        with self._meta_lock:
                            self.last_error = (
                                f"{name}: timed out after {provider_timeout}s"
                            )
                        continue
                else:
                    df = fetch_fn()
                if df is not None and not df.empty and len(df) >= 50:
                    with self._meta_lock:
                        self.last_provider = name
                    if self.use_cache:
                        _set_cached(ticker, period, "cache", df)
                    return df
            except Exception as e:
                with self._meta_lock:
                    prev = self.last_error
                    err = f"{name}: {e!s}"
                    self.last_error = err if not prev else f"{prev}; {err}"
                logger.debug("Provider %s failed for %s: %s", name, ticker, e)
                continue

        with self._meta_lock:
            if not self.last_error:
                self.last_error = "All providers failed"
            else:
                self.last_error = f"All providers failed ({self.last_error})"
        logger.info("All providers failed for %s: %s", ticker, self.last_error)
        return None

    def fetch_index(
        self, ticker: str, period: str = "1y", provider_timeout: float | None = 30.0
    ) -> pd.DataFrame | None:
        """Fetch index data with provider fallback (bounded by default)."""
        with self._meta_lock:
            self.last_provider = None

        if self.use_cache:
            cached = _get_cached(ticker, period, "index_cache")
            if cached is not None:
                with self._meta_lock:
                    self.last_provider = "cache"
                return cached

        providers = [
            ("yfinance", lambda: _fetch_yfinance_index(ticker, period)),
            ("jugaad", lambda: _fetch_jugaad_index(ticker, period)),
            ("marketlens", lambda: _fetch_marketlens_index(ticker, period)),
        ]

        for name, fetch_fn in providers:
            try:
                if provider_timeout:
                    df = _call_with_timeout(fetch_fn, provider_timeout)
                    if df is _TIMEOUT:
                        logger.debug("Index provider %s timed out for %s", name, ticker)
                        continue
                else:
                    df = fetch_fn()
                if df is not None and not df.empty:
                    with self._meta_lock:
                        self.last_provider = name
                    if self.use_cache:
                        _set_cached(ticker, period, "index_cache", df)
                    return df
            except Exception as e:
                logger.info("Index provider %s failed for %s: %s", name, ticker, e)
                continue

        return None

    def fetch_fundamentals(
        self, ticker: str, provider_timeout: float | None = None
    ) -> dict | None:
        """
        Fetch fundamentals from ALL free providers, save each raw result,
        then compile the four score fields.

        Each provider's result is cached under its own key the moment it
        arrives (failures are never cached), so a later compile combines
        sources fetched at different times and only re-fetches what is
        still missing. Fields merge first-writer-wins; providers that
        cannot add a missing field are skipped. The compiled dict is
        cached only when all four fields are set — partial compiles are
        recompiled from the raws on the next call.

        Args:
            provider_timeout: When set, each provider call is capped at this
                  many seconds. A provider that exceeds the cap is treated
                  like one that returned no data.
        """
        with self._meta_lock:
            self.last_provider = None

        if self.use_cache:
            cached = _FUNDAMENTALS_CACHE.get(_FUNDAMENTALS_CACHE.make_key(ticker))
            if cached is not None:
                with self._meta_lock:
                    self.last_provider = "cache"
                return cached

        providers = [
            ("finnhub", lambda: _fetch_fundamentals_finnhub(ticker)),
            ("alpha_vantage", lambda: _fetch_fundamentals_alpha_vantage(ticker)),
            ("screener", lambda: _fetch_fundamentals_screener(ticker)),
            ("marketlens", lambda: _fetch_fundamentals_marketlens(ticker)),
            ("yfinance", lambda: _fetch_fundamentals_yfinance(ticker)),
            ("nselib", lambda: _fetch_fundamentals_nselib(ticker)),
        ]

        merged: dict = {}
        contributors: list[str] = []
        for name, fetch_fn in providers:
            needed = {f for f in _FUND_SCORE_FIELDS if f not in merged}
            if not needed:
                break
            if not set(_PROVIDER_FIELDS[name]) & needed:
                continue
            fund = None
            if self.use_cache:
                fund = _FUNDAMENTALS_CACHE.get(
                    _FUNDAMENTALS_CACHE.make_key(ticker, name)
                )
            if fund is None:
                try:
                    if provider_timeout:
                        fund = _call_with_timeout(fetch_fn, provider_timeout)
                        if fund is _TIMEOUT:
                            fund = None
                    else:
                        fund = fetch_fn()
                except Exception as e:
                    logger.info(
                        "Fundamentals provider %s failed for %s: %s", name, ticker, e
                    )
                    fund = None
                if fund and self.use_cache:
                    # Save the raw result — failures stay uncached so they
                    # retry on the next compile.
                    _FUNDAMENTALS_CACHE.set(
                        _FUNDAMENTALS_CACHE.make_key(ticker, name), fund
                    )
            if not fund:
                continue
            contributed = False
            for f in fund:
                if f in needed and merged.get(f) is None and fund[f] is not None:
                    merged[f] = fund[f]
                    contributed = True
            if contributed:
                contributors.append(name)

        if not merged:
            return None
        with self._meta_lock:
            self.last_provider = "+".join(contributors)
        if self.use_cache and len(merged) == len(_FUND_SCORE_FIELDS):
            _FUNDAMENTALS_CACHE.set(_FUNDAMENTALS_CACHE.make_key(ticker), merged)
        return merged
