"""Single sqlite3 database for all persistent scanner state (stdlib only).

Tables:
    settings     one row (id=1): the settings dict as JSON
    scans        scan history — save_results appends, load_results reads latest
    scan_rows    each scan's result rows (row JSON + rank for ordering)
    kv           generic namespaced JSON cache with optional wall-clock expiry
    price_cache  OHLCV parquet payload stored as a BLOB with an expiry stamp
    ohlcv_archive permanent daily OHLCV (backtest archive, no expiry)

Connections are per-thread (sqlite objects cannot cross threads), WAL so
scan-writes and UI-reads do not block each other, busy_timeout so the two
writers (scan thread, settings save) queue instead of failing.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
# Migration steps for future schema bumps: {new_version: fn(conn)}. On bump,
# add the step here AND update the CREATE script above — fresh dbs are built
# from the script and stamped directly; only pre-existing dbs walk this map.
_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {}
_DEFAULT_DB = str(Path(__file__).resolve().parent.parent / "scanner.db")

_lock = threading.RLock()
_db_path: str | None = None
_conns: dict[int, sqlite3.Connection] = {}
_bootstrapped: set[str] = set()


def db_path() -> str:
    """Current database file (SCANNER_DB env override, else scanner/scanner.db)."""
    global _db_path
    with _lock:
        if _db_path is None:
            _db_path = os.environ.get("SCANNER_DB") or _DEFAULT_DB
        return _db_path


def set_db_path(path: str | None) -> None:
    """Redirect the store (tests): resets path, schema cache, open connections."""
    global _db_path
    with _lock:
        _db_path = str(path) if path else None
        _bootstrapped.clear()
        for conn in _conns.values():
            try:
                conn.close()
            except Exception:
                logger.debug("Closing stale db connection failed", exc_info=True)
        _conns.clear()


def get_conn() -> sqlite3.Connection:
    """Thread-local connection, schema bootstrapped once per path."""
    ident = threading.get_ident()
    with _lock:
        path = db_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = _conns.get(ident)
        if conn is None:
            conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            _conns[ident] = conn
        if path not in _bootstrapped:
            _create_schema(conn)
            _bootstrapped.add(path)
        return conn


def _close_all_connections() -> None:
    """Release every cached connection (process exit — see atexit below)."""
    with _lock:
        for conn in _conns.values():
            try:
                conn.close()
            except Exception:
                logger.debug("Closing db connection failed", exc_info=True)
        _conns.clear()


# Connections are per-thread and nothing else ever closes them; without this
# a weeks-long session accumulates idle sqlite handles.
atexit.register(_close_all_connections)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS settings (
            id    INTEGER PRIMARY KEY CHECK (id = 1),
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scans (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scan_rows (
            scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
            rank    INTEGER,
            row     TEXT NOT NULL,
            PRIMARY KEY (scan_id, rank)
        );
        CREATE TABLE IF NOT EXISTS kv (
            namespace TEXT NOT NULL,
            key       TEXT NOT NULL,
            value     TEXT NOT NULL,
            expires   REAL,
            PRIMARY KEY (namespace, key)
        );
        CREATE INDEX IF NOT EXISTS idx_kv_expires ON kv(expires);
        CREATE TABLE IF NOT EXISTS price_cache (
            cache_key TEXT PRIMARY KEY,
            payload   BLOB NOT NULL,
            expires   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ohlcv_archive (
            symbol TEXT    NOT NULL,
            date   TEXT    NOT NULL,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        );
        """
    )
    _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Stamp a fresh db; walk ``_MIGRATIONS`` for dbs behind SCHEMA_VERSION.

    Each step runs in one transaction together with its version stamp, so a
    failing migration leaves the db at the last fully-applied version.
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > SCHEMA_VERSION:
        logger.warning(
            "db schema v%d is newer than this app (v%d) — left untouched",
            current,
            SCHEMA_VERSION,
        )
        return
    if current == 0:
        legacy = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='scans'"
        ).fetchone()
        if legacy is None:
            # Fresh db: the CREATE script above already built it at current.
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return
        # Pre-stamp db: the tables are exactly v1, so never treat it as fresh.
        current = 1
        conn.execute(f"PRAGMA user_version = {current}")
    for version in range(current + 1, SCHEMA_VERSION + 1):
        step = _MIGRATIONS.get(version)
        if step is None:
            # Never stamp past an unapplied step — retry (and warn) next open.
            logger.warning("missing migration for schema v%d", version)
            break
        with transaction(conn):
            step(conn)
            conn.execute(f"PRAGMA user_version = {version}")


@contextmanager
def transaction(conn: sqlite3.Connection | None = None) -> Iterator[sqlite3.Connection]:
    """Explicit write transaction (connections run in autocommit mode)."""
    c = conn or get_conn()
    c.execute("BEGIN IMMEDIATE")
    try:
        yield c
    except BaseException:
        c.execute("ROLLBACK")
        raise
    else:
        c.execute("COMMIT")


# ── kv helpers (namespaced JSON cache with optional expiry) ──────────────────


def kv_get(namespace: str, key: str) -> str | None:
    row = (
        get_conn()
        .execute(
            "SELECT value FROM kv WHERE namespace = ? AND key = ?"
            " AND (expires IS NULL OR expires > ?)",
            (namespace, key, time.time()),
        )
        .fetchone()
    )
    return row["value"] if row else None


def kv_put(namespace: str, key: str, value: str, expires: float | None = None) -> None:
    get_conn().execute(
        "INSERT INTO kv (namespace, key, value, expires) VALUES (?, ?, ?, ?)"
        " ON CONFLICT (namespace, key)"
        " DO UPDATE SET value = excluded.value, expires = excluded.expires",
        (namespace, key, value, expires),
    )


def kv_get_json(namespace: str, key: str):
    raw = kv_get(namespace, key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def kv_put_json(namespace: str, key: str, value, expires: float | None = None) -> None:
    """Upsert ``value`` as JSON; non-serializable values are a silent no-op."""
    try:
        payload = json.dumps(value)
    except (TypeError, ValueError):
        return
    kv_put(namespace, key, payload, expires)


def kv_items(namespace: str) -> dict[str, str]:
    """Live ``{key: value}`` rows for a namespace (expired rows skipped)."""
    rows = (
        get_conn()
        .execute(
            "SELECT key, value FROM kv WHERE namespace = ?"
            " AND (expires IS NULL OR expires > ?)",
            (namespace, time.time()),
        )
        .fetchall()
    )
    return {r["key"]: r["value"] for r in rows}


def kv_count(namespace: str) -> int:
    row = (
        get_conn()
        .execute(
            "SELECT COUNT(*) AS n FROM kv WHERE namespace = ?"
            " AND (expires IS NULL OR expires > ?)",
            (namespace, time.time()),
        )
        .fetchone()
    )
    return int(row["n"]) if row else 0


def kv_delete(namespace: str, key: str) -> None:
    get_conn().execute(
        "DELETE FROM kv WHERE namespace = ? AND key = ?", (namespace, key)
    )


def kv_clear(namespace: str) -> int:
    cur = get_conn().execute("DELETE FROM kv WHERE namespace = ?", (namespace,))
    return cur.rowcount


def kv_prune() -> int:
    """Delete every expired kv row across namespaces. Returns rows removed."""
    cur = get_conn().execute(
        "DELETE FROM kv WHERE expires IS NOT NULL AND expires <= ?", (time.time(),)
    )
    return cur.rowcount


def kv_import_src(namespace: str, path: str | os.PathLike):
    """Legacy JSON to fold into an empty namespace once; else None.

    None when the namespace already holds rows, the file is missing or
    unreadable, or the payload is not a JSON object.
    """
    if kv_count(namespace) or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ── ohlcv_archive (permanent daily bars for backtests) ───────────────────────


def archive_put(symbol: str, df) -> int:
    """Upsert a daily OHLCV frame (DatetimeIndex, OHLCV columns) for a symbol.

    Dates are stored as YYYY-MM-DD; conflicts replace the row (refresh of a
    later fetch wins). pandas is imported lazily so this module stays
    stdlib-importable. Returns rows written.
    """
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return 0
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    if d.index.tz is not None:
        d.index = d.index.tz_convert("Asia/Kolkata").tz_localize(None)
    has_vol = "volume" in d.columns
    rows = [
        (
            symbol,
            dt.strftime("%Y-%m-%d"),
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            float(r["volume"]) if has_vol and not pd.isna(r["volume"]) else 0.0,
        )
        for dt, r in d.iterrows()
        if not pd.isna(r["close"])
    ]
    if not rows:
        return 0
    with transaction() as conn:
        conn.executemany(
            "INSERT INTO ohlcv_archive (symbol, date, open, high, low, close, volume)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (symbol, date) DO UPDATE SET"
            " open=excluded.open, high=excluded.high, low=excluded.low,"
            " close=excluded.close, volume=excluded.volume",
            rows,
        )
    return len(rows)


def archive_range(symbol: str) -> tuple[str, str] | None:
    """(min_date, max_date) archived for a symbol as YYYY-MM-DD, else None."""
    row = (
        get_conn()
        .execute(
            "SELECT MIN(date) AS lo, MAX(date) AS hi FROM ohlcv_archive WHERE symbol = ?",
            (symbol,),
        )
        .fetchone()
    )
    return (row["lo"], row["hi"]) if row and row["lo"] else None


def archive_get(symbol: str, start: str | None = None, end: str | None = None):
    """Daily bars for a symbol as an OHLCV DataFrame (DatetimeIndex); empty frame if none."""
    import pandas as pd

    sql = (
        "SELECT date, open, high, low, close, volume FROM ohlcv_archive"
        " WHERE symbol = ?"
    )
    args: list = [symbol]
    if start:
        sql += " AND date >= ?"
        args.append(start)
    if end:
        sql += " AND date <= ?"
        args.append(end)
    sql += " ORDER BY date"
    rows = get_conn().execute(sql, args).fetchall()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        [tuple(r) for r in rows],
        columns=["date", "open", "high", "low", "close", "volume"],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")
