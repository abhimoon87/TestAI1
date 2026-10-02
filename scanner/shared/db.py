"""Single sqlite3 database for all persistent scanner state (stdlib only).

Tables:
    settings     one row (id=1): the settings dict as JSON
    scans        scan history — save_results appends, load_results reads latest
    scan_rows    each scan's result rows (row JSON + rank for ordering)
    kv           generic namespaced JSON cache with optional wall-clock expiry
    price_cache  OHLCV parquet payload stored as a BLOB with an expiry stamp

Connections are per-thread (sqlite objects cannot cross threads), WAL so
scan-writes and UI-reads do not block each other, busy_timeout so the two
writers (scan thread, settings save) queue instead of failing.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
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
        """
    )
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


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
