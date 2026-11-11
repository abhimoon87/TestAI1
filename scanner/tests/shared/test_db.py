"""Unit tests for scanner.shared.db — schema, kv helpers, transactions, threads."""

import sqlite3
import threading
import time

import pytest

from scanner.shared import db


def test_schema_version_and_tables():
    conn = db.get_conn()
    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"settings", "scans", "scan_rows", "kv", "price_cache"} <= tables
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION


def test_set_db_path_switches_files(tmp_path):
    db.kv_put("t", "k", "v1")
    prev = db.db_path()
    alt = tmp_path / "alt.db"
    db.set_db_path(str(alt))
    try:
        assert db.kv_get("t", "k") is None  # fresh file, no carry-over
        db.kv_put("t", "k", "v2")
        assert db.kv_get("t", "k") == "v2"
    finally:
        db.set_db_path(prev)
    assert db.kv_get("t", "k") == "v1"  # original file still holds v1


def test_kv_expiry_and_prune():
    db.kv_put("t", "alive", "1", expires=time.time() + 3600)
    db.kv_put("t", "dead", "0", expires=time.time() - 1)
    assert db.kv_get("t", "alive") == "1"
    assert db.kv_get("t", "dead") is None
    assert db.kv_items("t") == {"alive": "1"}
    assert db.kv_prune() == 1
    n = (
        db.get_conn()
        .execute("SELECT COUNT(*) FROM kv WHERE key = 'dead'")
        .fetchone()[0]
    )
    assert n == 0


def test_kv_count_and_clear():
    db.kv_put("ns", "a", "1")
    db.kv_put("ns", "b", "2")
    db.kv_put("other", "c", "3")
    assert db.kv_count("ns") == 2
    assert db.kv_clear("ns") == 2
    assert db.kv_count("ns") == 0
    assert db.kv_get("other", "c") == "3"  # sibling namespace untouched


def test_json_helpers_and_bad_payload():
    db.kv_put_json("j", "ok", {"x": [1, 2]})
    assert db.kv_get_json("j", "ok") == {"x": [1, 2]}
    db.kv_put_json("j", "bad", object())  # not serializable → silent no-op
    assert db.kv_get_json("j", "bad") is None
    db.kv_put("j", "corrupt", "{not json")
    assert db.kv_get_json("j", "corrupt") is None


def test_transaction_rolls_back_on_error():
    try:
        with db.transaction():
            db.kv_put("tx", "k", "v")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert db.kv_get("tx", "k") is None


def test_transaction_commits():
    with db.transaction():
        db.kv_put("tx2", "k", "v")
    assert db.kv_get("tx2", "k") == "v"


def test_threads_use_own_connection_shared_file():
    db.kv_put("th", "k", "main")
    out = []

    def worker():
        out.append(db.kv_get("th", "k"))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert out == ["main"]


def test_close_all_connections_clears_cache_and_conn_still_usable():
    conn = db.get_conn()
    assert threading.get_ident() in db._conns
    db._close_all_connections()
    assert db._conns == {}
    # A later get_conn transparently opens a fresh, working connection.
    fresh = db.get_conn()
    assert fresh is not conn
    assert fresh.execute("SELECT 1").fetchone()[0] == 1


# ── Schema migration hook (PRAGMA user_version + _MIGRATIONS) ───────────────


def _db_at_version(path, version):
    """Standalone sqlite file stamped at `version` (an 'old' app's db)."""
    conn = sqlite3.connect(str(path))
    conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
    conn.close()


def _user_version(path):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def test_migration_walks_in_order_and_stamps(monkeypatch, tmp_path):
    path = tmp_path / "old.db"
    _db_at_version(path, 1)
    calls = []
    monkeypatch.setattr(db, "SCHEMA_VERSION", 3)
    monkeypatch.setattr(
        db,
        "_MIGRATIONS",
        {2: lambda c: calls.append(2), 3: lambda c: calls.append(3)},
    )
    db.set_db_path(str(path))

    conn = db.get_conn()

    assert calls == [2, 3]  # oldest first
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 3


def test_missing_migration_stops_before_stamping(monkeypatch, tmp_path):
    path = tmp_path / "old.db"
    _db_at_version(path, 1)
    monkeypatch.setattr(db, "SCHEMA_VERSION", 3)
    monkeypatch.setattr(db, "_MIGRATIONS", {3: lambda c: None})  # v2 missing
    db.set_db_path(str(path))

    db.get_conn()

    assert _user_version(path) == 1  # stuck at last applied version


def test_failing_migration_rolls_back_and_keeps_version(monkeypatch, tmp_path):
    path = tmp_path / "old.db"
    _db_at_version(path, 1)

    def boom(c):
        raise RuntimeError("half-applied")

    monkeypatch.setattr(db, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(db, "_MIGRATIONS", {2: boom})
    db.set_db_path(str(path))

    with pytest.raises(RuntimeError):
        db.get_conn()
    assert _user_version(path) == 1  # step + stamp share one transaction


def test_newer_db_is_left_untouched(tmp_path):
    path = tmp_path / "future.db"
    _db_at_version(path, 99)
    db.set_db_path(str(path))

    db.get_conn()

    assert _user_version(path) == 99  # never downgrade a newer schema


def test_pre_stamp_db_is_not_treated_as_fresh(monkeypatch, tmp_path):
    """A db created before user_version existed reads 0 but is really v1."""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE scans (id INTEGER PRIMARY KEY, created_at TEXT)")
    conn.commit()
    conn.close()  # user_version stays 0
    calls = []
    monkeypatch.setattr(db, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(db, "_MIGRATIONS", {2: lambda c: calls.append(2)})
    db.set_db_path(str(path))

    db.get_conn()

    assert calls == [2]  # migration ran instead of being skipped as fresh
    assert _user_version(path) == 2


def test_fresh_db_stamped_without_running_migrations(monkeypatch, tmp_path):
    def boom(c):
        raise AssertionError("fresh db must not replay migrations")

    monkeypatch.setattr(db, "SCHEMA_VERSION", 1)
    monkeypatch.setattr(db, "_MIGRATIONS", {1: boom})
    db.set_db_path(str(tmp_path / "fresh.db"))

    conn = db.get_conn()

    # The CREATE script builds fresh dbs at current — no steps to replay.
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
