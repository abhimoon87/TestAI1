"""Unit tests for scanner.shared.db — schema, kv helpers, transactions, threads."""

import threading
import time

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
