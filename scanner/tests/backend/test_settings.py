"""Unit tests for scanner settings persistence (negative-cache TTL)."""

import json

import pytest

import scanner.backend.settings_store as store_mod


def test_ttl_default_present():
    """DEFAULT_SETTINGS ships with a sane dead-symbol cache TTL."""
    assert store_mod.DEFAULT_SETTINGS["negative_cache_ttl_hours"] == 24


def test_ttl_persists_through_load_save_round_trip(tmp_path, monkeypatch):
    """A custom TTL survives save_settings -> load_settings."""
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(store_mod, "SETTINGS_FILE", str(settings_file))

    s = store_mod.load_settings()
    assert s["negative_cache_ttl_hours"] == 24  # default when nothing saved

    s["negative_cache_ttl_hours"] = 6
    store_mod.save_settings(s)

    reloaded = store_mod.load_settings()
    assert reloaded["negative_cache_ttl_hours"] == 6
    # Unrelated defaults survive the round trip too
    assert reloaded["timeframe"] == "D"
    assert reloaded["min_score"] == 50.0


def test_stale_age_default_and_round_trip(tmp_path, monkeypatch):
    """stale_member_max_age_days defaults to 45 and persists through save/load."""
    assert store_mod.DEFAULT_SETTINGS["stale_member_max_age_days"] == 45.0
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(store_mod, "SETTINGS_FILE", str(settings_file))

    s = store_mod.load_settings()
    assert s["stale_member_max_age_days"] == 45.0  # default when nothing saved

    s["stale_member_max_age_days"] = 60.0
    store_mod.save_settings(s)

    reloaded = store_mod.load_settings()
    assert reloaded["stale_member_max_age_days"] == 60.0


def test_ttl_loaded_from_corrupt_file_falls_back_to_default(tmp_path, monkeypatch):
    """A corrupt settings file must not break the TTL default."""
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(store_mod, "SETTINGS_FILE", str(settings_file))

    s = store_mod.load_settings()
    assert s["negative_cache_ttl_hours"] == 24


def test_ttl_persists_verbatim_in_db(tmp_path, monkeypatch):
    """The TTL is persisted verbatim in the settings row."""
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(store_mod, "SETTINGS_FILE", str(settings_file))

    s = store_mod.load_settings()
    s["negative_cache_ttl_hours"] = 3
    store_mod.save_settings(s)

    from scanner.shared import db

    row = db.get_conn().execute("SELECT value FROM settings WHERE id = 1").fetchone()
    raw = json.loads(row["value"])
    assert raw["negative_cache_ttl_hours"] == 3


def test_reduce_motion_defaults_off_and_round_trips(tmp_path, monkeypatch):
    """reduce_motion is a bool setting: default False, persists verbatim."""
    assert store_mod.DEFAULT_SETTINGS["reduce_motion"] is False
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(store_mod, "SETTINGS_FILE", str(settings_file))
    s = store_mod.load_settings()
    assert s["reduce_motion"] is False
    s["reduce_motion"] = True
    store_mod.save_settings(s)
    assert store_mod.load_settings()["reduce_motion"] is True


def test_reduce_motion_rejects_non_bool(tmp_path, monkeypatch):
    """A garbage value for reduce_motion falls back to the default."""
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"reduce_motion": "yes"}), encoding="utf-8")
    monkeypatch.setattr(store_mod, "SETTINGS_FILE", str(settings_file))
    assert store_mod.load_settings()["reduce_motion"] is False


# ── Results persistence ──────────────────────────────────────────────────────


def test_results_round_trip(tmp_path, monkeypatch):
    """save_results -> load_results returns the same dict rows."""
    results_file = tmp_path / "last_results.json"
    monkeypatch.setattr(store_mod, "RESULTS_FILE", str(results_file))

    rows = [
        {"ticker": "TCS", "total": 72.5, "ma_bullish": True, "px_tail": [1.0, 2.0]},
        {"ticker": "INFY", "total": 61.0, "ma_bullish": False},
    ]
    store_mod.save_results(rows)
    assert store_mod.load_results() == rows


def test_results_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "RESULTS_FILE", str(tmp_path / "nope.json"))
    assert store_mod.load_results() == []


def test_results_corrupt_file_returns_empty(tmp_path, monkeypatch):
    results_file = tmp_path / "last_results.json"
    results_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(store_mod, "RESULTS_FILE", str(results_file))
    assert store_mod.load_results() == []


def test_results_non_list_returns_empty(tmp_path, monkeypatch):
    results_file = tmp_path / "last_results.json"
    results_file.write_text(json.dumps({"ticker": "TCS"}), encoding="utf-8")
    monkeypatch.setattr(store_mod, "RESULTS_FILE", str(results_file))
    assert store_mod.load_results() == []


def test_results_filters_non_dict_entries(tmp_path, monkeypatch):
    results_file = tmp_path / "last_results.json"
    results_file.write_text(
        json.dumps([{"ticker": "TCS"}, "garbage", 42]), encoding="utf-8"
    )
    monkeypatch.setattr(store_mod, "RESULTS_FILE", str(results_file))
    assert store_mod.load_results() == [{"ticker": "TCS"}]


def test_results_numpy_scalars_serialized(tmp_path, monkeypatch):
    """numpy bool/float must not crash json.dump (coerced via default=)."""
    np = pytest.importorskip("numpy")
    results_file = tmp_path / "last_results.json"
    monkeypatch.setattr(store_mod, "RESULTS_FILE", str(results_file))

    rows = [
        {
            "ticker": "TCS",
            "ma_bullish": np.bool_(True),
            "close": np.float64(123.45),
            "total": np.float64(70.0),
        }
    ]
    store_mod.save_results(rows)
    loaded = store_mod.load_results()
    assert loaded[0]["ma_bullish"] is True
    assert loaded[0]["close"] == 123.45
    assert loaded[0]["total"] == 70.0


# ── Scan history (every save_results run is kept) ─────────────────────────────


def test_scan_history_keeps_previous_scans(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "RESULTS_FILE", str(tmp_path / "out.json"))
    rows1 = [{"ticker": "OLD", "total": 55.0}]
    rows2 = [{"ticker": "NEW", "total": 61.0}, {"ticker": "NEW2", "total": 60.0}]
    store_mod.save_results(rows1)
    store_mod.save_results(rows2)

    assert store_mod.load_results() == rows2  # latest scan wins

    from scanner.shared import db

    scans = db.get_conn().execute("SELECT id FROM scans ORDER BY id").fetchall()
    assert len(scans) == 2
    old = [
        json.loads(r["row"])["ticker"]
        for r in db.get_conn().execute(
            "SELECT row FROM scan_rows WHERE scan_id = ? ORDER BY rank",
            (scans[0]["id"],),
        )
    ]
    assert old == ["OLD"]  # older scan still queryable


def test_save_results_empty_clears_history_view(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "RESULTS_FILE", str(tmp_path / "out.json"))
    store_mod.save_results([{"ticker": "X"}])
    store_mod.save_results([])
    assert store_mod.load_results() == []  # cleared, not resurrected


def test_history_capped_at_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "RESULTS_FILE", str(tmp_path / "out.json"))
    for i in range(store_mod.SCAN_HISTORY_LIMIT + 5):
        store_mod.save_results([{"ticker": f"T{i}"}])

    from scanner.shared import db

    n = db.get_conn().execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    assert n == store_mod.SCAN_HISTORY_LIMIT
    assert (
        store_mod.load_results()[0]["ticker"] == f"T{store_mod.SCAN_HISTORY_LIMIT + 4}"
    )
