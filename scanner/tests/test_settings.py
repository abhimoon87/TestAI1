"""Unit tests for scanner settings persistence (negative-cache TTL)."""

import json

import scanner.settings_store as store_mod


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


def test_ttl_survives_raw_file_write(tmp_path, monkeypatch):
    """The TTL is persisted verbatim in the JSON on disk."""
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(store_mod, "SETTINGS_FILE", str(settings_file))

    s = store_mod.load_settings()
    s["negative_cache_ttl_hours"] = 3
    store_mod.save_settings(s)

    with open(settings_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["negative_cache_ttl_hours"] == 3