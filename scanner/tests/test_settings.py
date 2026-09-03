"""Unit tests for scanner.app settings persistence (negative-cache TTL)."""

import json

import scanner.app as app_mod


def test_ttl_default_present():
    """DEFAULT_SETTINGS ships with a sane dead-symbol cache TTL."""
    assert app_mod.DEFAULT_SETTINGS["negative_cache_ttl_hours"] == 24


def test_ttl_persists_through_load_save_round_trip(tmp_path, monkeypatch):
    """A custom TTL survives save_settings -> load_settings."""
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(app_mod, "SETTINGS_FILE", str(settings_file))

    s = app_mod.load_settings()
    assert s["negative_cache_ttl_hours"] == 24  # default when nothing saved

    s["negative_cache_ttl_hours"] = 6
    app_mod.save_settings(s)

    reloaded = app_mod.load_settings()
    assert reloaded["negative_cache_ttl_hours"] == 6
    # Unrelated defaults survive the round trip too
    assert reloaded["timeframe"] == "D"
    assert reloaded["min_score"] == 50.0


def test_ttl_loaded_from_corrupt_file_falls_back_to_default(tmp_path, monkeypatch):
    """A corrupt settings file must not break the TTL default."""
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(app_mod, "SETTINGS_FILE", str(settings_file))

    s = app_mod.load_settings()
    assert s["negative_cache_ttl_hours"] == 24


def test_ttl_survives_raw_file_write(tmp_path, monkeypatch):
    """The TTL is persisted verbatim in the JSON on disk."""
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(app_mod, "SETTINGS_FILE", str(settings_file))

    s = app_mod.load_settings()
    s["negative_cache_ttl_hours"] = 3
    app_mod.save_settings(s)

    with open(settings_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["negative_cache_ttl_hours"] == 3