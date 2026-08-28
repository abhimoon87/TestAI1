"""Tests for settings_store — DEFAULT_SETTINGS, load_settings, save_settings."""

from __future__ import annotations

import json
import os
import tempfile

from scanner.settings_store import DEFAULT_SETTINGS, load_settings, save_settings


class TestDefaultSettings:
    def test_has_all_ma_keys(self):
        for key in ("fast_ma_type", "fast_ma_len", "slow_ma_type", "slow_ma_len"):
            assert key in DEFAULT_SETTINGS

    def test_has_technical_keys(self):
        for key in ("rsi_len", "rs_length", "vol_ma_len", "atr_len"):
            assert key in DEFAULT_SETTINGS

    def test_has_scanner_keys(self):
        for key in ("min_score", "data_period", "timeframe", "crossover_lookback"):
            assert key in DEFAULT_SETTINGS

    def test_crossover_lookback_is_20(self):
        assert DEFAULT_SETTINGS["crossover_lookback"] == 20

    def test_min_score_is_50(self):
        assert DEFAULT_SETTINGS["min_score"] == 50.0


class TestLoadSettings:
    def test_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scanner.settings_store.SETTINGS_FILE",
                            str(tmp_path / "nonexistent.json"))
        result = load_settings()
        assert result == DEFAULT_SETTINGS

    def test_loads_saved_settings(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "settings.json"
        saved = {**DEFAULT_SETTINGS, "fast_ma_len": 30, "min_score": 60.0}
        settings_file.write_text(json.dumps(saved))
        monkeypatch.setattr("scanner.settings_store.SETTINGS_FILE",
                            str(settings_file))
        result = load_settings()
        assert result["fast_ma_len"] == 30
        assert result["min_score"] == 60.0

    def test_corrupt_file_returns_defaults(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text("NOT JSON {{{")
        monkeypatch.setattr("scanner.settings_store.SETTINGS_FILE",
                            str(settings_file))
        result = load_settings()
        assert result == DEFAULT_SETTINGS

    def test_partial_saved_settings_merge(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "settings.json"
        saved = {"fast_ma_len": 30}  # only one key
        settings_file.write_text(json.dumps(saved))
        monkeypatch.setattr("scanner.settings_store.SETTINGS_FILE",
                            str(settings_file))
        result = load_settings()
        assert result["fast_ma_len"] == 30
        assert result["slow_ma_len"] == DEFAULT_SETTINGS["slow_ma_len"]


class TestSaveSettings:
    def test_creates_file(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("scanner.settings_store.SETTINGS_FILE",
                            str(settings_file))
        save_settings({"foo": "bar"})
        assert settings_file.exists()
        data = json.loads(settings_file.read_text())
        assert data["foo"] == "bar"

    def test_roundtrip(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("scanner.settings_store.SETTINGS_FILE",
                            str(settings_file))
        original = {**DEFAULT_SETTINGS, "fast_ma_len": 99}
        save_settings(original)
        loaded = load_settings()
        assert loaded["fast_ma_len"] == 99
