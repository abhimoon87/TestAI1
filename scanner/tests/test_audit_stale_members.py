"""Unit tests for scanner.audit_stale_members — all fetches mocked offline."""

import pandas as pd

import scanner.audit_stale_members as audit


def _frame_ending(days_ago, n=20):
    """OHLCV frame whose last bar is ``days_ago`` calendar days before today."""
    end = pd.Timestamp.now().normalize() - pd.Timedelta(days=days_ago)
    dates = pd.bdate_range(end - pd.Timedelta(days=n), periods=n)
    return pd.DataFrame({"close": [1.0] * n, "volume": [1] * n}, index=dates)


def _patch_fetch(monkeypatch, batch):
    monkeypatch.setattr(audit, "fetch_batch_yfinance",
                        lambda *a, **k: batch)
    # Keep every offline test hermetic: no per-ticker probe, no mainboard
    # lookup, no rename-verification download.
    monkeypatch.setattr(audit, "fetch_stock_data", lambda *a, **k: None)
    monkeypatch.setattr(audit, "_nse_mainboard", list)


def test_splits_stale_into_annotated_and_not(monkeypatch):
    batch = {
        "RELIANCE": _frame_ending(2),    # fresh
        "GSPL": _frame_ending(120),      # stale, already annotated
        "STALECO": _frame_ending(300),   # stale, NOT annotated
    }
    _patch_fetch(monkeypatch, batch)

    res = audit.audit_stale_members(list(batch), period="3y", max_age_days=45)

    assert res["fetched"] == 3
    assert [t for t, _ in res["unannotated_stale"]] == ["STALECO"]
    assert [t for t, _ in res["annotated_stale"]] == ["GSPL"]
    assert res["annotated_fresh"] == []
    assert res["missing"] == []


def test_annotated_but_fresh_is_flagged_for_removal(monkeypatch):
    monkeypatch.setattr(
        "scanner.universes.SUSPENDED_OR_DELISTED",
        {"GSPL": "suspended", "RESUMEDCO": "delisted"},
    )
    batch = {
        "GSPL": _frame_ending(120),    # still stale -> stays annotated
        "RESUMEDCO": _frame_ending(3),  # trading again -> remove candidate
    }
    _patch_fetch(monkeypatch, batch)

    res = audit.audit_stale_members(list(batch), period="3y", max_age_days=45)

    assert [t for t, _ in res["annotated_stale"]] == ["GSPL"]
    assert res["annotated_fresh"] == ["RESUMEDCO"]


def test_cutoff_respected(monkeypatch):
    batch = {"BORDER": _frame_ending(30)}
    _patch_fetch(monkeypatch, batch)

    assert audit.audit_stale_members(["BORDER"], max_age_days=45)["unannotated_stale"] == []
    res = audit.audit_stale_members(["BORDER"], max_age_days=20)
    assert [t for t, _ in res["unannotated_stale"]] == ["BORDER"]


def test_missing_and_membership(monkeypatch):
    batch = {"RELIANCE": _frame_ending(1), "GSPL": _frame_ending(120)}
    _patch_fetch(monkeypatch, batch)

    res = audit.audit_stale_members(["RELIANCE", "GSPL", "GONE"], period="3y")

    assert res["missing"] == ["GONE"]
    assert res["neg_cache_skipped"] == []
    assert "NIFTY 50" in res["membership"]["RELIANCE"]
    assert "FnO STOCKS" in res["membership"]["GSPL"]


def test_missing_from_dead_symbol_cache_is_flagged(monkeypatch):
    import time

    import scanner.data_fetcher as df
    monkeypatch.setattr(df, "_negative_cache", {"GONE": time.time()})
    batch = {"RELIANCE": _frame_ending(1)}
    _patch_fetch(monkeypatch, batch)

    res = audit.audit_stale_members(["RELIANCE", "GONE"], period="3y")

    assert res["missing"] == ["GONE"]
    assert res["neg_cache_skipped"] == ["GONE"]
    report = audit.format_report(res)
    assert "dead-symbol cache" in report
    assert "clearing it" in report


def test_format_report_sections_and_paste_ready(monkeypatch):
    batch = {"STALECO": _frame_ending(300), "GSPL": _frame_ending(120)}
    _patch_fetch(monkeypatch, batch)
    res = audit.audit_stale_members(list(batch), period="3y")

    report = audit.format_report(res)
    assert "STALE — NOT ANNOTATED" in report
    assert "STALECO" in report
    assert "paste-ready" in report
    assert '"STALECO": "no trades since' in report
    assert "STALE — ALREADY ANNOTATED" in report
    assert "GSPL" in report
    assert "ANNOTATED BUT FRESH" in report
    assert "NO DATA IN WINDOW" in report


def test_universe_symbols_are_real(monkeypatch):
    """The corrected NSE symbols are present; the bogus ones are gone."""
    from scanner.universes import (
        BSE_SMALLCAP,
        NIFTY_MIDCAP_100,
        NIFTY_NEXT_50,
        NIFTY_SMALLCAP_100,
    )

    for lst in (NIFTY_NEXT_50, NIFTY_SMALLCAP_100, BSE_SMALLCAP, NIFTY_MIDCAP_100):
        assert "ASTER" not in lst and "BIRLASOFT" not in lst and "AVALONLABS" not in lst
    assert "AVALON" in NIFTY_NEXT_50
    assert "ASTERDM" in NIFTY_SMALLCAP_100
    assert "ASTERDM" in BSE_SMALLCAP
    assert "BSOFT" in NIFTY_SMALLCAP_100
    assert "BSOFT" in BSE_SMALLCAP


def test_main_unknown_universe_exits(monkeypatch, capsys):
    import sys
    from unittest.mock import patch as _patch

    with _patch.object(sys, "argv", ["audit", "--universe", "NOT_A_UNIVERSE"]):
        with _patch.object(sys, "exit") as mock_exit:
            audit.main()
    assert mock_exit.called
    assert "unknown static universe" in capsys.readouterr().err


# ══════════════════════════════════════════════════════════════════════════════
# Symbol validation — rename suggestion ranking
# ══════════════════════════════════════════════════════════════════════════════


def test_candidate_budget_round_robin_keeps_every_ticker():
    """The 60-candidate cap is fair — no ticker loses every candidate."""
    ranked = {f"BAD{i:02d}": [f"C{i:02d}{j}" for j in range(10)] for i in range(20)}

    picked = audit._candidate_budget(ranked, budget=60)

    assert len(picked) == 60
    assert {f"C{i:02d}0" for i in range(20)} <= set(picked)  # top pick of each
    assert len({c[:3] for c in picked}) == 20  # all 20 tickers represented


def test_candidate_budget_respects_smaller_budget_and_dedupes():
    """Shared candidates are fetched once; budget is honored when small."""
    ranked = {"BAD1": ["C1A", "SHARED"], "BAD2": ["C2A", "SHARED"]}

    picked = audit._candidate_budget(ranked, budget=3)

    assert picked == ["C1A", "C2A", "SHARED"]  # round-robin, no duplicate fetch


def test_rank_rename_candidates_prefix_rules():
    """Bad ticker that prefixes the real symbol, and one that over-extends it."""
    mainboard = ["ASTERDM", "ASTERMINDS", "AVALON", "RELIANCE", "TCS"]
    ranked = audit._rank_rename_candidates(["ASTER", "AVALONLABS"], mainboard)
    assert ranked["ASTER"][0] == "ASTERDM"
    assert ranked["AVALONLABS"][0] == "AVALON"


def test_rank_rename_candidates_fuzzy_match():
    """BIRLASOFT -> BSOFT has no prefix relation — only fuzzy similarity."""
    ranked = audit._rank_rename_candidates(
        ["BIRLASOFT"], ["RELIANCE", "BSOFT", "TCS"]
    )
    assert ranked["BIRLASOFT"][0] == "BSOFT"


def test_rank_rename_candidates_valid_symbol_or_no_match():
    """A real NSE symbol with no data is not a rename; empty input is safe."""
    assert audit._rank_rename_candidates(["GSPL"], ["GSPL", "TCS"])["GSPL"] == []
    assert audit._rank_rename_candidates(["X"], [])["X"] == []


def test_suggest_renames_verifies_candidate_has_data(monkeypatch):
    """Only candidates that actually return data are suggested."""
    def fake_fetch(tickers, **k):
        return {"ASTERDM": _frame_ending(2)}  # ASTERMINDS has no data
    monkeypatch.setattr(audit, "fetch_batch_yfinance", fake_fetch)
    monkeypatch.setattr(audit, "fetch_stock_data", lambda *a, **k: None)
    monkeypatch.setattr(audit, "_nse_mainboard",
                        lambda: ["ASTERDM", "ASTERMINDS"])

    res = audit.audit_stale_members(["ASTER"], period="3y")

    assert res["missing"] == ["ASTER"]
    assert res["rename_suggestions"] == {"ASTER": "ASTERDM"}


def test_suggest_renames_skips_when_mainboard_unavailable(monkeypatch):
    """No mainboard list -> no guesses, no extra download."""
    calls = []
    monkeypatch.setattr(audit, "fetch_batch_yfinance",
                        lambda t, **k: calls.append(list(t)) or {})
    monkeypatch.setattr(audit, "fetch_stock_data", lambda *a, **k: None)
    monkeypatch.setattr(audit, "_nse_mainboard", list)

    res = audit.audit_stale_members(["GONE"], period="3y")

    assert res["missing"] == ["GONE"]
    assert res["rename_suggestions"] == {}
    assert calls == [["GONE"]]  # only the main fetch, no verification call


def test_live_probe_recovers_symbol_marked_dead(monkeypatch):
    """A name the batch skipped (negative cache) is re-fetched per-ticker."""
    import time

    import scanner.data_fetcher as df
    monkeypatch.setattr(df, "_negative_cache", {"GONE": time.time()})
    monkeypatch.setattr(audit, "fetch_batch_yfinance",
                        lambda *a, **k: {"RELIANCE": _frame_ending(1)})
    monkeypatch.setattr(audit, "fetch_stock_data",
                        lambda *a, **k: _frame_ending(1, n=60))
    monkeypatch.setattr(audit, "_nse_mainboard", list)

    res = audit.audit_stale_members(["RELIANCE", "GONE"], period="3y")

    assert res["fetched"] == 2
    assert res["missing"] == []
    assert res["neg_cache_skipped"] == []


def test_live_probe_parallel_recovers_all_and_reports_failures(monkeypatch):
    """Pooled probes recover every name with data; failures stay missing."""
    import scanner.data_fetcher as df
    monkeypatch.setattr(df, "_negative_cache", {})
    monkeypatch.setattr(audit, "fetch_batch_yfinance",
                        lambda *a, **k: {"RELIANCE": _frame_ending(1)})
    monkeypatch.setattr(
        audit, "fetch_stock_data",
        lambda t, *a, **k: _frame_ending(1, n=60) if t in {"G1", "G2"} else None,
    )
    monkeypatch.setattr(audit, "_nse_mainboard", list)

    res = audit.audit_stale_members(
        ["RELIANCE", "G1", "G2", "DEAD1", "DEAD2"], period="3y")

    assert res["fetched"] == 3          # RELIANCE + G1 + G2 via the pool
    assert res["missing"] == ["DEAD1", "DEAD2"]
    assert res["rename_suggestions"] == {}


def test_live_probe_failed_names_still_reported_missing(monkeypatch):
    """Probe failure keeps the name in section 4 (with the dead-cache tag)."""
    monkeypatch.setattr(audit, "fetch_batch_yfinance",
                        lambda *a, **k: {"RELIANCE": _frame_ending(1)})
    monkeypatch.setattr(audit, "fetch_stock_data", lambda *a, **k: None)
    monkeypatch.setattr(audit, "_nse_mainboard", list)

    res = audit.audit_stale_members(["RELIANCE", "GONE"], period="3y")

    assert res["missing"] == ["GONE"]
    assert res["rename_suggestions"] == {}


# ══════════════════════════════════════════════════════════════════════════════
# All-universes mode + report formatting
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# --fix: apply verified renames + annotation updates to universes.py
# ══════════════════════════════════════════════════════════════════════════════


def _fake_universes_file(path):
    """A miniature universes.py with the same structural features."""
    path.write_text(
        'SUSPENDED_OR_DELISTED = {\n'
        '    "GSPL": "suspended — no trades since 2026-05-11",\n'
        '    "TATAMETALI": "delisted — merged into Tata Steel (last trade 2024-02-05)",\n'
        '}\n'
        '\n'
        'NIFTY_SMALLCAP_100 = [\n'
        '    "AFFLE", "ASTER",\n'
        '    "BSOFT",\n'
        ']\n'
        '\n'
        'SECTOR_MAP = {\n'
        '    "ASTER": "Chemicals",\n'
        '}\n',
        encoding="utf-8",
    )


_FIX_RES = {
    "rename_suggestions": {"ASTER": "ASTERDM"},
    "unannotated_stale": [("STALECO", "2025-11-10")],
    "annotated_fresh": ["RESUMEDCO"],
}


def test_apply_fixes_inserts_keep_dict_keys_sorted(tmp_path):
    """New SUSPENDED_OR_DELISTED entries merge in alphabetical order."""
    p = tmp_path / "universes.py"
    _fake_universes_file(p)
    res = {"unannotated_stale": [
        ("ZZCO", "2026-01-02"), ("AAACO", "2026-01-01"), ("MIDCO", "2026-01-03"),
    ]}

    audit.apply_fixes(res, path=str(p))

    out = p.read_text(encoding="utf-8")
    block = out.split("SUSPENDED_OR_DELISTED = {", 1)[1].split("}", 1)[0]
    keys = __import__("re").findall(r'"([A-Za-z0-9._-]+)":', block)
    assert keys == sorted(keys)  # dict keys stay alphabetical
    assert keys == ["AAACO", "GSPL", "MIDCO", "TATAMETALI", "ZZCO"]


def test_apply_fixes_skips_add_when_key_already_present(tmp_path):
    """Adding a name already annotated is a no-op, not a duplicate key."""
    p = tmp_path / "universes.py"
    _fake_universes_file(p)
    res = {"unannotated_stale": [("GSPL", "2026-05-11")]}

    summary = audit.apply_fixes(res, path=str(p))

    assert summary["annotated_added"] == []
    assert summary["changed"] is False
    assert p.read_text(encoding="utf-8").count('"GSPL":') == 1


def test_apply_fixes_dedupes_sector_map_after_rename(tmp_path):
    """Renaming onto a known sector ticker leaves ONE SECTOR_MAP entry."""
    p = tmp_path / "universes.py"
    p.write_text(
        "SUSPENDED_OR_DELISTED = {\n}\n"
        "NIFTY_SMALLCAP_100 = [\n    \"ASTER\",\n]\n"
        "SECTOR_MAP = {\n"
        '    "ASTERDM": "Health", "ASTER": "Chemicals",\n'
        '    "RELIANCE": "OilGas",\n'
        "}\n",
        encoding="utf-8",
    )

    summary = audit.apply_fixes(
        {"rename_suggestions": {"ASTER": "ASTERDM"}}, path=str(p))

    out = p.read_text(encoding="utf-8")
    block = out.split("SECTOR_MAP = {", 1)[1].split("}", 1)[0]
    assert block.count('"ASTERDM":') == 1
    assert '"Chemicals"' in block and '"Health"' not in block  # last kept
    assert summary["sector_map_deduped"] == ["ASTERDM"]
    assert summary["renamed"] == [("ASTER", "ASTERDM", 2)]
    text = audit.format_fix_summary(summary)
    assert "Sector map: ASTERDM kept its existing entry" in text


def test_apply_fixes_sector_dedupe_last_entry_without_trailing_comma(tmp_path):
    """Comma handling when the later duplicate closes the dict block."""
    p = tmp_path / "universes.py"
    p.write_text(
        "SUSPENDED_OR_DELISTED = {\n}\n"
        "SECTOR_MAP = {\n"
        '    "ASTER": "Chemicals",\n'
        '    "ASTERDM": "IT"\n'
        "}\n",
        encoding="utf-8",
    )

    summary = audit.apply_fixes(
        {"rename_suggestions": {"ASTER": "ASTERDM"}}, path=str(p))

    out = p.read_text(encoding="utf-8")
    block = out.split("SECTOR_MAP = {", 1)[1].split("}", 1)[0]
    assert block.count('"ASTERDM":') == 1
    assert '"IT"' in block and '"Chemicals"' not in block
    assert summary["sector_map_deduped"] == ["ASTERDM"]


def test_apply_fixes_no_sector_dup_no_dedupe_change(tmp_path):
    """Distinct SECTOR_MAP keys are untouched (nothing to dedupe)."""
    p = tmp_path / "universes.py"
    _fake_universes_file(p)  # SECTOR_MAP has only "ASTER" -> renamed, no dup

    summary = audit.apply_fixes(
        {"rename_suggestions": {"ASTER": "ASTERDM"}}, path=str(p))

    assert "sector_map_deduped" not in summary
    assert summary["renamed"] == [("ASTER", "ASTERDM", 2)]


def test_apply_fixes_renames_adds_and_removes(tmp_path):
    """--fix rewrites symbols everywhere, inserts section-1, removes section-3."""
    p = tmp_path / "universes.py"
    _fake_universes_file(p)
    res = _FIX_RES
    res = dict(res)
    # Removal target inside the block:
    p.write_text(p.read_text(encoding="utf-8").replace(
        '    "TATAMETALI": "delisted — merged into Tata Steel (last trade 2024-02-05)",\n',
        '    "RESUMEDCO": "delisted — old name",\n'
        '    "TATAMETALI": "delisted — merged into Tata Steel (last trade 2024-02-05)",\n',
    ), encoding="utf-8")

    summary = audit.apply_fixes(res, path=str(p))

    out = p.read_text(encoding="utf-8")
    assert '"ASTERDM"' in out and '"ASTER"' not in out       # renamed (list + sector)
    assert '"STALECO": "no trades since 2025-11-10"' in out   # added
    assert '"RESUMEDCO"' not in out                            # removed
    assert summary["changed"] is True
    assert summary["renamed"] == [("ASTER", "ASTERDM", 2)]
    assert summary["annotated_added"] == [("STALECO", "2025-11-10")]
    assert summary["annotated_removed"] == ["RESUMEDCO"]
    assert summary["backup"] == str(p) + ".bak"
    assert (tmp_path / "universes.py.bak").exists()
    import ast
    ast.parse(out)  # still valid Python


def test_apply_fixes_rename_never_hits_longer_symbols(tmp_path):
    """Quoted replacement must not corrupt ASTERMINDS-style neighbours."""
    p = tmp_path / "universes.py"
    _fake_universes_file(p)
    p.write_text(p.read_text(encoding="utf-8").replace(
        '    "AFFLE", "ASTER",\n', '    "AFFLE", "ASTER", "ASTERMINDS",\n',
    ), encoding="utf-8")

    audit.apply_fixes({"rename_suggestions": {"ASTER": "ASTERDM"}},
                      path=str(p))

    out = p.read_text(encoding="utf-8")
    assert '"ASTERDM"' in out and '"ASTERMINDS"' in out
    assert '"ASTER"' not in out


def test_apply_fixes_noop_leaves_file_untouched(tmp_path):
    """Nothing to fix -> no write, no .bak."""
    p = tmp_path / "universes.py"
    _fake_universes_file(p)
    before = p.read_text(encoding="utf-8")

    summary = audit.apply_fixes({}, path=str(p))

    assert summary["changed"] is False
    assert p.read_text(encoding="utf-8") == before
    assert not (tmp_path / "universes.py.bak").exists()


def test_apply_fixes_not_found_ticker_is_skipped(tmp_path):
    """A rename for a symbol absent from the file (already fixed) is skipped."""
    p = tmp_path / "universes.py"
    _fake_universes_file(p)
    before = p.read_text(encoding="utf-8")

    summary = audit.apply_fixes(
        {"rename_suggestions": {"AVALONLABS": "AVALON"}}, path=str(p))

    assert summary["not_found"] == ["AVALONLABS"]
    assert summary["changed"] is False
    assert p.read_text(encoding="utf-8") == before


def test_apply_fixes_missing_block_only_renames(tmp_path):
    """No SUSPENDED_OR_DELISTED block -> annotation edits skipped, renames kept."""
    p = tmp_path / "universes.py"
    p.write_text('NIFTY_SMALLCAP_100 = [\n    "ASTER",\n]\n', encoding="utf-8")

    summary = audit.apply_fixes(_FIX_RES, path=str(p))

    out = p.read_text(encoding="utf-8")
    assert '"ASTERDM"' in out
    assert "STALECO" not in out
    assert "skipped" in summary
    assert summary["changed"] is True


def test_format_fix_summary_lines():
    """Summary text covers each action type."""
    s = {"changed": True, "renamed": [("ASTER", "ASTERDM", 2)],
         "annotated_added": [("STALECO", "2025-11-10")],
         "annotated_removed": ["RESUMEDCO"], "not_found": ["AVALONLABS"],
         "backup": "/x/universes.py.bak"}
    text = audit.format_fix_summary(s)
    assert "ASTER -> ASTERDM (2 occurrence" in text
    assert "STALECO to SUSPENDED_OR_DELISTED" in text
    assert "RESUMEDCO from SUSPENDED_OR_DELISTED" in text
    assert "AVALONLABS" in text
    assert ".bak" in text

    assert "nothing to apply" in audit.format_fix_summary({"changed": False})


def test_static_universes_excludes_dynamic_placeholders():
    """Live/full-market placeholders are not audited (they mirror NIFTY_BROAD)."""
    keys = set(audit._STATIC_UNIVERSES)
    assert "NSE ALL (Live ~2,200)" not in keys
    assert "BSE ALL (Live ~4,500)" not in keys
    assert "FULL MARKET (NSE+BSE ~5,900)" not in keys
    assert "NIFTY 50" in keys and "ALL (Combined)" in keys


def test_all_universes_single_union_fetch_with_breakdown(monkeypatch):
    """--all: one fetch for the whole union, per-universe missing attributed."""
    calls = []

    def fake_fetch(tickers, **k):
        calls.append(sorted(tickers))
        return {t: _frame_ending(2) for t in tickers if t != "GONE"}
    monkeypatch.setattr(audit, "fetch_batch_yfinance", fake_fetch)
    monkeypatch.setattr(audit, "fetch_stock_data", lambda *a, **k: None)
    monkeypatch.setattr(audit, "_nse_mainboard", list)
    monkeypatch.setattr(audit, "_STATIC_UNIVERSES", {
        "U1": ["A", "B", "GONE"],
        "U2": ["B", "C"],
    })

    res = audit.audit_all_universes(period="3y")

    assert calls == [["A", "B", "C", "GONE"]]  # one union fetch, no repeats
    pu = res["per_universe"]
    assert pu["U1"]["members"] == 3 and pu["U1"]["missing"] == ["GONE"]
    assert pu["U1"]["fetched"] == 2
    assert pu["U2"]["members"] == 2 and pu["U2"]["missing"] == []
    assert res["missing"] == ["GONE"]


def test_format_report_rename_section_and_per_universe():
    """Section 5 lists verified renames; --all prints the per-universe block."""
    res = {
        "period": "3y", "max_age_days": 45.0,
        "tickers": 4, "fetched": 3, "stale_total": 0,
        "unannotated_stale": [], "annotated_stale": [],
        "annotated_fresh": [], "missing": ["ASTER"],
        "neg_cache_skipped": [], "rename_suggestions": {"ASTER": "ASTERDM"},
        "membership": {}, "annotated": [],
        "per_universe": {
            "NIFTY SMALLCAP 100": {"members": 51, "fetched": 50,
                                    "missing": ["ASTER"], "stale_unannotated": []},
            "NIFTY 50": {"members": 51, "fetched": 51,
                          "missing": [], "stale_unannotated": []},
        },
    }

    report = audit.format_report(res)

    assert "RENAME SUGGESTED" in report
    assert "ASTER -> ASTERDM" in report
    assert "Per-universe (missing / members)" in report
    assert "1/51 missing" in report and "0/51 missing" in report