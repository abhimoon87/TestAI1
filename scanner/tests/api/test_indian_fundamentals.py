"""Promoter snapshot history — record once/day, prune, change math."""

import datetime
import json

import scanner.api.indian_fundamentals as fund_mod


def _iso(monkeypatch, tmp_path):
    path = tmp_path / "promoter_history.json"
    monkeypatch.setattr(fund_mod, "_PROMOTER_HISTORY_PATH", str(path))
    return path


def test_record_creates_series(monkeypatch, tmp_path):
    _iso(monkeypatch, tmp_path)
    series = fund_mod.record_promoter_snapshot("RELIANCE", 51.8)
    today = datetime.date.today().isoformat()
    assert series == [[today, 51.8]]
    from scanner.shared import db

    assert db.kv_get_json("promoter_history", "all") == {"RELIANCE": [[today, 51.8]]}


def test_same_day_appends_once_first_write_wins(monkeypatch, tmp_path):
    _iso(monkeypatch, tmp_path)
    fund_mod.record_promoter_snapshot("TCS", 72.0)
    series = fund_mod.record_promoter_snapshot("TCS", 72.3)
    assert len(series) == 1
    assert series[0][1] == 72.0


def test_next_day_appends_and_prunes_to_keep(monkeypatch, tmp_path):
    path = _iso(monkeypatch, tmp_path)
    old = [[f"2020-01-{i:02d}", 50.0 + i] for i in range(1, 13)]  # 12 entries
    path.write_text(json.dumps({"INFY": old}), encoding="utf-8")
    series = fund_mod.record_promoter_snapshot("INFY", 55.5)
    assert len(series) == fund_mod._PROMOTER_KEEP  # 12 old + 1 new, pruned
    assert series[-1] == [datetime.date.today().isoformat(), 55.5]
    assert series[0] == old[1]  # oldest dropped


def test_zero_or_negative_pct_is_ignored(monkeypatch, tmp_path):
    path = _iso(monkeypatch, tmp_path)
    assert fund_mod.record_promoter_snapshot("X", 0) == []
    assert fund_mod.record_promoter_snapshot("X", -1.0) == []
    assert not path.exists()


def test_corrupt_history_file_recovers(monkeypatch, tmp_path):
    path = _iso(monkeypatch, tmp_path)
    path.write_text("{not json", encoding="utf-8")
    series = fund_mod.record_promoter_snapshot("Y", 40.0)
    assert len(series) == 1


def test_change_needs_two_snapshots():
    assert fund_mod.promoter_position_change([]) == (None, None)
    assert fund_mod.promoter_position_change([["2026-01-01", 50.0]]) == (None, None)


def test_change_math_total_and_recent():
    assert fund_mod.promoter_position_change([["a", 50.0], ["b", 50.5]]) == (0.5, 0.5)
    total, recent = fund_mod.promoter_position_change(
        [["a", 50.0], ["b", 50.2], ["c", 50.5]]
    )
    assert total == 0.5
    assert recent == 0.3


def test_change_negative_direction():
    total, recent = fund_mod.promoter_position_change([["a", 52.0], ["b", 51.5]])
    assert total == -0.5
    assert recent == -0.5


# ── Screener quarterly shareholding parse ──────────────────────────────────

_SHP_HTML = """
<html><body>
<div class="tab-buttons">
  <button data-tab-id="quarterly-shp" type="button">Quarterly</button>
  <button data-tab-id="yearly-shp" type="button">Yearly</button>
</div>
<div id="quarterly-shp">
  <div class="responsive-holder">
    <table class="data-table">
      <thead><tr><th class="text"></th><th>Sep 2023</th><th>Dec 2023</th><th>Jun 2026</th></tr></thead>
      <tbody>
        <tr class="stripe"><td class="text"><button onclick="Company.showShareholders('promoters', 'quarterly', this)">Promoters&nbsp;</button></td>
          <td>50.27%</td><td>50.30%</td><td>50.48%</td></tr>
        <tr><td class="text"><button onclick="Company.showShareholders('foreign_institutions', 'quarterly', this)">FIIs&nbsp;</button></td>
          <td>22.60%</td><td>22.13%</td><td>17.19%</td></tr>
        <tr><td class="text"><button onclick="Company.showShareholders('domestic_institutions', 'quarterly', this)">DIIs&nbsp;</button></td>
          <td>16.00%</td><td>20.00%</td><td>21.10%</td></tr>
        <tr><td class="text"><button onclick="Company.showShareholders('promoters', 'yearly', this)">Promoters&nbsp;</button></td>
          <td>1.00%</td></tr>
      </tbody>
    </table>
  </div>
</div>
<div id="yearly-shp"><table class="data-table"><thead><tr><th></th><th>2026</th></tr>
<tbody><tr><td>yearly junk</td><td>9.99%</td></tr></tbody></table></div>
</body></html>
"""


def test_shareholding_parse_series_and_quarter():
    out = fund_mod.parse_shareholding_html(_SHP_HTML)
    assert out["quarter"] == "Jun 2026"
    assert out["series"]["promoters"] == {
        "latest": 50.48,
        "total": 0.21,
        "recent": 0.18,
    }
    assert out["series"]["foreign_institutions"] == {
        "latest": 17.19,
        "total": -5.41,
        "recent": -4.94,
    }
    assert out["series"]["domestic_institutions"] == {
        "latest": 21.1,
        "total": 5.1,
        "recent": 1.1,
    }


def test_shareholding_parse_ignores_yearly_rows_and_missing_div():
    assert fund_mod.parse_shareholding_html("<html>nothing here</html>") is None
    # Only the promoters row survives; yearly-period rows are never matched.
    promoters_only = (
        _SHP_HTML.split("foreign_institutions", 1)[0]
        + "</tbody></table></div></div></body></html>"
    )
    out = fund_mod.parse_shareholding_html(promoters_only)
    assert set(out["series"]) == {"promoters"}


def test_shareholding_fetch_cache_only_and_network(monkeypatch):
    fund_mod._SHP_CACHE.clear()
    import scanner.api.market_lens as ml_mod

    try:
        assert fund_mod.fetch_shareholding_pattern("X", cache_only=True) is None

        class _Resp:
            status_code = 200
            text = _SHP_HTML

        calls = []
        monkeypatch.setattr(
            fund_mod.requests,
            "get",
            lambda *a, **k: calls.append(a) or _Resp(),
        )
        monkeypatch.setattr(ml_mod, "get_shareholding", lambda t: None)
        out = fund_mod.fetch_shareholding_pattern("X")
        again = fund_mod.fetch_shareholding_pattern("X", cache_only=True)
        assert out == again and out["quarter"] == "Jun 2026"
        assert len(calls) == 1  # second read served from cache
    finally:
        fund_mod._SHP_CACHE.clear()


def test_shareholding_merges_market_lens_promoters(monkeypatch):
    fund_mod._SHP_CACHE.clear()
    import scanner.api.market_lens as ml_mod

    try:

        class _Resp:
            status_code = 200
            text = _SHP_HTML

        monkeypatch.setattr(fund_mod.requests, "get", lambda *a, **k: _Resp())
        monkeypatch.setattr(
            ml_mod,
            "get_shareholding",
            lambda t: [
                {
                    "quarterEnd": "30-Jun-2026",
                    "promoters": "61.5",
                    "publicHolding": "9.2",
                },
                {
                    "quarterEnd": "31-Mar-2026",
                    "promoters": "60.0",
                    "publicHolding": "9.5",
                },
                {
                    "quarterEnd": "31-Dec-2025",
                    "promoters": "59.0",
                    "publicHolding": "9.9",
                },
            ],
        )
        out = fund_mod.fetch_shareholding_pattern("MERGE")
        # Promoters from Market Lens (latest 61.5, total +2.5, recent +1.5)…
        assert out["series"]["promoters"] == {
            "latest": 61.5,
            "total": 2.5,
            "recent": 1.5,
        }
        # …FII/DII and the quarter label from Screener.in.
        assert out["quarter"] == "Jun 2026"
        assert out["series"]["foreign_institutions"]["latest"] == 17.19
        assert out["series"]["domestic_institutions"]["latest"] == 21.1
    finally:
        fund_mod._SHP_CACHE.clear()


def test_shareholding_market_lens_only_when_screener_down(monkeypatch):
    fund_mod._SHP_CACHE.clear()
    import scanner.api.market_lens as ml_mod

    try:

        class _Resp:
            status_code = 404
            text = ""

        monkeypatch.setattr(fund_mod.requests, "get", lambda *a, **k: _Resp())
        monkeypatch.setattr(
            ml_mod,
            "get_shareholding",
            lambda t: [
                {
                    "quarterEnd": "30-Jun-2026",
                    "promoters": "61.5",
                    "publicHolding": "9.2",
                },
                {
                    "quarterEnd": "31-Mar-2026",
                    "promoters": "60.0",
                    "publicHolding": "9.5",
                },
            ],
        )
        out = fund_mod.fetch_shareholding_pattern("LENS_ONLY")
        assert out["quarter"] == "Jun 2026"
        assert out["series"]["promoters"]["latest"] == 61.5
        assert "foreign_institutions" not in out["series"]
    finally:
        fund_mod._SHP_CACHE.clear()


def test_shareholding_fetch_404_returns_none(monkeypatch):
    fund_mod._SHP_CACHE.clear()
    import scanner.api.market_lens as ml_mod

    try:

        class _Resp:
            status_code = 404
            text = ""

        monkeypatch.setattr(fund_mod.requests, "get", lambda *a, **k: _Resp())
        monkeypatch.setattr(ml_mod, "get_shareholding", lambda t: None)
        assert fund_mod.fetch_shareholding_pattern("NOPE") is None
    finally:
        fund_mod._SHP_CACHE.clear()
