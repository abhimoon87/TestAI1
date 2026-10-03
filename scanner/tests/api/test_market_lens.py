"""Market Lens core: caching, consumer shapes, universe filter, chain links."""

import scanner.api.market_lens as ml_mod


class _Resp:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            err = ml_mod.requests.HTTPError(str(self.status_code))
            err.response = self
            raise err


def _resp(payload, status=200):
    return _Resp(payload, status)


class TestTickerCleaning:
    def test_strips_exchanges_and_case(self):
        assert ml_mod._clean_ticker("reliance.ns") == "RELIANCE"
        assert ml_mod._clean_ticker("TCS.BO") == "TCS"
        assert ml_mod._clean_ticker(" HINDUNILVR.NSE ") == "HINDUNILVR"


class TestGetJson:
    def setup_method(self):
        ml_mod._STOCK_CACHE.clear()
        ml_mod._META_CACHE.clear()
        ml_mod._INDEX_CACHE.clear()
        ml_mod._PRICE_CACHE.clear()

    def teardown_method(self):
        ml_mod._STOCK_CACHE.clear()
        ml_mod._META_CACHE.clear()
        ml_mod._INDEX_CACHE.clear()
        ml_mod._PRICE_CACHE.clear()

    def test_success_caches_and_returns(self, monkeypatch):
        monkeypatch.setattr(
            ml_mod.requests,
            "get",
            lambda *a, **k: _resp({"success": True, "data": {"x": 1}}),
        )
        cache = ml_mod._STOCK_CACHE
        k = cache.make_key("t1")
        assert ml_mod._get_json("/stocks/T1", cache, k) == {"x": 1}
        # cached: no second HTTP even if it now fails
        monkeypatch.setattr(
            ml_mod.requests,
            "get",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
        )
        assert ml_mod._get_json("/stocks/T1", cache, k) == {"x": 1}

    def test_404_caches_missing_sentinel(self, monkeypatch):
        calls = []

        def fake_get(*a, **k):
            calls.append(1)
            return _resp({}, status=404)

        monkeypatch.setattr(ml_mod.requests, "get", fake_get)
        cache = ml_mod._STOCK_CACHE
        k = cache.make_key("t404")
        assert ml_mod._get_json("/stocks/T404", cache, k) == ml_mod._MISS
        assert ml_mod._get_json("/stocks/T404", cache, k) == ml_mod._MISS
        assert len(calls) == 1  # definitive miss — not refetched

    def test_missing_sentinel_filters_to_none(self, monkeypatch):
        monkeypatch.setattr(
            ml_mod.requests, "get", lambda *a, **k: _resp({}, status=404)
        )
        ml_mod._STOCK_CACHE.clear()
        assert ml_mod.get_stock("GONE") is None
        # second call is negative-cached: no HTTP at all
        monkeypatch.setattr(
            ml_mod.requests,
            "get",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
        )
        assert ml_mod.get_stock("GONE") is None

    def test_500_returns_none_uncached(self, monkeypatch):
        calls = []

        def fake_get(*a, **k):
            calls.append(1)
            return _resp({}, status=500)

        monkeypatch.setattr(ml_mod.requests, "get", fake_get)
        cache = ml_mod._STOCK_CACHE
        k = cache.make_key("t500")
        assert ml_mod._get_json("/stocks/T500", cache, k) is None
        assert ml_mod._get_json("/stocks/T500", cache, k) is None
        assert len(calls) == 2  # transient failure — retried, not cached

    def test_error_payload_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            ml_mod.requests, "get", lambda *a, **k: _resp({"success": False})
        )
        assert ml_mod._get_json("/stocks/X", ml_mod._STOCK_CACHE, "kk") is None

    def test_empty_data_returns_none_uncached(self, monkeypatch):
        monkeypatch.setattr(
            ml_mod.requests, "get", lambda *a, **k: _resp({"success": True, "data": []})
        )
        assert ml_mod._get_json("/indices", ml_mod._INDEX_CACHE, "kk") is None


class TestYoyGrowth:
    def test_full_window_newest_first(self):
        rows = [{"totalIncome": 200}, {}, {}, {}, {"totalIncome": 100}]
        assert ml_mod.yoy_growth(rows, "totalIncome") == 100.0

    def test_short_window_returns_none(self):
        assert ml_mod.yoy_growth([{"eps": 1}, {"eps": 2}], "eps") is None
        assert ml_mod.yoy_growth(None, "eps") is None

    def test_zero_or_bad_base_returns_none(self):
        rows = [{"eps": 5}, {}, {}, {}, {"eps": 0}]
        assert ml_mod.yoy_growth(rows, "eps") is None
        rows[4]["eps"] = None
        assert ml_mod.yoy_growth(rows, "eps") is None


class TestShareholdingShape:
    def test_sorts_descending_and_computes_deltas(self):
        raw = [
            {"quarterEnd": "31-Mar-2026", "promoters": "60.0"},
            {"quarterEnd": "30-Jun-2026", "promoters": "61.5"},
            {"quarterEnd": "31-Dec-2025", "promoters": "59.0"},
        ]
        out = ml_mod.ml_shareholding_shape(raw)
        assert out["quarter"] == "Jun 2026"
        prom = out["series"]["promoters"]
        assert prom["latest"] == 61.5
        assert prom["total"] == 2.5  # 61.5 - 59.0
        assert prom["recent"] == 1.5  # 61.5 - 60.0

    def test_single_quarter_has_none_deltas(self):
        out = ml_mod.ml_shareholding_shape(
            [{"quarterEnd": "30-Jun-2026", "promoters": "52.0"}]
        )
        assert out["series"]["promoters"] == {
            "latest": 52.0,
            "total": None,
            "recent": None,
        }

    def test_garbage_returns_none(self):
        assert ml_mod.ml_shareholding_shape([]) is None
        assert (
            ml_mod.ml_shareholding_shape([{"quarterEnd": "junk", "promoters": "1"}])
            is None
        )
        assert ml_mod.ml_shareholding_shape([{"quarterEnd": "30-Jun-2026"}]) is None


class TestFilterUniverse:
    @staticmethod
    def _no_criteria_returns_untouched(monkeypatch):
        def boom(t):
            raise AssertionError("must not fetch with no criteria")

        monkeypatch.setattr(ml_mod, "get_stock", boom)
        tickers = ["A", "B"]
        assert ml_mod.filter_universe(tickers) is tickers

    def test_no_criteria_zero_http(self, monkeypatch):
        self._no_criteria_returns_untouched(monkeypatch)

    def test_blank_criteria_zero_http(self, monkeypatch):
        def boom(t):
            raise AssertionError("must not fetch")

        monkeypatch.setattr(ml_mod, "get_stock", boom)
        tickers = ["A"]
        assert (
            ml_mod.filter_universe(tickers, sectors="  ", pe_max=0, mcap_min_cr=0)
            is tickers
        )

    def test_pe_and_mcap_and_sector(self, monkeypatch):
        profiles = {
            "OK": {"peRatio": 15, "marketCap": 5e11, "sector": "Banks"},
            "RICH": {"peRatio": 90, "marketCap": 5e11, "sector": "Banks"},
            "TINY": {"peRatio": 15, "marketCap": 1e7, "sector": "Banks"},  # ₹1 Cr
            "WRONG": {"peRatio": 15, "marketCap": 5e11, "sector": "IT"},
            "NOPROF": None,
        }
        monkeypatch.setattr(ml_mod, "get_stock", lambda t: profiles[t])
        out = ml_mod.filter_universe(
            list(profiles), sectors="Banks", pe_max=50, mcap_min_cr=100
        )
        assert out == ["OK"]

    def test_large_universe_all_fetched(self, monkeypatch):
        seen = []

        def fake_get(t):
            seen.append(t)
            return {"peRatio": 10, "sector": "Banks"}

        monkeypatch.setattr(ml_mod, "get_stock", fake_get)
        tickers = [f"T{i}" for i in range(501)]
        out = ml_mod.filter_universe(tickers, "Banks")
        assert sorted(seen) == sorted(tickers)  # no ceiling — every ticker verified
        assert out == tickers


class TestChainLinks:
    def test_fundamentals_link_shapes(self, monkeypatch):
        import scanner.api.data_providers as dp

        monkeypatch.setattr(ml_mod, "get_stock", lambda t: {"peRatio": 22.5})
        monkeypatch.setattr(
            ml_mod,
            "get_quarterly",
            lambda t: [
                {"eps": 10, "totalIncome": 200},
                {},
                {},
                {},
                {"eps": 8, "totalIncome": 100},
            ],
        )
        fund = dp._fetch_fundamentals_marketlens("RELIANCE")
        assert fund["pe_ratio"] == 22.5
        assert fund["eps_growth"] == 25.0
        assert fund["rev_growth"] == 100.0
        assert fund["roe"] is None

    def test_fundamentals_link_all_none_is_none(self, monkeypatch):
        import scanner.api.data_providers as dp

        monkeypatch.setattr(ml_mod, "get_stock", lambda t: {})
        monkeypatch.setattr(ml_mod, "get_quarterly", lambda t: None)
        assert dp._fetch_fundamentals_marketlens("X") is None

    def test_index_link_synthesizes_two_closes(self, monkeypatch):
        import scanner.api.data_providers as dp

        monkeypatch.setattr(
            ml_mod,
            "get_indices",
            lambda: [
                {"ticker": "NIFTY_50", "value": 22421.95, "change": -0.88},
                {"ticker": "BANKNIFTY", "value": 54450.75, "change": -0.33},
            ],
        )
        df = dp._fetch_marketlens_index("^NSEI", "1y")
        expected_prev = 22421.95 / (1 + (-0.88 / 100.0))
        assert list(df["close"]) == [expected_prev, 22421.95]
        # change is a percent: prev close reproduces the day's pct
        prev, last = df["close"].iloc[0], df["close"].iloc[1]
        assert round((last - prev) / prev * 100, 2) == -0.88

    def test_index_link_ignores_unmapped_indices(self, monkeypatch):
        import scanner.api.data_providers as dp

        monkeypatch.setattr(ml_mod, "get_indices", list)
        assert dp._fetch_marketlens_index("^CNXIT", "1y") is None
        assert dp._fetch_marketlens_index("^NSEI", "1y") is None


class TestPriceLink:
    @staticmethod
    def _rows(n=60, start="2026-01-01"):
        import pandas as pd

        days = pd.date_range(start, periods=n)
        return [
            {"date": d.strftime("%Y-%m-%d"), "price": 100.0 + i, "volume": 1000.0 + i}
            for i, d in enumerate(days)
        ]

    def test_maps_close_only_rows_to_ohlcv_frame(self, monkeypatch):
        import scanner.api.data_providers as dp

        monkeypatch.setattr(ml_mod, "get_price_history", lambda t, p: self._rows(60))
        df = dp._fetch_marketlens_price("RELIANCE", "1y")

        assert df is not None
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df["high"].isna().all() and df["low"].isna().all()
        assert list(df["close"]) == [100.0 + i for i in range(60)]
        assert (df["open"] == df["close"]).all()
        assert df.index.is_monotonic_increasing

    def test_unsorted_rows_are_sorted_ascending(self, monkeypatch):
        import scanner.api.data_providers as dp

        rows = list(reversed(self._rows(60)))
        monkeypatch.setattr(ml_mod, "get_price_history", lambda t, p: rows)
        df = dp._fetch_marketlens_price("RELIANCE", "1y")
        assert df.index.is_monotonic_increasing

    def test_period_maps_to_ml_windows(self, monkeypatch):
        import scanner.api.data_providers as dp

        seen = []
        monkeypatch.setattr(
            ml_mod, "get_price_history", lambda t, p: seen.append(p) or None
        )
        for period in ("6mo", "1y", "2y", "3y", "5y", "bogus"):
            dp._fetch_marketlens_price("X", period)
        assert seen == ["6M", "1Y", "5Y", "5Y", "5Y", "1Y"]

    def test_no_rows_is_none(self, monkeypatch):
        import scanner.api.data_providers as dp

        monkeypatch.setattr(ml_mod, "get_price_history", lambda t, p: None)
        assert dp._fetch_marketlens_price("X", "1y") is None
        monkeypatch.setattr(ml_mod, "get_price_history", lambda t, p: [])
        assert dp._fetch_marketlens_price("X", "1y") is None

    def test_garbage_rows_skipped(self, monkeypatch):
        import scanner.api.data_providers as dp

        rows = [{"date": "junk", "price": "x"}, {"price": 50.0}] + self._rows(55)
        monkeypatch.setattr(ml_mod, "get_price_history", lambda t, p: rows)
        df = dp._fetch_marketlens_price("X", "1y")
        assert df is not None
        assert len(df) == 55


class TestGetPriceHistory:
    def setup_method(self):
        ml_mod._PRICE_CACHE.clear()

    def teardown_method(self):
        ml_mod._PRICE_CACHE.clear()

    def test_success_returns_list_and_caches(self, monkeypatch):
        rows = [{"date": "2026-01-05", "price": 100.0, "volume": 10.0}]
        monkeypatch.setattr(
            ml_mod.requests,
            "get",
            lambda *a, **k: _resp({"success": True, "data": rows}),
        )
        assert ml_mod.get_price_history("RELIANCE", "1Y") == rows
        monkeypatch.setattr(
            ml_mod.requests,
            "get",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
        )
        assert ml_mod.get_price_history("RELIANCE", "1Y") == rows

    def test_clean_ticker_and_bad_ticker(self, monkeypatch):
        seen = []

        def fake_get(url, **k):
            seen.append(url)
            return _resp({"success": True, "data": [1]})

        monkeypatch.setattr(ml_mod.requests, "get", fake_get)
        ml_mod.get_price_history("reliance.ns", "1Y")
        assert "/stocks/RELIANCE/price-history?period=1Y" in seen[0]
        assert ml_mod.get_price_history("  ", "1Y") is None
        assert len(seen) == 1
