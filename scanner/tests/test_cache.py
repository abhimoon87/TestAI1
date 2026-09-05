"""Unit tests for scanner.cache.TTLCache — the shared in-memory TTL cache
used by all enrichment provider modules.

Expiry is tested with a fake monotonic clock so the suite never depends on
real sleep timing."""

import threading
import time

import pytest

from scanner.cache import TTLCache


class _FakeClock:
    """Deterministic stand-in for time.monotonic (advance manually)."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def cache():
    # Generous TTL: expiry is exercised separately with a fake clock
    return TTLCache(ttl=3600, namespace="test")


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(time, "monotonic", fake.monotonic)
    return fake


class TestTTLCacheBasics:
    def test_set_get_roundtrip(self, cache):
        cache.set("k", {"a": 1})
        assert cache.get("k") == {"a": 1}

    def test_get_missing_returns_none(self, cache):
        assert cache.get("nope") is None

    def test_expiry_returns_none_and_evicts(self, clock):
        c = TTLCache(ttl=50, namespace="x")
        c.set("k", 42)
        clock.now += 49
        assert c.get("k") == 42  # still fresh
        clock.now += 2  # 51 >= ttl 50
        assert c.get("k") is None  # expired -> evicted
        assert c.size == 0

    def test_overwrite_updates_value(self, cache):
        cache.set("k", 1)
        cache.set("k", 2)
        assert cache.get("k") == 2

    def test_clear(self, cache):
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

    def test_clear_expired_counts_only_stale(self, clock):
        c = TTLCache(ttl=50, namespace="x")
        c.set("stale", 1)
        clock.now += 50  # "stale" expired, exactly at the boundary
        c.set("fresh", 2)
        removed = c.clear_expired()
        assert removed == 1  # only "stale" expired
        assert c.get("fresh") == 2
        assert c.get("stale") is None


class TestTTLCacheNamespacing:
    def test_namespaces_are_isolated(self):
        a = TTLCache(ttl=3600, namespace="alpha")
        b = TTLCache(ttl=3600, namespace="beta")
        a.set("same-key", "A")
        b.set("same-key", "B")
        assert a.get("same-key") == "A"
        assert b.get("same-key") == "B"


class TestTTLCacheKeys:
    def test_make_key_is_deterministic(self):
        c = TTLCache(ttl=3600, namespace="k")
        assert c.make_key("x", "y") == c.make_key("x", "y")

    def test_make_key_distinguishes_parts(self):
        c = TTLCache(ttl=3600, namespace="k")
        assert c.make_key("ab", "c") != c.make_key("a", "bc")

    def test_make_key_unhashed(self):
        c = TTLCache(ttl=3600, namespace="k")
        assert c.make_key("a", "b", hashed=False) == "a:b"


class TestTTLCacheThreadSafety:
    def test_concurrent_set_get(self):
        cache = TTLCache(ttl=3600, namespace="threads")
        errors = []

        def worker(n):
            try:
                for i in range(200):
                    cache.set("k", (n, i))
                    cache.get("k")
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cache.get("k") is not None
        assert cache.size == 1
