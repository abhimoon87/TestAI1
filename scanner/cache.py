"""
Shared in-memory TTL cache — replaces 9x copy-pasted _cache_get/_cache_set.

Usage:
    from .cache import TTLCache
    _FREE_API_CACHE = TTLCache[dict](ttl=4*3600, namespace="free_api")
    _FREE_API_CACHE.get(key) -> dict | None
    _FREE_API_CACHE.set(key, value)
    _FREE_API_CACHE.make_key("mandi", commodity, state)

Features:
- Per-namespace isolation (avoids collision on same md5)
- Generic[T] for dict vs list (symbol_fetcher)
- time.monotonic() (NTP-safe) + threading.Lock + delete-on-expiry
- Centralized hashlib.md5(usedforsecurity=False)
"""

import hashlib
import logging
import threading
import time
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, ttl: int = 4 * 3600, namespace: str = ""):
        self.ttl = ttl
        self.namespace = namespace
        self._store: dict[str, tuple[T, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            value, ts = item
            if time.monotonic() - ts < self.ttl:
                return value
            # Expired — evict
            try:
                del self._store[key]
            except KeyError:
                pass
            return None

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._store[key] = (value, time.monotonic())

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def clear_expired(self) -> int:
        now = time.monotonic()
        removed = 0
        with self._lock:
            for k in list(self._store.keys()):
                _, ts = self._store[k]
                if now - ts >= self.ttl:
                    del self._store[k]
                    removed += 1
        return removed

    def make_key(self, *parts: str, hashed: bool = True) -> str:
        raw = ":".join(str(p) for p in parts)
        if not hashed:
            return raw
        return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)
