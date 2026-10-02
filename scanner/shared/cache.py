"""
Shared TTL cache — memory first, sqlite kv store as the restart backstop.

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
- Named caches persist to the shared sqlite db (kv table), so values
  survive restarts; clear() wipes both layers. Caches constructed without
  a namespace are memory-only.
"""

__all__ = ["TTLCache"]

import copy
import hashlib
import threading
import time
from typing import Generic, TypeVar

from .db import kv_clear, kv_get_json, kv_put_json

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, ttl: int = 4 * 3600, namespace: str = ""):
        try:
            ttl = int(ttl)
        except (TypeError, ValueError):
            ttl = 4 * 3600
        if ttl <= 0:
            ttl = 4 * 3600
        self.ttl = ttl
        self.namespace = namespace
        self._store: dict[str, tuple[T, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        with self._lock:
            item = self._store.get(key)
            if item is not None:
                value, ts = item
                if time.monotonic() - ts < self.ttl:
                    return copy.deepcopy(value)
                # Expired — evict, then try the persistent backstop
                try:
                    del self._store[key]
                except KeyError:
                    pass
        return self._db_get(key)

    def _db_get(self, key: str) -> T | None:
        # ponytail: unnamed caches are memory-only (no namespace, no isolation)
        if not self.namespace:
            return None
        try:
            value = kv_get_json(self.namespace, key)
        except Exception:
            return None
        if value is None:
            return None
        with self._lock:
            self._store[key] = (copy.deepcopy(value), time.monotonic())
        return copy.deepcopy(value)

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._store[key] = (copy.deepcopy(value), time.monotonic())
        if self.namespace:
            # ponytail: non-JSON-serializable values stay memory-only
            try:
                kv_put_json(self.namespace, key, value, expires=time.time() + self.ttl)
            except Exception:
                pass

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
        if self.namespace:
            try:
                kv_clear(self.namespace)
            except Exception:
                pass

    def make_key(self, *parts: str, hashed: bool = True) -> str:
        prefix = f"{self.namespace}:" if self.namespace else ""
        raw = prefix + ":".join(str(p) for p in parts)
        if not hashed:
            return raw
        return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)
