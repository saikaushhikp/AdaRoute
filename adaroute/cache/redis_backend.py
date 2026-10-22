"""Shared Redis and In-Memory Caching Backends for AdaRoute.

Provides:
- BaseCacheBackend interface
- RedisCacheBackend with connection pooling, JSON serialization, and TTL
- InMemoryCacheBackend with dict storage and expiration
- CacheEntry model with real vs simulated provenance
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any

from adaroute.schemas import PrivacyLevel

logger = logging.getLogger("adaroute.cache")

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False


class CacheEntry:
    """Provenance-aware cache entry."""

    def __init__(
        self,
        key: str,
        response: dict[str, Any],
        origin_model: str,
        origin_provider: str,
        origin_privacy: PrivacyLevel | str,
        origin_quality: float,
        origin_policy: str = "adaptive",
        content_version: str = "v1",
        ttl_seconds: float = 300.0,
        is_simulated: bool = False,
        source: str = "",
        created_at: float | None = None,
    ):
        self.key = key
        self.response = response
        self.origin_model = origin_model
        self.origin_provider = origin_provider
        if isinstance(origin_privacy, PrivacyLevel):
            self.origin_privacy = origin_privacy
        else:
            self.origin_privacy = PrivacyLevel.from_str(str(origin_privacy))
        self.origin_quality = float(origin_quality)
        self.origin_policy = origin_policy
        self.content_version = content_version
        self.ttl_seconds = float(ttl_seconds)
        self.is_simulated = is_simulated or ("simulated" in str(source).lower())
        self.source = source or ("simulated" if self.is_simulated else "real")
        self.created_at = created_at if created_at is not None else time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "response": self.response,
            "origin_model": self.origin_model,
            "origin_provider": self.origin_provider,
            "origin_privacy": self.origin_privacy.value,
            "origin_quality": self.origin_quality,
            "origin_policy": self.origin_policy,
            "content_version": self.content_version,
            "ttl_seconds": self.ttl_seconds,
            "is_simulated": self.is_simulated,
            "source": self.source,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        return cls(
            key=data["key"],
            response=data["response"],
            origin_model=data["origin_model"],
            origin_provider=data["origin_provider"],
            origin_privacy=PrivacyLevel.from_str(data["origin_privacy"]),
            origin_quality=data["origin_quality"],
            origin_policy=data.get("origin_policy", "adaptive"),
            content_version=data.get("content_version", "v1"),
            ttl_seconds=data.get("ttl_seconds", 300.0),
            is_simulated=data.get("is_simulated", False),
            source=data.get("source", ""),
            created_at=data.get("created_at"),
        )


class BaseCacheBackend(ABC):
    """Abstract interface for AdaRoute caching backends."""

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None:
        """Retrieve an entry by key."""

    @abstractmethod
    def set(self, key: str, entry: CacheEntry, ttl_seconds: float) -> None:
        """Store an entry with a time-to-live in seconds."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete an entry by key. Returns True if existed."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Delete all entries matching a prefix. Returns count of deleted keys."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all entries managed by this cache backend."""


class InMemoryCacheBackend(BaseCacheBackend):
    """In-memory dictionary cache backend with TTL eviction."""

    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}

    def get(self, key: str) -> CacheEntry | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._store[key]
            return None
        return entry

    def set(self, key: str, entry: CacheEntry, ttl_seconds: float) -> None:
        entry.ttl_seconds = ttl_seconds
        self._store[key] = entry

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def delete_prefix(self, prefix: str) -> int:
        keys_to_delete = [k for k in self._store if k.startswith(prefix)]
        for k in keys_to_delete:
            del self._store[k]
        return len(keys_to_delete)

    def clear(self) -> None:
        self._store.clear()


class RedisCacheBackend(BaseCacheBackend):
    """Shared Redis caching backend with connection pool, JSON serialization, and prefixing."""

    def __init__(
        self,
        redis_url: str | None = None,
        redis_host: str | None = None,
        redis_port: int | None = None,
        redis_db: int = 0,
        key_prefix: str = "adaroute:cache:",
        socket_timeout: float = 1.5,
        socket_connect_timeout: float = 1.5,
        client: Any | None = None,
    ) -> None:
        if not REDIS_AVAILABLE and client is None:
            raise RuntimeError("The 'redis' package is not installed. Install redis>=5.0.0.")

        self.key_prefix = key_prefix
        if client is not None:
            self.client = client
            self.pool = None
            return

        resolved_url = redis_url or os.getenv("REDIS_URL")
        resolved_host = redis_host or os.getenv("REDIS_HOST")
        resolved_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        resolved_db = redis_db or int(os.getenv("REDIS_DB", "0"))

        if resolved_url:
            self.pool = redis.ConnectionPool.from_url(
                resolved_url,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=True,
            )
        else:
            self.pool = redis.ConnectionPool(
                host=resolved_host or "localhost",
                port=resolved_port,
                db=resolved_db,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=True,
            )

        self.client = redis.Redis(connection_pool=self.pool)

    def ping(self) -> bool:
        """Checks if Redis server is reachable."""
        return bool(self.client.ping())

    def _format_key(self, key: str) -> str:
        if key.startswith(self.key_prefix):
            return key
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> CacheEntry | None:
        full_key = self._format_key(key)
        raw_val = self.client.get(full_key)
        if not raw_val:
            return None
        try:
            data = json.loads(raw_val)
            entry = CacheEntry.from_dict(data)
            if entry.is_expired():
                self.client.delete(full_key)
                return None
            return entry
        except Exception as e:
            logger.warning(f"Failed to deserialize cache entry from Redis for key {key}: {e}")
            return None

    def set(self, key: str, entry: CacheEntry, ttl_seconds: float) -> None:
        full_key = self._format_key(key)
        data_str = json.dumps(entry.to_dict())
        ttl_int = max(1, int(ttl_seconds))
        self.client.set(full_key, data_str, ex=ttl_int)

    def delete(self, key: str) -> bool:
        full_key = self._format_key(key)
        count = self.client.delete(full_key)
        return bool(count > 0)

    def delete_prefix(self, prefix: str) -> int:
        match_pattern = f"{self._format_key(prefix)}*"
        deleted_count = 0
        batch: list[str] = []
        for key in self.client.scan_iter(match=match_pattern, count=100):
            batch.append(key)
            if len(batch) >= 100:
                deleted_count += self.client.delete(*batch)
                batch = []
        if batch:
            deleted_count += self.client.delete(*batch)
        return deleted_count

    def clear(self) -> None:
        match_pattern = f"{self.key_prefix}*"
        batch: list[str] = []
        for key in self.client.scan_iter(match=match_pattern, count=100):
            batch.append(key)
            if len(batch) >= 100:
                self.client.delete(*batch)
                batch = []
        if batch:
            self.client.delete(*batch)
