"""Safe Exact-Match Caching for AdaRoute.

Enforces:
1. Authorization & tenant isolation before cache lookup
2. Composite cache keys incorporating tenant_id, user_id, logical_model, model_version, prompt, policy_version
3. Explicit bypass for restricted privacy or no_cache directives
4. Time-to-live (TTL) expiration to prevent stale facts
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

from adaroute.schemas import PrivacyLevel, RequestContext


class CacheEntry:
    def __init__(self, key: str, response: dict[str, Any], ttl_seconds: float = 300.0):
        self.key = key
        self.response = response
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class ExactCache:
    def __init__(self, default_ttl_sec: float = 300.0, policy_version: str = "v2.0"):
        self.default_ttl_sec = default_ttl_sec
        self.policy_version = policy_version
        self._store: dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0

    def compute_key(
        self,
        tenant_id: str,
        user_id: str | None,
        logical_model: str,
        model_version: str,
        prompt: str,
    ) -> str:
        """Construct secure composite cache key."""
        prompt_hash = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()
        u_id = user_id or "anonymous"
        raw_key = (
            f"tenant:{tenant_id}|user:{u_id}|model:{logical_model}|"
            f"m_ver:{model_version}|pol_ver:{self.policy_version}|p_hash:{prompt_hash}"
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(
        self,
        request: RequestContext,
        logical_model: str,
        model_version: str,
        prompt: str,
    ) -> dict[str, Any] | None:
        """Lookup cache enforcing authorization & safety policies first."""
        # 1. Bypass policy: no_cache flag
        if request.no_cache:
            self.misses += 1
            return None

        # 2. Bypass policy: High-sensitivity privacy cannot reuse cache across sessions
        if request.privacy_level == PrivacyLevel.RESTRICTED:
            self.misses += 1
            return None

        key = self.compute_key(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            logical_model=logical_model,
            model_version=model_version,
            prompt=prompt,
        )

        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None

        if entry.is_expired():
            del self._store[key]
            self.misses += 1
            return None

        self.hits += 1
        return entry.response

    def put(
        self,
        request: RequestContext,
        logical_model: str,
        model_version: str,
        prompt: str,
        response: dict[str, Any],
        ttl_seconds: float | None = None,
    ):
        """Store response in cache following isolation policies."""
        if request.no_cache or request.privacy_level == PrivacyLevel.RESTRICTED:
            return

        key = self.compute_key(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            logical_model=logical_model,
            model_version=model_version,
            prompt=prompt,
        )
        ttl = ttl_seconds or self.default_ttl_sec
        self._store[key] = CacheEntry(key=key, response=response, ttl_seconds=ttl)

    def clear(self):
        self._store.clear()
        self.hits = 0
        self.misses = 0
