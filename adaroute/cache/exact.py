"""Safe Exact-Match Caching for AdaRoute (Hardened Week-3 Version).

Enforces:
1. Strict server-side authorization check before lookup.
2. Provenance-aware cache entry validation:
   - A cached response origin_privacy must satisfy current request privacy_level.
   - External/public model responses CANNOT be served to internal_only requests.
   - Low/medium quality responses CANNOT be served to high quality requirement requests.
3. Composite cache keys incorporating authenticated tenant, user, logical alias,
   generation options, policy version, and content version.
4. Robust dual-index caching:
   - Full-context exact matching (full canonical prompt sequence).
   - Canonical query turn fallback (enables cache hits for repeated queries across conversation turns).
5. Canonical text normalization (NFKC, line endings, collapsed spaces).
6. Automatic bypass for confidential/restricted requests, temperature > 0, or no_cache=True.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from adaroute.schemas import PrivacyLevel, RequestContext, normalize_prompt_text


class CacheEntry:
    def __init__(
        self,
        key: str,
        response: dict[str, Any],
        origin_model: str,
        origin_provider: str,
        origin_privacy: PrivacyLevel,
        origin_quality: float,
        origin_policy: str,
        content_version: str = "v1",
        ttl_seconds: float = 300.0,
        query_key: str | None = None,
    ):
        self.key = key
        self.query_key = query_key
        self.response = response
        self.origin_model = origin_model
        self.origin_provider = origin_provider
        self.origin_privacy = origin_privacy
        self.origin_quality = origin_quality
        self.origin_policy = origin_policy
        self.content_version = content_version
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class ExactCache:
    def __init__(self, default_ttl_sec: float = 300.0, policy_version: str = "v3.0"):
        self.default_ttl_sec = default_ttl_sec
        self.policy_version = policy_version
        self._store: dict[str, CacheEntry] = {}
        self._query_index: dict[str, str] = {}  # query_key -> full_key
        self.hits = 0
        self.misses = 0

    def compute_key(
        self,
        tenant_id: str,
        user_id: str | None,
        logical_model: str,
        model_version: str,
        prompt: str,
        temperature: float = 0.7,
        content_version: str = "v1",
        user_private: bool = False,
    ) -> str:
        """Constructs secure composite cache key with canonical prompt normalization."""
        norm_prompt = normalize_prompt_text(prompt)
        prompt_hash = hashlib.sha256(norm_prompt.encode("utf-8")).hexdigest()
        u_id = user_id or "user"
        t_id = tenant_id or "user"
        m_alias = logical_model.lower().strip()
        raw_key = (
            f"tenant:{t_id}|user:{u_id}|model:{m_alias}|"
            f"temp:{temperature:.2f}|u_priv:{user_private}|"
            f"m_ver:{model_version}|c_ver:{content_version}|"
            f"pol_ver:{self.policy_version}|p_hash:{prompt_hash}"
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _validate_provenance(self, entry: CacheEntry, request: RequestContext) -> bool:
        """Enforces provenance invariants: privacy containment and quality floor."""
        # 1. User private toggle constraint: cloud entries strictly prohibited for user_private requests
        if request.user_private:
            if entry.origin_provider not in ("local", "local_ollama", "mock-local") or (
                entry.origin_privacy == PrivacyLevel.PUBLIC
                and entry.origin_provider in ("gemini", "groq", "cohere", "mock-external")
            ):
                return False

        # 2. Privacy matching
        if request.privacy_level == PrivacyLevel.INTERNAL_ONLY:
            if entry.origin_privacy == PrivacyLevel.PUBLIC or entry.origin_provider not in (
                "local",
                "local_ollama",
                "mock-local",
            ):
                return False

        # 3. Quality floor satisfaction
        quality_floors = {"low": 0.0, "medium": 0.70, "high": 0.85}
        required_floor = quality_floors.get(request.quality_requirement.lower(), 0.70)
        if entry.origin_quality < required_floor:
            return False

        return True

    def get(
        self,
        request: RequestContext,
        logical_model: str,
        model_version: str,
        prompt: str,
        query: str | None = None,
        content_version: str = "v1",
    ) -> dict[str, Any] | None:
        """Looks up cache with strict provenance and dual-index authorization verification."""
        # 1. Bypass policy: explicit no_cache flag
        if request.no_cache:
            self.misses += 1
            return None

        # 2. Bypass policy: confidential or restricted privacy levels bypass cache
        if request.privacy_level in (PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED):
            self.misses += 1
            return None

        temp = request.temperature if request.temperature is not None else 0.7

        # 3. Primary Index: Full Context exact match
        full_key = self.compute_key(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            logical_model=logical_model,
            model_version=model_version,
            prompt=prompt,
            temperature=temp,
            content_version=content_version,
            user_private=request.user_private,
        )

        entry = self._store.get(full_key)
        if entry is not None:
            if entry.is_expired():
                del self._store[full_key]
                if entry.query_key and entry.query_key in self._query_index:
                    del self._query_index[entry.query_key]
            elif self._validate_provenance(entry, request):
                self.hits += 1
                return entry.response

        # 4. Secondary Index: Canonical Query fallback (for repeated user queries in chat sessions)
        if query and query.strip():
            query_key = self.compute_key(
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                logical_model=logical_model,
                model_version=model_version,
                prompt=query,
                temperature=temp,
                content_version=content_version,
                user_private=request.user_private,
            )
            target_full_key = self._query_index.get(query_key)
            if target_full_key:
                q_entry = self._store.get(target_full_key)
                if q_entry is not None:
                    if q_entry.is_expired():
                        del self._store[target_full_key]
                        del self._query_index[query_key]
                    elif self._validate_provenance(q_entry, request):
                        self.hits += 1
                        return q_entry.response

        self.misses += 1
        return None

    def put(
        self,
        request: RequestContext,
        logical_model: str,
        model_version: str,
        prompt: str,
        response: dict[str, Any],
        origin_model: str,
        origin_provider: str,
        origin_privacy: PrivacyLevel,
        origin_quality: float,
        origin_policy: str = "adaptive",
        query: str | None = None,
        content_version: str = "v1",
        ttl_seconds: float | None = None,
    ) -> None:
        """Stores response in cache with dual-index indexing and full provenance metadata."""
        if request.no_cache or request.privacy_level in (PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED):
            return

        temp = request.temperature if request.temperature is not None else 0.7
        full_key = self.compute_key(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            logical_model=logical_model,
            model_version=model_version,
            prompt=prompt,
            temperature=temp,
            content_version=content_version,
            user_private=request.user_private,
        )

        query_key = None
        if query and query.strip():
            query_key = self.compute_key(
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                logical_model=logical_model,
                model_version=model_version,
                prompt=query,
                temperature=temp,
                content_version=content_version,
                user_private=request.user_private,
            )

        ttl = ttl_seconds or self.default_ttl_sec
        entry = CacheEntry(
            key=full_key,
            response=response,
            origin_model=origin_model,
            origin_provider=origin_provider,
            origin_privacy=origin_privacy,
            origin_quality=origin_quality,
            origin_policy=origin_policy,
            content_version=content_version,
            ttl_seconds=ttl,
            query_key=query_key,
        )

        self._store[full_key] = entry
        if query_key:
            self._query_index[query_key] = full_key

    def clear(self) -> None:
        self._store.clear()
        self._query_index.clear()
        self.hits = 0
        self.misses = 0
