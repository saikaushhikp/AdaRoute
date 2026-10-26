"""Safe Exact-Match Caching for AdaRoute with Shared Redis Backend & In-Memory Fallback.

Key Features (Week-4 Hardened Architecture):
1. Shared Redis caching backend with connection pool, JSON serialization, and configurable TTL.
2. Automatic transparent fallback to an in-memory dictionary backend if Redis is unreachable
   or the redis package is unavailable.
3. Lossless composite cache keying incorporating:
   - Full ordered messages context ([{"role": m.role, "content": m.content}])
   - Bounded generation options (max_tokens, temperature)
   - Tenant isolation & authenticated user partition
   - Logical model alias, model version, and routing policy name
   - User private privacy toggle
4. Complete removal of unsafe secondary query_key fallback index to eliminate context poisoning (W2-FB001).
5. Safe prompt normalization preserving code indentation and internal spacing for multiline code.
6. Real vs Simulated provenance verification: simulated responses strictly rejected for real requests.
7. Comprehensive cache invalidation: invalidate(key), invalidate_prefix(prefix), clear().
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from adaroute.cache.redis_backend import (
    REDIS_AVAILABLE,
    BaseCacheBackend,
    CacheEntry,
    InMemoryCacheBackend,
    RedisCacheBackend,
)
from adaroute.cache.semantic import SemanticCache
from adaroute.schemas import ChatMessage, PrivacyLevel, RequestContext, normalize_prompt_text

logger = logging.getLogger("adaroute.cache")


class ExactCache:
    """Safe Exact-Match Cache supporting Redis and In-Memory storage with provenance isolation."""

    def __init__(
        self,
        default_ttl_sec: float = 300.0,
        policy_version: str = "v4.0",
        redis_url: str | None = None,
        backend: str | BaseCacheBackend | None = None,
    ):
        self.default_ttl_sec = default_ttl_sec
        self.policy_version = policy_version
        self.hits = 0
        self.misses = 0

        if isinstance(backend, BaseCacheBackend):
            self._backend: BaseCacheBackend = backend
            self.backend_type = "custom"
        elif backend == "memory":
            self._backend = InMemoryCacheBackend()
            self.backend_type = "memory"
        else:
            self._backend, self.backend_type = self._init_backend(
                redis_url=redis_url,
                force_redis=(backend == "redis"),
            )
        self.semantic_cache = SemanticCache(similarity_threshold=0.95, redis_url=redis_url)

    def _init_backend(
        self,
        redis_url: str | None = None,
        force_redis: bool = False,
    ) -> tuple[BaseCacheBackend, str]:
        """Attempts connection to shared Redis backend; falls back transparently to in-memory on error."""
        target_url = redis_url or os.getenv("REDIS_URL")
        redis_host = os.getenv("REDIS_HOST")

        should_try_redis = force_redis or target_url is not None or redis_host is not None
        if not should_try_redis:
            # Default to checking if localhost redis is reachable without crashing if not
            target_url = "redis://localhost:6379/0"

        if REDIS_AVAILABLE:
            try:
                rb = RedisCacheBackend(redis_url=target_url, redis_host=redis_host)
                rb.ping()
                logger.info(f"Connected to Redis cache backend at {target_url or redis_host or 'localhost'}")
                return rb, "redis"
            except Exception as e:
                logger.warning(f"Redis cache unreachable ({e}). Automatically falling back to in-memory cache backend.")

        return InMemoryCacheBackend(), "memory"

    @property
    def _store(self) -> dict[str, CacheEntry]:
        """Exposes store for backward-compatible inspections in tests."""
        if isinstance(self._backend, InMemoryCacheBackend):
            return self._backend._store
        return {}

    def compute_key(
        self,
        tenant_id: str,
        user_id: str | None,
        logical_model: str,
        model_version: str,
        prompt: str | None = None,
        messages: list[ChatMessage] | list[dict[str, str]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        policy_name: str = "adaptive",
        content_version: str = "v1",
        user_private: bool = False,
    ) -> str:
        """Constructs secure composite cache key with lossless full conversation context."""
        canonical_messages: list[dict[str, str]] = []

        if messages:
            for m in messages:
                if isinstance(m, dict):
                    role = str(m.get("role", "user")).strip().lower()
                    content = normalize_prompt_text(str(m.get("content", "")))
                else:
                    role = str(getattr(m, "role", "user")).strip().lower()
                    content = normalize_prompt_text(str(getattr(m, "content", "")))
                canonical_messages.append({"role": role, "content": content})
        elif prompt:
            canonical_messages.append({"role": "user", "content": normalize_prompt_text(prompt)})

        # Lossless ordered JSON representation of messages context
        messages_json = json.dumps(canonical_messages, separators=(",", ":"), ensure_ascii=False)
        context_hash = hashlib.sha256(messages_json.encode("utf-8")).hexdigest()

        u_id = user_id or "user"
        t_id = tenant_id or "user"
        m_alias = logical_model.lower().strip()
        pol = (policy_name or "adaptive").lower().strip()
        temp = f"{float(temperature):.2f}" if temperature is not None else "0.70"
        max_tok = str(max_tokens) if max_tokens is not None else "default"

        raw_key = (
            f"tenant:{t_id}|user:{u_id}|model:{m_alias}|"
            f"temp:{temp}|max_tokens:{max_tok}|u_priv:{user_private}|"
            f"m_ver:{model_version}|pol:{pol}|c_ver:{content_version}|"
            f"pol_ver:{self.policy_version}|ctx_hash:{context_hash}"
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _validate_provenance(self, entry: CacheEntry, request: RequestContext) -> bool:
        """Enforces provenance invariants: simulation isolation, privacy containment, and quality floor."""
        # 1. Real vs Simulated isolation (W2-FB001):
        # A simulated response CANNOT satisfy a real-only request (execution_mode == 'real')
        req_mode = getattr(request, "execution_mode", None) or os.getenv("EXECUTION_MODE", "hybrid")
        if str(req_mode).strip().lower() == "real" and (
            entry.is_simulated or "simulated" in getattr(entry, "source", "").lower()
        ):
            return False

        # 2. User private toggle constraint: cloud entries strictly prohibited for user_private requests
        if request.user_private:
            if entry.origin_provider not in ("local", "local_ollama", "mock-local") or (
                entry.origin_privacy == PrivacyLevel.PUBLIC
                and entry.origin_provider in ("gemini", "groq", "cohere", "mock-external")
            ):
                return False

        # 3. Privacy matching
        if request.privacy_level == PrivacyLevel.INTERNAL_ONLY:
            if entry.origin_privacy == PrivacyLevel.PUBLIC or entry.origin_provider not in (
                "local",
                "local_ollama",
                "mock-local",
            ):
                return False

        # 4. Quality floor satisfaction
        quality_floors = {"low": 0.0, "medium": 0.70, "high": 0.85}
        required_floor = quality_floors.get(request.quality_requirement.lower(), 0.70)
        if entry.origin_quality < required_floor:
            return False

        return True

    def get(
        self,
        request: RequestContext,
        logical_model: str,
        model_version: str = "1.0.0",
        prompt: str | None = None,
        messages: list[ChatMessage] | list[dict[str, str]] | None = None,
        policy_name: str | None = None,
        content_version: str = "v1",
        query: str | None = None,  # Deprecated / ignored: secondary index removed
    ) -> dict[str, Any] | None:
        """Looks up cache with lossless context key and provenance validation."""
        # 1. Bypass policy: explicit no_cache flag
        if request.no_cache:
            self.misses += 1
            return None

        # 2. Bypass policy: confidential or restricted privacy levels bypass cache
        if request.privacy_level in (PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED):
            self.misses += 1
            return None

        temp = request.temperature if request.temperature is not None else 0.7
        pol = policy_name or getattr(request, "policy_name", "adaptive")
        max_tok = request.max_tokens

        # Primary Index: Lossless full context exact match
        full_key = self.compute_key(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            logical_model=logical_model,
            model_version=model_version,
            prompt=prompt,
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
            policy_name=pol,
            content_version=content_version,
            user_private=request.user_private,
        )

        try:
            entry = self._backend.get(full_key)
        except Exception as e:
            logger.warning(f"Cache backend get failed ({e}), switching to in-memory fallback.")
            self._backend = InMemoryCacheBackend()
            self.backend_type = "memory"
            entry = None

        if entry is not None:
            if entry.is_expired():
                try:
                    self._backend.delete(full_key)
                except Exception:
                    pass
            elif self._validate_provenance(entry, request):
                self.hits += 1
                return entry.response

        # NOTE: Unsafe secondary query_key fallback index is DISABLED and REMOVED (W2-FB001).
        # Context poisoning is eliminated by keying losslessly on the full ordered context.

        # Dynamic Semantic Caching: approximate vector similarity check (tau >= 0.95)
        # Bounded strictly to standalone user prompts; multi-turn messages / system prompts enforce exact match
        has_system_or_history = messages and (
            len(messages) > 1
            or any(
                (m.role if hasattr(m, "role") else m.get("role", "")).lower() == "system"
                for m in messages
            )
        )
        if prompt and self.semantic_cache and not has_system_or_history:
            try:
                sem_hit = self.semantic_cache.get(request, prompt)
                if sem_hit:
                    self.hits += 1
                    return sem_hit
            except Exception:
                pass

        self.misses += 1
        return None

    def put(
        self,
        request: RequestContext,
        logical_model: str,
        model_version: str = "1.0.0",
        prompt: str | None = None,
        messages: list[ChatMessage] | list[dict[str, str]] | None = None,
        response: dict[str, Any] | None = None,
        origin_model: str = "",
        origin_provider: str = "",
        origin_privacy: PrivacyLevel = PrivacyLevel.PUBLIC,
        origin_quality: float = 0.8,
        origin_policy: str = "adaptive",
        policy_name: str | None = None,
        is_simulated: bool = False,
        source: str = "",
        content_version: str = "v1",
        ttl_seconds: float | None = None,
        query: str | None = None,  # Deprecated / ignored
    ) -> None:
        """Stores response in cache with lossless context key and provenance metadata."""
        if request.no_cache or request.privacy_level in (PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED):
            return

        temp = request.temperature if request.temperature is not None else 0.7
        pol = policy_name or getattr(request, "policy_name", origin_policy)
        max_tok = request.max_tokens

        full_key = self.compute_key(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            logical_model=logical_model,
            model_version=model_version,
            prompt=prompt,
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
            policy_name=pol,
            content_version=content_version,
            user_private=request.user_private,
        )

        ttl = ttl_seconds or self.default_ttl_sec
        entry = CacheEntry(
            key=full_key,
            response=response or {},
            origin_model=origin_model,
            origin_provider=origin_provider,
            origin_privacy=origin_privacy,
            origin_quality=origin_quality,
            origin_policy=origin_policy,
            content_version=content_version,
            ttl_seconds=ttl,
            is_simulated=is_simulated,
            source=source,
        )

        try:
            self._backend.set(full_key, entry, ttl_seconds=ttl)
        except Exception as e:
            logger.warning(f"Cache backend set failed ({e}), switching to in-memory fallback.")
            self._backend = InMemoryCacheBackend()
            self.backend_type = "memory"
            self._backend.set(full_key, entry, ttl_seconds=ttl)

        # Index in SemanticCache for vector similarity retrieval
        has_system_or_history = messages and (
            len(messages) > 1
            or any(
                (m.role if hasattr(m, "role") else m.get("role", "")).lower() == "system"
                for m in messages
            )
        )
        if prompt and self.semantic_cache and not has_system_or_history:
            try:
                self.semantic_cache.put(
                    request=request,
                    prompt_text=prompt,
                    response_content=(response or {}).get("content", ""),
                    model=origin_model,
                    replica_id=(response or {}).get("replica_id", "cache"),
                    output_tokens=(response or {}).get("output_tokens", 0),
                    quality_score=origin_quality,
                    is_simulated=is_simulated,
                )
            except Exception as e:
                logger.debug(f"Semantic cache index failed: {e}")

    def invalidate(self, key: str) -> bool:
        """Invalidates a single cache entry by its key."""
        if self.semantic_cache:
            try:
                self.semantic_cache.clear()
            except Exception:
                pass
        try:
            return self._backend.delete(key)
        except Exception as e:
            logger.warning(f"Cache backend invalidate failed: {e}")
            return False

    def invalidate_prefix(self, prefix: str) -> int:
        """Invalidates all cache entries matching a key prefix."""
        if self.semantic_cache:
            try:
                self.semantic_cache.clear()
            except Exception:
                pass
        try:
            return self._backend.delete_prefix(prefix)
        except Exception as e:
            logger.warning(f"Cache backend invalidate_prefix failed: {e}")
            return 0

    def clear(self) -> None:
        """Clears all entries in the cache."""
        if self.semantic_cache:
            try:
                self.semantic_cache.clear()
            except Exception:
                pass
        try:
            self._backend.clear()
        except Exception as e:
            logger.warning(f"Cache backend clear failed ({e}), resetting to fresh in-memory store.")
            self._backend = InMemoryCacheBackend()
            self.backend_type = "memory"
        if hasattr(self, "semantic_cache") and self.semantic_cache:
            try:
                self.semantic_cache.clear()
            except Exception:
                pass
        self.hits = 0
        self.misses = 0
