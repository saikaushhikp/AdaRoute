"""Dynamic Redis Vector Semantic Caching Engine (AdaRoute hardened architecture).

Provides approximate vector similarity caching alongside exact-match caching:
- Dense normalized semantic vector representations (FastEmbed with sentence-transformers/all-MiniLM-L6-v2)
- Injectable BaseEmbeddingProvider for fast, isolated, deterministic offline unit testing
- Real Redis Search indexed KNN vector queries (FT.CREATE / FT.SEARCH) with structured TagField filters
- Zero KEYS or full-table scans in the lookup path
- Robust in-memory vector index fallback on Redis / Search module outage
- Semantic guards against false positives:
  * Negation polarity guard (detects "not", "never", "cannot", etc.)
  * Numeric literal identity guard (rejects mismatched numbers / ports / dates)
  * Format & constraint entity guard (rejects conflicting formats e.g. JSON vs YAML)
- Strict safety & governance enforcement:
  * Tenant and User isolation (Tenant A never hits Tenant B; user partition respected)
  * Privacy-level partitioning (public cached entries NEVER satisfy internal_only/confidential)
  * Confidential / Restricted requests bypass semantic cache
  * Execution mode isolation (simulated cache entries never satisfy real requests)
  * Quality floor guard (Q_cached >= Q_required)
  * Generation parameter preservation (max_tokens output bound)
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None


def _vec_to_bytes(vec: list[float]) -> bytes:
    if np is not None:
        return np.array(vec, dtype=np.float32).tobytes()
    return struct.pack(f"{len(vec)}f", *vec)

from adaroute.schemas import PrivacyLevel, RequestContext

logger = logging.getLogger("adaroute.cache.semantic")

try:
    import redis
    from redis.commands.search.field import NumericField, TagField, VectorField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    from redis.exceptions import ResponseError
except ImportError:
    redis = None
    NumericField = None
    TagField = None
    VectorField = None
    IndexDefinition = None
    IndexType = None
    Query = None
    ResponseError = Exception

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None


# -----------------------------------------------------------------------------
# Base Embedding Provider & Concrete Implementations
# -----------------------------------------------------------------------------


class BaseEmbeddingProvider(ABC):
    """Abstract interface for dense semantic embedding providers."""

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        """Encodes text into a unit-normalized dense float vector."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns vector dimension."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns provider model name."""

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Calculates cosine similarity between two vectors."""
        if len(v1) != len(v2) or not v1:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        return max(-1.0, min(1.0, dot))


class FastEmbedProvider(BaseEmbeddingProvider):
    """Real, locally hosted CPU/ONNX embedding provider using fastembed.

    Default model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions, cosine distance).
    Executes locally in ~1-4ms on CPU with zero external API calls.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dimension: int = 384):
        self._model_name = model_name
        self._dimension = dimension
        self._model: TextEmbedding | None = None
        if TextEmbedding is not None:
            try:
                self._model = TextEmbedding(model_name=self._model_name)
            except Exception as e:
                logger.warning(f"Failed to initialize FastEmbed model {model_name}: {e}")
                self._model = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._model_name

    def encode(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self._dimension

        if self._model is not None:
            try:
                vec_iter = self._model.embed([text])
                vec = list(next(vec_iter))
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 1e-9:
                    return [float(x / norm) for x in vec]
                return [float(x) for x in vec]
            except Exception as e:
                logger.warning(f"FastEmbed encoding failed: {e}; falling back to deterministic hashing.")

        # Fallback if fastembed model failed or is unavailable
        return DeterministicTestEmbeddingProvider(dimension=self._dimension).encode(text)


class DeterministicTestEmbeddingProvider(BaseEmbeddingProvider):
    """High-speed, deterministic dense semantic embedding provider for isolation testing.

    Computes unit-normalized D-dimensional dense embeddings using subword n-gram hashing
    and lexical distributions. Zero external download; fast (<0.2ms) and reproducible.
    """

    def __init__(self, dimension: int = 384, name: str = "deterministic-hash-v1"):
        self._dimension = dimension
        self._name = name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._name

    def encode(self, text: str) -> list[float]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower()).strip()
        tokens = cleaned.split()
        if not tokens:
            return [0.0] * self._dimension

        vec = [0.0] * self._dimension

        # 1. Word-level hashing
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dimension
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign * 1.5

        # 2. Character 3-gram hashing for morphological / subword similarity
        chars = f" {cleaned} "
        for i in range(len(chars) - 2):
            gram = chars[i : i + 3]
            h = int(hashlib.sha256(gram.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dimension
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign * 0.5

        # 3. L2 Normalize to unit hypersphere
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            return [round(x / norm, 6) for x in vec]
        return [0.0] * self._dimension


# Backward compatibility alias
LocalEmbeddingProvider = DeterministicTestEmbeddingProvider


# -----------------------------------------------------------------------------
# Semantic Guards
# -----------------------------------------------------------------------------


class SemanticGuards:
    """Conservative pre-guards rejecting semantic near-misses that share vector similarity

    but differ fundamentally in polarity, numerical literals, or constraint keywords.
    """

    NEGATION_TOKENS = {
        "not",
        "never",
        "no",
        "n't",
        "none",
        "neither",
        "nor",
        "cannot",
        "cant",
        "without",
        "hardly",
        "barely",
        "false",
        "deny",
        "refuse",
        "reject",
    }

    FORMAT_KEYWORDS = {
        "json",
        "yaml",
        "xml",
        "csv",
        "sql",
        "html",
        "toml",
        "markdown",
        "text",
    }

    LANG_KEYWORDS = {
        "python",
        "javascript",
        "typescript",
        "rust",
        "golang",
        "go",
        "java",
        "c++",
        "cpp",
        "c#",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "bash",
        "shell",
    }

    @classmethod
    def check_guards(cls, query_text: str, candidate_text: str) -> tuple[bool, str | None]:
        q_lower = query_text.lower()
        c_lower = candidate_text.lower()

        q_words = set(re.findall(r"\b[\w']+\b", q_lower))
        c_words = set(re.findall(r"\b[\w']+\b", c_lower))

        # 1. Negation Guard: Polarity check
        q_negs = q_words & cls.NEGATION_TOKENS
        c_negs = c_words & cls.NEGATION_TOKENS
        if bool(q_negs) != bool(c_negs) or q_negs != c_negs:
            return False, f"Negation polarity mismatch (query={sorted(q_negs)} vs cached={sorted(c_negs)})"

        # 2. Number / Quantity Guard: Numeric literal check
        q_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", q_lower))
        c_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", c_lower))
        if q_nums != c_nums:
            return False, f"Numeric literal mismatch (query={sorted(q_nums)} vs cached={sorted(c_nums)})"

        # 3. Format Entity Guard
        q_fmts = q_words & cls.FORMAT_KEYWORDS
        c_fmts = c_words & cls.FORMAT_KEYWORDS
        if q_fmts != c_fmts:
            return False, f"Format constraint mismatch (query={sorted(q_fmts)} vs cached={sorted(c_fmts)})"

        # 4. Language Entity Guard
        q_langs = q_words & cls.LANG_KEYWORDS
        c_langs = c_words & cls.LANG_KEYWORDS
        if q_langs != c_langs:
            return False, f"Language constraint mismatch (query={sorted(q_langs)} vs cached={sorted(c_langs)})"

        return True, None


# -----------------------------------------------------------------------------
# Cache Entry Dataclass
# -----------------------------------------------------------------------------


@dataclass
class SemanticCacheEntry:
    """Represents a cached response indexed with its dense semantic embedding."""

    key: str
    prompt: str
    vector: list[float]
    response_content: str
    model: str
    replica_id: str
    output_tokens: int
    quality_score: float
    privacy_level: PrivacyLevel
    tenant_id: str
    is_simulated: bool
    temperature: float = 0.7
    max_tokens: int | None = None
    user_private: bool = False
    user_id: str = "user"
    logical_model: str = "general"
    execution_mode: str = "hybrid"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "prompt": self.prompt,
            "vector": self.vector,
            "response_content": self.response_content,
            "model": self.model,
            "replica_id": self.replica_id,
            "output_tokens": self.output_tokens,
            "quality_score": self.quality_score,
            "privacy_level": self.privacy_level.value,
            "tenant_id": self.tenant_id,
            "is_simulated": self.is_simulated,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "user_private": self.user_private,
            "user_id": self.user_id,
            "logical_model": self.logical_model,
            "execution_mode": self.execution_mode,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticCacheEntry:
        return cls(
            key=data["key"],
            prompt=data["prompt"],
            vector=data["vector"],
            response_content=data["response_content"],
            model=data["model"],
            replica_id=data["replica_id"],
            output_tokens=int(data["output_tokens"]),
            quality_score=float(data["quality_score"]),
            privacy_level=PrivacyLevel.from_str(data["privacy_level"]),
            tenant_id=data["tenant_id"],
            is_simulated=bool(data["is_simulated"]),
            temperature=float(data.get("temperature", 0.7)),
            max_tokens=int(data["max_tokens"])
            if data.get("max_tokens") is not None and int(data["max_tokens"]) >= 0
            else None,
            user_private=bool(data.get("user_private", False)),
            user_id=str(data.get("user_id", "user")),
            logical_model=str(data.get("logical_model", "general")),
            execution_mode=str(data.get("execution_mode", "hybrid")),
            created_at=float(data.get("created_at", time.time())),
        )


def _escape_tag_value(val: str) -> str:
    """Escapes special characters in RediSearch tag query values: ,.<>{}[]"':;!@#$%^&*()-+=~|/\\."""
    clean = str(val or "").strip()
    return re.sub(r"([,.<>{}\[\]\"':;!@#$%^&*()\-+=~|/\\])", r"\\\1", clean)


# -----------------------------------------------------------------------------
# SemanticCache: Redis Vector Search with In-Memory Fallback
# -----------------------------------------------------------------------------


class SemanticCache:
    """Redis-indexed vector semantic search cache with conservative guards and safe fallback."""

    INDEX_VERSION = "v1"
    INDEX_NAME = f"idx:adaroute:vsem:{INDEX_VERSION}"
    KEY_PREFIX = f"adaroute:vsem:{INDEX_VERSION}:"

    QUALITY_MAP = {
        "low": 0.0,
        "medium": 0.30,
        "high": 0.85,
    }

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        dimension: int = 384,
        embedder: BaseEmbeddingProvider | None = None,
        redis_url: str | None = None,
        default_ttl: int = 86400,
        index_name: str | None = None,
    ):
        self.threshold = similarity_threshold
        self.default_ttl = default_ttl
        self.index_name = index_name or self.INDEX_NAME

        # Embedding Provider Setup
        if embedder is not None:
            self.embedder = embedder
        elif TextEmbedding is not None:
            self.embedder = FastEmbedProvider(dimension=dimension)
        else:
            self.embedder = DeterministicTestEmbeddingProvider(dimension=dimension)

        self.dimension = self.embedder.dimension
        self._memory_entries: list[SemanticCacheEntry] = []
        self.redis_client = None
        self.has_search = False

        # Attempt Redis connection and index setup
        r_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        if redis is not None and r_url:
            self._init_redis(r_url)

    def _init_redis(self, r_url: str) -> None:
        """Initializes Redis client and ensures RediSearch vector index exists."""
        try:
            client = redis.from_url(r_url, decode_responses=False, socket_connect_timeout=2.0)
            client.ping()
            self.redis_client = client

            # Check if RediSearch module is available
            try:
                self.redis_client.ft(self.index_name).info()
                self.has_search = True
                logger.info(f"RediSearch vector index '{self.index_name}' already active.")
            except ResponseError as e:
                if "unknown index" in str(e).lower() or "no such index" in str(e).lower():
                    self._create_search_index()
                else:
                    logger.warning(f"RediSearch info error: {e}. Falling back to in-memory index.")
                    self.has_search = False
            except Exception as e:
                logger.warning(f"RediSearch module not available ({e}). Using in-memory fallback.")
                self.has_search = False
        except Exception as e:
            logger.info(f"Redis unavailable for semantic cache ({e}). Using in-memory fallback.")
            self.redis_client = None
            self.has_search = False

    def _create_search_index(self) -> None:
        """Creates the RediSearch schema for vector KNN and metadata filtering."""
        if not self.redis_client or VectorField is None:
            return
        try:
            schema = (
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.dimension,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
                TagField("tenant_id"),
                TagField("user_id"),
                TagField("logical_model"),
                TagField("privacy_level"),
                TagField("execution_mode"),
                TagField("user_private"),
                TagField("quality_tier"),
                NumericField("quality"),
                NumericField("created_at"),
            )
            definition = IndexDefinition(
                prefix=[self.KEY_PREFIX],
                index_type=IndexType.HASH,
            )
            self.redis_client.ft(self.index_name).create_index(schema, definition=definition)
            self.has_search = True
            logger.info(f"Successfully created RediSearch vector index '{self.index_name}'.")
        except ResponseError as e:
            if "already exists" in str(e).lower():
                self.has_search = True
            else:
                logger.warning(f"Failed to create RediSearch index '{self.index_name}': {e}")
                self.has_search = False
        except Exception as e:
            logger.warning(f"Index creation exception: {e}")
            self.has_search = False

    def get(
        self,
        request: RequestContext,
        prompt_text: str,
        threshold: float | None = None,
    ) -> dict[str, Any] | None:
        """Finds closest cached entry exceeding the semantic similarity threshold

        using Redis Search KNN indexed query with structured metadata filters.
        Never performs KEYS or full-table scans.
        """
        if getattr(request, "no_cache", False):
            return None
        if request.privacy_level in (PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED):
            return None

        tau = threshold if threshold is not None else self.threshold
        req_quality = self.QUALITY_MAP.get(request.quality_requirement.lower(), 0.30)
        is_real_mode = (getattr(request, "execution_mode", None) or "hybrid").lower() == "real"
        user_priv_tag = "true" if getattr(request, "user_private", False) else "false"
        u_id = getattr(request, "user_id", None) or "user"

        try:
            query_vec = self.embedder.encode(prompt_text)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}; returning semantic miss.")
            return None

        # 1. Try Redis Search Vector Query
        if self.redis_client and self.has_search:
            try:
                escaped_tenant = _escape_tag_value(request.tenant_id)
                escaped_user = _escape_tag_value(u_id)
                escaped_priv = _escape_tag_value(request.privacy_level.value)
                escaped_upriv = _escape_tag_value(user_priv_tag)

                filter_clause = (
                    f"(@tenant_id:{{{escaped_tenant}}} "
                    f"@user_id:{{{escaped_user}}} "
                    f"@privacy_level:{{{escaped_priv}}} "
                    f"@user_private:{{{escaped_upriv}}})"
                )

                knn_expr = f"{filter_clause}=>[KNN 5 @embedding $query_vec AS vector_score]"
                q = (
                    Query(knn_expr)
                    .sort_by("vector_score")
                    .return_fields(
                        "prompt",
                        "response_content",
                        "model",
                        "replica_id",
                        "output_tokens",
                        "quality",
                        "privacy_level",
                        "tenant_id",
                        "user_id",
                        "is_simulated",
                        "user_private",
                        "vector_score",
                    )
                    .dialect(2)
                )

                vec_bytes = _vec_to_bytes(query_vec)
                res = self.redis_client.ft(self.index_name).search(q, query_params={"query_vec": vec_bytes})

                for doc in res.docs:
                    dist = float(doc.vector_score)
                    sim = max(-1.0, min(1.0, 1.0 - dist))

                    if sim < tau:
                        continue

                    doc_tenant = doc.tenant_id.decode() if isinstance(doc.tenant_id, bytes) else str(doc.tenant_id)
                    if doc_tenant != request.tenant_id:
                        continue

                    doc_user = doc.user_id.decode() if isinstance(doc.user_id, bytes) else str(doc.user_id)
                    if doc_user != u_id:
                        continue

                    doc_upriv = (
                        doc.user_private.decode() if isinstance(doc.user_private, bytes) else str(doc.user_private)
                    )
                    if (doc_upriv.lower() == "true") != getattr(request, "user_private", False):
                        continue

                    doc_priv = (
                        doc.privacy_level.decode() if isinstance(doc.privacy_level, bytes) else str(doc.privacy_level)
                    )
                    if doc_priv != request.privacy_level.value:
                        continue

                    doc_is_sim = str(doc.is_simulated).lower() in ("true", "1", "b'true'")
                    if is_real_mode and doc_is_sim:
                        continue

                    out_tokens = int(doc.output_tokens)
                    if request.max_tokens is not None and request.max_tokens > 0:
                        if out_tokens > request.max_tokens:
                            continue

                    q_score = float(doc.quality)
                    if q_score < req_quality:
                        continue

                    doc_prompt = doc.prompt.decode() if isinstance(doc.prompt, bytes) else str(doc.prompt)

                    passed_guard, guard_reason = SemanticGuards.check_guards(prompt_text, doc_prompt)
                    if not passed_guard:
                        logger.info(f"Semantic hit rejected by guard: {guard_reason}")
                        continue

                    doc_resp = (
                        doc.response_content.decode()
                        if isinstance(doc.response_content, bytes)
                        else str(doc.response_content)
                    )
                    doc_model = doc.model.decode() if isinstance(doc.model, bytes) else str(doc.model)
                    doc_replica = doc.replica_id.decode() if isinstance(doc.replica_id, bytes) else str(doc.replica_id)

                    return {
                        "content": doc_resp,
                        "model": doc_model,
                        "replica_id": doc_replica,
                        "output_tokens": out_tokens,
                        "similarity": round(sim, 4),
                        "cache_type": "semantic",
                        "matched_prompt": doc_prompt,
                    }

            except Exception as e:
                logger.warning(f"Redis Search query failed: {e}; falling back to in-memory search.")

        # 2. In-Memory Fallback
        best_entry: SemanticCacheEntry | None = None
        best_sim = -1.0

        for entry in self._memory_entries:
            if entry.tenant_id != request.tenant_id:
                continue
            if getattr(entry, "user_id", "user") != u_id:
                continue
            if getattr(entry, "user_private", False) != getattr(request, "user_private", False):
                continue
            if entry.privacy_level != request.privacy_level:
                continue
            if is_real_mode and entry.is_simulated:
                continue
            if request.max_tokens is not None and request.max_tokens > 0:
                if entry.output_tokens > request.max_tokens:
                    continue
            if entry.quality_score < req_quality:
                continue

            sim = self.embedder.cosine_similarity(query_vec, entry.vector)
            if sim >= tau and sim > best_sim:
                passed_guard, _ = SemanticGuards.check_guards(prompt_text, entry.prompt)
                if passed_guard:
                    best_sim = sim
                    best_entry = entry

        if best_entry:
            return {
                "content": best_entry.response_content,
                "model": best_entry.model,
                "replica_id": best_entry.replica_id,
                "output_tokens": best_entry.output_tokens,
                "similarity": round(best_sim, 4),
                "cache_type": "semantic",
                "matched_prompt": best_entry.prompt,
            }

        return None

    def put(
        self,
        request: RequestContext,
        prompt_text: str,
        response_content: str,
        model: str,
        replica_id: str,
        output_tokens: int,
        quality_score: float = 0.85,
        is_simulated: bool = False,
    ) -> str:
        """Stores a new response entry indexed by its semantic vector."""
        if getattr(request, "no_cache", False):
            return ""
        if request.privacy_level in (PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED):
            return ""

        vec = self.embedder.encode(prompt_text)
        entry_hash = hashlib.sha256(f"{request.tenant_id}:{prompt_text}:{time.time()}".encode()).hexdigest()[:16]
        key = f"{self.KEY_PREFIX}{request.tenant_id}:{entry_hash}"
        u_id = getattr(request, "user_id", None) or "user"
        upriv = getattr(request, "user_private", False)
        exec_mode = getattr(request, "execution_mode", "hybrid") or "hybrid"

        entry = SemanticCacheEntry(
            key=key,
            prompt=prompt_text,
            vector=vec,
            response_content=response_content,
            model=model,
            replica_id=replica_id,
            output_tokens=output_tokens,
            quality_score=quality_score,
            privacy_level=request.privacy_level,
            tenant_id=request.tenant_id,
            is_simulated=is_simulated,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens,
            user_private=upriv,
            user_id=u_id,
            logical_model=request.task_type or "general",
            execution_mode=exec_mode,
        )

        # Store in Redis Search Hash
        if self.redis_client:
            try:
                vec_bytes = _vec_to_bytes(vec)
                mapping = {
                    "embedding": vec_bytes,
                    "tenant_id": request.tenant_id,
                    "user_id": u_id,
                    "logical_model": request.task_type or "general",
                    "privacy_level": request.privacy_level.value,
                    "execution_mode": exec_mode,
                    "user_private": "true" if upriv else "false",
                    "quality_tier": request.quality_requirement or "medium",
                    "quality": float(quality_score),
                    "created_at": float(entry.created_at),
                    "prompt": prompt_text,
                    "response_content": response_content,
                    "model": model,
                    "replica_id": replica_id,
                    "output_tokens": int(output_tokens),
                    "temperature": float(entry.temperature),
                    "max_tokens": int(request.max_tokens) if request.max_tokens is not None else -1,
                    "is_simulated": "true" if is_simulated else "false",
                }
                self.redis_client.hset(key, mapping=mapping)
                self.redis_client.expire(key, int(self.default_ttl))
            except Exception as e:
                logger.warning(f"Failed to store entry in Redis semantic cache: {e}")

        # Always store in memory fallback list (bounded FIFO)
        self._memory_entries.append(entry)
        if len(self._memory_entries) > 500:
            self._memory_entries.pop(0)

        return key

    def clear(self, tenant_id: str | None = None) -> None:
        """Clears semantic cache entries safely."""
        if self.redis_client:
            try:
                pattern = f"{self.KEY_PREFIX}{tenant_id}:*" if tenant_id else f"{self.KEY_PREFIX}*"
                cursor = 0
                keys_to_del = []
                while True:
                    cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
                    if keys:
                        keys_to_del.extend(keys)
                    if cursor == 0:
                        break
                if keys_to_del:
                    self.redis_client.delete(*keys_to_del)
            except Exception as e:
                logger.debug(f"Redis clear failed: {e}")

        if tenant_id:
            self._memory_entries = [e for e in self._memory_entries if e.tenant_id != tenant_id]
        else:
            self._memory_entries.clear()
