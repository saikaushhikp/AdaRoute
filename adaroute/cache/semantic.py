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

# Canonical provider provenance definitions
CANONICAL_LOCAL_PROVIDERS = frozenset({"local", "vllm", "local_vllm", "local_ollama", "ollama", "mock-local"})
CANONICAL_CLOUD_PROVIDERS = frozenset({"gemini", "google", "groq", "cohere", "openai", "litellm", "mock-external"})

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

    CONFLICTING_INSTRUCTION_PAIRS = [
        (
            {"summarize", "summary", "brief", "briefly", "short"},
            {"expand", "elaborate", "detail", "detailed", "comprehensive", "long"},
        ),
        ({"encode"}, {"decode"}),
        ({"encrypt"}, {"decrypt"}),
        ({"compress"}, {"decompress"}),
    ]

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

        # 5. Conflicting Directives Guard
        for group_a, group_b in cls.CONFLICTING_INSTRUCTION_PAIRS:
            if (q_words & group_a and c_words & group_b) or (q_words & group_b and c_words & group_a):
                return False, "Conflicting instruction directives detected between query and cached entry"

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
    origin_provider: str = "unknown"
    finish_reason: str = "unknown"
    policy_name: str = "adaptive"
    policy_version: str = "v4.0"
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
            "privacy_level": self.privacy_level.value
            if hasattr(self.privacy_level, "value")
            else str(self.privacy_level),
            "tenant_id": self.tenant_id,
            "is_simulated": self.is_simulated,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "user_private": self.user_private,
            "user_id": self.user_id,
            "logical_model": self.logical_model,
            "execution_mode": self.execution_mode,
            "origin_provider": self.origin_provider,
            "finish_reason": self.finish_reason,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
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
            temperature=float(data["temperature"]) if data.get("temperature") is not None else 0.7,
            max_tokens=int(data["max_tokens"])
            if data.get("max_tokens") is not None and int(data["max_tokens"]) >= 0
            else None,
            user_private=bool(data.get("user_private", False)),
            user_id=str(data.get("user_id", "user")),
            logical_model=str(data.get("logical_model", "general")),
            execution_mode=str(data.get("execution_mode", "hybrid")),
            origin_provider=str(data.get("origin_provider", "unknown")),
            finish_reason=str(data.get("finish_reason", "unknown")),
            policy_name=str(data.get("policy_name", "adaptive")),
            policy_version=str(data.get("policy_version", "v4.0")),
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

    INDEX_VERSION = "v3"
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
                TagField("origin_provider"),
                TagField("finish_reason"),
                TagField("policy_name"),
                TagField("policy_version"),
                NumericField("temperature"),
                NumericField("max_tokens"),
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

    @staticmethod
    def _is_entry_eligible(
        entry_data: dict[str, Any],
        request: RequestContext,
        prompt_text: str,
        is_real_mode: bool,
        u_id: str,
        req_quality: float,
        req_task: str,
        explicit_model: str | None,
        req_temp: float,
    ) -> tuple[bool, str | None]:
        """Strict equivalence & scope check enforced identically across Redis Search and in-memory fallback.

        Enforces:
        1. Tenant and User isolation
        2. Privacy level and user_private toggle matching
        3. Canonical local vs cloud provenance (unknown fails closed)
        4. Real vs Simulated execution isolation
        5. Logical capability or explicit model ID matching
        6. Generation options: temperature (including deterministic 0.0)
        7. Response limits & completion status (truncated finish_reason='length' rejected)
        8. Quality floor satisfaction
        9. Semantic guards (negation, numeric literals, format, language, instruction conflicts)
        """
        # 1. Tenant & User partition isolation (fail-closed on missing)
        entry_tenant = str(entry_data.get("tenant_id") or "").strip()
        req_tenant = str(request.tenant_id or "user").strip()
        if not entry_tenant or entry_tenant != req_tenant:
            return False, "Tenant mismatch"

        entry_user = str(entry_data.get("user_id") or "").strip()
        req_user = str(u_id).strip()
        if not entry_user or entry_user != req_user:
            return False, "User partition mismatch"

        # 2. Privacy & Private toggle (fail-closed on missing)
        entry_upriv_raw = entry_data.get("user_private")
        if entry_upriv_raw is None or entry_upriv_raw == "":
            return False, "Missing user_private metadata"
        entry_upriv = str(entry_upriv_raw).lower() in ("true", "1", "b'true'")
        req_upriv = bool(getattr(request, "user_private", False))
        if entry_upriv != req_upriv:
            return False, "user_private toggle mismatch"

        entry_priv = str(entry_data.get("privacy_level") or "").strip().lower()
        req_priv = str(request.privacy_level.value).strip().lower()
        if not entry_priv or entry_priv != req_priv:
            return False, "privacy_level mismatch"

        # 3. Canonical Local/Cloud Provenance validation (unknown fails closed)
        origin_prov = str(entry_data.get("origin_provider") or "").strip().lower()
        if not origin_prov or origin_prov == "unknown":
            return False, "Unknown or missing origin provider rejected"

        if req_upriv:
            if origin_prov not in CANONICAL_LOCAL_PROVIDERS or origin_prov in CANONICAL_CLOUD_PROVIDERS:
                return False, f"Provider '{origin_prov}' ineligible for user_private request"

        if request.privacy_level in (PrivacyLevel.INTERNAL_ONLY, PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED):
            if origin_prov not in CANONICAL_LOCAL_PROVIDERS or origin_prov in CANONICAL_CLOUD_PROVIDERS:
                return False, f"Provider '{origin_prov}' ineligible for internal/restricted privacy"

        # 4. Real vs Simulated execution isolation
        is_sim = str(entry_data.get("is_simulated", "")).lower() in ("true", "1", "b'true'")
        if is_real_mode and (
            is_sim or origin_prov in ("mock", "mock-local", "mock-external", "unknown", "") or not origin_prov
        ):
            return False, "Simulated or unverified origin rejected for real request"

        # 5. Policy and effective policy version matching
        req_policy = getattr(request, "policy", None) or getattr(request, "policy_name", None) or "adaptive"
        entry_policy = entry_data.get("policy_name")
        if req_policy and entry_policy:
            if str(entry_policy).strip().lower() != str(req_policy).strip().lower():
                return False, f"Policy mismatch (req={req_policy} vs cached={entry_policy})"

        req_pol_ver = getattr(request, "policy_version", "v4.0") or "v4.0"
        entry_pol_ver = entry_data.get("policy_version")
        if entry_pol_ver and str(entry_pol_ver).strip() != str(req_pol_ver).strip():
            return False, f"Policy version mismatch (req={req_pol_ver} vs cached={entry_pol_ver})"

        # 6. Logical Model (Capability) or Explicit Model Matching
        entry_model = str(entry_data.get("model", "")).strip()
        if explicit_model:
            if entry_model.lower() != explicit_model.lower():
                return False, f"Explicit model mismatch (expected {explicit_model}, got {entry_model})"
        else:
            cached_logical = str(entry_data.get("logical_model", "")).strip().lower()
            if not cached_logical or cached_logical != req_task.lower():
                return False, f"Capability mismatch (expected {req_task}, got {cached_logical})"

        # 7. Strict Temperature Matching (eliminating drift tolerance)
        cached_temp_raw = entry_data.get("temperature")
        if cached_temp_raw is None:
            return False, "Missing temperature in cached entry"
        try:
            cached_temp = float(cached_temp_raw)
        except (ValueError, TypeError):
            return False, "Invalid cached temperature"

        if abs(cached_temp - req_temp) > 1e-6:
            return False, f"Temperature mismatch (req={req_temp} vs cached={cached_temp})"

        # 8. Output Limits, Original max_tokens Budget & Completion Status
        cached_finish = str(entry_data.get("finish_reason") or "").strip().lower()
        if not cached_finish or cached_finish in ("length", "unknown", "incomplete"):
            return False, f"Unsatisfactory finish reason '{cached_finish}' cannot satisfy complete request"

        # Check original max_tokens generation budget vs requested max_tokens
        cached_max_tok = entry_data.get("max_tokens")
        if cached_max_tok is not None:
            try:
                c_mt = int(cached_max_tok)
                if c_mt > 0 and request.max_tokens is not None and request.max_tokens > 0:
                    if request.max_tokens > c_mt:
                        return (
                            False,
                            f"Requested max_tokens {request.max_tokens} exceeds cached max_tokens budget {c_mt}",
                        )
            except (ValueError, TypeError):
                pass

        try:
            out_tokens = int(entry_data.get("output_tokens", 0))
        except (ValueError, TypeError):
            out_tokens = 0

        if request.max_tokens is not None and request.max_tokens > 0:
            if out_tokens > request.max_tokens:
                return False, f"Output token count {out_tokens} exceeds requested max_tokens {request.max_tokens}"

        # 9. Quality Floor
        try:
            q_score = float(entry_data.get("quality", entry_data.get("quality_score", 0.0)))
        except (ValueError, TypeError):
            q_score = 0.0

        if q_score < req_quality:
            return False, f"Quality {q_score} below required floor {req_quality}"

        # 10. Conservative Semantic Guards
        cached_prompt = str(entry_data.get("prompt", "")).strip()
        passed_guard, guard_reason = SemanticGuards.check_guards(prompt_text, cached_prompt)
        if not passed_guard:
            return False, f"Guard rejected: {guard_reason}"

        return True, None

    def get(
        self,
        request: RequestContext,
        prompt_text: str,
        threshold: float | None = None,
    ) -> dict[str, Any] | None:
        """Finds closest cached entry exceeding the semantic similarity threshold

        using Redis Search KNN indexed query with structured metadata filters.
        Enforces identical scope, provenance, generation option, and guard checks on both paths.
        Never performs KEYS or full-table scans.
        """
        if getattr(request, "no_cache", False):
            return None
        if request.privacy_level in (PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED):
            return None

        # Semantic matching is strictly restricted to safe standalone single-turn requests.
        # Multi-turn context or system prompt variations cannot safely be matched by prompt embedding alone.
        msgs = getattr(request, "messages", None)
        if msgs and (
            len(msgs) > 1
            or any((m.role if hasattr(m, "role") else m.get("role", "")).lower() == "system" for m in msgs)
        ):
            return None

        tau = threshold if threshold is not None else self.threshold
        req_quality = self.QUALITY_MAP.get(request.quality_requirement.lower(), 0.30)
        is_real_mode = (getattr(request, "execution_mode", None) or "hybrid").lower() == "real"
        user_priv_tag = "true" if getattr(request, "user_private", False) else "false"
        u_id = getattr(request, "user_id", None) or "user"
        req_task = getattr(request, "task_type", None) or getattr(request, "model", None) or "general"
        req_model_str = getattr(request, "model", None)
        explicit_model = req_model_str if req_model_str and ("/" in req_model_str or ":" in req_model_str) else None
        req_temp = request.temperature if getattr(request, "temperature", None) is not None else 0.7

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
                escaped_task = _escape_tag_value(req_task)

                filter_clause = (
                    f"(@tenant_id:{{{escaped_tenant}}} "
                    f"@user_id:{{{escaped_user}}} "
                    f"@privacy_level:{{{escaped_priv}}} "
                    f"@user_private:{{{escaped_upriv}}} "
                    f"@logical_model:{{{escaped_task}}})"
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
                        "logical_model",
                        "temperature",
                        "max_tokens",
                        "origin_provider",
                        "finish_reason",
                        "policy_name",
                        "policy_version",
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

                    # Unpack fields safely (bytes or str) - fail closed on absent values
                    def _decode(val: Any, default: str = "") -> str:
                        if val is None:
                            return default
                        return val.decode("utf-8", errors="replace") if isinstance(val, bytes) else str(val)

                    entry_dict = {
                        "tenant_id": _decode(getattr(doc, "tenant_id", None), ""),
                        "user_id": _decode(getattr(doc, "user_id", None), ""),
                        "privacy_level": _decode(getattr(doc, "privacy_level", None), ""),
                        "user_private": _decode(getattr(doc, "user_private", None), ""),
                        "logical_model": _decode(getattr(doc, "logical_model", None), ""),
                        "model": _decode(getattr(doc, "model", None), ""),
                        "origin_provider": _decode(getattr(doc, "origin_provider", None), "unknown"),
                        "finish_reason": _decode(getattr(doc, "finish_reason", None), "unknown"),
                        "is_simulated": getattr(doc, "is_simulated", None),
                        "temperature": getattr(doc, "temperature", None),
                        "max_tokens": getattr(doc, "max_tokens", None),
                        "policy_name": _decode(getattr(doc, "policy_name", None), ""),
                        "policy_version": _decode(getattr(doc, "policy_version", None), ""),
                        "output_tokens": getattr(doc, "output_tokens", 0),
                        "quality": getattr(doc, "quality", 0.0),
                        "prompt": _decode(getattr(doc, "prompt", None), ""),
                    }

                    eligible, reason = self._is_entry_eligible(
                        entry_data=entry_dict,
                        request=request,
                        prompt_text=prompt_text,
                        is_real_mode=is_real_mode,
                        u_id=u_id,
                        req_quality=req_quality,
                        req_task=req_task,
                        explicit_model=explicit_model,
                        req_temp=req_temp,
                    )
                    if not eligible:
                        logger.debug(f"Redis Search semantic candidate rejected: {reason}")
                        continue

                    doc_resp = _decode(getattr(doc, "response_content", None), "")
                    doc_model = entry_dict["model"]
                    doc_replica = _decode(getattr(doc, "replica_id", None), "cache")
                    out_tokens = int(entry_dict["output_tokens"])

                    return {
                        "content": doc_resp,
                        "model": doc_model,
                        "replica_id": doc_replica,
                        "output_tokens": out_tokens,
                        "similarity": round(sim, 4),
                        "cache_type": "semantic",
                        "matched_prompt": entry_dict["prompt"],
                        "provider": "cache",
                        "finish_reason": entry_dict["finish_reason"],
                    }

            except Exception as e:
                logger.warning(f"Redis Search query failed: {e}; falling back to in-memory search.")

        # 2. In-Memory Fallback
        best_entry: SemanticCacheEntry | None = None
        best_sim = -1.0

        for entry in self._memory_entries:
            sim = self.embedder.cosine_similarity(query_vec, entry.vector)
            if sim < tau or sim <= best_sim:
                continue

            entry_dict = {
                "tenant_id": entry.tenant_id,
                "user_id": getattr(entry, "user_id", "user"),
                "privacy_level": getattr(entry.privacy_level, "value", str(entry.privacy_level)),
                "user_private": getattr(entry, "user_private", False),
                "logical_model": getattr(entry, "logical_model", "general"),
                "model": entry.model,
                "origin_provider": getattr(entry, "origin_provider", "unknown"),
                "finish_reason": getattr(entry, "finish_reason", "unknown"),
                "is_simulated": entry.is_simulated,
                "temperature": entry.temperature,
                "output_tokens": entry.output_tokens,
                "max_tokens": getattr(entry, "max_tokens", None),
                "policy_name": getattr(entry, "policy_name", "adaptive"),
                "policy_version": getattr(entry, "policy_version", "v4.0"),
                "quality": entry.quality_score,
                "prompt": entry.prompt,
            }

            eligible, reason = self._is_entry_eligible(
                entry_data=entry_dict,
                request=request,
                prompt_text=prompt_text,
                is_real_mode=is_real_mode,
                u_id=u_id,
                req_quality=req_quality,
                req_task=req_task,
                explicit_model=explicit_model,
                req_temp=req_temp,
            )
            if not eligible:
                continue

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
                "provider": "cache",
                "finish_reason": best_entry.finish_reason,
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
        origin_provider: str = "local",
        finish_reason: str = "stop",
    ) -> str:
        """Stores a new response entry indexed by its semantic vector."""
        if getattr(request, "no_cache", False):
            return ""
        if request.privacy_level in (PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED):
            return ""

        # Inadequate completion status: truncated responses must NOT be indexed in semantic cache
        if finish_reason == "length":
            return ""

        vec = self.embedder.encode(prompt_text)
        entry_hash = hashlib.sha256(f"{request.tenant_id}:{prompt_text}:{time.time()}".encode()).hexdigest()[:16]
        key = f"{self.KEY_PREFIX}{request.tenant_id}:{entry_hash}"
        u_id = getattr(request, "user_id", None) or "user"
        upriv = getattr(request, "user_private", False)
        exec_mode = getattr(request, "execution_mode", "hybrid") or "hybrid"
        temp_val = request.temperature if getattr(request, "temperature", None) is not None else 0.7
        pol_name = getattr(request, "policy_name", "adaptive") or "adaptive"
        pol_ver = getattr(request, "policy_version", "v4.0") or "v4.0"

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
            temperature=temp_val,
            max_tokens=request.max_tokens,
            user_private=upriv,
            user_id=u_id,
            logical_model=request.task_type or "general",
            execution_mode=exec_mode,
            origin_provider=origin_provider,
            finish_reason=finish_reason,
            policy_name=pol_name,
            policy_version=pol_ver,
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
                    "temperature": float(temp_val),
                    "max_tokens": int(request.max_tokens) if request.max_tokens is not None else -1,
                    "is_simulated": "true" if is_simulated else "false",
                    "origin_provider": origin_provider,
                    "finish_reason": finish_reason,
                    "policy_name": pol_name,
                    "policy_version": pol_ver,
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
