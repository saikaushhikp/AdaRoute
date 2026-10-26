"""Tests for Week-5 Commitment 2: Dynamic Semantic Caching Engine (Milestone M10).

Verifies:
- Near-paraphrase semantic hit when similarity >= 0.95
- Dissimilar query semantic miss when similarity < 0.95
- Strict privacy allow-list isolation (public cache never satisfies internal_only)
- Strict cross-tenant isolation (Tenant A never hits Tenant B)
- Quality floor guard (low quality entry cannot satisfy high quality requirement)
- Real execution mode isolation (simulated entries never satisfy real requests)
"""

from __future__ import annotations

from adaroute.cache.semantic import LocalEmbeddingProvider, SemanticCache
from adaroute.schemas import PrivacyLevel, RequestContext


def test_tc_m10_01_embedding_cosine_similarity_properties():
    embedder = LocalEmbeddingProvider(dimension=128)

    # Identical text -> similarity == 1.0
    v1 = embedder.encode("what is the capital of France")
    v2 = embedder.encode("what is the capital of France")
    sim_identical = embedder.cosine_similarity(v1, v2)
    assert round(sim_identical, 4) == 1.0000

    # Near paraphrase -> high similarity >= 0.85
    v_paraphrase = embedder.encode("what is the capital of France?")
    sim_para = embedder.cosine_similarity(v1, v_paraphrase)
    assert sim_para >= 0.95

    # Completely different query -> low similarity < 0.50
    v_unrelated = embedder.encode("write a python script to implement quicksort")
    sim_unrelated = embedder.cosine_similarity(v1, v_unrelated)
    assert sim_unrelated < 0.50


def test_tc_m10_02_semantic_cache_hit_on_paraphrase():
    cache = SemanticCache(similarity_threshold=0.90)
    cache.clear()

    req = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="user",
    )

    # Store entry for original query
    prompt_orig = "what is the capital of France?"
    cache.put(
        request=req,
        prompt_text=prompt_orig,
        response_content="Paris is the capital of France.",
        model="cohere/command-r-08-2024",
        replica_id="cloud-direct",
        output_tokens=8,
        quality_score=0.88,
    )

    # Query with exact or slight variation
    prompt_var = "what is the capital of France"
    hit = cache.get(req, prompt_var, threshold=0.90)

    assert hit is not None
    assert hit["content"] == "Paris is the capital of France."
    assert hit["cache_type"] == "semantic"
    assert hit["similarity"] >= 0.90


def test_tc_m10_03_semantic_cache_miss_on_dissimilar_query():
    cache = SemanticCache(similarity_threshold=0.95)
    cache.clear()

    req = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="user",
    )

    cache.put(
        request=req,
        prompt_text="what is the capital of France?",
        response_content="Paris is the capital of France.",
        model="cohere/command-r-08-2024",
        replica_id="cloud-direct",
        output_tokens=8,
        quality_score=0.88,
    )

    # Query with completely unrelated prompt
    prompt_diff = "explain the theory of general relativity"
    miss = cache.get(req, prompt_diff)

    assert miss is None


def test_tc_m10_04_semantic_cache_privacy_isolation():
    cache = SemanticCache(similarity_threshold=0.90)
    cache.clear()

    # Public request cached
    req_pub = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="user",
    )
    cache.put(
        request=req_pub,
        prompt_text="what is the capital of France?",
        response_content="Paris is the capital of France.",
        model="cohere/command-r-08-2024",
        replica_id="cloud-direct",
        output_tokens=8,
        quality_score=0.88,
    )

    # Internal confidential request with same question must MISS (no privacy leak)
    req_internal = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.INTERNAL_ONLY,
        tenant_id="user",
    )
    res = cache.get(req_internal, "what is the capital of France?")
    assert res is None


def test_tc_m10_05_semantic_cache_tenant_isolation():
    cache = SemanticCache(similarity_threshold=0.90)
    cache.clear()

    # Cached by tenant_A
    req_a = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="tenant_A",
    )
    cache.put(
        request=req_a,
        prompt_text="summarize quarterly earnings report",
        response_content="Company revenue grew 12% year-over-year.",
        model="cohere/command-r-08-2024",
        replica_id="cloud-direct",
        output_tokens=10,
    )

    # Queried by tenant_B -> must MISS
    req_b = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="tenant_B",
    )
    res = cache.get(req_b, "summarize quarterly earnings report")
    assert res is None


def test_tc_m10_06_semantic_cache_quality_floor_guard():
    cache = SemanticCache(similarity_threshold=0.90)
    cache.clear()

    req = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        quality_requirement="high",  # Floor = 0.85
        tenant_id="user",
    )

    # Store entry with low quality (0.65)
    cache.put(
        request=req,
        prompt_text="what is the capital of France?",
        response_content="Paris",
        model="local-model",
        replica_id="local-r1",
        output_tokens=2,
        quality_score=0.65,  # Below 0.85
    )

    # High quality request must reject the 0.65 quality entry
    hit = cache.get(req, "what is the capital of France?")
    assert hit is None


def test_tc_m10_07_semantic_cache_simulation_isolation():
    cache = SemanticCache(similarity_threshold=0.90)
    cache.clear()

    req_sim = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        execution_mode="simulation",
        tenant_id="user",
    )

    # Store simulated entry
    cache.put(
        request=req_sim,
        prompt_text="hello world prompt",
        response_content="Simulated response output",
        model="mock-model",
        replica_id="mock-rep",
        output_tokens=5,
        is_simulated=True,
    )

    # Real mode request must not accept simulated cache entry
    req_real = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        execution_mode="real",
        tenant_id="user",
    )
    hit = cache.get(req_real, "hello world prompt")
    assert hit is None
