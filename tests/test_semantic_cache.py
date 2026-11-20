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

    # Medium quality request (floor 0.30): accepts 0.65
    req_med = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        quality_requirement="medium",  # Floor = 0.30
        tenant_id="user",
    )
    hit_med = cache.get(req_med, "what is the capital of France?")
    assert hit_med is not None

    # Entry with quality 0.20 is rejected for medium requirement (floor 0.30)
    cache.put(
        request=req_med,
        prompt_text="what is quantum computing?",
        response_content="Computing with qubits",
        model="local-model",
        replica_id="local-r1",
        output_tokens=3,
        quality_score=0.20,  # Below 0.30
    )
    hit_sub_floor = cache.get(req_med, "what is quantum computing?")
    assert hit_sub_floor is None


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


def test_tc_m10_08_semantic_cache_adversarial_negation_guard():
    cache = SemanticCache(similarity_threshold=0.80)
    cache.clear()

    req = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="user",
    )

    cache.put(
        request=req,
        prompt_text="the production database is running normally",
        response_content="DB status: ONLINE and HEALTHY.",
        model="local-model",
        replica_id="r1",
        output_tokens=6,
    )

    # Adversarial near-miss with negation: must NOT hit!
    neg_prompt = "the production database is not running normally"
    hit = cache.get(req, neg_prompt)
    assert hit is None, "Negation polarity mismatch must be rejected by semantic guards"


def test_tc_m10_09_semantic_cache_adversarial_numeric_guard():
    cache = SemanticCache(similarity_threshold=0.80)
    cache.clear()

    req = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="user",
    )

    cache.put(
        request=req,
        prompt_text="forward metrics traffic to port 8000",
        response_content="Forwarding configured to port 8000.",
        model="local-model",
        replica_id="r1",
        output_tokens=6,
    )

    # Adversarial near-miss with different number: must NOT hit!
    diff_num_prompt = "forward metrics traffic to port 9000"
    hit = cache.get(req, diff_num_prompt)
    assert hit is None, "Numeric literal mismatch must be rejected by semantic guards"


def test_tc_m10_10_semantic_cache_adversarial_format_and_lang_guard():
    cache = SemanticCache(similarity_threshold=0.80)
    cache.clear()

    req = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="user",
    )

    cache.put(
        request=req,
        prompt_text="generate configuration schema as json",
        response_content='{"schema": "v1"}',
        model="local-model",
        replica_id="r1",
        output_tokens=6,
    )

    # Conflicting format entity: YAML vs JSON
    hit_yaml = cache.get(req, "generate configuration schema as yaml")
    assert hit_yaml is None, "Conflicting format constraint must be rejected"


def test_tc_m10_11_semantic_cache_cross_user_isolation():
    cache = SemanticCache(similarity_threshold=0.85)
    cache.clear()

    # Cached by user_1
    req_u1 = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="tenant_X",
        user_id="user_1",
    )
    cache.put(
        request=req_u1,
        prompt_text="display account balance and history",
        response_content="Account balance is $4,500.00.",
        model="local-model",
        replica_id="r1",
        output_tokens=7,
    )

    # Queried by user_2 in the same tenant -> must MISS
    req_u2 = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="tenant_X",
        user_id="user_2",
    )
    hit_u2 = cache.get(req_u2, "display account balance and history")
    assert hit_u2 is None, "Cross-user cache hit strictly forbidden"


def test_tc_m10_12_semantic_cache_max_tokens_bound():
    cache = SemanticCache(similarity_threshold=0.85)
    cache.clear()

    req = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="user",
        max_tokens=20,  # Request allows max 20 tokens
    )

    # Entry generated with 80 tokens
    cache.put(
        request=req,
        prompt_text="explain distributed consensus algorithms",
        response_content="Long text..." * 20,
        model="local-model",
        replica_id="r1",
        output_tokens=80,
    )

    # Must MISS because cached output exceeds client max_tokens
    hit = cache.get(req, "explain distributed consensus algorithms")
    assert hit is None, "Cached response exceeding max_tokens must be rejected"


def test_tc_m10_13_semantic_cache_redis_vector_search_live():
    """Live verification against Redis Search (redis-stack-server:7.4.0-v8)."""
    import os

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache = SemanticCache(similarity_threshold=0.85, redis_url=redis_url)

    if not cache.has_search:
        return  # Skip if environment lacks RediSearch

    cache.clear()

    req = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="redis_test_tenant",
        user_id="redis_user",
    )

    cache.put(
        request=req,
        prompt_text="how do I initialize a git repository",
        response_content="Run 'git init' in your directory.",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        replica_id="vllm-r1",
        output_tokens=8,
        quality_score=0.85,
    )

    # Query with identical prompt
    hit = cache.get(req, "how do I initialize a git repository")
    assert hit is not None
    assert hit["content"] == "Run 'git init' in your directory."
    assert hit["similarity"] >= 0.85
    assert hit["cache_type"] == "semantic"


def test_tc_m10_14_fastembed_real_embedding_paraphrase():
    """Verification using real CPU ONNX FastEmbedProvider (sentence-transformers/all-MiniLM-L6-v2)."""
    from adaroute.cache.semantic import FastEmbedProvider

    embedder = FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2", dimension=384)
    cache = SemanticCache(similarity_threshold=0.85, embedder=embedder)
    cache.clear()

    req = RequestContext(
        task_type="coding",
        prompt_tokens=12,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="fastembed_tenant",
    )

    cache.put(
        request=req,
        prompt_text="how do I reverse a string in python",
        response_content="Use slicing: text[::-1]",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        replica_id="vllm-r1",
        output_tokens=6,
    )

    # Genuine paraphrase query
    paraphrase = "python code to reverse string"
    hit = cache.get(req, paraphrase, threshold=0.80)
    assert hit is not None
    assert hit["content"] == "Use slicing: text[::-1]"
    assert hit["similarity"] >= 0.80


def test_tc_m10_15_semantic_cache_failure_and_outage_safety():
    """Embedding or Redis failures must gracefully yield observable miss without raising exceptions."""

    class BrokenEmbedder:
        dimension = 384
        name = "broken"

        def encode(self, text: str):
            raise RuntimeError("Embedder service unreachable")

        def cosine_similarity(self, v1, v2):
            return 0.0

    cache = SemanticCache(embedder=BrokenEmbedder())
    req = RequestContext(
        task_type="general",
        prompt_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="user",
    )

    # Get must not raise an unhandled exception
    try:
        hit = cache.get(req, "sample query")
        assert hit is None
    except Exception as e:
        assert False, f"cache.get() raised unhandled exception on embedder outage: {e}"
