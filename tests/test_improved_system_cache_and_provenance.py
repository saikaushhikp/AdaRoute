import pytest
from httpx import ASGITransport, AsyncClient

from adaroute.api import app, cache
from adaroute.cache.exact import ExactCache
from adaroute.cache.redis_backend import CacheEntry
from adaroute.cache.semantic import SemanticCache, SemanticGuards
from adaroute.schemas import PrivacyLevel, RequestContext


@pytest.fixture(autouse=True)
def clean_cache():
    cache.clear()


def test_TC_M4_22_vllm_private_exact_hit_reproduction():
    """Reproduces and tests that private vLLM responses pass exact provenance check."""
    test_cache = ExactCache(backend="memory")
    req = RequestContext(
        task_type="general",
        prompt_tokens=10,
        estimated_output_tokens=20,
        quality_requirement="low",
        latency_slo=2.0,
        privacy_level=PrivacyLevel.PUBLIC,
        budget=1.0,
        tenant_id="report-cache-216",
        user_id="review",
        request_id="req-1",
        user_private=True,
        execution_mode="real",
    )
    entry = CacheEntry(
        key="test_key",
        response={
            "content": "Paris is the capital of France.",
            "model": "unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
            "replica_id": "vllm-gpu-gguf-qwen3-r2",
            "output_tokens": 15,
            "provider": "vllm",
            "finish_reason": "stop",
        },
        origin_model="unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
        origin_provider="vllm",
        origin_privacy=PrivacyLevel.PUBLIC,
        origin_quality=0.44,
        is_simulated=False,
        source="vllm_real",
        finish_reason="stop",
    )

    is_valid = test_cache._validate_provenance(entry, req)
    assert is_valid is True, "Legitimate local vLLM response must be eligible for private exact reuse"


def test_TC_M4_23_cloud_and_unknown_provenance_fails_closed_for_private_and_restricted():
    """Cloud-origin or unknown-provenance entries MUST fail closed for private or internal_only requests."""
    test_cache = ExactCache(backend="memory")
    req_private = RequestContext(
        task_type="general",
        quality_requirement="low",
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="t1",
        user_id="u1",
        user_private=True,
        execution_mode="real",
    )
    req_internal = RequestContext(
        task_type="general",
        quality_requirement="low",
        privacy_level=PrivacyLevel.INTERNAL_ONLY,
        tenant_id="t1",
        user_id="u1",
        user_private=False,
        execution_mode="real",
    )

    # Cloud provider entries
    for cloud_prov in ["gemini", "groq", "cohere", "openai", "litellm", "mock-external"]:
        entry = CacheEntry(
            key="k",
            response={"content": "cloud answer"},
            origin_model="cloud-model",
            origin_provider=cloud_prov,
            origin_privacy=PrivacyLevel.PUBLIC,
            origin_quality=0.9,
            is_simulated=False,
            source="cloud_real",
        )
        assert test_cache._validate_provenance(entry, req_private) is False, (
            f"Cloud provider {cloud_prov} must fail closed for private request"
        )
        assert test_cache._validate_provenance(entry, req_internal) is False, (
            f"Cloud provider {cloud_prov} must fail closed for internal_only request"
        )

    # Unknown provider
    unknown_entry = CacheEntry(
        key="k_unknown",
        response={"content": "mystery"},
        origin_model="unknown-model",
        origin_provider="unverified_third_party",
        origin_privacy=PrivacyLevel.PUBLIC,
        origin_quality=0.9,
        is_simulated=False,
        source="unknown",
    )
    assert test_cache._validate_provenance(unknown_entry, req_private) is False, (
        "Unknown provenance must fail closed for private request"
    )
    assert test_cache._validate_provenance(unknown_entry, req_internal) is False, (
        "Unknown provenance must fail closed for internal_only request"
    )


def test_TC_M4_24_simulated_entry_rejected_in_real_mode():
    """Simulated entries must never satisfy execution_mode='real'."""
    test_cache = ExactCache(backend="memory")
    req_real = RequestContext(
        task_type="general",
        quality_requirement="low",
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="t1",
        user_id="u1",
        execution_mode="real",
    )
    sim_entry = CacheEntry(
        key="k_sim",
        response={"content": "sim answer"},
        origin_model="unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
        origin_provider="vllm",
        origin_privacy=PrivacyLevel.PUBLIC,
        origin_quality=0.8,
        is_simulated=True,
        source="simulated_vllm",
    )
    assert test_cache._validate_provenance(sim_entry, req_real) is False, (
        "Simulated entry must be rejected in real mode"
    )


def test_TC_M4_25_temperature_zero_preserved_without_collapsing():
    """temperature=0.0 must not be collapsed to 0.7 through truthiness and must produce distinct keys."""
    test_cache = ExactCache(backend="memory")
    key_zero = test_cache.compute_key(
        tenant_id="t1",
        user_id="u1",
        logical_model="general",
        model_version="1.0",
        prompt="deterministic query",
        temperature=0.0,
    )
    key_default = test_cache.compute_key(
        tenant_id="t1",
        user_id="u1",
        logical_model="general",
        model_version="1.0",
        prompt="deterministic query",
        temperature=0.7,
    )
    assert key_zero != key_default, "temperature=0.0 and temperature=0.7 must produce different cache keys"


def test_TC_M4_26_semantic_cache_checks_capability_and_options():
    """SemanticCache must enforce scope, task_type, temperature, and finish_reason."""
    sem_cache = SemanticCache(similarity_threshold=0.85)

    req_coding = RequestContext(
        task_type="coding",
        quality_requirement="medium",
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="t1",
        user_id="u1",
        temperature=0.0,
        max_tokens=100,
    )

    # Put a coding answer
    sem_cache.put(
        request=req_coding,
        prompt_text="Write a function to compute factorial in python",
        response_content="def factorial(n): ...",
        model="unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M",
        replica_id="vllm-r1",
        output_tokens=30,
        quality_score=0.85,
        is_simulated=False,
        origin_provider="vllm",
        finish_reason="stop",
    )

    # Query with same capability and options -> HIT
    hit = sem_cache.get(req_coding, "Write a function to calculate factorial in python")
    assert hit is not None, "Similar prompt with same capability and temperature must hit"
    assert hit["cache_type"] == "semantic"

    # Query with different capability (reasoning) -> MISS
    req_reasoning = RequestContext(
        task_type="reasoning",
        quality_requirement="medium",
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="t1",
        user_id="u1",
        temperature=0.0,
        max_tokens=100,
    )
    miss_cap = sem_cache.get(req_reasoning, "Write a function to calculate factorial in python")
    assert miss_cap is None, "Mismatched capability/task_type must miss"

    # Query with different temperature (0.9 vs 0.0) -> MISS
    req_creative = RequestContext(
        task_type="coding",
        quality_requirement="medium",
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="t1",
        user_id="u1",
        temperature=0.9,
        max_tokens=100,
    )
    miss_temp = sem_cache.get(req_creative, "Write a function to calculate factorial in python")
    assert miss_temp is None, "Mismatched temperature must miss"


def test_TC_M4_27_finish_reason_length_declines_semantic_indexing():
    """A response truncated by output token limit (finish_reason='length') must NOT be indexed in semantic cache."""
    sem_cache = SemanticCache(similarity_threshold=0.80)
    req = RequestContext(
        task_type="general",
        quality_requirement="low",
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="t1",
        user_id="u1",
    )
    # Attempt to put truncated response
    key = sem_cache.put(
        request=req,
        prompt_text="Explain quantum mechanics",
        response_content="Quantum mechanics is a fundamental theory in physics that",
        model="unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
        replica_id="vllm-r1",
        output_tokens=10,
        quality_score=0.44,
        is_simulated=False,
        origin_provider="vllm",
        finish_reason="length",
    )
    # Should decline indexing
    assert key == "" or len(sem_cache._memory_entries) == 0, "Truncated response must not be indexed in semantic cache"
    hit = sem_cache.get(req, "Explain quantum mechanics theory")
    assert hit is None, "Truncated response must never be returned as a semantic hit"


def test_TC_M4_28_adversarial_semantic_guards():
    """Test semantic guards against negation, numeric, format, and entity changes."""
    # 1. Negation mismatch
    pass_neg, reason_neg = SemanticGuards.check_guards("is liquid nitrogen safe?", "is liquid nitrogen not safe?")
    assert pass_neg is False
    assert "Negation" in reason_neg

    # 2. Number mismatch
    pass_num, reason_num = SemanticGuards.check_guards("Report experiment 216", "Report experiment 217")
    assert pass_num is False
    assert "Numeric" in reason_num

    # 3. Format mismatch
    pass_fmt, reason_fmt = SemanticGuards.check_guards("output as JSON format", "output as YAML format")
    assert pass_fmt is False
    assert "Format" in reason_fmt

    # 4. Language mismatch
    pass_lang, reason_lang = SemanticGuards.check_guards("write quicksort in Rust", "write quicksort in Python")
    assert pass_lang is False
    assert "Language" in reason_lang


@pytest.mark.asyncio
async def test_TC_M4_29_endpoint_exact_repeat_for_private_vllm():
    """Endpoint test: private real vLLM request repeated identically must hit exact (cache_type='exact')."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "model": "general",
            "messages": [{"role": "user", "content": "Report experiment 216: What is the capital of France?"}],
            "tenant_id": "report-cache-216",
            "user_id": "review",
            "user_private": True,
            "quality_requirement": "low",
            "max_tokens": 40,
            "execution_mode": "hybrid",
        }
        # 1. Cold miss
        res1 = await client.post("/v1/chat/completions", json=payload)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["routing_metadata"]["cache_hit"] is False

        # 2. Identical repeat MUST be an EXACT hit
        res2 = await client.post("/v1/chat/completions", json=payload)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["routing_metadata"]["cache_hit"] is True
        assert data2["routing_metadata"]["cache_type"] == "exact", "Identical repeat must hit exact, not semantic"
        assert data2["choices"][0]["message"]["content"] == data1["choices"][0]["message"]["content"]

        # Default diagnostics must NOT leak matched_prompt
        assert data2["routing_metadata"].get("matched_prompt") is None, "Default diagnostics must not leak raw prompts"
