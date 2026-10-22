from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from adaroute.api import app, cache


@pytest.fixture(autouse=True)
def clear_cache():
    cache.clear()


@pytest.mark.asyncio
async def test_tc_m4_01_repeat_request_cache_hit():
    """[TC_M4_01] Milestone M4: Repeated identical request hits cache and returns cached response."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "model": "general",
            "prompt": "What is the capital of Japan?",
            "tenant_id": "tenant-alpha",
        }
        # First call -> Cache miss
        res1 = await client.post("/v1/chat/completions", json=payload)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["routing_metadata"]["cache_hit"] is False

        # Second call -> Cache hit!
        res2 = await client.post("/v1/chat/completions", json=payload)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["routing_metadata"]["cache_hit"] is True
        assert data2["choices"][0]["message"]["content"] == data1["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_tc_m4_02_cross_tenant_cache_isolation():
    """[TC_M4_02] Milestone M4: Tenant isolation invariant. Tenant B must NEVER hit Tenant A's cache."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        prompt = "Confidential business summary query"
        # Tenant A populates cache
        res_a = await client.post(
            "/v1/chat/completions",
            json={"model": "general", "prompt": prompt, "tenant_id": "tenant-A"},
        )
        assert res_a.status_code == 200
        assert res_a.json()["routing_metadata"]["cache_hit"] is False

        # Tenant B queries identical prompt -> MUST MISS CACHE
        res_b = await client.post(
            "/v1/chat/completions",
            json={"model": "general", "prompt": prompt, "tenant_id": "tenant-B"},
        )
        assert res_b.status_code == 200
        # Invariant: Tenant B gets a cache miss due to tenant partition key!
        assert res_b.json()["routing_metadata"]["cache_hit"] is False


@pytest.mark.asyncio
async def test_tc_m4_03_no_cache_bypass():
    """[TC_M4_03] Milestone M4: Requests with no_cache=True must bypass cache lookup and write."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"model": "general", "prompt": "Bypass test", "no_cache": True}
        res1 = await client.post("/v1/chat/completions", json=payload)
        res2 = await client.post("/v1/chat/completions", json=payload)

        assert res1.json()["routing_metadata"]["cache_hit"] is False
        assert res2.json()["routing_metadata"]["cache_hit"] is False


@pytest.mark.asyncio
async def test_tc_m4_04_canonical_normalization_cache_hit():
    """[TC_M4_04] Robust Caching: Single user message and prompt with whitespace variations produce identical cache hit."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. First call uses standalone prompt text with trailing whitespace
        payload1 = {
            "model": "general",
            "prompt": "   Explain quantum cryptography principles   \n",
        }
        res1 = await client.post("/v1/chat/completions", json=payload1)
        assert res1.status_code == 200
        assert res1.json()["routing_metadata"]["cache_hit"] is False

        # 2. Second call uses messages array with different inline/trailing spacing -> Must HIT cache!
        payload2 = {
            "model": "general",
            "messages": [{"role": "user", "content": "Explain  quantum cryptography  principles"}],
        }
        res2 = await client.post("/v1/chat/completions", json=payload2)
        assert res2.status_code == 200
        assert res2.json()["routing_metadata"]["cache_hit"] is True
        assert res2.json()["choices"][0]["message"]["content"] == res1.json()["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_tc_m4_05_lossless_full_context_cache_and_poisoning_prevention():
    """[TC_M4_05] P0 Assessment (W2-FB001): Full ordered context keying prevents context poisoning.

    Changing system facts (e.g. balance 100 vs 900) or changing max_tokens (64 vs 1)
    causes a cache MISS, while identical full-context calls produce a cache HIT.
    Unsafe secondary query_key fallback is completely disabled.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Turn 1: System prompt states balance 100 with max_tokens=64 -> Cache MISS
        req1 = {
            "model": "general",
            "messages": [
                {"role": "system", "content": "You are a banking assistant. Customer balance is 100."},
                {"role": "user", "content": "What is my current balance?"},
            ],
            "max_tokens": 64,
        }
        res1 = await client.post("/v1/chat/completions", json=req1)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["routing_metadata"]["cache_hit"] is False

        # Turn 2: Critical Context Fact Changed: Balance is now 900 with max_tokens=64 -> MUST MISS CACHE!
        # (Old unsafe query_key secondary index poisoned this turn by returning cached balance 100 response)
        req2 = {
            "model": "general",
            "messages": [
                {"role": "system", "content": "You are a banking assistant. Customer balance is 900."},
                {"role": "user", "content": "What is my current balance?"},
            ],
            "max_tokens": 64,
        }
        res2 = await client.post("/v1/chat/completions", json=req2)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["routing_metadata"]["cache_hit"] is False, "Changing system context must produce cache MISS"

        # Turn 3: Context options changed: Same messages as Turn 2, but max_tokens=1 -> MUST MISS CACHE!
        req3 = {
            "model": "general",
            "messages": [
                {"role": "system", "content": "You are a banking assistant. Customer balance is 900."},
                {"role": "user", "content": "What is my current balance?"},
            ],
            "max_tokens": 1,
        }
        res3 = await client.post("/v1/chat/completions", json=req3)
        assert res3.status_code == 200
        data3 = res3.json()
        assert data3["routing_metadata"]["cache_hit"] is False, "Changing max_tokens must produce cache MISS"

        # Turn 4: Identical full-context and generation options as Turn 3 -> MUST HIT CACHE!
        req4 = {
            "model": "general",
            "messages": [
                {"role": "system", "content": "You are a banking assistant. Customer balance is 900."},
                {"role": "user", "content": "What is my current balance?"},
            ],
            "max_tokens": 1,
        }
        res4 = await client.post("/v1/chat/completions", json=req4)
        assert res4.status_code == 200
        data4 = res4.json()
        assert data4["routing_metadata"]["cache_hit"] is True, "Identical full-context request must produce cache HIT"
        assert data4["choices"][0]["message"]["content"] == data3["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_tc_m4_06_gemini_model_registered_and_spec_verified():
    """[TC_M4_06] Gemini Model: gemini/gemini-flash-latest is registered with accurate specs."""
    from adaroute.policy.registry import default_registry

    gemini_spec = default_registry.get_model("gemini/gemini-flash-latest")
    assert gemini_spec is not None
    assert gemini_spec.provider == "gemini"
    assert gemini_spec.is_external is True
    assert gemini_spec.is_cloud is True
    assert gemini_spec.max_context == 1048576
    assert gemini_spec.input_price_per_1k == 0.000075
    assert gemini_spec.output_price_per_1k == 0.000300
    assert gemini_spec.quality_scores["general"] == 0.89


@pytest.mark.asyncio
async def test_tc_m4_07_user_private_cache_partition_and_isolation():
    """[TC_M4_07] user_private cache isolation: A prompt cached under public/cloud mode MUST NOT hit when user_private=True."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        prompt = "Write an asynchronous script for latency benchmarking."

        # 1. Turn 1: user_private=False, policy=strongest (selects external cloud model)
        res1 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": prompt,
                "policy": "strongest",
                "user_private": False,
            },
        )
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["routing_metadata"]["cache_hit"] is False
        assert "external" in data1["routing_metadata"]["provider"] or data1["routing_metadata"]["provider"] in (
            "gemini",
            "groq",
            "cohere",
            "mock-external",
        )

        # 2. Turn 2: Exact same prompt with user_private=True -> MUST MISS CACHE and route LOCAL!
        res2 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": prompt,
                "policy": "strongest",
                "user_private": True,
            },
        )
        assert res2.status_code == 200
        data2 = res2.json()
        # Invariant: Cache MUST miss because user_private=True cannot receive cloud entries!
        assert data2["routing_metadata"]["cache_hit"] is False
        assert data2["routing_metadata"]["provider"] in ("local", "local_ollama", "mock-local")

        # 3. Turn 3: Repeat identical prompt with user_private=True -> MUST HIT local cache!
        res3 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": prompt,
                "policy": "strongest",
                "user_private": True,
            },
        )
        assert res3.status_code == 200
        data3 = res3.json()
        assert data3["routing_metadata"]["cache_hit"] is True
        assert data3["routing_metadata"]["provider"] in ("local", "local_ollama", "mock-local")
        assert data3["choices"][0]["message"]["content"] == data2["choices"][0]["message"]["content"]

        # 4. Turn 4: Repeat identical prompt with user_private=False -> MUST HIT cloud cache!
        res4 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": prompt,
                "policy": "strongest",
                "user_private": False,
            },
        )
        assert res4.status_code == 200
        data4 = res4.json()
        assert data4["routing_metadata"]["cache_hit"] is True
        assert data4["choices"][0]["message"]["content"] == data1["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_tc_m4_08_code_indentation_preservation_in_cache():
    """[TC_M4_08] Code indentation preservation: Leading spaces/tabs and spaces within multiline code are preserved."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Multiline Python code with 4-space indentation and internal spacing
        code_prompt = (
            "def calculate_balance(principal, rate):\n"
            "    # Leading 4-space indent with internal double spaces\n"
            "    interest =  principal * rate\n"
            "    return  principal + interest"
        )
        res1 = await client.post("/v1/chat/completions", json={"model": "general", "prompt": code_prompt})
        assert res1.status_code == 200
        assert res1.json()["routing_metadata"]["cache_hit"] is False

        # Repeated request with exact same code indentation -> Cache HIT
        res2 = await client.post("/v1/chat/completions", json={"model": "general", "prompt": code_prompt})
        assert res2.status_code == 200
        assert res2.json()["routing_metadata"]["cache_hit"] is True

        # Modified code with different indentation (2 spaces instead of 4) -> Cache MISS
        different_indent_code = (
            "def calculate_balance(principal, rate):\n"
            "  # Leading 2-space indent\n"
            "  interest =  principal * rate\n"
            "  return  principal + interest"
        )
        res3 = await client.post("/v1/chat/completions", json={"model": "general", "prompt": different_indent_code})
        assert res3.status_code == 200
        assert res3.json()["routing_metadata"]["cache_hit"] is False


@pytest.mark.asyncio
async def test_tc_m4_09_simulation_provenance_isolation():
    """[TC_M4_09] Real vs Simulated Isolation: Simulated cache entry CANNOT satisfy execution_mode='real'."""
    from adaroute.cache.exact import ExactCache
    from adaroute.schemas import PrivacyLevel, RequestContext

    test_cache = ExactCache(backend="memory")
    prompt = "Simulated isolation test prompt"

    # Context 1: Simulated execution entry stored
    ctx_sim = RequestContext(
        task_type="general",
        prompt_tokens=10,
        estimated_output_tokens=20,
        privacy_level=PrivacyLevel.PUBLIC,
        execution_mode="hybrid",
    )
    test_cache.put(
        request=ctx_sim,
        logical_model="general",
        prompt=prompt,
        response={"content": "Simulated output", "model": "test-model", "replica_id": "r1"},
        origin_model="test-model",
        origin_provider="local",
        origin_privacy=PrivacyLevel.PUBLIC,
        origin_quality=0.85,
        is_simulated=True,
        source="simulated_local_cpu",
    )

    # Context 2: Request requiring strict real-only execution
    ctx_real = RequestContext(
        task_type="general",
        prompt_tokens=10,
        estimated_output_tokens=20,
        privacy_level=PrivacyLevel.PUBLIC,
        execution_mode="real",
    )
    hit_real = test_cache.get(request=ctx_real, logical_model="general", prompt=prompt)
    assert hit_real is None, "Simulated cache entry must NOT satisfy execution_mode='real'"

    # Context 3: Request allowing hybrid execution
    hit_hybrid = test_cache.get(request=ctx_sim, logical_model="general", prompt=prompt)
    assert hit_hybrid is not None, "Simulated cache entry CAN satisfy execution_mode='hybrid'"
    assert hit_hybrid["content"] == "Simulated output"


@pytest.mark.asyncio
async def test_tc_m4_10_cache_invalidation_methods():
    """[TC_M4_10] Cache Invalidation: invalidate(key), invalidate_prefix(prefix), clear()."""
    from adaroute.cache.exact import ExactCache
    from adaroute.schemas import PrivacyLevel, RequestContext

    test_cache = ExactCache(backend="memory")
    ctx = RequestContext(
        task_type="general",
        prompt_tokens=5,
        estimated_output_tokens=10,
        privacy_level=PrivacyLevel.PUBLIC,
    )

    test_cache.put(
        request=ctx,
        logical_model="general",
        prompt="Alpha prompt",
        response={"content": "Response Alpha", "model": "m1", "replica_id": "r1"},
        origin_model="m1",
        origin_provider="local",
        origin_privacy=PrivacyLevel.PUBLIC,
        origin_quality=0.85,
    )
    k_alpha = test_cache.compute_key(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        logical_model="general",
        model_version="1.0.0",
        prompt="Alpha prompt",
    )

    # Verify hit
    assert test_cache.get(request=ctx, logical_model="general", prompt="Alpha prompt") is not None

    # Invalidate specific key
    invalidated = test_cache.invalidate(k_alpha)
    assert invalidated is True
    assert test_cache.get(request=ctx, logical_model="general", prompt="Alpha prompt") is None

    # Populate two items
    test_cache.put(
        request=ctx,
        logical_model="general",
        prompt="Item 1",
        response={"content": "R1", "model": "m1", "replica_id": "r1"},
        origin_model="m1",
        origin_provider="local",
        origin_privacy=PrivacyLevel.PUBLIC,
        origin_quality=0.85,
    )
    test_cache.put(
        request=ctx,
        logical_model="general",
        prompt="Item 2",
        response={"content": "R2", "model": "m1", "replica_id": "r1"},
        origin_model="m1",
        origin_provider="local",
        origin_privacy=PrivacyLevel.PUBLIC,
        origin_quality=0.85,
    )
    k1 = test_cache.compute_key(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        logical_model="general",
        model_version="1.0.0",
        prompt="Item 1",
    )

    # Invalidate prefix matching k1[:8]
    prefix = k1[:8]
    count = test_cache.invalidate_prefix(prefix)
    assert count >= 1

    # Clear all
    test_cache.clear()
    assert test_cache.hits == 0
    assert test_cache.misses == 0
    assert len(test_cache._store) == 0


@pytest.mark.asyncio
async def test_tc_m4_11_redis_backend_and_transparent_fallback():
    """[TC_M4_11] Shared Redis Backend: TTL, JSON serialization, and automatic transparent fallback."""
    from unittest.mock import MagicMock

    from adaroute.cache.exact import ExactCache
    from adaroute.cache.redis_backend import CacheEntry, InMemoryCacheBackend, RedisCacheBackend
    from adaroute.schemas import PrivacyLevel

    # 1. Test Redis backend operations with mock Redis client
    mock_redis_client = MagicMock()
    mock_storage: dict[str, str] = {}

    def mock_get(key: str):
        return mock_storage.get(key)

    def mock_set(key: str, val: str, ex: int | None = None):
        mock_storage[key] = val
        return True

    def mock_delete(*keys: str):
        deleted = 0
        for k in keys:
            if k in mock_storage:
                del mock_storage[k]
                deleted += 1
        return deleted

    mock_redis_client.get.side_effect = mock_get
    mock_redis_client.set.side_effect = mock_set
    mock_redis_client.delete.side_effect = mock_delete
    mock_redis_client.ping.return_value = True
    mock_redis_client.scan_iter.side_effect = lambda match, count: [
        k for k in mock_storage if k.startswith("adaroute:cache:")
    ]

    redis_backend = RedisCacheBackend(client=mock_redis_client)
    entry = CacheEntry(
        key="test-key-1",
        response={"content": "redis response", "model": "qwen"},
        origin_model="qwen",
        origin_provider="local",
        origin_privacy=PrivacyLevel.PUBLIC,
        origin_quality=0.85,
        ttl_seconds=300.0,
    )
    redis_backend.set("test-key-1", entry, ttl_seconds=300.0)
    assert mock_redis_client.set.called

    fetched = redis_backend.get("test-key-1")
    assert fetched is not None
    assert fetched.response["content"] == "redis response"
    assert fetched.origin_quality == 0.85

    # Test delete
    deleted = redis_backend.delete("test-key-1")
    assert deleted is True
    assert redis_backend.get("test-key-1") is None

    # 2. Test Transparent Fallback when Redis server is unreachable
    cache_with_bad_redis = ExactCache(redis_url="redis://nonexistent-host-9999.invalid:6379/0")
    # Must transparently fall back to memory without throwing an unhandled exception!
    assert cache_with_bad_redis.backend_type == "memory"
    assert isinstance(cache_with_bad_redis._backend, InMemoryCacheBackend)


@pytest.mark.asyncio
async def test_tc_m4_14_prose_punctuation_and_case_normalization():
    """[TC_M4_14] Natural language query normalization: trailing punctuation (?, !, .) and casing match."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Prompt 1: "can you read my prompt?" (with question mark)
        res1 = await client.post(
            "/v1/chat/completions",
            json={"model": "general", "prompt": "can you read my prompt?", "user_private": True},
        )
        assert res1.status_code == 200
        assert res1.json()["routing_metadata"]["cache_hit"] is False

        # Prompt 2: "can you read my prompt" (without question mark) -> MUST HIT CACHE!
        res2 = await client.post(
            "/v1/chat/completions",
            json={"model": "general", "prompt": "can you read my prompt", "user_private": True},
        )
        assert res2.status_code == 200
        assert res2.json()["routing_metadata"]["cache_hit"] is True

        # Prompt 3: "Can you read my prompt? " (capitalized with spaces) -> MUST HIT CACHE!
        res3 = await client.post(
            "/v1/chat/completions",
            json={"model": "general", "prompt": "Can you read my prompt? ", "user_private": True},
        )
        assert res3.status_code == 200
        assert res3.json()["routing_metadata"]["cache_hit"] is True


@pytest.mark.asyncio
async def test_tc_m4_15_external_failover_when_primary_rate_limited(monkeypatch):
    """[TC_M4_15] External Failover: When primary model encounters 429 RateLimitError,

    router fails over to alternate candidate rather than returning a fake simulated response.
    """
    from adaroute.backends.litellm_backend import ExternalBackendUnavailableError, LiteLLMBackend

    # Mock Gemini failing with 429 RateLimitError
    async def mock_execute(self, request, model_spec, replica_id, prompt_text, messages=None):
        if "gemini" in model_spec.model_id.lower():
            raise ExternalBackendUnavailableError("LiteLLM call to gemini failed: 429 Quota Exceeded")
        return {
            "content": f"Real response from fallback model {model_spec.model_id}",
            "prompt_tokens": 5,
            "output_tokens": 15,
            "latency_ms": 25.0,
            "replica_id": replica_id,
            "model": model_spec.model_id,
            "provider": model_spec.provider,
            "source": "litellm_external_real",
            "actual_cost": 0.00005,
        }

    monkeypatch.setattr(LiteLLMBackend, "execute", mock_execute)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": "Test failover query",
                "privacy_level": "public",
                "policy": "adaptive",
                "no_cache": True,
            },
        )
        assert res.status_code == 200
        data = res.json()
        meta = data["routing_metadata"]

        # Invariant: Attempts > 1 proving transparent failover occurred
        assert meta["attempts"] >= 2
        # Invariant: Failed attempt recorded with cost_state="unknown_unbilled"
        assert meta["attempt_history"][0]["status"] == "failed"
        assert meta["attempt_history"][0]["cost_state"] == "unknown_unbilled"
        # Invariant: Final response was fulfilled by an alternate model, NOT a fake simulated gemini response
        assert "Simulated cloud response" not in data["content"]
        assert meta["selected_model"] != "gemini/gemini-flash-latest"
