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
async def test_tc_m4_05_multi_turn_repeat_query_cache_hit():
    """[TC_M4_05] Robust Caching: Exact query repeated in multi-turn conversation hits cache via secondary index."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        query = "What are the principles of zero-trust network architecture?"

        # Turn 1: Initial query -> Miss
        res1 = await client.post(
            "/v1/chat/completions",
            json={"model": "general", "messages": [{"role": "user", "content": query}]},
        )
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["routing_metadata"]["cache_hit"] is False
        assistant_reply = data1["choices"][0]["message"]["content"]

        # Turn 2: User repeats the same query in multi-turn conversation -> Must HIT cache!
        res2 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": assistant_reply},
                    {"role": "user", "content": query},
                ],
            },
        )
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["routing_metadata"]["cache_hit"] is True
        assert data2["choices"][0]["message"]["content"] == assistant_reply


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
