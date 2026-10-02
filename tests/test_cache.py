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
