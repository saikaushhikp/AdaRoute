"""End-to-End Integration Tests for UI Reverse-Proxy and Gateway.
Validates HTTP boundary between UI container and AdaRoute Gateway.
"""

import httpx
import pytest

UI_BASE_URL = "http://localhost:8080"


@pytest.mark.asyncio
async def test_ui_static_asset_serving():
    """Verify UI container serves index.html and assets."""
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(f"{UI_BASE_URL}/", timeout=5.0)
            assert res.status_code == 200
            assert "AdaRoute AI Gateway" in res.text
            assert '<div id="root"></div>' in res.text
        except httpx.ConnectError:
            pytest.skip("UI container is not currently running on port 8080 (optional live test)")


@pytest.mark.asyncio
async def test_ui_health_reverse_proxy():
    """Verify UI container reverse-proxies /health to Gateway."""
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(f"{UI_BASE_URL}/health", timeout=5.0)
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "healthy"
            assert "models" in data
            assert "replicas" in data
        except httpx.ConnectError:
            pytest.skip("UI container is not currently running on port 8080")


@pytest.mark.asyncio
async def test_ui_chat_completion_reverse_proxy_and_cache():
    """Verify UI reverse-proxies /v1/chat/completions, returns metadata, and respects caching."""
    payload = {
        "model": "general",
        "messages": [{"role": "user", "content": "E2E Reverse Proxy Verification Test"}],
        "privacy_level": "public",
        "budget": 1.0,
        "max_tokens": 128,
    }

    async with httpx.AsyncClient() as client:
        try:
            # 1. First execution (miss)
            res1 = await client.post(
                f"{UI_BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=15.0,
            )
            assert res1.status_code == 200
            data1 = res1.json()
            assert "routing_metadata" in data1
            assert data1["routing_metadata"]["logical_model"] == "general"

            # 2. Second execution (cache hit)
            res2 = await client.post(
                f"{UI_BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=15.0,
            )
            assert res2.status_code == 200
            data2 = res2.json()
            assert data2["routing_metadata"]["cache_hit"] is True
            assert data2["routing_metadata"]["estimated_cost"] == 0.0
        except httpx.ConnectError:
            pytest.skip("UI container is not currently running on port 8080")
