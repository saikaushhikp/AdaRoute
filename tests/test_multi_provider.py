import pytest
from httpx import ASGITransport, AsyncClient

from adaroute.api import app


@pytest.mark.asyncio
async def test_tc_m2_01_local_model_execution():
    """[TC_M2_01] Milestone M2: Local model backend executes successfully."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": "Test local inference",
                "privacy_level": "internal_only",  # Forces local model
                "budget": 1.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["routing_metadata"]["provider"] == "local"
        assert "local" in data["model"] or "qwen" in data["model"]


@pytest.mark.asyncio
async def test_tc_m2_02_alternate_provider_execution():
    """[TC_M2_02] Milestone M2: Alternate/cloud provider executes successfully when eligible."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": "Test external inference",
                "privacy_level": "public",
                "policy": "strongest",  # Strongest policy selects high-capability external provider
                "budget": 1.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["routing_metadata"]["provider"] == "mock-external"
        assert "gpt-4o" in data["model"]


@pytest.mark.asyncio
async def test_tc_m2_03_transparent_provider_switch_same_client():
    """[TC_M2_03] Milestone M2: Client uses the identical API payload; gateway switches

    provider transparently based on policy/privacy without caller code alteration.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Request 1: Internal -> routed to local
        res_local = await client.post(
            "/v1/chat/completions",
            json={"model": "general", "prompt": "Hi internal", "privacy_level": "internal_only", "no_cache": True},
        )
        # Request 2: Public + Strongest -> routed to external
        res_ext = await client.post(
            "/v1/chat/completions",
            json={"model": "general", "prompt": "Hi external", "privacy_level": "public", "policy": "strongest", "no_cache": True},
        )

        assert res_local.status_code == 200
        assert res_ext.status_code == 200

        data_local = res_local.json()
        data_ext = res_ext.json()

        # Transparent switch verified
        assert data_local["routing_metadata"]["provider"] == "local"
        assert data_ext["routing_metadata"]["provider"] == "mock-external"
