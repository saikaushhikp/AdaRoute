import httpx
import pytest
import pytest_asyncio

from adaroute.api import app


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


@pytest.mark.asyncio
async def test_TC_M1_11_unified_api_routing_public_task(client):
    """[] Milestone M1: Validate that the unified API routes a public task."""
    response = await client.post(
        "/v1/chat/completions",
        json={
            "prompt": "Hello world, what is the capital of France?",
            "task_type": "qa",
            "privacy_level": "public",
            "budget": 1.0,
            "user_private": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["content"]) > 0
    assert data["provider"] in ("mock-local", "local", "vllm", "local_vllm", "local_ollama")
    assert len(data["replica_id"]) > 0


@pytest.mark.asyncio
async def test_TC_M1_12_unified_api_routing_privacy_constraint(client):
    """[] Milestone M1: Validate that the unified API enforces privacy constraints."""
    response = await client.post(
        "/v1/chat/completions",
        json={
            "prompt": "Summarize this internal proprietary document",
            "task_type": "summarization",
            "privacy_level": "internal",
            "budget": 1.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] in ("mock-local", "local", "vllm", "local_vllm", "local_ollama")
    assert "external" not in data["provider"]


@pytest.mark.asyncio
async def test_TC_M1_13_unified_api_budget_constraint(client):
    """[] Milestone M1: Validate that the unified API enforces budget constraints."""
    # Very low budget constraint should force routing to the cheaper model (mock-local or cost-compliant model)
    # Even if we change the queue state (which we mock statically for now), budget is a hard constraint.
    response = await client.post(
        "/v1/chat/completions",
        json={
            "prompt": "Write a poem",
            "task_type": "creative",
            "privacy_level": "public",
            "budget": 0.01,  # Too low for gpt-4o
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] != "gpt-4o-external"
    assert data["routing_metadata"]["actual_cost"] <= 0.01
