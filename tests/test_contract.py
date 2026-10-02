import pytest
from httpx import ASGITransport, AsyncClient

from adaroute.api import app


@pytest.mark.asyncio
async def test_tc_m1_01_logical_capability_alias_routing():
    """[TC_M1_01] Milestone M1: Single unified endpoint accepts logical capability alias

    (e.g., model='coding', model='general') rather than provider-specific SDK calls.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "coding",
                "prompt": "def fibonacci(n):",
                "quality_requirement": "medium",
                "privacy_level": "public",
                "budget": 1.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["logical_model"] == "coding"
        assert "choices" in data and len(data["choices"]) > 0
        assert "routing_metadata" in data
        assert data["routing_metadata"]["logical_model"] == "coding"


@pytest.mark.asyncio
async def test_tc_m1_02_standard_chat_messages_contract():
    """[TC_M1_02] Milestone M1: Standard messages list [role, content] is parsed and executed."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2 + 2?"},
                ],
                "budget": 1.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert data["usage"]["total_tokens"] > 0
