import pytest
from httpx import ASGITransport, AsyncClient

from adaroute.api import app, registry
from adaroute.policy.eligibility import EligibilityChecker
from adaroute.schemas import PrivacyLevel, RequestContext


@pytest.fixture
def eligibility_checker():
    return EligibilityChecker(registry)


@pytest.mark.asyncio
async def test_tc_p2_05_impossible_budget_safely_rejected():
    """[TC_P2_05] Budget Ceiling: If budget is lower than any eligible candidate cost,

    the gateway must reject safely (503 / 400) without running inference.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "prompt": "Explain quantum mechanics in depth",
                "task_type": "reasoning",
                "privacy_level": "public",
                "budget": 0.000001,  # Sub-micro-cent budget, impossible for all backends
            },
        )
        assert response.status_code == 503
        data = response.json()
        assert "Budget exceeded" in data["detail"] or "Routing rejected" in data["detail"]


@pytest.mark.asyncio
async def test_tc_p2_06_negative_budget_validation_error():
    """[TC_P2_06] Negative budget must fail validation immediately."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "prompt": "Hello",
                "budget": -5.0,
            },
        )
        assert response.status_code in (400, 422)


def test_tc_p2_07_budget_differentiates_model_tier(eligibility_checker):
    """[TC_P2_07] Intermediate budget permits local cheap model but excludes expensive cloud model."""
    local_spec = registry.get_model("qwen-2.5-local")
    ext_spec = registry.get_model("gpt-4o-external")

    # Moderate budget sufficient for local ($0.001/k) but too low for GPT-4o ($0.015/k)
    req = RequestContext(
        task_type="coding",
        prompt_tokens=500,
        estimated_output_tokens=300,
        privacy_level=PrivacyLevel.PUBLIC,
        budget=0.002,  # Local cost: ~0.0007, External cost: ~0.007
    )

    local_ok, _, _local_cost = eligibility_checker.check_model_eligibility(local_spec, req)
    ext_ok, ext_reason, _ext_cost = eligibility_checker.check_model_eligibility(ext_spec, req)

    assert local_ok is True
    assert ext_ok is False
    assert "Budget exceeded" in ext_reason
