import pytest
from httpx import ASGITransport, AsyncClient

from adaroute.api import app, registry
from adaroute.experiments.baseline_routing import DeterministicRouter
from adaroute.policy.eligibility import EligibilityChecker
from adaroute.schemas import BackendState, PrivacyLevel, RequestContext


@pytest.fixture
def eligibility_checker():
    return EligibilityChecker(registry)


@pytest.mark.asyncio
async def test_tc_m3_04_privacy_adversarial_forbidden_choice():
    """[TC_M3_04] Adversarial Privacy Test: External backend is cheaper and less loaded,

    but request is internal_only. External MUST NEVER be called or selected.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Send internal_only request
        response = await client.post(
            "/v1/chat/completions",
            json={
                "prompt": "Confidential proprietary quarterly financial roadmap",
                "task_type": "general",
                "privacy_level": "internal_only",
                "budget": 1.0,
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Invariant: Must be served by local provider, never external
        assert data["routing_metadata"]["provider"] == "local"
        assert "local" in data["model"] or "qwen" in data["model"]
        assert "external" not in data["routing_metadata"]["provider"]


@pytest.mark.asyncio
async def test_tc_m3_05_unknown_privacy_fails_closed():
    """[TC_M3_05] Fail-Closed Validation: Unknown privacy labels must be rejected

    with HTTP 400, rather than silently falling back to public.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "prompt": "Test prompt",
                "task_type": "general",
                "privacy_level": "super_secret_unknown_level",
                "budget": 1.0,
            },
        )
        # Invariant: Validation error / 400 Bad Request
        assert response.status_code in (400, 422)


def test_tc_m3_06_unavailable_local_fails_closed_no_leak():
    """[TC_M3_06] Fallback Isolation: When local privacy-compliant backends are offline,

    system must reject rather than leak internal request to external provider.
    """
    # Deterministic router fixture with only external available
    backends = [
        BackendState(
            provider="local",
            model="qwen2.5:1.5b",
            replica_id="local-1",
            availability=False,  # Local is offline!
            queue_depth=0,
        ),
        BackendState(
            provider="mock-external",
            model="gpt-4o-external",
            replica_id="ext-1",
            availability=True,  # External is available & cheaper!
            queue_depth=0,
        ),
    ]
    router = DeterministicRouter(backends)
    req = RequestContext(
        task_type="general",
        prompt_tokens=100,
        estimated_output_tokens=50,
        privacy_level=PrivacyLevel.INTERNAL_ONLY,
        budget=1.0,
    )

    chosen = router.route(req)
    # Invariant: Must NOT select external; must return None (safe rejection)
    assert chosen is None


def test_tc_m3_07_confidential_and_restricted_enforcement(eligibility_checker):
    """[TC_M3_07] Confidential & Restricted labels must be strictly filtered from external models."""
    ext_spec = registry.get_model("gpt-4o-external")
    assert ext_spec is not None

    for priv in [PrivacyLevel.INTERNAL_ONLY, PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED]:
        req = RequestContext(
            task_type="general",
            prompt_tokens=50,
            estimated_output_tokens=50,
            privacy_level=priv,
            budget=10.0,
        )
        is_ok, reason, _ = eligibility_checker.check_model_eligibility(ext_spec, req)
        assert is_ok is False
        assert "Privacy violation" in reason
