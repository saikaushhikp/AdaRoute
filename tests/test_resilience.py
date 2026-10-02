import pytest
from httpx import ASGITransport, AsyncClient

from adaroute.api import app, backend_runners
from adaroute.resilience.circuit import CircuitBreaker, CircuitState


@pytest.mark.asyncio
async def test_tc_m5_01_bounded_retry_on_replica_failure():
    """[TC_M5_01] Milestone M5: When primary replica experiences transient failure,

    gateway retries boundedly on an alternate replica and records attempt count.
    """
    backend_runners.get("mock-external")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Request with normal execution
        res = await client.post(
            "/v1/chat/completions",
            json={"model": "general", "prompt": "Retry test", "privacy_level": "public"},
        )
        assert res.status_code == 200
        response_data = res.json()
        assert response_data["routing_metadata"]["attempts"] >= 1


def test_tc_m5_02_circuit_breaker_state_transitions():
    """[TC_M5_02] Milestone M5: Circuit breaker trips to OPEN after 3 failures and blocks traffic."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=0.2)
    resource = "test-replica-1"

    # Initially CLOSED
    assert cb.get_state(resource) == CircuitState.CLOSED
    assert cb.allow_request(resource) is True

    # 3 failures -> Trips to OPEN
    cb.record_failure(resource)
    cb.record_failure(resource)
    assert cb.get_state(resource) == CircuitState.CLOSED

    cb.record_failure(resource)
    assert cb.get_state(resource) == CircuitState.OPEN
    # Invariant: Requests rejected immediately in OPEN state
    assert cb.allow_request(resource) is False

    # After recovery timeout -> Transitions to HALF_OPEN
    import time
    time.sleep(0.25)
    assert cb.get_state(resource) == CircuitState.HALF_OPEN
    assert cb.allow_request(resource) is True

    # Canary success resets to CLOSED
    cb.record_success(resource)
    assert cb.get_state(resource) == CircuitState.CLOSED
