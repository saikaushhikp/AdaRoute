import pytest

from adaroute.api import cache, circuit_breaker, registry


@pytest.fixture(autouse=True)
def reset_gateway_state():
    cache.clear()
    circuit_breaker.reset()
    for rep in registry.get_all_replicas():
        rep.consecutive_failures = 0
        rep.availability = True
        rep.workload_tokens = 0
        rep.active_requests = 0
    yield
    cache.clear()
    circuit_breaker.reset()
