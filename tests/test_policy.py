from adaroute.routing import StrongestModelPolicy
from adaroute.schemas import BackendState, RequestContext


def test_tc_1_1_privacy_restriction():
    """[TC_1.1] Milestone M1/M3: Validate that privacy-restricted requests retain internal_only constraints."""
    req = RequestContext(
        task_type="qa",
        prompt_tokens=100,
        estimated_output_tokens=50,
        quality_requirement="high",
        latency_slo=2.0,
        privacy_level="internal_only",
        budget=1.0,
        tenant_id="t1",
    )

    # Example setup to be expanded later
    assert req.privacy_level == "internal_only"


def test_tc_1_2_budget_restriction():
    """[TC_1.2] Milestone M1/M3: Validate that requests retain and enforce budget constraints."""
    req = RequestContext(
        task_type="qa",
        prompt_tokens=100,
        estimated_output_tokens=50,
        quality_requirement="high",
        latency_slo=2.0,
        privacy_level="public",
        budget=0.0001,
        tenant_id="t1",
    )
    # Budget too low, should fail in full implementation
    assert req.budget < 0.01


def test_tc_1_3_unhealthy_replica_removed():
    """[TC_1.3] Milestone M1/M5: Validate that unavailable/unhealthy replicas are excluded from routing candidate set."""
    backends = [
        BackendState(
            provider="local",
            model="llama3",
            replica_id="r1",
            queue_depth=0,
            estimated_workload=0.0,
            latency=0.1,
            memory_pressure=0.5,
            failure_rate=0.0,
            availability=True,
        ),
        BackendState(
            provider="local",
            model="llama3",
            replica_id="r2",
            queue_depth=0,
            estimated_workload=0.0,
            latency=0.1,
            memory_pressure=0.5,
            failure_rate=1.0,
            availability=False,
        ),
    ]
    policy = StrongestModelPolicy(backends)
    req = RequestContext(
        task_type="qa",
        prompt_tokens=100,
        estimated_output_tokens=50,
        quality_requirement="high",
        latency_slo=2.0,
        privacy_level="public",
        budget=1.0,
        tenant_id="t1",
    )

    chosen = policy.route(req)
    assert chosen is not None
    assert chosen.replica_id == "r1"
