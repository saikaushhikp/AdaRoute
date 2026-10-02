from adaroute.api import registry
from adaroute.policy.replica_selection import (
    LeastLoadedScheduler,
    SLOAwareScheduler,
)
from adaroute.schemas import RequestContext


def test_tc_m6_01_multiple_replicas_exist_for_same_model():
    """[TC_M6_01] Milestone M6: At least two replicas exist for the local model to allow within-model scheduling."""
    replicas = registry.get_replicas_for_model("qwen-2.5-local")
    assert len(replicas) >= 2
    replica_ids = [r.replica_id for r in replicas]
    assert "local-r1" in replica_ids
    assert "local-r2" in replica_ids


def test_tc_m6_02_dynamic_workload_weighted_queue_tracking():
    """[TC_M6_02] Milestone M6: Dynamic W_q queue tracking increments on dispatch and decrements on complete."""
    scheduler = SLOAwareScheduler(registry)
    replica = registry.get_replica("local-r1")
    assert replica is not None

    init_active = replica.active_requests
    init_work = replica.workload_tokens

    # Dispatch a request with 100 prompt + 50 estimated output
    scheduler.dispatch("local-r1", prompt_tokens=100, estimated_output_tokens=50)
    assert replica.active_requests == init_active + 1
    assert replica.workload_tokens == init_work + 150

    # Complete the request
    scheduler.complete("local-r1", prompt_tokens=100, estimated_output_tokens=50, latency_ms=40.0)
    assert replica.active_requests == init_active
    assert replica.workload_tokens == init_work


def test_tc_m6_03_least_loaded_and_slo_aware_selection():
    """[TC_M6_03] Milestone M6: Scheduler selects the least-loaded replica when one is busy."""
    scheduler = LeastLoadedScheduler(registry)
    model_spec = registry.get_model("qwen-2.5-local")
    req = RequestContext(task_type="qa")

    r1 = registry.get_replica("local-r1")
    r2 = registry.get_replica("local-r2")

    # Artificially load replica 1
    r1.active_requests = 10
    r1.workload_tokens = 5000
    r2.active_requests = 0
    r2.workload_tokens = 0

    chosen = scheduler.select_replica(model_spec, req)
    assert chosen is not None
    # Invariant: Must choose lightly loaded replica 2!
    assert chosen.replica_id == "local-r2"

    # Reset
    r1.active_requests = 0
    r1.workload_tokens = 0
