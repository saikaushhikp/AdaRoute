from adaroute.api import registry
from adaroute.policy.replica_selection import (
    LeastLoadedScheduler,
    SLOAwareScheduler,
)
from adaroute.schemas import ReplicaState, RequestContext


def test_tc_m6_01_multiple_replicas_exist_for_same_model():
    """[TC_M6_01] Milestone M6: At least two replicas exist for the local model to allow within-model scheduling."""
    local_models = [m for m in registry.get_all_models() if not m.is_external and not m.is_cloud]
    assert len(local_models) > 0, "No local models found in registry"
    target = next((m for m in local_models if len(registry.get_replicas_for_model(m.model_id)) >= 2), local_models[0])
    replicas = registry.get_replicas_for_model(target.model_id)
    assert len(replicas) >= 2, f"Expected at least 2 replicas for {target.model_id}, found {len(replicas)}"


def test_tc_m6_02_dynamic_workload_weighted_queue_tracking():
    """[TC_M6_02] Milestone M6: Dynamic W_q queue tracking increments on dispatch and decrements on complete."""
    scheduler = SLOAwareScheduler(registry)
    local_models = [m for m in registry.get_all_models() if not m.is_external and not m.is_cloud]
    assert len(local_models) > 0, "No local models found in registry"
    model = local_models[0]
    replicas = registry.get_replicas_for_model(model.model_id)
    assert len(replicas) > 0, f"No replicas registered for model {model.model_id}"
    replica = replicas[0]
    rep_id = replica.replica_id

    init_active = replica.active_requests
    init_work = replica.workload_tokens

    # Dispatch a request with 100 prompt + 50 estimated output
    scheduler.dispatch(rep_id, prompt_tokens=100, estimated_output_tokens=50)
    assert replica.active_requests == init_active + 1
    assert replica.workload_tokens == init_work + 150

    # Complete the request
    scheduler.complete(rep_id, prompt_tokens=100, estimated_output_tokens=50, latency_ms=40.0)
    assert replica.active_requests == init_active
    assert replica.workload_tokens == init_work


def test_tc_m6_03_least_loaded_and_slo_aware_selection():
    """[TC_M6_03] Milestone M6: Scheduler selects the least-loaded replica when one is busy."""
    scheduler = LeastLoadedScheduler(registry)
    local_models = [m for m in registry.get_all_models() if not m.is_external and not m.is_cloud]
    assert len(local_models) > 0, "No local models found in registry"
    model_spec = next(
        (m for m in local_models if len(registry.get_replicas_for_model(m.model_id)) >= 2), local_models[0]
    )
    replicas = registry.get_replicas_for_model(model_spec.model_id)
    if len(replicas) < 2:
        r2 = ReplicaState(
            replica_id=f"{replicas[0].replica_id}-r2",
            model_id=model_spec.model_id,
            provider=model_spec.provider,
            endpoint=replicas[0].endpoint,
            availability=True,
            weight=1.0,
        )
        registry.register_replica(r2)
        model_spec.replicas.append(r2.replica_id)
        replicas = registry.get_replicas_for_model(model_spec.model_id)

    r1 = replicas[0]
    r2 = replicas[1]
    req = RequestContext(task_type="qa")

    # Artificially load replica 1
    r1.active_requests = 10
    r1.workload_tokens = 5000
    r2.active_requests = 0
    r2.workload_tokens = 0

    chosen = scheduler.select_replica(model_spec, req)
    assert chosen is not None
    # Invariant: Must choose lightly loaded replica 2!
    assert chosen.replica_id == r2.replica_id

    # Reset
    r1.active_requests = 0
    r1.workload_tokens = 0
