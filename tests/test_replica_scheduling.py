from adaroute.api import registry
from adaroute.policy.replica_selection import (
    LeastLoadedScheduler,
    SLOAwareScheduler,
)
from adaroute.schemas import ReplicaState, RequestContext


def test_TC_M6_01_multiple_replicas_exist_for_same_model():
    """[] Milestone M6: At least two replicas exist for the local model to allow within-model scheduling."""
    local_models = [m for m in registry.get_all_models() if not m.is_external and not m.is_cloud]
    assert len(local_models) > 0, "No local models found in registry"
    target = next((m for m in local_models if len(registry.get_replicas_for_model(m.model_id)) >= 2), local_models[0])
    replicas = registry.get_replicas_for_model(target.model_id)
    assert len(replicas) >= 2, f"Expected at least 2 replicas for {target.model_id}, found {len(replicas)}"


def test_TC_M6_02_dynamic_workload_weighted_queue_tracking():
    """[] Milestone M6: Dynamic W_q queue tracking increments on dispatch and decrements on complete."""
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


def test_TC_M6_03_least_loaded_and_slo_aware_selection():
    """[] Milestone M6: Scheduler selects the least-loaded replica when one is busy."""
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


def test_TC_M6_04_round_robin_cycles_healthy_replicas():
    """[] Milestone M6: RoundRobinScheduler cycles deterministically through replicas."""
    from adaroute.policy.replica_selection import RoundRobinScheduler

    scheduler = RoundRobinScheduler(registry)
    local_models = [m for m in registry.get_all_models() if not m.is_external and not m.is_cloud]
    model_spec = next(
        (m for m in local_models if len(registry.get_replicas_for_model(m.model_id)) >= 2), local_models[0]
    )
    replicas = registry.get_replicas_for_model(model_spec.model_id)
    req = RequestContext(task_type="qa")

    first = scheduler.select_replica(model_spec, req)
    second = scheduler.select_replica(model_spec, req)
    assert first is not None and second is not None
    assert first.replica_id != second.replica_id
    assert {first.replica_id, second.replica_id} == {replicas[0].replica_id, replicas[1].replica_id}


def test_TC_M6_05_prometheus_signals_affect_least_loaded():
    """[] Milestone M6: Real vLLM waiting/running request signals steer traffic away from busy engines."""
    scheduler = LeastLoadedScheduler(registry)
    local_models = [m for m in registry.get_all_models() if not m.is_external and not m.is_cloud]
    model_spec = next(
        (m for m in local_models if len(registry.get_replicas_for_model(m.model_id)) >= 2), local_models[0]
    )
    replicas = registry.get_replicas_for_model(model_spec.model_id)
    r1, r2 = replicas[0], replicas[1]
    req = RequestContext(task_type="coding")

    # In-flight counters equal
    r1.active_requests = 0
    r2.active_requests = 0

    # r1 has 3 waiting requests reported by Prometheus
    scheduler.update_metrics(r1.replica_id, {"num_requests_waiting": 3.0, "num_requests_running": 1.0})
    scheduler.update_metrics(r2.replica_id, {"num_requests_waiting": 0.0, "num_requests_running": 0.0})

    chosen = scheduler.select_replica(model_spec, req)
    assert chosen.replica_id == r2.replica_id

    # Reset metrics
    scheduler.update_metrics(r1.replica_id, {"num_requests_waiting": 0.0, "num_requests_running": 0.0})


def test_TC_M6_06_unhealthy_replica_avoidance_and_recovery():
    """[] Milestone M6: Schedulers strictly avoid dead replicas and recover when marked alive."""
    scheduler = LeastLoadedScheduler(registry)
    local_models = [m for m in registry.get_all_models() if not m.is_external and not m.is_cloud]
    model_spec = next(
        (m for m in local_models if len(registry.get_replicas_for_model(m.model_id)) >= 2), local_models[0]
    )
    replicas = registry.get_replicas_for_model(model_spec.model_id)
    r1, r2 = replicas[0], replicas[1]
    req = RequestContext(task_type="qa")

    # Mark r1 unavailable
    r1.availability = False
    chosen = scheduler.select_replica(model_spec, req)
    assert chosen.replica_id == r2.replica_id

    # Recover r1
    r1.availability = True
    r2.availability = False
    chosen_rec = scheduler.select_replica(model_spec, req)
    assert chosen_rec.replica_id == r1.replica_id

    # Restore r2
    r2.availability = True


def test_TC_M6_07_replica_metrics_parser():
    """[] Milestone M6: Parse real Prometheus vLLM 0.34.0 metric format."""
    from adaroute.policy.replica_metrics import ReplicaMetricsCollector

    sample_metrics = """
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="test"} 2.0
# HELP vllm:num_requests_waiting Number of requests currently waiting in queue.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="test"} 5.0
# HELP vllm:kv_cache_usage_perc Percentage of KV cache currently used.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="test"} 0.34
"""
    parsed = ReplicaMetricsCollector.parse_prometheus_text(sample_metrics)
    assert parsed["num_requests_running"] == 2.0
    assert parsed["num_requests_waiting"] == 5.0
    assert parsed["kv_cache_usage_perc"] == 0.34
