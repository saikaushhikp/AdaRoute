"""Stage 2: Replica Selection & Dynamic Workload-Weighted Scheduling.

Evaluates within-model replicas based on active connections, dynamic workload W_q,
memory pressure, and recent latency to prevent queue buildup and SLO violations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from adaroute.schemas import ModelSpec, ReplicaState, RequestContext

from .registry import ModelRegistry


class BaseReplicaScheduler(ABC):
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    @abstractmethod
    def select_replica(
        self,
        model: ModelSpec,
        request: RequestContext,
        exclude_ids: set[str] | None = None,
        candidate_replicas: list[Any] | None = None,
    ) -> ReplicaState | None:
        """Select specific replica among healthy instances of the selected model."""

    def _resolve_replicas(
        self,
        model: ModelSpec,
        candidate_replicas: list[Any] | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[ReplicaState]:
        if candidate_replicas is not None:
            resolved = []
            for cr in candidate_replicas:
                r_id = getattr(cr, "replica_id", str(cr))
                reg_r = self.registry.get_replica(r_id)
                if reg_r:
                    resolved.append(reg_r)
                elif hasattr(cr, "is_alive") or hasattr(cr, "availability"):
                    resolved.append(cr)
            reps = resolved
        else:
            reps = self.registry.get_replicas_for_model(model.model_id)

        healthy = [
            r
            for r in reps
            if (r.is_alive() if hasattr(r, "is_alive") else getattr(r, "availability", True))
            and (not exclude_ids or getattr(r, "replica_id", str(r)) not in exclude_ids)
        ]
        return healthy

    def update_metrics(self, replica_id: str, metrics: dict[str, float]):
        """Updates real-time engine telemetry (from Prometheus /metrics)."""
        import time

        replica = self.registry.get_replica(replica_id)
        if replica:
            replica.vllm_running = int(metrics.get("num_requests_running", 0.0))
            replica.vllm_waiting = int(metrics.get("num_requests_waiting", 0.0))
            # Use real KV cache usage for memory pressure when available
            if "kv_cache_usage_perc" in metrics:
                replica.memory_pressure = float(metrics["kv_cache_usage_perc"])
            replica.metrics_last_polled = time.time()
            replica.metrics_source = "live_prometheus" if not metrics.get("is_stale", False) else "stale_prometheus"

    def dispatch(self, replica_id: str, prompt_tokens: int, estimated_output_tokens: int):
        """Update replica dynamic state on dispatch (W_q increases)."""
        replica = self.registry.get_replica(replica_id)
        if replica:
            replica.active_requests += 1
            replica.workload_tokens += prompt_tokens + estimated_output_tokens

    def complete(
        self,
        replica_id: str,
        prompt_tokens: int,
        estimated_output_tokens: int,
        latency_ms: float,
        success: bool = True,
    ):
        """Update replica dynamic state on completion (W_q decreases)."""
        import time

        replica = self.registry.get_replica(replica_id)
        if replica:
            replica.active_requests = max(0, replica.active_requests - 1)
            replica.workload_tokens = max(0, replica.workload_tokens - (prompt_tokens + estimated_output_tokens))
            # Exponential moving average for recent latency
            replica.recent_latency_ms = 0.8 * replica.recent_latency_ms + 0.2 * latency_ms
            if success:
                replica.consecutive_failures = 0
                replica.last_heartbeat = time.time()
            else:
                replica.consecutive_failures += 1
                if replica.consecutive_failures >= 3:
                    replica.availability = False


class RoundRobinScheduler(BaseReplicaScheduler):
    """Cycles through healthy replicas of the selected model in round-robin order."""

    def __init__(self, registry: ModelRegistry):
        super().__init__(registry)
        self._indices: dict[str, int] = {}

    def select_replica(
        self,
        model: ModelSpec,
        request: RequestContext,
        exclude_ids: set[str] | None = None,
        candidate_replicas: list[Any] | None = None,
    ) -> ReplicaState | None:
        healthy = self._resolve_replicas(model, candidate_replicas=candidate_replicas, exclude_ids=exclude_ids)
        if not healthy:
            return None

        idx = self._indices.get(model.model_id, 0) % len(healthy)
        chosen = healthy[idx]
        self._indices[model.model_id] = idx + 1
        return chosen


class LeastLoadedScheduler(BaseReplicaScheduler):
    """Picks the healthy replica with the fewest active requests and queue depth."""

    def select_replica(
        self,
        model: ModelSpec,
        request: RequestContext,
        exclude_ids: set[str] | None = None,
        candidate_replicas: list[Any] | None = None,
    ) -> ReplicaState | None:
        healthy = self._resolve_replicas(model, candidate_replicas=candidate_replicas, exclude_ids=exclude_ids)
        if not healthy:
            return None

        # Prioritize live vLLM engine queue (waiting + running) if available, falling back to in-flight tracking
        def _load_metric(r: ReplicaState):
            live_active = getattr(r, "vllm_waiting", 0) + getattr(r, "vllm_running", 0)
            return (live_active, r.active_requests, r.workload_tokens)

        healthy.sort(key=_load_metric)
        return healthy[0]


class SLOAwareScheduler(BaseReplicaScheduler):
    """Evaluates dynamic workload-weighted load W_q, queue depth, and memory pressure.

    W_q = sum(prompt_tokens + estimated_output_tokens).
    Prefers replicas with low W_q and minimal recent latency to meet SLO targets.
    """

    def select_replica(
        self,
        model: ModelSpec,
        request: RequestContext,
        exclude_ids: set[str] | None = None,
        candidate_replicas: list[Any] | None = None,
    ) -> ReplicaState | None:
        healthy = self._resolve_replicas(model, candidate_replicas=candidate_replicas, exclude_ids=exclude_ids)
        if not healthy:
            return None

        if len(healthy) == 1:
            return healthy[0]

        best_score = float("inf")
        best_replica = None

        for r in healthy:
            # Score balances workload-weighted queue (tokens), recent latency, and memory
            score = (
                (r.workload_tokens / 1000.0) * 1.5
                + (r.recent_latency_ms / 50.0) * 1.0
                + (r.memory_pressure * 2.0)
                + (r.active_requests * 1.0)
            )

            # Incorporate live Prometheus engine queues if available
            live_waiting = getattr(r, "vllm_waiting", 0)
            live_running = getattr(r, "vllm_running", 0)
            score += live_waiting * 3.0 + live_running * 1.0

            # Request SLO evaluation: penalize heavily if recent latency approaches or exceeds SLO
            if request.latency_slo and r.recent_latency_ms > 0:
                latency_sec = r.recent_latency_ms / 1000.0
                if latency_sec > request.latency_slo:
                    score += 20.0
                elif latency_sec > request.latency_slo * 0.8:
                    score += 5.0

            # Extra penalty if degraded
            if getattr(r, "is_degraded", False):
                score += 10.0

            if score < best_score:
                best_score = score
                best_replica = r

        return best_replica or healthy[0]
