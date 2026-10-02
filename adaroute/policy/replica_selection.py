"""Stage 2: Replica Selection & Dynamic Workload-Weighted Scheduling.

Evaluates within-model replicas based on active connections, dynamic workload W_q,
memory pressure, and recent latency to prevent queue buildup and SLO violations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from adaroute.schemas import ModelSpec, ReplicaState, RequestContext

from .registry import ModelRegistry


class BaseReplicaScheduler(ABC):
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    @abstractmethod
    def select_replica(self, model: ModelSpec, request: RequestContext) -> ReplicaState | None:
        """Select specific replica among healthy instances of the selected model."""

    def dispatch(self, replica_id: str, prompt_tokens: int, estimated_output_tokens: int):
        """Update replica dynamic state on dispatch (W_q increases)."""
        replica = self.registry.get_replica(replica_id)
        if replica:
            replica.active_requests += 1
            replica.workload_tokens += (prompt_tokens + estimated_output_tokens)

    def complete(
        self,
        replica_id: str,
        prompt_tokens: int,
        estimated_output_tokens: int,
        latency_ms: float,
        success: bool = True,
    ):
        """Update replica dynamic state on completion (W_q decreases)."""
        replica = self.registry.get_replica(replica_id)
        if replica:
            replica.active_requests = max(0, replica.active_requests - 1)
            replica.workload_tokens = max(0, replica.workload_tokens - (prompt_tokens + estimated_output_tokens))
            # Exponential moving average for recent latency
            replica.recent_latency_ms = 0.8 * replica.recent_latency_ms + 0.2 * latency_ms
            if success:
                replica.consecutive_failures = 0
            else:
                replica.consecutive_failures += 1
                if replica.consecutive_failures >= 3:
                    replica.availability = False


class RoundRobinScheduler(BaseReplicaScheduler):
    """Cycles through healthy replicas of the selected model in round-robin order."""

    def __init__(self, registry: ModelRegistry):
        super().__init__(registry)
        self._indices: dict[str, int] = {}

    def select_replica(self, model: ModelSpec, request: RequestContext) -> ReplicaState | None:
        replicas = self.registry.get_replicas_for_model(model.model_id)
        healthy = [r for r in replicas if r.is_alive()]
        if not healthy:
            return None

        idx = self._indices.get(model.model_id, 0) % len(healthy)
        chosen = healthy[idx]
        self._indices[model.model_id] = idx + 1
        return chosen


class LeastLoadedScheduler(BaseReplicaScheduler):
    """Picks the healthy replica with the fewest active requests."""

    def select_replica(self, model: ModelSpec, request: RequestContext) -> ReplicaState | None:
        replicas = self.registry.get_replicas_for_model(model.model_id)
        healthy = [r for r in replicas if r.is_alive()]
        if not healthy:
            return None

        healthy.sort(key=lambda r: (r.active_requests, r.workload_tokens))
        return healthy[0]


class SLOAwareScheduler(BaseReplicaScheduler):
    """Evaluates dynamic workload-weighted load W_q and memory pressure.

    W_q = sum(prompt_tokens + estimated_output_tokens).
    Prefers replicas with low W_q and minimal recent latency to meet SLO targets.
    """

    def select_replica(self, model: ModelSpec, request: RequestContext) -> ReplicaState | None:
        replicas = self.registry.get_replicas_for_model(model.model_id)
        healthy = [r for r in replicas if r.is_alive()]
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

            # Extra penalty if degraded
            if r.is_degraded:
                score += 10.0

            if score < best_score:
                best_score = score
                best_replica = r

        return best_replica or healthy[0]
