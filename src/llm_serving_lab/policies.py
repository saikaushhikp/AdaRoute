"""Independent model-class and replica-selection policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .workloads import Request


@dataclass(frozen=True)
class Model:
    model_id: str
    quality: float
    cost_per_million_tokens: float
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    max_context_tokens: int
    privacy_levels: frozenset[str]
    reliability: float


@dataclass(frozen=True)
class Replica:
    replica_id: str
    model_id: str
    speed_factor: float
    capacity: int
    reliability: float


@dataclass(frozen=True)
class ReplicaState:
    available_at: float = 0.0
    active: int = 0
    completions: int = 0
    recent_latency: float = 0.0
    healthy: bool = True
    lane_available_at: tuple[float, ...] = ()


class ModelSelector(Protocol):
    def select(self, request: Request, candidates: list[Model]) -> Model: ...


class ReplicaSelector(Protocol):
    def select(
        self, request: Request, replicas: list[Replica], states: dict[str, ReplicaState], now: float
    ) -> Replica: ...


class StrongestModel:
    def select(self, request: Request, candidates: list[Model]) -> Model:
        return max(candidates, key=lambda model: (model.quality, -model.cost_per_million_tokens))


class CheapestModel:
    def select(self, request: Request, candidates: list[Model]) -> Model:
        return min(candidates, key=lambda model: (model.cost_per_million_tokens, -model.quality))


class StaticRuleModel:
    def select(self, request: Request, candidates: list[Model]) -> Model:
        required = request.minimum_quality + (0.04 if request.task in {"code", "reasoning"} else 0)
        qualified = [model for model in candidates if model.quality >= required]
        return min(qualified or candidates, key=lambda model: model.cost_per_million_tokens)


class AdaptiveModel:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {"quality": 0.42, "cost": 0.22, "latency": 0.24, "reliability": 0.12}

    def select(self, request: Request, candidates: list[Model]) -> Model:
        max_cost = max(model.cost_per_million_tokens for model in candidates) or 1.0
        max_delay = max(
            (request.prompt_tokens / model.prefill_tokens_per_second)
            + (request.expected_output_tokens / model.decode_tokens_per_second)
            for model in candidates
        ) or 1.0

        def score(model: Model) -> float:
            estimated_cost = (
                (request.prompt_tokens + request.expected_output_tokens)
                * model.cost_per_million_tokens
                / 1_000_000
            )
            estimated_latency = (
                request.prompt_tokens / model.prefill_tokens_per_second
                + request.expected_output_tokens / model.decode_tokens_per_second
            )
            quality = min(model.quality / max(request.minimum_quality, 0.01), 1.2) / 1.2
            cost = 1 - min(estimated_cost / max(request.max_cost, 1e-9), 1.0)
            latency = 1 - min(estimated_latency / max(request.latency_target_seconds, 1e-9), 1.0)
            reliability = model.reliability
            # A small normalization term keeps incomparable model profiles on a stable scale.
            cost = max(cost, 1 - model.cost_per_million_tokens / max_cost * 0.35)
            latency = max(latency, 1 - estimated_latency / max_delay * 0.35)
            return (
                self.weights.get("quality", 0.42) * quality
                + self.weights.get("cost", 0.22) * cost
                + self.weights.get("latency", 0.24) * latency
                + self.weights.get("reliability", 0.12) * reliability
            )

        return max(candidates, key=lambda model: (score(model), model.quality))


class RoundRobinReplica:
    def __init__(self) -> None:
        self._next_by_model: dict[str, int] = {}

    def select(self, request: Request, replicas: list[Replica], states: dict[str, ReplicaState], now: float) -> Replica:
        ordered = sorted(replicas, key=lambda replica: replica.replica_id)
        index = self._next_by_model.get(ordered[0].model_id, 0) % len(ordered)
        self._next_by_model[ordered[0].model_id] = index + 1
        return ordered[index]


class LeastLoadedReplica:
    def select(self, request: Request, replicas: list[Replica], states: dict[str, ReplicaState], now: float) -> Replica:
        return min(
            replicas,
            key=lambda replica: (
                max(0.0, states[replica.replica_id].available_at - now),
                states[replica.replica_id].active / max(replica.capacity, 1),
                -replica.speed_factor,
                replica.replica_id,
            ),
        )


class AdaptiveReplica:
    def select(self, request: Request, replicas: list[Replica], states: dict[str, ReplicaState], now: float) -> Replica:
        def score(replica: Replica) -> float:
            state = states[replica.replica_id]
            wait = max(0.0, state.available_at - now)
            capacity_pressure = state.active / max(replica.capacity, 1)
            history = state.recent_latency
            return wait + 0.35 * capacity_pressure + 0.25 * history - 0.2 * replica.speed_factor - 0.5 * replica.reliability

        return min(replicas, key=lambda replica: (score(replica), replica.replica_id))


@dataclass(frozen=True)
class Policy:
    name: str
    model_selector: ModelSelector
    replica_selector: ReplicaSelector


def build_policies(weights: dict[str, float] | None = None) -> list[Policy]:
    return [
        Policy("strongest", StrongestModel(), LeastLoadedReplica()),
        Policy("cheapest", CheapestModel(), LeastLoadedReplica()),
        Policy("static_rule", StaticRuleModel(), LeastLoadedReplica()),
        Policy("adaptive", AdaptiveModel(weights), AdaptiveReplica()),
        Policy("round_robin", StaticRuleModel(), RoundRobinReplica()),
        Policy("least_loaded", StaticRuleModel(), LeastLoadedReplica()),
    ]