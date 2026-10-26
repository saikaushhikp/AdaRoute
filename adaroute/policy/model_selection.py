"""Stage 1: Model & Provider Selection Policies.

Implements genuine distinct baselines (Strongest, Cheapest, Static) and
the multi-objective Adaptive Model Policy over eligible candidates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from adaroute.schemas import ModelSpec, RequestContext

from .eligibility import EligibilityChecker
from .registry import ModelRegistry


class BaseModelPolicy(ABC):
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.checker = EligibilityChecker(registry)

    @abstractmethod
    def rank_models(self, request: RequestContext) -> list[tuple[ModelSpec, float]]:
        """Rank all eligible models according to policy. Returns ordered list of (ModelSpec, estimated_cost)."""

    def select_model(self, request: RequestContext) -> tuple[ModelSpec | None, float, str | None]:
        """Select eligible model. Returns (selected_model, estimated_cost, failure_reason)."""
        ranked = self.rank_models(request)
        if not ranked:
            return None, 0.0, "No eligible models satisfy hard constraints (privacy/budget/availability)"
        return ranked[0][0], ranked[0][1], None


class StrongestModelPolicy(BaseModelPolicy):
    """B1: Always selects the highest-quality eligible model, regardless of cost."""

    def rank_models(self, request: RequestContext) -> list[tuple[ModelSpec, float]]:
        eligible = self.checker.get_eligible_models(request)
        task = request.task_type.lower()
        eligible.sort(
            key=lambda x: x[0].quality_scores.get(task, x[0].quality_scores.get("general", 0.7)),
            reverse=True,
        )
        return eligible


class CheapestModelPolicy(BaseModelPolicy):
    """B2: Always selects the lowest-cost eligible model that meets constraints."""

    def rank_models(self, request: RequestContext) -> list[tuple[ModelSpec, float]]:
        eligible = self.checker.get_eligible_models(request)
        eligible.sort(key=lambda x: x[1])
        return eligible


class StaticRulePolicy(BaseModelPolicy):
    """B3: Routes strictly according to predetermined static task mapping (Config-Driven).

    System-agnostic: uses declarative static_task_map from config.yaml, or resolves
    the first declared model advertising the requested capability. Never inserts
    undeclared models.
    """

    def __init__(self, registry: ModelRegistry, static_map: dict[str, str] | None = None):
        super().__init__(registry)
        cfg_map = registry.get_static_task_map()
        self.static_map = static_map if static_map is not None else (dict(cfg_map) if cfg_map else {})

    def rank_models(self, request: RequestContext) -> list[tuple[ModelSpec, float]]:
        eligible = self.checker.get_eligible_models(request)
        task = request.task_type.lower()
        preferred_id = self.static_map.get(task)
        if not preferred_id:
            # System-agnostic dynamic resolution: select the first declared model advertising this task capability
            candidate_specs = self.registry.get_models_for_capability(task)
            preferred_id = candidate_specs[0].model_id if candidate_specs else None

        preferred = [x for x in eligible if x[0].model_id == preferred_id] if preferred_id else []
        others = sorted([x for x in eligible if x[0].model_id != preferred_id], key=lambda x: x[1])
        return preferred + others


class AdaptiveModelPolicy(BaseModelPolicy):
    """B4 / Proposed: Evaluates normalized multi-objective objective J(m).

    J(m) = w_c * C_norm + w_l * L_norm + w_q * Q_loss_norm + w_s * SLO_penalty + w_f * Failure_penalty
    Objective weights are loaded dynamically from config.yaml policies.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        weights: dict[str, float] | None = None,
        policy_name: str = "adaptive",
    ):
        super().__init__(registry)
        cfg_weights = registry.get_policy_weights(policy_name)
        self.weights = (
            weights
            or cfg_weights
            or {
                "cost": 0.35,
                "latency": 0.20,
                "quality": 0.30,
                "slo": 0.10,
                "failure": 0.05,
            }
        )

    def rank_models(self, request: RequestContext) -> list[tuple[ModelSpec, float]]:
        eligible = self.checker.get_eligible_models(request)
        if len(eligible) <= 1:
            return eligible

        task = request.task_type.lower()
        max_cost = max(c for _, c in eligible) or 1.0
        max_quality = max(s.quality_scores.get(task, s.quality_scores.get("general", 0.7)) for s, _ in eligible) or 1.0

        scored = []
        for spec, cost in eligible:
            # 1. Normalized cost (Cost factors preserved for budgeting & rate-limiting)
            c_norm = cost / max_cost

            # 2. Normalized quality loss
            q_val = spec.quality_scores.get(task, spec.quality_scores.get("general", 0.7))
            q_loss_norm = (max_quality - q_val) / max_quality

            # 3. Estimated latency (based on replica averages)
            replicas = self.registry.get_replicas_for_model(spec.model_id)
            alive_replicas = [r for r in replicas if r.is_alive()]
            avg_latency = (
                sum(r.recent_latency_ms for r in alive_replicas) / len(alive_replicas) if alive_replicas else 100.0
            )
            l_norm = min(1.0, avg_latency / 1000.0)

            # 4. SLO violation risk
            slo_penalty = 0.0
            if request.latency_slo and (avg_latency / 1000.0) > request.latency_slo:
                slo_penalty = 1.0

            # 5. Failure penalty
            fail_rate = (
                sum(r.consecutive_failures for r in alive_replicas) / (len(alive_replicas) * 3.0)
                if alive_replicas
                else 0.0
            )

            # Combined Objective Score
            j = (
                self.weights.get("cost", 0.35) * c_norm
                + self.weights.get("quality", 0.30) * q_loss_norm
                + self.weights.get("latency", 0.20) * l_norm
                + self.weights.get("slo", 0.10) * slo_penalty
                + self.weights.get("failure", 0.05) * fail_rate
            )
            scored.append((j, spec, cost))

        scored.sort(key=lambda x: x[0])
        return [(spec, cost) for _, spec, cost in scored]


class LatencyFirstPolicy(AdaptiveModelPolicy):
    """Named preset: Prioritizes latency above cost."""

    def __init__(self, registry: ModelRegistry):
        super().__init__(registry, policy_name="latency_first")


class CostFirstPolicy(AdaptiveModelPolicy):
    """Named preset: Prioritizes cost reduction above latency and quality."""

    def __init__(self, registry: ModelRegistry):
        super().__init__(registry, policy_name="cost_first")


class QualityFirstPolicy(AdaptiveModelPolicy):
    """Named preset: Prioritizes high quality while satisfying constraints."""

    def __init__(self, registry: ModelRegistry):
        super().__init__(registry, policy_name="quality_first")
