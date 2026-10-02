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
    def select_model(self, request: RequestContext) -> tuple[ModelSpec | None, float, str | None]:
        """Select eligible model. Returns (selected_model, estimated_cost, failure_reason)."""


class StrongestModelPolicy(BaseModelPolicy):
    """B1: Always selects the highest-quality eligible model, regardless of cost."""

    def select_model(self, request: RequestContext) -> tuple[ModelSpec | None, float, str | None]:
        eligible = self.checker.get_eligible_models(request)
        if not eligible:
            return None, 0.0, "No eligible models satisfy hard constraints (privacy/budget/availability)"

        # Sort by quality score for the given task descending
        task = request.task_type.lower()
        eligible.sort(
            key=lambda x: x[0].quality_scores.get(task, x[0].quality_scores.get("general", 0.7)),
            reverse=True,
        )
        chosen_spec, cost = eligible[0]
        return chosen_spec, cost, None


class CheapestModelPolicy(BaseModelPolicy):
    """B2: Always selects the lowest-cost eligible model that meets constraints."""

    def select_model(self, request: RequestContext) -> tuple[ModelSpec | None, float, str | None]:
        eligible = self.checker.get_eligible_models(request)
        if not eligible:
            return None, 0.0, "No eligible models satisfy hard constraints (privacy/budget/availability)"

        # Sort by estimated cost ascending
        eligible.sort(key=lambda x: x[1])
        chosen_spec, cost = eligible[0]
        return chosen_spec, cost, None


class StaticRulePolicy(BaseModelPolicy):
    """B3: Routes strictly according to predetermined static task mapping."""

    STATIC_MAP = {
        "coding": "gpt-4o-external",
        "reasoning": "gpt-4o-external",
        "summarization": "qwen-2.5-local",
        "qa": "qwen-2.5-local",
        "general": "qwen-2.5-local",
    }

    def select_model(self, request: RequestContext) -> tuple[ModelSpec | None, float, str | None]:
        eligible = self.checker.get_eligible_models(request)
        if not eligible:
            return None, 0.0, "No eligible models satisfy hard constraints"

        preferred_id = self.STATIC_MAP.get(request.task_type.lower(), "qwen-2.5-local")
        for spec, cost in eligible:
            if spec.model_id == preferred_id:
                return spec, cost, None

        # Fallback to cheapest eligible if preferred model violates privacy/budget
        sorted_eligible = sorted(eligible, key=lambda x: x[1])
        return sorted_eligible[0][0], sorted_eligible[0][1], None


class AdaptiveModelPolicy(BaseModelPolicy):
    """B4 / Proposed: Evaluates normalized multi-objective objective J(m).

    J(m) = w_c * C_norm + w_l * L_norm + w_q * Q_loss_norm + w_s * SLO_penalty + w_f * Failure_penalty
    """

    def __init__(
        self,
        registry: ModelRegistry,
        weights: dict[str, float] | None = None,
    ):
        super().__init__(registry)
        self.weights = weights or {
            "cost": 0.35,
            "latency": 0.20,
            "quality": 0.30,
            "slo": 0.10,
            "failure": 0.05,
        }

    def select_model(self, request: RequestContext) -> tuple[ModelSpec | None, float, str | None]:
        eligible = self.checker.get_eligible_models(request)
        if not eligible:
            return None, 0.0, "No eligible models satisfy hard constraints"

        if len(eligible) == 1:
            return eligible[0][0], eligible[0][1], None

        task = request.task_type.lower()
        max_cost = max(c for _, c in eligible) or 1.0
        max_quality = max(
            s.quality_scores.get(task, s.quality_scores.get("general", 0.7)) for s, _ in eligible
        ) or 1.0

        best_score = float("inf")
        best_spec = None
        best_cost = 0.0

        for spec, cost in eligible:
            # 1. Normalized cost
            c_norm = cost / max_cost

            # 2. Normalized quality loss
            q_val = spec.quality_scores.get(task, spec.quality_scores.get("general", 0.7))
            q_loss_norm = (max_quality - q_val) / max_quality

            # 3. Estimated latency (based on replica averages)
            replicas = self.registry.get_replicas_for_model(spec.model_id)
            alive_replicas = [r for r in replicas if r.is_alive()]
            avg_latency = (
                sum(r.recent_latency_ms for r in alive_replicas) / len(alive_replicas)
                if alive_replicas
                else 100.0
            )
            l_norm = min(1.0, avg_latency / 1000.0)

            # 4. SLO violation risk
            slo_penalty = 0.0
            if request.latency_slo and (avg_latency / 1000.0) > request.latency_slo:
                slo_penalty = 1.0

            # 5. Failure penalty
            fail_rate = sum(r.consecutive_failures for r in alive_replicas) / (len(alive_replicas) * 3.0)

            # Combined Objective Score
            j = (
                self.weights["cost"] * c_norm
                + self.weights["quality"] * q_loss_norm
                + self.weights["latency"] * l_norm
                + self.weights["slo"] * slo_penalty
                + self.weights["failure"] * fail_rate
            )

            if j < best_score:
                best_score = j
                best_spec = spec
                best_cost = cost

        return best_spec, best_cost, None
