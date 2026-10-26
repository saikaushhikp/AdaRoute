"""Hard Constraint Eligibility Checker (Hardened Week-3 Version).

Applies strict filtering across:
1. Privacy allow-lists (Fail-closed).
2. Hard budget ceilings (Estimated and Worst-Case max_tokens spend).
3. All configured quality floors (low, medium, high).
4. Replica health and circuit breaker states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from adaroute.schemas import ModelSpec, RequestContext

if TYPE_CHECKING:
    from .registry import ModelRegistry


class EligibilityChecker:
    QUALITY_FLOORS = {
        "low": 0.0,
        "medium": 0.70,
        "high": 0.85,
    }

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def estimate_cost(self, spec: ModelSpec, prompt_tokens: int, estimated_output_tokens: int) -> float:
        """Calculate pre-inference estimated cost in dollars."""
        in_cost = (prompt_tokens / 1000.0) * spec.input_price_per_1k
        out_cost = (estimated_output_tokens / 1000.0) * spec.output_price_per_1k
        return in_cost + out_cost

    def check_model_eligibility(
        self,
        spec: ModelSpec,
        request: RequestContext,
    ) -> tuple[bool, str | None, float]:
        """Check if a candidate ModelSpec satisfies all hard constraints.

        Returns (is_eligible, rejection_reason, estimated_cost).
        """
        # 0. HARD CONSTRAINT: Real Mode Mock Exclusion
        import os

        exec_mode = os.getenv("EXECUTION_MODE", "hybrid").lower()
        if exec_mode == "real" and (spec.execution_type == "mock" or spec.provider.startswith("mock")):
            return (
                False,
                f"Real execution mode active: mock candidate '{spec.model_id}' is excluded",
                0.0,
            )

        # 0.1 HARD CONSTRAINT: User Private Toggle (Fail-closed against cloud/external models)
        if request.user_private and spec.is_cloud:
            return (
                False,
                f"User private toggle active: cloud model '{spec.provider}/{spec.model_id}' is excluded",
                0.0,
            )

        # 1. HARD CONSTRAINT: Privacy (Fail-closed)
        if request.privacy_level not in spec.allowed_privacy:
            return (
                False,
                f"Privacy violation: {request.privacy_level} not permitted on {spec.provider}/{spec.model_id}",
                0.0,
            )

        # 2. HARD CONSTRAINT: Budget Ceiling (Pre-inference and Independent Worst-Case C_max Gate)
        est_cost = self.estimate_cost(spec, request.prompt_tokens, request.estimated_output_tokens)
        if request.budget is not None:
            if est_cost > request.budget:
                return (
                    False,
                    f"Budget exceeded: Estimated cost ${est_cost:.5f} > budget ${request.budget:.5f}",
                    est_cost,
                )

            # Independent bounded worst-case output reservation: C_max must NOT exceed budget
            if request.max_tokens is not None and request.max_tokens > 0:
                max_possible_cost = spec.calculate_max_cost(request.prompt_tokens, request.max_tokens)
                if max_possible_cost > request.budget:
                    return (
                        False,
                        f"Budget ceiling exceeded: Max output cost ${max_possible_cost:.5f} > budget ${request.budget:.5f}",
                        est_cost,
                    )

        # 2.5 HARD CONSTRAINT: Context Window Capacity
        needed_tokens = request.prompt_tokens + (request.max_tokens if (request.max_tokens and request.max_tokens > 0) else request.estimated_output_tokens)
        if spec.max_context and needed_tokens > spec.max_context:
            return (
                False,
                f"Context window exceeded: required {needed_tokens} tokens > limit {spec.max_context} for '{spec.model_id}'",
                est_cost,
            )

        # 3. HARD CONSTRAINT: Quality Floor (Enforced across ALL floors: low, medium, high!)
        min_quality = self.QUALITY_FLOORS.get(request.quality_requirement.lower(), 0.70)
        model_q = spec.quality_scores.get(request.task_type.lower(), spec.quality_scores.get("general", 0.75))
        if model_q < min_quality:
            return (
                False,
                f"Quality floor not met: model quality {model_q:.2f} < required {min_quality:.2f} for '{request.quality_requirement}'",
                est_cost,
            )

        # 4. HARD CONSTRAINT: Replica Availability (Only for self-hosted/local models with managed replicas)
        if not spec.is_cloud:
            replicas = self.registry.get_replicas_for_model(spec.model_id)
            healthy_replicas = [r for r in replicas if r.is_alive()]
            if not healthy_replicas:
                return False, f"No healthy replicas available for model {spec.model_id}", est_cost

        return True, None, est_cost

    def get_eligible_models(
        self,
        request: RequestContext,
        candidate_models: list[ModelSpec] | None = None,
    ) -> list[tuple[ModelSpec, float]]:
        """Filter candidates down to strictly eligible models with their estimated costs."""
        candidates = candidate_models or self.registry.get_models_for_capability(request.task_type)
        eligible = []

        for spec in candidates:
            is_ok, _reason, cost = self.check_model_eligibility(spec, request)
            if is_ok:
                eligible.append((spec, cost))

        return eligible
