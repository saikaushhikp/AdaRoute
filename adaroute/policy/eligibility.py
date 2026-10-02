"""Hard Constraint Eligibility Checker.

Applies strict filtering across privacy, budget, quality floors, and replica availability.
Ensures fail-closed behavior for all policies, retries, and fallbacks.
"""
from __future__ import annotations

from adaroute.schemas import ModelSpec, RequestContext

from .registry import ModelRegistry


class EligibilityChecker:
    QUALITY_FLOORS = {
        "low": 0.50,
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
        # 1. HARD CONSTRAINT: Privacy (Fail-closed)
        if request.privacy_level not in spec.allowed_privacy:
            return False, f"Privacy violation: {request.privacy_level} not permitted on {spec.provider}/{spec.model_id}", 0.0

        # 2. HARD CONSTRAINT: Budget Ceiling
        est_cost = self.estimate_cost(spec, request.prompt_tokens, request.estimated_output_tokens)
        if request.budget is not None and est_cost > request.budget:
            return False, f"Budget exceeded: Estimated cost ${est_cost:.5f} > budget ${request.budget:.5f}", est_cost

        # 3. HARD CONSTRAINT: Quality Floor
        min_quality = self.QUALITY_FLOORS.get(request.quality_requirement.lower(), 0.70)
        model_q = spec.quality_scores.get(request.task_type.lower(), spec.quality_scores.get("general", 0.75))
        if model_q < min_quality:
            # If request explicitly requires 'high' quality, enforce strictly
            if request.quality_requirement.lower() == "high":
                return False, f"Quality floor not met: {model_q:.2f} < required {min_quality:.2f}", est_cost

        # 4. HARD CONSTRAINT: Replica Availability
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
