"""Hard Constraint Eligibility Engine (Week-4 Revamp).

Evaluates all mandatory hard feasibility & safety constraints before soft scoring:
1. Capability support & modality fit
2. Privacy allow-lists (Fail-closed)
3. User-private mode (Excludes cloud/external providers)
4. Context window capacity: (prompt_tokens + reservation_tokens) <= context_window
5. Credential availability for cloud providers
6. Hard budget ceilings: C_est <= B and C_max <= B
7. Session cumulative budget remaining
8. Provider cooldown and circuit breaker state
9. Replica availability (at least 1 healthy replica for local deployments)
10. Execution mode compatibility (e.g. mock exclusion in 'real' mode)
11. Quality floor (low: 0.0, medium: 0.70, high: 0.85)

Returns a typed EligibilityDecision containing all failed constraint reasons.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from adaroute.schemas import RequestContext

from .candidates import CandidateDeployment
from .estimator import OutputEstimate


@dataclass
class EligibilityDecision:
    """Typed decision containing the eligibility outcome and detailed rejection reasons."""

    candidate_id: str
    is_eligible: bool
    rejection_reasons: list[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    worst_case_cost: float = 0.0

    @property
    def primary_reason(self) -> str | None:
        return self.rejection_reasons[0] if self.rejection_reasons else None


class HardConstraintEngine:
    """Evaluates all hard constraints in a deterministic, provider-independent order."""

    QUALITY_FLOORS = {
        "low": 0.0,
        "medium": 0.70,
        "high": 0.85,
    }

    def __init__(self, execution_mode: str = "hybrid"):
        self.execution_mode = execution_mode.lower()

    def evaluate(
        self,
        candidate: CandidateDeployment,
        request: RequestContext,
        estimate: OutputEstimate,
        session_budget_remaining: float | None = None,
        circuit_breaker: Any = None,
        failed_replicas: set[str] | None = None,
    ) -> EligibilityDecision:
        rejection_reasons: list[str] = []
        eff_mode = (
            "real"
            if (self.execution_mode == "real" or (request.execution_mode and request.execution_mode.lower() == "real"))
            else (request.execution_mode or self.execution_mode or "hybrid").lower()
        )
        now = time.time()

        # 1. Capability & Modality Support
        cap_prof = candidate.get_capability_profile(request.task_type)
        if cap_prof and not cap_prof.supported:
            rejection_reasons.append(
                f"Capability '{request.task_type}' is unsupported on deployment '{candidate.deployment_id}'"
            )

        # 2. Execution Mode Mock Exclusion
        if eff_mode == "real" and candidate.is_mock:
            rejection_reasons.append(
                f"Real execution mode active: mock candidate '{candidate.deployment_id}' is excluded"
            )

        # 3. User Private Toggle
        if request.user_private and candidate.is_cloud:
            rejection_reasons.append(
                f"User private toggle active: cloud model '{candidate.provider_id}/{candidate.model_id}' is excluded"
            )

        # 4. Privacy Allow-list (Fail-closed)
        if request.privacy_level not in candidate.allowed_privacy:
            rejection_reasons.append(
                f"Privacy violation: level '{request.privacy_level.value}' not allowed on '{candidate.deployment_id}'"
            )

        # 5. Context Window Capacity
        eff_out = request.max_tokens if (request.max_tokens and request.max_tokens > 0) else estimate.reservation_tokens
        total_tokens_needed = request.prompt_tokens + eff_out
        if total_tokens_needed > candidate.context_window:
            rejection_reasons.append(
                f"Context window exceeded: required {total_tokens_needed} tokens > limit {candidate.context_window}"
            )

        # 6. Credential Availability (Cloud deployments)
        if candidate.is_cloud and candidate.credential_reference:
            cred_val = os.getenv(candidate.credential_reference)
            if not cred_val:
                rejection_reasons.append(
                    f"Missing credential: environment variable '{candidate.credential_reference}' is not set"
                )

        # 7. Provider Cooldown / Circuit Breaker
        if candidate.runtime.is_in_cooldown(now):
            rejection_reasons.append(
                f"Provider cooldown active on '{candidate.deployment_id}' until {candidate.runtime.cooldown_until:.1f}"
            )

        # 8. Local Replica Availability
        if candidate.is_local:
            excluded = failed_replicas or set()
            available_reps = [
                r
                for r in candidate.replicas
                if r.availability
                and r.replica_id not in excluded
                and (circuit_breaker is None or circuit_breaker.allow_request(r.replica_id))
            ]
            if not available_reps:
                rejection_reasons.append(
                    f"No healthy replicas available for local deployment '{candidate.deployment_id}'"
                )

        # 9. Cost & Worst-Case Budget Ceilings
        est_cost = candidate.calculate_expected_cost(request.prompt_tokens, estimate.expected_tokens)
        eff_max_tokens = request.max_tokens if (request.max_tokens and request.max_tokens > 0) else estimate.reservation_tokens
        worst_case_cost = candidate.calculate_max_cost(request.prompt_tokens, eff_max_tokens)

        if request.budget is not None:
            if est_cost > request.budget:
                rejection_reasons.append(
                    f"Budget exceeded: Estimated cost ${est_cost:.5f} > budget ${request.budget:.5f}"
                )
            if worst_case_cost > request.budget:
                rejection_reasons.append(
                    f"Worst-case budget exceeded: Max output cost ${worst_case_cost:.5f} > budget ${request.budget:.5f}"
                )

        # 10. Cumulative Session Budget
        if session_budget_remaining is not None:
            if est_cost > session_budget_remaining:
                rejection_reasons.append(
                    f"Session budget exhausted: Cost ${est_cost:.5f} > remaining session budget ${session_budget_remaining:.5f}"
                )

        # 11. Quality Floor
        min_quality = self.QUALITY_FLOORS.get(request.quality_requirement.lower(), 0.70)
        cand_q = candidate.get_capability_quality(request.task_type)
        if cand_q < min_quality:
            rejection_reasons.append(
                f"Quality floor not met: model quality {cand_q:.2f} < required {min_quality:.2f} for '{request.quality_requirement}'"
            )

        is_eligible = len(rejection_reasons) == 0
        return EligibilityDecision(
            candidate_id=candidate.deployment_id,
            is_eligible=is_eligible,
            rejection_reasons=rejection_reasons,
            estimated_cost=est_cost,
            worst_case_cost=worst_case_cost,
        )
