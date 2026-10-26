"""Structured Explainability & Provenance Telemetry (Week-4 Revamp).

Formats transparent, audit-ready, secret-safe explanations of every routing decision:
- Request signals and 'When to Reason' justification
- Hard constraint rejections and per-candidate reasons
- Pareto dominance pruning results (survivors vs dominated)
- Multi-objective score breakdown for winning and alternative candidates
- Fallback sequence and provenance labels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .candidates import CandidateDeployment
from .constraints import EligibilityDecision
from .objectives import CandidateScoreRecord
from .signals import SemanticSignals


@dataclass
class RoutingExplanation:
    """Detailed explainable breakdown of a routing decision."""

    decision_version: str = "4.0.0"
    policy_name: str = "adaptive"
    capability: str = "general"
    signals: dict[str, Any] = field(default_factory=dict)
    hard_rejections: list[dict[str, Any]] = field(default_factory=list)
    pareto_dominated: list[str] = field(default_factory=list)
    pareto_survivors: list[str] = field(default_factory=list)
    selected_deployment: str = "none"
    selected_model: str = "none"
    selected_provider: str = "none"
    objective_contributions: dict[str, float] = field(default_factory=dict)
    fallback_sequence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_version": self.decision_version,
            "policy_name": self.policy_name,
            "capability": self.capability,
            "signals": self.signals,
            "hard_rejections": self.hard_rejections,
            "pareto_dominated": self.pareto_dominated,
            "pareto_survivors": self.pareto_survivors,
            "selected_deployment": self.selected_deployment,
            "selected_model": self.selected_model,
            "selected_provider": self.selected_provider,
            "objective_contributions": self.objective_contributions,
            "fallback_sequence": self.fallback_sequence,
        }


class ExplanationBuilder:
    """Builds clean, sanitized explanations for UI diagnostics and ledger outbox."""

    @classmethod
    def build(
        cls,
        policy_name: str,
        capability: str,
        signals: SemanticSignals | None,
        rejected: list[EligibilityDecision],
        dominated_ids: list[str],
        ranked: list[tuple[CandidateDeployment, CandidateScoreRecord]],
    ) -> RoutingExplanation:
        hard_rej_records = [
            {"candidate_id": r.candidate_id, "reasons": r.rejection_reasons}
            for r in rejected
        ]

        if ranked:
            winner_cand, winner_score = ranked[0]
            sel_dep = winner_cand.deployment_id
            sel_mod = winner_cand.model_id
            sel_prov = winner_cand.provider_id
            contribs = winner_score.contributions
            fallbacks = [c[0].deployment_id for c in ranked[1:]]
            survivors = [c[0].deployment_id for c in ranked]
        else:
            sel_dep = "none"
            sel_mod = "none"
            sel_prov = "none"
            contribs = {}
            fallbacks = []
            survivors = []

        return RoutingExplanation(
            decision_version="4.0.0",
            policy_name=policy_name,
            capability=capability,
            signals=signals.to_dict() if signals else {},
            hard_rejections=hard_rej_records,
            pareto_dominated=dominated_ids,
            pareto_survivors=survivors,
            selected_deployment=sel_dep,
            selected_model=sel_mod,
            selected_provider=sel_prov,
            objective_contributions=contribs,
            fallback_sequence=fallbacks,
        )
