"""Generic Declarative Objective Engine (Week-4 Revamp).

Supports arbitrary, configuration-defined objective dimensions, weights, and directions
without hard-coding objective names or formulas in Python code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .features import CandidateFeatures
from .normalization import NormalizationEngine


@dataclass
class ObjectiveDefinition:
    """Definition of a single objective dimension."""

    name: str
    direction: str = "minimize"  # "minimize" or "maximize"
    weight: float = 1.0
    missing_policy: str = "penalize"  # "penalize" or "reject"

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError(f"Objective weight for '{self.name}' must be non-negative, got {self.weight}")


@dataclass
class CandidateScoreRecord:
    """Detailed objective score contribution breakdown for a candidate."""

    candidate_id: str
    total_score: float  # Minimization scale: 0.0 is best
    contributions: dict[str, float] = field(default_factory=dict)
    normalized_values: dict[str, float] = field(default_factory=dict)
    raw_values: dict[str, float] = field(default_factory=dict)


class ConfigurableObjectiveEngine:
    """Evaluates candidates against arbitrary declarative objectives."""

    def __init__(
        self,
        objectives: list[ObjectiveDefinition] | list[dict[str, Any]] | None = None,
        normalization: NormalizationEngine | None = None,
    ):
        self.normalization = normalization or NormalizationEngine()
        self.objectives: list[ObjectiveDefinition] = []

        raw_objs = objectives or [
            {"name": "quality_loss", "direction": "minimize", "weight": 0.30},
            {"name": "cost", "direction": "minimize", "weight": 0.35},
            {"name": "latency", "direction": "minimize", "weight": 0.20},
            {"name": "reliability", "direction": "maximize", "weight": 0.10},
            {"name": "uncertainty", "direction": "minimize", "weight": 0.05},
        ]

        total_weight = 0.0
        for obj in raw_objs:
            if isinstance(obj, ObjectiveDefinition):
                odef = obj
            else:
                odef = ObjectiveDefinition(
                    name=obj["name"],
                    direction=obj.get("direction", "minimize"),
                    weight=float(obj.get("weight", 1.0)),
                    missing_policy=obj.get("missing_policy", "penalize"),
                )
            self.objectives.append(odef)
            total_weight += odef.weight

        # Normalize weights so sum == 1.0 if positive
        if total_weight > 0.0:
            for odef in self.objectives:
                odef.weight = odef.weight / total_weight

    def score_candidate(self, features: CandidateFeatures) -> CandidateScoreRecord:
        """Calculates total scalar objective cost J(c) where lower is better."""
        total_score = 0.0
        contributions: dict[str, float] = {}
        normalized_vals: dict[str, float] = {}
        raw_vals: dict[str, float] = {}

        for obj in self.objectives:
            raw_v = features.get_metric_value(obj.name)
            if raw_v is None:
                norm_v = 1.0  # Penalize missing value
                raw_v = 0.0
            else:
                norm_v, _ = self.normalization.normalize_feature(obj.name, raw_v)

            contribution = obj.weight * norm_v
            total_score += contribution

            contributions[obj.name] = round(contribution, 5)
            normalized_vals[obj.name] = round(norm_v, 5)
            raw_vals[obj.name] = round(raw_v, 5)

        return CandidateScoreRecord(
            candidate_id=features.candidate_id,
            total_score=round(total_score, 6),
            contributions=contributions,
            normalized_values=normalized_vals,
            raw_values=raw_vals,
        )

    def rank_candidates(
        self, candidates: list[CandidateFeatures]
    ) -> list[tuple[CandidateFeatures, CandidateScoreRecord]]:
        """Ranks candidates by total objective score with deterministic tie-breaking."""
        scored = [(c, self.score_candidate(c)) for c in candidates]

        # Stable deterministic tie-breaker:
        # 1. total_score (ascending, lower is better)
        # 2. quality (descending, higher is better)
        # 3. failure_rate (ascending, lower is better)
        # 4. latency_p50 (ascending)
        # 5. cost_expected (ascending)
        # 6. candidate_id (alphabetical)
        scored.sort(
            key=lambda item: (
                item[1].total_score,
                -item[0].quality.value,
                item[0].failure_rate.value,
                item[0].latency_p50.value,
                item[0].cost_expected.value,
                item[0].candidate_id,
            )
        )
        return scored
