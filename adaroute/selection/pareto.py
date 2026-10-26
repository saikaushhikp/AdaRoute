"""Multi-Objective Pareto Dominance Pruning (Week-4 Revamp).

Filters out strictly dominated deployment candidates before weighted objective scoring.
Candidate A dominates Candidate B iff:
1. A is no worse than B on every configured Pareto objective.
2. A is strictly better than B on at least one objective.

Preserves the complete non-dominated trade-off frontier.
"""

from __future__ import annotations

from typing import Any

from .features import CandidateFeatures
from .normalization import NormalizationEngine


class ParetoFrontierEngine:
    """Computes the non-dominated Pareto frontier for eligible candidates."""

    def __init__(
        self,
        objectives: list[dict[str, Any]] | None = None,
        normalization: NormalizationEngine | None = None,
        missing_metric_policy: str = "penalize",  # "penalize" or "reject"
    ):
        self.objectives = objectives or [
            {"name": "cost", "direction": "minimize"},
            {"name": "quality", "direction": "maximize"},
            {"name": "latency", "direction": "minimize"},
            {"name": "reliability", "direction": "maximize"},
        ]
        self.normalization = normalization or NormalizationEngine()
        self.missing_metric_policy = missing_metric_policy.lower()

    def _get_normalized_vector(self, feat: CandidateFeatures) -> list[float] | None:
        """Converts candidate features into a list of normalized minimization metrics (0.0 is best)."""
        vec: list[float] = []
        for obj in self.objectives:
            name = obj["name"]
            raw_val = feat.get_metric_value(name)
            if raw_val is None:
                if self.missing_metric_policy == "reject":
                    return None  # Candidate is rejected due to missing metric
                # Penalize with worst possible normalized score (1.0)
                norm_v = 1.0
            else:
                norm_v, _ = self.normalization.normalize_feature(name, raw_val)
            vec.append(norm_v)
        return vec

    def dominates(self, vec_a: list[float], vec_b: list[float], eps: float = 1e-6) -> bool:
        """Returns True if candidate A dominates candidate B on all metrics (lower is better)."""
        at_least_as_good = True
        strictly_better = False

        for a_val, b_val in zip(vec_a, vec_b):
            if a_val > b_val + eps:
                # A is strictly worse on this objective
                return False
            if a_val < b_val - eps:
                strictly_better = True

        return at_least_as_good and strictly_better

    def prune_dominated(
        self, candidates: list[CandidateFeatures]
    ) -> tuple[list[CandidateFeatures], list[str]]:
        """Removes all candidates dominated on the configured Pareto objectives.

        Returns (survivors, rejected_dominated_candidate_ids).
        """
        if len(candidates) <= 1:
            return list(candidates), []

        # Build normalized vectors
        candidate_vectors: list[tuple[CandidateFeatures, list[float]]] = []
        for c in candidates:
            vec = self._get_normalized_vector(c)
            if vec is not None:
                candidate_vectors.append((c, vec))

        survivors: list[CandidateFeatures] = []
        dominated_ids: list[str] = []

        for i, (cand_i, vec_i) in enumerate(candidate_vectors):
            is_dominated = False
            for j, (cand_j, vec_j) in enumerate(candidate_vectors):
                if i == j:
                    continue
                if self.dominates(vec_j, vec_i):
                    is_dominated = True
                    dominated_ids.append(cand_i.candidate_id)
                    break
            if not is_dominated:
                survivors.append(cand_i)

        return survivors, dominated_ids
