"""Tests for Section 12.5: Objective Engine and Pareto Pruning.

Verifies:
- Distinct objective behaviors (cost-first, quality-first, latency-first)
- Custom objective dimensions defined purely in configuration
- Multi-objective Pareto dominance pruning (dominated candidate removed)
- Pareto trade-off frontier preservation (non-dominated candidates survive)
- Candidate-set stability: adding a dominated candidate does not change surviving rankings
- Single candidate handling without divide-by-zero
"""

from __future__ import annotations

from adaroute.selection.features import CandidateFeatures, ProvenanceValue
from adaroute.selection.objectives import ConfigurableObjectiveEngine
from adaroute.selection.pareto import ParetoFrontierEngine


def make_features(
    cid: str,
    cost: float,
    quality: float,
    latency: float,
    failure: float = 0.0,
    session_continuity: float = 0.0,
    uncertainty: float = 0.1,
) -> CandidateFeatures:
    return CandidateFeatures(
        candidate_id=cid,
        model_id=cid,
        deployment_id=cid,
        provider_id="test_prov",
        cost_expected=ProvenanceValue(cost, "test"),
        cost_reserved=ProvenanceValue(cost * 1.2, "test"),
        cost_worst_case=ProvenanceValue(cost * 1.5, "test"),
        quality=ProvenanceValue(quality, "test"),
        latency_p50=ProvenanceValue(latency, "test"),
        latency_p95=ProvenanceValue(latency * 1.5, "test"),
        failure_rate=ProvenanceValue(failure, "test"),
        session_continuity=ProvenanceValue(session_continuity, "test"),
        semantic_fit=ProvenanceValue(quality, "test"),
        uncertainty=ProvenanceValue(uncertainty, "test"),
        is_cloud=False,
        is_local=True,
        is_mock=False,
        context_window=32768,
        queue_depth=0,
    )


def test_tc_m3_34_pareto_dominance_pruning():
    """Candidate A dominates Candidate B iff A is <= B on all metrics and < on at least one."""
    pareto = ParetoFrontierEngine(
        objectives=[
            {"name": "cost", "direction": "minimize"},
            {"name": "quality", "direction": "maximize"},
            {"name": "latency", "direction": "minimize"},
        ]
    )

    # Candidate 1: High quality (0.95), higher cost ($0.010), high latency (800ms)
    c1 = make_features("c1-frontier", cost=0.010, quality=0.95, latency=800.0)
    # Candidate 2: Cheap ($0.001), low latency (200ms), lower quality (0.75)
    c2 = make_features("c2-frontier", cost=0.001, quality=0.75, latency=200.0)
    # Candidate 3: Strictly worse than c2 on cost ($0.005), latency (300ms), and quality (0.70)
    c3_dominated = make_features("c3-dominated", cost=0.005, quality=0.70, latency=300.0)

    survivors, dominated_ids = pareto.prune_dominated([c1, c2, c3_dominated])

    assert len(survivors) == 2
    survivor_ids = {s.candidate_id for s in survivors}
    assert survivor_ids == {"c1-frontier", "c2-frontier"}
    assert dominated_ids == ["c3-dominated"]


def test_tc_m3_35_candidate_set_stability():
    """Adding a dominated candidate does not alter the relative order of existing frontier candidates."""
    obj_engine = ConfigurableObjectiveEngine(
        objectives=[
            {"name": "cost", "direction": "minimize", "weight": 0.50},
            {"name": "quality_loss", "direction": "minimize", "weight": 0.50},
        ]
    )

    c1 = make_features("c1", cost=0.002, quality=0.90, latency=300.0)
    c2 = make_features("c2", cost=0.001, quality=0.80, latency=200.0)
    c_dominated = make_features("c_dom", cost=0.010, quality=0.60, latency=1000.0)

    # Rank pair
    ranked_pair = obj_engine.rank_candidates([c1, c2])
    pair_order = [r[0].candidate_id for r in ranked_pair]

    # Rank with dominated candidate added
    ranked_trio = obj_engine.rank_candidates([c1, c2, c_dominated])
    trio_order = [r[0].candidate_id for r in ranked_trio if r[0].candidate_id != "c_dom"]

    # Invariant: relative order of c1 and c2 is invariant
    assert pair_order == trio_order


def test_tc_m3_36_custom_declarative_objective():
    """A completely new objective dimension operates without any Python code changes."""
    custom_engine = ConfigurableObjectiveEngine(
        objectives=[
            {"name": "queue_depth", "direction": "minimize", "weight": 0.80},
            {"name": "cost", "direction": "minimize", "weight": 0.20},
        ]
    )

    c1 = make_features("c1-idle", cost=0.002, quality=0.85, latency=300.0)
    # Give c2 queue depth
    c2 = CandidateFeatures(
        candidate_id="c2-busy",
        model_id="c2",
        deployment_id="c2",
        provider_id="test_prov",
        cost_expected=ProvenanceValue(0.001, "test"),
        cost_reserved=ProvenanceValue(0.002, "test"),
        cost_worst_case=ProvenanceValue(0.003, "test"),
        quality=ProvenanceValue(0.85, "test"),
        latency_p50=ProvenanceValue(300.0, "test"),
        latency_p95=ProvenanceValue(450.0, "test"),
        failure_rate=ProvenanceValue(0.0, "test"),
        session_continuity=ProvenanceValue(0.0, "test"),
        semantic_fit=ProvenanceValue(0.85, "test"),
        uncertainty=ProvenanceValue(0.1, "test"),
        is_cloud=False,
        is_local=True,
        is_mock=False,
        context_window=32768,
        queue_depth=15,  # Heavy queue
    )

    ranked = custom_engine.rank_candidates([c1, c2])
    # c1-idle wins because queue_depth has 80% weight
    assert ranked[0][0].candidate_id == "c1-idle"


def test_tc_m3_37_single_candidate_normalization_safety():
    """Single candidate set evaluates smoothly without division-by-zero."""
    engine = ConfigurableObjectiveEngine()
    c = make_features("single", cost=0.001, quality=0.80, latency=100.0)
    ranked = engine.rank_candidates([c])
    assert len(ranked) == 1
    assert ranked[0][0].candidate_id == "single"
    assert ranked[0][1].total_score >= 0.0
