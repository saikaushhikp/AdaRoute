"""Unified Model & Deployment Selector (Week-4 Revamp).

Orchestrates the complete 12-step selection pipeline:
1. Signal Extraction: deterministic code, math, constraint, and reasoning signals
2. Candidate Resolution: capability/deployment lookup from catalog
3. Output Estimation: deployment-aware expected and reservation tokens
4. Hard Constraints: typed eligibility evaluation with per-constraint reasons
5. Runtime Feature Snapshots: immutable feature captures with metric provenance
6. Pareto Pruning: non-dominated frontier filtering
7. Declarative Objective Scoring: configured weights and directions
8. Session Continuity: soft preference for active session deployment
9. RoutingDecision: typed, explainable output with fallback order
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from adaroute.schemas import RequestContext

from .candidates import CandidateDeployment, CandidateResolver
from .catalog import SystemCatalog
from .constraints import EligibilityDecision, HardConstraintEngine
from .estimator import DeploymentOutputEstimator
from .explanations import ExplanationBuilder, RoutingExplanation
from .features import CandidateFeatures, FeatureSnapshotBuilder
from .normalization import NormalizationEngine
from .objectives import CandidateScoreRecord, ConfigurableObjectiveEngine
from .pareto import ParetoFrontierEngine
from .session import SessionState, SessionStore
from .signals import SignalExtractor


@dataclass
class RoutingDecision:
    """Typed result of the selection pipeline containing the selected deployment, fallbacks, and explanation."""

    request_id: str
    selected_candidate: CandidateDeployment | None
    ranked_candidates: list[tuple[CandidateDeployment, CandidateScoreRecord]]
    rejected_candidates: list[EligibilityDecision]
    explanation: RoutingExplanation
    decision_version: str = "4.0.0"
    policy_name: str = "adaptive"

    @property
    def fallback_candidates(self) -> list[CandidateDeployment]:
        return [c[0] for c in self.ranked_candidates[1:]]

    @property
    def estimated_cost(self) -> float:
        if self.selected_candidate and self.ranked_candidates:
            # Look up estimated cost in feature record if available
            return self.ranked_candidates[0][1].raw_values.get("cost", 0.0)
        return 0.0


class UnifiedModelSelector:
    """System-agnostic, deployment-aware model selection engine."""

    def __init__(
        self,
        catalog: SystemCatalog,
        estimator: DeploymentOutputEstimator | None = None,
        session_store: SessionStore | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.catalog = catalog
        self.config = config or {}
        self.resolver = CandidateResolver(catalog)
        self.estimator = estimator or DeploymentOutputEstimator()
        self.session_store = session_store or SessionStore()

        # Configurable signal extractor
        sig_cfg = self.config.get("semantic_signals", {}).get("reasoning", {})
        on_thresh = float(sig_cfg.get("on_threshold", 0.65))
        off_thresh = float(sig_cfg.get("off_threshold", 0.30))
        self.signal_extractor = SignalExtractor(
            reasoning_on_threshold=on_thresh,
            reasoning_off_threshold=off_thresh,
        )

        # Hard constraint engine
        exec_mode = self.config.get("gateway", {}).get("execution_mode", "hybrid")
        self.constraint_engine = HardConstraintEngine(execution_mode=exec_mode)

        # Normalization engine
        self.normalization_engine = NormalizationEngine()

        # Pareto frontier engine
        self.pareto_engine = ParetoFrontierEngine(normalization=self.normalization_engine)

        # Policies map
        self.policies: dict[str, ConfigurableObjectiveEngine] = {}
        self._load_policies_from_config()

    def _load_policies_from_config(self) -> None:
        """Loads declarative policy objectives from configuration."""
        routing_cfg = self.config.get("routing", {})
        policies_cfg = routing_cfg.get("policies", {})

        # Default built-in policies
        self.policies["adaptive"] = ConfigurableObjectiveEngine(
            objectives=[
                {"name": "quality_loss", "direction": "minimize", "weight": 0.30},
                {"name": "cost", "direction": "minimize", "weight": 0.35},
                {"name": "latency", "direction": "minimize", "weight": 0.20},
                {"name": "reliability", "direction": "maximize", "weight": 0.10},
                {"name": "session_switch", "direction": "minimize", "weight": 0.05},
            ],
            normalization=self.normalization_engine,
        )
        self.policies["latency_first"] = ConfigurableObjectiveEngine(
            objectives=[
                {"name": "latency", "direction": "minimize", "weight": 0.65},
                {"name": "cost", "direction": "minimize", "weight": 0.15},
                {"name": "quality_loss", "direction": "minimize", "weight": 0.15},
                {"name": "reliability", "direction": "maximize", "weight": 0.05},
            ],
            normalization=self.normalization_engine,
        )
        self.policies["cost_first"] = ConfigurableObjectiveEngine(
            objectives=[
                {"name": "cost", "direction": "minimize", "weight": 0.70},
                {"name": "quality_loss", "direction": "minimize", "weight": 0.10},
                {"name": "latency", "direction": "minimize", "weight": 0.15},
                {"name": "reliability", "direction": "maximize", "weight": 0.05},
            ],
            normalization=self.normalization_engine,
        )
        self.policies["quality_first"] = ConfigurableObjectiveEngine(
            objectives=[
                {"name": "quality_loss", "direction": "minimize", "weight": 0.70},
                {"name": "latency", "direction": "minimize", "weight": 0.15},
                {"name": "cost", "direction": "minimize", "weight": 0.10},
                {"name": "reliability", "direction": "maximize", "weight": 0.05},
            ],
            normalization=self.normalization_engine,
        )
        self.policies["strongest"] = self.policies["quality_first"]
        self.policies["cheapest"] = self.policies["cost_first"]

        # Parse user-defined policies from config
        if isinstance(policies_cfg, dict):
            for pname, pdef in policies_cfg.items():
                if isinstance(pdef, dict):
                    objs = pdef.get("objectives")
                    if objs:
                        self.policies[pname] = ConfigurableObjectiveEngine(
                            objectives=objs,
                            normalization=self.normalization_engine,
                        )
                    elif "weights" in pdef:
                        # Legacy format conversion
                        w = pdef["weights"]
                        legacy_objs = [
                            {"name": "cost", "direction": "minimize", "weight": float(w.get("cost", 0.35))},
                            {"name": "quality_loss", "direction": "minimize", "weight": float(w.get("quality", 0.30))},
                            {"name": "latency", "direction": "minimize", "weight": float(w.get("latency", 0.20))},
                            {"name": "reliability", "direction": "maximize", "weight": float(w.get("slo", 0.10))},
                            {"name": "failure_rate", "direction": "minimize", "weight": float(w.get("failure", 0.05))},
                        ]
                        self.policies[pname] = ConfigurableObjectiveEngine(
                            objectives=legacy_objs,
                            normalization=self.normalization_engine,
                        )

    def route(
        self,
        request: RequestContext,
        prompt_text: str = "",
        session_id: str | None = None,
        circuit_breaker: Any = None,
        failed_replicas: set[str] | None = None,
    ) -> RoutingDecision:
        """Executes the complete selection pipeline and returns a typed RoutingDecision."""
        req_id = request.request_id or "req-" + str(int(time.time() * 1000))
        policy_name = (request.policy_name or "adaptive").lower()
        active_policy = self.policies.get(policy_name, self.policies["adaptive"])

        # 1. Signals Extraction
        signals = self.signal_extractor.extract(prompt_text, request.prompt_tokens)

        # 2. Session Context Resolution
        sess_state: SessionState | None = None
        eff_session_id = session_id or getattr(request, "session_id", None)
        if eff_session_id and self.session_store:
            sess_state = self.session_store.get_session(eff_session_id, request.tenant_id)

        # 3. Candidate Resolution
        target_cap = request.task_type
        # If 'When to Reason' signals strict reasoning demand and user requested general, adjust target
        if signals.reasoning_mode == "reasoning_required" and target_cap.lower() in ("general", "qa"):
            target_cap = "reasoning"

        candidates = self.resolver.resolve_candidates(target_cap)
        if not candidates and target_cap != "general":
            # Fall back to general capability if specific task has no registered candidates
            candidates = self.resolver.resolve_candidates("general")

        # 4. Hard Constraints Evaluation
        eligible_candidates: list[tuple[CandidateDeployment, CandidateFeatures]] = []
        rejected_candidates: list[EligibilityDecision] = []

        for cand in candidates:
            # Deployment-aware estimation per candidate
            est = self.estimator.estimate(
                deployment_id=cand.deployment_id,
                capability=request.task_type,
                prompt_tokens=request.prompt_tokens,
                max_tokens=request.max_tokens,
            )

            sess_budget_rem = sess_state.session_budget_remaining if sess_state else None
            decision = self.constraint_engine.evaluate(
                candidate=cand,
                request=request,
                estimate=est,
                session_budget_remaining=sess_budget_rem,
                circuit_breaker=circuit_breaker,
                failed_replicas=failed_replicas,
            )

            if not decision.is_eligible:
                rejected_candidates.append(decision)
                continue

            # Build CandidateFeatures snapshot with provenance
            features = FeatureSnapshotBuilder.build(
                candidate=cand,
                capability=request.task_type,
                prompt_tokens=request.prompt_tokens,
                estimate=est,
                max_tokens=request.max_tokens,
                signals=signals,
                previous_deployment_id=sess_state.previous_deployment_id if sess_state else None,
            )
            eligible_candidates.append((cand, features))

        if not eligible_candidates:
            # No eligible candidates survive hard constraints
            explanation = ExplanationBuilder.build(
                policy_name=policy_name,
                capability=request.task_type,
                signals=signals,
                rejected=rejected_candidates,
                dominated_ids=[],
                ranked=[],
            )
            return RoutingDecision(
                request_id=req_id,
                selected_candidate=None,
                ranked_candidates=[],
                rejected_candidates=rejected_candidates,
                explanation=explanation,
                decision_version="4.0.0",
                policy_name=policy_name,
            )

        # 5. Pareto Dominance Pruning
        all_features = [item[1] for item in eligible_candidates]
        pareto_survivors, dominated_ids = self.pareto_engine.prune_dominated(all_features)
        survivor_cand_ids = {s.candidate_id for s in pareto_survivors}

        # Keep only survivor candidates
        survivor_pairs = [item for item in eligible_candidates if item[0].deployment_id in survivor_cand_ids]

        # 6. Declarative Objective Scoring & Ranking
        survivor_features = [item[1] for item in survivor_pairs]
        ranked_scored_features = active_policy.rank_candidates(survivor_features)

        # Map back to CandidateDeployment objects
        cand_map = {item[0].deployment_id: item[0] for item in survivor_pairs}
        ranked_candidates: list[tuple[CandidateDeployment, CandidateScoreRecord]] = [
            (cand_map[feat.candidate_id], score_rec)
            for feat, score_rec in ranked_scored_features
            if feat.candidate_id in cand_map
        ]

        # 7. Build Explanation Record
        explanation = ExplanationBuilder.build(
            policy_name=policy_name,
            capability=request.task_type,
            signals=signals,
            rejected=rejected_candidates,
            dominated_ids=dominated_ids,
            ranked=ranked_candidates,
        )

        selected = ranked_candidates[0][0] if ranked_candidates else None

        return RoutingDecision(
            request_id=req_id,
            selected_candidate=selected,
            ranked_candidates=ranked_candidates,
            rejected_candidates=rejected_candidates,
            explanation=explanation,
            decision_version="4.0.0",
            policy_name=policy_name,
        )
