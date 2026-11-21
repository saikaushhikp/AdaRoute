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
from .vllm_router import VLLMSemanticRouterClient, classify_semantic_intent


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
        self.static_task_map: dict[str, str] = dict(self.config.get("routing", {}).get("static_task_map", {}))

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

        # vLLM Semantic Router Client
        sr_cfg = self.config.get("semantic_router", {})
        self.vllm_router_client = VLLMSemanticRouterClient(
            router_url=sr_cfg.get("router_url"),
            timeout=float(sr_cfg.get("timeout_seconds", 2.0)),
        )

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
        self.policies["strongest"] = ConfigurableObjectiveEngine(
            objectives=[
                {"name": "quality_loss", "direction": "minimize", "weight": 1.0},
            ],
            normalization=self.normalization_engine,
        )
        self.policies["cheapest"] = ConfigurableObjectiveEngine(
            objectives=[
                {"name": "cost", "direction": "minimize", "weight": 1.0},
            ],
            normalization=self.normalization_engine,
        )
        self.policies["static"] = ConfigurableObjectiveEngine(
            objectives=[
                {"name": "cost", "direction": "minimize", "weight": 0.50},
                {"name": "quality_loss", "direction": "minimize", "weight": 0.50},
            ],
            normalization=self.normalization_engine,
        )
        self.policies["vllm_semantic_router"] = ConfigurableObjectiveEngine(
            objectives=[
                {"name": "quality_loss", "direction": "minimize", "weight": 0.35},
                {"name": "cost", "direction": "minimize", "weight": 0.35},
                {"name": "latency", "direction": "minimize", "weight": 0.30},
            ],
            normalization=self.normalization_engine,
        )

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
        active_policy = self.policies.get(policy_name)
        if not active_policy:
            raise ValueError(f"Unknown routing policy: '{policy_name}'. Valid policies: {sorted(self.policies.keys())}")

        # 1. Signals Extraction
        signals = self.signal_extractor.extract(prompt_text, request.prompt_tokens)

        # 2. Session Context Resolution
        sess_state: SessionState | None = None
        eff_session_id = session_id or getattr(request, "session_id", None)
        if eff_session_id and self.session_store:
            sess_state = self.session_store.get_session(
                eff_session_id, request.tenant_id, initial_budget=getattr(request, "session_budget", None)
            )

        # 3. Candidate Resolution
        target_cap = request.task_type
        is_explicit_model = (
            self.catalog.get_model(target_cap) is not None
            or self.catalog.get_deployment(target_cap) is not None
            or any(
                d.model_id.lower() == target_cap.lower() or d.deployment_id.lower() == target_cap.lower()
                for d in self.catalog.deployments.values()
            )
        )
        # If 'When to Reason' signals strict reasoning demand and user requested general, adjust target
        if (
            not is_explicit_model
            and signals.reasoning_mode == "reasoning_required"
            and target_cap.lower() in ("general", "qa")
        ):
            target_cap = "reasoning"

        candidates = self.resolver.resolve_candidates(target_cap)
        if not candidates and not is_explicit_model and target_cap != "general":
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

        if policy_name == "static":
            preferred_model_id = self.static_task_map.get(request.task_type.lower())
            if preferred_model_id:
                pref_clean = preferred_model_id.lower()
                cand_map_temp = {item[0].deployment_id: item[0] for item in survivor_pairs}

                def _static_sort_key(item: tuple[CandidateFeatures, CandidateScoreRecord]) -> int:
                    c_dep = cand_map_temp.get(item[0].candidate_id)
                    if c_dep and (c_dep.model_id.lower() == pref_clean or c_dep.deployment_id.lower() == pref_clean):
                        return 0
                    return 1

                ranked_scored_features.sort(key=_static_sort_key)

        # Map back to CandidateDeployment objects
        cand_map = {item[0].deployment_id: item[0] for item in survivor_pairs}
        ranked_candidates: list[tuple[CandidateDeployment, CandidateScoreRecord]] = [
            (cand_map[feat.candidate_id], score_rec)
            for feat, score_rec in ranked_scored_features
            if feat.candidate_id in cand_map
        ]

        # Append dominated eligible candidates at the end for resilient fallback
        dominated_pairs = [item for item in eligible_candidates if item[0].deployment_id in dominated_ids]
        if dominated_pairs:
            dominated_features = [item[1] for item in dominated_pairs]
            ranked_dominated = active_policy.rank_candidates(dominated_features)
            dominated_map = {item[0].deployment_id: item[0] for item in dominated_pairs}
            for feat, score_rec in ranked_dominated:
                if feat.candidate_id in dominated_map:
                    ranked_candidates.append((dominated_map[feat.candidate_id], score_rec))

        # 6b. vLLM Semantic Router Policy Evaluation & Hard Admission Authority
        router_metadata: dict[str, Any] = {}
        if policy_name == "vllm_semantic_router":
            router_messages = []
            if getattr(request, "messages", None):
                for m in request.messages:
                    if isinstance(m, dict):
                        router_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
                    else:
                        router_messages.append(
                            {"role": getattr(m, "role", "user"), "content": getattr(m, "content", "")}
                        )
            else:
                router_messages = [{"role": "user", "content": prompt_text}]

            router_res = self.vllm_router_client.preview_route(
                messages=router_messages,
                model="vllm-sr/auto",
                session_id=eff_session_id,
            )

            router_type = router_res.raw_payload.get("router_type") or (
                "bundled_heuristic" if "50050" in self.vllm_router_client.router_url else "upstream_preview"
            )
            router_metadata = {
                "router_success": router_res.success,
                "router_status": router_res.selection_status,
                "router_selected_model": router_res.selected_model,
                "router_confidence": router_res.confidence,
                "router_reason": router_res.reason,
                "router_algorithm": router_res.algorithm,
                "router_latency_ms": round(router_res.latency_ms, 2),
                "router_type": router_type,
                "actual_policy_used": f"vllm_semantic_router_{router_type}",
            }
            if router_res.error:
                router_metadata["router_error"] = router_res.error

            matched_cand_dep = None
            if router_res.success and router_res.selected_model:
                s_clean = router_res.selected_model.lower().strip()
                for c_dep, _ in eligible_candidates:
                    caps = getattr(c_dep.model, "capability_profiles", {}) or getattr(c_dep.model, "capabilities", {})
                    cap_list = list(caps.keys()) if isinstance(caps, dict) else list(caps)
                    # Strict identity matching: exact deployment_id, exact model_id, exact provider_model_name, or exact capability
                    if (
                        c_dep.deployment_id.lower() == s_clean
                        or c_dep.model_id.lower() == s_clean
                        or getattr(c_dep, "provider_model_name", "").lower() == s_clean
                        or any(cap.lower() == s_clean for cap in cap_list)
                    ):
                        matched_cand_dep = c_dep
                        break

            if matched_cand_dep is not None:
                ranked_candidates.sort(
                    key=lambda pair: 0 if pair[0].deployment_id == matched_cand_dep.deployment_id else 1
                )
                router_metadata["admission_status"] = "admitted"
                router_metadata["decision"] = (
                    f"vLLM-SR chose '{router_res.selected_model}', mapped to deployment '{matched_cand_dep.deployment_id}'"
                )
            else:
                # _, fb_intent, fb_conf, fb_reason = classify_semantic_intent(prompt_text, request.prompt_tokens)
                _, fb_intent, _, _ = classify_semantic_intent(prompt_text, request.prompt_tokens)
                if fb_intent == "fast_qa":
                    # Simple query: sort candidates prioritizing lowest cost & latency (efficiency)
                    ranked_candidates.sort(
                        key=lambda pair: (
                            pair[1].raw_values.get("cost", 0.0),
                            pair[1].raw_values.get("latency", 0.0),
                            -pair[0].get_capability_quality(request.task_type),
                        )
                    )
                elif fb_intent == "coding":
                    # Coding query: prioritize models with coding capability
                    ranked_candidates.sort(
                        key=lambda pair: 0 if "coding" in getattr(pair[0].model, "capabilities", {}) else 1
                    )
                else:
                    # Complex / reasoning query: prioritize highest quality
                    ranked_candidates.sort(key=lambda pair: -pair[0].get_capability_quality(request.task_type))

                if router_res.selected_model:
                    router_metadata["admission_status"] = "rejected_by_hard_constraints"
                    router_metadata["decision"] = (
                        f"vLLM-SR chose '{router_res.selected_model}' which violates AdaRoute hard constraints "
                        f"(privacy/budget/quality/mode). Safely fell back to intent-aware ranking (intent: {fb_intent})."
                    )
                else:
                    router_metadata["admission_status"] = "fallback_on_unavailability"
                    router_metadata["decision"] = (
                        f"vLLM-SR unavailable or timed out ({router_res.error}). Safely fell back to intent-aware ranking (intent: {fb_intent})."
                    )

        # 7. Build Explanation Record
        explanation = ExplanationBuilder.build(
            policy_name=policy_name,
            capability=request.task_type,
            signals=signals,
            rejected=rejected_candidates,
            dominated_ids=dominated_ids,
            ranked=ranked_candidates,
            router_metadata=router_metadata,
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
