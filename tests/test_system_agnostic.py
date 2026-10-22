"""System-Agnostic Model Selection & Configuration Validation Tests.

Proves that AdaRoute is a true system-agnostic AI Gateway:
- Switching between System S1 (Edge Workstation with Ollama & Gemini) and
  System S2 (Enterprise Node with Phi4-mini & Cohere) requires ZERO Python code changes.
- All model lists, capabilities, replicas, endpoints, credentials, estimation priors,
  and multi-objective policy weights are driven purely by declarative config.yaml & .env.
- Cost factors, budget enforcement, and telemetry accounting are strictly preserved.
"""

from __future__ import annotations

from pathlib import Path

from adaroute.policy.eligibility import EligibilityChecker
from adaroute.policy.model_selection import (
    AdaptiveModelPolicy,
    StaticRulePolicy,
)
from adaroute.policy.registry import ModelRegistry
from adaroute.schemas import PrivacyLevel, RequestContext
from adaroute.workloads.estimator import OutputLengthEstimator

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_tc_m3_11_system_s1_declarative_routing():
    """[TC_M3_11] System S1: Edge Workstation with Ollama Qwen and Google Gemini."""
    s1_config = FIXTURES_DIR / "system_s1.yaml"
    assert s1_config.exists()

    registry = ModelRegistry(config_path=s1_config)

    # 1. Model Catalog Isolation: Only S1 models configured
    s1_model_ids = {m.model_id for m in registry.get_all_models()}
    assert s1_model_ids == {"qwen2.5:1.5b", "gemini/gemini-flash-latest"}
    # Invariant: No legacy hardcoded or S2 models injected
    assert "phi4-mini:3.8b" not in s1_model_ids
    assert "gpt-4o-external" not in s1_model_ids
    assert "llama-3-8b" not in s1_model_ids

    # 2. Config-Driven Estimator Priors for S1
    estimator = OutputLengthEstimator.from_config(registry.raw_config)
    assert estimator.task_priors["coding"] == 320
    assert estimator.default_prior == 90
    assert estimator.max_token_ratio == 0.18

    # Estimator produces S1-specific expected length
    est_coding = estimator.estimate(task_type="coding", prompt_tokens=50)
    assert est_coding == 320

    # 3. Config-Driven Policy Weights for S1 (Quality-first priorities: w_q=0.70, w_c=0.05)
    policy = AdaptiveModelPolicy(registry, policy_name="adaptive")
    assert policy.weights["quality"] == 0.70
    assert policy.weights["cost"] == 0.05

    # 4. Route Public Coding Request: Quality weight favors Gemini
    req_public = RequestContext(
        task_type="coding",
        prompt_tokens=100,
        estimated_output_tokens=est_coding,
        privacy_level=PrivacyLevel.PUBLIC,
        budget=1.0,
    )
    ranked = policy.rank_models(req_public)
    assert len(ranked) == 2
    # Gemini (quality 0.89) ranked first under quality-weighted policy
    assert ranked[0][0].model_id == "gemini/gemini-flash-latest"

    # 5. Route Private Request: User private toggle or internal privacy isolates to local Qwen
    req_private = RequestContext(
        task_type="coding",
        prompt_tokens=100,
        estimated_output_tokens=est_coding,
        privacy_level=PrivacyLevel.INTERNAL_ONLY,
        budget=1.0,
    )
    ranked_private = policy.rank_models(req_private)
    assert len(ranked_private) == 1
    assert ranked_private[0][0].model_id == "qwen2.5:1.5b"


def test_tc_m3_12_system_s2_declarative_routing():
    """[TC_M3_12] System S2: Enterprise On-Prem Node with Phi4-mini and Cohere Command R."""
    s2_config = FIXTURES_DIR / "system_s2.yaml"
    assert s2_config.exists()

    registry = ModelRegistry(config_path=s2_config)

    # 1. Model Catalog Isolation: Only S2 models configured
    s2_model_ids = {m.model_id for m in registry.get_all_models()}
    assert s2_model_ids == {"phi4-mini:3.8b", "cohere/command-r-08-2024"}
    # Invariant: No S1 models or legacy defaults exist
    assert "qwen2.5:1.5b" not in s2_model_ids
    assert "gemini/gemini-flash-latest" not in s2_model_ids
    assert "gpt-4o-external" not in s2_model_ids

    # 2. Config-Driven Estimator Priors for S2
    estimator = OutputLengthEstimator.from_config(registry.raw_config)
    assert estimator.task_priors["coding"] == 190
    assert estimator.default_prior == 65
    assert estimator.max_token_ratio == 0.12

    est_coding = estimator.estimate(task_type="coding", prompt_tokens=50)
    assert est_coding == 190

    # 3. Config-Driven Policy Weights for S2 (Cost-first priorities: w_c=0.60, w_q=0.15)
    policy = AdaptiveModelPolicy(registry, policy_name="adaptive")
    assert policy.weights["cost"] == 0.60
    assert policy.weights["quality"] == 0.15

    # 4. Route Public Coding Request: Cost-first weight favors local Phi4-mini
    req_public = RequestContext(
        task_type="coding",
        prompt_tokens=100,
        estimated_output_tokens=est_coding,
        privacy_level=PrivacyLevel.PUBLIC,
        budget=1.0,
    )
    ranked = policy.rank_models(req_public)
    assert len(ranked) == 2
    # Phi4-mini (cost $0.000005) ranked first under cost-heavy policy
    assert ranked[0][0].model_id == "phi4-mini:3.8b"

    # 5. Static Policy uses S2 static_task_map
    static_policy = StaticRulePolicy(registry)
    ranked_static = static_policy.rank_models(req_public)
    assert ranked_static[0][0].model_id == "phi4-mini:3.8b"


def test_tc_m3_13_cost_factors_strictly_preserved_for_telemetry_and_budgets():
    """[TC_M3_13] Cost factors remain active for budget enforcement and telemetry."""
    s1_config = FIXTURES_DIR / "system_s1.yaml"
    registry = ModelRegistry(config_path=s1_config)
    checker = EligibilityChecker(registry)

    gemini_spec = registry.get_model("gemini/gemini-flash-latest")
    assert gemini_spec is not None
    assert gemini_spec.input_price_per_1k == 0.000075
    assert gemini_spec.output_price_per_1k == 0.000300

    # Pre-inference cost estimation is accurate and non-zero
    est_cost = checker.estimate_cost(gemini_spec, prompt_tokens=1000, estimated_output_tokens=500)
    expected_cost = (1000 / 1000.0) * 0.000075 + (500 / 1000.0) * 0.000300
    assert abs(est_cost - expected_cost) < 1e-9

    # Strict budget rejection when request budget is exceeded
    req_low_budget = RequestContext(
        task_type="general",
        prompt_tokens=1000,
        estimated_output_tokens=500,
        budget=0.00005,  # Lower than est_cost
    )
    is_eligible, reason, _ = checker.check_model_eligibility(gemini_spec, req_low_budget)
    assert is_eligible is False
    assert "Budget exceeded" in reason


def test_tc_m3_14_online_historical_quantile_adaptation():
    """[TC_M3_14] Estimator updates running observations without future leakage."""
    estimator = OutputLengthEstimator(
        task_priors={"coding": 200},
        default_prior=80,
        historical_quantile=0.80,
    )

    # Initial prior estimate
    initial_est = estimator.estimate("coding", prompt_tokens=50)
    assert initial_est == 200

    # Record 5 actual post-inference observations (e.g. coding queries were concise: 50, 60, 55, 65, 70)
    for actual in [50, 60, 55, 65, 70]:
        estimator.record_observation("coding", estimated=initial_est, actual=actual)

    # Adaptive quantile kicks in: estimate now reflects empirical 80th percentile of actual traffic
    adapted_est = estimator.estimate("coding", prompt_tokens=50)
    assert adapted_est <= 70  # Shifted from 200 towards the empirical distribution!

    stats = estimator.get_stats()
    assert stats["samples"] == 5
    assert "mae_coding" in stats
