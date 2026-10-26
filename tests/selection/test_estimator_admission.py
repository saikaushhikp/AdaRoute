"""Tests for Section 12.3: Output Estimation and Admission.

Verifies:
- Deployment-specific priors
- Prompt-length bucket handling
- Explicit max_tokens capping
- Reservation bound >= expected output
- Minimum observations warm-up guard
- Online quantile adaptation after warm-up
- Deployment isolation (observations for Deployment A do not affect Deployment B)
- Bounded worst-case cost admission (C_max > B rejects even if C_est <= B)
- Context window capacity check
"""

from __future__ import annotations

from adaroute.schemas import PrivacyLevel, RequestContext
from adaroute.selection.candidates import CandidateDeployment
from adaroute.selection.catalog import (
    DeploymentDefinition,
    ModelDefinition,
    ProviderDefinition,
    SystemCatalog,
)
from adaroute.selection.constraints import HardConstraintEngine
from adaroute.selection.estimator import DeploymentOutputEstimator


def test_tc_m3_27_deployment_specific_prior():
    est = DeploymentOutputEstimator(
        default_prior=80,
        task_priors={"coding": 250},
        deployment_priors={"qwen-deploy": {"coding": 180}, "gemini-deploy": {"coding": 350}},
    )
    res_qwen = est.estimate("qwen-deploy", "coding", prompt_tokens=50)
    res_gemini = est.estimate("gemini-deploy", "coding", prompt_tokens=50)

    assert res_qwen.expected_tokens == 180
    assert res_gemini.expected_tokens == 350
    assert res_gemini.reservation_tokens >= res_gemini.expected_tokens


def test_tc_m3_28_prompt_length_bucket_adjustment():
    est = DeploymentOutputEstimator(default_prior=100)
    # Short prompt (<100 tokens)
    short_res = est.estimate("dep-1", "general", prompt_tokens=30)
    # Long prompt (>500 tokens) gets 1.25 expansion
    long_res = est.estimate("dep-1", "general", prompt_tokens=600)

    assert short_res.expected_tokens == 100
    assert long_res.expected_tokens == 125


def test_tc_m3_29_explicit_max_tokens_capping():
    est = DeploymentOutputEstimator(default_prior=300)
    # If client requests max_tokens = 60, expected must clamp to 60
    res = est.estimate("dep-1", "general", prompt_tokens=50, max_tokens=60)
    assert res.expected_tokens == 60
    assert res.reservation_tokens == 60
    assert res.strategy == "explicit_max"


def test_tc_m3_30_historical_warmup_and_adaptation():
    est = DeploymentOutputEstimator(default_prior=100, minimum_observations=5, expected_quantile=0.75)

    # 4 observations (< minimum_observations 5): should NOT adapt yet
    for _ in range(4):
        est.record_observation("dep-1", "qa", prompt_tokens=50, actual_output_tokens=30)
    res_before = est.estimate("dep-1", "qa", prompt_tokens=50)
    assert res_before.strategy == "configured_prior"
    assert res_before.expected_tokens == 100

    # 5th observation: minimum observations reached, adapts via quantile!
    est.record_observation("dep-1", "qa", prompt_tokens=50, actual_output_tokens=30)
    res_after = est.estimate("dep-1", "qa", prompt_tokens=50)
    assert res_after.strategy == "online_quantile"
    assert res_after.expected_tokens == 30


def test_tc_m3_31_deployment_isolation_in_estimation():
    est = DeploymentOutputEstimator(default_prior=100, minimum_observations=5)
    # Record 10 observations for dep-qwen
    for _ in range(10):
        est.record_observation("dep-qwen", "qa", prompt_tokens=50, actual_output_tokens=40)

    res_qwen = est.estimate("dep-qwen", "qa", prompt_tokens=50)
    res_other = est.estimate("dep-gemini", "qa", prompt_tokens=50)

    # dep-qwen adapted to 40
    assert res_qwen.strategy == "online_quantile"
    assert res_qwen.expected_tokens == 40

    # dep-gemini remains at configured prior 100
    assert res_other.strategy == "configured_prior"
    assert res_other.expected_tokens == 100


def test_tc_m3_32_worst_case_cost_and_context_rejection():
    """C_max > budget rejects candidate even if C_est <= budget."""
    catalog = SystemCatalog(strict_catalog=True)
    p = ProviderDefinition(provider_id="p1", adapter_kind="mock", execution_type="mock")
    catalog.register_provider(p)
    m = ModelDefinition(model_id="m1", context_window=1000)
    catalog.register_model(m)
    dep = DeploymentDefinition(
        deployment_id="d1",
        model_id="m1",
        provider_id="p1",
        input_price_per_1k=0.01,
        output_price_per_1k=0.02,
        allowed_privacy={PrivacyLevel.PUBLIC},
    )
    catalog.register_deployment(dep)
    cand = CandidateDeployment(dep, m, p, catalog.get_snapshot("d1"))

    engine = HardConstraintEngine(execution_mode="hybrid")
    est = DeploymentOutputEstimator(default_prior=50)
    est_out = est.estimate("d1", "general", prompt_tokens=100)

    # 1. Budget where expected cost ($0.002) <= budget ($0.005) but max cost (max_tokens=500 -> $0.011) > budget
    req_budget = RequestContext(
        task_type="general",
        prompt_tokens=100,
        budget=0.005,
        max_tokens=500,
        privacy_level=PrivacyLevel.PUBLIC,
    )
    dec_budget = engine.evaluate(cand, req_budget, est_out)
    assert dec_budget.is_eligible is False
    assert any("Worst-case budget exceeded" in r for r in dec_budget.rejection_reasons)

    # 2. Context window capacity check: prompt (800) + reservation (300) > context_window (1000)
    req_context = RequestContext(
        task_type="general",
        prompt_tokens=800,
        budget=1.0,
        max_tokens=300,
        privacy_level=PrivacyLevel.PUBLIC,
    )
    dec_context = engine.evaluate(cand, req_context, est_out)
    assert dec_context.is_eligible is False
    assert any("Context window exceeded" in r for r in dec_context.rejection_reasons)
