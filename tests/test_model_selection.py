import pytest

from adaroute.api import registry
from adaroute.policy.model_selection import (
    AdaptiveModelPolicy,
    CheapestModelPolicy,
    StaticRulePolicy,
    StrongestModelPolicy,
)
from adaroute.schemas import PrivacyLevel, RequestContext


@pytest.fixture
def policies():
    return {
        "strongest": StrongestModelPolicy(registry),
        "cheapest": CheapestModelPolicy(registry),
        "static": StaticRulePolicy(registry),
        "adaptive": AdaptiveModelPolicy(registry),
    }


def test_tc_m3_01_strongest_distinct_from_cheapest(policies):
    """[TC_M3_01] Milestone M3: Strongest and Cheapest must NOT be identical stubs.

    When both local and external are eligible, Strongest picks GPT-4o and Cheapest picks local.
    """
    req = RequestContext(
        task_type="coding",
        prompt_tokens=200,
        estimated_output_tokens=150,
        privacy_level=PrivacyLevel.PUBLIC,
        budget=1.0,
    )

    strong_spec, strong_cost, _ = policies["strongest"].select_model(req)
    cheap_spec, cheap_cost, _ = policies["cheapest"].select_model(req)

    assert strong_spec is not None
    assert cheap_spec is not None

    # Invariant: Distinct models selected!
    assert strong_spec.model_id != cheap_spec.model_id
    assert strong_spec.model_id == "gpt-4o-external"
    assert cheap_spec.model_id == "qwen-2.5-local"

    # Invariant: Quality and cost orderings hold
    assert strong_cost > cheap_cost
    assert (
        strong_spec.quality_scores["coding"]
        > cheap_spec.quality_scores["coding"]
    )


def test_tc_m3_02_static_rule_follows_task_mapping(policies):
    """[TC_M3_02] Milestone M3: StaticRulePolicy routes coding -> external, qa -> local."""
    req_coding = RequestContext(
        task_type="coding",
        prompt_tokens=100,
        estimated_output_tokens=100,
        privacy_level=PrivacyLevel.PUBLIC,
        budget=1.0,
    )
    req_qa = RequestContext(
        task_type="qa",
        prompt_tokens=100,
        estimated_output_tokens=50,
        privacy_level=PrivacyLevel.PUBLIC,
        budget=1.0,
    )

    spec_code, _, _ = policies["static"].select_model(req_coding)
    spec_qa, _, _ = policies["static"].select_model(req_qa)

    assert spec_code.model_id == "gpt-4o-external"
    assert spec_qa.model_id == "qwen-2.5-local"


def test_tc_m3_03_adaptive_policy_balances_objectives(policies):
    """[TC_M3_03] Milestone M3: Adaptive policy adapts model selection based on cost weight

    versus quality weight.
    """
    req = RequestContext(
        task_type="general",
        prompt_tokens=100,
        estimated_output_tokens=50,
        privacy_level=PrivacyLevel.PUBLIC,
        budget=1.0,
    )

    # Cost-heavy adaptive policy favors cheap local model
    cost_heavy_adaptive = AdaptiveModelPolicy(registry, weights={"cost": 0.8, "quality": 0.1, "latency": 0.1, "slo": 0.0, "failure": 0.0})
    spec_cost, _, _ = cost_heavy_adaptive.select_model(req)
    assert spec_cost.model_id == "qwen-2.5-local"

    # Quality-heavy adaptive policy favors strongest model
    quality_heavy_adaptive = AdaptiveModelPolicy(registry, weights={"cost": 0.1, "quality": 0.8, "latency": 0.1, "slo": 0.0, "failure": 0.0})
    spec_qual, _, _ = quality_heavy_adaptive.select_model(req)
    assert spec_qual.model_id == "gpt-4o-external"
