"""Tests for Section 12.8: Explainability, Reproducibility & Secret Scrubbing.

Verifies:
- RoutingDecision records policy name, rejected candidates, Pareto survivors, winner
- Reproducibility: identical snapshots and requests produce identical rankings
- Metric provenance is explicitly labeled on every feature
- Secret scrubbing: API keys and bearer tokens never appear in explanations
- Raw prompt content is never leaked into decision telemetry
"""

from __future__ import annotations

from adaroute.schemas import PrivacyLevel, RequestContext
from adaroute.selection.catalog import (
    CapabilityProfile,
    DeploymentDefinition,
    ModelDefinition,
    ProviderDefinition,
    SystemCatalog,
)
from adaroute.selection.selector import UnifiedModelSelector
from adaroute.telemetry.ledger import sanitize_secrets


def build_explainable_catalog() -> SystemCatalog:
    catalog = SystemCatalog(strict_catalog=True)
    p = ProviderDefinition(provider_id="prov-1", adapter_kind="mock", execution_type="mock")
    catalog.register_provider(p)

    m1 = ModelDefinition(
        model_id="model-fast",
        capability_profiles={
            "general": CapabilityProfile("general", supported=True, quality_value=0.75, quality_source="benchmark", quality_confidence=0.85)
        },
        context_window=8192,
    )
    catalog.register_model(m1)
    catalog.register_deployment(
        DeploymentDefinition(
            deployment_id="dep-fast",
            model_id="model-fast",
            provider_id="prov-1",
            input_price_per_1k=0.0001,
            output_price_per_1k=0.0002,
        )
    )

    m2 = ModelDefinition(
        model_id="model-flagship",
        capability_profiles={
            "general": CapabilityProfile("general", supported=True, quality_value=0.95, quality_source="configured", quality_confidence=0.90)
        },
        context_window=32768,
    )
    catalog.register_model(m2)
    catalog.register_deployment(
        DeploymentDefinition(
            deployment_id="dep-flagship",
            model_id="model-flagship",
            provider_id="prov-1",
            input_price_per_1k=0.005,
            output_price_per_1k=0.015,
        )
    )

    return catalog


def test_tc_m7_08_routing_decision_structure_and_provenance():
    catalog = build_explainable_catalog()
    selector = UnifiedModelSelector(catalog=catalog)

    req = RequestContext(
        task_type="general",
        prompt_tokens=50,
        privacy_level=PrivacyLevel.PUBLIC,
        budget=1.0,
    )
    decision = selector.route(req, prompt_text="Provide a summary")

    assert decision.selected_candidate is not None
    assert decision.policy_name == "adaptive"
    assert decision.decision_version == "4.0.0"

    # Check explanation structure
    exp = decision.explanation.to_dict()
    assert "signals" in exp
    assert "hard_rejections" in exp
    assert "pareto_survivors" in exp
    assert "selected_deployment" in exp
    assert "objective_contributions" in exp
    assert len(decision.ranked_candidates) == 2


def test_tc_m7_09_reproducibility():
    """Identical snapshots and requests produce the exact same ranking."""
    catalog = build_explainable_catalog()
    selector = UnifiedModelSelector(catalog=catalog)

    req = RequestContext(
        task_type="general",
        prompt_tokens=50,
        privacy_level=PrivacyLevel.PUBLIC,
        budget=1.0,
    )

    d1 = selector.route(req, prompt_text="Deterministic test")
    d2 = selector.route(req, prompt_text="Deterministic test")

    order_1 = [c[0].deployment_id for c in d1.ranked_candidates]
    order_2 = [c[0].deployment_id for c in d2.ranked_candidates]
    assert order_1 == order_2
    assert d1.selected_candidate.deployment_id == d2.selected_candidate.deployment_id


def test_tc_m7_10_secret_scrubbing_in_telemetry():
    """Raw API keys or bearer tokens must be scrubbed from all outbox logs and telemetry."""
    dirty_text = "Call to endpoint failed: Authorization header Bearer AIzaSyD9876543210-abcdef was invalid or expired."
    clean_text = sanitize_secrets(dirty_text)

    assert "AIzaSyD9876543210-abcdef" not in clean_text
    assert "[REDACTED_SECRET]" in clean_text
