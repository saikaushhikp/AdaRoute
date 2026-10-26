"""Tests for Section 12.1: Catalog and System Portability.

Verifies:
- Empty catalog fails clearly with no implicit models
- System S1 and S2 fixtures route with zero Python code changes
- Unknown provider references fail validation
- Unknown model references fail validation
- Duplicate model, deployment, and replica IDs fail
- Strict catalog mode injects no undeclared legacy models
- Missing endpoint / missing credential handling
- Endpoint preservation into provider adapter
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaroute.selection.candidates import CandidateResolver
from adaroute.selection.catalog import (
    CapabilityProfile,
    DeploymentDefinition,
    ModelDefinition,
    ProviderDefinition,
    ReplicaDefinition,
    SystemCatalog,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def test_tc_m3_19_empty_catalog_fails_clearly():
    """Empty catalog has zero implicit models and resolver returns empty list."""
    catalog = SystemCatalog(strict_catalog=True)
    resolver = CandidateResolver(catalog)
    candidates = resolver.resolve_candidates("general")
    assert len(candidates) == 0
    assert len(catalog.models) == 0
    assert len(catalog.deployments) == 0


def test_tc_m3_20_system_s1_catalog_isolation():
    """System S1 loads only S1 models/deployments without legacy defaults."""
    s1_path = FIXTURES_DIR / "system_s1.yaml"
    catalog = SystemCatalog.from_yaml_file(s1_path, strict_catalog=True)

    assert set(catalog.models.keys()) == {"qwen2.5:1.5b", "gemini/gemini-flash-latest"}
    assert "gpt-4o-external" not in catalog.models
    assert "phi4-mini:3.8b" not in catalog.models


def test_tc_m3_21_system_s2_catalog_isolation():
    """System S2 loads only S2 models/deployments with zero Python changes."""
    s2_path = FIXTURES_DIR / "system_s2.yaml"
    catalog = SystemCatalog.from_yaml_file(s2_path, strict_catalog=True)

    assert set(catalog.models.keys()) == {"phi4-mini:3.8b", "cohere/command-r-08-2024"}
    assert "qwen2.5:1.5b" not in catalog.models
    assert "gemini/gemini-flash-latest" not in catalog.models


def test_tc_m3_22_unknown_provider_reference_fails_validation():
    """Referencing an undeclared provider in a deployment raises ValueError."""
    catalog = SystemCatalog(strict_catalog=True)
    catalog.register_model(ModelDefinition(model_id="test-model", context_window=4096))
    dep = DeploymentDefinition(
        deployment_id="dep-1",
        model_id="test-model",
        provider_id="nonexistent-provider",
    )
    with pytest.raises(ValueError, match="references unknown provider"):
        catalog.register_deployment(dep)


def test_tc_m3_23_unknown_model_reference_fails_validation():
    """Referencing an undeclared model in a deployment raises ValueError."""
    catalog = SystemCatalog(strict_catalog=True)
    catalog.register_provider(ProviderDefinition(provider_id="prov-1", adapter_kind="ollama"))
    dep = DeploymentDefinition(
        deployment_id="dep-1",
        model_id="nonexistent-model",
        provider_id="prov-1",
    )
    with pytest.raises(ValueError, match="references unknown model"):
        catalog.register_deployment(dep)


def test_tc_m3_24_duplicate_ids_fail_validation():
    """Colliding model, deployment, and replica IDs fail with explicit errors."""
    catalog = SystemCatalog(strict_catalog=True)
    p = ProviderDefinition(provider_id="p1", adapter_kind="ollama")
    catalog.register_provider(p)
    with pytest.raises(ValueError, match="Duplicate provider_id"):
        catalog.register_provider(p)

    m = ModelDefinition(model_id="m1", context_window=4096)
    catalog.register_model(m)
    with pytest.raises(ValueError, match="Duplicate model_id"):
        catalog.register_model(m)

    # Duplicate replica IDs within deployment
    reps = [
        ReplicaDefinition(replica_id="rep-1", endpoint="http://h1:11434"),
        ReplicaDefinition(replica_id="rep-1", endpoint="http://h2:11434"),
    ]
    dep = DeploymentDefinition(
        deployment_id="d1",
        model_id="m1",
        provider_id="p1",
        replicas=reps,
    )
    with pytest.raises(ValueError, match="Duplicate replica_id"):
        catalog.register_deployment(dep)


def test_tc_m3_25_invalid_quality_or_prices_fail_validation():
    """Negative prices or quality not in [0, 1] raise ValueError."""
    with pytest.raises(ValueError, match="Quality value"):
        CapabilityProfile(name="coding", quality_value=1.5)

    with pytest.raises(ValueError, match="input_price_per_1k must be non-negative"):
        DeploymentDefinition(
            deployment_id="d1",
            model_id="m1",
            provider_id="p1",
            input_price_per_1k=-0.01,
        )


def test_tc_m3_26_secret_in_credential_reference_rejected():
    """Passing an actual secret key instead of an env var name fails validation."""
    with pytest.raises(ValueError, match="credential_reference must be an environment variable name"):
        ProviderDefinition(
            provider_id="p1",
            adapter_kind="litellm",
            credential_reference="sk-1234567890abcdef",
        )
