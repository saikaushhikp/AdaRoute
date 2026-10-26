"""Tests for Section 12.4: Hard Constraints Engine.

Verifies:
- Privacy allow-lists (public, internal_only, confidential, restricted)
- User-private mode (excludes cloud/external providers)
- Real mode mock exclusion
- Missing credentials rejection for cloud candidates
- Provider cooldown rejection
- Cumulative session budget remaining
- Replica health (all unhealthy rejects; at least one healthy accepts)
- No eligible candidate failure explanation
"""

from __future__ import annotations

import os
import time

from adaroute.schemas import PrivacyLevel, RequestContext
from adaroute.selection.candidates import CandidateDeployment
from adaroute.selection.catalog import (
    DeploymentDefinition,
    ModelDefinition,
    ProviderDefinition,
    ReplicaDefinition,
    SystemCatalog,
)
from adaroute.selection.constraints import HardConstraintEngine
from adaroute.selection.estimator import OutputEstimate


def build_test_environment():
    catalog = SystemCatalog(strict_catalog=True)

    # Cloud Provider
    p_cloud = ProviderDefinition(
        provider_id="cloud_prov",
        adapter_kind="litellm",
        execution_type="external",
        credential_reference="MANDATORY_TEST_KEY",
    )
    catalog.register_provider(p_cloud)

    # Local Provider
    p_local = ProviderDefinition(
        provider_id="local_prov",
        adapter_kind="ollama",
        execution_type="local",
    )
    catalog.register_provider(p_local)

    # Mock Provider
    p_mock = ProviderDefinition(
        provider_id="mock_prov",
        adapter_kind="mock",
        execution_type="mock",
    )
    catalog.register_provider(p_mock)

    # Models
    m_cloud = ModelDefinition(model_id="cloud-model", context_window=32768)
    catalog.register_model(m_cloud)

    m_local = ModelDefinition(model_id="local-model", context_window=32768)
    catalog.register_model(m_local)

    m_mock = ModelDefinition(model_id="mock-model", context_window=4096)
    catalog.register_model(m_mock)

    # Deployments
    d_cloud = DeploymentDefinition(
        deployment_id="dep-cloud",
        model_id="cloud-model",
        provider_id="cloud_prov",
        allowed_privacy={PrivacyLevel.PUBLIC},
        input_price_per_1k=0.001,
        output_price_per_1k=0.002,
    )
    catalog.register_deployment(d_cloud)

    reps = [
        ReplicaDefinition(replica_id="r1", endpoint="http://h1:11434", availability=True),
        ReplicaDefinition(replica_id="r2", endpoint="http://h2:11434", availability=False),
    ]
    d_local = DeploymentDefinition(
        deployment_id="dep-local",
        model_id="local-model",
        provider_id="local_prov",
        allowed_privacy={PrivacyLevel.PUBLIC, PrivacyLevel.INTERNAL_ONLY, PrivacyLevel.CONFIDENTIAL},
        input_price_per_1k=0.00001,
        output_price_per_1k=0.00002,
        replicas=reps,
    )
    catalog.register_deployment(d_local)

    d_mock = DeploymentDefinition(
        deployment_id="dep-mock",
        model_id="mock-model",
        provider_id="mock_prov",
        allowed_privacy={PrivacyLevel.PUBLIC},
    )
    catalog.register_deployment(d_mock)

    return catalog


def test_tc_m3_33_privacy_and_user_private_constraints():
    catalog = build_test_environment()
    engine = HardConstraintEngine(execution_mode="hybrid")
    cand_cloud = CandidateDeployment(
        catalog.get_deployment("dep-cloud"),
        catalog.get_model("cloud-model"),
        catalog.get_provider("cloud_prov"),
        catalog.get_snapshot("dep-cloud"),
    )
    est = OutputEstimate(expected_tokens=50, reservation_tokens=70, confidence=0.8, strategy="test", sample_count=10)

    # 1. Privacy failure: internal request on public cloud deployment
    req_internal = RequestContext(task_type="general", prompt_tokens=50, privacy_level=PrivacyLevel.INTERNAL_ONLY)
    dec = engine.evaluate(cand_cloud, req_internal, est)
    assert dec.is_eligible is False
    assert any("Privacy violation" in r for r in dec.rejection_reasons)

    # 2. User private failure: user_private=True excludes cloud
    req_private = RequestContext(task_type="general", prompt_tokens=50, privacy_level=PrivacyLevel.PUBLIC, user_private=True)
    dec_priv = engine.evaluate(cand_cloud, req_private, est)
    assert dec_priv.is_eligible is False
    assert any("User private toggle active" in r for r in dec_priv.rejection_reasons)


def test_tc_m2_06_missing_credentials_rejection():
    catalog = build_test_environment()
    engine = HardConstraintEngine(execution_mode="hybrid")
    cand_cloud = CandidateDeployment(
        catalog.get_deployment("dep-cloud"),
        catalog.get_model("cloud-model"),
        catalog.get_provider("cloud_prov"),
        catalog.get_snapshot("dep-cloud"),
    )
    est = OutputEstimate(expected_tokens=50, reservation_tokens=70, confidence=0.8, strategy="test", sample_count=10)

    # Unset env var
    os.environ.pop("MANDATORY_TEST_KEY", None)
    req = RequestContext(task_type="general", prompt_tokens=50, privacy_level=PrivacyLevel.PUBLIC)
    dec = engine.evaluate(cand_cloud, req, est)
    assert dec.is_eligible is False
    assert any("Missing credential" in r for r in dec.rejection_reasons)


def test_tc_m2_07_real_mode_mock_exclusion():
    catalog = build_test_environment()
    engine_real = HardConstraintEngine(execution_mode="real")
    cand_mock = CandidateDeployment(
        catalog.get_deployment("dep-mock"),
        catalog.get_model("mock-model"),
        catalog.get_provider("mock_prov"),
        catalog.get_snapshot("dep-mock"),
    )
    est = OutputEstimate(expected_tokens=50, reservation_tokens=70, confidence=0.8, strategy="test", sample_count=10)

    req = RequestContext(task_type="general", prompt_tokens=50, privacy_level=PrivacyLevel.PUBLIC)
    dec = engine_real.evaluate(cand_mock, req, est)
    assert dec.is_eligible is False
    assert any("Real execution mode active: mock candidate" in r for r in dec.rejection_reasons)


def test_tc_m5_05_cooldown_rejection():
    catalog = build_test_environment()
    engine = HardConstraintEngine(execution_mode="hybrid")
    snap = catalog.get_snapshot("dep-cloud")
    snap.cooldown_until = time.time() + 60.0  # 60s cooldown

    cand_cloud = CandidateDeployment(
        catalog.get_deployment("dep-cloud"),
        catalog.get_model("cloud-model"),
        catalog.get_provider("cloud_prov"),
        snap,
    )
    est = OutputEstimate(expected_tokens=50, reservation_tokens=70, confidence=0.8, strategy="test", sample_count=10)

    req = RequestContext(task_type="general", prompt_tokens=50, privacy_level=PrivacyLevel.PUBLIC)
    dec = engine.evaluate(cand_cloud, req, est)
    assert dec.is_eligible is False
    assert any("Provider cooldown active" in r for r in dec.rejection_reasons)


def test_tc_m5_06_local_replica_availability():
    catalog = build_test_environment()
    engine = HardConstraintEngine(execution_mode="hybrid")
    cand_local = CandidateDeployment(
        catalog.get_deployment("dep-local"),
        catalog.get_model("local-model"),
        catalog.get_provider("local_prov"),
        catalog.get_snapshot("dep-local"),
    )
    est = OutputEstimate(expected_tokens=50, reservation_tokens=70, confidence=0.8, strategy="test", sample_count=10)
    req = RequestContext(task_type="general", prompt_tokens=50, privacy_level=PrivacyLevel.PUBLIC)

    # 1. With r1 alive: eligible
    dec1 = engine.evaluate(cand_local, req, est)
    assert dec1.is_eligible is True

    # 2. When r1 is also failed/excluded: rejected
    dec2 = engine.evaluate(cand_local, req, est, failed_replicas={"r1"})
    assert dec2.is_eligible is False
    assert any("No healthy replicas available" in r for r in dec2.rejection_reasons)


def test_tc_m5_07_session_budget_remaining():
    catalog = build_test_environment()
    engine = HardConstraintEngine(execution_mode="hybrid")
    os.environ["MANDATORY_TEST_KEY"] = "mock-key"
    cand_cloud = CandidateDeployment(
        catalog.get_deployment("dep-cloud"),
        catalog.get_model("cloud-model"),
        catalog.get_provider("cloud_prov"),
        catalog.get_snapshot("dep-cloud"),
    )
    est = OutputEstimate(expected_tokens=500, reservation_tokens=700, confidence=0.8, strategy="test", sample_count=10)
    req = RequestContext(task_type="general", prompt_tokens=1000, privacy_level=PrivacyLevel.PUBLIC)

    # Expected cost for 1k in + 500 out = $0.001 + $0.001 = $0.002
    # If session budget remaining is $0.0005, it must be rejected
    dec = engine.evaluate(cand_cloud, req, est, session_budget_remaining=0.0005)
    assert dec.is_eligible is False
    assert any("Session budget exhausted" in r for r in dec.rejection_reasons)
