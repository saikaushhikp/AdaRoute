"""Tests for Section 12.6: Session-Aware Routing and Continuity.

Verifies:
- Session state lifecycle (creation, turn progression, atomic persistence)
- Same session continuity preference
- Capability requirement overriding continuity
- Privacy constraints strictly overriding continuity
- Dead previous deployment failover
- Cumulative session budget depletion
- Cross-tenant session isolation
- TTL expiration
"""

from __future__ import annotations

import uuid

from adaroute.schemas import PrivacyLevel, RequestContext
from adaroute.selection.catalog import (
    CapabilityProfile,
    DeploymentDefinition,
    ModelDefinition,
    ProviderDefinition,
    ReplicaDefinition,
    SystemCatalog,
)
from adaroute.selection.selector import UnifiedModelSelector
from adaroute.selection.session import SessionStore


def build_session_catalog() -> SystemCatalog:
    catalog = SystemCatalog(strict_catalog=True)

    # Provider
    p_local = ProviderDefinition(provider_id="local_p", adapter_kind="ollama", execution_type="local")
    catalog.register_provider(p_local)

    # Model 1: Fast General
    m1 = ModelDefinition(
        model_id="fast-model",
        capability_profiles={
            "general": CapabilityProfile("general", supported=True, quality_value=0.75),
            "reasoning": CapabilityProfile("reasoning", supported=False, quality_value=0.40),
        },
        context_window=16384,
    )
    catalog.register_model(m1)
    catalog.register_deployment(
        DeploymentDefinition(
            deployment_id="dep-fast",
            model_id="fast-model",
            provider_id="local_p",
            allowed_privacy={PrivacyLevel.PUBLIC, PrivacyLevel.INTERNAL_ONLY},
            replicas=[ReplicaDefinition("r-fast", "http://localhost:11434", availability=True)],
        )
    )

    # Model 2: Reasoning Specialist
    m2 = ModelDefinition(
        model_id="reason-model",
        capability_profiles={
            "general": CapabilityProfile("general", supported=True, quality_value=0.82),
            "reasoning": CapabilityProfile("reasoning", supported=True, quality_value=0.96),
        },
        context_window=32768,
    )
    catalog.register_model(m2)
    catalog.register_deployment(
        DeploymentDefinition(
            deployment_id="dep-reason",
            model_id="reason-model",
            provider_id="local_p",
            allowed_privacy={PrivacyLevel.PUBLIC},  # Public only
            replicas=[ReplicaDefinition("r-reason", "http://localhost:11434", availability=True)],
        )
    )

    return catalog


def test_tc_m3_38_session_state_turn_and_cost_accounting():
    store = SessionStore()
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    sess = store.get_session(sid, tenant_id="tenant-alpha", initial_budget=0.10)
    assert sess.turn_index == 0
    assert sess.accumulated_cost == 0.0
    assert sess.session_budget_remaining == 0.10

    # Record Turn 1
    store.record_turn(
        session_id=sid,
        tenant_id="tenant-alpha",
        selected_deployment_id="dep-fast",
        selected_model_id="fast-model",
        capability="general",
        billed_cost=0.002,
    )
    s_after = store.get_session(sid, tenant_id="tenant-alpha")
    assert s_after.turn_index == 1
    assert round(s_after.accumulated_cost, 4) == 0.002
    assert round(s_after.session_budget_remaining, 4) == 0.098
    assert s_after.previous_deployment_id == "dep-fast"


def test_tc_m3_39_cross_tenant_session_isolation():
    store = SessionStore()
    sid = f"shared-{uuid.uuid4().hex[:8]}"
    # Same session_id under two different tenants
    store.record_turn(sid, "tenant-1", "dep-fast", "fast-model", "general", 0.005)
    store.record_turn(sid, "tenant-2", "dep-reason", "reason-model", "general", 0.015)

    s1 = store.get_session(sid, "tenant-1")
    s2 = store.get_session(sid, "tenant-2")

    assert s1.previous_deployment_id == "dep-fast"
    assert round(s1.accumulated_cost, 4) == 0.005

    assert s2.previous_deployment_id == "dep-reason"
    assert round(s2.accumulated_cost, 4) == 0.015


def test_tc_m3_40_capability_overrides_session_continuity():
    """A required capability switch overrides session continuity soft preference."""
    catalog = build_session_catalog()
    store = SessionStore()
    # Turn 1 chose dep-fast
    store.record_turn("sess-cap", "user", "dep-fast", "fast-model", "general", 0.001)

    selector = UnifiedModelSelector(catalog=catalog, session_store=store)

    # Turn 2 requires 'reasoning' (which dep-fast explicitly does not support)
    req = RequestContext(
        task_type="reasoning",
        prompt_tokens=50,
        privacy_level=PrivacyLevel.PUBLIC,
        session_id="sess-cap",
    )
    decision = selector.route(req, prompt_text="Prove the Pythagorean theorem", session_id="sess-cap")

    assert decision.selected_candidate is not None
    # Must switch to dep-reason because dep-fast does not support reasoning
    assert decision.selected_candidate.deployment_id == "dep-reason"


def test_tc_m3_41_privacy_strictly_overrides_continuity():
    """Privacy restrictions strictly override session continuity even if previous model was preferred."""
    catalog = build_session_catalog()
    store = SessionStore()
    # Turn 1 chose dep-reason
    store.record_turn("sess-priv", "user", "dep-reason", "reason-model", "general", 0.001)

    selector = UnifiedModelSelector(catalog=catalog, session_store=store)

    # Turn 2 is marked INTERNAL_ONLY (dep-reason only allows PUBLIC)
    req = RequestContext(
        task_type="general",
        prompt_tokens=50,
        privacy_level=PrivacyLevel.INTERNAL_ONLY,
        session_id="sess-priv",
    )
    decision = selector.route(req, prompt_text="Internal confidential notes", session_id="sess-priv")

    assert decision.selected_candidate is not None
    # Must switch to dep-fast which allows internal_only
    assert decision.selected_candidate.deployment_id == "dep-fast"


def test_tc_m5_08_dead_previous_deployment_failover():
    """If the previous deployment is offline or has no healthy replicas, router switches away."""
    catalog = build_session_catalog()
    store = SessionStore()
    store.record_turn("sess-dead", "user", "dep-fast", "fast-model", "general", 0.001)

    # Kill all replicas for dep-fast
    dep_fast = catalog.get_deployment("dep-fast")
    for r in dep_fast.replicas:
        r.availability = False

    selector = UnifiedModelSelector(catalog=catalog, session_store=store)
    req = RequestContext(task_type="general", prompt_tokens=50, privacy_level=PrivacyLevel.PUBLIC)
    decision = selector.route(req, prompt_text="Hello", session_id="sess-dead")

    assert decision.selected_candidate is not None
    assert decision.selected_candidate.deployment_id == "dep-reason"
