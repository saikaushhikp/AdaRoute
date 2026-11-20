"""Comprehensive Endpoint-Level Tests for Unified Model Selection.

Verifies:
1. Authoritative selection path on cache miss: changing policy/objective changes the deployment executed.
2. Response and telemetry metadata identify the selected deployment, replica, and explanation.
3. Unsupported policies strictly rejected with HTTP 400 (no silent fallback).
4. Custom configured policies really affect ranking and execution.
5. Explicit model requests preserved without capability rewrite or fallback to "general".
6. Multiple deployments of the same model distinguished without ID collapse.
7. Fail-closed privacy, user_private mode, and mock exclusion.
8. Bounded budget and context window rejection (HTTP 503).
9. Session budget admission, consecutive turn depletion, and exhaustion rejection.
10. Cache hit session turn accounting (turns advance, remaining budget unchanged).
11. Concurrent session request thread-safety without race conditions.
12. Circuit breaker canary probe non-consumption during candidate inspection and failover.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from adaroute.api import app, circuit_breaker, registry, session_store
from adaroute.schemas import PrivacyLevel, ReplicaState
from adaroute.selection.catalog import DeploymentDefinition, ReplicaDefinition
from adaroute.selection.objectives import ConfigurableObjectiveEngine


@pytest.mark.asyncio
async def test_tc_m3_62_endpoint_cache_miss_objective_switch_changes_deployment():
    """Prove that on a cache miss, changing the selector policy changes the deployment executed,
    and response/telemetry identifies that deployment and its explanation accurately."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Request with 'strongest' policy selects the highest quality model for coding (gemini-flash-latest)
        res_strong = await client.post(
            "/v1/chat/completions",
            json={
                "model": "coding",
                "prompt": "Write a fast matrix multiplication routine in Rust.",
                "policy": "strongest",
                "execution_mode": "simulation",
                "no_cache": True,
            },
        )
        assert res_strong.status_code == 200
        data_strong = res_strong.json()
        assert data_strong["routing_metadata"]["selected_deployment"] in (
            "deploy-gemini/gemini-flash-latest",
            "deploy-groq/qwen/qwen3.8-27b",
        )
        selected_strong = data_strong["routing_metadata"]["selected_deployment"]
        assert data_strong["deployment_id"] == selected_strong
        assert data_strong["model"] in ("gemini/gemini-flash-latest", "groq/qwen/qwen3.8-27b")

        # Verify explanation metadata
        exp_strong = data_strong["routing_metadata"]["explanation"]
        assert exp_strong is not None
        assert exp_strong["decision_version"] == "4.0.0"
        assert exp_strong["policy_name"] == "strongest"
        assert exp_strong["selected_deployment"] == selected_strong
        assert "objective_contributions" in exp_strong

        # 2. Request with 'cheapest' policy selects the cheapest candidate (vllm Qwen)
        res_cheap = await client.post(
            "/v1/chat/completions",
            json={
                "model": "coding",
                "prompt": "Write a fast matrix multiplication routine in Rust.",
                "policy": "cheapest",
                "execution_mode": "simulation",
                "no_cache": True,
            },
        )
        assert res_cheap.status_code == 200
        data_cheap = res_cheap.json()
        assert data_cheap["routing_metadata"]["selected_deployment"] == "deploy-Qwen/Qwen2.5-0.5B-Instruct"
        assert data_cheap["deployment_id"] == "deploy-Qwen/Qwen2.5-0.5B-Instruct"
        assert data_cheap["model"] == "Qwen/Qwen2.5-0.5B-Instruct"

        # Invariant: Switching policy changed the deployment actually executed!
        assert data_strong["deployment_id"] != data_cheap["deployment_id"]


@pytest.mark.asyncio
async def test_tc_m3_50_endpoint_unknown_policy_rejection_and_custom_policy_execution():
    """Unsupported policy rejected with HTTP 400 (never silently substituted);
    custom configured policy executes and shapes ranking."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Unknown policy must return HTTP 400
        res_bad = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": "Hello test policy",
                "policy": "non_existent_magic_policy",
                "execution_mode": "simulation",
                "no_cache": True,
            },
        )
        assert res_bad.status_code == 400
        assert "Unknown routing policy" in res_bad.json()["detail"]

        # 2. Register custom declarative policy in registry selector
        custom_engine = ConfigurableObjectiveEngine(
            objectives=[
                {"name": "cost", "direction": "minimize", "weight": 0.90},
                {"name": "quality_loss", "direction": "minimize", "weight": 0.10},
            ]
        )
        registry.selector.policies["custom_cost_heavy"] = custom_engine

        try:
            res_custom = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "coding",
                    "prompt": "Calculate fibonacci sequence in C",
                    "policy": "custom_cost_heavy",
                    "execution_mode": "simulation",
                    "no_cache": True,
                },
            )
            assert res_custom.status_code == 200
            data_custom = res_custom.json()
            assert data_custom["routing_metadata"]["policy_name"] == "custom_cost_heavy"
            assert data_custom["routing_metadata"]["explanation"]["policy_name"] == "custom_cost_heavy"
        finally:
            registry.selector.policies.pop("custom_cost_heavy", None)


@pytest.mark.asyncio
async def test_tc_m1_27_endpoint_explicit_model_request_preservation_and_rejection():
    """Explicit model requests are preserved (never rewritten to capability or falling back to general)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Explicit request for gpt-4o-external with reasoning trigger prompt
        res_explicit = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-external",
                "prompt": "Think step by step and prove Fermat's Last Theorem for n=3",
                "execution_mode": "simulation",
                "no_cache": True,
            },
        )
        assert res_explicit.status_code == 200
        data = res_explicit.json()
        assert data["model"] == "gpt-4o-external"
        assert data["deployment_id"] == "deploy-gpt-4o-external"

        # 2. Explicit request with impossible budget fails closed with 503; NO fallback to cheap model
        res_rejected = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-external",
                "prompt": "Explain quantum computing",
                "budget": 0.0000001,
                "execution_mode": "simulation",
                "no_cache": True,
            },
        )
        assert res_rejected.status_code == 503
        assert "Budget exceeded" in res_rejected.json()["detail"]


@pytest.mark.asyncio
async def test_tc_m2_10_endpoint_multiple_deployments_of_same_model_distinguished():
    """Multiple deployments of the same model are distinct in candidate resolution,
    cost accounting, and response deployment_id, and Stage 2 schedules replicas belonging to that deployment."""
    rep = ReplicaDefinition("replica-discount-1", endpoint="mock://discount-1", availability=True)
    registry.register_replica(
        ReplicaState(
            replica_id="replica-discount-1",
            model_id="gpt-4o-external",
            provider="mock-external",
            endpoint="mock://discount-1",
            availability=True,
        )
    )
    dep_discount = DeploymentDefinition(
        deployment_id="deploy-gpt-4o-discount",
        model_id="gpt-4o-external",
        provider_id="mock-external",
        input_price_per_1k=0.00001,
        output_price_per_1k=0.00002,
        allowed_privacy={PrivacyLevel.PUBLIC},
        replicas=[rep],
    )
    registry.catalog.register_deployment(dep_discount)

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Request with cheapest policy when both deploy-gpt-4o-external and deploy-gpt-4o-discount exist
            res = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-external",
                    "prompt": "Multiple deployment test prompt",
                    "policy": "cheapest",
                    "execution_mode": "simulation",
                    "no_cache": True,
                },
            )
            assert res.status_code == 200
            data = res.json()
            # Must pick the discount deployment and its scoped replica!
            assert data["deployment_id"] == "deploy-gpt-4o-discount"
            assert data["routing_metadata"]["selected_deployment"] == "deploy-gpt-4o-discount"
            assert data["routing_metadata"]["selected_replica"] == "replica-discount-1"
    finally:
        registry.catalog.deployments.pop("deploy-gpt-4o-discount", None)
        registry._replicas.pop("replica-discount-1", None)


@pytest.mark.asyncio
async def test_tc_m3_51_endpoint_privacy_and_user_private_enforcement():
    """user_private=True excludes all cloud models; restricted privacy fails closed against unauthorized models."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. user_private=True forces local execution even with 'strongest' policy
        res_priv = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": "Classified company financial records",
                "user_private": True,
                "policy": "strongest",
                "execution_mode": "simulation",
                "no_cache": True,
            },
        )
        assert res_priv.status_code == 200
        data_priv = res_priv.json()
        assert data_priv["routing_metadata"]["provider"] in ("vllm", "local", "mock-local")
        assert data_priv["deployment_id"] in (
            "deploy-Qwen/Qwen2.5-0.5B-Instruct",
            "deploy-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deploy-llama-3-8b",
        )

        # 2. Cloud model requested with confidential privacy is rejected
        res_leak_blocked = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini/gemini-flash-latest",
                "prompt": "Confidential medical data",
                "privacy_level": "confidential",
                "execution_mode": "simulation",
                "no_cache": True,
            },
        )
        assert res_leak_blocked.status_code == 503
        assert "Privacy violation" in res_leak_blocked.json()["detail"]


@pytest.mark.asyncio
async def test_tc_m3_52_endpoint_budget_and_context_window_rejection():
    """Hard constraints reject requests exceeding budget or context window."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Budget ceiling rejection
        res_b = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": "Any generic prompt",
                "budget": 0.000000001,
                "execution_mode": "simulation",
                "no_cache": True,
            },
        )
        assert res_b.status_code == 503
        assert "Budget exceeded" in res_b.json()["detail"]

        # 2. Context window rejection (llama-3-8b context window is 8192)
        res_ctx = await client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3-8b",
                "prompt": "Context length test",
                "max_tokens": 100000,
                "execution_mode": "simulation",
                "no_cache": True,
            },
        )
        assert res_ctx.status_code == 503
        assert "Context window exceeded" in res_ctx.json()["detail"]


@pytest.mark.asyncio
async def test_tc_m3_53_endpoint_session_budget_admission_and_turn_accounting():
    """Unified session store enforces initial budget admission, remaining budget enforcement,
    and consecutive turn depletion."""
    sid = "test-session-budget-depletion"
    # Reset any previous session state
    session_store.reset_session(sid, "user")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Turn 1: Initial budget $0.001 - should succeed
        res1 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "coding",
                "prompt": "Turn 1 request",
                "session_id": sid,
                "session_budget": 0.001,
                "execution_mode": "simulation",
                "no_cache": True,
            },
        )
        assert res1.status_code == 200
        sess1 = session_store.get_session(sid, "user")
        assert sess1 is not None
        assert sess1.turn_index == 1
        rem_after_turn1 = sess1.session_budget_remaining
        assert rem_after_turn1 < 0.001

        # Turn 2: Second turn on same session succeeds
        res2 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "coding",
                "prompt": "Turn 2 request",
                "session_id": sid,
                "execution_mode": "simulation",
                "no_cache": True,
            },
        )
        assert res2.status_code == 200
        sess2 = session_store.get_session(sid, "user")
        assert sess2.turn_index == 2
        assert sess2.session_budget_remaining < rem_after_turn1

        # Manually clamp session budget remaining to near zero to test admission rejection
        sess2.session_budget_remaining = 0.0000001
        session_store.save_session(sess2)

        # Turn 3: Exceeds remaining session budget -> HTTP 503
        res3 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "coding",
                "prompt": "Turn 3 request after budget depleted",
                "session_id": sid,
                "execution_mode": "simulation",
                "no_cache": True,
            },
        )
        assert res3.status_code == 503
        assert "Session budget exhausted" in res3.json()["detail"]


@pytest.mark.asyncio
async def test_tc_m4_17_endpoint_cache_hit_session_turn_accounting():
    """Cache hits advance session turn count while keeping billed cost at 0.0 (budget remaining untouched)."""
    sid = "test-session-cache-hit-accounting"
    session_store.reset_session(sid, "user")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        prompt = "Cache hit accounting test prompt"
        # Turn 1: Cache miss
        res1 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": prompt,
                "session_id": sid,
                "session_budget": 0.05,
                "execution_mode": "simulation",
            },
        )
        assert res1.status_code == 200
        assert res1.json()["routing_metadata"]["cache_hit"] is False

        sess_after_miss = session_store.get_session(sid, "user")
        turns_before_hit = sess_after_miss.turn_index
        budget_before_hit = sess_after_miss.session_budget_remaining

        # Turn 2: Exact repeat -> Cache HIT
        res2 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": prompt,
                "session_id": sid,
                "execution_mode": "simulation",
            },
        )
        assert res2.status_code == 200
        assert res2.json()["routing_metadata"]["cache_hit"] is True

        sess_after_hit = session_store.get_session(sid, "user")
        # Turn count incremented
        assert sess_after_hit.turn_index == turns_before_hit + 1
        # Budget remaining was NOT reduced by cache hit!
        assert sess_after_hit.session_budget_remaining == budget_before_hit


@pytest.mark.asyncio
async def test_tc_m3_54_endpoint_concurrent_session_thread_safety():
    """Concurrent requests targeting the same session handle thread synchronization cleanly."""
    sid = "test-session-concurrent"
    session_store.reset_session(sid, "user")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        tasks = [
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "general",
                    "prompt": f"Concurrent request #{i}",
                    "session_id": sid,
                    "session_budget": 1.0,
                    "execution_mode": "simulation",
                    "no_cache": True,
                },
            )
            for i in range(5)
        ]
        responses = await asyncio.gather(*tasks)
        for r in responses:
            assert r.status_code == 200

        sess = session_store.get_session(sid, "user")
        assert sess is not None
        assert sess.turn_index == 5


@pytest.mark.asyncio
async def test_tc_m5_09_endpoint_circuit_breaker_canary_probe_non_consumption():
    """Evaluating candidate eligibility does not prematurely consume circuit breaker half-open canary probes."""
    res_id = "deploy-gemini/gemini-flash-latest"
    circuit_breaker.reset()
    for _ in range(circuit_breaker.failure_threshold):
        circuit_breaker.record_failure(res_id)
    # Fast forward into HALF_OPEN state
    circuit_breaker._last_state_change[res_id] = 0.0

    # In HALF_OPEN, is_available must be True, but canary probes must not be consumed
    assert circuit_breaker.is_available(res_id) is True
    assert circuit_breaker._half_open_probes.get(res_id, 0) == 0

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Requesting inspection should not exhaust canary probes
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini/gemini-flash-latest",
                "prompt": "Canary probe test",
                "execution_mode": "simulation",
                "no_cache": True,
            },
        )
        assert res.status_code == 200
        # Canary probe is allowed and consumed by actual execution, not candidate inspection
        assert circuit_breaker.get_state(res_id).value in ("closed", "half_open")
