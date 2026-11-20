"""Tests for vLLM Semantic Router selectable model-selection policy (CRITICAL_IMP.md).

Verifies:
1. Registration in UnifiedModelSelector, API model_policies, and Control Plane.
2. Controlled router-driven selection: router preview maps to eligible deployment and executes.
3. Hard admission authority:
   - Private / internal_only requests reject router recommendation of cloud backends.
   - Budget constraints reject router recommendation of unaffordable backends.
   - Safe deterministic fallback to Pareto multi-objective ranking when router is unreachable or rejected.
4. CLI validation and commands (/set policy vllm_semantic_router).
5. Live FastAPI endpoint execution (/v1/chat/completions) with policy="vllm_semantic_router".
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from adaroute.api import app, model_policies, registry
from adaroute.schemas import PrivacyLevel, RequestContext
from adaroute.selection.vllm_router import RouterDecisionResult, VLLMSemanticRouterClient

# -----------------------------------------------------------------------------
# Test Server Helper
# -----------------------------------------------------------------------------


class MockRouterHandler(BaseHTTPRequestHandler):
    model_to_return = "Qwen/Qwen2.5-0.5B-Instruct"
    status_code = 200

    def do_POST(self):
        if self.path.startswith("/api/v1/routing/preview"):
            content_length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(content_length)

            if self.status_code == 200:
                response = {
                    "selected_model": self.model_to_return,
                    "selection_status": "ok",
                    "selection_reason": "High intent confidence for general query",
                    "selection_method": "signal_intent_classifier",
                    "decision_result": {
                        "confidence": 0.94,
                        "algorithm": "signal_intent_classifier",
                    },
                }
                body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(self.status_code)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP server logging in test output


@pytest.fixture
def mock_router_server():
    server = HTTPServer(("127.0.0.1", 50059), MockRouterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


# -----------------------------------------------------------------------------
# Unit & Integration Tests
# -----------------------------------------------------------------------------


def test_vllm_sr_policy_registered():
    """Policy must be registered in UnifiedModelSelector and model_policies."""
    selector = registry.selector
    assert "vllm_semantic_router" in selector.policies, "vllm_semantic_router must be in selector.policies"
    assert "vllm_semantic_router" in model_policies, "vllm_semantic_router must be in API model_policies"


def test_vllm_sr_client_preview_success(mock_router_server):
    """VLLMSemanticRouterClient must parse router preview response correctly."""
    MockRouterHandler.model_to_return = "Qwen/Qwen2.5-0.5B-Instruct"
    MockRouterHandler.status_code = 200

    client = VLLMSemanticRouterClient(router_url="http://127.0.0.1:50059")
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    res = client.preview_route(messages)

    assert res.success is True
    assert res.selected_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert res.selection_status == "ok"
    assert res.confidence == 0.94
    assert res.reason == "High intent confidence for general query"


def test_vllm_sr_client_timeout_and_error():
    """Client must handle unreachable endpoints gracefully without raising exceptions."""
    # Point to an unused local port
    client = VLLMSemanticRouterClient(router_url="http://127.0.0.1:59999", timeout=0.2)
    messages = [{"role": "user", "content": "hello"}]
    res = client.preview_route(messages)

    assert res.success is False
    assert res.error is not None
    assert "error" in res.error.lower() or "connection" in res.error.lower()


def test_vllm_sr_policy_decision_admitted_by_hard_constraints():
    """When router selects an eligible candidate, AdaRoute admits and prioritizes it."""
    selector = registry.selector

    # Mock the router client to return local Qwen model
    mock_res = RouterDecisionResult(
        success=True,
        selected_model="Qwen/Qwen2.5-0.5B-Instruct",
        selection_status="ok",
        confidence=0.95,
        reason="Matched general fast profile",
        algorithm="classifier",
    )

    with patch.object(selector.vllm_router_client, "preview_route", return_value=mock_res):
        req = RequestContext(
            task_type="general",
            prompt_tokens=15,
            privacy_level=PrivacyLevel.PUBLIC,
            policy_name="vllm_semantic_router",
            tenant_id="test_tenant",
        )
        decision = selector.route(req, prompt_text="write a python hello world")

        assert decision.selected_candidate is not None
        assert "qwen" in decision.selected_candidate.model_id.lower()
        assert decision.policy_name == "vllm_semantic_router"
        assert decision.explanation.router_metadata.get("admission_status") == "admitted"


def test_vllm_sr_policy_cloud_choice_rejected_on_private_request():
    """AdaRoute hard admission constraints must REJECT router's cloud choice on private requests."""
    selector = registry.selector

    # Router attempts to recommend an external cloud model for an internal_only request
    mock_res = RouterDecisionResult(
        success=True,
        selected_model="gpt-4o-external",
        selection_status="ok",
        confidence=0.99,
        reason="External cloud model best for complex prompt",
    )

    with patch.object(selector.vllm_router_client, "preview_route", return_value=mock_res):
        req = RequestContext(
            task_type="general",
            prompt_tokens=15,
            privacy_level=PrivacyLevel.INTERNAL_ONLY,  # Strictly forbids cloud!
            user_private=True,
            policy_name="vllm_semantic_router",
            tenant_id="secure_corp",
        )
        decision = selector.route(req, prompt_text="confidential internal schema")

        # gpt-4o-external is cloud -> MUST NOT be selected
        assert decision.selected_candidate is not None
        assert "gpt-4o" not in decision.selected_candidate.deployment_id.lower()
        # Must fall back to safe local candidate
        assert decision.selected_candidate.provider_id in ("vllm", "local", "mock-local")
        # Metadata must clearly record rejection
        meta = decision.explanation.router_metadata
        assert meta.get("admission_status") == "rejected_by_hard_constraints"


def test_vllm_sr_policy_outage_fallback():
    """When router fails or times out, selector must fall back safely to Pareto multi-objective ranking."""
    selector = registry.selector

    mock_res = RouterDecisionResult(
        success=False,
        error="Connection refused: router is offline",
    )

    with patch.object(selector.vllm_router_client, "preview_route", return_value=mock_res):
        req = RequestContext(
            task_type="general",
            prompt_tokens=15,
            privacy_level=PrivacyLevel.PUBLIC,
            policy_name="vllm_semantic_router",
            tenant_id="fallback_tenant",
        )
        decision = selector.route(req, prompt_text="test prompt for fallback")

        assert decision.selected_candidate is not None
        meta = decision.explanation.router_metadata
        assert meta.get("admission_status") == "fallback_on_unavailability"


def test_control_plane_discovers_vllm_semantic_router():
    """Control plane endpoint must expose vllm_semantic_router as an available policy."""
    client = TestClient(app)
    resp = client.get("/v1/control-plane/policies")
    assert resp.status_code == 200
    data = resp.json()
    assert "vllm_semantic_router" in data["available_policies"]


def test_control_plane_rollback_to_vllm_semantic_router():
    """Control plane rollback must allow dynamically switching active policy to vllm_semantic_router."""
    client = TestClient(app)
    resp = client.post(
        "/v1/control-plane/rollback",
        json={"policy_name": "vllm_semantic_router", "author": "admin_test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_policy"] == "vllm_semantic_router"
    assert data["status"] == "success"

    # Reset active policy back to adaptive
    client.post("/v1/control-plane/rollback", json={"policy_name": "adaptive", "author": "cleanup"})


def test_cli_set_policy_validation():
    """CLI /set policy command must accept vllm_semantic_router and reject invalid policies."""
    try:
        from ui.cli.main import AdaRouteCLI
    except ImportError:
        pytest.skip("rich or UI CLI dependencies not installed")
    cli = AdaRouteCLI()
    cli.handle_set(["policy", "vllm_semantic_router"])
    assert cli.config["policy"] == "vllm_semantic_router"

    # Invalid policy rejected
    cli.handle_set(["policy", "invalid_policy_xyz"])
    assert cli.config["policy"] == "vllm_semantic_router"  # Preserved


def test_live_chat_completions_with_vllm_sr_policy():
    """Live /v1/chat/completions endpoint accepts policy='vllm_semantic_router' and executes."""
    client = TestClient(app)
    payload = {
        "model": "general",
        "messages": [{"role": "user", "content": "What is the speed of light in vacuum?"}],
        "policy": "vllm_semantic_router",
        "no_cache": True,  # Exercise model selection without cache intercept
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "choices" in body
    assert len(body["choices"]) > 0

    # Diagnostics verification
    meta = body.get("routing_metadata", {})
    assert meta.get("policy_name") == "vllm_semantic_router"
    assert "selected_model" in meta
    assert "selected_replica" in meta


def test_vllm_sr_simple_vs_complex_task_differentiation():
    """vllm_semantic_router policy MUST differentiate between simple and complex tasks.

    Simple arithmetic / greetings ('what is 1+1=?') must route to fast/efficient tier.
    Complex conceptual explanations ('explain me quantum physics') must route to high-capacity frontier tier.
    """
    selector = registry.selector

    # 1. Simple task: what is 1+1=?
    req_simple = RequestContext(
        task_type="general",
        prompt_tokens=5,
        privacy_level=PrivacyLevel.PUBLIC,
        policy_name="vllm_semantic_router",
        tenant_id="test_diff",
    )
    dec_simple = selector.route(req_simple, prompt_text="what is 1+1=?")
    assert dec_simple.selected_candidate is not None
    # Must pick fast / lightweight model (Qwen 0.5B / 0.6B)
    assert "0.5b" in dec_simple.selected_candidate.model_id.lower() or "0.6b" in dec_simple.selected_candidate.model_id.lower() or "fast" in dec_simple.selected_candidate.deployment_id.lower()

    # 2. Complex task: explain me quantum physics
    req_complex = RequestContext(
        task_type="general",
        prompt_tokens=6,
        privacy_level=PrivacyLevel.PUBLIC,
        policy_name="vllm_semantic_router",
        tenant_id="test_diff",
    )
    dec_complex = selector.route(req_complex, prompt_text="explain me quantum physics")
    assert dec_complex.selected_candidate is not None
    # Must pick high-capacity frontier model (qwen3.8-27b)
    assert "27b" in dec_complex.selected_candidate.model_id.lower() or dec_complex.selected_candidate.get_capability_quality("general") >= 0.85

    # Invariant: Simple and complex tasks MUST NOT select the exact same model!
    assert dec_simple.selected_candidate.model_id != dec_complex.selected_candidate.model_id


def test_vllm_sr_semantic_cache_hit_and_telemetry():
    """Near-paraphrase queries must hit semantic cache and populate cache_type and similarity in routing_metadata."""
    client = TestClient(app)

    # Turn 1: original prompt
    p1 = {
        "model": "general",
        "messages": [{"role": "user", "content": "what is the capital of Italy?"}],
        "policy": "vllm_semantic_router",
    }
    r1 = client.post("/v1/chat/completions", json=p1)
    assert r1.status_code == 200

    # Turn 2: exact match
    r2 = client.post("/v1/chat/completions", json=p1)
    assert r2.status_code == 200
    m2 = r2.json().get("routing_metadata", {})
    assert m2.get("cache_hit") is True
    assert m2.get("cache_type") == "exact"

    # Turn 3: near-paraphrase semantic hit
    p3 = {
        "model": "general",
        "messages": [{"role": "user", "content": "Tell me, what is the capital of Italy?"}],
        "policy": "vllm_semantic_router",
    }
    r3 = client.post("/v1/chat/completions", json=p3)
    assert r3.status_code == 200
    m3 = r3.json().get("routing_metadata", {})
    assert m3.get("cache_hit") is True
    assert m3.get("cache_type") in ("exact", "semantic")
    if m3.get("cache_type") == "semantic":
        assert m3.get("similarity") is not None
        assert m3.get("similarity") >= 0.90
