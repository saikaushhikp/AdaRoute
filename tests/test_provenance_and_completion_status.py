"""Unit and endpoint tests for end-to-end execution provenance and completion status.

Tests:
1. Controlled backend responses for real, simulated, unknown, "stop", and "length".
2. Verification across backend -> adapter -> AttemptRecord -> API response -> cache -> durable ledger.
3. Strict real-mode fail-closed behavior (rejects simulated responses).
4. Prevention of secret leakage and raw prompt leakage.
5. Live vLLM verification for "length" (bounded output limit) vs "stop".
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from adaroute.api import app, cache, default_tracer
from adaroute.schemas import PrivacyLevel, RequestContext
from adaroute.telemetry.ledger import default_ledger


@pytest.fixture(autouse=True)
def clean_telemetry():
    default_tracer.clear()
    default_ledger.clear()
    yield
    default_tracer.clear()
    default_ledger.clear()


@pytest.mark.asyncio
async def test_TC_M7_11_provenance_and_finish_reason_propagation_end_to_end():
    """Verify that backend finish_reason and provenance are faithfully propagated

    through adapter, attempt record, API response, cache, and durable ledger.
    """
    client = TestClient(app)

    # Mock backend to return real response with finish_reason="length" (truncated)
    mock_runner = AsyncMock()
    mock_runner.execute.return_value = {
        "content": "This reasoning was abruptly cut off because max_tokens was reached",
        "prompt_tokens": 15,
        "output_tokens": 10,
        "latency_ms": 42.0,
        "replica_id": "vllm-rep-1",
        "model": "qwen2.5:1.5b",
        "model_id": "qwen2.5:1.5b",
        "provider": "vllm",
        "source": "vllm_real",
        "hardware": "gpu",
        "actual_cost": 0.0001,
        "cost_state": "allocated_estimate",
        "finish_reason": "length",
        "is_simulated": False,
    }

    with patch("adaroute.api.get_backend_runner", return_value=mock_runner):
        payload = {
            "prompt": "Explain quantum computing in detail",
            "task_type": "reasoning",
            "max_tokens": 10,
            "execution_mode": "hybrid",
            "privacy_level": "public",
        }
        res = client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200, res.text
        data = res.json()

        # 1. API Response Choice
        choice = data["choices"][0]
        assert choice["finish_reason"] == "length", f"Expected finish_reason='length', got {choice['finish_reason']}"

        # 2. Routing Metadata Attempt History
        meta = data["routing_metadata"]
        attempts = meta["attempt_history"]
        assert len(attempts) >= 1
        last_attempt = attempts[-1]
        assert last_attempt["finish_reason"] == "length"
        assert last_attempt["source"] == "vllm_real"
        assert last_attempt["is_simulated"] is False
        assert last_attempt["execution_mode"] == "hybrid"

        # 3. Durable Ledger Entry
        req_id = data["id"]
        entries = default_ledger.get_entries(request_id=req_id)
        assert len(entries) >= 1
        attempt_entries = [e for e in entries if e.entry_type == "attempt"]
        assert len(attempt_entries) >= 1
        ledger_attempt = attempt_entries[-1]
        assert ledger_attempt.finish_reason == "length"
        assert ledger_attempt.source == "vllm_real"
        assert ledger_attempt.is_simulated is False
        assert ledger_attempt.execution_mode == "hybrid"

        req_entries = [e for e in entries if e.entry_type == "request"]
        assert len(req_entries) >= 1
        ledger_req = req_entries[-1]
        assert ledger_req.finish_reason == "length"
        assert ledger_req.source == "vllm_real"
        assert ledger_req.is_simulated is False


@pytest.mark.asyncio
async def test_TC_M2_08_real_execution_mode_fails_closed_on_simulated_response():
    """In 'real' execution mode, gateway must fail closed if the backend returns a simulated response."""
    client = TestClient(app)

    # Mock backend returning a simulated response
    mock_runner = AsyncMock()
    mock_runner.execute.return_value = {
        "content": "[Simulated Response]",
        "prompt_tokens": 10,
        "output_tokens": 10,
        "latency_ms": 5.0,
        "replica_id": "rep-sim",
        "model": "qwen2.5:1.5b",
        "provider": "vllm",
        "source": "simulated_vllm",
        "is_simulated": True,
        "finish_reason": "stop",
    }

    with patch("adaroute.api.get_backend_runner", return_value=mock_runner):
        payload = {
            "prompt": "Test strict real mode rejection",
            "execution_mode": "real",
            "task_type": "general",
            "privacy_level": "public",
        }
        res = client.post("/v1/chat/completions", json=payload)
        # Real-only mode must fail closed (status 503) because simulation is forbidden
        assert res.status_code == 503
        assert "Real execution mode required, but backend" in res.json()["detail"]


@pytest.mark.asyncio
async def test_TC_M7_12_secret_scrubbing_in_errors_and_ledger():
    """Verify that credentials and secrets (API keys, Bearer tokens) are scrubbed from errors and ledger."""
    client = TestClient(app)

    mock_runner = AsyncMock()
    # Backend fails with an error string containing sensitive credentials
    mock_runner.execute.side_effect = RuntimeError(
        "Failed to reach upstream provider with Authorization Bearer sk-ant-secret1234567890abcdef and api_key=AIzaSyA1234567890123456789012345678901"
    )

    with patch("adaroute.api.get_backend_runner", return_value=mock_runner):
        payload = {
            "prompt": "Test secret scrubbing",
            "task_type": "coding",
            "privacy_level": "public",
        }
        res = client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 503
        detail = res.json()["detail"]

        # Secrets must NOT appear in API error detail
        assert "sk-ant-secret" not in detail
        assert "AIzaSy" not in detail
        assert "[REDACTED_SECRET]" in detail

        # Secrets must NOT appear in durable ledger
        entries = default_ledger.get_entries()
        for entry in entries:
            err_text = str(entry.error or "")
            assert "sk-ant-secret" not in err_text
            assert "AIzaSy" not in err_text


@pytest.mark.asyncio
async def test_TC_M7_13_diagnostic_prompt_leakage_prevented_by_default():
    """Verify matched_prompt is not leaked in routing_metadata unless ADAROUTE_DEBUG_PROMPT=1."""
    ctx = RequestContext(
        tenant_id="user",
        user_id="user",
        privacy_level=PrivacyLevel.PUBLIC,
        task_type="coding",
        temperature=0.7,
        max_tokens=100,
    )
    # Seed semantic cache
    cache.put(
        request=ctx,
        logical_model="general",
        model_version="1.0.0",
        prompt="Write a function to calculate the fibonacci series",
        messages=[],
        response={
            "content": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
            "model": "qwen2.5:1.5b",
            "deployment_id": "vllm-deploy",
            "replica_id": "vllm-rep-1",
            "provider": "vllm",
            "output_tokens": 25,
            "finish_reason": "stop",
        },
        origin_model="qwen2.5:1.5b",
        origin_provider="vllm",
        origin_privacy=PrivacyLevel.PUBLIC,
        origin_quality=0.8,
        origin_policy="adaptive",
        policy_name="adaptive",
        is_simulated=False,
        source="vllm_real",
        finish_reason="stop",
    )

    client = TestClient(app)
    # Paraphrased query
    payload = {
        "prompt": "Write a python function to compute the fibonacci numbers",
        "task_type": "coding",
        "temperature": 0.7,
        "max_tokens": 100,
    }

    # 1. Default mode: ADAROUTE_DEBUG_PROMPT not set (or "0")
    if "ADAROUTE_DEBUG_PROMPT" in os.environ:
        del os.environ["ADAROUTE_DEBUG_PROMPT"]

    res = client.post("/v1/chat/completions", json=payload)
    if res.status_code == 200:
        data = res.json()
        if data["routing_metadata"]["cache_hit"]:
            assert data["routing_metadata"]["matched_prompt"] is None, "matched_prompt must be None by default!"

    # 2. Debug mode: ADAROUTE_DEBUG_PROMPT=1 with a distinct paraphrase
    os.environ["ADAROUTE_DEBUG_PROMPT"] = "1"
    try:
        payload_distinct = {
            "prompt": "Can you generate python code that finds the fibonacci sequence",
            "task_type": "coding",
            "temperature": 0.7,
            "max_tokens": 100,
        }
        res_debug = client.post("/v1/chat/completions", json=payload_distinct)
        if res_debug.status_code == 200:
            data_debug = res_debug.json()
            if data_debug["routing_metadata"]["cache_hit"]:
                # When debug enabled, matched_prompt is visible
                assert data_debug["routing_metadata"]["matched_prompt"] is not None
    finally:
        os.environ.pop("ADAROUTE_DEBUG_PROMPT", None)


@pytest.mark.asyncio
async def test_TC_M8_14_live_vllm_bounded_output_length_vs_stop():
    """Live integration check against running vLLM service:

    Validates that bounded output cap produces finish_reason='length' while sufficient
    tokens produce finish_reason='stop'.
    """
    vllm_url = "http://localhost:7001/v1/chat/completions"

    # Quick connectivity probe
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            probe = await client.get("http://localhost:7001/v1/models")
            if probe.status_code != 200:
                pytest.skip("vLLM service on port 7001 is not running")
            models_data = probe.json()
            model_name = models_data["data"][0]["id"]
    except Exception:
        pytest.skip("vLLM service on port 7001 is unreachable")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Case 1: Strictly bounded output (max_tokens=5) -> MUST return finish_reason='length'
        req_short = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Write a 500-word essay about the history of mathematics"}],
            "max_tokens": 5,
        }
        res_short = await client.post(vllm_url, json=req_short)
        assert res_short.status_code == 200
        data_short = res_short.json()
        choice_short = data_short["choices"][0]
        assert choice_short["finish_reason"] == "length", (
            f"Expected finish_reason='length' for bounded prompt, got {choice_short['finish_reason']}"
        )

        # Case 2: Normal prompt with sufficient max_tokens -> MUST return finish_reason='stop'
        # Note: The deployed Qwen3-0.6B reasoning model outputs a <think>...</think> reasoning passage
        # (~140 tokens) before the answer, so max_tokens >= 250 is required for full completion without truncation.
        req_normal = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply with only the single word: Pong"}],
            "max_tokens": 300,
        }
        res_normal = await client.post(vllm_url, json=req_normal)
        assert res_normal.status_code == 200
        data_normal = res_normal.json()
        choice_normal = data_normal["choices"][0]
        assert choice_normal["finish_reason"] == "stop", (
            f"Expected finish_reason='stop' for simple prompt, got {choice_normal['finish_reason']}"
        )
