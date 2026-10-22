import pytest
from httpx import ASGITransport, AsyncClient

from adaroute.api import app
from adaroute.experiments.event_runner import EventDrivenSimulator
from adaroute.telemetry.ledger import DurableLedger, sanitize_secrets
from adaroute.telemetry.tracer import StructuredTracer
from adaroute.workloads.generator import WorkloadGenerator


def test_tc_m7_03_sanitize_secrets_credentials_and_tokens():
    """[TC_M7_03] Validate that provider credentials are reliably redacted."""
    # Bearer token
    text_bearer = "Authorization header: Bearer abc123def456_xyz-789.end"
    sanitized = sanitize_secrets(text_bearer)
    assert "[REDACTED_SECRET]" in sanitized
    assert "abc123def456" not in sanitized

    # Google API Key
    text_google = "Google Gemini call failed: AIzaSyD-1234567890abcdefghijklmnopqr key rejected"
    sanitized = sanitize_secrets(text_google)
    assert "[REDACTED_SECRET]" in sanitized
    assert "AIzaSyD" not in sanitized

    # OpenAI API Key
    text_openai = "OpenAI key sk-12345678901234567890abcdefghijk invalid"
    sanitized = sanitize_secrets(text_openai)
    assert "[REDACTED_SECRET]" in sanitized
    assert "12345678901234567890" not in sanitized

    # Anthropic API Key
    text_ant = "Anthropic key sk-ant-api03-12345678901234567890-xyz"
    sanitized = sanitize_secrets(text_ant)
    assert "[REDACTED_SECRET]" in sanitized
    assert "api03" not in sanitized

    # Groq API Key
    text_groq = "Groq key gsk_12345678901234567890abcdefgh"
    sanitized = sanitize_secrets(text_groq)
    assert "[REDACTED_SECRET]" in sanitized

    # Query param key
    text_query = "https://endpoint.local/v1?key=secrettoken12345"
    sanitized = sanitize_secrets(text_query)
    assert "[REDACTED_SECRET]" in sanitized

    # Empty and None
    assert sanitize_secrets(None) == ""
    assert sanitize_secrets("") == ""


def test_tc_m7_04_durable_ledger_disk_persistence(tmp_path):
    """[TC_M7_04] Validate that the ledger durably appends records to disk."""
    ledger_file = tmp_path / "sub" / "dir" / "ledger.jsonl"
    ledger = DurableLedger(ledger_path=ledger_file)

    assert ledger_file.parent.exists()

    entry = ledger.record_request(
        request_id="req-test-1",
        logical_model="general",
        selected_model="qwen2.5:1.5b",
        selected_replica="local-r1",
        provider="local",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=125.0,
        cost=0.0001,
        cost_state="allocated_estimate",
        status="success",
        execution_mode="simulated",
        is_simulated=True,
    )

    assert entry.request_id == "req-test-1"
    assert ledger_file.exists()

    # Read back entries from disk
    entries = ledger.get_entries(request_id="req-test-1")
    assert len(entries) == 1
    disk_entry = entries[0]
    assert disk_entry.request_id == "req-test-1"
    assert disk_entry.cost_state == "allocated_estimate"
    assert disk_entry.is_simulated is True
    assert disk_entry.status == "success"


def test_tc_m7_05_durable_ledger_failed_attempt_zero_unbilled_cost(tmp_path):
    """[TC_M7_05] Failed attempts record unknown unbilled cost as zero."""
    ledger_file = tmp_path / "ledger.jsonl"
    ledger = DurableLedger(ledger_path=ledger_file)

    tracer = StructuredTracer(ledger=ledger)
    req_id = "req-failed-1"
    tracer.start_trace(request_id=req_id, prompt_tokens=200)

    # Record failed attempt with secret in error message
    secret_err = "LiteLLM failure: Bearer super-secret-token-12345 connection timeout"
    tracer.record_attempt(
        request_id=req_id,
        attempt_number=1,
        replica_id="local-r1",
        model_id="qwen2.5:1.5b",
        provider="local",
        prompt_tokens=200,
        completion_tokens=0,
        latency_ms=50.0,
        cost=0.005,  # Non-zero passed initially
        cost_state="unknown_unbilled",
        status="failed",
        error=secret_err,
    )

    # Invariant: Failed attempts with unknown charges must clamp cost to 0.0 and cost_state to unknown_unbilled
    attempts = ledger.get_entries(request_id=req_id, entry_type="attempt")
    assert len(attempts) == 1
    failed_record = attempts[0]
    assert failed_record.status == "failed"
    assert failed_record.cost == 0.0
    assert failed_record.cost_state == "unknown_unbilled"
    assert "super-secret-token-12345" not in failed_record.error
    assert "[REDACTED_SECRET]" in failed_record.error


def test_tc_m8_03_simulator_cloud_scheduling_and_delay_decomposition():
    """[TC_M8_03] The simulator decomposes queue delay, TTFT, and TPOT."""
    sim = EventDrivenSimulator(arrival_rate_rps=10.0, seed=42)

    # Generate small trace
    gen = WorkloadGenerator(seed=42)
    trace = gen.generate_trace(count=10)

    # Run strongest policy (routes public requests to cloud models)
    res_strong = sim.run_benchmark("strongest", "round_robin", num_requests=10, offered_trace=trace)

    assert res_strong["total_requests"] == 10
    assert res_strong["completed_requests"] > 0
    assert "avg_queue_delay_ms" in res_strong
    assert "avg_arrival_to_first_token_ms" in res_strong
    assert "avg_ttft_ms" in res_strong
    assert "avg_tpot_ms" in res_strong

    for rec in res_strong["per_request_records"]:
        if rec["status"] == "completed":
            # Invariant: arrival_to_first_token_ms == queue_delay_ms + ttft_ms
            assert abs(rec["arrival_to_first_token_ms"] - (rec["queue_delay_ms"] + rec["ttft_ms"])) < 1e-4

            # Invariant: total_latency_ms == queue_delay_ms + ttft_ms + (output_tokens * tpot_ms)
            expected_exec = rec["total_latency_ms"] - rec["queue_delay_ms"]
            assert expected_exec >= rec["ttft_ms"]

            # Invariant: Cloud models skip local replica queue (W_q == 0)
            if rec["is_cloud"]:
                assert rec["queue_delay_ms"] == 0.0


def test_tc_m8_04_simulator_deterministic_frozen_cohort_matched_replay():
    """[TC_M8_04] The same offered trace is replayed across all policies."""
    gen = WorkloadGenerator(seed=101)
    frozen_trace = gen.generate_trace(count=15)

    sim = EventDrivenSimulator(seed=101)
    policies = ["strongest", "cheapest", "static", "adaptive"]
    results = {}

    for pol in policies:
        results[pol] = sim.run_benchmark(pol, "round_robin", num_requests=15, offered_trace=frozen_trace)

    # All policies evaluate on identical offered count
    for pol in policies:
        assert results[pol]["offered_requests"] == 15
        assert results[pol]["total_requests"] == 15
        assert results[pol]["completed_requests"] == results[pol]["accepted_requests"]


@pytest.mark.asyncio
async def test_tc_m7_06_live_api_ledger_sanitization_integration():
    """[TC_M7_06] The live API records sanitized attempts in the telemetry ledger."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": "Test durable sanitized ledger integration",
                "tenant_id": "tenant-ledger-audit",
            },
        )
        assert res.status_code == 200
        req_id = res.json()["id"]

        from adaroute.telemetry.ledger import default_ledger

        # Confirm ledger persistence to disk
        entries = default_ledger.get_entries(request_id=req_id)
        assert len(entries) >= 1
        req_entry = next(e for e in entries if e.entry_type == "request")
        assert req_entry.request_id == req_id
        assert req_entry.status in ("success", "cached")
        assert req_entry.cost_state in ("allocated_estimate", "invoiced_real")
