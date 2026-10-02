import pytest
from httpx import ASGITransport, AsyncClient

from adaroute.api import app, tracer


@pytest.mark.asyncio
async def test_tc_m7_01_complete_structured_trace_record():
    """[TC_M7_01] Milestone M7: Every request generates a full correlation trace record

    with latency, attempt count, cost, and routing decisions.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": "Observability trace test",
                "tenant_id": "tenant-xyz",
            },
        )
        assert res.status_code == 200
        req_id = res.json()["id"]

        trace = tracer.get_trace(req_id)
        assert trace is not None
        assert trace.request_id == req_id
        assert trace.tenant_id == "tenant-xyz"
        assert trace.total_latency_ms > 0
        assert trace.selected_model is not None
        assert trace.selected_replica is not None
        assert len(trace.attempts) >= 1
        assert trace.actual_output_tokens > 0


@pytest.mark.asyncio
async def test_tc_m7_02_secret_prompt_redaction_in_traces():
    """[TC_M7_02] Milestone M7: Raw prompt and proprietary messages are excluded

    from the default tracer data structures to prevent secret leakage.
    """
    secret_text = "API_KEY_SUPER_SECRET_VALUE_DO_NOT_LOG"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "general",
                "prompt": f"Process this sensitive token: {secret_text}",
            },
        )
        assert res.status_code == 200
        req_id = res.json()["id"]

        trace = tracer.get_trace(req_id)
        trace_json = trace.model_dump_json()

        # Invariant: Secret text must NOT appear in the trace event log!
        assert secret_text not in trace_json
