"""Tests for Live Production Benchmarking and Poisson Load Test Harness (Week-5 Milestone M9)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adaroute.experiments.load_test import LiveLoadTestRunner, LoadTestMetrics


@pytest.mark.asyncio
async def test_TC_M8_09_live_load_test_metrics_calculation():
    """Validates empirical metrics computation (p50, p90, p95, p99, RPS, savings)."""
    runner = LiveLoadTestRunner(gateway_url="http://localhost:7000")

    mock_results = [
        {
            "success": True,
            "latency_ms": 20.0,
            "cache_hit": False,
            "cost": 0.0001,
            "prompt_tokens": 100,
            "output_tokens": 50,
            "model": "qwen2.5:1.5b",
        },
        {
            "success": True,
            "latency_ms": 35.0,
            "cache_hit": True,
            "cost": 0.0,
            "prompt_tokens": 100,
            "output_tokens": 50,
            "model": "qwen2.5:1.5b",
        },
        {
            "success": True,
            "latency_ms": 50.0,
            "cache_hit": False,
            "cost": 0.0002,
            "prompt_tokens": 200,
            "output_tokens": 100,
            "model": "gemini-1.5-flash",
        },
        {
            "success": False,
            "latency_ms": 150.0,
            "error": "HTTP 503: Service Unavailable",
        },
    ]

    metrics = runner._compute_metrics(mock_results, total_offered=4, duration_sec=2.0)

    assert isinstance(metrics, LoadTestMetrics)
    assert metrics.total_offered == 4
    assert metrics.completed == 3
    assert metrics.failed == 1
    assert metrics.cache_hits == 1
    assert metrics.cache_misses == 2
    assert metrics.effective_rps == 1.5
    assert metrics.p50_latency_ms == 35.0
    assert metrics.p90_latency_ms == 50.0
    assert metrics.p95_latency_ms == 50.0
    assert metrics.p99_latency_ms == 50.0
    assert metrics.mean_latency_ms == pytest.approx(35.0, rel=1e-2)
    assert metrics.total_prompt_tokens == 400
    assert metrics.total_output_tokens == 200
    assert metrics.actual_incurred_cost == pytest.approx(0.0003, rel=1e-4)

    # Frontier cost = (400 / 1000) * 0.005 + (200 / 1000) * 0.015 = 0.002 + 0.003 = 0.005
    # Attributable savings = 0.005 - 0.0003 = 0.0047
    assert metrics.frontier_benchmark_cost == pytest.approx(0.005, rel=1e-4)
    assert metrics.attributable_savings == pytest.approx(0.0047, rel=1e-4)
    assert metrics.savings_percentage > 90.0

    d = metrics.to_dict()
    assert "p50_latency_ms" in d
    assert "attributable_savings" in d


@pytest.mark.asyncio
async def test_TC_M8_10_live_load_test_runner_mock_execution(tmp_path):
    """Executes full multi-client load test loop with mocked httpx client."""
    runner = LiveLoadTestRunner(
        gateway_url="http://test-gw:7000",
        arrival_rate_rps=100.0,
        concurrency=3,
        seed=123,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "{}"
    mock_resp.json.return_value = {
        "id": "req-mock-1",
        "routing_metadata": {
            "selected_model": "qwen2.5:1.5b",
            "actual_cost": 0.00005,
            "cache_hit": False,
        },
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25,
        },
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        metrics = await runner.run(total_requests=6)

    assert metrics.total_offered == 6
    assert metrics.completed == 6
    assert metrics.failed == 0
    assert metrics.total_prompt_tokens == 6 * 15
    assert metrics.total_output_tokens == 6 * 25
    assert metrics.effective_rps > 0.0


@pytest.mark.asyncio
async def test_TC_M8_11_live_load_test_network_error_handling():
    """Verifies that network exceptions are safely converted to failures without aborting the run."""
    runner = LiveLoadTestRunner(
        gateway_url="http://unreachable-gw:7000",
        arrival_rate_rps=1000.0,
        concurrency=2,
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=RuntimeError("Connection refused")):
        metrics = await runner.run(total_requests=4)

    assert metrics.total_offered == 4
    assert metrics.completed == 0
    assert metrics.failed == 4
    assert metrics.cache_hits == 0
    assert metrics.actual_incurred_cost == 0.0


def test_TC_M8_12_diverse_prompt_catalog_and_scenarios():
    """Validates diverse prompt catalog and scenario request generation."""
    runner = LiveLoadTestRunner(gateway_url="http://localhost:7000", arrival_rate_rps=20.0, seed=42)

    # 1. Catalog diversity checks
    categories = {p["category"] for p in runner.DIVERSE_PROMPT_CATALOG}
    expected_categories = {"coding", "reasoning", "qa", "summarization", "structured_extraction", "fast", "complex"}
    assert expected_categories.issubset(categories)

    lengths = {p["length_tier"] for p in runner.DIVERSE_PROMPT_CATALOG}
    assert {"short", "medium", "long"}.issubset(lengths)

    # 2. Standard scenario
    std_reqs = runner.generate_requests(scenario="standard", total_requests=10)
    assert len(std_reqs) == 10
    assert all("payload" in r and "target_arrival_time" in r for r in std_reqs)

    # 3. Stress scenario forces no_cache and unique prompt nonces
    stress_reqs = runner.generate_requests(scenario="stress", total_requests=8)
    assert len(stress_reqs) == 8
    for r in stress_reqs:
        assert r["payload"]["no_cache"] is True
        assert "[nonce:" in r["payload"]["prompt"]

    # 4. Burst scenario schedules concurrent arrival batches
    burst_reqs = runner.generate_requests(scenario="burst", total_requests=12)
    assert len(burst_reqs) == 12
    # At least some requests share identical arrival timestamps (burst waves)
    times = [r["target_arrival_time"] for r in burst_reqs]
    assert len(set(times)) < len(times)

    # 5. Local-only scenario forces user_private = True
    local_reqs = runner.generate_requests(scenario="local_only", total_requests=6)
    assert len(local_reqs) == 6
    assert all(r["payload"]["user_private"] is True for r in local_reqs)
    assert all(r["payload"]["privacy_level"] == "internal_only" for r in local_reqs)


@pytest.mark.asyncio
async def test_TC_M8_13_live_policy_benchmark_mock():
    """Validates live policy benchmark matrix evaluation across multiple policies."""
    runner = LiveLoadTestRunner(gateway_url="http://test-gw:7000", arrival_rate_rps=50.0, concurrency=2)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "{}"
    mock_resp.json.return_value = {
        "id": "req-mock-pol",
        "routing_metadata": {
            "selected_model": "unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M",
            "actual_cost": 0.00001,
            "cache_hit": False,
        },
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 30,
        },
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        matrix = await runner.run_live_policy_benchmark(
            requests_per_policy=3,
            policies=["adaptive", "cost_first", "latency_first"],
            no_cache=True,
        )

    assert "adaptive" in matrix
    assert "cost_first" in matrix
    assert "latency_first" in matrix
    assert matrix["adaptive"]["completed"] == 3
    assert matrix["adaptive"]["failed"] == 0
    assert "model_distribution" in matrix["adaptive"]
    assert "capability_distribution" in matrix["adaptive"]
