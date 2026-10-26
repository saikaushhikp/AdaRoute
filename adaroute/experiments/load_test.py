"""Live Production Load Testing & Concurrency Benchmark Harness (Week-5 Milestone M9).

Executes end-to-end load tests against live containerized gateway and serving replicas
under continuous multi-client Poisson arrival streams:
- Generates Poisson arrival process: inter-arrival Delta t ~ Exponential(lambda)
- Controls concurrent async client worker pools
- Measures end-to-end latency percentiles (p50, p90, p95, p99)
- Records throughput (effective RPS), cache hit ratio, error rates
- Calculates attributable cost savings: S_p = B_p - A_p vs frontier flagship pricing
- Serializes structured telemetry to results/live_load_test_summary.json
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class LoadTestMetrics:
    """Aggregated empirical benchmark metrics from a live load test."""

    total_offered: int
    completed: int
    failed: int
    cache_hits: int
    cache_misses: int
    duration_sec: float
    effective_rps: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    total_prompt_tokens: int
    total_output_tokens: int
    actual_incurred_cost: float
    frontier_benchmark_cost: float
    attributable_savings: float
    savings_percentage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_offered": self.total_offered,
            "completed": self.completed,
            "failed": self.failed,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "duration_sec": round(self.duration_sec, 2),
            "effective_rps": round(self.effective_rps, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p90_latency_ms": round(self.p90_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "actual_incurred_cost": round(self.actual_incurred_cost, 6),
            "frontier_benchmark_cost": round(self.frontier_benchmark_cost, 6),
            "attributable_savings": round(self.attributable_savings, 6),
            "savings_percentage": round(self.savings_percentage, 2),
        }


class LiveLoadTestRunner:
    """Multi-client load generator driving Poisson arrival traffic against live gateway."""

    SAMPLE_PROMPTS = [
        {"model": "general", "prompt": "What is the capital of France?", "privacy_level": "public"},
        {"model": "general", "prompt": "Explain the difference between TCP and UDP.", "privacy_level": "public"},
        {"model": "coding", "prompt": "Write a python function to compute factorial recursively.", "privacy_level": "public"},
        {"model": "qa", "prompt": "What is Moore's Law and does it still apply?", "privacy_level": "internal_only"},
        {"model": "general", "prompt": "Summarize the primary benefits of microservice architecture.", "privacy_level": "public"},
        {"model": "reasoning", "prompt": "Solve step-by-step: If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?", "privacy_level": "public"},
    ]

    def __init__(
        self,
        gateway_url: str = "http://localhost:8000",
        arrival_rate_rps: float = 10.0,
        concurrency: int = 5,
        timeout_sec: float = 15.0,
        seed: int = 42,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.arrival_rate_rps = arrival_rate_rps
        self.concurrency = concurrency
        self.timeout_sec = timeout_sec
        self.rng = random.Random(seed)

    async def _client_worker(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        results: list[dict[str, Any]],
        client: httpx.AsyncClient,
    ) -> None:
        """Worker task processing requests from the arrival queue."""
        while not queue.empty():
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            req_payload = item["payload"]
            target_time = item["target_arrival_time"]

            # Wait until Poisson arrival time
            now = time.perf_counter()
            if target_time > now:
                await asyncio.sleep(target_time - now)

            start_t = time.perf_counter()
            try:
                res = await client.post(
                    f"{self.gateway_url}/v1/chat/completions",
                    json=req_payload,
                    headers={"Content-Type": "application/json"},
                )
                lat_ms = (time.perf_counter() - start_t) * 1000.0
                if res.status_code == 200:
                    data = res.json()
                    meta = data.get("routing_metadata", {})
                    usage = data.get("usage", {})
                    results.append(
                        {
                            "success": True,
                            "latency_ms": lat_ms,
                            "cache_hit": meta.get("cache_hit", False),
                            "cost": meta.get("actual_cost", 0.0),
                            "prompt_tokens": usage.get("prompt_tokens", 10),
                            "output_tokens": usage.get("completion_tokens", 20),
                            "model": meta.get("selected_model", "unknown"),
                        }
                    )
                else:
                    results.append(
                        {
                            "success": False,
                            "latency_ms": lat_ms,
                            "error": f"HTTP {res.status_code}: {res.text}",
                        }
                    )
            except Exception as e:
                lat_ms = (time.perf_counter() - start_t) * 1000.0
                results.append(
                    {
                        "success": False,
                        "latency_ms": lat_ms,
                        "error": str(e),
                    }
                )
            finally:
                queue.task_done()

    async def run(self, total_requests: int = 30) -> LoadTestMetrics:
        """Executes the load test and returns empirical metrics."""
        # 1. Schedule Poisson arrival stream
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        current_arrival = time.perf_counter()
        for i in range(total_requests):
            sample = self.SAMPLE_PROMPTS[i % len(self.SAMPLE_PROMPTS)].copy()
            # Inter-arrival delta ~ Exponential(lambda)
            delta_t = self.rng.expovariate(self.arrival_rate_rps)
            current_arrival += delta_t
            queue.put_nowait({"payload": sample, "target_arrival_time": current_arrival})

        results: list[dict[str, Any]] = []
        bench_start = time.perf_counter()

        # 2. Launch concurrent client worker tasks
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            workers = [
                asyncio.create_task(self._client_worker(queue, results, client))
                for _ in range(self.concurrency)
            ]
            await asyncio.gather(*workers)

        duration = time.perf_counter() - bench_start
        return self._compute_metrics(results, total_requests, duration)

    def _compute_metrics(
        self, results: list[dict[str, Any]], total_offered: int, duration_sec: float
    ) -> LoadTestMetrics:
        completed_results = [r for r in results if r.get("success")]
        failed_count = total_offered - len(completed_results)

        latencies = sorted([r["latency_ms"] for r in completed_results])
        n = len(latencies)

        def pct(p: float) -> float:
            if not latencies:
                return 0.0
            idx = math.ceil(p * n) - 1
            idx = max(0, min(idx, n - 1))
            return latencies[idx]

        p50 = pct(0.50)
        p90 = pct(0.90)
        p95 = pct(0.95)
        p99 = pct(0.99)
        mean_lat = sum(latencies) / n if n > 0 else 0.0

        cache_hits = sum(1 for r in completed_results if r.get("cache_hit"))
        cache_misses = len(completed_results) - cache_hits

        total_p_tok = sum(r.get("prompt_tokens", 0) for r in completed_results)
        total_out_tok = sum(r.get("output_tokens", 0) for r in completed_results)
        actual_cost = sum(r.get("cost", 0.0) for r in completed_results)

        # Frontier benchmark cost (e.g. GPT-4o pricing: $0.005/1k in, $0.015/1k out)
        frontier_cost = (total_p_tok / 1000.0) * 0.005 + (total_out_tok / 1000.0) * 0.015
        savings = max(0.0, frontier_cost - actual_cost)
        savings_pct = (savings / frontier_cost * 100.0) if frontier_cost > 0.0 else 0.0
        eff_rps = len(completed_results) / duration_sec if duration_sec > 0.0 else 0.0

        return LoadTestMetrics(
            total_offered=total_offered,
            completed=len(completed_results),
            failed=failed_count,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            duration_sec=duration_sec,
            effective_rps=eff_rps,
            p50_latency_ms=p50,
            p90_latency_ms=p90,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            mean_latency_ms=mean_lat,
            total_prompt_tokens=total_p_tok,
            total_output_tokens=total_out_tok,
            actual_incurred_cost=actual_cost,
            frontier_benchmark_cost=frontier_cost,
            attributable_savings=savings,
            savings_percentage=savings_pct,
        )


def _save_summary(out_file: Path, data: dict[str, Any]) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


async def main():
    gateway_url = os.getenv("GATEWAY_URL", "http://localhost:8000")
    runner = LiveLoadTestRunner(
        gateway_url=gateway_url,
        arrival_rate_rps=15.0,
        concurrency=6,
        timeout_sec=20.0,
    )
    print(f"Starting Live Production Load Test against {gateway_url} (30 requests)...")
    metrics = await runner.run(total_requests=30)
    print(json.dumps(metrics.to_dict(), indent=2))

    out_file = Path("results/live_load_test_summary.json")
    await asyncio.to_thread(_save_summary, out_file, metrics.to_dict())
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
