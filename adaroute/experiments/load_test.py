"""Live Production Load Testing & Concurrency Benchmark Harness (Week-5 Milestone M9).

Executes end-to-end load tests against live containerized gateway and serving replicas
under continuous multi-client arrival streams (Poisson, Burst, and Stress profiles):
- Generates Poisson arrival process: inter-arrival Delta t ~ Exponential(lambda)
- Supports burst spikes, high-concurrency stress tests, and cache-evaluation profiles
- Features rich multi-domain prompt diversity: coding, reasoning, qa, summarization,
  structured extraction, fast checks, and complex system design
- Evaluates data sovereignty and privacy constraints (public cloud vs internal_only local vLLM)
- Measures end-to-end latency percentiles (p50, p90, p95, p99, min, max)
- Tracks per-model and per-capability request distributions and error codes
- Calculates attributable cost savings: S_p = B_p - A_p vs frontier flagship pricing
- Supports live multi-policy benchmarking across adaptive, cost_first, latency_first, strongest, cheapest
- Serializes structured telemetry to results/live_load_test_summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
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
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    model_distribution: dict[str, int] = field(default_factory=dict)
    capability_distribution: dict[str, int] = field(default_factory=dict)
    error_details: dict[str, int] = field(default_factory=dict)
    scenario: str = "standard"

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
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "actual_incurred_cost": round(self.actual_incurred_cost, 6),
            "frontier_benchmark_cost": round(self.frontier_benchmark_cost, 6),
            "attributable_savings": round(self.attributable_savings, 6),
            "savings_percentage": round(self.savings_percentage, 2),
            "model_distribution": self.model_distribution,
            "capability_distribution": self.capability_distribution,
            "error_details": self.error_details,
            "scenario": self.scenario,
        }


class LiveLoadTestRunner:
    """Multi-client load generator driving Poisson and burst arrival traffic against live gateway."""

    # Preserved for backward compatibility with existing tests
    SAMPLE_PROMPTS = [
        {"model": "general", "prompt": "What is the capital of France?", "privacy_level": "public"},
        {"model": "general", "prompt": "Explain the difference between TCP and UDP.", "privacy_level": "public"},
        {
            "model": "coding",
            "prompt": "Write a python function to compute factorial recursively.",
            "privacy_level": "public",
        },
        {"model": "qa", "prompt": "What is Moore's Law and does it still apply?", "privacy_level": "internal_only"},
        {
            "model": "general",
            "prompt": "Summarize the primary benefits of microservice architecture.",
            "privacy_level": "public",
        },
        {
            "model": "reasoning",
            "prompt": "Solve step-by-step: If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?",
            "privacy_level": "public",
        },
    ]

    # Diverse Multi-Domain, Multi-Tier, Multi-Constraint Catalog
    DIVERSE_PROMPT_CATALOG: list[dict[str, Any]] = [
        # --- 1. Coding (Algorithms, Concurrency, SQL, Architecture, Bug Fixing) ---
        {
            "category": "coding",
            "model": "coding",
            "prompt": "Write a python function to compute factorial recursively with memoization and type annotations.",
            "length_tier": "short",
            "privacy_level": "public",
            "user_private": False,
        },
        {
            "category": "coding",
            "model": "coding",
            "prompt": "Write an async Python function using asyncio to download multiple URLs concurrently with a Semaphore limit of 5. Include proper timeout handling and error recovery.",
            "length_tier": "medium",
            "privacy_level": "public",
            "user_private": False,
        },
        {
            "category": "coding",
            "model": "coding",
            "prompt": "Write a SQL query using window functions (e.g. DENSE_RANK or ROW_NUMBER) to find the top 3 highest-earning employees in each department from an employee table with columns (id, name, department_id, salary).",
            "length_tier": "medium",
            "privacy_level": "public",
            "user_private": False,
        },
        {
            "category": "coding",
            "model": "coding",
            "prompt": "Implement a thread-safe in-memory LRU Cache class in Python with O(1) get(key) and put(key, value) operations, using an OrderedDict guarded by a threading.Lock.",
            "length_tier": "medium",
            "privacy_level": "internal_only",
            "user_private": True,
        },
        {
            "category": "coding",
            "model": "coding",
            "prompt": "Review the following code and identify memory leaks or concurrency race conditions:\n\n```python\nimport threading\nglobal_data = {}\ndef process_item(k, v):\n    if k not in global_data:\n        global_data[k] = []\n    global_data[k].append(v)\nthreads = [threading.Thread(target=process_item, args=(i % 5, i)) for i in range(1000)]\nfor t in threads: t.start()\nfor t in threads: t.join()\n```\nExplain the race conditions and provide a robust thread-safe alternative using locks or defaultdict.",
            "length_tier": "long",
            "privacy_level": "internal_only",
            "user_private": True,
        },
        # --- 2. Reasoning & Math (Logic Puzzles, Probability, Traversal, Word Problems) ---
        {
            "category": "reasoning",
            "model": "reasoning",
            "prompt": "Solve step-by-step: If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?",
            "length_tier": "short",
            "privacy_level": "public",
            "user_private": False,
        },
        {
            "category": "reasoning",
            "model": "reasoning",
            "prompt": "A box contains 4 red balls, 5 blue balls, and 6 green balls. If three balls are drawn randomly without replacement, calculate the exact probability of drawing one ball of each color. Show all fractional steps.",
            "length_tier": "medium",
            "privacy_level": "public",
            "user_private": False,
        },
        {
            "category": "reasoning",
            "model": "reasoning",
            "prompt": "Solve this deduction puzzle step by step: Alice, Bob, and Charlie each own a pet (cat, dog, parrot). 1. Alice does not own the dog. 2. Bob does not own the parrot. 3. The cat owner is older than Alice. 4. Bob is younger than the cat owner. Who owns which pet?",
            "length_tier": "medium",
            "privacy_level": "public",
            "user_private": False,
        },
        {
            "category": "reasoning",
            "model": "reasoning",
            "prompt": "A train leaves City A at 8:00 AM traveling at 60 mph towards City B. Another train leaves City B at 9:00 AM traveling at 90 mph towards City A on parallel tracks. If the distance between City A and City B is 360 miles, at what time will the two trains cross each other?",
            "length_tier": "medium",
            "privacy_level": "internal_only",
            "user_private": True,
        },
        # --- 3. Factual QA & System Architecture ---
        {
            "category": "qa",
            "model": "qa",
            "prompt": "Explain the difference between TCP and UDP in terms of connection state, transport overhead, and congestion control.",
            "length_tier": "short",
            "privacy_level": "public",
            "user_private": False,
        },
        {
            "category": "qa",
            "model": "qa",
            "prompt": "What is Moore's Law, what physical and thermal scaling limits have hindered it over the last decade, and how has the hardware industry adapted?",
            "length_tier": "medium",
            "privacy_level": "internal_only",
            "user_private": True,
        },
        {
            "category": "qa",
            "model": "qa",
            "prompt": "Compare the CAP theorem with the PACELC theorem. Explain how distributed databases like CockroachDB and Cassandra navigate consistency versus latency trade-offs during normal operations versus partitions.",
            "length_tier": "medium",
            "privacy_level": "public",
            "user_private": False,
        },
        {
            "category": "qa",
            "model": "qa",
            "prompt": "Explain how Linux epoll multiplexing works under the hood compared to select and poll. Describe the difference between Level-Triggered (LT) and Edge-Triggered (ET) modes and why ET requires non-blocking sockets.",
            "length_tier": "medium",
            "privacy_level": "internal_only",
            "user_private": True,
        },
        # --- 4. Summarization & Incident Analysis ---
        {
            "category": "summarization",
            "model": "summarization",
            "prompt": "Summarize the primary benefits of microservice architecture versus monolithic architecture in 3 concise bullet points.",
            "length_tier": "short",
            "privacy_level": "public",
            "user_private": False,
        },
        {
            "category": "summarization",
            "model": "summarization",
            "prompt": "Summarize the following incident report into Root Cause, Impact, and Remediation items:\nIncident Report - 2026-11-12:\nAt 08:14 UTC, the primary API gateway began shedding requests with HTTP 503 errors. The telemetry service detected CPU exhaustion on the Redis cluster following an unthrottled batch invalidation query dispatched by the cache warmer. Over 45,000 requests failed before the rate limiter was deployed at 08:35 UTC. Redis cluster memory pressure was relieved by clearing ephemeral session keys, and normal operations resumed at 08:50 UTC. Future actions: add token-bucket rate limits on cache invalidations and partition keys by tenant.",
            "length_tier": "long",
            "privacy_level": "public",
            "user_private": False,
        },
        # --- 5. Structured Data Extraction ---
        {
            "category": "structured_extraction",
            "model": "general",
            "prompt": "Extract customer_name, order_id, items, and total_amount as valid strict JSON from this message:\n'Hello, this is David Miller confirming receipt of order #ORD-77492. The shipment contained 2 Mechanical Keyboards ($60 each) and 1 Monitor Stand ($40). Total paid was $160.'",
            "length_tier": "medium",
            "privacy_level": "public",
            "user_private": False,
        },
        {
            "category": "structured_extraction",
            "model": "general",
            "prompt": "Extract server_hostname, timestamp, severity, and error_code as JSON from this log line:\n'2026-11-14T22:15:08.192Z [node-worker-04] CRITICAL db_connection_pool_depleted: Unable to acquire connection within 5000ms'",
            "length_tier": "short",
            "privacy_level": "internal_only",
            "user_private": True,
        },
        # --- 6. Fast & Low-Latency Checks ---
        {
            "category": "fast",
            "model": "fast",
            "prompt": "What is the capital of France? Answer in one word.",
            "length_tier": "short",
            "privacy_level": "public",
            "user_private": False,
        },
        {
            "category": "fast",
            "model": "fast",
            "prompt": "Is HTTP/2 binary or textual? Answer in one word.",
            "length_tier": "short",
            "privacy_level": "public",
            "user_private": False,
        },
        # --- 7. Complex System Design ---
        {
            "category": "complex",
            "model": "complex",
            "prompt": "Design a high-throughput, low-latency rate limiting service capable of handling 500,000 requests per second. Compare Token Bucket, Leaky Bucket, and Sliding Window Counter algorithms in terms of memory footprint, burst handling, and distributed synchronization across Redis instances.",
            "length_tier": "long",
            "privacy_level": "public",
            "user_private": False,
        },
    ]

    def __init__(
        self,
        gateway_url: str = "http://localhost:7000",
        arrival_rate_rps: float = 10.0,
        concurrency: int = 5,
        timeout_sec: float = 30.0,
        seed: int = 42,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.arrival_rate_rps = arrival_rate_rps
        self.concurrency = concurrency
        self.timeout_sec = timeout_sec
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_requests(
        self,
        scenario: str = "standard",
        total_requests: int = 30,
        no_cache: bool | None = None,
        policy: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generates structured arrival stream payloads according to the selected scenario."""
        catalog = self.DIVERSE_PROMPT_CATALOG

        if scenario == "local_only":
            # Filter or force local private workloads
            pool = [p for p in catalog if p.get("user_private")] or catalog
            force_private = True
        elif scenario == "cloud_only":
            pool = [p for p in catalog if not p.get("user_private")] or catalog
            force_private = False
        else:
            pool = catalog
            force_private = None

        scheduled_items: list[dict[str, Any]] = []
        current_arrival = time.perf_counter()

        if scenario == "burst":
            # Clustered burst arrival: groups of requests arrive simultaneously
            burst_size = min(10, max(2, self.concurrency * 2))
            burst_interval_sec = 2.0
            for i in range(total_requests):
                base_item = pool[i % len(pool)].copy()
                payload = {
                    "model": base_item["model"],
                    "prompt": base_item["prompt"],
                    "privacy_level": base_item.get("privacy_level", "public"),
                }
                if base_item.get("user_private"):
                    payload["user_private"] = True
                if no_cache is not None:
                    payload["no_cache"] = no_cache
                else:
                    payload["no_cache"] = True  # Stress/Burst forces real model execution
                if policy:
                    payload["policy"] = policy

                # Batch arrivals
                if i > 0 and i % burst_size == 0:
                    current_arrival += burst_interval_sec

                scheduled_items.append({"payload": payload, "target_arrival_time": current_arrival})

        else:
            # Poisson arrival process: Delta t ~ Exponential(lambda)
            for i in range(total_requests):
                base_item = pool[i % len(pool)].copy()
                payload = {
                    "model": base_item["model"],
                    "prompt": base_item["prompt"],
                    "privacy_level": base_item.get("privacy_level", "public"),
                }
                if force_private is True or base_item.get("user_private"):
                    payload["user_private"] = True
                    payload["privacy_level"] = "internal_only"
                elif force_private is False:
                    payload["user_private"] = False
                    payload["privacy_level"] = "public"

                if scenario == "stress":
                    # Stress test forces no_cache and adds unique request nonce to force live inference
                    payload["no_cache"] = True
                    payload["prompt"] = f"{payload['prompt']} [nonce:{i}:{self.rng.randint(1000, 9999)}]"
                elif scenario == "cache_evaluation":
                    # 50% repeat queries (testing cache latency), 50% unique fresh queries
                    if i % 2 == 0:
                        payload["no_cache"] = False
                    else:
                        payload["no_cache"] = True
                        payload["prompt"] = f"{payload['prompt']} [unique:{i}]"
                elif no_cache is not None:
                    payload["no_cache"] = no_cache

                if policy:
                    payload["policy"] = policy

                delta_t = self.rng.expovariate(self.arrival_rate_rps)
                current_arrival += delta_t
                scheduled_items.append({"payload": payload, "target_arrival_time": current_arrival})

        return scheduled_items

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

            # Wait until Poisson / Burst arrival time
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
                            "provider": meta.get("provider", "unknown"),
                            "capability": req_payload.get("model", "unknown"),
                            "status_code": res.status_code,
                        }
                    )
                else:
                    results.append(
                        {
                            "success": False,
                            "latency_ms": lat_ms,
                            "error": f"HTTP {res.status_code}: {res.text[:200]}",
                            "status_code": res.status_code,
                        }
                    )
            except Exception as e:
                lat_ms = (time.perf_counter() - start_t) * 1000.0
                results.append(
                    {
                        "success": False,
                        "latency_ms": lat_ms,
                        "error": str(e),
                        "status_code": 0,
                    }
                )
            finally:
                queue.task_done()

    async def run(
        self,
        total_requests: int = 30,
        scenario: str = "standard",
        no_cache: bool | None = None,
        policy: str | None = None,
    ) -> LoadTestMetrics:
        """Executes the load test and returns empirical metrics."""
        # 1. Schedule arrival stream
        scheduled_requests = self.generate_requests(
            scenario=scenario,
            total_requests=total_requests,
            no_cache=no_cache,
            policy=policy,
        )

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for item in scheduled_requests:
            queue.put_nowait(item)

        results: list[dict[str, Any]] = []
        bench_start = time.perf_counter()

        # 2. Launch concurrent client worker tasks
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            workers = [
                asyncio.create_task(self._client_worker(queue, results, client)) for _ in range(self.concurrency)
            ]
            await asyncio.gather(*workers)

        duration = time.perf_counter() - bench_start
        metrics = self._compute_metrics(results, total_requests, duration)
        metrics.scenario = scenario
        return metrics

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
        min_lat = latencies[0] if latencies else 0.0
        max_lat = latencies[-1] if latencies else 0.0

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

        # Model and capability distributions
        model_dist: dict[str, int] = {}
        cap_dist: dict[str, int] = {}
        for r in completed_results:
            m = r.get("model", "unknown")
            model_dist[m] = model_dist.get(m, 0) + 1
            c = r.get("capability", "unknown")
            cap_dist[c] = cap_dist.get(c, 0) + 1

        error_dist: dict[str, int] = {}
        for r in results:
            if not r.get("success"):
                err = str(r.get("error", "Unknown error"))[:80]
                error_dist[err] = error_dist.get(err, 0) + 1

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
            min_latency_ms=min_lat,
            max_latency_ms=max_lat,
            model_distribution=model_dist,
            capability_distribution=cap_dist,
            error_details=error_dist,
        )

    async def run_live_policy_benchmark(
        self,
        requests_per_policy: int = 10,
        policies: list[str] | None = None,
        no_cache: bool = True,
    ) -> dict[str, Any]:
        """Runs empirical comparisons across routing policies against the live gateway."""
        if policies is None:
            policies = ["adaptive", "cheapest", "latency_first", "cost_first", "strongest"]

        benchmark_matrix: dict[str, Any] = {}
        for pol in policies:
            print(f"--> Benchmarking Live Policy: '{pol}' ({requests_per_policy} requests, no_cache={no_cache})...")
            m = await self.run(
                total_requests=requests_per_policy,
                scenario="standard",
                no_cache=no_cache,
                policy=pol,
            )
            benchmark_matrix[pol] = m.to_dict()

        return benchmark_matrix


def _save_summary(out_file: Path, data: dict[str, Any]) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


async def main():
    parser = argparse.ArgumentParser(
        description="AdaRoute Production Load & Concurrency Benchmark Harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gateway-url",
        default=os.getenv("GATEWAY_URL", "http://localhost:7000"),
        help="URL of the running AdaRoute gateway",
    )
    parser.add_argument(
        "--scenario",
        default="standard",
        choices=["standard", "stress", "burst", "local_only", "cloud_only", "cache_evaluation", "live_benchmark"],
        help="Workload test scenario to execute",
    )
    parser.add_argument(
        "-n",
        "--requests",
        type=int,
        default=30,
        help="Total requests to generate",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent worker tasks",
    )
    parser.add_argument(
        "-r",
        "--rps",
        type=float,
        default=10.0,
        help="Target Poisson arrival rate in requests per second",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Optional policy override (e.g. adaptive, strongest, cheapest, cost_first, latency_first)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Force bypass cache on all requests to ensure 100%% model execution",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results/live_load_test_summary.json",
        help="Output JSON summary path",
    )

    args = parser.parse_args()

    runner = LiveLoadTestRunner(
        gateway_url=args.gateway_url,
        arrival_rate_rps=args.rps,
        concurrency=args.concurrency,
        timeout_sec=args.timeout,
    )

    out_file = Path(args.output)

    if args.scenario == "live_benchmark":
        print(f"Starting Live Policy Benchmark Matrix against {args.gateway_url}...")
        results = await runner.run_live_policy_benchmark(
            requests_per_policy=max(5, args.requests // 5),
            no_cache=True,
        )
        print(json.dumps(results, indent=2))
        await asyncio.to_thread(_save_summary, out_file, results)
        print(f"Live Policy Benchmark results saved to {out_file}")
    else:
        print(
            f"Starting Live Load Test against {args.gateway_url} "
            f"(scenario={args.scenario}, requests={args.requests}, concurrency={args.concurrency}, rps={args.rps})..."
        )
        metrics = await runner.run(
            total_requests=args.requests,
            scenario=args.scenario,
            no_cache=args.no_cache if args.no_cache else None,
            policy=args.policy,
        )
        print(json.dumps(metrics.to_dict(), indent=2))
        await asyncio.to_thread(_save_summary, out_file, metrics.to_dict())
        print(f"Results saved to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
