"""Benchmark Runner and Evaluation Harness for AdaRoute.

Executes baseline policy combinations over independent workload traces,
recording latency, throughput, cost, quality attainment, and estimator MAE.
Supports both ACTUAL live model inference against the live gateway and
offline discrete mathematical simulation for regression testing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any

import httpx

from adaroute.policy.model_selection import (
    AdaptiveModelPolicy,
    CheapestModelPolicy,
    StaticRulePolicy,
    StrongestModelPolicy,
)
from adaroute.policy.registry import ModelRegistry
from adaroute.policy.replica_selection import (
    LeastLoadedScheduler,
    RoundRobinScheduler,
    SLOAwareScheduler,
)
from adaroute.schemas import RequestContext
from adaroute.workloads.estimator import OutputLengthEstimator
from adaroute.workloads.generator import SyntheticRequest, WorkloadGenerator


class BenchmarkRunner:
    def __init__(
        self,
        seed: int = 42,
        gateway_url: str | None = None,
        live: bool = False,
        timeout_sec: float = 30.0,
    ):
        self.seed = seed
        self.gateway_url = (gateway_url or os.getenv("GATEWAY_URL", "http://localhost:7000")).rstrip("/")
        self.live = live
        self.timeout_sec = timeout_sec
        self.registry = ModelRegistry()
        self.generator = WorkloadGenerator(seed=seed)
        self.estimator = OutputLengthEstimator()

    def run_policy_combination(
        self,
        model_policy_name: str,
        scheduler_name: str,
        requests: list[SyntheticRequest],
        live: bool | None = None,
    ) -> dict[str, Any]:
        is_live = self.live if live is None else live

        if not is_live:
            # Deterministic simulation code (preserves regression test invariants)
            registry = ModelRegistry()
            estimator = OutputLengthEstimator()

            model_policies = {
                "strongest": StrongestModelPolicy(registry),
                "cheapest": CheapestModelPolicy(registry),
                "static": StaticRulePolicy(registry),
                "adaptive": AdaptiveModelPolicy(registry),
            }

            replica_schedulers = {
                "round_robin": RoundRobinScheduler(registry),
                "least_loaded": LeastLoadedScheduler(registry),
                "slo_aware": SLOAwareScheduler(registry),
            }

            m_policy = model_policies[model_policy_name]
            r_scheduler = replica_schedulers[scheduler_name]

            latencies_ms = []
            costs = []
            quality_scores = []
            slo_attained_count = 0
            total_tokens = 0
            rejected_count = 0

            start_time = time.perf_counter()

            for req in requests:
                est_out = estimator.estimate(
                    task_type=req.task_type,
                    prompt_tokens=req.prompt_tokens,
                )

                ctx = RequestContext(
                    task_type=req.task_type,
                    prompt_tokens=req.prompt_tokens,
                    estimated_output_tokens=est_out,
                    quality_requirement=req.quality_requirement,
                    latency_slo=req.latency_slo,
                    privacy_level=req.privacy_level,
                    budget=req.budget,
                    tenant_id=req.tenant_id,
                )

                # Stage 1: Model Selection
                model_spec, _est_cost, _err = m_policy.select_model(ctx)
                if not model_spec:
                    rejected_count += 1
                    continue

                # Stage 2: Replica Selection
                replica = r_scheduler.select_replica(model_spec, ctx)
                if not replica:
                    rejected_count += 1
                    continue

                # Simulate Execution
                r_scheduler.dispatch(replica.replica_id, req.prompt_tokens, est_out)

                exec_lat = replica.recent_latency_ms + (req.actual_output_tokens * 0.1)
                latencies_ms.append(exec_lat)

                actual_cost = (req.prompt_tokens / 1000.0) * model_spec.input_price_per_1k + (
                    req.actual_output_tokens / 1000.0
                ) * model_spec.output_price_per_1k
                costs.append(actual_cost)

                q_val = model_spec.quality_scores.get(req.task_type, 0.75)
                quality_scores.append(q_val)

                if (exec_lat / 1000.0) <= req.latency_slo:
                    slo_attained_count += 1

                total_tokens += req.prompt_tokens + req.actual_output_tokens
                estimator.record_observation(req.task_type, est_out, req.actual_output_tokens)
                r_scheduler.complete(replica.replica_id, req.prompt_tokens, est_out, exec_lat, success=True)

            total_duration = time.perf_counter() - start_time
            processed = len(requests) - rejected_count

            latencies_ms.sort()
            p95 = latencies_ms[int(0.95 * len(latencies_ms))] if latencies_ms else 0.0
            p99 = latencies_ms[int(0.99 * len(latencies_ms))] if latencies_ms else 0.0
            mean_lat = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
            est_stats = estimator.get_stats()

            return {
                "model_policy": model_policy_name,
                "replica_scheduler": scheduler_name,
                "requests_total": len(requests),
                "requests_processed": processed,
                "requests_rejected": rejected_count,
                "avg_latency_ms": round(mean_lat, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "throughput_req_per_sec": round(processed / max(0.01, total_duration), 2),
                "total_cost_usd": round(sum(costs), 5),
                "avg_cost_per_req_usd": round(sum(costs) / max(1, processed), 6),
                "avg_quality_score": round(sum(quality_scores) / max(1, len(quality_scores)), 4),
                "slo_attainment_pct": round((slo_attained_count / max(1, processed)) * 100.0, 2),
                "estimator_mae": est_stats.get("mae", 0.0),
                "execution_mode": "simulated",
            }

        # ACTUAL Live Model Inference against Live Gateway
        latencies_ms = []
        costs = []
        quality_scores = []
        slo_attained_count = 0
        total_tokens = 0
        rejected_count = 0
        processed_count = 0

        start_time = time.perf_counter()

        with httpx.Client(timeout=self.timeout_sec) as client:
            for req in requests:
                est_out = self.estimator.estimate(
                    task_type=req.task_type,
                    prompt_tokens=req.prompt_tokens,
                )

                payload = {
                    "model": req.task_type,
                    "prompt": req.prompt,
                    "policy": model_policy_name,
                    "scheduler": scheduler_name,
                    "privacy_level": req.privacy_level.value
                    if hasattr(req.privacy_level, "value")
                    else str(req.privacy_level),
                    "budget": req.budget,
                    "latency_slo": req.latency_slo,
                    "no_cache": True,
                }

                req_start = time.perf_counter()
                try:
                    res = client.post(
                        f"{self.gateway_url}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    exec_lat = (time.perf_counter() - req_start) * 1000.0

                    if res.status_code == 200:
                        data = res.json()
                        meta = data.get("routing_metadata", {})
                        usage = data.get("usage", {})

                        cost = meta.get("actual_cost", 0.0)
                        lat = meta.get("latency_ms", exec_lat)
                        latencies_ms.append(lat)
                        costs.append(cost)

                        p_tok = usage.get("prompt_tokens", req.prompt_tokens)
                        out_tok = usage.get("completion_tokens", 20)
                        total_tokens += p_tok + out_tok

                        # Empirical quality score
                        q_val = 0.85
                        if "explanation" in meta and "objective_contributions" in meta["explanation"]:
                            q_val = round(
                                1.0 - meta["explanation"]["objective_contributions"].get("quality_loss", 0.15),
                                4,
                            )
                        quality_scores.append(q_val)

                        if meta.get("slo_attained", (lat / 1000.0) <= req.latency_slo):
                            slo_attained_count += 1

                        self.estimator.record_observation(req.task_type, est_out, out_tok)
                        processed_count += 1
                    else:
                        rejected_count += 1
                except Exception:
                    rejected_count += 1

        total_duration = time.perf_counter() - start_time
        latencies_ms.sort()
        n = len(latencies_ms)

        def pct(p: float) -> float:
            if not latencies_ms:
                return 0.0
            idx = max(0, min(math.ceil(p * n) - 1, n - 1))
            return latencies_ms[idx]

        p95 = pct(0.95)
        p99 = pct(0.99)
        mean_lat = sum(latencies_ms) / n if n else 0.0
        est_stats = self.estimator.get_stats()

        return {
            "model_policy": model_policy_name,
            "replica_scheduler": scheduler_name,
            "requests_total": len(requests),
            "requests_processed": processed_count,
            "requests_rejected": rejected_count,
            "avg_latency_ms": round(mean_lat, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "throughput_req_per_sec": round(processed_count / max(0.01, total_duration), 2),
            "total_cost_usd": round(sum(costs), 5),
            "avg_cost_per_req_usd": round(sum(costs) / max(1, processed_count), 6),
            "avg_quality_score": round(sum(quality_scores) / max(1, len(quality_scores)), 4),
            "slo_attainment_pct": round((slo_attained_count / max(1, processed_count)) * 100.0, 2),
            "estimator_mae": est_stats.get("mae", 0.0),
            "execution_mode": "live",
        }

    def run_all(
        self,
        num_requests: int = 50,
        output_dir: str = "results",
        live: bool | None = None,
    ) -> list[dict[str, Any]]:
        is_live = self.live if live is None else live
        os.makedirs(output_dir, exist_ok=True)
        requests = self.generator.generate_trace(count=num_requests)

        # Save trace
        trace_path = os.path.join(output_dir, "workload_trace.jsonl")
        self.generator.save_trace(trace_path, requests)

        combinations = [
            ("strongest", "round_robin"),
            ("cheapest", "round_robin"),
            ("static", "round_robin"),
            ("adaptive", "round_robin"),
            ("adaptive", "least_loaded"),
            ("adaptive", "slo_aware"),
        ]

        results = []
        for m_pol, r_sched in combinations:
            if is_live:
                print(f"--> [LIVE] Benchmarking combination: {m_pol} + {r_sched} ({len(requests)} requests)...")
            res = self.run_policy_combination(m_pol, r_sched, requests, live=is_live)
            results.append(res)

        summary_path = os.path.join(output_dir, "benchmark_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AdaRoute Benchmark Matrix Harness (Actual Live Model Inference & Simulation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gateway-url",
        default=os.getenv("GATEWAY_URL", "http://localhost:7000"),
        help="URL of running AdaRoute gateway",
    )
    parser.add_argument(
        "--live",
        dest="live",
        action="store_true",
        default=True,
        help="Execute against actual models via live gateway (default: True)",
    )
    parser.add_argument(
        "--simulated",
        dest="live",
        action="store_false",
        help="Run offline mathematical simulation instead of live model inference",
    )
    parser.add_argument(
        "-n",
        "--requests",
        type=int,
        default=5,
        help="Number of requests per policy combination",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results",
        help="Output directory for benchmark summary",
    )

    args = parser.parse_args()

    mode_label = "ACTUAL Live Model Inference" if args.live else "Offline Simulation"
    print(f"Running AdaRoute benchmark matrix ({mode_label}) against {args.gateway_url}...")
    runner = BenchmarkRunner(seed=42, gateway_url=args.gateway_url, live=args.live)
    out = runner.run_all(num_requests=args.requests, output_dir=args.output, live=args.live)
    for r in out:
        print(
            f"[{r['model_policy']} + {r['replica_scheduler']}]: Cost=${r['total_cost_usd']} | "
            f"Quality={r['avg_quality_score']} | SLO={r['slo_attainment_pct']}% | "
            f"Latency(p95)={r['p95_latency_ms']}ms | Mode={r.get('execution_mode', 'unknown')}"
        )
