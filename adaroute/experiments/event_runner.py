"""Event-Driven Benchmark & Concurrency Simulation Harness (Week-4 Hardened Version).

Replaces uncalibrated simulation loops with an asynchronous discrete-event simulator:
- Models Poisson / burst arrival processes with overlapping request lifetimes.
- Unified OutputLengthEstimator pre-inference estimation (no circular generation leak).
- Cloud vs Local execution path parity matching api.py (Cloud models skip local replica queue).
- Replays identical frozen deterministic request cohorts across all evaluated policies.
- Explicitly decomposes queue waiting delay (W_q), TTFT, and TPOT:
  * queue_delay_ms
  * arrival_to_first_token_ms = queue_delay_ms + ttft_ms
  * total_latency_ms = queue_delay_ms + ttft_ms + (output_tokens * tpot_ms)
- Reports matched-cohort service metrics (accepted, rejected, completed, quality, cost).
- Resolves official pricing unit reporting ($0.000300/1k for Gemini Flash = $0.30/1M).
"""

from __future__ import annotations

import heapq
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass(order=True)
class Event:
    time_ms: float
    event_type: str = field(compare=False)  # "arrival" | "completion"
    request_id: str = field(compare=False)
    data: dict[str, Any] = field(compare=False, default_factory=dict)


class EventDrivenSimulator:
    def __init__(
        self,
        arrival_rate_rps: float = 15.0,  # Arrival intensity
        seed: int = 42,
        estimator: OutputLengthEstimator | None = None,
        registry: ModelRegistry | None = None,
    ):
        self.arrival_rate_rps = arrival_rate_rps
        self.seed = seed
        self.rng = random.Random(seed)
        self.registry = registry or ModelRegistry()
        self.estimator = estimator or OutputLengthEstimator.from_config(
            self.registry.get_estimation_config() if hasattr(self.registry, "get_estimation_config") else None
        )

    def run_benchmark(
        self,
        policy_name: str,
        scheduler_name: str,
        num_requests: int = 40,
        offered_trace: list[SyntheticRequest] | None = None,
    ) -> dict[str, Any]:
        registry = self.registry

        policies = {
            "adaptive": AdaptiveModelPolicy(registry),
            "strongest": StrongestModelPolicy(registry),
            "cheapest": CheapestModelPolicy(registry),
            "static": StaticRulePolicy(registry),
        }
        schedulers = {
            "slo_aware": SLOAwareScheduler(registry),
            "least_loaded": LeastLoadedScheduler(registry),
            "round_robin": RoundRobinScheduler(registry),
        }

        policy = policies[policy_name]
        scheduler = schedulers[scheduler_name]

        # Use frozen deterministic offered trace if provided, else generate
        if offered_trace is not None:
            requests = list(offered_trace[:num_requests])
        else:
            generator = WorkloadGenerator(seed=self.seed)
            requests = generator.generate_trace(count=num_requests)

        event_queue: list[Event] = []
        for idx, req in enumerate(requests):
            heapq.heappush(
                event_queue,
                Event(
                    time_ms=req.arrival_offset_ms,
                    event_type="arrival",
                    request_id=req.request_id,
                    data={"request": req, "prompt_text": req.prompt, "index": idx},
                ),
            )

        # Simulation State Tracking: timeline when each local replica becomes free
        replica_free_at: dict[str, float] = {r.replica_id: 0.0 for r in registry.get_all_replicas()}

        completed_latencies_ms: list[float] = []
        queue_delay_measurements_ms: list[float] = []
        arrival_to_first_token_measurements_ms: list[float] = []
        ttft_measurements_ms: list[float] = []
        tpot_measurements_ms: list[float] = []
        costs: list[float] = []
        qualities: list[float] = []
        slo_attained_count = 0
        rejected_count = 0
        cloud_completed_count = 0
        local_completed_count = 0
        per_request_records: list[dict[str, Any]] = []

        sim_clock_ms = 0.0

        while event_queue:
            event = heapq.heappop(event_queue)
            sim_clock_ms = event.time_ms

            if event.event_type == "arrival":
                req_workload = event.data["request"]

                # P1 Assessment GAP-07: Use unified OutputLengthEstimator (NOT int(prompt_tokens * 1.5))
                est_out = self.estimator.estimate(
                    task_type=req_workload.task_type,
                    prompt_tokens=req_workload.prompt_tokens,
                    max_tokens=getattr(req_workload, "max_tokens", None),
                )

                ctx = RequestContext(
                    task_type=req_workload.task_type,
                    prompt_tokens=req_workload.prompt_tokens,
                    estimated_output_tokens=est_out,
                    quality_requirement=req_workload.quality_requirement,
                    latency_slo=req_workload.latency_slo,
                    privacy_level=req_workload.privacy_level,
                    budget=req_workload.budget,
                    tenant_id=req_workload.tenant_id,
                    max_tokens=getattr(req_workload, "max_tokens", None),
                )

                # Stage 1: Model Selection
                spec, _est_cost, _err = policy.select_model(ctx)
                if not spec:
                    rejected_count += 1
                    per_request_records.append(
                        {
                            "request_id": req_workload.request_id,
                            "status": "rejected",
                            "reason": "model_selection_rejected",
                        }
                    )
                    continue

                # Stage 2: Scheduling (Parity with api.py: Cloud models skip local replica queue)
                is_cloud = spec.is_cloud
                if is_cloud:
                    # Cloud models execute via direct provider endpoint without local replica bottleneck
                    rep_id = f"{spec.provider}-cloud"
                    queue_delay_ms = 0.0
                    ttft_ms = 65.0 + (ctx.prompt_tokens * 0.04)
                    tpot_ms = 4.5
                else:
                    replica = scheduler.select_replica(spec, ctx)
                    if not replica:
                        rejected_count += 1
                        per_request_records.append(
                            {
                                "request_id": req_workload.request_id,
                                "status": "rejected",
                                "reason": "replica_selection_rejected",
                                "model": spec.model_id,
                            }
                        )
                        continue
                    rep_id = replica.replica_id
                    free_time = replica_free_at.get(rep_id, sim_clock_ms)
                    queue_delay_ms = max(0.0, free_time - sim_clock_ms)
                    ttft_ms = 25.0 + (ctx.prompt_tokens * 0.12)
                    tpot_ms = 10.0

                actual_out_tokens = req_workload.actual_output_tokens

                # Explicit delay decomposition:
                # 1. queue_delay_ms = W_q
                # 2. arrival_to_first_token_ms = queue_delay_ms + ttft_ms
                # 3. total_latency_ms = queue_delay_ms + ttft_ms + (output_tokens * tpot_ms)
                arrival_to_first_token_ms = queue_delay_ms + ttft_ms
                execution_duration_ms = ttft_ms + (actual_out_tokens * tpot_ms)
                total_latency_ms = queue_delay_ms + execution_duration_ms

                start_serving_ms = sim_clock_ms + queue_delay_ms
                completion_time_ms = start_serving_ms + execution_duration_ms

                # Local replica queue state update
                if not is_cloud:
                    replica_free_at[rep_id] = completion_time_ms
                    scheduler.dispatch(rep_id, ctx.prompt_tokens, ctx.estimated_output_tokens)

                # Cost & Quality accounting
                req_cost = (ctx.prompt_tokens / 1000.0) * spec.input_price_per_1k + (
                    actual_out_tokens / 1000.0
                ) * spec.output_price_per_1k
                costs.append(req_cost)

                model_q = spec.quality_scores.get(ctx.task_type, 0.80)
                qualities.append(model_q)

                completed_latencies_ms.append(total_latency_ms)
                queue_delay_measurements_ms.append(queue_delay_ms)
                arrival_to_first_token_measurements_ms.append(arrival_to_first_token_ms)
                ttft_measurements_ms.append(ttft_ms)
                tpot_measurements_ms.append(tpot_ms)

                slo_met = (total_latency_ms / 1000.0) <= req_workload.latency_slo
                if slo_met:
                    slo_attained_count += 1

                if is_cloud:
                    cloud_completed_count += 1
                else:
                    local_completed_count += 1

                per_request_records.append(
                    {
                        "request_id": req_workload.request_id,
                        "model": spec.model_id,
                        "provider": spec.provider,
                        "is_cloud": is_cloud,
                        "replica_id": rep_id,
                        "prompt_tokens": ctx.prompt_tokens,
                        "output_tokens": actual_out_tokens,
                        "queue_delay_ms": round(queue_delay_ms, 2),
                        "ttft_ms": round(ttft_ms, 2),
                        "arrival_to_first_token_ms": round(arrival_to_first_token_ms, 2),
                        "total_latency_ms": round(total_latency_ms, 2),
                        "cost": round(req_cost, 6),
                        "quality": round(model_q, 4),
                        "slo_met": slo_met,
                        "status": "completed",
                    }
                )

                # Schedule completion event
                heapq.heappush(
                    event_queue,
                    Event(
                        time_ms=completion_time_ms,
                        event_type="completion",
                        request_id=event.request_id,
                        data={
                            "replica_id": rep_id,
                            "prompt_tokens": ctx.prompt_tokens,
                            "est_out_tokens": ctx.estimated_output_tokens,
                            "latency_ms": total_latency_ms,
                            "is_cloud": is_cloud,
                        },
                    ),
                )

            elif event.event_type == "completion":
                if not event.data.get("is_cloud", False):
                    rep_id = event.data["replica_id"]
                    p_tokens = event.data["prompt_tokens"]
                    est_out = event.data["est_out_tokens"]
                    lat = event.data["latency_ms"]
                    scheduler.complete(rep_id, p_tokens, est_out, lat, success=True)

        # Compute summary percentiles
        completed_latencies_ms.sort()
        queue_delay_measurements_ms.sort()
        arrival_to_first_token_measurements_ms.sort()

        p50 = completed_latencies_ms[int(len(completed_latencies_ms) * 0.50)] if completed_latencies_ms else 0.0
        p95 = completed_latencies_ms[int(len(completed_latencies_ms) * 0.95)] if completed_latencies_ms else 0.0
        p99 = completed_latencies_ms[int(len(completed_latencies_ms) * 0.99)] if completed_latencies_ms else 0.0
        p95_queue = (
            queue_delay_measurements_ms[int(len(queue_delay_measurements_ms) * 0.95)]
            if queue_delay_measurements_ms
            else 0.0
        )
        p95_arr_first = (
            arrival_to_first_token_measurements_ms[int(len(arrival_to_first_token_measurements_ms) * 0.95)]
            if arrival_to_first_token_measurements_ms
            else 0.0
        )

        slo_rate = (slo_attained_count / len(completed_latencies_ms) * 100.0) if completed_latencies_ms else 0.0
        total_sim_sec = max(0.001, sim_clock_ms / 1000.0)
        throughput = len(completed_latencies_ms) / total_sim_sec

        return {
            "policy": policy_name,
            "scheduler": scheduler_name,
            "seed": self.seed,
            "total_requests": len(requests),
            "offered_requests": len(requests),
            "accepted_requests": len(completed_latencies_ms),
            "completed_requests": len(completed_latencies_ms),
            "rejected_requests": rejected_count,
            "rejection_rate_pct": round((rejected_count / len(requests)) * 100.0, 2),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_p99_ms": round(p99, 2),
            "queue_wait_p95_ms": round(p95_queue, 2),
            "avg_queue_delay_ms": round(sum(queue_delay_measurements_ms) / max(1, len(queue_delay_measurements_ms)), 2),
            "avg_ttft_ms": round(sum(ttft_measurements_ms) / max(1, len(ttft_measurements_ms)), 2),
            "avg_tpot_ms": round(sum(tpot_measurements_ms) / max(1, len(tpot_measurements_ms)), 2),
            "avg_arrival_to_first_token_ms": round(
                sum(arrival_to_first_token_measurements_ms) / max(1, len(arrival_to_first_token_measurements_ms)), 2
            ),
            "arrival_to_first_token_p95_ms": round(p95_arr_first, 2),
            "slo_attainment_pct": round(slo_rate, 2),
            "total_cost": round(sum(costs), 6),
            "avg_quality": round(sum(qualities) / max(1, len(qualities)), 4),
            "effective_throughput_rps": round(throughput, 2),
            "simulated_duration_sec": round(total_sim_sec, 2),
            "cloud_requests_completed": cloud_completed_count,
            "local_requests_completed": local_completed_count,
            "per_request_records": per_request_records,
        }


def run_full_event_benchmark_matrix() -> dict[str, Any]:
    """Runs all 6 baseline policy combinations across multiple seeds replaying identical offered traces."""
    matrix = [
        ("strongest", "round_robin"),
        ("cheapest", "round_robin"),
        ("static", "round_robin"),
        ("adaptive", "round_robin"),
        ("adaptive", "least_loaded"),
        ("adaptive", "slo_aware"),
    ]

    seeds = [42, 101, 2026]
    all_results: dict[str, list[dict[str, Any]]] = {f"{pol}+{sched}": [] for pol, sched in matrix}

    # Pre-generate and freeze one deterministic request trace per seed
    # This guarantees that for each seed, all 6 policy combinations replay the identical offered cohort
    frozen_traces: dict[int, list[SyntheticRequest]] = {}
    for seed in seeds:
        gen = WorkloadGenerator(seed=seed)
        frozen_traces[seed] = gen.generate_trace(count=30)

    for pol, sched in matrix:
        key = f"{pol}+{sched}"
        for seed in seeds:
            sim = EventDrivenSimulator(arrival_rate_rps=10.0, seed=seed)
            res = sim.run_benchmark(pol, sched, num_requests=30, offered_trace=frozen_traces[seed])
            all_results[key].append(res)

    # Compute aggregates (mean across seeds)
    summary_report: dict[str, Any] = {
        "metadata": {
            "description": "AdaRoute Event-Driven Discrete-Event Simulator Benchmark Matrix (Week-4 Hardened)",
            "unit_pricing_audit": {
                "gemini/gemini-flash-latest": {
                    "input_price_per_1k": 0.000075,
                    "output_price_per_1k": 0.000300,
                    "input_price_per_1M": 0.075,
                    "output_price_per_1M": 0.30,
                    "source": "Official Google AI Studio Pricing ($0.30 / 1M tokens = $0.000300 / 1k tokens, resolving 1000x discrepancy)",
                }
            },
            "evaluation_parity": {
                "cloud_scheduling": "Cloud models execute direct without local replica queue bottleneck (parity with api.py)",
                "estimation": "Unified OutputLengthEstimator pre-inference estimation",
                "cohort_matching": "Deterministic frozen trace replayed identically across all policies per seed",
                "delay_decomposition": "Queue delay W_q separated from TTFT and TPOT",
            },
        }
    }

    for key, seed_runs in all_results.items():
        n = len(seed_runs)
        summary_report[key] = {
            "policy": seed_runs[0]["policy"],
            "scheduler": seed_runs[0]["scheduler"],
            "num_runs": n,
            "avg_offered_requests": round(sum(r["offered_requests"] for r in seed_runs) / n, 1),
            "avg_completed_requests": round(sum(r["completed_requests"] for r in seed_runs) / n, 1),
            "avg_rejected_requests": round(sum(r["rejected_requests"] for r in seed_runs) / n, 1),
            "avg_rejection_rate_pct": round(sum(r["rejection_rate_pct"] for r in seed_runs) / n, 2),
            "avg_latency_p95_ms": round(sum(r["latency_p95_ms"] for r in seed_runs) / n, 2),
            "avg_latency_p50_ms": round(sum(r["latency_p50_ms"] for r in seed_runs) / n, 2),
            "avg_queue_wait_p95_ms": round(sum(r["queue_wait_p95_ms"] for r in seed_runs) / n, 2),
            "avg_queue_delay_ms": round(sum(r["avg_queue_delay_ms"] for r in seed_runs) / n, 2),
            "avg_ttft_ms": round(sum(r["avg_ttft_ms"] for r in seed_runs) / n, 2),
            "avg_tpot_ms": round(sum(r["avg_tpot_ms"] for r in seed_runs) / n, 2),
            "avg_arrival_to_first_token_ms": round(sum(r["avg_arrival_to_first_token_ms"] for r in seed_runs) / n, 2),
            "avg_slo_attainment_pct": round(sum(r["slo_attainment_pct"] for r in seed_runs) / n, 2),
            "avg_total_cost": round(sum(r["total_cost"] for r in seed_runs) / n, 6),
            "avg_quality": round(sum(r["avg_quality"] for r in seed_runs) / n, 4),
            "effective_throughput_rps": round(sum(r["effective_throughput_rps"] for r in seed_runs) / n, 2),
            "runs": seed_runs,
        }

    # Persist summary JSON
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / "event_benchmark_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    return summary_report


if __name__ == "__main__":
    print("Running AdaRoute Event-Driven Benchmark Simulation Matrix (Week-4 Hardened)...")
    report = run_full_event_benchmark_matrix()
    for combo, data in report.items():
        if combo == "metadata":
            continue
        print(
            f"[{combo}]: p95={data['avg_latency_p95_ms']}ms | "
            f"Queue_Wait_p95={data['avg_queue_wait_p95_ms']}ms | "
            f"Arrival_to_First_Token={data['avg_arrival_to_first_token_ms']}ms | "
            f"Cost=${data['avg_total_cost']} | "
            f"Quality={data['avg_quality']} | "
            f"SLO={data['avg_slo_attainment_pct']}% | "
            f"Throughput={data['effective_throughput_rps']} rps | "
            f"Serviced={data['avg_completed_requests']}/{data['avg_offered_requests']}"
        )
