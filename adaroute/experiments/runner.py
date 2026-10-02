"""Benchmark Runner and Evaluation Harness for AdaRoute.

Executes baseline combinations over independent workload traces,
recording latency, throughput, cost, quality attainment, and estimator MAE.
"""

import json
import os
import time
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


class BenchmarkRunner:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.registry = ModelRegistry()
        self.generator = WorkloadGenerator(seed=seed)
        self.estimator = OutputLengthEstimator()

    def run_policy_combination(
        self,
        model_policy_name: str,
        scheduler_name: str,
        requests: list[SyntheticRequest],
    ) -> dict[str, Any]:
        # Fresh registry and policies per run to isolate state
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

            # Simulated latency
            exec_lat = replica.recent_latency_ms + (req.actual_output_tokens * 0.1)
            latencies_ms.append(exec_lat)

            # Realized cost
            actual_cost = (
                (req.prompt_tokens / 1000.0) * model_spec.input_price_per_1k
                + (req.actual_output_tokens / 1000.0) * model_spec.output_price_per_1k
            )
            costs.append(actual_cost)

            # Quality attained
            q_val = model_spec.quality_scores.get(req.task_type, 0.75)
            quality_scores.append(q_val)

            # SLO check
            if (exec_lat / 1000.0) <= req.latency_slo:
                slo_attained_count += 1

            total_tokens += (req.prompt_tokens + req.actual_output_tokens)

            # Estimator error tracking
            estimator.record_observation(req.task_type, est_out, req.actual_output_tokens)

            # Replica completion
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
        }

    def run_all(self, num_requests: int = 50, output_dir: str = "results") -> list[dict[str, Any]]:
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
            res = self.run_policy_combination(m_pol, r_sched, requests)
            results.append(res)

        summary_path = os.path.join(output_dir, "benchmark_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        return results


if __name__ == "__main__":
    runner = BenchmarkRunner(seed=42)
    print("Running AdaRoute benchmark matrix...")
    out = runner.run_all(num_requests=60)
    for r in out:
        print(f"[{r['model_policy']} + {r['replica_scheduler']}]: Cost=${r['total_cost_usd']} | Quality={r['avg_quality_score']} | SLO={r['slo_attainment_pct']}% | Latency(p95)={r['p95_latency_ms']}ms")
