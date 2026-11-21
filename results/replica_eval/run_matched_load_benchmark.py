#!/usr/bin/env python3
"""Matched-Load Multi-Replica Scheduling Benchmark for AdaRoute.

Compares Stage 2 local replica schedulers:
  - round_robin
  - least_loaded
  - slo_aware

Under:
  - Scenario 1: Balanced offered load across identical deterministic prompt mixes.
  - Scenario 2: Asymmetric load contention (replica 1 loaded, testing dynamic steering).

Measures:
  - End-to-end p50, p95, p99, mean latency
  - Throughput (req/s)
  - Replica dispatch distribution (Replica 1 vs Replica 2)
  - SLO attainment and error counts
  - Engine-side Prometheus metrics (/metrics)
  - GPU memory utilization via nvidia-smi

Saves artifacts in results/replica_eval/runs/<timestamp>/:
  - manifest.json
  - benchmark_results.json
  - benchmark_summary.md
"""

from __future__ import annotations

import datetime
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:7000")
REPLICA_MODEL = "unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M"

WORKLOAD_PROMPTS = [
    {
        "id": "p1_fib",
        "task": "coding",
        "prompt": "Write a python function to compute fibonacci numbers up to n.",
        "max_tokens": 40,
        "slo_ms": 1200.0,
    },
    {
        "id": "p2_qa",
        "task": "qa",
        "prompt": "What is the capital of France and what is its primary river?",
        "max_tokens": 25,
        "slo_ms": 800.0,
    },
    {
        "id": "p3_prime",
        "task": "coding",
        "prompt": "Write a python function is_prime(n) that checks if n is prime.",
        "max_tokens": 40,
        "slo_ms": 1200.0,
    },
    {
        "id": "p4_summary",
        "task": "summarization",
        "prompt": "Summarize what a distributed lock is and why it is used in 1 sentence.",
        "max_tokens": 30,
        "slo_ms": 1000.0,
    },
    {
        "id": "p5_stack",
        "task": "coding",
        "prompt": "Implement a python Stack class with push and pop methods.",
        "max_tokens": 45,
        "slo_ms": 1400.0,
    },
    {
        "id": "p6_concept",
        "task": "general",
        "prompt": "Explain the difference between concurrency and parallelism in 1 sentence.",
        "max_tokens": 30,
        "slo_ms": 1000.0,
    },
]


def run_cmd(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def get_gpu_info() -> dict[str, Any]:
    # 1. Try querying gateway control plane
    try:
        url = f"{GATEWAY_URL}/v1/control-plane/gpu-info"
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "gpu_name" in data and not data.get("error"):
                return data
    except Exception:
        pass

    # 2. Fallback to host nvidia-smi if running on host
    try:
        out = run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        return {
            "gpu_name": parts[0],
            "total_mib": float(parts[1]),
            "free_mib": float(parts[2]),
            "used_mib": float(parts[3]),
            "driver_version": parts[4],
        }
    except Exception as e:
        return {"error": str(e)}


def read_engine_prometheus(port: int) -> dict[str, float]:
    # 1. Try querying gateway control plane
    try:
        url = f"{GATEWAY_URL}/v1/control-plane/replica/metrics?port={port}"
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "generation_tokens_total" in data:
                return data
    except Exception:
        pass

    # 2. Fallback to direct /metrics probe
    metrics = {
        "generation_tokens_total": 0.0,
        "num_requests_running": 0.0,
        "num_requests_waiting": 0.0,
        "kv_cache_usage_perc": 0.0,
    }
    try:
        with urllib.request.urlopen(f"http://vllm:{port}/metrics", timeout=2.0) as resp:
            for line in resp.read().decode("utf-8").splitlines():
                line = line.strip()
                if line.startswith("vllm:generation_tokens_total") and not line.startswith("#"):
                    try:
                        metrics["generation_tokens_total"] = float(line.split()[-1])
                    except Exception:
                        pass
                elif line.startswith(
                    ("vllm:num_requests_running", "vllm:num_requests_running{")
                ) and not line.startswith("#"):
                    try:
                        metrics["num_requests_running"] = float(line.split()[-1])
                    except Exception:
                        pass
                elif line.startswith(
                    ("vllm:num_requests_waiting", "vllm:num_requests_waiting{")
                ) and not line.startswith("#"):
                    try:
                        metrics["num_requests_waiting"] = float(line.split()[-1])
                    except Exception:
                        pass
                elif line.startswith(
                    ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_factor")
                ) and not line.startswith("#"):
                    try:
                        metrics["kv_cache_usage_perc"] = float(line.split()[-1])
                    except Exception:
                        pass
    except Exception:
        pass
    return metrics


def set_replica_asymmetric_load(
    r1_waiting: float = 0.0,
    r1_running: float = 0.0,
    r1_active: int = 0,
    r1_tokens: int = 0,
) -> None:
    payload = {
        "replica_id": "vllm-gpu-gguf-qwen2.5-r1",
        "metrics": {
            "num_requests_waiting": r1_waiting,
            "num_requests_running": r1_running,
        },
        "active_requests": r1_active,
        "workload_tokens": r1_tokens,
    }
    req = urllib.request.Request(
        f"{GATEWAY_URL}/v1/control-plane/replica-state",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0):
            pass
    except Exception as e:
        print(f"  Warning: failed to set replica state: {e}")


def send_request(
    model: str,
    prompt: str,
    scheduler: str,
    max_tokens: int,
    slo_ms: float = 2000.0,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "no_cache": True,
        "user_private": True,
        "execution_mode": "real",
        "scheduler": scheduler,
        "latency_slo": slo_ms / 1000.0,
    }
    req = urllib.request.Request(
        f"{GATEWAY_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            lat = (time.perf_counter() - t0) * 1000.0
            data = json.loads(resp.read().decode("utf-8"))
            data["_client_lat_ms"] = lat
            return data
    except urllib.error.HTTPError as e:
        lat = (time.perf_counter() - t0) * 1000.0
        err_msg = f"HTTP Error {e.code}: {e.read().decode('utf-8', errors='ignore')}"
        return {"error": err_msg, "_client_lat_ms": lat}
    except Exception as e:
        lat = (time.perf_counter() - t0) * 1000.0
        return {"error": str(e), "_client_lat_ms": lat}


def run_workload_for_scheduler(
    scheduler: str,
    scenario_name: str,
    asymmetric_load: bool = False,
) -> dict[str, Any]:
    print(f"\n  Running Scheduler: [{scheduler}] | Scenario: [{scenario_name}]...")
    records = []
    t_start = time.perf_counter()

    # If asymmetric scenario, simulate an active load / queue on replica 1
    if asymmetric_load:
        set_replica_asymmetric_load(
            r1_waiting=4.0,
            r1_running=2.0,
            r1_active=3,
            r1_tokens=300,
        )

    for idx, item in enumerate(WORKLOAD_PROMPTS):
        res = send_request(
            model=REPLICA_MODEL,
            prompt=item["prompt"],
            scheduler=scheduler,
            max_tokens=item["max_tokens"],
            slo_ms=item["slo_ms"],
        )
        r_meta = res.get("routing_metadata", {})
        err = res.get("error")
        lat = res.get("_client_lat_ms", 0.0)
        sel_rep = r_meta.get("selected_replica", "unknown")
        slo_attained = lat <= item["slo_ms"] and err is None

        records.append(
            {
                "prompt_id": item["id"],
                "task": item["task"],
                "scheduler": scheduler,
                "selected_replica": sel_rep,
                "latency_ms": round(lat, 2),
                "slo_ms": item["slo_ms"],
                "slo_attained": slo_attained,
                "error": err,
                "content": res.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "prompt_tokens": r_meta.get("attempt_history", [{}])[0].get("prompt_tokens", 0),
                "completion_tokens": r_meta.get("attempt_history", [{}])[0].get("completion_tokens", 0),
            }
        )
        time.sleep(0.3)

    t_total = time.perf_counter() - t_start

    # Reset asymmetric metrics
    if asymmetric_load:
        set_replica_asymmetric_load(
            r1_waiting=0.0,
            r1_running=0.0,
            r1_active=0,
            r1_tokens=0,
        )

    latencies = [r["latency_ms"] for r in records if not r["error"]]
    latencies.sort()
    count = len(latencies)

    p50 = latencies[int(count * 0.50)] if count else 0.0
    p95 = latencies[int(count * 0.95)] if count else 0.0
    p99 = latencies[int(count * 0.99)] if count else 0.0
    mean_lat = statistics.mean(latencies) if count else 0.0
    throughput = round(len(records) / max(t_total, 0.001), 2)

    r1_count = sum(1 for r in records if r["selected_replica"] == "vllm-gpu-gguf-qwen2.5-r1")
    r2_count = sum(1 for r in records if r["selected_replica"] == "vllm-gpu-gguf-qwen2.5-r2")
    slo_rate = round(sum(1 for r in records if r["slo_attained"]) / max(len(records), 1), 3)

    stats = {
        "scheduler": scheduler,
        "scenario": scenario_name,
        "total_requests": len(records),
        "successful_requests": count,
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "throughput_req_per_sec": throughput,
        "slo_attainment_rate": slo_rate,
        "replica_distribution": {
            "vllm-gpu-gguf-qwen2.5-r1": r1_count,
            "vllm-gpu-gguf-qwen2.5-r2": r2_count,
        },
    }

    print(
        f"    p50: {p50:.1f}ms | p95: {p95:.1f}ms | Throughput: {throughput} req/s | Dispatches: r1={r1_count}, r2={r2_count} | SLO Attainment: {slo_rate * 100:.1f}%"
    )
    return {"stats": stats, "records": records}


def main() -> int:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out_dir = REPO_ROOT / "results" / "replica_eval" / "runs" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("AdaRoute Matched-Load Multi-Replica Scheduling Benchmark")
    print(f"Timestamp: {timestamp}")
    print(f"Artifacts Directory: {out_dir}")
    print("=" * 78)

    # 1. Warm-up
    print("\n[Phase 1/4] Warming up inference engines...")
    send_request(REPLICA_MODEL, "Warmup ping.", "round_robin", 10)
    time.sleep(1.0)

    # 2. Snapshot Initial Hardware & Metrics
    gpu_initial = get_gpu_info()
    r1_metrics_initial = read_engine_prometheus(7002)
    r2_metrics_initial = read_engine_prometheus(7003)

    schedulers = ["round_robin", "least_loaded", "slo_aware"]
    results_by_scenario: dict[str, Any] = {}

    # 3. Scenario A: Balanced Load
    print("\n[Phase 2/4] Scenario A: Balanced Load (Uniform Offered Workload)")
    balanced_results = {}
    for s in schedulers:
        res = run_workload_for_scheduler(s, scenario_name="balanced_load", asymmetric_load=False)
        balanced_results[s] = res
    results_by_scenario["balanced_load"] = balanced_results

    # 4. Scenario B: Asymmetric Load
    print("\n[Phase 3/4] Scenario B: Asymmetric Load (Replica 1 Pressure Steering)")
    asymmetric_results = {}
    for s in schedulers:
        res = run_workload_for_scheduler(s, scenario_name="asymmetric_contention", asymmetric_load=True)
        asymmetric_results[s] = res
    results_by_scenario["asymmetric_contention"] = asymmetric_results

    # Snapshot Final Hardware & Metrics
    gpu_final = get_gpu_info()
    r1_metrics_final = read_engine_prometheus(7002)
    r2_metrics_final = read_engine_prometheus(7003)

    # 5. Build Artifacts
    print("\n[Phase 4/4] Writing benchmark evaluation manifest and summary...")
    manifest = {
        "run_id": f"benchmark_{timestamp}",
        "timestamp": timestamp,
        "execution_mode": "real",
        "model_id": REPLICA_MODEL,
        "schedulers_evaluated": schedulers,
        "gpu_initial": gpu_initial,
        "gpu_final": gpu_final,
        "r1_port": 7002,
        "r2_port": 7003,
        "r1_metrics_initial": r1_metrics_initial,
        "r1_metrics_final": r1_metrics_final,
        "r2_metrics_initial": r2_metrics_initial,
        "r2_metrics_final": r2_metrics_final,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    full_output = {
        "manifest": manifest,
        "scenarios": results_by_scenario,
    }
    with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    # Markdown Summary
    md = f"""# AdaRoute Matched-Load Multi-Replica Benchmark Summary

**Run Timestamp:** {timestamp}  
**GPU Hardware:** {gpu_final.get("gpu_name")} (VRAM: {gpu_final.get("used_mib")} / {gpu_final.get("total_mib")} MiB)  
**Evaluated Model:** `{REPLICA_MODEL}`  
**Replicas:** `vllm-gpu-gguf-qwen2.5-r1` (Port 7002) and `vllm-gpu-gguf-qwen2.5-r2` (Port 7003)

## 1. Scenario A: Balanced Load (Equal Offered Requests)

| Scheduler | p50 Latency (ms) | p95 Latency (ms) | Mean (ms) | Throughput (req/s) | Dispatches (R1 / R2) | SLO Attainment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for s in schedulers:
        st = balanced_results[s]["stats"]
        r1_c = st["replica_distribution"]["vllm-gpu-gguf-qwen2.5-r1"]
        r2_c = st["replica_distribution"]["vllm-gpu-gguf-qwen2.5-r2"]
        md += f"| **{s}** | {st['p50_latency_ms']:.1f} | {st['p95_latency_ms']:.1f} | {st['mean_latency_ms']:.1f} | {st['throughput_req_per_sec']} | {r1_c} / {r2_c} | {st['slo_attainment_rate'] * 100:.1f}% |\n"

    md += """
## 2. Scenario B: Asymmetric Load Contention (Replica 1 Under Queue Pressure)

| Scheduler | p50 Latency (ms) | p95 Latency (ms) | Mean (ms) | Throughput (req/s) | Dispatches (R1 / R2) | Steered to R2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for s in schedulers:
        st = asymmetric_results[s]["stats"]
        r1_c = st["replica_distribution"]["vllm-gpu-gguf-qwen2.5-r1"]
        r2_c = st["replica_distribution"]["vllm-gpu-gguf-qwen2.5-r2"]
        steered = "YES" if r2_c > r1_c else "NO"
        md += f"| **{s}** | {st['p50_latency_ms']:.1f} | {st['p95_latency_ms']:.1f} | {st['mean_latency_ms']:.1f} | {st['throughput_req_per_sec']} | {r1_c} / {r2_c} | **{steered}** |\n"

    tokens_r1 = r1_metrics_final.get("generation_tokens_total", 0) - r1_metrics_initial.get(
        "generation_tokens_total", 0
    )
    tokens_r2 = r2_metrics_final.get("generation_tokens_total", 0) - r2_metrics_initial.get(
        "generation_tokens_total", 0
    )

    md += f"""
## 3. Independent Engine Prometheus Metrics
- **Port 7002 (Replica 1):** +{tokens_r1:.0f} tokens generated
- **Port 7003 (Replica 2):** +{tokens_r2:.0f} tokens generated

## 4. Honest Disclosures & Resource Bounds
- **Single Container Contention:** Both engines run inside the single `adaroute-vllm` Docker container and share the single GPU ({gpu_final.get("gpu_name")}). Multi-replica serving enables parallel request handling and independent process queues, but shares compute units and PCIe bandwidth.
- **Tail Percentile Notice:** Sample sizes (6 requests per scenario run) are illustrative benchmarks demonstrating scheduling dynamics, queue steering, and independent endpoint reachability, not statistically bound high-volume tail guarantees.
"""
    with open(out_dir / "benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + "=" * 78)
    print("Matched-Load Benchmark Complete!")
    print(f"Results and summary written to: {out_dir}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
