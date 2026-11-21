# AdaRoute Matched-Load Multi-Replica Benchmark Summary

**Run Timestamp:** 20261121T112744.199772Z  
**GPU Hardware:** NVIDIA RTX PRO 4000 Blackwell SFF Edition (VRAM: 8082.0 / 24467.0 MiB)  
**Evaluated Model:** `unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M`  
**Replicas:** `vllm-gpu-gguf-qwen2.5-r1` (Port 7002) and `vllm-gpu-gguf-qwen2.5-r2` (Port 7003)

## 1. Scenario A: Balanced Load (Equal Offered Requests)

| Scheduler | p50 Latency (ms) | p95 Latency (ms) | Mean (ms) | Throughput (req/s) | Dispatches (R1 / R2) | SLO Attainment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **round_robin** | 565.5 | 644.2 | 516.5 | 1.22 | 3 / 3 | 100.0% |
| **least_loaded** | 589.8 | 652.1 | 507.9 | 1.24 | 6 / 0 | 100.0% |
| **slo_aware** | 579.1 | 652.2 | 504.3 | 1.24 | 6 / 0 | 100.0% |

## 2. Scenario B: Asymmetric Load Contention (Replica 1 Under Queue Pressure)

| Scheduler | p50 Latency (ms) | p95 Latency (ms) | Mean (ms) | Throughput (req/s) | Dispatches (R1 / R2) | Steered to R2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **round_robin** | 571.0 | 636.5 | 498.4 | 1.25 | 3 / 3 | **NO** |
| **least_loaded** | 571.9 | 662.0 | 505.4 | 1.24 | 4 / 2 | **NO** |
| **slo_aware** | 564.4 | 641.8 | 491.4 | 1.26 | 0 / 6 | **YES** |

## 3. Independent Engine Prometheus Metrics
- **Port 7002 (Replica 1):** +707 tokens generated
- **Port 7003 (Replica 2):** +511 tokens generated

## 4. Honest Disclosures & Resource Bounds
- **Single Container Contention:** Both engines run inside the single `adaroute-vllm` Docker container and share the single GPU (NVIDIA RTX PRO 4000 Blackwell SFF Edition). Multi-replica serving enables parallel request handling and independent process queues, but shares compute units and PCIe bandwidth.
- **Tail Percentile Notice:** Sample sizes (6 requests per scenario run) are illustrative benchmarks demonstrating scheduling dynamics, queue steering, and independent endpoint reachability, not statistically bound high-volume tail guarantees.
