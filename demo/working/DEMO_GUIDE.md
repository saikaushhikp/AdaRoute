# AdaRoute Live Demonstration Guide (`just cli`)

This document outlines the **authoritative 15-step demonstration script** for presenting AdaRoute through the interactive terminal workbench (`just cli`).

Every step has been systematically engineered to demonstrate a core gateway capability, prove an architectural invariant, and showcase real-time telemetry, routing, caching, and fail-safe controls to an audience.

---

## Pre-Demo Quickstart: Fresh Environment Setup

To ensure a pristine demo state with empty cache and zero in-flight state:
```bash
# 1. Spin down and bring up fresh GPU containers
just down
just gpu-up

# 2. Wait ~20-30s until gateway and all 3 vLLM engine replicas report healthy
curl -s http://localhost:7000/health | jq .status

# 3. Launch the terminal workbench
just cli
```

---

## The 15-Step Master Demonstration Sequence

```
Step  1:  /health                                      [Cluster Health & Topology]
Step  2:  /set policy cheapest                         [Policy: Cost Minimization]
Step  3:  Write a python function to compute...       [Local 0.5B Model Dispatch]
Step  4:  /set policy strongest                        [Policy: Capability Maximization]
Step  5:  Write a python function to compute...       [Cloud 27B Model Dispatch]
Step  6:  /private                                     [Hard Privacy Constraint ON]
Step  7:  Confidential patient record: calculate...    [Sovereign Local Enforcement]
Step  8:  /diag                                        [Deep Audit & Rejection Telemetry]
Step  9:  /private (OFF) + /set cap coding             [Local Multi-Replica Environment]
          /set policy cheapest + /set scheduler round_robin
Step 10:  Echo test dispatch alpha                     [Replica 1 Dispatch (Port 7002)]
Step 11:  Echo test dispatch beta                      [Replica 2 Dispatch (Port 7003)]
Step 12:  What is the boiling point of pure water...   [Cache Cold Seed -> CACHE MISS]
Step 13:  What is the boiling point of pure water...   [Exact Repeat -> CACHE HIT exact]
Step 14:  At what temperature does pure water boil?    [Semantic Paraphrase -> CACHE HIT]
Step 15:  /set budget 0.0000001                        [Budget Ceiling Admission Failure]
          Explain quantum computing in detail.
          /set budget 1.0                              [Restore Budget & /exit]
```

---

## Detailed Step-by-Step Walkthrough

### Step 1: System Health & Multi-Replica Discovery
- **Action / Command:**
  ```text
  /health
  ```
- **What Viewer Sees:**
  ```text
  ✔ Gateway is healthy and reachable!
  Available Models: unsloth/Qwen3-0.6B-GGUF:Q4_K_M, unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M, gemini/gemini-flash-latest, groq/qwen/qwen3.8-27b, cohere/command-r-08-2024, gpt-4o-external, llama-3-8b
  Active Replicas: vllm-gpu-gguf-qwen3-r1, vllm-gpu-gguf-qwen2.5-r1, vllm-gpu-gguf-qwen2.5-r2, replica-gpt-4o-1, replica-llama-1
  ```
- **Why This Step is Placed Here:**
  Establishes baseline trust. Proves to the audience that AdaRoute is not a mock stub; it is connected to a live model catalog and actively supervising three independent local GPU vLLM engine processes (`qwen3-r1` on port 7001, `qwen2.5-r1` on port 7002, `qwen2.5-r2` on port 7003) alongside cloud endpoints.
- **Presenter Talking Point:**
  > *"Notice that our cluster health endpoint exposes both logical models and distinct physical replica instances. Three vLLM engines are running concurrently on our local NVIDIA RTX PRO 4000 GPU, ready for multi-replica load scheduling."*

---

### Step 2 & 3: Cost-Minimizing Model Policy (`cheapest`)
- **Action / Command:**
  ```text
  /set policy cheapest
  ```
- **Prompt:**
  ```text
  Write a python function to compute fibonacci(n).
  ```
- **What Viewer Sees:**
  The assistant streams or prints a Python function. The mini-telemetry bar displays:
  ```text
  Model: unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M
  Replica: vllm-gpu-gguf-qwen2.5-r1 (or r2)
  Latency: ~500ms
  Cost: $0.000001
  CACHE MISS  SLO: ✅
  ```
- **Why This Step is Placed Here:**
  Demonstrates Stage 1 objective routing. For a standard coding task, the `cheapest` policy evaluates catalog prices and selects the lightweight on-premises local GPU model, costing fractions of a cent ($0.000001) while attaining the latency SLO.

---

### Step 4 & 5: Capability-Maximizing Model Policy (`strongest`)
- **Action / Command:**
  ```text
  /set policy strongest
  ```
- **Prompt:** (Exact same prompt as Step 3)
  ```text
  Write a python function to compute fibonacci(n).
  ```
- **What Viewer Sees:**
  The assistant returns a comprehensive, type-annotated implementation. The mini-telemetry bar displays:
  ```text
  Model: groq/qwen/qwen3.8-27b (or gemini/gemini-flash-latest)
  Replica: cloud-direct
  Latency: ~400-700ms
  Cost: ~$0.000180
  CACHE MISS  SLO: ✅
  ```
- **Why This Step is Placed Here:**
  **Directly proves model selection is dynamic and working.** For the exact same prompt, simply changing the policy from `cheapest` to `strongest` causes AdaRoute to dynamically re-evaluate the Pareto frontier and route from a 0.5B local model to a 27B parameter frontier cloud model. Not all requests go to the same model!
- **Presenter Talking Point:**
  > *"Notice that without changing the prompt, switching policy from cheapest to strongest re-routed the request from our local GGUF engine to a 27B cloud model. AdaRoute's Stage 1 Pareto router dynamically balances cost versus capability."*

---

### Step 6 & 7: Hard Privacy Constraints & Sovereign Local Override
- **Action / Command:**
  ```text
  /private
  ```
- **Prompt:**
  ```text
  Confidential patient record: calculate average heart rate for patient 991.
  ```
- **What Viewer Sees:**
  Header shows `User Private: ON (local only)`. The mini-telemetry bar displays:
  ```text
  Model: unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M
  Replica: vllm-gpu-gguf-qwen2.5-r1
  Cost: $0.000001
  CACHE MISS  SLO: ✅
  ```
- **Why This Step is Placed Here:**
  Demonstrates hard constraint enforcement over objective scoring. Even though the active policy is still set to `strongest` (which normally prefers Groq/Gemini), the `user_private: true` constraint **hard-rejects all external cloud providers** and strictly confines inference to the sovereign local GPU engine.
- **Presenter Talking Point:**
  > *"Here, policy is still set to strongest, but the private toggle is ON. Look at the model: it did NOT send patient data to cloud providers. Privacy is an uncompromising hard filter in AdaRoute, not an objective that can be traded away for higher quality."*

---

### Step 8: Telemetry, Audit Trail & Decision Diagnostics
- **Action / Command:**
  ```text
  /diag
  ```
- **What Viewer Sees:**
  A comprehensive Rich diagnostics table opens:
  ```text
  ╭──────────────────────────────── Last Routing Telemetry ────────────────────────────────╮
  │ Selected Concrete Model   unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M         │
  │ Selected Serving Replica  vllm-gpu-gguf-qwen2.5-r1                                     │
  │ Execution Provider        vllm                                                         │
  │ Policy Applied            strongest                                                    │
  │ Privacy Partition         user_private=True (Local isolated)                           │
  │ Hard Rejections           deploy-groq/qwen/qwen3.8-27b (Excluded: cloud provider...) │
  │ Pareto Survivors          deploy-unsloth/Qwen2.5-Coder...                              │
  ╰────────────────────────────────────────────────────────────────────────────────────────╯
  ```
- **Why This Step is Placed Here:**
  Demonstrates auditability, explainability, and regulatory compliance. The user can inspect the exact mathematical reasoning for every routing decision, proving that cloud candidates were explicitly excluded by the privacy guard.

---

### Step 9: Reset Privacy & Configure Multi-Replica Environment
- **Action / Commands:**
  ```text
  /private
  /set cap coding
  /set policy cheapest
  /set scheduler round_robin
  ```
- **What Viewer Sees:**
  User Private mode turns OFF. Capability set to `coding`, policy set to `cheapest`, scheduler set to `round_robin`.
- **Why This Step is Placed Here:**
  Prepares the Stage 2 multi-replica testbed. The coding capability maps to `Qwen2.5-Coder-0.5B`, which has two independent live replica processes running on the GPU: `vllm-gpu-gguf-qwen2.5-r1` (Port 7002) and `vllm-gpu-gguf-qwen2.5-r2` (Port 7003).

---

### Step 10: Multi-Replica Dispatch 1 (Target: Replica 1)
- **Prompt:**
  ```text
  Echo test dispatch alpha
  ```
- **What Viewer Sees:**
  Mini-telemetry bar shows:
  ```text
  Model: unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M
  Replica: vllm-gpu-gguf-qwen2.5-r1
  Latency: ~180-250ms
  ```
- **Why This Step is Placed Here:**
  First dispatch under round-robin. It hits **Replica 1 (`r1`)** on container port 7002.

---

### Step 11: Multi-Replica Dispatch 2 (Target: Replica 2)
- **Prompt:**
  ```text
  Echo test dispatch beta
  ```
- **What Viewer Sees:**
  Mini-telemetry bar shows:
  ```text
  Model: unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M
  Replica: vllm-gpu-gguf-qwen2.5-r2
  Latency: ~180-250ms
  ```
- **Why This Step is Placed Here:**
  **Directly proves replica scheduling is working (not all requests go to the same replica).** The very next request is automatically steered to sibling **Replica 2 (`r2`)** on container port 7003, demonstrating true process-level load balancing across independent vLLM engines on the shared GPU.
- **Presenter Talking Point:**
  > *"Look at the Replica field: Request 1 went to Replica 1 (port 7002); Request 2 went to Replica 2 (port 7003). Even though both run on the same Blackwell GPU, AdaRoute balances traffic across distinct process queues to prevent head-of-line blocking."*

---

### Step 12: Cache Cold Seed (Cache Miss)
- **Prompt:**
  ```text
  What is the boiling point of pure water in Celsius?
  ```
- **What Viewer Sees:**
  ```text
  Model: unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M
  Latency: ~300-500ms
  Cost: $0.000001
  CACHE MISS  SLO: ✅
  ```
- **Why This Step is Placed Here:**
  Initiates the caching demonstration. Seeds the Redis vector index with an initial prompt-completion pair and records the baseline live inference latency (~300ms).

---

### Step 13: Exact Response Caching (Instant Zero-Cost Hit)
- **Prompt:** (Identical to Step 12)
  ```text
  What is the boiling point of pure water in Celsius?
  ```
- **What Viewer Sees:**
  ```text
  Model: unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M
  Latency: 0.4ms
  Cost: $0.000000
  CACHE HIT (exact)  SLO: ✅
  ```
- **Why This Step is Placed Here:**
  Proves exact response caching. Latency drops from **~350ms to 0.4ms** (a 99.8% speedup) and cost drops to **$0.000000**, with zero compute load on the GPU or cloud backends.
- **Presenter Talking Point:**
  > *"Look at that latency: 0.4 milliseconds and zero cost. For identical recurring prompts, AdaRoute serves responses directly from RAM/Redis before reaching any model selector or inference engine."*

---

### Step 14: Semantic Vector Caching (Natural Language Paraphrase Hit)
- **Prompt:** (Semantically identical, completely different wording)
  ```text
  At what temperature in Celsius does pure water boil?
  ```
- **What Viewer Sees:**
  ```text
  Model: unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M
  Latency: 3.7ms
  Cost: $0.000000
  CACHE HIT (semantic: 90.8%)  SLO: ✅
  ```
- **Why This Step is Placed Here:**
  **Demonstrates AI-native semantic vector search caching.** Even though the prompt string is entirely different, AdaRoute computes MiniLM embeddings, queries Redis Vector Search (`HNSW`), identifies a 90.8% cosine similarity match, and returns the cached answer in 3.7ms without executing model inference.
- **Presenter Talking Point:**
  > *"The prompt wording was completely different, but notice the badge: CACHE HIT (semantic: 90.8%). AdaRoute understands semantic intent and avoided calling the LLM, returning the answer in 3.7 milliseconds."*

---

### Step 15: Economic Guardrails, Budget Rejection & Clean Reset
- **Action / Command:**
  ```text
  /set budget 0.0000001
  ```
- **Prompt:**
  ```text
  Explain quantum computing in detail.
  ```
- **What Viewer Sees:**
  A red error panel appears:
  ```text
  ╭─────────────────────────────── Gateway Error ────────────────────────────────╮
  │ HTTP 503 Error:                                                              │
  │ Routing rejected: Budget exceeded: Estimated cost $0.00030 > budget $0.00000 │
  ╰──────────────────────────────────────────────────────────────────────────────╯
  ```
- **Reset Command & Exit:**
  ```text
  /set budget 1.0
  /exit
  ```
- **Why This Step is Placed Here:**
  Demonstrates financial safety & fail-closed admission control. When a request would violate the caller's budget ceiling, AdaRoute terminates the request immediately before any provider is invoked. Resetting the budget to $1.00 leaves the workbench ready for subsequent testing before exiting cleanly.
- **Presenter Talking Point:**
  > *"Notice our fail-closed economic safety: an impossibly low budget rejected the query before sending a single token to any model. AdaRoute protects organizations from unexpected runaway bills."*

---

## Quick Reference Summary Table

| Step # | Action / Input | Expected Badge / Output | Core Invariant Demonstrated |
| :---: | :--- | :--- | :--- |
| **1** | `/health` | `✔ Gateway is healthy (3 replicas)` | Live cluster topology & 3 vLLM processes |
| **2-3**| `/set policy cheapest` + Fibonacci | `Model: Qwen2.5-Coder-0.5B`, `$1e-6` | Cost-minimizing Stage 1 policy |
| **4-5**| `/set policy strongest` + Fibonacci | `Model: qwen3.8-27b`, `cloud-direct` | **Dynamic Model Selection (Not same model)** |
| **6-7**| `/private` + Patient Record | `Model: Qwen2.5-Coder-0.5B` | **Privacy Override (Forces local GPU)** |
| **8** | `/diag` | `Hard Rejections: cloud excluded` | Deep observability & audit trail |
| **9-10**| `/set scheduler round_robin` + Echo Alpha | `Replica: vllm-gpu-gguf-qwen2.5-r1` | Stage 2 Scheduling (Replica 1, Port 7002) |
| **11** | Echo Beta | `Replica: vllm-gpu-gguf-qwen2.5-r2` | **Dynamic Replica Scheduling (Not same replica)** |
| **12** | Water boiling point | `CACHE MISS (Latency: ~350ms)` | Cold cache seed lifecycle |
| **13** | Water boiling point (repeat) | `CACHE HIT (exact, Latency: 0.4ms)`| Exact prompt caching (99.8% speedup) |
| **14** | Water boiling point (paraphrase) | `CACHE HIT (semantic: 90.8%, 3.7ms)`| AI-native semantic vector search cache |
| **15** | `/set budget 0.0000001` + Quantum | `HTTP 503: Budget exceeded` + `/exit`| Fail-closed economic spend protection |
