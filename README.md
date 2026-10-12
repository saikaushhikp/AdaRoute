# AdaRoute: Multi-Objective Adaptive LLM Gateway

**AdaRoute** is a provider-independent AI gateway designed with cost-quality routing, safe caching, and scalable model serving. It addresses the challenges of serving heterogeneous LLM workloads across local open models and external cloud providers by decoupling serving decisions into two evaluation stages:
1. **Stage 1 — Model & Provider Selection:** Evaluates incoming request requirements (task classification, quality targets, cost budget, privacy policies, `user_private` mode) to identify the appropriate model class and provider.
2. **Stage 2 — Replica Scheduling:** Evaluates real-time backend state (workload-weighted queue depth $W_q$, memory pressure, moving-average latency, health) to route requests to the optimal replica instance for local/self-hosted models. For external cloud models (served via LiteLLM APIs), replica scheduling is bypassed since replicas are managed remotely by the provider.

---

## 1. Incremental Progress: Week 1, Week 2 & Week 3

### Week 1: Foundation & Initial Contract
- **Packaging & Tooling**: Configured `pyproject.toml`, `.gitignore`, `Dockerfile` (python:3.10-slim), and `justfile` for reproducible execution.
- **Initial Contract**: Exposed unified `/v1/chat/completions` prototype in `adaroute/api.py`.
- **Core Schemas**: Established typed `RequestContext` and `BackendState` models.
- **Initial Tests**: Implemented baseline invariant unit tests (`TC_1.1` to `TC_1.6`).

### Week 2: Robust Vertical Slice & Policy Engine
- **P0 Privacy Fail-Closed & Canonical Validation**:
  - Implemented validated `PrivacyLevel` enum (`PUBLIC`, `INTERNAL_ONLY`, `CONFIDENTIAL`, `RESTRICTED`).
  - Added explicit backend allow-lists: third-party cloud providers only receive `PUBLIC` traffic; internal/confidential requests strictly stay on local models.
  - Fail-closed validation rejects unknown privacy labels with HTTP 400.
- **P0 Adversarial Safety Tests**:
  - Created adversarial tests where the external provider is cheaper and idle while the local backend is loaded; verified internal requests *never* leak externally.
  - Implemented hard budget ceiling evaluation using pre-inference output projections: $C_{\text{est}} \le B$.
- **M1 Stable Logical Capability Contract**:
  - Standardized `/v1/chat/completions` accepting logical capability aliases (`model="general"`, `model="coding"`, `model="reasoning"`).
- **M2 Real Local Model + Alternate/Mock Provider**:
  - Connected `LocalBackend` (supporting local Ollama open models like Qwen/Llama with local CPU fallback) alongside `MockBackend` (external cloud provider simulation).
- **M3 Genuine Distinct Baselines & Model Registry**:
  - Built `ModelRegistry` tracking capability tiers, quality scores, pricing, and allowed privacy.
  - Implemented distinct Stage 1 policies: `StrongestModelPolicy`, `CheapestModelPolicy`, `StaticRulePolicy`, and multi-objective `AdaptiveModelPolicy`.
- **M4 Safe Exact Caching**:
  - Built `ExactCache` with SHA-256 composite keys (`tenant_id`, `user_id`, `logical_model`, `model_version`, `policy_version`, `prompt_hash`).
  - Enforced authorization-first checking, strict tenant isolation, and privacy-restricted bypass.
- **M5 Resilience & Circuit Breaker**:
  - Implemented `CircuitBreaker` (`CLOSED` $\to$ `OPEN` $\to$ `HALF-OPEN`) to prevent cascading retry storms.
  - Implemented bounded retries (max 2 attempts) for transient errors, with privacy-safe fallback.
- **M6 SLO-Aware Dynamic Load Balancing**:
  - Deployed $\ge 2$ replicas for the local model (`local-r1`, `local-r2`) for within-model scheduling.
  - Implemented dynamic workload-weighted queue tracking: $W_q = \sum(\text{prompt}\textunderscore\text{tokens} + \hat{L}_{\{\text{out}\}})$.
  - Implemented `RoundRobinScheduler`, `LeastLoadedScheduler`, and `SLOAwareScheduler`.
- **M7 Structured Observability & Correlation Traces**:
  - Implemented `RequestTracer` tracking request IDs, attempt-level records, latency, tokens, costs, and SLO attainment, with raw prompt redaction.
- **M8 Independent Workload Generator & Benchmark Harness**:
  - Built `WorkloadGenerator` decoupling ground truth demand from pre-inference estimation.
  - Built `BenchmarkRunner` evaluating all 6 baseline combinations, exporting JSON summaries.

### Week 3: Production Hardening, Single-User Model, Live Provider Execution & Containerization
- **Single-User Architecture & Unhindered Gateway Access (`TC_W3_01`)**:
  - Removed multi-tenant auth layer barriers; configured system for single-user workflow (`user`), where all local models, replicas, cloud API keys, and caches belong to the single user. Requests execute directly without unnecessary authorization friction.
- **User-Controlled Privacy Toggle (`user_private`) (`TC_W3_02`)**:
  - Added user toggle `user_private: bool` (default: `False`). When toggled ON at the prompt stage, requests route exclusively to local self-hosted models (Ollama/vLLM), strictly filtering external cloud providers. When toggled OFF, external cloud models via LiteLLM are included in candidate selection.
- **Replica Scheduling Skipped for Cloud Models**:
  - Cloud models hosted remotely (via LiteLLM APIs) do not manage local replica pools. The Stage 2 replica scheduler is automatically bypassed for cloud models (`replica_id="cloud-direct"`, `scheduler_name="skipped"`), while local models undergo full multi-replica load balancing.
- **Provenance-Aware Semantic Cache Containment (`TC_W3_03`, `TC_W3_04`)**:
  - Upgraded `ExactCache` to record explicit cache item provenance (`origin_privacy`, `origin_quality`) and incorporate request temperature into composite keys.
  - Enforced unidirectional provenance guards: public cached responses can never fulfill `internal_only` requests, and low-quality cached responses can never fulfill high-quality floor requests.
- **Strict Contract Validation & Quality Floor Enforcement (`TC_W3_05`, `TC_W3_06`)**:
  - Enforced strict Pydantic model validation with `extra="forbid"`, mandatory non-empty prompts, and explicit rejection of unsupported streaming requests.
  - Added universal quality floor filtering (`low=0.0`, `medium=0.70`, `high=0.85`) across Stage 1 eligibility checking.
  - Hardened budget feasibility to evaluate worst-case token spend: $C_{\text{max}} = p \cdot P_{\text{in}} + \text{max\_tokens} \cdot P_{\text{out}} \le B$.
- **Real Local CPU Serving with Ollama (`TC_W3_07`, `TC_W3_08`)**:
  - Integrated `LocalBackend` directly with local Ollama HTTP `/api/chat` supporting `qwen2.5:1.5b` and `deepseek-r1:1.5b`.
  - Captures native token metrics directly from Ollama response telemetry (`prompt_eval_count`, `eval_count`) and labels hardware execution (`hardware: "cpu"`).
  - Implemented strict fail-closed policy (`LocalBackendUnavailableError`) returning HTTP 503 when the local daemon is unreachable in real mode.
- **Real External Cloud Serving via LiteLLM**:
  - Implemented `LiteLLMBackend` and `LiteLLMRoutingAdapter` using authentic API keys from `.env` for Groq (`qwen-2.5-32b`, `llama-3.3-70b-versatile`) and Cohere (`command-r-08-2024`).
- **Canonical Model Configuration Catalog (`config.yaml`)**:
  - Centralized all model metadata, per-token pricing, SLA latency targets, quality scores, and replica allocations in root `config.yaml`.
- **Circuit Breaker State Machine Hardening (`TC_W3_09`)**:
  - Enhanced `CircuitBreaker` with single-probe half-open throttling and `reset()` / `clear()` lifecycle management.
- **Sanitized Multi-Attempt Observability Ledgers (`TC_W3_10`)**:
  - Regex-based scrubbing of API keys and credentials from all tracer ledgers, attempt records, and error messages.
- **Asynchronous Poisson Process Event Simulation Benchmark (`TC_W3_11`, `TC_W3_12`)**:
  - Built `event_runner.py` implementing an asynchronous discrete-event clock simulator under Poisson request arrival rates ($\lambda = 2.0$ req/s).
  - Evaluates true First-Token Latency (TTFT), Time-Per-Output-Token (TPOT), p95 queue waiting delay, costs, and quality across all 6 baseline pairings.
- **Full Docker Containerization & Compose Support**:
  - Completely containerized all services (including Ollama, AdaRoute Gateway, and Workbench UI) with reproducible `just` recipes.

---

## 2. Milestone Progress Mapping (M1 – M8)

| Milestone | Description | Status | Current Deliverables & Invariants |
| :--- | :--- | :--- | :--- |
| **M1** | **Stable AI Gateway Contract** | **Verified** | Standard `/v1/chat/completions` accepting capability aliases; Single-user context, `user_private` toggle, and strict schema validation (`extra="forbid"`) (`TC_M1_01`, `TC_M1_02`, `TC_W3_01`, `TC_W3_02`, `TC_W3_05`). |
| **M2** | **Multi-Provider Execution** | **Verified** | Live Ollama CPU local model (`qwen2.5:1.5b`, `deepseek-r1:1.5b`) with native token telemetry + authentic cloud execution via LiteLLM (Groq, Cohere) under unified interface (`TC_M2_01`–`TC_M2_03`, `TC_W3_07`, `TC_W3_08`). |
| **M3** | **Cost / Quality Routing** | **Verified** | Distinct baselines (`Strongest`, `Cheapest`, `Static`, `Adaptive`) operating over canonical `config.yaml`, evaluating quality floors and worst-case spend bounds $C_{\text{max}} \le B$ (`TC_M3_01`–`TC_M3_03`, `TC_W3_06`). |
| **M4** | **Safe Caching** | **Verified** | Composite key hashing with user isolation, authorization-first checks, and unidirectional provenance tags (`origin_privacy`, `origin_quality`) preventing public/low-grade cache leakage (`TC_M4_01`–`TC_M4_03`, `TC_W3_03`, `TC_W3_04`). |
| **M5** | **Resilience & Quotas** | **Verified** | Circuit breaker state machine with single-probe half-open throttling, reset lifecycle, and privacy-preserving bounded retries (`TC_M5_01`, `TC_M5_02`, `TC_W3_09`). |
| **M6** | **SLO-Aware Load Balancing** | **Verified** | Multiple replicas per local model evaluated via dynamic workload queue $W_q = \sum(\text{prompt}\textunderscore\text{tokens} + \hat{L}_{\{\text{out}\}})$; cloud models bypass replica selection (`TC_M6_01`–`TC_M6_03`). |
| **M7** | **Usage & Observability** | **Verified** | Structured correlation event traces and multi-attempt execution ledgers with regex-sanitized secret scrubbing (`TC_M7_01`, `TC_M7_02`, `TC_W3_10`). |
| **M8** | **Reproducible Benchmark** | **Verified** | Decoupled Poisson arrival event simulator (`event_runner.py`) measuring TTFT, TPOT, p95 queue wait, and attributable cost savings vs frontier models (`TC_M8_01`, `TC_M8_02`, `TC_W3_11`, `TC_W3_12`). |

---

## 3. Formal Test Suite & Test Case Matrix

| Testcase ID | Milestone | Scope | Test Function | Verified Invariant / Property | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TC_P2_01` | **P0 Correctness** | Integration | `test_tc_p2_01_privacy_adversarial_forbidden_choice` | Internal request NEVER routes to external provider even if cheaper/idle. | **PASSED** |
| `TC_P2_02` | **P0 Correctness** | Integration | `test_tc_p2_02_unknown_privacy_fails_closed` | Unknown privacy labels fail closed with HTTP 400. | **PASSED** |
| `TC_P2_03` | **P0 Correctness** | Unit | `test_tc_p2_03_unavailable_local_fails_closed_no_leak` | Offline local replica triggers safe rejection; never leaks to external. | **PASSED** |
| `TC_P2_04` | **P0 Correctness** | Unit | `test_tc_p2_04_confidential_and_restricted_enforcement` | Confidential and restricted labels strictly filtered from external models. | **PASSED** |
| `TC_P2_05` | **P0 Correctness** | Integration | `test_tc_p2_05_impossible_budget_safely_rejected` | Impossible budget ceiling is safely rejected before inference. | **PASSED** |
| `TC_P2_06` | **P0 Correctness** | Integration | `test_tc_p2_06_negative_budget_validation_error` | Negative budget amounts fail validation with HTTP 400. | **PASSED** |
| `TC_P2_07` | **P0 Correctness** | Unit | `test_tc_p2_07_budget_differentiates_model_tier` | Intermediate budget permits cheap local model while filtering expensive model. | **PASSED** |
| `TC_M1_01` | **M1 Contract** | Integration | `test_tc_m1_01_logical_capability_alias_routing` | Unified API accepts logical capability aliases (`coding`, `general`). | **PASSED** |
| `TC_M1_02` | **M1 Contract** | Integration | `test_tc_m1_02_standard_chat_messages_contract` | Standard OpenAI messages schema accepted and executed. | **PASSED** |
| `TC_M2_01` | **M2 Multi-Provider**| Integration | `test_tc_m2_01_local_model_execution` | Local model backend executes successfully. | **PASSED** |
| `TC_M2_02` | **M2 Multi-Provider**| Integration | `test_tc_m2_02_alternate_provider_execution` | Alternate/mock cloud provider executes successfully when eligible. | **PASSED** |
| `TC_M2_03` | **M2 Multi-Provider**| Integration | `test_tc_m2_03_transparent_provider_switch_same_client` | Same client payload switches transparently based on policy constraints. | **PASSED** |
| `TC_M3_01` | **M3 Baselines** | Unit | `test_tc_m3_01_strongest_distinct_from_cheapest` | Strongest and Cheapest policies pick distinct models matching their objectives. | **PASSED** |
| `TC_M3_02` | **M3 Baselines** | Unit | `test_tc_m3_02_static_rule_follows_task_mapping` | Static policy follows predetermined task-to-model mapping. | **PASSED** |
| `TC_M3_03` | **M3 Baselines** | Unit | `test_tc_m3_03_adaptive_policy_balances_objectives` | Adaptive policy balances cost and quality weights in multi-objective scoring. | **PASSED** |
| `TC_M4_01` | **M4 Safe Cache** | Integration | `test_tc_m4_01_repeat_request_cache_hit` | Repeated identical request hits cache and returns cached payload. | **PASSED** |
| `TC_M4_02` | **M4 Safe Cache** | Integration | `test_tc_m4_02_cross_tenant_cache_isolation` | Tenant isolation invariant: Partitioned users cannot retrieve each other's cached responses. | **PASSED** |
| `TC_M4_03` | **M4 Safe Cache** | Integration | `test_tc_m4_03_no_cache_bypass` | Requests with `no_cache=True` bypass cache lookup and write. | **PASSED** |
| `TC_M4_04` | **M4 Safe Cache** | Integration | `test_tc_m4_04_canonical_normalization_cache_hit` | Canonical prompt whitespace and line ending normalization produces identical cache hit. | **PASSED** |
| `TC_M4_05` | **M4 Safe Cache** | Integration | `test_tc_m4_05_multi_turn_repeat_query_cache_hit` | Multi-turn chat repeated query hits cache via secondary canonical query index. | **PASSED** |
| `TC_M4_06` | **M4 Model Specs** | Unit | `test_tc_m4_06_gemini_model_registered_and_spec_verified` | Verified `gemini/gemini-flash-latest` pricing ($0.075/1M in, $0.30/1M out) and 1M context. | **PASSED** |
| `TC_M4_07` | **M4 Safe Cache** | Integration | `test_tc_m4_07_user_private_cache_partition_and_isolation` | `user_private` cache isolation: Public cloud cache entries NEVER hit when private. | **PASSED** |
| `TC_M5_01` | **M5 Resilience** | Integration | `test_tc_m5_01_bounded_retry_on_replica_failure` | Primary replica transient error triggers bounded retry on alternate replica. | **PASSED** |
| `TC_M5_02` | **M5 Resilience** | Unit | `test_tc_m5_02_circuit_breaker_state_transitions` | Circuit breaker transitions CLOSED $\to$ OPEN $\to$ HALF-OPEN $\to$ CLOSED. | **PASSED** |
| `TC_M6_01` | **M6 Scheduler** | Unit | `test_tc_m6_01_multiple_replicas_exist_for_same_model` | At least two replicas exist for local model enabling within-model scheduling. | **PASSED** |
| `TC_M6_02` | **M6 Scheduler** | Unit | `test_tc_m6_02_dynamic_workload_weighted_queue_tracking`| Dynamic workload-weighted queue $W_q$ increments on dispatch and decrements on completion. | **PASSED** |
| `TC_M6_03` | **M6 Scheduler** | Unit | `test_tc_m6_03_least_loaded_and_slo_aware_selection` | Scheduler selects least-loaded replica when one is artificially saturated. | **PASSED** |
| `TC_M7_01` | **M7 Telemetry** | Integration | `test_tc_m7_01_complete_structured_trace_record` | Every request produces a full correlation event trace record. | **PASSED** |
| `TC_M7_02` | **M7 Telemetry** | Integration | `test_tc_m7_02_secret_prompt_redaction_in_traces` | Raw prompts and proprietary secrets are excluded from tracer records. | **PASSED** |
| `TC_M8_01` | **M8 Benchmark** | Unit | `test_tc_m8_01_independent_workload_and_estimator_error_tracking` | Decoupled ground truth workload generation enables non-circular estimation error tracking. | **PASSED** |
| `TC_M8_02` | **M8 Benchmark** | Unit | `test_tc_m8_02_reproducible_benchmark_execution` | Fixed seed produces identical benchmark metrics across independent runs. | **PASSED** |
| `TC_W3_01` | **M1 Contract** | Integration | `test_tc_w3_01_single_user_context_no_auth_wall` | Single-user requests execute cleanly without authentication barriers. | **PASSED** |
| `TC_W3_02` | **M1 / Policy** | Integration | `test_tc_w3_02_user_private_toggle_enforcement` | `user_private=True` strictly restricts execution to local models, while cloud models skip replica selection when eligible. | **PASSED** |
| `TC_W3_03` | **M4 Safe Cache** | Integration | `test_tc_w3_03_cache_provenance_privacy_leak_prevention` | Public cache entries cannot fulfill `internal_only` requests. | **PASSED** |
| `TC_W3_04` | **M4 Safe Cache** | Integration | `test_tc_w3_04_cache_provenance_quality_floor_enforcement` | Cache entries generated by lower-quality models cannot fulfill higher quality floor requests. | **PASSED** |
| `TC_W3_05` | **M1 Contract** | Integration | `test_tc_w3_05_strict_contract_validation` | Extra unknown parameters, empty prompt bodies, and unsupported stream requests fail validation with HTTP 400. | **PASSED** |
| `TC_W3_06` | **M3 Budget** | Integration | `test_tc_w3_06_bounded_worst_case_spend_clamping` | Worst-case spend $C_{\text{max}} = p \cdot P_{\text{in}} + \text{max\_tokens} \cdot P_{\text{out}}$ exceeding budget is rejected with HTTP 400. | **PASSED** |
| `TC_W3_07` | **M2 Multi-Provider**| Integration | `test_tc_w3_07_real_local_mode_fails_closed_when_offline` | In real execution mode, offline Ollama daemon fails closed with HTTP 503 instead of falling back. | **PASSED** |
| `TC_W3_08` | **M2 Multi-Provider**| Integration | `test_tc_w3_08_real_local_model_native_tokens_and_hardware_tags` | Real Ollama execution returns native token counts and labels `hardware: "cpu"`. | **PASSED** |
| `TC_W3_09` | **M5 Resilience** | Unit | `test_tc_w3_09_circuit_breaker_check_on_all_alternates` | Circuit breaker half-open state throttles concurrent probes and transitions accurately. | **PASSED** |
| `TC_W3_10` | **M7 Observability** | Integration | `test_tc_w3_10_multi_attempt_ledger_accounting` | Multi-attempt ledger records every execution attempt and scrubs all API keys and secrets. | **PASSED** |
| `TC_W3_11` | **M8 Benchmark** | Unit | `test_tc_w3_11_async_event_benchmark_poisson_arrival` | Poisson event clock benchmark computes accurate TTFT, TPOT, and queue delays under load. | **PASSED** |
| `TC_W3_12` | **M8 Benchmark** | Unit | `test_tc_w3_12_attributable_savings_accounting_vs_frontier` | Verifies measurable cost reduction against theoretical frontier models. | **PASSED** |

---

## 4. How to Use

AdaRoute can be used via three interfaces depending on your workflow:
1. **Interactive Terminal CLI (`ui/cli`)** — Fast, keyboard-driven workbench.
2. **AdaRoute Workbench Web UI (`ui/`)** — Visual browser-based interface with live telemetry drawer.
3. **Programmatic OpenAI-Compatible API (`/v1/chat/completions`)** — Direct integration via `curl` or HTTP clients.

---

### Step 1: Launch Services with Docker

Start the complete stack (AdaRoute Gateway, Ollama service, and Workbench UI):

```bash
# Launch Gateway, UI, and Ollama in the background
just up

# Check status of running containers
docker compose ps
```

Verify that the Gateway is healthy:
```bash
curl -s http://localhost:8000/health | jq .
```
Expected output:
```json
{
  "status": "healthy",
  "models": ["qwen2.5:1.5b", "deepseek-r1:1.5b", "groq/qwen/qwen3.8-27b", "cohere/command-r-08-2024", "gpt-4o-external", "llama-3-8b"],
  "replicas": ["local-r1", "local-r2", "local-deepseek-r1", "replica-llama-1"]
}
```

---

### Step 2: Using the Interactive Terminal CLI (`ui/cli`)

Launch the terminal client:
```bash
just cli
```

#### Available Commands:
- Type your prompt and press `Enter` to send.
- `/private` or `/user_private` — Toggle **User Private** mode on/off. When ON, all traffic stays strictly on local models.
- `/set cap <general|coding|reasoning>` — Switch logical capability alias.
- `/set privacy <public|internal_only|confidential|restricted>` — Set privacy level.
- `/set budget <float>` — Set max request budget ceiling (e.g., `/set budget 0.05`).
- `/set quality <low|medium|high>` — Set quality floor requirement.
- `/set policy <adaptive|strongest|cheapest|static>` — Override Stage 1 policy.
- `/set scheduler <slo_aware|least_loaded|round_robin>` — Override Stage 2 scheduler.
- `/diag` — Display deep routing diagnostics for the last completion.
- `/clear` — Reset conversation history.
- `/exit` — Exit the CLI.

#### Example CLI Session:

```text
╭───────────────────────────────── AdaRoute Workbench ─────────────────────────────────╮
│ ⚡ AdaRoute AI Gateway (http://localhost:8000)       ● Online                        │
│ Cap: general  Privacy: public  User Private: OFF  Budget: $1.0000  Policy: adaptive  │
╰───────────────────────────────────────────────────────────────────────────────────────╯

user > What are the principles of zero-trust network architecture?

[● Routing through AdaRoute Gateway...]

╭──────────────────────────────── AdaRoute Assistant ────────────────────────────────╮
│ Zero-trust network architecture is founded on the principle of "never trust, always │
│ verify." Its key tenets include continuous identity verification, least privilege, │
│ explicit microsegmentation, and assuming breach.                                    │
╰─────────────────────────────────────────────────────────────────────────────────────╯
[CACHE MISS]  [Model: groq/qwen/qwen3.8-27b]  [Replica: cloud-direct]  [Provider: groq]  [184ms]  [Tokens: 142]  [Cost: $0.00008]  ✅ SLO Met

user > /private
User Private mode is now ENABLED (local only)

user > Write a quick Python function to reverse a linked list.

[● Routing through AdaRoute Gateway...]

╭──────────────────────────────── AdaRoute Assistant ────────────────────────────────╮
│ def reverse_linked_list(head):                                                      │
│     prev, curr = None, head                                                         │
│     while curr:                                                                     │
│         nxt = curr.next                                                             │
│         curr.next = prev                                                            │
│         prev = curr                                                                 │
│         curr = nxt                                                                  │
│     return prev                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────╯
[CACHE MISS]  [Model: qwen2.5:1.5b]  [Replica: local-r1]  [Provider: local]  [612ms]  [Tokens: 98]  [Cost: $0.00001]  ✅ SLO Met

user > /diag
╭─────────────────────────────────── Last Routing Telemetry ───────────────────────────────────╮
│ Request ID:        59a8039c-eb2f-48d1-be61-2670d8a6797a                                      │
│ Logical Alias:     general                                                                   │
│ Selected Model:    qwen2.5:1.5b                                                              │
│ Selected Replica:  local-r1                                                                  │
│ Provider:          local (CPU Ollama)                                                        │
│ Latency:           612.45 ms                                                                 │
│ Total Tokens:      98 (Prompt: 24, Output: 74)                                               │
│ Total Billed Cost: $0.000012                                                                 │
│ Cache State:       MISS                                                                      │
│ SLO Attained:      True                                                                      │
│ Policy Applied:    adaptive                                                                  │
│ Scheduler:         slo_aware                                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
```

---

### Step 3: Using the AdaRoute Workbench Web UI

1. Open your browser at [http://localhost:8080](http://localhost:8080).
2. **Select Logical Capability**: Choose `General`, `Coding`, or `Reasoning` from the sidebar.
3. **Toggle User Private**:
   - Click the `🛡️ User Private: OFF` badge in the composer footer or toggle **User Private** in the sidebar.
   - When active, the badge displays `🛡️ User Private: ON (Local)`. All requests are guaranteed to route exclusively to local open models (`qwen2.5:1.5b` or `deepseek-r1:1.5b`), completely excluding external cloud APIs.
4. **Tune Constraints**:
   - Set Budget Ceilings ($).
   - Set Quality Floors (`Low`, `Medium`, `High`).
   - Adjust Target Latency SLO (seconds).
5. **Inspect Live Diagnostics**:
   - Click **Diagnostics** in the top-right header or click the mini-telemetry pill on any assistant message bubble to open the slide-over telemetry drawer.

---

### Step 4: Programmatic API Execution via HTTP / `curl`

AdaRoute exposes an OpenAI-compatible `/v1/chat/completions` endpoint. Because AdaRoute is designed for single-user deployments, **no authorization headers or API keys are required**.

#### Example 1: Standard Public Chat Request (Cloud & Local Eligible)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "general",
    "prompt": "Explain the difference between TCP and UDP in two sentences.",
    "privacy_level": "public",
    "budget": 1.0
  }' | jq .
```

#### Example 2: Local-Only Secure Execution via `user_private`

When `user_private: true` is passed, the request will strictly execute on local self-hosted models, even if public and with unlimited budget:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coding",
    "prompt": "Write a thread-safe singleton in Python.",
    "user_private": true,
    "quality_requirement": "medium"
  }' | jq .
```

#### Example 3: Budget-Constrained & Quality-Floor Gated Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "reasoning",
    "prompt": "Solve this riddle: The more you take, the more you leave behind.",
    "budget": 0.005,
    "quality_requirement": "high",
    "latency_slo": 3.0
  }' | jq .
```

#### Example Response Body:

```json
{
  "id": "c1f7b82e-9d2a-4318-8f12-0943ebc102a9",
  "model": "deepseek-r1:1.5b",
  "logical_model": "reasoning",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Footsteps."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 4,
    "total_tokens": 22
  },
  "routing_metadata": {
    "request_id": "c1f7b82e-9d2a-4318-8f12-0943ebc102a9",
    "logical_model": "reasoning",
    "selected_model": "deepseek-r1:1.5b",
    "selected_replica": "local-deepseek-r1",
    "provider": "local",
    "latency_ms": 482.12,
    "attempts": 1,
    "cache_hit": false,
    "estimated_cost": 0.000022,
    "actual_cost": 0.000022,
    "slo_attained": true,
    "policy_name": "adaptive",
    "scheduler_name": "slo_aware",
    "attempt_history": [
      {
        "attempt_number": 1,
        "replica_id": "local-deepseek-r1",
        "model_id": "deepseek-r1:1.5b",
        "provider": "local",
        "prompt_tokens": 18,
        "completion_tokens": 4,
        "latency_ms": 482.12,
        "cost": 0.000022,
        "status": "success"
      }
    ]
  },
  "content": "Footsteps.",
  "replica_id": "local-deepseek-r1",
  "provider": "local"
}
```

---

## 5. Development, Verification & Benchmarking Guide

All development, testing, and benchmark execution is managed via `just`:

```bash
# Build the reproducible container image
just build

# Execute full test suite inside Docker (48 tests across all milestones)
just test

# Execute full test suite locally (uses local Python/Conda environment)
just test-local

# Execute the asynchronous Poisson event benchmark runner inside Docker
just benchmark-event

# Run linting inside Docker
just lint

# Run UI component and client tests
just ui-test

# Start Gateway and Workbench UI together via Docker Compose
just up

# Tear down Docker Compose services
just down
```

### Configuration & Environment Setup
1. Copy the example environment template:
   ```bash
   cp .env.example .env
   ```
2. Configure credentials in `.env`:
   - `GROQ_API_KEY`: Groq API key for cloud models (`qwen-2.5-32b`, `llama-3.3-70b-versatile`).
   - `COHERE_API_KEY`: Cohere API key for cloud models (`command-r-08-2024`).
   - `EXECUTION_MODE`: `real`, `simulation`, or `hybrid` (default: `hybrid`).
   - `OLLAMA_BASE_URL`: Local Ollama HTTP endpoint (default: `http://localhost:11434`).
3. Model catalog and routing metadata are centrally defined in `config.yaml`.
