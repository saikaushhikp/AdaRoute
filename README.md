# AdaRoute: Multi-Objective Adaptive LLM Gateway

**AdaRoute** is a provider-independent AI gateway designed with cost-quality routing, safe caching, and scalable model serving. It addresses the challenges of serving heterogeneous LLM workloads across local open models and external cloud providers by decoupling serving decisions into two evaluation stages:
1. **Stage 1 — Model & Provider Selection:** Evaluates incoming request requirements (task classification, quality targets, cost budget, privacy policies) to identify the appropriate model class and provider.
2. **Stage 2 — Replica Scheduling:** Evaluates real-time backend state (workload-weighted queue depth $W_q$, memory pressure, moving-average latency, health) to route requests to the optimal replica instance.

---

## 1. Incremental Progress: Week 1 & Week 2

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
  - Implemented dynamic workload-weighted queue tracking: $W_q = \sum (\text{prompt\_tokens} + \hat{L}_{\text{out}})$.
  - Implemented `RoundRobinScheduler`, `LeastLoadedScheduler`, and `SLOAwareScheduler`.
- **M7 Structured Observability & Correlation Traces**:
  - Implemented `RequestTracer` tracking request IDs, attempt-level records, latency, tokens, costs, and SLO attainment, with raw prompt redaction.
- **M8 Independent Workload Generator & Benchmark Harness**:
  - Built `WorkloadGenerator` decoupling ground truth demand from pre-inference estimation.
  - Built `BenchmarkRunner` evaluating all 6 baseline combinations, exporting JSON summaries.

---

## 2. Milestone Progress Mapping (M1 – M8)

| Milestone | Description | Status | Week 2 Deliverables & Verification |
| :--- | :--- | :--- | :--- |
| **M1** | **Stable AI Gateway Contract** | **Verified** | `/v1/chat/completions` accepts logical capability aliases and standard messages without SDK lock-in (`TC_M1_01`, `TC_M1_02`). |
| **M2** | **Multi-Provider Execution** | **Verified** | Real local model (`LocalBackend`) + simulated cloud provider (`MockBackend`) switch seamlessly under the same contract (`TC_M2_01`–`TC_M2_03`). |
| **M3** | **Cost / Quality Routing** | **Verified** | Distinct baselines (`Strongest`, `Cheapest`, `Static`, `Adaptive`) operating over central `ModelRegistry` (`TC_M3_01`–`TC_M3_03`). |
| **M4** | **Safe Caching** | **Verified** | Composite key hashing with tenant isolation, authorization-first checking, and privacy bypass (`TC_M4_01`–`TC_M4_03`). |
| **M5** | **Resilience & Quotas** | **Verified** | Circuit breaker state machine and bounded retries with privacy-preserving fallback (`TC_M5_01`, `TC_M5_02`). |
| **M6** | **SLO-Aware Load Balancing** | **Verified** | $\ge 2$ replicas evaluated with dynamic workload queue $W_q$ and SLO-aware scheduling (`TC_M6_01`–`TC_M6_03`). |
| **M7** | **Usage & Observability** | **Verified** | Correlation event traces tracking attempt latency, tokens, costs, and redacted secrets (`TC_M7_01`, `TC_M7_02`). |
| **M8** | **Reproducible Benchmark** | **Verified** | Independent workload generator, estimator MAE tracking, and benchmark runner producing machine-readable artifacts (`TC_M8_01`, `TC_M8_02`). |

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
| `TC_M4_02` | **M4 Safe Cache** | Integration | `test_tc_m4_02_cross_tenant_cache_isolation` | Tenant isolation invariant: Tenant B cannot retrieve Tenant A's cached responses. | **PASSED** |
| `TC_M4_03` | **M4 Safe Cache** | Integration | `test_tc_m4_03_no_cache_bypass` | Requests with `no_cache=True` bypass cache lookup and write. | **PASSED** |
| `TC_M5_01` | **M5 Resilience** | Integration | `test_tc_m5_01_bounded_retry_on_replica_failure` | Primary replica transient error triggers bounded retry on alternate replica. | **PASSED** |
| `TC_M5_02` | **M5 Resilience** | Unit | `test_tc_m5_02_circuit_breaker_state_transitions` | Circuit breaker transitions CLOSED $\to$ OPEN $\to$ HALF-OPEN $\to$ CLOSED. | **PASSED** |
| `TC_M6_01` | **M6 Scheduler** | Unit | `test_tc_m6_01_multiple_replicas_exist_for_same_model` | At least two replicas exist for local model enabling within-model scheduling. | **PASSED** |
| `TC_M6_02` | **M6 Scheduler** | Unit | `test_tc_m6_02_dynamic_workload_weighted_queue_tracking`| Dynamic workload-weighted queue $W_q$ increments on dispatch and decrements on completion. | **PASSED** |
| `TC_M6_03` | **M6 Scheduler** | Unit | `test_tc_m6_03_least_loaded_and_slo_aware_selection` | Scheduler selects least-loaded replica when one is artificially saturated. | **PASSED** |
| `TC_M7_01` | **M7 Telemetry** | Integration | `test_tc_m7_01_complete_structured_trace_record` | Every request produces a full correlation event trace record. | **PASSED** |
| `TC_M7_02` | **M7 Telemetry** | Integration | `test_tc_m7_02_secret_prompt_redaction_in_traces` | Raw prompts and proprietary secrets are excluded from tracer records. | **PASSED** |
| `TC_M8_01` | **M8 Benchmark** | Unit | `test_tc_m8_01_independent_workload_and_estimator_error_tracking` | Decoupled ground truth workload generation enables non-circular estimation error tracking. | **PASSED** |
| `TC_M8_02` | **M8 Benchmark** | Unit | `test_tc_m8_02_reproducible_benchmark_execution` | Fixed seed produces identical benchmark metrics across independent runs. | **PASSED** |

---

## 4. Development & Verification Guide

All development, testing, and benchmark execution is managed via `just`:

```bash
# Build the reproducible container image
just build

# Execute full test suite (Unit & Integration tests across all milestones)
just test

# Execute the reproducible benchmark runner (outputs results to results/)
just benchmark

# Check code linting
just lint

# Auto-format codebase
just format
```
