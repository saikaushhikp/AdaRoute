# Week-2 Comprehensive Deliverable Report: AdaRoute Policy Engine & Gateway Slice

## 1. Executive Summary

During Week 2, AdaRoute successfully transitioned from the preliminary specification and mock setup of Week 1 into a **verified, robust vertical slice**. In direct response to the Week 1 assessment feedback, all identified defects were remediated, genuine baseline algorithms were implemented, real local model execution was integrated alongside alternate/mock providers, and a reproducible benchmark harness was established.

---

## 2. P0 Correctness & Security Fixes

### A. Privacy Fail-Closed Implementation
- **The Defect in W1**: The prototype checked `request.privacy_level in ["internal", "high"]`, while specifications and tests used `internal_only`, allowing sensitive requests to leak to external mock providers when local queues were saturated.
- **W2 Remediation**:
  - Implemented a canonical `PrivacyLevel` enum (`PUBLIC`, `INTERNAL_ONLY`, `CONFIDENTIAL`, `RESTRICTED`) in [`adaroute/schemas.py`](file:///home/kaushik/AdaRoute/adaroute/schemas.py).
  - Configured explicit backend allow-lists: `gpt-4o-external` is strictly restricted to `{PrivacyLevel.PUBLIC}`.
  - Fail-Closed Behavior: Unknown privacy strings fail validation immediately with HTTP 400.
  - Enforced uniformly across Stage 1 model selection, Stage 2 scheduling, retries, fallbacks, and cache lookups.

### B. Adversarial Testing (`TC_P2_01` to `TC_P2_04`)
- Created test scenarios where the external mock is intentionally configured to be **significantly cheaper** and **idle (0 queue)** while the local model is expensive and busy (10 queue).
- **Verified Invariant**: Requests with `internal_only` *never* select or call the external provider under any queue or cost condition.
- Verified that if all local replicas are offline, the gateway fails closed with HTTP 503 rather than leaking private data to available external providers.

### C. Hard Budget Ceilings (`TC_P2_05` to `TC_P2_07`)
- Replaced the naive `model_price <= budget` check with a pre-inference cost projection:
  $$C_{\text{est}} = \frac{\text{prompt\_tokens}}{1000} \cdot P_{\text{in}} + \frac{\hat{L}_{\text{out}}}{1000} \cdot P_{\text{out}}$$
- If no capable candidate satisfies $C_{\text{est}} \le B$, the request is safely rejected with HTTP 503 before any inference call occurs.

---

## 3. Milestone Incremental Progress (M1 – M8)

### M1: Stable AI Gateway Contract
- Exposes standardized OpenAI-compatible `/v1/chat/completions` endpoint in [`adaroute/api.py`](file:///home/kaushik/AdaRoute/adaroute/api.py).
- Callers invoke logical capability aliases (`model="general"`, `model="coding"`, `model="reasoning"`) without provider-specific SDK dependencies (`TC_M1_01`, `TC_M1_02`).

### M2: Multi-Provider Execution
- Connects a **real self-hosted local model backend** (`LocalBackend`) supporting Ollama / local CPU execution alongside a simulated external cloud provider (`MockBackend`).
- Unchanged client requests switch seamlessly between local and cloud models based on tenant constraints and policy decisions (`TC_M2_01` – `TC_M2_03`).

### M3: Cost / Quality Routing & Distinct Baselines
- Implemented [`ModelRegistry`](file:///home/kaushik/AdaRoute/adaroute/policy/registry.py) tracking capability tiers, quality scores, token prices, and privacy allow-lists.
- Replaced identical stubs with genuinely distinct Stage 1 policies:
  - `StrongestModelPolicy`: Selects highest quality score.
  - `CheapestModelPolicy`: Selects lowest estimated cost.
  - `StaticRulePolicy`: Deterministic task-to-model mapping.
  - `AdaptiveModelPolicy`: Evaluates normalized multi-objective function $J(m)$ (`TC_M3_01` – `TC_M3_03`).

### M4: Safe Exact Caching
- Implemented [`ExactCache`](file:///home/kaushik/AdaRoute/adaroute/cache/exact.py) with composite keys:
  $$K = \text{SHA-256}(\text{tenant\_id} \parallel \text{user\_id} \parallel \text{logical\_model} \parallel \text{model\_version} \parallel \text{policy\_version} \parallel \text{SHA-256}(\text{prompt}))$$
- Enforces authorization-first lookup; strictly guarantees tenant isolation and bypasses for restricted privacy or `no_cache=True` (`TC_M4_01` – `TC_M4_03`).

### M5: Resilience & Circuit Breaker
- Integrated [`CircuitBreaker`](file:///home/kaushik/AdaRoute/adaroute/resilience/circuit.py) state machine (`CLOSED` $\to$ `OPEN` $\to$ `HALF-OPEN`) to prevent retry storms upon consecutive failures.
- Implemented bounded retries (maximum 2 attempts) for transient errors, with privacy-safe fallback (`TC_M5_01`, `TC_M5_02`).

### M6: SLO-Aware Dynamic Load Balancing
- Deployed $\ge 2$ replicas for the local model (`local-r1`, `local-r2`) to evaluate within-model scheduling.
- Implemented dynamic, event-driven workload queue tracking:
  $$W_q = \sum_{i \in \text{queue}} (\text{prompt\_tokens}_i + \hat{L}_{\text{out}, i})$$
  which increments on dispatch and decrements on completion.
- Evaluated `RoundRobinScheduler`, `LeastLoadedScheduler`, and `SLOAwareScheduler` (`TC_M6_01` – `TC_M6_03`).

### M7: Usage & Observability
- Implemented structured [`RequestTracer`](file:///home/kaushik/AdaRoute/adaroute/telemetry/tracer.py) recording complete correlation events: `request_id`, attempt number, selected replica, latency, tokens, estimated vs realized cost, and SLO attainment.
- Automatically redacts raw prompt text from logs to prevent secret leakage (`TC_M7_01`, `TC_M7_02`).

### M8: Reproducible Benchmark Harness
- Built an independent [`WorkloadGenerator`](file:///home/kaushik/AdaRoute/adaroute/workloads/generator.py) decoupling ground-truth output lengths from pre-inference estimates, eliminating circular evaluation.
- Implemented continuous estimator error tracking ($e = |\hat{L}_{\text{out}} - L_{\text{actual}}|$, MAE).
- Built [`BenchmarkRunner`](file:///home/kaushik/AdaRoute/adaroute/experiments/runner.py) evaluating all 6 baseline combinations, exporting machine-readable JSON artifacts (`TC_M8_01`, `TC_M8_02`).

---

## 4. Verification Evidence & Summary

All 26 test cases pass within the containerized Docker environment via `just test`:
- **P0 Correctness Suite**: 7/7 tests passed.
- **Milestone Verification Suite (M1–M8)**: 19/19 tests passed.
- **Linter & Code Quality**: 100% compliant with `ruff check .`.
