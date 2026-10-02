# AdaRoute System Architecture (Week 2 Specification)

## 1. High-Level Architecture Overview

AdaRoute decouples serving decisions into two evaluation stages: **Model & Provider Selection (Stage 1)** and **Replica Scheduling (Stage 2)**.

```text
                           CLIENT APPLICATION
                                  │
                                  ▼
                 POST /v1/chat/completions (OpenAI Compatible)
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │     AdaRoute Gateway      │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    Pre-Routing Characterizer
          (Tokenization, Pre-Inference Estimation L̂_out, Schema Validation)
                                  │
                                  ▼
                    Safe Exact Cache (M4)
          (Authorization-First, Tenant Isolation, Hash Key Check)
                                  │ (Miss)
                                  ▼
               STAGE 1: MODEL & PROVIDER SELECTION (M3)
             ┌─────────────────────────────────────────┐
             │ Hard Constraints Eligibility Filtering  │
             │   - Privacy (Fail-closed enum)          │
             │   - Budget Ceiling: C_est <= B          │
             │   - Quality Floor: Q(model) >= Q_min    │
             │   - Health: At least 1 replica alive    │
             │                                         │
             │ Policy Optimization:                    │
             │   - Strongest / Cheapest / Static /     │
             │   - Adaptive Multi-Objective J(m)       │
             └────────────────────┬────────────────────┘
                                  │ Chosen Model Class (e.g., qwen-2.5-local)
                                  ▼
                 STAGE 2: REPLICA SCHEDULER (M6)
             ┌─────────────────────────────────────────┐
             │ Dynamic Telemetry Evaluation:           │
             │   - Workload-Weighted Queue W_q         │
             │   - Recent Moving-Average Latency       │
             │   - Memory Pressure & Staleness         │
             │                                         │
             │ Scheduling Policy:                      │
             │   - Round-Robin / Least-Loaded /        │
             │   - SLO-Aware (Latency Violation Risk)  │
             └────────────────────┬────────────────────┘
                                  │ Chosen Replica (e.g., local-r2)
                                  ▼
               RESILIENCE & EXECUTION CONTROLLER (M5)
             ┌─────────────────────────────────────────┐
             │   - Circuit Breaker State Check         │
             │   - Dynamic State Dispatch (W_q ↑)      │
             │   - Backend Execution (Local / Mock)    │
             │   - Bounded Retry (Max 2 Attempts)      │
             │   - Dynamic State Completion (W_q ↓)    │
             └────────────────────┬────────────────────┘
                                  │ Response Payload
                                  ▼
               OBSERVABILITY & TRACING PIPELINE (M7)
             ┌─────────────────────────────────────────┐
             │   - Secret-Redacted Event Trace Record  │
             │   - Attempt-Level Correlation ID        │
             │   - Realized Cost & Token Accounting    │
             │   - Continuous Estimator Calibration    │
             └────────────────────┬────────────────────┘
                                  │
                                  ▼
                           CLIENT RESPONSE
```

---

## 2. Formal Mathematical Formulation

### A. Stage 1: Model Selection
For incoming request $r = (\text{prompt}, \text{task}, \text{privacy}, B, Q_{\min}, L_{slo})$ and candidates $M$:

$$\text{Eligible}(M) = \{ m \in M \mid \text{privacy} \in \text{Allowed}(m) \land C_{\text{est}}(m) \le B \land Q(m, \text{task}) \ge Q_{\min} \land \text{Alive}(m) \}$$

Over eligible candidates, the **Adaptive Policy** evaluates:

$$J(m) = w_c \cdot \frac{C_{\text{est}}(m)}{\max C} + w_q \cdot \frac{Q_{\max} - Q(m, \text{task})}{Q_{\max}} + w_l \cdot \frac{L(m)}{\max L} + w_s \cdot \mathbb{I}[L(m) > L_{slo}] + w_f \cdot \text{FailRate}(m)$$

### B. Stage 2: Replica Scheduling
Among replicas $R(m)$ for selected model $m$:

$$W_q(r) = \sum_{i \in \text{Queue}(r)} (\text{prompt\_tokens}_i + \hat{L}_{out, i})$$

$$\text{Score}(r) = \alpha \cdot \frac{W_q(r)}{1000} + \beta \cdot \frac{\text{Latency}_{\text{recent}}(r)}{50} + \gamma \cdot \text{MemoryPressure}(r) + \delta \cdot \text{ActiveReq}(r)$$

The replica with minimal $\text{Score}(r)$ is selected.

---

## 3. Safe Caching Invariants (M4)

Cache identity is constructed via a collision-resistant SHA-256 digest:

$$K = \text{SHA-256}(\text{tenant\_id} \parallel \text{user\_id} \parallel \text{logical\_model} \parallel \text{model\_version} \parallel \text{policy\_version} \parallel \text{SHA-256}(\text{prompt}))$$

- **Invariants**:
  1. Requests with `privacy_level == RESTRICTED` or `no_cache == True` strictly bypass cache.
  2. Authorization and tenant isolation are verified **before** cache lookup.
  3. Tenant A can never retrieve cached responses generated by Tenant B.
