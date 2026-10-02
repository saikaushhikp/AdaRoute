# Benchmark & Workload Generation Design (Week 2 Specification)

## 1. Principles of Independent Workload Generation

To eliminate circular evaluation where mock generation lengths were sampled around the router's own estimate (identified as a major flaw in W1 assessment), Week 2 introduces an **Independent Workload Generator** (`adaroute.workloads.generator.WorkloadGenerator`):

```text
                       WORKLOAD GENERATOR
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
      Pre-Routing Features            Hidden Ground Truth
   (prompt, task_type, privacy,      (actual_output_tokens,
      budget, SLO target)            arrival_offset_ms)
               │                               │
               ▼                               │
        AdaRoute Router                        │
   (Pre-inference Estimator)                   │
               │                               │
               ▼                               ▼
       Estimated Load L̂_out           Real Execution Demand
               │                               │
               └───────────────┬───────────────┘
                               ▼
               Estimator Error: e = |L̂_out - L_actual|
```

---

## 2. Workload Distribution Profiles

| Task Category | Prompt Length Range | Actual Output Distribution $(\mu, \sigma)$ | Pre-Inference Estimator Prior | Primary Quality Evaluator |
| :--- | :--- | :--- | :--- | :--- |
| **QA / Factual** | 15 – 60 tokens | $(45, 15)$ tokens | 50 tokens | Exact Match / Substring Correctness |
| **Coding** | 40 – 180 tokens | $(310, 60)$ tokens | 280 tokens | Unit Test Execution Pass Rate |
| **Summarization**| 120 – 600 tokens | $(175, 40)$ tokens | 160 tokens | Factual Consistency Rubric |
| **Reasoning** | 30 – 120 tokens | $(230, 50)$ tokens | 220 tokens | Mathematical Final Answer Accuracy |
| **Creative** | 25 – 80 tokens | $(270, 70)$ tokens | 250 tokens | Fluency & Task Completion |

---

## 3. Evaluated Policy Combinations

The benchmark harness evaluates 6 distinct combinations to isolate the contribution of Stage 1 (Model Selection) from Stage 2 (Replica Scheduling):

| Combination ID | Stage 1: Model Selection Policy | Stage 2: Replica Scheduling Policy | Purpose / Research Question Addressed |
| :--- | :--- | :--- | :--- |
| **Comb-1** | `StrongestModelPolicy` | `RoundRobinScheduler` | Upper-bound quality reference; what is the cost of always using top-tier models? |
| **Comb-2** | `CheapestModelPolicy` | `RoundRobinScheduler` | Lower-bound cost reference; what quality degradation occurs under purely greedy cost routing? |
| **Comb-3** | `StaticRulePolicy` | `RoundRobinScheduler` | Industry standard baseline; how well do fixed task-to-model rules perform? |
| **Comb-4** | `AdaptiveModelPolicy` | `RoundRobinScheduler` | Isolates Stage 1 improvement when coupled with naive load balancing. |
| **Comb-5** | `AdaptiveModelPolicy` | `LeastLoadedScheduler` | Traditional gateway load balancing coupled with multi-objective model selection. |
| **Comb-6** | `AdaptiveModelPolicy` | `SLOAwareScheduler` | **Proposed AdaRoute System**: Joint adaptive model selection and workload-weighted dynamic scheduling ($W_q$). |

---

## 4. Evaluated Metrics Catalog

1. **Latency**:
   - Time to First Token (TTFT): Initial token emission latency.
   - Mean, P95, and P99 End-to-End Latency.
2. **Economic**:
   - Total Cost ($): Sum of input and realized output token expenses.
   - Cost per Successful Request ($/req).
3. **Quality & Service Reliability**:
   - Mean Attained Quality Score (0.0 to 1.0).
   - SLO Attainment Rate (% of requests meeting latency targets).
   - Rejection Rate (% of requests rejected due to budget or privacy constraints).
4. **Estimation Accuracy**:
   - Mean Absolute Error (MAE): $\frac{1}{N}\sum |\hat{L}_{out} - L_{actual}|$.
   - Mean Signed Error: $\frac{1}{N}\sum (\hat{L}_{out} - L_{actual})$.
