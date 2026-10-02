I went through the five supplied project documents as a single progression:

**Problem definition → tentative milestones → deeper project understanding → submitted Week-0 proposal → Week-0 assessment.**

The key point is that **Week 1 is not primarily about implementing the full gateway yet**. The assessment explicitly positions W1 as the stage where the project definition is made technically precise, experimentally defensible, traceable to prior work, and supported by an initial repository/setup. 

# Week 1 — What You Actually Need to Do

## 1. First: Freeze/refine the problem definition

Your current proposal has the right research direction, but one important correction is required.

### Current wording to avoid

You currently frame the objective around finding an **"optimal"** routing decision.

The assessment explicitly says this is too strong unless the objective and constraints are formally defined. 

### W1 action

Replace the vague notion of:

> "optimal routing"

with a **measurable constrained decision problem**.

For example, conceptually:

$$
\text{Choose action }a
$$

subject to hard constraints such as:

$$
Privacy(a)=allowed
$$

$$
Cost(a)\le B
$$

$$
Quality(a)\ge Q_{\min}
$$

$$
P(L>L_{SLO})\le\epsilon
$$

while optimizing a defined objective involving:

$$
C,\quad L,\quad Q_{\text{loss}},\quad SLO\ violations
$$

The important distinction is:

### Hard constraints

* Privacy restrictions
* Disallowed providers/models
* Maximum budget, where applicable
* Required capability
* Possibly strict SLO requirements

### Soft objectives/preferences

* Lower cost
* Lower latency
* Better quality
* Better utilization
* Lower failure risk

This needs to become part of the formal project specification.

---

# 2. Define the Request Characterization Model

This is one of the **most important W1 tasks**.

Your proposal lists request attributes, but does not yet explain **how the system obtains them**. The assessment explicitly identifies this gap. 

You need to define something like:

| Feature                | How obtained                                              |
| ---------------------- | --------------------------------------------------------- |
| Task type              | Rule-based classifier / metadata / lightweight classifier |
| Prompt length          | Model tokenizer                                           |
| Expected output length | Pre-inference estimator                                   |
| Required quality       | User/application policy                                   |
| Latency target         | Application/tenant SLO                                    |
| Privacy                | Data classification / tenant policy                       |
| Budget                 | Request/tenant/workload configuration                     |
| Queue state            | Serving-runtime telemetry                                 |
| Memory pressure        | GPU/runtime telemetry                                     |
| Recent latency         | Runtime metrics                                           |
| Failure state          | Backend health monitor                                    |

This should become an actual **request schema**.

For example, conceptually:

```text
RequestContext
 ├── task_type
 ├── prompt_tokens
 ├── estimated_output_tokens
 ├── quality_requirement
 ├── latency_slo
 ├── privacy_level
 ├── budget
 ├── tenant_id
 └── cache_eligibility
```

And separately:

```text
BackendState
 ├── provider
 ├── model
 ├── replica_id
 ├── queue_depth
 ├── estimated_workload
 ├── GPU_utilization
 ├── memory_pressure
 ├── recent_latency
 ├── failure_rate
 └── availability
```

The project already establishes that LLM requests should not be treated as uniform API calls because prompt/output sizes and service requirements vary significantly. 

---

# 3. Solve the "Expected Output Length" Problem

This is a **specific technical flaw identified by the evaluator**.

Your proposal currently treats expected generation length as if it were known before inference.

It isn't.

The assessment explicitly says:

> Do **not** use the actual output length as an input to the router.

That would leak future information and create artificially optimistic results. 

## W1 research/design task

Define an estimator:

$$
\hat L_{out}
=
f(
task,
prompt,
max\_tokens,
history,
model,
format
)
$$

Possible inputs:

* Task type
* Prompt/token characteristics
* Requested `max_tokens`
* Historical outputs
* Model behavior
* Requested response format

### And importantly:

You must plan to measure:

$$
Error =
|\hat L_{out}-L_{out}|
$$

and investigate how routing performance changes when the estimate is inaccurate.

### W1 deliverable

A short **Output-Length Estimation Specification**:

```text
Input
   ↓
Request characterization
   ↓
Output-length estimator
   ↓
estimated_generation_tokens
   ↓
routing policy
   ↓
actual inference
   ↓
compare estimate vs actual
```

Do **not** build a sophisticated ML estimator yet unless necessary.

A simple baseline estimator is enough for W1.

---

# 4. Make the Two-Stage Routing Architecture Explicit

This is arguably the **central architectural decision of the project**.

The proposal already identifies:

### Stage 1

**Model/provider selection**

> Which model/provider should serve this request?

### Stage 2

**Replica selection**

> Which instance of that model should execute it?

The evaluator wants this separation to become experimentally explicit. 

So W1 should formalize:

```text
                 Request
                    │
                    ▼
          Request Characterization
                    │
                    ▼
        ┌─────────────────────────┐
        │ Stage 1                 │
        │ Model / Provider Router │
        └────────────┬────────────┘
                     │
                     ▼
             Selected Model
                     │
                     ▼
        ┌─────────────────────────┐
        │ Stage 2                 │
        │ Replica Scheduler       │
        └────────────┬────────────┘
                     │
                     ▼
                Replica
```

And then evaluate them **both independently and jointly**.

---

# 5. Define the W1 Experimental Baselines

This is another explicit requirement from the assessment.

You need to lock down the baseline policies **before implementation**.

## Model/provider routing

At minimum:

### B1 — Strongest model

```text
all requests → strongest model
```

### B2 — Cheapest model

```text
all requests → cheapest model
```

### B3 — Static rule-based

```text
task → predetermined model
```

### B4 — Adaptive model routing

```text
request state + backend state
             ↓
       policy decision
```

---

## Replica scheduling

### B5 — Round robin

```text
R1 → R2 → R3 → R1 ...
```

### B6 — Least-loaded

Based on:

```text
queue / connections / load
```

### B7 — Adaptive/SLO-aware

Based on:

```text
queue
prompt workload
estimated output
GPU pressure
recent latency
SLO
```

---

# 6. Build the Combined Experimental Matrix

This is particularly important because otherwise you won't know **where the improvement comes from**.

The assessment explicitly requires comparison of combinations such as static/adaptive model selection with different replica policies. 

Your W1 experiment specification should therefore include:

| Model Policy               | Replica Policy     |
| -------------------------- | ------------------ |
| Static                     | Round-robin        |
| Static                     | Least-loaded       |
| Adaptive                   | Round-robin        |
| Adaptive                   | Least-loaded       |
| Adaptive                   | Adaptive/SLO-aware |
| Oracle/offline upper bound | If feasible        |

The critical scientific question becomes:

> Does the improvement come from **better model selection**, **better replica selection**, or their combination?

That is much stronger than simply demonstrating that "adaptive routing works."

---

# 7. Define the Workload Taxonomy

The assessment explicitly asks for this. 

You should define at least:

1. **Factual QA**
2. **Reasoning**
3. **Coding**
4. **Summarization**
5. **Structured extraction**
6. **Long-context tasks**

Then characterize each workload along dimensions such as:

```text
Task type
Prompt length
Expected output length
Quality requirement
Latency SLO
Privacy requirement
Budget
Arrival rate
```

This directly supports the project's underlying idea that requests have heterogeneous computational and service requirements. 

---

# 8. Define How "Quality" Will Be Measured

This is **not optional**.

Your project optimizes a cost/quality/latency trade-off, so "quality" cannot remain an abstract variable.

The evaluator specifically asks you to define quality metrics per task category. 

Create something like:

| Task                  | Candidate quality metric           |
| --------------------- | ---------------------------------- |
| Factual QA            | Exact match / reference similarity |
| Reasoning             | Task-specific correctness          |
| Coding                | Unit-test pass rate                |
| Summarization         | Reference/semantic evaluation      |
| Structured extraction | Schema validity + field accuracy   |
| Long context          | Task-specific benchmark score      |

The important requirement is:

> **Do not only report one aggregate quality number.**

Quality should be reported **per task category**, because an aggregate can hide poor routing for particular workloads. 

---

# 9. Define Dynamic Backend-State Semantics

Your proposal currently says:

```text
queue
GPU utilization
memory pressure
latency
failure rate
```

But W1 needs to answer:

### How fresh is this information?

Define:

* Metric publication interval
* Maximum tolerated staleness
* Missing telemetry behavior
* Queue measurement definition
* Whether active tokens are considered
* Replica unhealthy criteria
* Recovery criteria
* Oscillation prevention

These are explicitly listed by the assessment. 

### Particularly important:

Do **not** define load merely as:

```text
queue = 10 requests
```

because:

```text
10 × short requests
```

may be substantially less work than:

```text
2 × huge-context requests
```

The project already identifies heterogeneous request demand as a core issue. 

So W1 should investigate a **workload-weighted queue/load representation**.

For example:

$$
W_q =
\sum_{i\in queue}
(\hat p_i+\hat o_i)
$$

or a more refined estimated service-work metric.

You don't have to finalize the perfect formula in W1. But you **must define and justify the candidate representation**.

---

# 10. Define the Policy/Decision Function

This is where your project starts becoming a real systems/research project.

You need to specify what the adaptive policy actually computes.

The broader project framing already proposes:

$$
x_r = \text{request state}
$$

$$
x_b = \text{backend state}
$$

and a decision:

$$
a^* = \arg\min_a J(a|x_r,x_b)
$$

with cost, latency, quality loss, SLO violation and failure-related terms. 

For W1, turn this into a concrete specification.

For example:

$$
J(a)=
w_cC(a)
+w_lL(a)
+w_qQ_{loss}(a)
+w_sSLO(a)
+w_fF(a)
$$

subject to:

$$
Privacy(a)=allowed
$$

$$
Cost(a)\le B
$$

$$
Quality(a)\ge Q_{min}
$$

This does **not** mean implementing a sophisticated optimizer immediately.

The assessment actually supports beginning with an interpretable rule-based controller. 

---

# 11. Define the Architecture / System Design

W1 should produce a formal architecture rather than just the conceptual diagram.

The intended system currently looks like:

```text
                    CLIENT
                       │
                       ▼
              ┌─────────────────┐
              │   AI Gateway    │
              └────────┬────────┘
                       │
                       ▼
             Request Characterizer
                       │
                       ▼
                Policy Engine
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
    Model/Provider              Cache Policy
       Router                      │
          │                         │
          └──────────┬──────────────┘
                     ▼
              Replica Scheduler
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    Replica 1    Replica 2    External/Mock
        │            │            │
        └────────────┼────────────┘
                     ▼
                LLM Serving
                     │
                     ▼
                  Response

        ┌──────────────────────────────┐
        │ Metrics / Logs / Tracing     │
        │ Cost / Latency / SLO / Error │
        └──────────────────────────────┘
```

The deeper project work specifically argues for a **thin gateway/policy layer**, rather than attempting to recreate an entire production gateway or inference engine. 

---

# 12. Decide What Is Actually Being Implemented vs Simulated

This needs to be settled **in W1**, otherwise scope will explode later.

The project constraint is:

> Single-GPU or CPU/simulation fallback must always exist; multi-GPU experiments are optional depending on available resources. 

So explicitly classify components.

### Implement

* Gateway API
* Request schema
* Request characterization
* Output-length estimator
* Policy engine
* Model/provider abstraction
* Replica-selection logic
* Backend-state abstraction
* Baseline policies
* Experiment framework
* Metrics collection
* Evaluation scripts

### Implement if resources permit

* Real vLLM backend
* Multiple real replicas
* GPU telemetry
* Actual multi-GPU scheduling

### Simulate/mock when hardware is unavailable

* External provider
* Multiple replicas
* Failure injection
* Queue state
* Backend latency
* GPU pressure
* Provider outages

This is fully consistent with the project's stated resource constraints.

---

# 13. Research the Existing Systems — But With a Specific Purpose

The project has already done a substantial first-pass literature/system review.

The supplied literature includes:

* vLLM/PagedAttention
* RouteLLM
* SGLang
* Llumnix
* DistServe
* Envoy 

And the deeper analysis established that mature gateway products already cover many basic gateway features.

Therefore **W1 research should not be generic "read papers."**

Instead, create a structured comparison:

| Work/System          | What it solves              | Layer           | What we reuse            | What remains for our project |
| -------------------- | --------------------------- | --------------- | ------------------------ | ---------------------------- |
| vLLM                 | Efficient serving           | Serving         | Serving substrate        | Gateway policy               |
| RouteLLM             | Model routing               | Model selection | Routing concepts         | Backend-state integration    |
| SGLang               | Execution/cache             | Serving         | Cache/execution concepts | Gateway-level policy         |
| Llumnix              | Dynamic scheduling          | Replica serving | Scheduling ideas         | Gateway-level integration    |
| DistServe            | Prefill/decode              | Serving         | SLO insights             | Not implemented from scratch |
| Envoy                | Resilience/gateway          | Infrastructure  | Retry/circuit concepts   | LLM-specific policy          |
| Existing AI gateways | Provider/routing/resilience | Gateway         | Architectural reference  | Cross-layer adaptive policy  |

The goal is to establish:

> **What exactly are we borrowing, what are we implementing, and what are we experimentally investigating?**

---

# 14. Repository Initial Setup

The assessment explicitly lists:

> **Repository's initial setup**

as a W1 requirement. 

So W1 should produce an initial repository structure.

I would recommend something along these lines:

```text
AdaRoute/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
│
├── docs/
│   ├── problem-definition.md
│   ├── system-design.md
│   ├── assumptions.md
│   ├── literature-review.md
│   ├── baselines.md
│   ├── metrics.md
│   └── experiment-plan.md
│
├── gateway/
│   ├── api/
│   ├── schemas/
│   ├── characterization/
│   ├── routing/
│   ├── scheduling/
│   ├── caching/
│   ├── resilience/
│   ├── providers/
│   └── observability/
│
├── backends/
│   ├── mock/
│   └── local/
│
├── workloads/
│   ├── generators/
│   ├── traces/
│   └── configs/
│
├── experiments/
│   ├── baselines/
│   ├── scenarios/
│   ├── runners/
│   └── configs/
│
├── evaluation/
│   ├── metrics/
│   ├── analysis/
│   └── plots/
│
└── tests/
```

**You don't need to implement all these modules in W1.**

The point is to establish the project structure so subsequent milestones have a controlled place to live.

---

# 15. Write Acceptance Tests Before Serious Implementation

This is another important requirement from the assessment.

The final tests need to verify more than average latency. 

W1 should therefore define acceptance properties such as:

### Safety

```text
Privacy-restricted request
        ↓
Disallowed backend
        ↓
NEVER selected
```

### Budget

```text
Budget = B
Cost > B
     ↓
Backend rejected
```

### Reproducibility

```text
same request
+ same state
+ same config
        ↓
same routing decision
```

### Missing telemetry

```text
telemetry unavailable
        ↓
safe fallback policy
```

### Failed replica

```text
replica unhealthy
        ↓
removed from candidate set
```

### Recovery

```text
healthy again
        ↓
controlled re-entry
```

### Failover

```text
primary fails
        ↓
bounded retry
        ↓
fallback
        ↓
no silent request loss
```

### Traceability

Every request should allow you to reconstruct:

```text
request
 ↓
characterization
 ↓
model decision
 ↓
replica decision
 ↓
attempts
 ↓
cache outcome
 ↓
final result
```

---

# 16. Define the Measurement Framework

W1 should freeze the metrics that the future implementation will expose.

The project already has a strong metric set:

### Performance

* TTFT
* TPOT
* Throughput
* P95 latency
* P99 latency

### Resource

* GPU utilization
* GPU memory
* KV-cache utilization where available
* CPU utilization
* Queue/workload state

### Economics

* Cost/request
* Cost/token
* Cost/successful request

### Service

* SLO attainment
* Failure rate
* Fallback rate
* Cache hit rate
* Stale-cache rejection
* Retry amplification

These align with both the original milestone M8 and the W0 assessment.  

---

# 17. Define Experimental Reproducibility

This should be part of W1, not something added at the end.

Every experiment needs fixed:

```text
Workload trace
Policy version
Policy configuration
Hardware
Runtime
Model version
Random seed
Arrival process
Warm-up period
Number of repetitions
```

and outputs:

```text
Raw results
↓
Metric extraction
↓
Confidence intervals
↓
Plots
↓
Comparison
```

The evaluator explicitly requires this experimental discipline. 

---

# 18. What Should NOT Be Done in Week 1

This is equally important.

## Do NOT spend W1 trying to build:

### ❌ A new LLM

Not relevant.

### ❌ A new inference engine

vLLM/SGLang already exist.

### ❌ DistServe from scratch

The deeper project analysis explicitly places this outside scope. 

### ❌ Llumnix from scratch

Same reason.

### ❌ RL-based routing

Not yet.

Start with interpretable rules.

### ❌ Production Kubernetes infrastructure

Not required.

### ❌ Sophisticated semantic caching implementation

First define its **safety model and experimental role**.

### ❌ Full production gateway

The research contribution is the policy/decision layer, not reproducing LiteLLM/Kong/OpenRouter.

---

# 19. What About Coding in Week 1?

**Yes — but only foundational/prototyping code.**

The W0 assessment says W1 includes repository setup, while the broader project direction calls for a functional prototype and experimental artifacts.  

So I would divide coding into:

## Required W1 coding

### A. Project skeleton

Create repository/package structure.

### B. Request schema

Implement something like:

```python
RequestContext(
    task_type,
    prompt_tokens,
    estimated_output_tokens,
    quality_requirement,
    latency_slo,
    privacy_level,
    budget,
    tenant_id,
)
```

### C. Backend schema

```python
BackendState(
    provider,
    model,
    replica_id,
    queue_depth,
    estimated_workload,
    latency,
    memory_pressure,
    failure_rate,
    availability,
)
```

### D. Baseline policies

Implement:

```text
StrongestModelPolicy
CheapestModelPolicy
StaticRulePolicy
RoundRobinPolicy
LeastLoadedPolicy
```

### E. Initial adaptive policy

A simple interpretable rule/scoring policy.

### F. Mock backend

This is particularly important because it lets you test the entire architecture without GPU availability.

### G. Basic experiment runner

Something that can execute:

```text
workload
   ↓
policy
   ↓
mock backends
   ↓
results.json/csv
```

### H. Tests

Especially the safety/acceptance properties listed above.

---

# 20. What Research Needs to Be Done in W1?

I'd split it into **five concrete research tracks**.

## Research Track 1 — LLM routing

Study:

* RouteLLM
* model-selection approaches
* quality/cost routing
* task-aware routing

Goal:

> Define what information a model router needs and what your baseline/adaptive policy will contribute.

---

## Research Track 2 — LLM serving/scheduling

Study:

* vLLM/PagedAttention
* Llumnix
* DistServe
* SGLang

Goal:

> Understand what runtime information is meaningful for replica selection and what can realistically be exposed to the gateway.

---

## Research Track 3 — Gateway/resilience

Study:

* Envoy architecture
* retries
* retry budgets
* circuit breakers
* rate limiting
* failover

Goal:

> Reuse established distributed-system mechanisms instead of inventing them.

---

## Research Track 4 — Quality evaluation

Investigate:

> How can quality be measured separately for QA, reasoning, coding, summarization, structured extraction and long-context workloads?

This is essential because quality is one of your optimization dimensions.

---

## Research Track 5 — Output-length estimation

Investigate:

> How can pre-inference generation length be estimated without using future information?

And define the experiment for measuring estimator error.

---

# 21. The W1 Deliverables I Would Freeze

By the **end of Week 1**, I would expect the repository/project to contain approximately these artifacts:

```text
AdaRoute/Progress_Docs/W1/
│
├── 01_problem_definition.md
│
├── 02_assumptions.md
│
├── 03_literature_and_system_landscape.md
│
├── 04_research_gap.md
│
├── 05_system_architecture.md
│
├── 06_request_characterization.md
│
├── 07_output_length_estimation.md
│
├── 08_routing_policy_spec.md
│
├── 09_replica_scheduler_spec.md
│
├── 10_cache_safety_spec.md
│
├── 11_resilience_spec.md
│
├── 12_quality_evaluation.md
│
├── 13_baselines.md
│
├── 14_experiment_matrix.md
│
├── 15_metrics.md
│
├── 16_acceptance_tests.md
│
├── 17_project_plan.md
│
└── repository/
    ├── initial project skeleton
    ├── schemas
    ├── baseline policies
    ├── mock backend
    └── initial tests
```

Not necessarily all as separate files — they can be consolidated into a smaller technical specification — but **all of these topics need to be resolved**.

---

# 22. How W1 Fits Into M1–M8

Your original milestones remain useful. The deeper analysis says to **amend rather than replace them**. 

So:

```text
                 WEEK 0
                   │
                   ▼
        Problem + Initial Proposal
                   │
                   ▼
        ┌─────────────────────┐
        │       WEEK 1        │
        │                     │
        │ Formal Specification│
        │ Assumptions         │
        │ Literature          │
        │ Baselines           │
        │ Architecture        │
        │ Metrics             │
        │ Acceptance Tests    │
        │ Repo Setup          │
        │ Initial Prototype   │
        └──────────┬──────────┘
                   │
                   ▼
                  M1
          Stable Gateway Contract
                   │
                   ▼
                  M2
        Multi-provider execution
                   │
                   ▼
                  M3
       Adaptive Cost-Quality Router
                   │
                   ▼
                  M4
             Safe Caching
                   │
                   ▼
                  M5
              Resilience
                   │
                   ▼
                  M6
         SLO-aware Scheduler
                   │
                   ▼
                  M7
            Observability
                   │
                   ▼
                  M8
       Reproducible Evaluation
```

---

# 23. The Most Important W1 Objective

If I compress the entire W0 assessment into **one sentence**, it is:

> **Turn the current good research idea into a precise, testable system specification before scaling implementation.**

The evaluator's assessment is very clear that your proposal's **core idea is already strong**: separating model/provider selection from replica selection is technically meaningful, and the rule-based-first approach is sensible. 

The weakness is not the fundamental idea.

The weakness is that several parts are still **conceptual rather than operationally defined**.

---

# Final W1 Checklist

I would use this as the actual working checklist:

### A. Problem & assumptions

* [ ] Refine problem statement
* [ ] Remove/define "optimal"
* [ ] Define hard constraints vs soft objectives
* [ ] Freeze hardware/resource assumptions
* [ ] Define implementation vs simulation boundaries

### B. Request characterization

* [ ] Define request schema
* [ ] Define how task type is obtained
* [ ] Define prompt tokenization
* [ ] Define output-length estimation
* [ ] Define quality requirement
* [ ] Define latency/SLO
* [ ] Define privacy
* [ ] Define budget
* [ ] Measure characterization overhead

### C. Backend characterization

* [ ] Define backend schema
* [ ] Define queue metric
* [ ] Define workload-weighted load
* [ ] Define latency telemetry
* [ ] Define GPU/memory telemetry
* [ ] Define telemetry freshness
* [ ] Define missing telemetry behavior
* [ ] Define unhealthy/recovery states

### D. Routing

* [ ] Strongest baseline
* [ ] Cheapest baseline
* [ ] Static rule baseline
* [ ] Adaptive model router
* [ ] Define routing objective
* [ ] Define constraints

### E. Replica scheduling

* [ ] Round-robin
* [ ] Least-loaded
* [ ] Adaptive/SLO-aware
* [ ] Define scheduler score/objective
* [ ] Define oscillation prevention

### F. Quality

* [ ] Define workload taxonomy
* [ ] Define quality metric per task
* [ ] Define evaluation dataset/workload
* [ ] Define quality reporting by category

### G. Research

* [ ] vLLM/PagedAttention
* [ ] RouteLLM
* [ ] SGLang
* [ ] Llumnix
* [ ] DistServe
* [ ] Envoy
* [ ] Existing AI gateways
* [ ] Explicit research-gap statement

### H. Architecture

* [ ] API contract
* [ ] Request schema
* [ ] Backend abstraction
* [ ] Policy engine
* [ ] Model/provider router
* [ ] Replica scheduler
* [ ] Cache policy
* [ ] Resilience state machine
* [ ] Observability schema

### I. Evaluation

* [ ] Baseline matrix
* [ ] Workload matrix
* [ ] Failure scenarios
* [ ] TTFT
* [ ] TPOT
* [ ] Throughput
* [ ] P95/P99
* [ ] Cost
* [ ] Quality
* [ ] SLO attainment
* [ ] Resource utilization
* [ ] Failure recovery
* [ ] Reproducibility protocol

### J. Repository

* [ ] Initial repository
* [ ] Project skeleton
* [ ] Schemas
* [ ] Baseline policies
* [ ] Mock backend
* [ ] Initial adaptive policy
* [ ] Experiment runner
* [ ] Initial tests
* [ ] README/documentation

---

## Bottom line

**Week 1 is primarily a specification + research + experimental-design + repository-foundation week, with a small amount of foundational coding.**

The expected progression is:

$$
\boxed{
\text{W0 idea}
\rightarrow
\text{W1 precise specification}
\rightarrow
\text{M1--M7 implementation}
\rightarrow
\text{M8 experimental evidence}
}
$$

And the most important thing to protect is the project's actual research identity:

$$
\boxed{
\text{Request Characterization}
+
\text{Real-time Backend State}
+
\text{Policy Constraints}
\rightarrow
\text{Adaptive Model + Replica Decisions}
}
$$

rather than drifting into simply **"building another LLM gateway."** The deeper project analysis explicitly identifies the adaptive, cross-layer policy layer as the intended research contribution. 

The W1 assessment itself boils the required submission down to **problem definition, assumptions, literature/system landscape, alternatives, baselines, architecture, metrics, milestone acceptance tests, project plan, and initial repository setup**. 

**So I would not start M1 implementation yet. First, finish the W1 technical specification and the minimal mock-policy prototype; then use that specification as the contract for M1 onward.**
