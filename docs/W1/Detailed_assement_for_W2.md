# Week-2: What Must Be Done

I reconciled the **Week-0 assessment**, the **submitted Week-1 report**, the **Week-1 assessment**, the original project/milestone requirements, and the actual AdaRoute repository at the Week-1 commit you referenced.

The important point is that **Week 2 is not another specification/research-only week**. The evaluator is asking you to convert the Week-1 design into a **credible, testable vertical slice** and fix several concrete correctness problems first.

The Week-1 report established the intended system around hard privacy/budget/quality constraints, request characterization, two-stage model→replica decisions, caching, resilience, quality evaluation, baselines, metrics and acceptance tests.  

The Week-1 assessment then found that the implementation is still substantially narrower than the specification: **both backends are mocks, strongest and cheapest are effectively identical, queues are static, and the privacy invariant is actually broken for `internal_only`**. 

So Week 2 should be treated as:

> **Fix correctness → establish a real gateway vertical slice → make model/replica decisions genuinely separate → make the baselines meaningful → establish reproducible workload/measurement foundations.**

---

# 1. The highest-priority Week-2 objective

The evaluator's explicit Week-2 requirement is:

> **OSS choice + local/alternate vertical slice + API contract + initial isolation/failure/ledger evidence + official scope and benchmark design.** 

In practical terms, by the end of Week 2 you should be able to demonstrate this:

```text
                 SAME CLIENT
                     |
                     v
              AdaRoute Gateway
                     |
          +----------+----------+
          |                     |
          v                     v
   Real self-hosted       Alternate/mock
       model                  backend
          |
      1+ replicas
```

with the gateway making a real routing decision, while enforcing privacy/budget constraints and producing traceable evidence.

This directly restores alignment with the original assignment, which requires **at least one self-hosted open model and one additional provider/mock through a provider-independent gateway**. 

---

# 2. P0 — FIX THE PRIVACY BUG FIRST

This is the most urgent item.

The current code contains:

```python
if request.privacy_level in ["internal", "high"]:
```

but the documented specification and test use:

```text
internal_only
```

Therefore an `internal_only` request can incorrectly reach the external backend.

The evaluator explicitly reproduced this failure. 

## What to change

### A. Create a canonical privacy model

Do **not** leave privacy as arbitrary strings.

For example:

```python
class PrivacyLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal_only"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
```

The exact categories can be smaller, but they must be canonical and validated.

### B. Define backend eligibility explicitly

Each backend should have something conceptually like:

```text
allowed_privacy_levels:
    public
    internal_only
```

or:

```python
allowed_privacy = {
    "mock-local": {...},
    "external-provider": {"public"},
}
```

Then:

```text
request privacy
        +
backend allowed privacy
        ↓
candidate eligibility
```

### C. Fail closed

Unknown privacy labels should **not** silently become public.

Required behavior:

```text
unknown privacy → reject request
```

### D. Apply the same check everywhere

The privacy eligibility function must be used by:

* adaptive routing
* strongest baseline
* cheapest baseline
* static baseline
* replica selection
* retry
* fallback
* cache lookup
* cache retrieval

This is specifically called out by the evaluator. 

---

# 3. P0 — REBUILD THE TESTS SO THEY CAN ACTUALLY CATCH THE BUGS

The current tests are too weak.

For example, this:

```python
assert req.privacy_level == "internal_only"
```

does not test routing safety.

The evaluator explicitly noted that some tests merely verify that fields are stored rather than proving that routing behavior is correct. 

## Week-2 tests must force forbidden choices

### Privacy test

Construct:

```text
local cost = 0.01
external cost = 0.001
```

so that the external provider is **more attractive**.

Then:

```text
privacy = internal_only
```

Expected:

```text
external call = NEVER
selected backend = local
```

Test both:

1. selected backend
2. absence of external execution

### Budget test

Make:

```text
local = 0.001
external = 0.05
budget = 0.01
```

Then verify external is excluded.

Also test:

```text
budget < every candidate cost
```

Expected:

```text
safe rejection
```

### Unknown privacy test

```text
privacy = "something_unknown"
```

Expected:

```text
400 / validation error
```

not routing.

### Unavailable-local test

Make the only privacy-allowed local backend unavailable.

Expected:

```text
do not leak request to external
→ reject safely
```

This is important because otherwise fallback can accidentally bypass privacy.

The Week-1 assessment explicitly requires stronger tests including all-invalid candidates, unknown labels, boundary budgets, unavailable local replicas and proving that no external call occurs. 

---

# 4. P0 — STOP CLAIMING M2 IS COMPLETE

Your Week-1 README calls M2:

> “Mocked / Ready”

but the evaluator correctly points out that **two mocks do not complete multi-provider execution**. 

The original milestone requires:

```text
self-hosted open model
+
external/mock provider
```

through the same logical capability. 

## Week-2 target

You need:

```text
One unchanged client request
        ↓
AdaRoute
        ↓
logical capability / model alias
        ↓
either
 ┌───────────────┐
 │ Local model   │
 │ real serving  │
 └───────────────┘

or

 ┌────────────────┐
 │ alternate/mock │
 │ provider       │
 └────────────────┘
```

### Real local backend

Use an existing serving runtime rather than writing one.

The evaluator specifically recommends using an existing OSS gateway/runtime and retaining the policy work as the contribution. 

A practical path is:

```text
vLLM / Ollama / similar local runtime
```

with a small model appropriate to the available GPU.

### Alternate backend

Keep the mock if necessary.

That is completely reasonable for:

* external-provider simulation
* failures
* queue experiments
* rate-limit scenarios
* outage injection

But it must coexist with one **real local model**.

---

# 5. P1 — CHOOSE AND INTEGRATE AN EXISTING OSS GATEWAY

This is an important modification to the project direction.

Your Week-1 report still implicitly presents AdaRoute as though you are building the entire gateway.

The evaluator is recommending:

> **preserve AdaRoute's policy contribution but reuse an existing OSS gateway for generic gateway functionality.** 

The explicitly recommended candidate is:

### LiteLLM

The assessment specifically calls out LiteLLM as the first integration spike because AdaRoute is Python-oriented and LiteLLM already supplies provider execution and routing/reliability hooks. 

You should therefore **investigate and decide**:

```text
LiteLLM
   vs
Bifrost
```

but **do not develop both as full stacks**.

The comparison should be short:

| Criterion         | Question                                             |
| ----------------- | ---------------------------------------------------- |
| Integration       | Can AdaRoute policy plug in cleanly?                 |
| Provider support  | Does local + alternate work?                         |
| Routing hooks     | Can policy control model choice?                     |
| Reliability hooks | Retry/fallback/limits available?                     |
| Observability     | Can request/attempt information be captured?         |
| OSS availability  | Are needed capabilities available in chosen version? |
| Overhead          | Does the integration add measurable latency?         |
| Reproducibility   | Can the whole setup be pinned?                       |

Then choose one.

### Important research correction

The Week-1 report says:

> existing gateways mostly use fixed rules.

That claim now needs to be narrowed.

The evaluator notes that LiteLLM already documents least-busy, latency-based, cost-based and custom routing. 

So the research claim should become:

> **The contribution is not basic routing or gateway infrastructure; it is the experimentally evaluated coordination of request characterization, backend state, policy constraints and model/replica decisions.**

This is consistent with the deeper project framing. 

---

# 6. P1 — MAKE THE TWO-STAGE ROUTING REAL

This is one of the strongest ideas in your project, but the evaluator found that the implementation does not actually implement it.

The intended design is:

```text
Request
   |
   v
Stage 1
MODEL / PROVIDER SELECTION
   |
   v
Chosen Model Class
   |
   v
Stage 2
REPLICA SELECTION
   |
   v
Replica
```

The report already specifies this separation. 

The Week-1 assessment explicitly says:

> the implementation does not separate model and replica choice. 

## Week-2 implementation

### Stage 1

Choose:

```text
logical_model
provider/model class
```

based on:

* privacy
* budget
* minimum quality
* task
* latency/SLO
* estimated workload

### Stage 2

Among replicas of that selected model:

```text
Replica A
Replica B
Replica C
```

choose using:

* queue/work
* recent latency
* memory pressure
* health
* SLO

---

# 7. P1 — MAKE THE BASELINES ACTUALLY DIFFERENT

This is another major correction.

Currently:

```python
StrongestModelPolicy → healthy[0]
CheapestModelPolicy  → healthy[0]
```

So they are effectively identical.

The evaluator explicitly flags this. 

## Build actual model metadata

Example:

```python
ModelSpec(
    model="small-local",
    capability="general",
    quality_score=0.72,
    input_price=...,
    output_price=...,
)

ModelSpec(
    model="strong-model",
    capability="reasoning",
    quality_score=0.92,
    input_price=...,
    output_price=...,
)
```

Then implement:

### B1 — Strongest

```text
highest quality among eligible models
```

### B2 — Cheapest

```text
lowest cost among eligible models
```

### B3 — Static

```text
task_type → predetermined model
```

### B4 — Adaptive

```text
eligible models
→ objective
→ selected model
```

The original milestone structure explicitly requires routing to be compared against static routing baselines. 

---

# 8. P1 — ADD MULTIPLE REPLICAS OF THE SAME MODEL

Right now you effectively have:

```text
local model → one replica
external model → one replica
```

That makes meaningful replica scheduling impossible.

The evaluator explicitly says:

> add two replicas of one model and live or event-simulated queue updates. 

Week 2 should therefore introduce:

```text
Local-Model-A
   ├── replica-1
   ├── replica-2
   └── replica-3
```

The external/mock provider can remain separate.

---

# 9. P1 — MAKE QUEUE STATE DYNAMIC

Your report says the scheduler uses workload-aware queueing:

$$
W_q=\sum_i(\text{prompt tokens}_i+\hat{output tokens}_i)
$$

which is a good direction. 

But the implementation currently uses static values.

That means the scheduler cannot react to workload.

## Week-2 implementation

At minimum, implement event/simulation state:

```text
request arrives
     ↓
queue work increases
     ↓
request dispatched
     ↓
queue work decreases
```

A replica state should evolve.

For example:

```python
replica.queue_work
replica.active_work
replica.last_update
replica.memory_pressure
replica.recent_latency
replica.health
```

Then test:

```text
same model
R1 = heavily loaded
R2 = lightly loaded

request → R2
```

and after load changes:

```text
R2 becomes overloaded
R1 becomes free

next request → R1
```

---

# 10. P1 — IMPLEMENT THE OUTPUT-LENGTH ESTIMATOR PROPERLY

Your Week-1 report defined an estimator:

$$
\hat L_{out}=f(task,prompt\_length,max\_tokens,user\_history)
$$

but the code simply uses:

```python
estimated_output_tokens = 50
```

The evaluator also found a deeper experimental problem:

> the mock output length is sampled around the estimate, making estimator validation circular. 

## Week-2 implementation

Start simple.

For example:

```text
if max_tokens explicitly supplied:
    estimate based on bounded request value
else:
    estimate from task-specific statistics
```

Example:

```text
QA             → 50
summarization  → 150
coding         → 300
reasoning      → 200
extraction     → 100
```

But do **not** generate the actual mock output from the same estimate.

Instead:

```text
workload generator
        |
        +--> actual output demand
        |
        +--> request seen by estimator
                  |
                  v
             estimated output
```

This creates a real estimation error:

$$
e = \hat L_{out}-L_{out}.
$$

Then log:

* MAE
* mean signed error
* p95 absolute error
* error by task

This is exactly the correction requested by the assessment. 

---

# 11. P1 — FIX TOKEN COUNTING

Current code:

```python
len(req.prompt.split())
```

is word count, not token count.

The evaluator explicitly calls this out. 

Week 2 should use:

```text
model-specific tokenizer
```

or a clearly documented calibrated approximation in simulation.

At minimum:

```text
prompt
 ↓
tokenizer
 ↓
prompt_tokens
```

and preserve the distinction between:

```text
words ≠ tokens
```

---

# 12. P1 — DEFINE REAL MODEL CAPABILITY / QUALITY METADATA

Your routing algorithm currently lacks meaningful quality information.

Yet your specification says:

```text
Quality(model) ≥ Qmin
```

and the objective contains:

$$
Q_{loss}.
$$

The evaluator says you need:

> versioned capability/quality and price tables behind a shared eligibility filter. 

So create a central registry such as:

```text
Model Registry
 ├── logical capability
 ├── model id
 ├── provider
 ├── quality tier
 ├── input price
 ├── output price
 ├── privacy permissions
 ├── max context
 └── version
```

This registry becomes the source of truth.

---

# 13. P1 — RECONCILE THE ROUTING OBJECTIVE

There is currently a specification mismatch.

The Week-1 report defines:

$$
J =
w_cC+w_lL+w_qQ_{loss}+w_sSLO+w_fF
$$

but the adaptive implementation discusses only:

$$
w_cC+w_lL+w_qQ_{loss}.
$$

The evaluator explicitly identified this inconsistency. 

## Week 2 decision

Choose a single formal objective.

For example:

$$
J(a)=
w_c\tilde C(a)
+w_l\tilde L(a)
+w_q\tilde Q_{loss}(a)
+w_s\tilde SLO(a)
+w_f\tilde F(a)
$$

where the tilde terms are normalized.

Then document:

* units
* normalization
* weights
* tie-breaking
* hard constraints
* soft preferences

Do not call the output “optimal” in an absolute sense. Call it:

> **minimum-scoring feasible action under the declared policy.**

That resolves the Week-0 “optimal routing” criticism as well. 

---

# 14. P1 — BUDGET MUST BECOME A REAL CONSTRAINT

Your current check is basically:

```text
model_price <= budget
```

The evaluator correctly notes that this does not guarantee the final request stays within budget because output tokens and retries have cost. 

Week 2 should introduce:

$$
C_{\text{estimated}}
=
C_{input}
+
C_{output}(\hat L_{out})
$$

and ideally:

$$
C_{\text{reserved}}
=
C_{\text{attempt}}
\times
N_{\text{allowed attempts}}.
$$

Then:

```text
budget check
   ↓
reserve worst/bounded cost
   ↓
execute
   ↓
reconcile actual usage
```

This becomes especially important once retries/fallback are implemented.

---

# 15. P1 — DEFINE STATE TELEMETRY SEMANTICS

Your report says the router will use live state, but does not yet define exactly what “live” means.

The Week-0 assessment already flagged:

* publication interval
* max staleness
* missing telemetry
* queue semantics
* active tokens
* unhealthy state
* recovery
* oscillation prevention. 

Week 2 needs to freeze these definitions.

For example:

```text
Telemetry publication:
    every 100 ms

Maximum accepted age:
    500 ms

Telemetry missing:
    mark degraded

Replica offline:
    3 consecutive failed health checks

Recovery:
    staged re-entry

Scheduling hysteresis:
    do not change replica unless score improvement > threshold
```

The exact numbers are design parameters; what matters is that they are **explicit and experimentally reproducible**.

---

# 16. P1 — BUILD THE FIRST REAL FAILURE PATH

Your current behavior:

```text
unavailable backend
      ↓
HTTP 503
```

is not yet resilience.

The assessment explicitly says availability filtering + 503 is not retry/failover/quota handling. 

Week 2 should at least start:

```text
attempt
  ↓
timeout / 429 / 5xx
  ↓
bounded retry?
  ↓
alternate eligible replica/model
  ↓
success
```

Keep retry count bounded.

Example:

```text
max_attempts = 2 or 3
```

and never retry:

```text
invalid request
privacy violation
budget impossible
```

The original milestone specifically requires bounded retry, fallback and circuit-breaker behavior. 

---

# 17. P1 — ADD REQUEST/ATTEMPT CORRELATION

The Week-1 report says observability should capture routing decisions, cache results, failures and traces. 

But the evaluator notes that response metadata is not a request/attempt trace. 

Week 2 should introduce at least:

```json
{
  "request_id": "...",
  "tenant_id": "...",
  "logical_capability": "...",
  "task_type": "...",
  "privacy_level": "...",
  "selected_model": "...",
  "selected_replica": "...",
  "attempt": 1,
  "cache": "miss",
  "outcome": "success",
  "latency_ms": 123,
  "estimated_tokens": 200,
  "actual_tokens": 183
}
```

For retries:

```text
request_id = same
attempt = 1
attempt = 2
attempt = 3
```

This gives you the audit trail you need later for M7/M8.

---

# 18. P1 — INTRODUCE THE STABLE LOGICAL API CONTRACT

The evaluator says your current API is a custom prompt schema without a logical model/capability alias. 

Week 2 should therefore freeze something like:

```json
{
  "model": "general",
  "messages": [...],
  "max_tokens": 200,
  "budget": 0.02,
  "privacy_level": "internal_only",
  "latency_slo": 2.0
}
```

where:

```text
model = logical capability
```

rather than:

```text
model = provider-specific actual model
```

For example:

```text
general
coding
reasoning
summarization
```

The gateway translates that into:

```text
logical capability
      ↓
candidate models
      ↓
provider
      ↓
replica
```

That makes M1 meaningful.

---

# 19. P1 — START SAFE CACHE IMPLEMENTATION, BUT KEEP IT SMALL

Caching is not the largest Week-2 priority, but you should at least establish the correct foundation because M4 remains mandatory.

The evaluator specifically says the current cache concept is insufficient because tenant/privacy alone does not protect against changes in:

* model
* options
* user state
* content state
* policy versions. 

The deeper specification gives the intended conceptual cache identity:

$$
K =
H(
tenant/user,
prompt,
conversation,
model,
model\_version,
system\_prompt\_version,
retrieval\_state,
policy\_state,
tool\_state
).
$$



## For Week 2

Implement **exact caching only**.

A good initial key:

```text
tenant_id
+ user_id
+ normalized_prompt
+ logical_model
+ model_version
+ policy_version
+ request options
```

and enforce:

```text
authorization
      ↓
cache eligibility
      ↓
cache lookup
```

not:

```text
cache lookup
      ↓
authorization
```

The Week-1 assessment specifically calls for authorization before cache lookup, invalidation/bypass, and cache-hit tracing. 

Semantic caching can wait.

---

# 20. P1 — BUILD AN INDEPENDENT WORKLOAD GENERATOR

This is one of the most important experimental tasks.

Right now your mock output depends on the estimate itself, which makes evaluation circular.

The assessment explicitly requires **independently generated demand**. 

Week 2 should therefore create:

```text
workloads/
   short.jsonl
   long.jsonl
   mixed.jsonl
   burst.jsonl
   repeated.jsonl
```

Each generated request should have known ground-truth characteristics:

```text
task_type
prompt_tokens
actual_output_tokens
privacy
quality_requirement
latency_slo
budget
arrival_time
```

The estimator only sees allowed **pre-inference information**.

The actual output length remains hidden from the router.

That creates a proper experimental separation:

```text
GROUND TRUTH
     |
     +----------------------+
     |                      |
     v                      v
actual demand         estimator input
     |                      |
     |                      v
     |                estimated demand
     |                      |
     +----------+-----------+
                |
                v
          routing decision
```

---

# 21. P1 — START THE BENCHMARK HARNESS

Do **not** wait until M8 to create the benchmark framework.

The Week-1 evaluator wants the benchmark design established now, and the original project explicitly ends in reproducible evaluation. 

Week 2 should create:

```text
experiments/
    configs/
    workloads/
    runners/
    results/
    analysis/
```

and a runner like:

```bash
python -m experiments.run \
    --policy adaptive \
    --scheduler rr \
    --workload mixed \
    --seed 42
```

Output:

```text
results/
  run_001/
     config.json
     workload.jsonl
     raw_results.jsonl
     summary.json
```

---

# 22. The first experimental matrix for Week 2

Do **not** attempt the entire E0–E11 matrix yet.

Week 2 should establish the infrastructure for a smaller slice.

I recommend:

| Experiment | Model policy | Replica policy | Workload         |
| ---------- | ------------ | -------------- | ---------------- |
| W2-E1      | Strongest    | RR             | steady           |
| W2-E2      | Cheapest     | RR             | steady           |
| W2-E3      | Static       | RR             | steady           |
| W2-E4      | Adaptive     | RR             | steady           |
| W2-E5      | Adaptive     | Least-loaded   | steady           |
| W2-E6      | Adaptive     | SLO-aware      | steady           |
| W2-E7      | Adaptive     | SLO-aware      | burst            |
| W2-E8      | Adaptive     | SLO-aware      | injected failure |

This directly evolves the five combinations requested by the evaluator:

1. static + RR
2. static + least-loaded
3. adaptive + RR
4. adaptive + least-loaded
5. adaptive + adaptive replica



---

# 23. Metrics that should start working in Week 2

You do **not** need the full final benchmark set yet, but start collecting the core fields now.

## Request-level

```text
request_id
task_type
prompt_tokens
estimated_output_tokens
actual_output_tokens
privacy
budget
quality_requirement
latency_slo
```

## Routing

```text
model_selected
provider_selected
replica_selected
routing_policy
routing_score
rejection_reason
```

## Execution

```text
attempt_count
success/failure
latency
TTFT if streaming is available
TPOT if measurable
output_tokens
```

## State

```text
queue_work
memory_pressure
health
telemetry_age
```

## Economic

```text
estimated_cost
actual_cost
```

The M8 target ultimately includes TTFT, token latency, throughput, p95/p99, memory/GPU, SLO attainment and estimated cost. 

---

# 24. Research to do in Week 2

The research should now become **implementation-directed**, not broad literature collection.

## A. OSS gateway integration

Investigate:

```text
LiteLLM
Bifrost
```

Specifically:

* provider abstraction
* custom routing hooks
* fallback
* retries
* load balancing
* observability
* metadata propagation
* local model integration
* licensing / edition limitations
* exact version capabilities

The evaluator explicitly recommends this comparison. 

### B. Local serving runtime

Investigate:

```text
vLLM
Ollama
```

based on actual available GPU/CPU resources.

Your foundational research already establishes vLLM as the serving substrate rather than something you should reimplement. 

### C. Output-length prediction

Research lightweight approaches based on:

* task class
* prompt length
* max tokens
* historical request statistics
* response format

Do **not** jump to a sophisticated ML predictor yet.

### D. LLM quality evaluation

Now determine what dataset you will actually use for:

```text
QA
reasoning
coding
summarization
structured extraction
```

The Week-1 assessment specifically says dataset IDs, quality floors, calibration/test split, judge protocol and real outputs are still missing. 

---

# 25. Quality evaluation must become concrete

Your report currently says:

* QA → exact match/similarity
* reasoning → correctness
* coding → pass@1
* summarization → ROUGE/BERTScore/judge



That is still too abstract.

For Week 2, freeze a **small development dataset**.

For each task category define:

```text
dataset
input
reference
metric
minimum acceptable quality
```

For example:

```text
coding
→ deterministic unit tests
→ pass/fail
→ Qmin = X

QA
→ reference answer
→ exact / task-specific score
→ Qmin = Y
```

The evaluator explicitly cautions that similarity metrics alone do not establish factual consistency. 

---

# 26. What NOT to do in Week 2

This is important because the project can easily explode in scope.

## Do NOT spend Week 2 on:

### RL routing

Your current project direction correctly avoids premature RL. The Week-0 assessment explicitly liked the rule-based starting point. 

### Full semantic cache

Exact cache first.

### Prefix/KV reuse

Optional later.

### Full DistServe implementation

No.

### Full Llumnix implementation

No.

### Kubernetes

No.

### Large dashboard

Not yet.

The evaluator explicitly says the dashboard/control-plane enhancement is **optional after core M1–M8 work**. 

### Building two gateway stacks

No.

Pick one OSS integration after the spike.

---

# 27. Week-2 repository structure I recommend

Your current repository is only ~16 files and largely foundational.

Move toward something like:

```text
AdaRoute/
├── adaroute/
│   ├── api.py
│   ├── schemas.py
│   ├── policy/
│   │   ├── eligibility.py
│   │   ├── model_selection.py
│   │   └── replica_selection.py
│   ├── backends/
│   │   ├── base.py
│   │   ├── local.py
│   │   └── mock.py
│   ├── cache/
│   │   ├── policy.py
│   │   └── exact.py
│   ├── resilience/
│   │   ├── retry.py
│   │   └── circuit_breaker.py
│   ├── telemetry/
│   │   └── events.py
│   └── experiments/
│       ├── workloads/
│       ├── configs/
│       ├── runners/
│       └── analysis/
│
├── tests/
│   ├── test_privacy.py
│   ├── test_budget.py
│   ├── test_model_selection.py
│   ├── test_replica_selection.py
│   ├── test_failover.py
│   ├── test_cache.py
│   └── test_contract.py
│
├── docs/
│   ├── architecture.md
│   ├── routing.md
│   ├── cache_safety.md
│   ├── benchmark.md
│   └── integration.md
│
├── workloads/
├── experiments/
├── Dockerfile
├── pyproject.toml
└── justfile
```

The exact structure is your implementation choice, but the separation should mirror the architecture.

---

# 28. Week-2 acceptance criteria

This is the part I would treat as the **actual definition of “Week 2 complete.”**

## P0 — Correctness

### TC-P2-01 Privacy

```text
internal_only + external cheaper
→ external NEVER called
```

### TC-P2-02 Unknown privacy

```text
invalid privacy label
→ request rejected
```

### TC-P2-03 Budget

```text
no eligible backend within budget
→ safe rejection
```

### TC-P2-04 Budget + fallback

```text
primary becomes unavailable
→ fallback still respects budget
```

---

## P1 — M1

### TC-M1-01

Same client invokes:

```text
logical capability
```

without specifying provider-specific model.

### TC-M1-02

Invalid/unsupported fields are rejected clearly.

---

## P1 — M2

### TC-M2-01

Real local model successfully serves request.

### TC-M2-02

Alternate/mock backend serves same logical capability.

### TC-M2-03

Caller code remains unchanged when route changes.

---

## P1 — M3

### TC-M3-01

Strongest ≠ Cheapest.

### TC-M3-02

Static rule actually follows task mapping.

### TC-M3-03

Adaptive routing uses:

```text
request + capability + cost + constraints
```

---

## P1 — M6

### TC-M6-01

At least two replicas of same model exist.

### TC-M6-02

Replica state changes dynamically.

### TC-M6-03

Scheduling reacts to queue/work.

---

## P1 — M7

### TC-M7-01

One request produces complete structured trace.

### TC-M7-02

Retry generates multiple attempts under one request ID.

---

## P1 — M8 foundation

### TC-M8-01

Same workload trace + same seed + same config produces reproducible routing results.

### TC-M8-02

Results are written to machine-readable files.

---

# 29. What should be researched vs coded vs documented

A useful Week-2 split is:

| Area                           | Research |   Code  | Documentation |
| ------------------------------ | :------: | :-----: | :-----------: |
| LiteLLM/Bifrost                |    ✓✓    |    ✓    |       ✓       |
| Local serving runtime          |    ✓✓    |    ✓✓   |       ✓       |
| Privacy model                  |     —    |    ✓✓   |       ✓       |
| Budget model                   |     ✓    |    ✓✓   |       ✓       |
| Model registry                 |     —    |    ✓✓   |       ✓       |
| Two-stage routing              |     ✓    |    ✓✓   |       ✓       |
| Replica simulation             |     ✓    |    ✓✓   |       ✓       |
| Output estimator               |    ✓✓    |    ✓    |       ✓       |
| Independent workload generator |     ✓    |    ✓✓   |       ✓       |
| Exact cache                    |     ✓    |    ✓    |       ✓       |
| Retry/failover                 |     ✓    |    ✓    |       ✓       |
| Circuit breaker                |     ✓    | partial |       ✓       |
| Quality dataset                |    ✓✓    |    ✓    |       ✓✓      |
| Benchmark harness              |     ✓    |    ✓✓   |       ✓✓      |
| Dashboard                      |     —    |  **No** |       —       |

---

# 30. The most important conceptual modification to the project

Your Week-1 project was moving toward:

```text
"Build an intelligent AI gateway"
```

Week 2 should firmly establish:

```text
                  AdaRoute
                     |
        +------------+------------+
        |                         |
        v                         v
   MODEL DECISION           REPLICA DECISION
        |                         |
        +------------+------------+
                     |
                     v
              EXECUTION LAYER
```

where the **research contribution is the policy layer**, while existing infrastructure performs gateway/serving work.

That is consistent with the deeper research conclusion that the gateway itself is infrastructure and the stronger contribution is **cross-layer adaptive decision-making** using request state, model quality/cost, cache state, replica state, SLO and failure state. 

---

# 31. Week-2 priority order

This is the order I recommend actually executing the work:

### **P0 — Must happen first**

1. Fix canonical privacy handling.
2. Make privacy tests genuinely adversarial.
3. Fix budget enforcement.
4. Make failed/invalid routing fail closed.
5. Freeze API contract.

### **P1 — Core implementation**

6. Choose LiteLLM/Bifrost integration path.
7. Connect one real local model.
8. Keep one alternate/mock provider.
9. Introduce logical model/capability aliases.
10. Build model metadata/quality/price registry.
11. Implement genuine strongest/cheapest/static policies.
12. Implement actual adaptive model policy.
13. Add ≥2 replicas for one model.
14. Implement separate replica scheduler.
15. Make queue state dynamic/event-driven.

### **P1 — Evaluation foundation**

16. Implement real output-length estimator.
17. Build independently generated workload traces.
18. Freeze seeds/configuration/versioning.
19. Build benchmark runner.
20. Add structured request/attempt traces.
21. Begin TTFT/TPOT/latency/cost measurements.

### **P1 — Required vertical slice**

22. Exact safe cache foundation.
23. Timeout/429/5xx injection.
24. Bounded retry.
25. Same-policy-safe fallback.
26. Start circuit-breaker state.

### **P1 — Research validation**

27. Freeze quality dataset + metric definitions.
28. Research and document OSS integration limitations.
29. Update research-gap statement.
30. Record real-vs-simulated evaluation boundaries.

---

# 32. What your Week-2 end-state should look like

The ideal Week-2 checkpoint is **not** “all M1–M8 are complete.”

It should look like:

```text
                  CLIENT
                    |
                    v
             ┌──────────────┐
             │ AdaRoute API │
             └──────┬───────┘
                    |
                    v
          Request Characterizer
                    |
                    v
             Eligibility
       (privacy / budget / quality)
                    |
                    v
           MODEL SELECTION
          /      |       \
      local    alternate   mock
                    |
                    v
           REPLICA SELECTION
             /          \
           R1            R2
            \            /
             \          /
              REAL SERVING
                    |
                    v
              Trace / Cost
                    |
                    v
              Experiment Log
```

And you should be able to demonstrate one concrete scenario such as:

> A client submits the same logical capability request; AdaRoute routes it to the real local model when privacy requires locality, routes eligible public requests according to model policy, then chooses among multiple replicas using current workload state, while recording the routing decision and attempt trace.

That would be a **substantial and defensible Week-2 advancement** over the current Week-1 state.

---

## Bottom line

The Week-1 assessment is **not asking you to broaden AdaRoute**. It is asking you to **make the existing idea real**.

The biggest gaps are:

**1. privacy correctness**
**2. real local + alternate execution**
**3. genuine model-policy baselines**
**4. genuine two-stage model→replica selection**
**5. dynamic replica state**
**6. independent workload generation**
**7. proper output-length estimation**
**8. real failure/fallback behavior**
**9. request/attempt tracing**
**10. reproducible benchmark foundation**

The original project still requires the broader M1–M8 path—stable gateway contract, multi-provider execution, cost/quality routing, safe caching, resilience/quotas, SLO-aware scheduling, observability, and reproducible benchmarking. 

So the correct Week-2 strategy is:

> **Do not add more features blindly. Close the evaluator-identified gaps one by one, starting with P0 privacy, and finish with a real local-model + alternate-provider vertical slice plus reproducible policy/replica experiments.**

That gives you a strong foundation for **M2 → M3 → M6 → M7 → M8**, rather than letting the project remain a mock gateway with documentation claiming capabilities that the implementation does not yet possess.
