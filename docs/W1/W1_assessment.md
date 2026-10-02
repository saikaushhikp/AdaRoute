# Overall Assessment

Your Week 1 submission makes clear progress from the initial research proposal: it defines hard constraints, a two-stage routing experiment, task-specific metrics and an inspectable mock gateway. The model-versus-replica distinction remains the strongest idea. Preserve it as a small policy extension on an existing OSS gateway rather than rebuilding the gateway platform. The current implementation is narrower than the report suggests: both backends are mocks, the named strongest/cheapest baselines are identical, queues are static, and a reproduced privacy-label mismatch can route an internal_only request to the external mock. Fix that defect first. Then demonstrate one real self-hosted model and an alternate/mock through the same client, and compare genuine baselines under an independently generated workload. Use AI tools extensively for integration, testing and analysis, while ensuring both members can explain, test and modify all submitted work. A project/company savings dashboard or control-plane enhancement is optional after core M1–M8 work is demonstrated.

# Strengths in the current submission

- Clear policy question. The report separates model/provider choice from replica choice and proposes five combinations to isolate their effects. This is a useful experimental structure.
- Improved specification. Privacy, budget and quality are identified as constraints; request features, output-length estimation, cache bypass and bounded retries are discussed. The implementation does not yet enforce the complete specification.
- Inspectible working component. The deterministic router filters availability and mock prices, then selects by queue depth. Three existing unit-test functions passed by direct invocation in the review runtime.
- Quality and resource awareness. Per-task quality measures, a characterization overhead target below 5 ms and CPU/simulation fallback are explicit. These support a feasible first evaluation.
- Useful engineering setup. FastAPI/Pydantic modules, mock providers, Dockerfile, justfile and six named tests provide a starting point. Their presence does not yet establish a reproducible benchmark or a full integration-suite pass.

# GitHub Repo review

The 16-file checkout includes six test functions.

Claims that need narrower wording
Strongest and cheapest both return the first healthy backend. The API registry has one mock per different model with static queue values. Availability filtering and 503 are not failover, quotas or a circuit breaker. Response metadata is not a request/attempt trace. A Dockerfile without pinned dependencies, workload and runtime versions does not establish identical benchmark conditions across hosts

# Minimum viable prototype

**Preserve the assigned project**
The policy experiment can be the team’s main contribution while existing OSS supplies gateway functions. Summary M1–M8 remain mandatory, including safe caching, quotas, observability and one real self-hosted model. The current two mocks are useful fixtures, but do not complete multi-provider execution

**Final anchor workflow**
One unchanged client invokes local and alternate providers, with routing, eligible cache reuse, quota enforcement and bounded failover under injected failures. The evaluator reproduces cost-quality-latency results. CPU/single-GPU/simulation limitations are declared; mock scheduling results are not presented as real model-quality or GPU-performance gains.

# Reuse existing gateways and fix the policy foundation

**LiteLLM gateway plus AdaRoute policy**
Recommended first spike because AdaRoute is already Python. Reuse provider execution and existing routing/reliability hooks; keep the bounded model/replica policy and evaluator as the contribution. LiteLLM already documents least-busy, latency-based, cost-based and custom routing [S1], so the report’s broad claim that gateways mostly use fixed rules needs revision.

**Bifrost plus policy adapter**
A viable comparison candidate for an HTTP gateway with existing provider, fallback, load-balancing and observability facilities [S2]. Compare integration hooks, edition coverage, setup effort and measured overhead using the same tiny workflow. Do not build both stacks long term.

**Real local model plus simulated replicas**
Use an existing local-serving runtime with a small open model; LiteLLM documents Ollama integration [S3]. Keep mocks for additional providers and fault/queue experiments. At least one real self-hosted model is still required. The current mock-local label does not satisfy M2.

**A bounded contribution**
Time-box the OSS comparison to one local-model plus alternate/mock workflow. Compare a gateway extension with the custom implementation using integration effort, policy hooks, constraint propagation, telemetry, edition coverage and reproducibility. Keep the policy and evaluation work. Rewrite the research claim as a testable hypothesis about estimator-aware, telemetry-aware scheduling under a declared workload, rather than claiming a generally missing gateway layer.

**P0 Privacy labels must fail closed**
Reproduced on the submitted code: internal_only with local queue 10 and external queue 0 selects mock-external. DeterministicRouter filters only internal/high, while the specification and TC_1.1 use internal_only. Define a validated canonical enum and explicit backend allow-list; reject unknown labels, rather than treating them as public. Bind policy and tenant identity to authenticated server-side configuration. Apply the same eligibility checks to every baseline, retry and fallback.

**P0 Tests must force the forbidden choice**
TC_1.1 and TC_1.2 only retain request values. The integration fixtures already favor the local backend, so privacy/budget tests can pass even if those filters are removed. Make the forbidden external backend cheaper or less loaded, test internal_only and all-invalid candidates, and assert both selection and absence of an external call. Test unknown labels, equality/negative budget boundaries and unavailable local replicas.

# Make routing and measurement meaningful
 	
**P1 Make baselines and the two stages real**
StrongestModelPolicy and CheapestModelPolicy both return healthy[0] and ignore request constraints. Implement distinct, versioned capability/quality and price tables behind a shared eligibility filter. Select an eligible model, then a replica of that model. The current registry has one replica per different model; it cannot isolate within-model scheduling. Add two replicas of one model and live or event-simulated queue updates.

**P1 Estimated quality and cost are not guarantees**
Reconcile the five-term and three-term objectives, normalize units, declare weights/tie-breaks and calibrate predictions on development data only. Quality feasibility is a measured proxy, not a per-response guarantee. A flat model price and an output-length estimate do not enforce a hard monetary ceiling. Reserve budget against bounded output/attempts, enforce max_tokens and total retry budget, then reconcile usage. Preserve privacy and budget checks during fallback.

**P1 Independent demand and measured state**
Use the correct tokenizer or a documented calibrated approximation per model; word count is not token count. Implement the stated estimator instead of a constant 50. Generate output demand independently of the estimator: both mock implementations currently sample output around the estimate. Freeze a separate workload trace, then inject estimator error and stale/missing telemetry. Define publication interval, age threshold, conservative fallback, queue updates and gradual recovery.

**Honest baseline experiment**
Keep the five model/replica combinations. Use a shared feasible set, workload trace, random seeds, model versions, arrival process, warm-up and failure schedule. Distinguish unconstrained diagnostic references from compliant baselines. Randomize run order and report repeated-run uncertainty, per-task results, failures and SLO attainment.

**Quality and estimator validity**
Freeze a small real-output dataset with per-task scoring and numeric quality floors. Use exact answers or coding tests where available; define a factual-consistency rubric for summaries instead of assuming ROUGE/BERTScore alone proves grounding. Split calibration from evaluation. Report estimator MAE and signed/tail errors per task, then measure routing sensitivity. Mocks validate mechanics, not real-model quality gains.

**Performance and simulation**
Report characterization and policy overhead separately, including p95/p99 against the proposed <5 ms characterization target. Measure TTFT/TPOT through streaming or serving instrumentation; the current sleep plus final mock response does not measure either. Use fixed short/long/burst profiles and separate real, simulated and estimated results. Record resource limits and simulator assumptions; a Dockerfile alone does not make results identical across hosts.

# Complete the core and optional brownie points

**Complete gateway safety and observability**
The API is a custom prompt schema without a logical model/capability alias, and neither provider executes the prompt on a real model. Add a documented stable contract and same-client model switch. Tenant/privacy-only cache keys and TTL omit model, options, user/content state and policy versions. Implement authorization before cache lookup, invalidation/bypass and cache-hit tracing. Availability filtering plus HTTP 503 is not retry/failover/quota handling. Add bounded deadlines, 429/backoff, circuit recovery and request/attempt correlation without raw prompts or secrets.

**Cost accounting**
Use versioned input/output prices and allocated local compute costs. Include billable failed/retried attempts, cache outcomes and gateway overhead. Report cost per successful task alongside failure and quality rates, with costs of failed requests still included in the numerator. Show estimates separately from invoiced spend and avoid double-counting caching and routing savings.

**Core versus bonus**
M3/M6 already require routing and replica-policy evaluation; the two-stage idea is therefore not automatically a bonus. M7/M8 already require usage and cost reporting. An existing dashboard or a new UI alone does not qualify. Propose a distinct improvement beyond frozen tests and demonstrate its added value.

**Useful bounded extension**
An auditable project/company savings view can show request and attempt counts by model, cache hits, failures, token usage and estimated spend. Optionally add versioned routing-policy dry run/apply/rollback with an audit trail. Do not build a general administrative platform. Scope dashboard access by authenticated project/tenant and restrict organization totals to the appropriate role.

**Savings with an explicit comparator**
For project p, estimated savings S_p = B_p − C_p, where B_p is the same workload under the declared static reference policy and C_p includes all actual gateway attempts plus allocated costs. Percentage = 100 × S_p/B_p only when B_p > 0; otherwise N/A. Company percentage = 100 × ΣS_p/ΣB_p, not the mean of project percentages. Show negative savings. If baseline execution was not measured, label its cost counterfactual/estimated; do not call it realized cash savings.

**Evidence for eligibility**
Use a hand-calculated multi-project ledger fixture with retries, failures and cache hits; reconcile totals and prevent cross-project visibility. Publish price/configuration versions and comparator quality/SLO checks. Claim one eligible BP lane per achievement with reproducible added value; no bonus is awarded for the idea alone.

Verify edition boundaries before choosing dashboard dependencies: LiteLLM documents some reporting and custom metadata features as Enterprise [S5]. Existing OSS functionality can still supply the base; independently validate the exact feature set in the pinned version.

---

| Week0 requirement | Status | Current evidence and remaining gap |
| --- | --- | --- |
| 01 Measurable objective | Partial | Hard constraints and a weighted objective are specified. The full objective includes SLO/failure terms while the later adaptive rule omits them; weights/scales and the quality feasibility test remain undefined. Code uses queue depth and mock prices only. |
| 02 Output-length estimation | Partial | Report explicitly avoids future-output leakage and defines a task/max_tokens estimator and absolute error. API always uses 50 output tokens; mock output is sampled around that estimate, making estimator validation circular. |
| 03 Operational characterization | Partial | Sources for task/privacy/SLO/budget and a \<5 ms target are described. Actual code counts words, fixes quality/SLO/output estimate and uses default-tenant. Tokenizer calibration and measured overhead are absent. |
| 04 Two-stage experiment | Partial | Five policy combinations and workload scenarios are explicit. Optional oracle is not essential. Implementation does not separate model and replica choice; strongest/cheapest both select the first healthy backend. |
| 05 Quality measurement | Partial | QA, math, coding and summarization metrics are named and per-class reporting is planned. Dataset IDs, quality floors, calibration/test split, judge protocol and real outputs remain absent. Similarity scores alone do not establish factual consistency. |
| 06 Dynamic backend state | Partial | Weighted queue, degradation and gradual recovery are described. Publication interval, T, maximum age, hysteresis and missing-state behavior are unspecified; current queue values are static. |
| 07 Experimental rigor | Partial | Workload scenarios and a Docker setup exist. No fixed traces/seeds, warm-up/repetitions, confidence-interval method, pinned model/runtime versions, raw results or runnable benchmark analysis are supplied. |
| 08 Verification properties | Partial | Seven invariants and six test IDs are listed. Only three unit functions were reproduced; two check stored fields. The internal_only privacy counterexample fails the intended invariant, and most robustness/trace properties remain untested. |

---

| Milestone | Proposed acceptance evidence | Target |
|---|---|---|
| M1 Stable contract | TC1.1: one documented logical alias/capability works through unchanged client code. TC1.2: unsupported/invalid fields fail clearly. Current custom prompt endpoint is partial. | W2 |
| M2 Multi-provider execution | TC2.1: real self-hosted model plus alternate/mock process requests. TC2.2: same capability switches without caller changes. Two mock labels are not completion. | W2–W3 |
| M3 Cost/quality routing | TC3.1: canonical privacy, quality and bounded cost eligibility. TC3.2: genuine strongest/cheapest/static versus adaptive comparison. Record hard-constraint rejection reasons. | W2–W5 |
| M4 Safe caching | TC4.1: eligible repeat hits within authorized tenant/user scope. TC4.2: state/content/model/policy changes invalidate or miss. Agree required cache modes before freeze; prefix/KV is optional. | W3–W4 |
| M5 Resilience and quotas | TC5.1: timeout/429/outage gives bounded retries/fallback and circuit recovery. TC5.2: quota exhaustion/permanent invalid requests do not amplify upstream calls. | W3–W4 |
| M6 SLO-aware serving | TC6.1: two replicas of one model react to queue work and memory/staleness. TC6.2: compare round-robin/least-loaded with adaptive scheduling under identical arrivals. | W3–W5 |
| M7 Usage and observability | TC7.1: request/attempt correlation covers success, errors and cache hits. TC7.2: tenant/project-scoped records and secret-safe logging. Response metadata alone is insufficient. | W2–W4 |
| M8 Reproducible benchmark | TC8.1: pinned setup and fixed mixed/burst trace produce required raw metrics. TC8.2: reproducible failure/quality/cost comparison with real/simulated results separated. | W3 harness; W5–W7 results |

# W2 Requirements

- Open-Source-Software choice, local+alternate vertical slice, API contract and initial isolation/failure/ledger evidence; official scope and benchmark design.
- Core architecture demonstrated and first substantial milestone set completed