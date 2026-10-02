# Overall Assessment

 	
This proposal contains an excellent core insight: model selection and replica selection are related but distinct decisions and should be evaluated separately and jointly. The narrowed research direction is more achievable than building an entire production gateway, but it risks deviating from the assigned Project 8 requirements. Preserve the routing-policy focus while implementing a thin gateway or simulator that provides traceable, reproducible evidence. The immediate W1 gaps are assumptions, prior-system/literature analysis, repository evidence, and traceability.

# Strengths

1. Strong central research question
The report asks a clear systems question:
How should an LLM-serving system characterize incoming demand and combine it with changing backend conditions to make adaptive serving decisions?

This is more precise and researchable than merely proposing an “intelligent router.”

The proposal also identifies meaningful request attributes: task type, prompt length, expected output length, quality requirement, latency target, privacy constraints, and budget. On the infrastructure side, it considers queue state, service time, memory pressure, cost, failures, and availability. 

2. Valuable separation of two decisions
The distinction between:
  1. Model/provider selection, and
  2. Replica/instance selection
is technically important.

A model may be suitable for a task but temporarily overloaded. Conversely, an available replica may host a model that does not satisfy the quality, privacy, or cost constraints. Evaluating these decisions separately and jointly could become the project’s strongest contribution.

3. Sensible avoidance of premature complexity
The proposal explicitly begins with an interpretable rule-based controller rather than immediately introducing reinforcement learning. This is a good decision.

The team should first establish whether request characterization and live system state materially outperform simple baselines. A more complex learned policy is justified only if the rule-based approach exposes measurable limitations.

4. Strong initial evaluation metrics
The proposed metrics are appropriate:
* Time to first token
* Time per output token
* Throughput
* P95/P99 latency
* Resource utilization
* Estimated cost
* SLO attainment
* Failure-recovery behavior

The inclusion of strongest-model, cheapest-model, round-robin, least-loaded, static-rule, and adaptive-policy baselines gives the project a meaningful experimental direction.

5. Realistic resource constraints
Designing the evaluation so that it can run using a single GPU or CPU simulation is sensible. Reproducibility on limited hardware is more valuable than proposing an experiment that depends on unavailable infrastructure.



# Requirements/Gaps/Suggestions/Next Steps

1. Replace “optimal routing” with a measurable objective
The term “optimal” is too strong unless the team defines the objective and constraints.

The team must explain which factors are hard constraints and which are soft preferences. Privacy restrictions, for example, should normally be constraints rather than weighted preferences.

2. Expected output length is not known in advance
The proposal treats expected generation length as a request attribute. In practice, the actual output length is unknown before inference.

The system therefore needs an estimator based on:
* Task type
* Prompt features
* Requested maximum tokens
* Historical outputs for similar requests
* Model behavior
* User-specified response format

Evaluation should report the estimator’s error and test how routing quality degrades when the estimate is inaccurate.

Do not evaluate using the actual output length as an input to the router. That would leak future information and produce artificially optimistic results.

3. Define request characterization operationally
The report lists useful features but does not explain how they are obtained.

For each feature, specify:
- Task type        : Rule-based classifier, user metadata, or lightweight model 
- Prompt length    : Model-specific tokenizer                                   
- Output length    : Pre-inference estimate                                     
- Required quality : User policy or task-class configuration                    
- Latency target   : Tenant/application SLO                                     
- Privacy          : Data classification or tenant policy                       
- Budget           : Request, tenant, or workload budget                        
- Queue state      : Runtime-exported metric                                    
- Memory pressure  : Serving-runtime or GPU telemetry                           

Also measure the latency overhead introduced by characterization and policy evaluation. An intelligent router that adds excessive delay can erase the benefit of selecting a faster backend.

4. Make the two-stage decision experiment explicit
The model and replica selectors are separated architecturally, but their interaction still needs evaluation.

Compare at least:
1. Static model + round-robin replica
2. Static model + least-loaded replica
3. Adaptive model + round-robin replica
4. Adaptive model + least-loaded replica
5. Adaptive model + adaptive replica
6. Oracle or offline upper-bound policy, where feasible

This design will reveal whether improvements come from better model selection, better replica selection, or their combination.

5. Define quality measurement
Cost and latency are straightforward to measure; response quality is not.

Create a workload taxonomy:
* Factual question answering
* Reasoning
* Coding
* Summarization
* Structured extraction
* Long-context tasks

For each task class, define a quality metric such as exact match, unit-test pass rate, structured-output validity, task-specific score, or carefully controlled judge-based evaluation.

Report quality separately by task category. An aggregate average could conceal poor routing for high-value tasks.

6. Treat dynamic backend state carefully
Queue depth, memory pressure, and recent latency can become stale between collection and routing.

The team should define:
* Metric publication interval
* Maximum permissible staleness
* Behavior when metrics are missing
* Queue-length versus workload-weighted queue measurement
* Whether current active tokens are considered
* How replicas are marked unhealthy
* How recovery is detected
* How oscillation between replicas is prevented

A queue containing two long-context requests may represent more load than ten short requests. Simple request counts are therefore insufficient, as the proposal correctly notes.

7. Establish experimental rigor
Every policy should receive the same workload trace, random seed, model configuration, and failure schedule.

The experiment package should record:
* Workload definition
* Policy version and configuration
* Hardware/runtime configuration
* Model versions
* Random seed
* Arrival process
* Warm-up period
* Number of repetitions
* Confidence intervals
* Raw results
* Analysis scripts

Without this, differences may be caused by run-to-run variance rather than routing policy.

8. Add verification properties
The final tests should verify more than average performance.

Required properties should include:
* Privacy-restricted requests never reach disallowed backends.
* Budget-limited requests do not exceed the configured cost limit.
* Routing decisions are reproducible for fixed state and configuration.
* Missing telemetry triggers a safe fallback.
* Failed replicas are removed from selection.
* Recovered replicas re-enter service without oscillation.
* Policy overhead remains below a declared threshold.
* No request is silently dropped during failover.
* Correlation traces connect request characterization, model choice, replica choice, attempts, and final outcome.

# W1 Requirements

- Problem definition, assumptions, literature/system landscape, alternatives, baseline, architecture, metrics, milestone acceptance tests and project plan
- Repository's initial setup

Requirements for W1 are not fully satisfied. The proposal is strong on problem definition, assumptions, and metrics, but it lacks a clear literature/system landscape, alternatives, baseline, architecture, milestone acceptance tests, and project plan. The repository's initial setup is also missing.
