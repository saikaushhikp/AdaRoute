# AdaRoute — Architecture Decision Records

**Project:** AdaRoute, Project ID 8, Group 41  
**Documentation date:** 21 November 2026  
**Repository:** <https://github.com/saikaushhikp/AdaRoute>  
**Reviewed baseline:** branch `psk/gpu-vllm`

The milestone labels used below follow the [final report](Project_Report.pdf): **M1** API contract; **M2** provider execution; **M3** model selection; **M4** caching; **M5** resilience and quotas; **M6** replica scheduling; **M7** observability; and **M8** evaluation. The listed alternatives and reasons summarize the project design; measured results are identified separately from architectural judgment.

## Decision index

| ADR | Decision | Status | Scope | Decision date |
|---|---|---|---|---|
| [ADR-001](#adr-001--use-a-unified-two-stage-gateway-and-client-interface) | Unified two-stage gateway and client interface | Accepted | M1, M3, M6 | 04-10-2026 |
| [ADR-002](#adr-002--enforce-privacy-and-budget-constraints-before-ranking) | Privacy and budget constraints before ranking | Accepted | M3–M5 | 04-10-2026 |
| [ADR-003](#adr-003--use-a-model-registry-and-comparable-selection-policies) | Model registry, estimation and selection policies | Accepted | M3 | 04-10-2026 |
| [ADR-004](#adr-004--use-exact-response-caching-with-compatibility-checks) | Exact response caching | Accepted | M4 | 04-10-2026 |
| [ADR-005](#adr-005--isolate-provider-execution-behind-backend-adapters) | Local, hosted and mock backend adapters | Accepted | M2 | 04-10-2026 |
| [ADR-006](#adr-006--schedule-eligible-replicas-and-bound-failure-recovery) | Replica scheduling, retries and circuit breakers | Accepted | M5–M6 | 04-10-2026 |
| [ADR-007](#adr-007--record-routing-evidence-and-compare-policy-baselines) | Telemetry, ledger and comparative evaluation | Accepted | M7–M8 | 04-10-2026 |
| [ADR-008](#adr-008--extend-model-selection-with-a-unified-deployment-aware-pipeline) | Unified deployment-aware selection and optional semantic-router policy | Accepted | M3, M7 | 03-11-2026 |
| [ADR-009](#adr-009--extend-caching-with-redis-vector-semantic-search) | Redis-backed state and vector semantic caching | Accepted | M3–M4 | 03-11-2026 |
| [ADR-010](#adr-010--serve-local-models-and-real-replicas-with-vllm) | vLLM local serving and real model replicas | Accepted | M2, M6 | 03-11-2026 |

## Mandatory methodology record

**Status at 9 October 2026, 11:59 PM IST: Reaffirmed; our methodology and approach did not change.** We retained the two-stage method: select an eligible model, then select a replica, and compare routing policies using tests and benchmark workloads. Later work extended the implementation and evaluation while retaining this approach.

---

## ADR-001 — Use a unified two-stage gateway and client interface

| Field | Record |
|---|---|
| **Decision date** | 04-10-2026 |
| **Status and scope** | **Accepted**; M1, M3, M6; gateway API, routing stages and clients. |
| **Context and decision drivers** | Applications need one interface for requests with different quality, privacy, cost and latency needs. Model choice and local replica choice use different information and should be independently inspectable. |
| **Options considered** | **(a)** Call each provider directly: straightforward initially but duplicates client and policy logic. **(b)** Make one joint decision across every endpoint: potentially holistic but harder to evaluate by stage. **(c)** Provide one gateway API and separate model and replica decisions. |
| **Decision and rationale** | We use **(c)**. Our OpenAI-style chat endpoint accepts logical capabilities or model IDs and request constraints. Stage 1 selects an eligible model/deployment; Stage 2 selects a local replica. The web workbench and terminal client use the gateway interface. |
| **Consequences and trade-offs** | A stable contract lets clients compare providers and policies without separate integrations. The stages add interfaces and may compound stale-state errors; our interface does not implement every OpenAI feature, including streaming. Contract changes must preserve or deliberately version existing clients. |
| **Evidence and validation** |  [API](../adaroute/api.py), [schemas](../adaroute/schemas.py), [contract tests](../tests/test_contract.py), [UI](../ui/) and [CLI](../ui/cli/main.py). |
| **Revisit trigger** | A required client contract cannot be supported without breaking existing contract tests, or a repeated matched-workload study justifies a joint optimizer. |

## ADR-002 — Enforce privacy and budget constraints before ranking

| Field | Record |
|---|---|
| **Decision date** | 04-10-2026 |
| **Status and scope** | **Accepted**; M3–M5; eligibility, request privacy and budget admission. |
| **Context and decision drivers** | A cheaper or higher-quality provider must not override a request's privacy restrictions or budget constraint. A local demonstration also needs to remain easy to operate. |
| **Options considered** | **(a)** Treat privacy and budget as score penalties: flexible, but another objective can outweigh a requirement. **(b)** Filter ineligible candidates before scoring. **(c)** Require production-grade authentication for every demonstration. |
| **Decision and rationale** | We use **(b)**: filter candidates against request and deployment constraints before ranking. Our prototype configuration permits optional authentication for demonstration; request-level privacy restrictions still apply. We estimate cost before execution using token estimates and configured prices. |
| **Consequences and trade-offs** | Rejection reasons are inspectable and privacy tests can exercise fail-closed paths. Estimates are not invoices or shared concurrent quotas. Both checked-in profiles set `auth_required: false`; caller-supplied identities therefore do not establish a trusted multi-tenant boundary. Administrative endpoints also need protection before untrusted exposure. |
| **Evidence and validation** | [Constraint engine](../adaroute/selection/constraints.py), [privacy tests](../tests/test_privacy.py), [budget tests](../tests/test_budget.py), [authentication manager](../adaroute/auth/manager.py), [CPU configuration](../config.yaml) and [GPU configuration](../config.gpu.yaml). |
| **Revisit trigger** | Any adversarial test selects an ineligible deployment, or a deployment needs untrusted users, authenticated multitenancy or shared quota enforcement. |

## ADR-003 — Use a model registry and comparable selection policies

| Field | Record |
|---|---|
| **Decision date** | 04-10-2026 |
| **Status and scope** | **Accepted**; M3; model metadata, output estimates and policy baselines. |
| **Context and decision drivers** | Available models differ in capabilities, quality, prices, privacy and runtime behavior. The gateway needs comparable choices before it can evaluate adaptive routing. |
| **Options considered** | **(a)** Route every request to one fixed model: simple but ignores request variation. **(b)** Hard-code provider rules throughout the gateway: quick to start but difficult to maintain. **(c)** Maintain model metadata and compare strongest, cheapest, static and adaptive policies. |
| **Decision and rationale** | We use **(c)**, with token/output estimates to support cost and workload decisions. The strongest, cheapest and static policies provide understandable references for adaptive selection. The later deployment-aware implementation is recorded in [ADR-008](#adr-008--extend-model-selection-with-a-unified-deployment-aware-pipeline). |
| **Consequences and trade-offs** | We can compare policy choices under a common contract, but configured quality and prices are proxies and must be maintained. Estimation errors affect both admission and ranking; changing policy behavior requires preserving baseline and constraint tests. |
| **Evidence and validation** |[model policy tests](../tests/test_model_selection.py), [registry](../adaroute/policy/registry.py), [configuration](../config.yaml) and [budget tests](../tests/test_budget.py). |
| **Revisit trigger** | A policy selects an ineligible model, catalog metadata becomes too inaccurate for the workload, or a controlled comparison requires additional decision features. |

## ADR-004 — Use exact response caching with compatibility checks

| Field | Record |
|---|---|
| **Decision date** | 04-10-2026 |
| **Status and scope** | **Accepted**; M4; exact response reuse and cache isolation. |
| **Context and decision drivers** | Repeated compatible requests should avoid unnecessary inference while preserving request and identity boundaries. |
| **Options considered** | **(a)** No response cache: simplest reuse safety, but every repeat incurs inference. **(b)** Exact request-derived keys and compatibility checks. **(c)** Approximate semantic matching only: broader reuse, with more uncertainty about equivalence. |
| **Decision and rationale** | We use **(b)** as the first cache check. Request context, identity and relevant options contribute to exact lookup and stored provenance checks; private or explicitly bypassed requests follow the applicable no-cache rules. |
| **Consequences and trade-offs** | Compatible exact hits can avoid an upstream call, while strict keys may reduce reuse. Incorrect or incomplete metadata can cause unsafe reuse. Changing key semantics requires cache invalidation/versioning and renewed isolation tests. Later semantic lookup is recorded in [ADR-009](#adr-009--extend-caching-with-redis-vector-semantic-search). |
| **Evidence and validation** | [Exact cache](../adaroute/cache/exact.py), [cache tests](../tests/test_cache.py), [cache and provenance tests](../tests/test_improved_system_cache_and_provenance.py), and [live cache cases](../results/report_evidence/runs/20261121T112814.754567Z/cache_evaluation.json). |
| **Revisit trigger** | A cross-user, private or generation-incompatible request reuses a cached response, or a request option affecting response compatibility is missing from the cache contract. |

## ADR-005 — Isolate provider execution behind backend adapters

| Field | Record |
|---|---|
| **Decision date** | 04-10-2026 |
| **Status and scope** | **Accepted**; M2; local, hosted and mock execution paths. |
| **Context and decision drivers** | Local inference, external providers and test doubles differ in protocol, availability and provenance. They must operate behind the same client contract. |
| **Options considered** | **(a)** Call providers directly from policy code: fewer abstractions but tight coupling. **(b)** Use only mocks: inexpensive tests but no evidence of real execution. **(c)** Use adapters for local, hosted and mock providers, with explicit execution modes. |
| **Decision and rationale** | We use **(c)**. The midterm implementation used Ollama locally and LiteLLM for hosted calls; the current local vLLM deployment is documented separately in [ADR-010](#adr-010--serve-local-models-and-real-replicas-with-vllm). Real-only execution is intended to reject simulated or unknown-origin results. |
| **Consequences and trade-offs** | Adapters make providers replaceable and allow offline tests, but each integration must propagate errors, usage, source and finish reason correctly. Hybrid or simulated execution must remain clearly distinguishable from real inference. |
| **Evidence and validation** | [backend adapters](../adaroute/backends/), [multi-provider tests](../tests/test_multi_provider.py) and [provenance tests](../tests/test_provenance_and_completion_status.py). |
| **Revisit trigger** | A real-only request accepts a simulated or unknown-origin response, or a required provider cannot be integrated without changing selection logic. |

## ADR-006 — Schedule eligible replicas and bound failure recovery

| Field | Record |
|---|---|
| **Decision date** | 04-10-2026 |
| **Status and scope** | **Accepted**; M5–M6; schedulers, retries and circuit breaking. |
| **Context and decision drivers** | Replicas may have different workloads or become unavailable; provider timeouts and rate limits must not cause unbounded retries. |
| **Options considered** | **(a)** Always choose a fixed replica and fail on error: simple but does not use available alternatives. **(b)** Round robin only: transparent but ignores unequal load. **(c)** Retain round robin and load-aware schedulers with bounded retries and circuit breakers. |
| **Decision and rationale** | We use **(c)**: round robin provides a baseline, while least-loaded and SLO-aware choices use workload and runtime signals. Transient failures can trigger bounded retries and circuit state transitions, subject to eligibility constraints. |
| **Consequences and trade-offs** | Scheduling and recovery can be evaluated independently of model choice, but counters and metrics may be stale or process-local. Retries add latency and uncertain charges. The later move to distinct real engine replicas is recorded in [ADR-010](#adr-010--serve-local-models-and-real-replicas-with-vllm). |
| **Evidence and validation** | [Scheduler](../adaroute/policy/replica_selection.py), [retry policy](../adaroute/resilience/retry.py), [circuit breaker](../adaroute/resilience/circuit.py), [scheduler tests](../tests/test_replica_scheduling.py) and [resilience tests](../tests/test_resilience.py). |
| **Revisit trigger** | A controlled failure causes unbounded attempts or an ineligible fallback, or a repeated concurrent workload shows load-aware scheduling fails an agreed SLO more often than round robin. |

## ADR-007 — Record routing evidence and compare policy baselines

| Field | Record |
|---|---|
| **Decision date** | 04-10-2026 |
| **Status and scope** | **Accepted**; M7–M8; tracing, ledger, workloads and evaluation. |
| **Context and decision drivers** | We need to explain selected models and replicas, account for attempts, and compare policy behavior rather than relying on a final answer alone. |
| **Options considered** | **(a)** Free-text logs and manual demonstrations: quick but hard to compare. **(b)** Unit tests alone: focused but insufficient for system behavior. **(c)** Structured request/attempt records plus repeatable baseline workloads and tests. |
| **Decision and rationale** | We use **(c)**: record routing, cache, attempt, token and cost-state information and evaluate policy combinations using generated workloads. We complement simulation/component tests with live evaluations where services are available. |
| **Consequences and trade-offs** | Saved records support debugging and reproducibility, but require access and retention controls. Configured quality, inferred timing and estimated cost must not be presented as independently verified accuracy, measured TTFT or invoice reconciliation. |
| **Evidence and validation** | [Tracer](../adaroute/telemetry/tracer.py), [ledger](../adaroute/telemetry/ledger.py), [benchmark runner](../adaroute/experiments/runner.py), [event runner](../adaroute/experiments/event_runner.py), [observability tests](../tests/test_observability.py) and [evaluation results](../results/review_20261121_rerun/). |
| **Revisit trigger** | A retry lacks an attributable attempt record, reported provenance is incorrect, or the current workloads cannot distinguish policies under agreed evaluation criteria. |

## ADR-008 — Extend model selection with a unified deployment-aware pipeline

| Field | Record |
|---|---|
| **Decision date** | 03-11-2026 |
| **Status and scope** | **Accepted**; M3, M7; deployment catalog, unified selector, estimation and semantic-router policy. |
| **Context and decision drivers** | The initial model policies need deployment-aware capabilities, clearer constraint explanations, configurable objectives and a way to compare a semantic-router policy without replacing the gateway. |
| **Options considered** | **(a)** Keep only the earlier fixed policy classes: simpler but less portable and explainable. **(b)** Replace the gateway with an external or learned router: potentially capable, but changes constraint ownership and requires integration evidence. **(c)** Use a unified selector with candidate resolution, eligibility, feature snapshots, Pareto filtering and configurable scores. |
| **Decision and rationale** | We use **(c)**, retaining comparable strongest, cheapest, static and adaptive policies. The selector uses deployment-aware expected and reserved output estimates from configured priors or observed quantiles. A separate Bayesian estimator is updated for calibration but is not the selector's current candidate estimator. We also expose `vllm_semantic_router` as an optional policy. |
| **Consequences and trade-offs** | Deployment identity and score contributions are easier to inspect, but selection depends on catalog metadata and configured quality proxies. The bundled router preview is an AdaRoute heuristic; an upstream vLLM Semantic Router deployment has not been demonstrated. A recommendation must still resolve to an eligible candidate. |
| **Evidence and validation** | [Unified selector](../adaroute/selection/selector.py), [catalog](../adaroute/selection/catalog.py), [estimator](../adaroute/selection/estimator.py), [Bayesian estimator](../adaroute/workloads/bayesian_estimator.py), [router client](../adaroute/selection/vllm_router.py), [bundled service](../adaroute/services/vllm_router_service.py), [selection tests](../tests/test_unified_selection_endpoint.py) and [router tests](../tests/test_vllm_semantic_router_policy.py). |
| **Revisit trigger** | A recommendation maps to an ambiguous or ineligible deployment, or a repeated matched evaluation shows another selector consistently meets agreed quality, latency and cost criteria better. |

## ADR-009 — Extend caching with Redis vector semantic search

| Field | Record |
|---|---|
| **Decision date** | 03-11-2026 |
| **Status and scope** | **Accepted**; M3–M4; Redis-backed cache/session state and semantic response lookup. |
| **Context and decision drivers** | Exact lookup misses paraphrases; shared cache and session state are useful across requests. At the same time, the development stack must remain usable when Redis Search is unavailable. |
| **Options considered** | **(a)** Keep exact caching only: simpler but misses semantic reuse. **(b)** Search embeddings only in process memory: easy for small data but not shared. **(c)** Use Redis Search vector KNN with identity and compatibility filters, plus a local fallback. |
| **Decision and rationale** | We use **(c)** after the exact-cache check. FastEmbed produces embeddings; Redis Search retrieves nearby candidates by cosine distance, and the cache applies a similarity threshold and additional eligibility checks. Redis is also used for supported shared state, with process-local fallback where implemented. |
| **Consequences and trade-offs** | Vector search enables candidate retrieval for paraphrases, while fallback supports development without Redis. Similarity does not prove equivalence, and fallback does not provide distributed consistency. This is a response cache, not vLLM's internal KV or prefix cache; changing storage requires revalidating isolation and provenance. |
| **Evidence and validation** | [Semantic cache](../adaroute/cache/semantic.py), [Redis backend](../adaroute/cache/redis_backend.py), [session store](../adaroute/selection/session.py), [semantic-cache tests](../tests/test_semantic_cache.py) and [live cache cases](../results/report_evidence/runs/20261121T112814.754567Z/cache_evaluation.json). The recorded cases are examples, not a general false-hit rate. |
| **Revisit trigger** | A labeled safety set reveals an unacceptable false reuse, Redis Search cannot meet an agreed latency target, or multiple workers require stronger fallback consistency. |

## ADR-010 — Serve local models and real replicas with vLLM

| Field | Record |
|---|---|
| **Decision date** | 03-11-2026 |
| **Status and scope** | **Accepted**; M2, M6; CPU/GPU profiles, local serving processes and replica endpoints. |
| **Context and decision drivers** | We need OpenAI-compatible local serving and a way to test the replica scheduler against independent engines. Multiple replica labels on the same proxy do not establish separate engine queues. The project retains a single vLLM container for this setup. |
| **Options considered** | **(a)** Keep Ollama and logical replica records: familiar, but labels alone do not prove separate engines. **(b)** Run each replica in a different container: clearer container boundaries, but changes the requested topology. **(c)** Use vLLM CPU/GPU profiles and supervise separate engines with distinct internal endpoints in one GPU container. |
| **Decision and rationale** | We use **(c)**. The GPU catalog configures one Qwen3 engine and two Qwen2.5-Coder engines on distinct internal ports. The two Qwen2.5 engines serve the same model identity while keeping separate process and endpoint identities; the gateway can poll their health and available metrics. |
| **Consequences and trade-offs** | The scheduler can route to and observe independent engine processes, but they share the container and GPU and are not hardware-isolated replicas. Three engines require sufficient memory and lifecycle management; image/plugin compatibility and stale metrics remain operational concerns. A different serving runtime or container layout requires revalidating endpoints, privacy and recovery. |
| **Evidence and validation** | [CPU Compose](../docker-compose.yml), [GPU override](../docker-compose.gpu.yml), [GPU configuration](../config.gpu.yaml), [vLLM launcher](../scripts/vllm_multi_server.py), [launcher tests](../tests/test_vllm_multi_server.py), [smoke log](../results/review_20261121_rerun/replica-smoke.log) and [replica benchmark](../results/replica_eval/runs/20261121T112744.199772Z/benchmark_results.json). The smoke run observed traffic to both engines and one recovery; the small scheduler study used injected imbalance. |
| **Revisit trigger** | Either replica cannot remain independently reachable during a controlled failure test, or the available GPU cannot run the configured engines with sufficient headroom. |
