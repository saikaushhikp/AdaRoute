# Problem Statement

Build an AI gateway with cost-quality routing, safe caching and scalable model serving
 
- **Context:** Different AI workloads may require local open models, external providers, different latency/quality targets and different cost budgets. Direct calls from every application create vendor lock-in and make quotas, failover, caching, observability and GPU utilization difficult to control.
 
- **Problem:** Create a provider-independent AI gateway that fronts at least one self-hosted open model and one additional provider/mock, supports policy-based routing, usage accounting, safe caching, retries/circuit breakers and SLO-aware load balancing across model replicas. Benchmark modern high-throughput serving techniques when suitable GPUs are available.
 
- **Minimum scope / constraints:** Assume shared academic Linux/CUDA access: every implementation must have a single-GPU or CPU/simulated fallback, while multi-GPU experiments are performed only if scheduled resources are available. Cache keys must include the state/version information necessary to prevent unsafe cross-user or stale reuse.

# Expected Solution(tentative Milestones)


- **M1:** Stable AI gateway contract: Application code invokes logical model/capability aliases through one API rather than provider-specific SDK calls.

- **M2:** Multi-provider execution: At least one self-hosted open model and one external/mock provider can serve the same logical capability and can be switched without changing caller code.

- **M3:** Cost/quality routing: Requests are routed using task, quality, latency, privacy/cost or availability signals, and the routing policy is evaluated against static routing baselines.

- **M4:** Safe caching: Exact/retrieval/semantic and optionally prefix/KV reuse are implemented only with explicit cache keys/invalidation rules for user state, content version and policy-sensitive requests.

- **M5:** Resilience and quotas: Timeout, rate-limit, provider outage and quota exhaustion trigger bounded retry, fallback or circuit-breaker behavior instead of uncontrolled retry storms.

- **M6:** SLO-aware serving/load balancing: Replica queue state, request size and GPU/memory pressure inform scheduling; round-robin/least-loaded baselines are compared with the proposed policy.

- **M7:** Usage and observability: Every request records routing decision, model/provider, latency, token/usage estimate, cache outcome, failure mode and correlation trace without exposing secrets by default.

- **M8:** Reproducible performance benchmark: Mixed short/long and burst workloads report TTFT, token latency, throughput, p95/p99 latency, memory/GPU use, SLO attainment and estimated cost.

**Done when:** The same client can transparently invoke local and alternate models through one gateway, observe routing/caching/failover/quotas under injected failures, and reproduce a benchmark showing the cost-quality-latency trade-offs of the chosen serving and load-balancing policies.