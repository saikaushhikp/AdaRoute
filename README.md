# AdaRoute

**AdaRoute** is an AI gateway designed with cost-quality routing, safe caching, and scalable model serving. It addresses the challenge of serving heterogeneous LLM workloads across diverse local and external backends by decoupling serving decisions into two evaluation stages:
1. **Stage 1 — Model & Provider Selection:** Evaluates incoming request requirements (task type, quality demands, cost budget, privacy policies) to identify the appropriate model class and provider.
2. **Stage 2 — Replica Scheduling:** Evaluates real-time backend state (queue depth, memory pressure, active workload weight, latency, health) to route requests to the optimal replica instance.

---

## 1. Week 1 Incremental Progress

During Week 1, the foundational engineering infrastructure, API contracts, baseline decision policies, mock execution backends, and initial test suites were designed and implemented:

### Repository & Environment Setup
- **Packaging & Dependency Management:** Initialized `pyproject.toml` specifying project dependencies (`fastapi`, `pydantic`, `uvicorn`) and development dependencies (`pytest`, `pytest-asyncio`, `ruff`, `httpx`).
- **Reproducible Containerization:** Created `Dockerfile` based on `python:3.10-slim` ensuring a self-contained, reproducible development and test execution environment.
- **Task Automation via `justfile`:** Configured standard developer workflows (`just build`, `just test`, `just lint`, `just format`, `just dev`).
- **Source Hygiene:** Configured `.gitignore` for Python environments, byte-code, IDE settings, and test caches.

### Unified Gateway Contract (`adaroute/api.py`)
- Implemented a unified FastAPI application exposing `/v1/chat/completions`.
- Standardized caller interaction through a single gateway interface (`UnifiedRequest` -> `UnifiedResponse`), abstracting away provider-specific SDK complexities.
- Built an automated characterization stage that parses raw requests into typed `RequestContext` instances with estimated output token projections.

### Data Schemas (`adaroute/schemas.py`)
- **`RequestContext`:** Encapsulates multi-dimensional demand including `task_type`, `prompt_tokens`, `estimated_output_tokens`, `quality_requirement`, `latency_slo`, `privacy_level`, `budget`, and `tenant_id`.
- **`BackendState`:** Models runtime metrics across providers and replicas, capturing `provider`, `model`, `replica_id`, `queue_depth`, `estimated_workload`, `latency`, `memory_pressure`, `failure_rate`, and `availability`.

### Baseline Routing & Scheduling (`adaroute/routing.py`, `adaroute/experiments/baseline_routing.py`)
- **Baseline Policies:** Implemented `StrongestModelPolicy` and `CheapestModelPolicy` to serve as reference points for future adaptive policies.
- **`DeterministicRouter`:** Built a two-tier decision baseline:
  - *Hard Constraint Enforcement:* Restricts internal/confidential requests from routing to external third-party providers; drops candidates whose cost exceeds request budgets.
  - *Load-Aware Routing:* Sorts valid candidates by runtime queue depth to balance replica utilization.

### Mock Provider Infrastructure (`adaroute/backends.py`, `adaroute/experiments/mock_provider.py`)
- Implemented `MockProvider` and `MockBackend` to simulate physical model execution.
- Emulates network and inference latency delays and probabilistic output token generation (Gaussian distribution around estimated output length), enabling empirical experimentation without requiring dedicated multi-GPU hardware.

---

## 2. Milestone Mapping & Progress Status

Below is the incremental progress mapped against the project milestones (M1–M8):

| Milestone | Description | Week 1 Status | Implemented Incremental Deliverables |
| :--- | :--- | :--- | :--- |
| **M1** | **Stable AI Gateway Contract** | **Prototype Achieved** | Exposed unified `/v1/chat/completions` API endpoint in `adaroute/api.py`. Callers invoke models via a single standardized schema without provider-specific SDKs. |
| **M2** | **Multi-Provider Execution** | **Mocked / Ready** | Multi-backend support built into `DeterministicRouter` and `MockProvider`, switching transparently between local (`mock-local` / `llama-3-8b`) and external (`mock-external` / `gpt-4o`) representations. |
| **M3** | **Cost / Quality Routing** | **Baseline Implemented** | Implemented `StrongestModelPolicy`, `CheapestModelPolicy`, and constraint-based deterministic routing enforcing privacy rules and budget caps against static baselines. |
| **M4** | **Safe Caching** | **Planned (Schema Ready)** | `RequestContext` models `tenant_id` and `privacy_level` as foundational attributes required for cache isolation and safe invalidation keys in subsequent iterations. |
| **M5** | **Resilience and Quotas** | **Baseline Filtering** | Implemented health-filtering logic (`availability=False` exclusion) preventing traffic assignment to degraded replicas, with fallback HTTP 503 error handling on exhaustion. |
| **M6** | **SLO-Aware Serving & Load Balancing** | **Queue-Aware Baseline** | Integrated queue depth tracking in `DeterministicRouter` to route requests to least-loaded replicas; workload schemas account for prompt and estimated output token loads. |
| **M7** | **Usage and Observability** | **Initial Telemetry** | Unified API response tracks and returns execution metadata (`replica_id`, `provider`, `model`, `tokens_used`) without secret leakage. |
| **M8** | **Reproducible Performance Benchmark** | **Infrastructure Ready** | Docker and `justfile` test environment ensures identical test and benchmark conditions regardless of host Linux configuration. |

---

## 3. Test Suite & Test Case Matrix

Every test in the project is assigned a formal `testcase_id` identifying its milestone association and verification scope:

| Testcase ID | Milestone | Scope | Location | Test Function | Verification Invariant / Objective | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `TC_1.1` | **M1** / M3 | Unit | `tests/test_policy.py` | `test_tc_1_1_privacy_restriction` | Validates that `RequestContext` preserves `privacy_level="internal_only"` constraints for downstream routing decisions. | **Passed** |
| `TC_1.2` | **M1** / M3 | Unit | `tests/test_policy.py` | `test_tc_1_2_budget_restriction` | Validates that `RequestContext` correctly retains request budget constraints to prevent budget overruns. | **Passed** |
| `TC_1.3` | **M1** / M5 | Unit | `tests/test_policy.py` | `test_tc_1_3_unhealthy_replica_removed` | Validates that the policy engine removes unhealthy/offline (`availability=False`) replicas from candidate routing pools. | **Passed** |
| `TC_1.4` | **M1** / M6 | Integration | `adaroute/experiments/test_integration.py` | `test_tc_1_4_unified_api_routing_public_task` | Validates that unified `/v1/chat/completions` transparently routes public traffic to the least-loaded replica (`replica-1`). | **Passed** |
| `TC_1.5` | **M1** / M2 | Integration | `adaroute/experiments/test_integration.py` | `test_tc_1_5_unified_api_routing_privacy_constraint` | Validates that unified API strictly restricts internal tasks from reaching external providers, routing exclusively to local backends. | **Passed** |
| `TC_1.6` | **M1** / M3 | Integration | `adaroute/experiments/test_integration.py` | `test_tc_1_6_unified_api_budget_constraint` | Validates that unified API enforces cost constraints, rejecting high-cost external models and routing to compliant backends. | **Passed** |

---

## 4. Quickstart & Verification

All build and testing routines are managed via `just`:

```bash
# Build the reproducible container image
just build

# Run the complete test suite (Unit & Integration tests with testcase IDs)
just test

# Check code linting
just lint

# Format code
just format
```
