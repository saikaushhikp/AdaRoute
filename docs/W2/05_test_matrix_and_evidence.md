# Test Matrix & Verification Evidence (Week 2 Catalog)

## 1. Test Suite Catalog

Every test case in the AdaRoute repository is indexed with a unique `testcase_id` identifying its priority and milestone association:

| Testcase ID | Milestone | Scope | File Location | Test Function | Verified Invariant / Property |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TC_P2_01` | **P0 Correctness** | Integration | `tests/test_privacy.py` | `test_tc_p2_01_privacy_adversarial_forbidden_choice` | When external provider is cheaper/less loaded, `internal_only` request MUST NEVER call or select external provider. |
| `TC_P2_02` | **P0 Correctness** | Integration | `tests/test_privacy.py` | `test_tc_p2_02_unknown_privacy_fails_closed` | Unknown privacy labels fail closed with HTTP 400 rather than silently falling back to public. |
| `TC_P2_03` | **P0 Correctness** | Unit | `tests/test_privacy.py` | `test_tc_p2_03_unavailable_local_fails_closed_no_leak` | When local replica is offline, internal request fails safely rather than leaking to available external candidate. |
| `TC_P2_04` | **P0 Correctness** | Unit | `tests/test_privacy.py` | `test_tc_p2_04_confidential_and_restricted_enforcement` | `CONFIDENTIAL` and `RESTRICTED` privacy levels are strictly excluded from third-party cloud models. |
| `TC_P2_05` | **P0 Correctness** | Integration | `tests/test_budget.py` | `test_tc_p2_05_impossible_budget_safely_rejected` | When budget ceiling is lower than any eligible candidate cost, request is safely rejected with HTTP 503. |
| `TC_P2_06` | **P0 Correctness** | Integration | `tests/test_budget.py` | `test_tc_p2_06_negative_budget_validation_error` | Negative budget amounts fail validation immediately with HTTP 400. |
| `TC_P2_07` | **P0 Correctness** | Unit | `tests/test_budget.py` | `test_tc_p2_07_budget_differentiates_model_tier` | Intermediate budget permits local cheap model ($0.001/k) while correctly filtering expensive cloud model ($0.015/k). |
| `TC_M1_01` | **M1 Contract** | Integration | `tests/test_contract.py` | `test_tc_m1_01_logical_capability_alias_routing` | Single unified API `/v1/chat/completions` accepts logical capability aliases (`model="coding"`, `model="general"`). |
| `TC_M1_02` | **M1 Contract** | Integration | `tests/test_contract.py` | `test_tc_m1_02_standard_chat_messages_contract` | Standard OpenAI messages schema (`[{"role": "...", "content": "..."}]`) is accepted and executed. |
| `TC_M2_01` | **M2 Multi-Provider**| Integration | `tests/test_multi_provider.py` | `test_tc_m2_01_local_model_execution` | Local open model backend executes successfully and returns valid responses. |
| `TC_M2_02` | **M2 Multi-Provider**| Integration | `tests/test_multi_provider.py` | `test_tc_m2_02_alternate_provider_execution` | Alternate/mock cloud provider executes successfully when eligible. |
| `TC_M2_03` | **M2 Multi-Provider**| Integration | `tests/test_multi_provider.py` | `test_tc_m2_03_transparent_provider_switch_same_client` | Same client payload switches transparently between local and cloud providers based on policy constraints. |
| `TC_M3_01` | **M3 Baselines** | Unit | `tests/test_model_selection.py` | `test_tc_m3_01_strongest_distinct_from_cheapest` | `StrongestModelPolicy` and `CheapestModelPolicy` are distinct; Strongest picks GPT-4o while Cheapest picks local model. |
| `TC_M3_02` | **M3 Baselines** | Unit | `tests/test_model_selection.py` | `test_tc_m3_02_static_rule_follows_task_mapping` | `StaticRulePolicy` deterministically routes coding to external cloud and QA to local model. |
| `TC_M3_03` | **M3 Baselines** | Unit | `tests/test_model_selection.py` | `test_tc_m3_03_adaptive_policy_balances_objectives` | `AdaptiveModelPolicy` balances cost vs quality weights in multi-objective scoring function $J(m)$. |
| `TC_M4_01` | **M4 Safe Cache** | Integration | `tests/test_cache.py` | `test_tc_m4_01_repeat_request_cache_hit` | Repeated identical request within same tenant scope hits cache and returns cached payload. |
| `TC_M4_02` | **M4 Safe Cache** | Integration | `tests/test_cache.py` | `test_tc_m4_02_cross_tenant_cache_isolation` | Tenant isolation invariant: Tenant B cannot retrieve Tenant A's cached responses. |
| `TC_M4_03` | **M4 Safe Cache** | Integration | `tests/test_cache.py` | `test_tc_m4_03_no_cache_bypass` | Requests with `no_cache=True` bypass cache lookup and write. |
| `TC_M5_01` | **M5 Resilience** | Integration | `tests/test_resilience.py` | `test_tc_m5_01_bounded_retry_on_replica_failure` | Primary replica transient error triggers bounded retry on alternate healthy replica. |
| `TC_M5_02` | **M5 Resilience** | Unit | `tests/test_resilience.py` | `test_tc_m5_02_circuit_breaker_state_transitions` | Circuit breaker transitions CLOSED $\to$ OPEN $\to$ HALF-OPEN $\to$ CLOSED, preventing cascading retry storms. |
| `TC_M6_01` | **M6 Scheduler** | Unit | `tests/test_replica_scheduling.py` | `test_tc_m6_01_multiple_replicas_exist_for_same_model` | At least two replicas exist for local model (`local-r1`, `local-r2`) enabling within-model scheduling. |
| `TC_M6_02` | **M6 Scheduler** | Unit | `tests/test_replica_scheduling.py` | `test_tc_m6_02_dynamic_workload_weighted_queue_tracking`| Dynamic workload-weighted queue $W_q$ increments on dispatch and decrements on completion. |
| `TC_M6_03` | **M6 Scheduler** | Unit | `tests/test_replica_scheduling.py` | `test_tc_m6_03_least_loaded_and_slo_aware_selection` | Scheduler selects least-loaded replica when one is artificially saturated. |
| `TC_M7_01` | **M7 Telemetry** | Integration | `tests/test_observability.py` | `test_tc_m7_01_complete_structured_trace_record` | Every request produces a full correlation event trace record with attempt counts, latency, and costs. |
| `TC_M7_02` | **M7 Telemetry** | Integration | `tests/test_observability.py` | `test_tc_m7_02_secret_prompt_redaction_in_traces` | Raw prompts and proprietary secrets are excluded from tracer records. |
| `TC_M8_01` | **M8 Benchmark** | Unit | `tests/test_benchmark.py` | `test_tc_m8_01_independent_workload_and_estimator_error_tracking` | Decoupled ground truth workload generation enables non-circular estimation error ($e = \|\hat{L} - L\|$) tracking. |
| `TC_M8_02` | **M8 Benchmark** | Unit | `tests/test_benchmark.py` | `test_tc_m8_02_reproducible_benchmark_execution` | Fixed seed produces identical benchmark metrics across independent runs, saved as machine-readable JSON. |
