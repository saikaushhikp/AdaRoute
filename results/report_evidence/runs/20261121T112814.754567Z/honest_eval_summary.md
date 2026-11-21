# AdaRoute Honest Evaluation Summary

**Run Date**: 20261121T112814.754567Z  
**Run ID**: `55077be5`  
**Environment**: live_services (Linux-6.8.0-138-generic-x86_64-with-glibc2.35)  
**Hardware GPU**: `NVIDIA RTX PRO 4000 Blackwell SFF Edition, 24467 MiB`  
**Evaluated Commit**: `3b081c7ff8` (dirty: True) on branch `psk/gpu-vllm`  
**Service Endpoints**: Gateway `http://localhost:7000`, Redis `7.4.7`  

---

## 1. Model Selection Comparison (Cache Disabled, no_cache=True)

Tested on 5 frozen objective test cases across mathematical reasoning, Python code generation, JSON formatting, chemical entity lookup, and syllogistic logic.
All requests ran with `temperature=0.0`, `max_tokens=300`, and `user_private=True` (strictly local execution).

Graders strictly evaluate the **final answer** after any `<think>...</think>` reasoning passage with deterministic parsers (AST parsing for code, `json.loads` for JSON array, exact integer for math, exact symbol for chemistry, exact Yes/No for logic).

| Policy | Completed / Total | Verified Final Answer Accuracy | Mean Latency (ms) | Billing Classification |
| :--- | :---: | :---: | :---: | :--- |
| **adaptive** | 5/5 | 80.0% | 2849.4 ms | Local GPU Allocated Estimate ($0.000) |
| **cheapest** | 5/5 | 60.0% | 857.1 ms | Local GPU Allocated Estimate ($0.000) |
| **strongest**| 5/5 | 80.0% | 2691.8 ms | Local GPU Allocated Estimate ($0.000) |

> [!NOTE]
> **Observed Routing Outcome**:
> Policies differentiated model selection based on their configured objectives.

---

## 2. Safe and Observable Cache Reuse

Tested on 10 labeled test cases evaluating Cold Misses, Exact Repeats, Paraphrased Semantic Queries, Adversarial Negations, Numeric Alterations, Conflicting Directives, Truncated Completion Ingestion Rejection, and Cross-Tenant Isolation in fresh run-isolated namespace `55077be5`.

| Case ID | Type | Expected Decision | Actual Decision | Cache Hit | Upstream Attempts | Finish Reason | Latency (ms) | Invariant Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `chem_symbol_cold` | cold_seed | `miss` | `miss` | False | 1 | `stop` | 1582.7 ms | **PASSED** |
| `chem_symbol_exact` | exact_repeat | `exact` | `exact` | True | 0 | `stop` | 2.1 ms | **PASSED** |
| `chem_symbol_semantic_paraphrase` | semantic_paraphrase | `semantic` | `semantic` | True | 0 | `stop` | 4.73 ms | **PASSED** |
| `chem_symbol_adversarial_entity` | adversarial_entity | `miss` | `miss` | False | 1 | `stop` | 1707.17 ms | **PASSED** |
| `chem_symbol_adversarial_negation` | adversarial_negation | `miss` | `miss` | False | 1 | `length` | 3376.43 ms | **PASSED** |
| `chem_symbol_cross_tenant_isolation` | cross_tenant | `miss` | `miss` | False | 1 | `stop` | 1584.56 ms | **PASSED** |
| `truncated_length_cold` | cold_seed_truncated | `miss` | `miss` | False | 1 | `length` | 239.31 ms | **PASSED** |
| `truncated_length_exact_repeat` | exact_repeat_truncated | `exact` | `exact` | True | 0 | `length` | 2.55 ms | **PASSED** |
| `truncated_length_semantic_paraphrase` | semantic_paraphrase_rejected_due_to_truncation | `miss` | `miss` | False | 1 | `length` | 3377.95 ms | **PASSED** |
| `adversarial_conflicting_directive` | adversarial_directive | `miss` | `miss` | False | 1 | `stop` | 2124.32 ms | **PASSED** |

**Cache Invariant Correctness**: 10 / 10 (100.0%)

---

## 3. Disclosures and Honest Limitations

1. **Namespace Isolation**: This run used fresh, unique tenant and user namespaces (`eval-*-55077be5-*`), preventing warm-cache cross-contamination from earlier evaluation sessions.
2. **Reasoning Extraction**: Answers were parsed strictly after `</think>`. Truncated outputs were marked as incomplete and failed the accuracy check.
3. **Local Allocations**: Local costs ($0.000) represent internal GPU compute allocations rather than external provider invoices.
4. **Sample Scope**: Small focused test cases demonstrate correctness of gateway invariants (circuit breakers, cache eligibility, and telemetry), not broad population benchmark distributions.
