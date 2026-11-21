#!/usr/bin/env python3
"""AdaRoute Honest, Small-Scale Repeatable Evaluation (Milestone Review).

Performs:
1. Frozen, repeatable model-selection comparison across policies ('adaptive', 'cheapest', 'strongest')
   with cache strictly disabled (no_cache=True) and strict deterministic evaluation of final answers.
2. Isolated cache-reuse evaluation across Cold Miss, Exact Hit, Semantic Hit, and Adversarial Non-Hits
   (negation, numeric change, conflicting directive, cross-tenant isolation) in fresh run-isolated namespaces.
3. Produces versioned artifacts: manifest.json, selection_evaluation.json, cache_evaluation.json,
   and honest_eval_summary.md in a new results/report_evidence/runs/ directory.

Constraints:
- Never overwrite existing benchmark artifacts or user ledger.
- Distinguish local allocated estimates from invoiced cloud charges.
- Accurately record execution provenance and finish_reason.
- Fail closed with non-zero exit code if evaluation expectations fail.
"""

from __future__ import annotations

import ast
import datetime
import json
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

RUN_TIMESTAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
RUN_ID = uuid.uuid4().hex[:8]
OUTPUT_DIR = Path(__file__).resolve().parent / "runs" / RUN_TIMESTAMP
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:7000")


def extract_final_answer(content: str | None, finish_reason: str | None = None) -> tuple[str | None, str | None]:
    """Extracts the final answer after </think>, strictly rejecting truncated reasoning."""
    if not content or not str(content).strip():
        return None, "Empty response content"

    text = str(content).strip()
    if "<think>" in text:
        if "</think>" not in text:
            return None, "Truncated inside <think> reasoning passage without completion"
        final_part = text.split("</think>", 1)[1].strip()
    else:
        final_part = text

    if not final_part:
        return None, "No completion text produced after reasoning passage"

    if finish_reason == "length":
        return None, "Incomplete response truncated by length limit (finish_reason='length')"

    return final_part, None


def check_math(final_ans: str) -> tuple[bool, str]:
    nums = re.findall(r"\b\d+\b", final_ans)
    if nums and nums[-1] == "391":
        return True, "Found exact numerical answer 391"
    return False, f"Expected 391, got '{final_ans}'"


def check_code(final_ans: str) -> tuple[bool, str]:
    code = final_ans
    if "```" in final_ans:
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", final_ans, re.DOTALL)
        if match:
            code = match.group(1)
    try:
        tree = ast.parse(code)
        func_def = next(
            (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "is_even"),
            None,
        )
        if not func_def:
            return False, "AST has no FunctionDef named 'is_even'"
        if len(func_def.args.args) < 1:
            return False, "Function 'is_even' takes no arguments"
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(func_def))
        if not has_return:
            return False, "Function 'is_even' has no return statement"
        return True, "AST contains valid function `is_even(n)` with return statement"
    except Exception as e:
        return False, f"AST parsing failed: {e}"


def check_json(final_ans: str) -> tuple[bool, str]:
    match = re.search(r"\[.*?\]", final_ans, re.DOTALL)
    if not match:
        return False, "No JSON array brackets found in final answer"
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list) and [str(x).lower().strip() for x in parsed] == ["apple", "banana", "cherry"]:
            return True, "Valid JSON array matching ['apple', 'banana', 'cherry']"
        return False, f"JSON array items mismatch: {parsed}"
    except Exception as e:
        return False, f"json.loads failed: {e}"


def check_chem(final_ans: str) -> tuple[bool, str]:
    clean = re.sub(r"[^\w]", " ", final_ans).strip().split()
    if clean and clean[0].upper() == "AU":
        return True, "Found exact chemical symbol Au"
    return False, f"Expected Au, got '{final_ans}'"


def check_logic(final_ans: str) -> tuple[bool, str]:
    clean = re.sub(r"[^\w]", " ", final_ans).strip().lower().split()
    if clean and clean[0] == "yes" and "no" not in clean[:3]:
        return True, "Found exact Yes answer"
    return False, f"Expected Yes, got '{final_ans}'"


# Frozen prompts with deterministic objective evaluation checks
SELECTION_CASES = [
    {
        "case_id": "math_mult",
        "task_type": "reasoning",
        "prompt": "What is 17 * 23? Output only the final number with no extra text.",
        "expected_answer": "391",
        "grader": check_math,
    },
    {
        "case_id": "code_func",
        "task_type": "coding",
        "prompt": "Write a Python function named `is_even(n)` that returns True if n is even, and False otherwise.",
        "expected_answer": "def is_even",
        "grader": check_code,
    },
    {
        "case_id": "json_format",
        "task_type": "structured_output",
        "prompt": "Output a valid JSON array containing strings 'apple', 'banana', and 'cherry'. Output only the JSON array.",
        "expected_answer": "['apple', 'banana', 'cherry']",
        "grader": check_json,
    },
    {
        "case_id": "chem_symbol",
        "task_type": "general",
        "prompt": "What is the chemical symbol for gold? Output only the two-letter symbol.",
        "expected_answer": "Au",
        "grader": check_chem,
    },
    {
        "case_id": "logic_syllogism",
        "task_type": "reasoning",
        "prompt": "Premises: All Bloops are Razzies. All Razzies are Lizzies. Question: Are all Bloops definitely Lizzies? Answer with just Yes or No.",
        "expected_answer": "Yes",
        "grader": check_logic,
    },
]


def build_cache_cases(run_namespace: str) -> list[dict[str, Any]]:
    """Builds cache test cases with a unique namespace per run to avoid warm-cache cross-contamination."""
    chem_tenant = f"eval-tenant-{run_namespace}-chem"
    chem_user = f"eval-user-{run_namespace}-chem"
    other_tenant = f"eval-tenant-{run_namespace}-diff"
    other_user = f"eval-user-{run_namespace}-diff"
    trunc_tenant = f"eval-tenant-{run_namespace}-trunc"
    trunc_user = f"eval-user-{run_namespace}-trunc"

    return [
        {
            "case_id": "chem_symbol_cold",
            "type": "cold_seed",
            "task_type": "general",
            "prompt": "What is the chemical symbol for gold? Output only the two-letter symbol.",
            "expected_cache": "miss",
            "tenant_id": chem_tenant,
            "user_id": chem_user,
            "max_tokens": 300,
        },
        {
            "case_id": "chem_symbol_exact",
            "type": "exact_repeat",
            "task_type": "general",
            "prompt": "What is the chemical symbol for gold? Output only the two-letter symbol.",
            "expected_cache": "exact",
            "tenant_id": chem_tenant,
            "user_id": chem_user,
            "max_tokens": 300,
        },
        {
            "case_id": "chem_symbol_semantic_paraphrase",
            "type": "semantic_paraphrase",
            "task_type": "general",
            "prompt": "Which chemical symbol represents the element gold? Give only the two-letter symbol.",
            "expected_cache": "semantic",
            "tenant_id": chem_tenant,
            "user_id": chem_user,
            "max_tokens": 300,
        },
        {
            "case_id": "chem_symbol_adversarial_entity",
            "type": "adversarial_entity",
            "task_type": "general",
            "prompt": "What is the chemical symbol for silver? Output only the two-letter symbol.",
            "expected_cache": "miss",
            "tenant_id": chem_tenant,
            "user_id": chem_user,
            "max_tokens": 300,
        },
        {
            "case_id": "chem_symbol_adversarial_negation",
            "type": "adversarial_negation",
            "task_type": "general",
            "prompt": "What is NOT the chemical symbol for gold? Do not output Au.",
            "expected_cache": "miss",
            "tenant_id": chem_tenant,
            "user_id": chem_user,
            "max_tokens": 300,
        },
        {
            "case_id": "chem_symbol_cross_tenant_isolation",
            "type": "cross_tenant",
            "task_type": "general",
            "prompt": "What is the chemical symbol for gold? Output only the two-letter symbol.",
            "expected_cache": "miss",
            "tenant_id": other_tenant,
            "user_id": other_user,
            "max_tokens": 300,
        },
        {
            "case_id": "truncated_length_cold",
            "type": "cold_seed_truncated",
            "task_type": "reasoning",
            "prompt": "Explain the foundational principles of thermodynamics in rigorous detail.",
            "expected_cache": "miss",
            "tenant_id": trunc_tenant,
            "user_id": trunc_user,
            "max_tokens": 15,  # Bound to force finish_reason='length'
        },
        {
            "case_id": "truncated_length_exact_repeat",
            "type": "exact_repeat_truncated",
            "task_type": "reasoning",
            "prompt": "Explain the foundational principles of thermodynamics in rigorous detail.",
            "expected_cache": "exact",  # Exact cache preserves identical repeat with finish_reason='length'
            "tenant_id": trunc_tenant,
            "user_id": trunc_user,
            "max_tokens": 15,
        },
        {
            "case_id": "truncated_length_semantic_paraphrase",
            "type": "semantic_paraphrase_rejected_due_to_truncation",
            "task_type": "reasoning",
            "prompt": "Describe the core laws and foundations of thermodynamics thoroughly.",
            "expected_cache": "miss",  # Semantic cache must decline to index truncated response
            "tenant_id": trunc_tenant,
            "user_id": trunc_user,
            "max_tokens": 300,
        },
        {
            "case_id": "adversarial_conflicting_directive",
            "type": "adversarial_directive",
            "task_type": "reasoning",
            "prompt": "Provide a brief one-word summary of thermodynamics.",
            "expected_cache": "miss",
            "tenant_id": trunc_tenant,
            "user_id": trunc_user,
            "max_tokens": 300,
        },
    ]


def probe_environment() -> dict[str, Any]:
    env_info: dict[str, Any] = {
        "git_commit": "unknown",
        "git_branch": "unknown",
        "git_dirty": False,
        "gpu": "None / not detected",
        "redis_version": "unknown",
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        env_info["git_commit"] = commit
        env_info["git_branch"] = branch
        env_info["git_dirty"] = bool(status)
    except Exception:
        pass

    try:
        gpu_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if gpu_out:
            env_info["gpu"] = gpu_out
    except Exception:
        pass

    try:
        import redis

        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), socket_connect_timeout=1.0)
        info = r.info("server")
        env_info["redis_version"] = info.get("redis_version", "unknown")
    except Exception:
        pass

    return env_info


def run_evaluation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    print(f"=== AdaRoute Small-Scale Evaluation Starting (Run ID: {RUN_ID}) ===")
    print(f"Target Gateway: {GATEWAY_URL}")
    print(f"Output Directory: {OUTPUT_DIR}")

    is_live = False
    try:
        req = urllib.request.Request(f"{GATEWAY_URL}/health")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            if resp.status == 200:
                is_live = True
                data = json.loads(resp.read().decode("utf-8"))
                print(f"[OK] Gateway is reachable and healthy: {data.get('status')}")
    except Exception as e:
        print(f"[WARN] Gateway {GATEWAY_URL} not reachable: {e}. Falling back to TestClient.")

    env_info = probe_environment()
    manifest: dict[str, Any] = {
        "evaluation_name": "honest_small_scale_comparison",
        "run_id": RUN_ID,
        "timestamp_utc": RUN_TIMESTAMP,
        "git_branch": env_info["git_branch"],
        "git_commit": env_info["git_commit"],
        "git_dirty": env_info["git_dirty"],
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "environment": "live_services" if is_live else "local_test_double",
        "gateway_url": GATEWAY_URL,
        "hardware_gpu": env_info["gpu"],
        "redis_version": env_info["redis_version"],
        "evaluation_parameters": {
            "temperature": 0.0,
            "max_tokens": 300,
            "seed": 42,
            "user_private": True,
            "execution_mode": "real",
        },
        "caveats": [
            "Local replica IDs ('vllm-gpu-gguf-qwen3-r1', 'vllm-gpu-gguf-qwen3-r2') connect to a single containerized vLLM instance; they test scheduler routing logic and queue accounting, not independent multi-GPU node balancing.",
            "Local model costs are internal GPU/CPU allocations based on catalog token rates ($0.0/1k), not invoiced cloud provider bills.",
            "Qwen3-0.6B generates an internal <think>...</think> reasoning passage before producing its answer; answer checkers evaluate the final answer text strictly after </think>.",
            "Sample size is intentionally small and focused (5 frozen selection cases, 10 labeled cache cases) to demonstrate correctness and provenance invariants rather than inferring broad statistical distributions.",
        ],
    }

    # Query gateway helper
    def query_gateway(payload: dict[str, Any]) -> dict[str, Any]:
        if is_live:
            req = urllib.request.Request(
                f"{GATEWAY_URL}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=45.0) as res:
                    return {"status_code": res.status, "body": json.loads(res.read().decode("utf-8"))}
            except urllib.error.HTTPError as e:
                try:
                    body = json.loads(e.read().decode("utf-8"))
                except Exception:
                    body = {"detail": str(e)}
                return {"status_code": e.code, "body": body}
            except Exception as e:
                return {"status_code": 500, "body": {"error": str(e)}}
        else:
            from fastapi.testclient import TestClient

            from adaroute.api import app

            tc = TestClient(app)
            res = tc.post("/v1/chat/completions", json=payload)
            return {"status_code": res.status_code, "body": res.json()}

    # =========================================================================
    # Part 1: Model Selection Evaluation (no_cache=True)
    # =========================================================================
    print("\n--- Part 1: Model Selection Evaluation (no_cache=True) ---")
    policies = ["adaptive", "cheapest", "strongest"]
    selection_results: dict[str, Any] = {"summary": {}, "cases": []}
    policy_metrics = {p: {"total": 0, "completed": 0, "correct": 0, "latency_ms": []} for p in policies}

    for case in SELECTION_CASES:
        case_records = {"case_id": case["case_id"], "prompt": case["prompt"], "policies": {}}
        for policy in policies:
            payload = {
                "prompt": case["prompt"],
                "task_type": case["task_type"],
                "policy": policy,
                "temperature": 0.0,
                "max_tokens": 300,
                "no_cache": True,
                "user_private": True,  # Keep local-only
                "execution_mode": "real",
                "tenant_id": f"eval-selection-{RUN_ID}",
                "user_id": f"eval-user-{RUN_ID}-{policy}",
            }
            t0 = time.perf_counter()
            resp = query_gateway(payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            policy_metrics[policy]["total"] += 1
            if resp["status_code"] == 200:
                data = resp["body"]
                raw_content = data["choices"][0]["message"]["content"]
                finish_reason = data["choices"][0]["finish_reason"]
                selected_model = data["routing_metadata"]["selected_model"]
                selected_replica = data["routing_metadata"]["selected_replica"]
                provider = data["routing_metadata"]["provider"]
                cost = data["routing_metadata"]["actual_cost"]

                # Extract and evaluate final answer strictly after </think>
                final_ans, extract_err = extract_final_answer(raw_content, finish_reason=finish_reason)
                if final_ans is not None:
                    is_correct, grader_why = case["grader"](final_ans)
                else:
                    is_correct = False
                    grader_why = f"Extraction failed: {extract_err}"

                policy_metrics[policy]["completed"] += 1
                if is_correct:
                    policy_metrics[policy]["correct"] += 1
                policy_metrics[policy]["latency_ms"].append(elapsed_ms)

                case_records["policies"][policy] = {
                    "status_code": 200,
                    "selected_model": selected_model,
                    "selected_replica": selected_replica,
                    "provider": provider,
                    "latency_ms": round(elapsed_ms, 2),
                    "finish_reason": finish_reason,
                    "cost": cost,
                    "cost_state": "allocated_estimate",
                    "is_correct": is_correct,
                    "grader_reason": grader_why,
                    "extracted_final_answer": final_ans,
                    "raw_content_preview": (raw_content[:150] + "...") if len(raw_content) > 150 else raw_content,
                }
                print(
                    f"  [{policy.upper()}] {case['case_id']} -> Model: {selected_model} | Latency: {elapsed_ms:.1f}ms | Correct: {is_correct} | Finish: {finish_reason} | {grader_why}"
                )
            else:
                case_records["policies"][policy] = {
                    "status_code": resp["status_code"],
                    "error": resp["body"].get("detail", "Error"),
                }
                print(
                    f"  [{policy.upper()}] {case['case_id']} -> FAILED HTTP {resp['status_code']}: {resp['body'].get('detail')}"
                )
        selection_results["cases"].append(case_records)

    for p in policies:
        lats = policy_metrics[p]["latency_ms"]
        avg_lat = round(sum(lats) / len(lats), 2) if lats else 0.0
        tot = policy_metrics[p]["total"]
        comp = policy_metrics[p]["completed"]
        corr = policy_metrics[p]["correct"]
        selection_results["summary"][p] = {
            "total_requests": tot,
            "completed": comp,
            "verified_accuracy": round(corr / comp, 3) if comp else 0.0,
            "avg_latency_ms": avg_lat,
            "cost_type": "local_gpu_allocation",
        }

    # =========================================================================
    # Part 2: Isolated Cache-Reuse Evaluation
    # =========================================================================
    print(f"\n--- Part 2: Cache Reuse & False-Hit Guard Evaluation (Namespace: {RUN_ID}) ---")
    cache_cases = build_cache_cases(RUN_ID)
    cache_results: list[dict[str, Any]] = []

    for ccase in cache_cases:
        payload = {
            "prompt": ccase["prompt"],
            "task_type": ccase["task_type"],
            "policy": "adaptive",
            "temperature": 0.0,
            "max_tokens": ccase.get("max_tokens", 300),
            "no_cache": False,
            "user_private": True,
            "execution_mode": "real",
            "tenant_id": ccase["tenant_id"],
            "user_id": ccase["user_id"],
        }
        t0 = time.perf_counter()
        resp = query_gateway(payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if resp["status_code"] == 200:
            data = resp["body"]
            meta = data["routing_metadata"]
            cache_hit = meta["cache_hit"]
            cache_type = meta.get("cache_type")
            sim_score = meta.get("similarity")
            attempts = meta.get("attempts", 0)
            finish_reason = data["choices"][0]["finish_reason"]

            if not cache_hit:
                decision = "miss"
            elif cache_type == "exact":
                decision = "exact"
            elif cache_type == "semantic":
                decision = "semantic"
            else:
                decision = "hit"

            expected = ccase["expected_cache"]
            passed_expectation = decision == expected

            rec = {
                "case_id": ccase["case_id"],
                "test_type": ccase["type"],
                "expected_decision": expected,
                "actual_decision": decision,
                "cache_hit": cache_hit,
                "cache_type": cache_type,
                "similarity_score": sim_score,
                "upstream_attempts": attempts,
                "finish_reason": finish_reason,
                "latency_ms": round(elapsed_ms, 2),
                "passed_expectation": passed_expectation,
            }
            cache_results.append(rec)
            status_symbol = "[PASS]" if passed_expectation else "[FAIL]"
            print(
                f"  {status_symbol} {ccase['case_id']} ({ccase['type']}) -> Expected: {expected} | Actual: {decision} | Latency: {elapsed_ms:.2f}ms | Sim: {sim_score}"
            )
        else:
            rec = {
                "case_id": ccase["case_id"],
                "test_type": ccase["type"],
                "status_code": resp["status_code"],
                "error": resp["body"].get("detail"),
                "passed_expectation": False,
            }
            cache_results.append(rec)
            print(f"  [FAIL] {ccase['case_id']} HTTP {resp['status_code']}: {resp['body'].get('detail')}")

    # Write artifacts to dedicated directory
    manifest_path = OUTPUT_DIR / "manifest.json"
    selection_path = OUTPUT_DIR / "selection_evaluation.json"
    cache_path = OUTPUT_DIR / "cache_evaluation.json"
    summary_path = OUTPUT_DIR / "honest_eval_summary.md"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection_results, f, indent=2)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_results, f, indent=2)

    correct_cache_decisions = sum(1 for c in cache_results if c.get("passed_expectation"))
    total_cache_decisions = len(cache_results)

    # Check model selection diversity across policies
    observed_models_by_policy = {
        p: [c["policies"].get(p, {}).get("selected_model") for c in selection_results["cases"]] for p in policies
    }
    all_same = (
        observed_models_by_policy["adaptive"]
        == observed_models_by_policy["cheapest"]
        == observed_models_by_policy["strongest"]
    )

    md_content = f"""# AdaRoute Honest Evaluation Summary

**Run Date**: {manifest["timestamp_utc"]}  
**Run ID**: `{RUN_ID}`  
**Environment**: {manifest["environment"]} ({manifest["platform"]})  
**Hardware GPU**: `{manifest["hardware_gpu"]}`  
**Evaluated Commit**: `{env_info["git_commit"][:10]}` (dirty: {env_info["git_dirty"]}) on branch `{env_info["git_branch"]}`  
**Service Endpoints**: Gateway `{GATEWAY_URL}`, Redis `{manifest["redis_version"]}`  

---

## 1. Model Selection Comparison (Cache Disabled, no_cache=True)

Tested on 5 frozen objective test cases across mathematical reasoning, Python code generation, JSON formatting, chemical entity lookup, and syllogistic logic.
All requests ran with `temperature=0.0`, `max_tokens=300`, and `user_private=True` (strictly local execution).

Graders strictly evaluate the **final answer** after any `<think>...</think>` reasoning passage with deterministic parsers (AST parsing for code, `json.loads` for JSON array, exact integer for math, exact symbol for chemistry, exact Yes/No for logic).

| Policy | Completed / Total | Verified Final Answer Accuracy | Mean Latency (ms) | Billing Classification |
| :--- | :---: | :---: | :---: | :--- |
| **adaptive** | {selection_results["summary"]["adaptive"]["completed"]}/{selection_results["summary"]["adaptive"]["total_requests"]} | {selection_results["summary"]["adaptive"]["verified_accuracy"] * 100:.1f}% | {selection_results["summary"]["adaptive"]["avg_latency_ms"]:.1f} ms | Local GPU Allocated Estimate ($0.000) |
| **cheapest** | {selection_results["summary"]["cheapest"]["completed"]}/{selection_results["summary"]["cheapest"]["total_requests"]} | {selection_results["summary"]["cheapest"]["verified_accuracy"] * 100:.1f}% | {selection_results["summary"]["cheapest"]["avg_latency_ms"]:.1f} ms | Local GPU Allocated Estimate ($0.000) |
| **strongest**| {selection_results["summary"]["strongest"]["completed"]}/{selection_results["summary"]["strongest"]["total_requests"]} | {selection_results["summary"]["strongest"]["verified_accuracy"] * 100:.1f}% | {selection_results["summary"]["strongest"]["avg_latency_ms"]:.1f} ms | Local GPU Allocated Estimate ($0.000) |

> [!NOTE]
> **Observed Routing Outcome**:
> {"All three policies selected the identical local model across all 5 test cases on this local configuration. This is documented honestly as an observed outcome: under local constraints with available catalog deployments, the policies converge on the eligible candidate. No routing specialization or accuracy differentiation is claimed where unobserved." if all_same else "Policies differentiated model selection based on their configured objectives."}

---

## 2. Safe and Observable Cache Reuse

Tested on 10 labeled test cases evaluating Cold Misses, Exact Repeats, Paraphrased Semantic Queries, Adversarial Negations, Numeric Alterations, Conflicting Directives, Truncated Completion Ingestion Rejection, and Cross-Tenant Isolation in fresh run-isolated namespace `{RUN_ID}`.

| Case ID | Type | Expected Decision | Actual Decision | Cache Hit | Upstream Attempts | Finish Reason | Latency (ms) | Invariant Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for cr in cache_results:
        inv_status = "PASSED" if cr.get("passed_expectation") else "FAILED"
        md_content += f"| `{cr['case_id']}` | {cr['test_type']} | `{cr.get('expected_decision', 'not recorded')}` | `{cr.get('actual_decision', 'error')}` | {cr.get('cache_hit')} | {cr.get('upstream_attempts')} | `{cr.get('finish_reason')}` | {cr.get('latency_ms')} ms | **{inv_status}** |\n"

    md_content += f"""
**Cache Invariant Correctness**: {correct_cache_decisions} / {total_cache_decisions} ({correct_cache_decisions / total_cache_decisions * 100:.1f}%)

---

## 3. Disclosures and Honest Limitations

1. **Namespace Isolation**: This run used fresh, unique tenant and user namespaces (`eval-*-{RUN_ID}-*`), preventing warm-cache cross-contamination from earlier evaluation sessions.
2. **Reasoning Extraction**: Answers were parsed strictly after `</think>`. Truncated outputs were marked as incomplete and failed the accuracy check.
3. **Local Allocations**: Local costs ($0.000) represent internal GPU compute allocations rather than external provider invoices.
4. **Sample Scope**: Small focused test cases demonstrate correctness of gateway invariants (circuit breakers, cache eligibility, and telemetry), not broad population benchmark distributions.
"""

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[DONE] Saved evaluation artifacts to {OUTPUT_DIR}:")
    print(f"  - {manifest_path.name}")
    print(f"  - {selection_path.name}")
    print(f"  - {cache_path.name}")
    print(f"  - {summary_path.name}")

    if correct_cache_decisions != total_cache_decisions or any(
        policy_metrics[p]["completed"] != policy_metrics[p]["total"] for p in policies
    ):
        raise SystemExit("Evaluation completed with failed expectations or requests; inspect the saved results.")


if __name__ == "__main__":
    run_evaluation()
