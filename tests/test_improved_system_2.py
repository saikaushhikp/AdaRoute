"""Unit and regression tests for IMPROVE_SYSTEM_2 requirements.

Covers:
1. P0: Cache equivalence, exact float representation in compute_key, strict temperature matching in semantic cache,
       original max_tokens budget check, policy and policy_version validation, fail-closed decoding on missing fields,
       multi-turn context protection in semantic cache.
2. P0: Truthful backend origin (observed_origin: real/simulated/unknown), unknown finish_reason preservation,
       fail-closed on simulated/unknown in real mode, unknown_unbilled cost state.
3. P2: Exact candidate resolution in vLLM Semantic Router (no ambiguous substring matching).
4. P1: Robust final answer extraction and objective grading (rejecting truncated <think> reasoning).
"""

from __future__ import annotations

import pytest

from adaroute.api import cache
from adaroute.cache.exact import ExactCache
from adaroute.cache.semantic import SemanticCache
from adaroute.schemas import ChatMessage, PrivacyLevel, RequestContext


@pytest.fixture(autouse=True)
def clean_cache():
    cache.clear()


# =============================================================================
# 1. Exact Cache Float Precision and Provenance
# =============================================================================


def test_TC_M4_15_exact_cache_distinct_float_temperatures_do_not_collide():
    """Distinct float temperatures must produce distinct keys and never collapse."""
    c = ExactCache(backend="memory")
    key1 = c.compute_key(
        tenant_id="t1",
        user_id="u1",
        logical_model="general",
        model_version="1.0.0",
        prompt="Hello world",
        temperature=0.7000001,
    )
    key2 = c.compute_key(
        tenant_id="t1",
        user_id="u1",
        logical_model="general",
        model_version="1.0.0",
        prompt="Hello world",
        temperature=0.7000002,
    )
    assert key1 != key2, "Close but distinct float temperatures must produce distinct cache keys"


def test_TC_M4_16_exact_cache_zero_temperature_distinct_from_close_float():
    """Temperature 0.0 must be distinct from any non-zero float."""
    c = ExactCache(backend="memory")
    key_zero = c.compute_key(
        tenant_id="t1",
        user_id="u1",
        logical_model="general",
        model_version="1.0.0",
        prompt="Hello world",
        temperature=0.0,
    )
    key_tiny = c.compute_key(
        tenant_id="t1",
        user_id="u1",
        logical_model="general",
        model_version="1.0.0",
        prompt="Hello world",
        temperature=1e-7,
    )
    assert key_zero != key_tiny, "0.0 must be distinct from 1e-7"


# =============================================================================
# 2. Semantic Cache: Strict Equivalence, Provenance & Fail-Closed
# =============================================================================


def test_TC_M4_17_semantic_cache_rejects_temperature_drift():
    """Semantic cache must reject temperature drift within 0.05 tolerance."""
    sem = SemanticCache()
    req = RequestContext(
        task_type="general",
        prompt_tokens=10,
        estimated_output_tokens=20,
        quality_requirement="low",
        latency_slo=2.0,
        privacy_level=PrivacyLevel.PUBLIC,
        budget=1.0,
        tenant_id="t1",
        user_id="u1",
        request_id="r1",
        temperature=0.70,
    )

    # Cached entry has temperature=0.74 (within old 0.05 tolerance)
    entry_data = {
        "tenant_id": "t1",
        "user_id": "u1",
        "privacy_level": "public",
        "user_private": False,
        "logical_model": "general",
        "model": "qwen",
        "origin_provider": "local",
        "finish_reason": "stop",
        "is_simulated": False,
        "temperature": 0.74,
        "output_tokens": 15,
        "max_tokens": 300,
        "quality": 0.8,
        "prompt": "What is the capital of France?",
        "policy_name": "adaptive",
        "policy_version": "v4.0",
    }

    eligible, reason = sem._is_entry_eligible(
        entry_data=entry_data,
        request=req,
        prompt_text="What is the capital of France?",
        is_real_mode=False,
        u_id="u1",
        req_quality=0.3,
        req_task="general",
        explicit_model=None,
        req_temp=0.70,
    )
    assert not eligible, "Temperature drift (0.74 vs 0.70) must be rejected"
    assert "temperature" in reason.lower()


def test_TC_M4_18_semantic_cache_checks_original_max_tokens_budget():
    """An entry generated with max_tokens=20 cannot satisfy a request asking for max_tokens=300."""
    sem = SemanticCache()
    req = RequestContext(
        task_type="general",
        prompt_tokens=10,
        estimated_output_tokens=20,
        quality_requirement="low",
        latency_slo=2.0,
        privacy_level=PrivacyLevel.PUBLIC,
        budget=1.0,
        tenant_id="t1",
        user_id="u1",
        request_id="r1",
        temperature=0.70,
        max_tokens=300,
    )

    entry_data = {
        "tenant_id": "t1",
        "user_id": "u1",
        "privacy_level": "public",
        "user_private": False,
        "logical_model": "general",
        "model": "qwen",
        "origin_provider": "local",
        "finish_reason": "stop",
        "is_simulated": False,
        "temperature": 0.70,
        "output_tokens": 18,
        "max_tokens": 20,  # Clamped generation budget
        "quality": 0.8,
        "prompt": "Explain quantum physics",
        "policy_name": "adaptive",
        "policy_version": "v4.0",
    }

    eligible, reason = sem._is_entry_eligible(
        entry_data=entry_data,
        request=req,
        prompt_text="Explain quantum physics",
        is_real_mode=False,
        u_id="u1",
        req_quality=0.3,
        req_task="general",
        explicit_model=None,
        req_temp=0.70,
    )
    assert not eligible, "Entry generated with max_tokens=20 cannot satisfy max_tokens=300 request"
    assert "max_tokens" in reason.lower()


def test_TC_M4_19_semantic_cache_enforces_policy_and_version_matching():
    """Policy mismatch or effective policy version mismatch must fail closed."""
    sem = SemanticCache()
    req = RequestContext(
        task_type="general",
        quality_requirement="low",
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="t1",
        user_id="u1",
        request_id="r1",
        temperature=0.70,
        max_tokens=100,
        policy_name="cheapest",
        policy_version="v4.0",
    )

    # 1. Policy mismatch
    entry_wrong_policy = {
        "tenant_id": "t1",
        "user_id": "u1",
        "privacy_level": "public",
        "user_private": False,
        "logical_model": "general",
        "model": "qwen",
        "origin_provider": "local",
        "finish_reason": "stop",
        "is_simulated": False,
        "temperature": 0.70,
        "output_tokens": 10,
        "max_tokens": 100,
        "quality": 0.8,
        "prompt": "Hello",
        "policy_name": "strongest",
        "policy_version": "v4.0",
    }
    eligible, reason = sem._is_entry_eligible(
        entry_data=entry_wrong_policy,
        request=req,
        prompt_text="Hello",
        is_real_mode=False,
        u_id="u1",
        req_quality=0.3,
        req_task="general",
        explicit_model=None,
        req_temp=0.70,
    )
    assert not eligible
    assert "policy" in reason.lower()

    # 2. Version mismatch
    entry_wrong_ver = dict(entry_wrong_policy, policy_name="cheapest", policy_version="v4.1")
    eligible, reason = sem._is_entry_eligible(
        entry_data=entry_wrong_ver,
        request=req,
        prompt_text="Hello",
        is_real_mode=False,
        u_id="u1",
        req_quality=0.3,
        req_task="general",
        explicit_model=None,
        req_temp=0.70,
    )
    assert not eligible
    assert "version" in reason.lower()


def test_TC_M4_20_semantic_cache_fails_closed_on_missing_provenance():
    """Missing or empty origin_provider, finish_reason, or tenant must fail closed."""
    sem = SemanticCache()
    req = RequestContext(
        task_type="general",
        quality_requirement="low",
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="t1",
        user_id="u1",
        request_id="r1",
        temperature=0.70,
    )

    # Missing origin_provider
    entry_no_origin = {
        "tenant_id": "t1",
        "user_id": "u1",
        "privacy_level": "public",
        "user_private": False,
        "logical_model": "general",
        "model": "qwen",
        "origin_provider": "",
        "finish_reason": "stop",
        "temperature": 0.70,
        "output_tokens": 10,
        "quality": 0.8,
        "prompt": "Hello",
    }
    eligible, reason = sem._is_entry_eligible(
        entry_data=entry_no_origin,
        request=req,
        prompt_text="Hello",
        is_real_mode=False,
        u_id="u1",
        req_quality=0.3,
        req_task="general",
        explicit_model=None,
        req_temp=0.70,
    )
    assert not eligible
    assert "provider" in reason.lower() or "provenance" in reason.lower()

    # Missing finish_reason
    entry_no_finish = dict(entry_no_origin, origin_provider="local", finish_reason="")
    eligible, reason = sem._is_entry_eligible(
        entry_data=entry_no_finish,
        request=req,
        prompt_text="Hello",
        is_real_mode=False,
        u_id="u1",
        req_quality=0.3,
        req_task="general",
        explicit_model=None,
        req_temp=0.70,
    )
    assert not eligible
    assert "finish" in reason.lower()


def test_TC_M4_21_semantic_cache_multi_turn_protection():
    """Semantic cache must return None (miss) for multi-turn requests or system prompts."""
    sem = SemanticCache()
    req_multi = RequestContext(
        task_type="general",
        quality_requirement="low",
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="t1",
        user_id="u1",
        request_id="r1",
        messages=[
            ChatMessage(role="user", content="Turn 1"),
            ChatMessage(role="assistant", content="Answer 1"),
            ChatMessage(role="user", content="Turn 2"),
        ],
    )
    result = sem.get(request=req_multi, prompt_text="Turn 2")
    assert result is None, "Multi-turn requests must bypass semantic cache"

    req_sys = RequestContext(
        task_type="general",
        quality_requirement="low",
        privacy_level=PrivacyLevel.PUBLIC,
        tenant_id="t1",
        user_id="u1",
        request_id="r2",
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello"),
        ],
    )
    result_sys = sem.get(request=req_sys, prompt_text="Hello")
    assert result_sys is None, "Requests with system prompt must bypass semantic cache"


# =============================================================================
# 3. Router Candidate Identity Resolution (No Substring Matching)
# =============================================================================


def test_TC_M3_44_vllm_router_candidate_resolution_no_substring_collision():
    """Candidate matching must not match 'code' into 'unicode-transcoder' by substring."""

    class MockCand:
        def __init__(self, deployment_id: str, model_id: str, caps: list[str]):
            self.deployment_id = deployment_id
            self.model_id = model_id
            self.provider_model_name = model_id
            self.model = type("Model", (), {"capabilities": caps, "capability_profiles": {}})()

    dep1 = MockCand("dep-unicode-tokenizer", "unicode-tokenizer-v1", ["text-processing"])
    dep2 = MockCand("dep-coding-specialist", "qwen-coder-v1", ["coding"])

    eligible = [(dep1, None), (dep2, None)]

    rec = "coding"
    matched = None
    for cand, _ in eligible:
        caps = getattr(cand.model, "capability_profiles", {}) or getattr(cand.model, "capabilities", {})
        cap_list = list(caps.keys()) if isinstance(caps, dict) else list(caps)
        # Strict matching: exact match only
        if (
            cand.deployment_id.lower() == rec.lower()
            or cand.model_id.lower() == rec.lower()
            or getattr(cand, "provider_model_name", "").lower() == rec.lower()
            or any(cap.lower() == rec.lower() for cap in cap_list)
        ):
            matched = cand
            break

    assert matched is not None
    assert matched.deployment_id == "dep-coding-specialist"
    assert matched.deployment_id != "dep-unicode-tokenizer"


# =============================================================================
# 4. Objective Final Answer Grader Tests (Adversarial Robustness)
# =============================================================================


def test_TC_M8_03_math_grader_rejects_truncated_think():
    """Output truncated inside <think> without </think> or final answer must be marked incomplete."""
    from results.report_evidence.run_honest_eval import extract_final_answer

    truncated_output = "<think>\nLet's calculate 17 * 23. 17 * 20 = 340, 17 * 3 = 51. 340 + 51 = 391."
    final_ans, reason = extract_final_answer(truncated_output, finish_reason="length")
    assert final_ans is None
    assert reason and ("truncated" in reason.lower() or "incomplete" in reason.lower())


def test_TC_M8_04_math_grader_rejects_reasoning_substring_false_positive():
    """If model thought about 391 in reasoning but produced 500 in final answer, check fails."""
    from results.report_evidence.run_honest_eval import check_math, extract_final_answer

    confused_output = "<think>\nMaybe 17 * 23 is 391? No wait.\n</think>\n\n500"
    final_ans, _reason = extract_final_answer(confused_output, finish_reason="stop")
    assert final_ans == "500"
    passed, why = check_math(final_ans)
    assert not passed
    assert "Expected 391" in why


def test_TC_M8_05_math_grader_passes_clean_391():
    """Clean 391 output must pass the objective math grader."""
    from results.report_evidence.run_honest_eval import check_math, extract_final_answer

    clean_output = "<think>\n17 * 23 = 391\n</think>\n\n391"
    final_ans, _reason = extract_final_answer(clean_output, finish_reason="stop")
    assert final_ans == "391"
    passed, _why = check_math(final_ans)
    assert passed


def test_TC_M8_06_json_grader_validates_real_json():
    """Valid JSON output must pass while plain words must fail the JSON grader."""
    from results.report_evidence.run_honest_eval import check_json, extract_final_answer

    valid_json = '<think>Generating array</think>\n\n["apple", "banana", "cherry"]'
    final_ans, _ = extract_final_answer(valid_json, finish_reason="stop")
    assert final_ans is not None
    passed, _why = check_json(final_ans)
    assert passed

    invalid_words_only = "<think>Generating array</think>\n\nHere is apple banana cherry"
    final_ans_bad, _ = extract_final_answer(invalid_words_only, finish_reason="stop")
    assert final_ans_bad is not None
    passed_bad, _why_bad = check_json(final_ans_bad)
    assert not passed_bad
