"""Tests for Section 12.2: Capability and Request Characterization.

Verifies:
- Resolves only models advertising requested capability
- Unknown capability fails closed without general fallback
- Concrete model ID restricts candidate set strictly to that model
- Per-capability quality differentiation (coding vs reasoning)
- Deterministic signal extraction: code presence, math, structured output, constraints
- 'When to Reason' signals and explainable evidence
"""

from __future__ import annotations

from adaroute.selection.candidates import CandidateResolver
from adaroute.selection.catalog import (
    CapabilityProfile,
    DeploymentDefinition,
    ModelDefinition,
    ProviderDefinition,
    SystemCatalog,
)
from adaroute.selection.signals import SignalExtractor


def create_sample_catalog() -> SystemCatalog:
    catalog = SystemCatalog(strict_catalog=True)
    p = ProviderDefinition(provider_id="local_p", adapter_kind="ollama", execution_type="local")
    catalog.register_provider(p)

    # Model 1: General + Coding specialist
    m1 = ModelDefinition(
        model_id="code-specialist",
        capability_profiles={
            "general": CapabilityProfile("general", supported=True, quality_value=0.75),
            "coding": CapabilityProfile("coding", supported=True, quality_value=0.92),
            "reasoning": CapabilityProfile("reasoning", supported=False, quality_value=0.50),
        },
        context_window=32768,
    )
    catalog.register_model(m1)
    catalog.register_deployment(
        DeploymentDefinition(deployment_id="dep-code", model_id="code-specialist", provider_id="local_p")
    )

    # Model 2: Reasoning specialist
    m2 = ModelDefinition(
        model_id="math-reasoner",
        capability_profiles={
            "general": CapabilityProfile("general", supported=True, quality_value=0.78),
            "reasoning": CapabilityProfile("reasoning", supported=True, quality_value=0.95),
            "coding": CapabilityProfile("coding", supported=False, quality_value=0.40),
        },
        context_window=16384,
    )
    catalog.register_model(m2)
    catalog.register_deployment(
        DeploymentDefinition(deployment_id="dep-math", model_id="math-reasoner", provider_id="local_p")
    )

    return catalog


def test_tc_m3_42_capability_resolution_isolation():
    catalog = create_sample_catalog()
    resolver = CandidateResolver(catalog)

    # Coding should only match code-specialist
    coding_cands = resolver.resolve_candidates("coding")
    assert len(coding_cands) == 1
    assert coding_cands[0].model_id == "code-specialist"

    # Reasoning should only match math-reasoner
    reasoning_cands = resolver.resolve_candidates("reasoning")
    assert len(reasoning_cands) == 1
    assert reasoning_cands[0].model_id == "math-reasoner"


def test_tc_m3_43_unknown_capability_fails_closed():
    catalog = create_sample_catalog()
    resolver = CandidateResolver(catalog)
    # Unknown capability returns empty list (no silent fallback to general)
    cands = resolver.resolve_candidates("nonexistent_task")
    assert len(cands) == 0


def test_tc_m3_44_concrete_model_id_direct_lookup():
    catalog = create_sample_catalog()
    resolver = CandidateResolver(catalog)
    # Target concrete model_id directly
    cands = resolver.resolve_candidates("code-specialist")
    assert len(cands) == 1
    assert cands[0].model_id == "code-specialist"


def test_tc_m3_45_per_capability_quality_differentiation():
    catalog = create_sample_catalog()
    m_code = catalog.get_model("code-specialist")
    assert m_code.get_capability_quality("coding") == 0.92
    assert m_code.get_capability_quality("general") == 0.75
    # Unsupported capability quality
    cap_reas = m_code.capability_profiles.get("reasoning")
    assert cap_reas.supported is False


def test_tc_m3_46_deterministic_signal_extraction_code():
    extractor = SignalExtractor()
    prompt = "```python\ndef fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n```"
    signals = extractor.extract(prompt, prompt_tokens=25)
    assert signals.has_code is True
    assert "code_syntax_detected" in signals.evidence
    assert signals.prompt_length_bucket == "short"


def test_tc_m3_47_deterministic_signal_extraction_reasoning():
    extractor = SignalExtractor()
    prompt = "Prove step-by-step why the square root of 2 is irrational under the following constraint: must use proof by contradiction."
    signals = extractor.extract(prompt, prompt_tokens=30)
    assert signals.reasoning_score >= 0.65
    assert signals.reasoning_mode == "reasoning_required"
    assert "proof_keyword" in signals.evidence
    assert "step_by_step_marker" in signals.evidence


def test_tc_m3_48_deterministic_signal_structured_output():
    extractor = SignalExtractor()
    prompt = "Format your response as a valid JSON object matching the provided schema."
    signals = extractor.extract(prompt, prompt_tokens=15)
    assert signals.has_structured_output is True
    assert "structured_output_requested" in signals.evidence
