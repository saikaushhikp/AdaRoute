"""Deterministic Request & Semantic Signal Extraction (Week-4 Revamp).

Extracts explainable, no-training signals from prompt text and request metadata:
- Code presence (markdown blocks, syntax patterns, keywords)
- Mathematical notation (LaTeX, math symbols, equations)
- Reasoning markers ('step-by-step', 'prove', 'derive', logic chains)
- Structured output demand (JSON, schema, tables, markdown)
- Explicit constraints count ('must', 'cannot', 'strictly', numbered rules)
- Prompt length bucket ('short', 'medium', 'long')
- 'When to Reason' composite score with evidence list
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SemanticSignals:
    """Immutable signals extracted deterministically from request prompt & metadata."""

    prompt_tokens: int
    prompt_length_bucket: str  # "short", "medium", "long"
    has_code: bool = False
    has_math: bool = False
    has_structured_output: bool = False
    constraint_count: int = 0
    reasoning_score: float = 0.0  # [0.0, 1.0]
    reasoning_mode: str = "standard"  # "fast", "standard", "reasoning_required"
    evidence: list[str] = field(default_factory=list)
    source: str = "deterministic_rules"

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "prompt_length_bucket": self.prompt_length_bucket,
            "has_code": self.has_code,
            "has_math": self.has_math,
            "has_structured_output": self.has_structured_output,
            "constraint_count": self.constraint_count,
            "reasoning_score": round(self.reasoning_score, 3),
            "reasoning_mode": self.reasoning_mode,
            "evidence": list(self.evidence),
            "source": self.source,
        }


class SignalExtractor:
    """Deterministic, rule-based signal extractor with zero offline training."""

    CODE_PATTERNS = [
        re.compile(r"```[a-zA-Z0-9_-]*"),  # Fenced code blocks
        re.compile(r"\b(def|class|import|function|return|const|let|var|SELECT|FROM|WHERE)\b"),
        re.compile(r"[{};()\[\]]{3,}"),  # Code syntax density
    ]

    MATH_PATTERNS = [
        re.compile(r"\\[a-zA-Z]+"),  # LaTeX commands like \sum, \int, \frac
        re.compile(r"[\$\^\{\}\_]{2,}"),  # Math formatting symbols
        re.compile(r"\b(equation|theorem|integral|derivative|matrix|proof|modulo|logarithm)\b", re.IGNORECASE),
        re.compile(r"[=><\+\-\*/]{2,}"),
    ]

    REASONING_PATTERNS = [
        (re.compile(r"\b(prove|derive|verify|deduce|infer)\b", re.IGNORECASE), "proof_keyword", 0.35),
        (re.compile(r"\b(step[-\s]by[-\s]step|think carefully|chain of thought|explain your reasoning)\b", re.IGNORECASE), "step_by_step_marker", 0.30),
        (re.compile(r"\b(why is|how come|analyze the trade-offs|counterexample)\b", re.IGNORECASE), "analytical_marker", 0.20),
        (re.compile(r"\b(solve|calculate|evaluate|optimize)\b", re.IGNORECASE), "problem_solving_marker", 0.15),
    ]

    STRUCTURED_PATTERNS = [
        re.compile(r"\b(json|yaml|xml|csv|schema|markdown table)\b", re.IGNORECASE),
        re.compile(r"\bformat your response as\b", re.IGNORECASE),
        re.compile(r"```json", re.IGNORECASE),
    ]

    CONSTRAINT_PATTERNS = [
        re.compile(r"\b(must|must not|cannot|should not|strictly|never|always|do not)\b", re.IGNORECASE),
        re.compile(r"^\s*([0-9]+\.|\*|\-)\s+[A-Z]", re.MULTILINE),  # Numbered/bulleted rules
    ]

    def __init__(
        self,
        reasoning_on_threshold: float = 0.65,
        reasoning_off_threshold: float = 0.30,
    ):
        self.on_threshold = reasoning_on_threshold
        self.off_threshold = reasoning_off_threshold

    def extract(self, prompt_text: str, prompt_tokens: int) -> SemanticSignals:
        evidence: list[str] = []

        # 1. Prompt bucket
        if prompt_tokens < 100:
            bucket = "short"
        elif prompt_tokens <= 500:
            bucket = "medium"
        else:
            bucket = "long"

        # 2. Code detection
        has_code = False
        for pat in self.CODE_PATTERNS:
            if pat.search(prompt_text):
                has_code = True
                evidence.append("code_syntax_detected")
                break

        # 3. Math detection
        has_math = False
        for pat in self.MATH_PATTERNS:
            if pat.search(prompt_text):
                has_math = True
                evidence.append("math_notation_detected")
                break

        # 4. Structured output detection
        has_struct = False
        for pat in self.STRUCTURED_PATTERNS:
            if pat.search(prompt_text):
                has_struct = True
                evidence.append("structured_output_requested")
                break

        # 5. Constraint count
        constraints = 0
        for pat in self.CONSTRAINT_PATTERNS:
            matches = pat.findall(prompt_text)
            constraints += len(matches)
        if constraints >= 2:
            evidence.append(f"multi_constraint_count_{constraints}")

        # 6. Reasoning score & When-to-Reason decision
        reasoning_score = 0.0
        for pat, tag, weight in self.REASONING_PATTERNS:
            if pat.search(prompt_text):
                reasoning_score += weight
                evidence.append(tag)

        if has_math:
            reasoning_score += 0.25
        if constraints >= 3:
            reasoning_score += 0.20

        # Cap score in [0.0, 1.0]
        reasoning_score = min(1.0, reasoning_score)

        if reasoning_score >= self.on_threshold:
            mode = "reasoning_required"
        elif reasoning_score <= self.off_threshold:
            mode = "fast"
        else:
            mode = "standard"

        return SemanticSignals(
            prompt_tokens=prompt_tokens,
            prompt_length_bucket=bucket,
            has_code=has_code,
            has_math=has_math,
            has_structured_output=has_struct,
            constraint_count=constraints,
            reasoning_score=reasoning_score,
            reasoning_mode=mode,
            evidence=evidence,
            source="deterministic_rules",
        )
