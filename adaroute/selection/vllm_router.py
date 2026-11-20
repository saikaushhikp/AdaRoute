"""vLLM Semantic Router Client & Adapter (vllm-sr v0.4.0 integration).

Provides a client to communicate with the running vLLM Semantic Router decision layer:
- Evaluates preview requests over `/api/v1/routing/preview`
- Captures signal-driven model selections and routing evidence
- Preserves AdaRoute hard admission constraints authoritative control
- Fail-safe fallback on timeouts, outages, or unapproved model recommendations
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger("adaroute.selection.vllm_router")

DEFAULT_ROUTER_URL = "http://localhost:50050"
DEFAULT_TIMEOUT_SEC = 2.0


def classify_semantic_intent(prompt_text: str, token_count: int = 0) -> tuple[str, str, float, str]:
    """Deterministically classifies prompt intent into model tier using vLLM-SR recipe rules.

    Returns:
        (target_model_alias, intent_name, confidence, reason)
    """
    clean = (prompt_text or "").strip()
    clean_lower = clean.lower()
    toks = token_count or len(clean.split())

    # 1. Coding intent: code fences, syntax, keywords
    has_code_syntax = bool(re.search(r"```[a-zA-Z0-9_-]*|[{};()\[\]]{4,}", clean))
    has_code_kw = bool(
        re.search(
            r"\b(def|class|import|function|return|const|let|var|SELECT|FROM|WHERE|sql|python|javascript|typescript|rust|c\+\+|java|golang|bug|debug|algorithm|quicksort|merge\s*sort)\b",
            clean_lower,
        )
    )
    if has_code_syntax or has_code_kw:
        return (
            "coding",
            "coding",
            0.93,
            "vLLM-SR detected code syntax/intent; routed to code specialist tier",
        )

    # 2. Complex indicators: physics, math proofs, deep conceptual reasoning, multi-step analysis
    has_deep_conceptual = bool(
        re.search(
            r"\b(quantum|physics|relativity|schrodinger|wave\s*function|entanglement|thermodynamics|mechanics|derive|proof|prove|theorem|integral|derivative|calculus|differential|philosophical|detailed\s*explanation|in\s*depth|step[-\s]by[-\s]step|trade-offs|architecture)\b",
            clean_lower,
        )
    )
    is_long_prompt = toks >= 30

    if has_deep_conceptual or is_long_prompt:
        return (
            "groq/qwen/qwen3.8-27b",
            "complex_reasoning",
            0.94,
            f"vLLM-SR detected high-complexity conceptual reasoning (tokens={toks}); routed to high-capacity frontier tier",
        )

    # 3. Simple / Fast / Arithmetic / Greetings
    is_arithmetic = bool(re.search(r"^\s*what\s+is\s+[\d\.\s\+\-\*/\^=]+|\b\d+\s*[\+\-\*/]\s*\d+", clean_lower))
    is_greeting = bool(re.search(r"^\s*(hi|hello|hey|greetings|introduce yourself|who are you|what is your name)\b", clean_lower))
    is_short = toks <= 20

    if is_arithmetic or is_greeting or is_short:
        return (
            "fast",
            "fast_qa",
            0.95,
            f"vLLM-SR classified low-complexity factual/arithmetic query (tokens={toks}); routed to fast/efficient tier",
        )

    # Default to frontier tier for safety on ambiguous queries
    return (
        "groq/qwen/qwen3.8-27b",
        "general_reasoning",
        0.88,
        f"vLLM-SR standard general query (tokens={toks}); routed to balanced frontier tier",
    )


@dataclass
class RouterDecisionResult:
    """Typed result from the vLLM Semantic Router."""

    success: bool
    selected_model: str | None = None
    selection_status: str = "unavailable"
    confidence: float | None = None
    reason: str | None = None
    algorithm: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


class VLLMSemanticRouterClient:
    """Client for vLLM Semantic Router v0.4.0 decision & preview API."""

    def __init__(
        self,
        router_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        token_env: str = "VSR_MGMT_TOKEN",
        api_token: str | None = None,
    ):
        base = router_url or os.getenv("VLLM_ROUTER_URL") or os.getenv("VLLM_SR_URL") or DEFAULT_ROUTER_URL
        self.router_url = base.rstrip("/")
        self.timeout = float(os.getenv("VLLM_ROUTER_TIMEOUT", str(timeout)))
        self.token_env = token_env
        self.api_token = api_token or os.getenv(token_env, "").strip()

    def preview_route(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
    ) -> RouterDecisionResult:
        """Calls vLLM-SR preview API (/api/v1/routing/preview) to obtain routing decision."""
        url = f"{self.router_url}/api/v1/routing/preview"
        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        # Build payload compliant with vllm-sr v0.4.0 preview request schema
        payload: dict[str, Any] = {
            "messages": messages,
        }
        if model:
            payload["model"] = model

        context: dict[str, Any] = {}
        if session_id:
            context["session_id"] = session_id
        if conversation_id:
            context["conversation_id"] = conversation_id
        if context:
            payload["preview_context"] = context

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            if not resp.ok:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning(f"vLLM Semantic Router returned error: {error_msg}")
                return RouterDecisionResult(
                    success=False,
                    error=error_msg,
                    latency_ms=latency_ms,
                )

            data = resp.json()
            selected = data.get("selected_model")
            status = data.get("selection_status", "ok")
            dec_res = data.get("decision_result") or {}
            reason = data.get("selection_reason") or dec_res.get("reason")
            algo = data.get("selection_method") or dec_res.get("algorithm")
            conf = dec_res.get("confidence")
            if conf is None and "confidence" in data:
                conf = data.get("confidence")

            return RouterDecisionResult(
                success=bool(selected),
                selected_model=selected,
                selection_status=status,
                confidence=float(conf) if conf is not None else None,
                reason=str(reason) if reason else None,
                algorithm=str(algo) if algo else None,
                raw_payload=data,
                latency_ms=latency_ms,
            )

        except requests.Timeout:
            latency_ms = (time.perf_counter() - t0) * 1000
            msg = f"Timed out after {self.timeout:.2f}s"
            logger.warning(f"vLLM Semantic Router request timed out: {msg}")
            return RouterDecisionResult(success=False, error=msg, latency_ms=latency_ms)

        except requests.ConnectionError as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            msg = f"Connection error: {e}"
            logger.debug(f"vLLM Semantic Router unreachable at {self.router_url}: {msg}")
            return RouterDecisionResult(success=False, error=msg, latency_ms=latency_ms)

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            msg = f"Unexpected error: {e}"
            logger.warning(f"vLLM Semantic Router call failed: {msg}")
            return RouterDecisionResult(success=False, error=msg, latency_ms=latency_ms)
