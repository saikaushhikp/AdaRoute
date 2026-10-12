"""Request and Attempt Correlation Tracer (Hardened Week-3 Version).

Provides comprehensive observability across:
1. Pre-validation and validation rejection traces.
2. Full multi-attempt cost and token ledger (sum of all billable attempts).
3. Secret and credential sanitization in error logs.
4. Aligned cache metrics and SLO attainment calculation.
"""

from __future__ import annotations

import re
import time

from pydantic import BaseModel, Field


def sanitize_message(text: str | None) -> str | None:
    """Scrub potential API keys, secrets, or bearer tokens from telemetry logs."""
    if not text:
        return text
    # Redact common key prefixes
    patterns = [
        r"(Bearer\s+)[A-Za-z0-9_\-\.]+",
        r"(gsk_[A-Za-z0-9]{20,})",
        r"(sk-[A-Za-z0-9_\-]{20,})",
        r"(key=)[A-Za-z0-9_\-\.]+",
        r"(AQ\.[A-Za-z0-9_\-]{20,})",
    ]
    sanitized = text
    for pat in patterns:
        sanitized = re.sub(pat, r"[REDACTED_SECRET]", sanitized, flags=re.IGNORECASE)
    return sanitized


class AttemptRecord(BaseModel):
    attempt_number: int
    replica_id: str
    model_id: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    status: str = "success"  # "success", "failed", "timeout", "rate_limited"
    error: str | None = None


class RequestEvent(BaseModel):
    request_id: str
    tenant_id: str = "default"
    user_id: str = "anonymous"
    logical_model: str = "general"
    task_type: str = "general"
    privacy_level: str = "public"
    budget: float = 1.0
    latency_slo: float | None = 2.0
    prompt_tokens: int = 0
    estimated_output_tokens: int = 50
    actual_output_tokens: int = 0
    selected_model: str | None = None
    selected_replica: str | None = None
    cache_outcome: str = "miss"  # "hit", "miss", "bypass"
    attempts: list[AttemptRecord] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    slo_attained: bool = True
    final_status: str = "success"  # "success", "rejected", "failed", "validation_error"
    error_detail: str | None = None
    timestamp: float = Field(default_factory=time.time)


class RequestTracer:
    def __init__(self):
        self._traces: dict[str, RequestEvent] = {}

    def start_trace(
        self,
        request_id: str,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        logical_model: str = "general",
        task_type: str = "general",
        privacy_level: str = "public",
        budget: float = 1.0,
        latency_slo: float | None = 2.0,
        prompt_tokens: int = 0,
        estimated_output_tokens: int = 50,
    ) -> RequestEvent:
        event = RequestEvent(
            request_id=request_id,
            tenant_id=tenant_id,
            user_id=user_id,
            logical_model=logical_model,
            task_type=task_type,
            privacy_level=privacy_level,
            budget=budget,
            latency_slo=latency_slo,
            prompt_tokens=prompt_tokens,
            estimated_output_tokens=estimated_output_tokens,
        )
        self._traces[request_id] = event
        return event

    def record_validation_error(self, request_id: str, error_detail: str) -> None:
        """Records validation failure occurring before pre-routing characterization."""
        event = self._traces.get(request_id)
        sanitized = sanitize_message(error_detail)
        if event:
            event.final_status = "validation_error"
            event.error_detail = sanitized
        else:
            event = RequestEvent(
                request_id=request_id,
                final_status="validation_error",
                error_detail=sanitized,
            )
            self._traces[request_id] = event

    def record_attempt(
        self,
        request_id: str,
        attempt_number: int,
        replica_id: str,
        model_id: str,
        provider: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        cost: float = 0.0,
        status: str = "success",
        error: str | None = None,
    ) -> None:
        event = self._traces.get(request_id)
        if event:
            event.attempts.append(
                AttemptRecord(
                    attempt_number=attempt_number,
                    replica_id=replica_id,
                    model_id=model_id,
                    provider=provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    cost=cost,
                    status=status,
                    error=sanitize_message(error),
                )
            )

    def finish_trace(
        self,
        request_id: str,
        selected_model: str,
        selected_replica: str,
        actual_output_tokens: int,
        total_latency_ms: float,
        estimated_cost: float,
        actual_cost: float | None = None,
        cache_outcome: str = "miss",
        final_status: str = "success",
        error_detail: str | None = None,
    ) -> RequestEvent | None:
        event = self._traces.get(request_id)
        if event:
            event.selected_model = selected_model
            event.selected_replica = selected_replica
            event.actual_output_tokens = actual_output_tokens
            event.total_latency_ms = total_latency_ms
            event.estimated_cost = estimated_cost
            event.cache_outcome = cache_outcome
            event.final_status = final_status
            event.error_detail = sanitize_message(error_detail)

            # Reconcile actual cost: sum of all billable attempts, or explicit cost
            if actual_cost is not None:
                event.actual_cost = actual_cost
            elif event.attempts:
                event.actual_cost = sum(a.cost for a in event.attempts)
            else:
                event.actual_cost = 0.0

            if event.latency_slo:
                event.slo_attained = (total_latency_ms / 1000.0) <= event.latency_slo
            else:
                event.slo_attained = True

        return event

    def get_trace(self, request_id: str) -> RequestEvent | None:
        return self._traces.get(request_id)

    def get_all_traces(self) -> list[RequestEvent]:
        return list(self._traces.values())

    def clear(self) -> None:
        self._traces.clear()


default_tracer = RequestTracer()
