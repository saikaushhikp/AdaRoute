"""Request and Attempt Correlation Tracer.

Provides full observability across routing stages, retry attempts,
cache decisions, and economic metrics while strictly redacting secrets and raw prompts.
"""
from __future__ import annotations

import time

from pydantic import BaseModel, Field


class AttemptRecord(BaseModel):
    attempt_number: int
    replica_id: str
    model_id: str
    provider: str
    latency_ms: float
    status: str  # "success", "failed", "timeout"
    error: str | None = None


class RequestEvent(BaseModel):
    request_id: str
    tenant_id: str
    user_id: str
    logical_model: str
    task_type: str
    privacy_level: str
    budget: float
    latency_slo: float | None
    prompt_tokens: int
    estimated_output_tokens: int
    actual_output_tokens: int = 0
    selected_model: str | None = None
    selected_replica: str | None = None
    cache_outcome: str = "miss"  # "hit", "miss", "bypass"
    attempts: list[AttemptRecord] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    slo_attained: bool = True
    final_status: str = "success"
    timestamp: float = Field(default_factory=time.time)


class RequestTracer:
    def __init__(self):
        self._traces: dict[str, RequestEvent] = {}

    def start_trace(
        self,
        request_id: str,
        tenant_id: str,
        user_id: str,
        logical_model: str,
        task_type: str,
        privacy_level: str,
        budget: float,
        latency_slo: float | None,
        prompt_tokens: int,
        estimated_output_tokens: int,
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

    def record_attempt(
        self,
        request_id: str,
        attempt_number: int,
        replica_id: str,
        model_id: str,
        provider: str,
        latency_ms: float,
        status: str,
        error: str | None = None,
    ):
        event = self._traces.get(request_id)
        if event:
            event.attempts.append(
                AttemptRecord(
                    attempt_number=attempt_number,
                    replica_id=replica_id,
                    model_id=model_id,
                    provider=provider,
                    latency_ms=latency_ms,
                    status=status,
                    error=error,
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
        actual_cost: float,
        cache_outcome: str = "miss",
        final_status: str = "success",
    ) -> RequestEvent | None:
        event = self._traces.get(request_id)
        if event:
            event.selected_model = selected_model
            event.selected_replica = selected_replica
            event.actual_output_tokens = actual_output_tokens
            event.total_latency_ms = total_latency_ms
            event.estimated_cost = estimated_cost
            event.actual_cost = actual_cost
            event.cache_outcome = cache_outcome
            event.final_status = final_status

            if event.latency_slo:
                event.slo_attained = (total_latency_ms / 1000.0) <= event.latency_slo
            else:
                event.slo_attained = True

        return event

    def get_trace(self, request_id: str) -> RequestEvent | None:
        return self._traces.get(request_id)

    def get_all_traces(self) -> list[RequestEvent]:
        return list(self._traces.values())

    def clear(self):
        self._traces.clear()


default_tracer = RequestTracer()
