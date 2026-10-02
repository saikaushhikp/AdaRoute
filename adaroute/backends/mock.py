"""Configurable Mock / Cloud Provider Backend for AdaRoute.

Simulates external APIs (e.g., GPT-4o), network latencies, fault injection,
and records execution counts to prove safety invariants in tests.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from adaroute.schemas import ModelSpec, RequestContext
from adaroute.workloads.tokenizer import count_tokens

from .base import BaseBackend


class MockBackend(BaseBackend):
    def __init__(
        self,
        provider_name: str = "mock-external",
        base_latency_ms: float = 40.0,
        inject_failure_status: int | None = None,
        inject_timeout: bool = False,
    ):
        self.provider_name = provider_name
        self.base_latency_ms = base_latency_ms
        self.inject_failure_status = inject_failure_status
        self.inject_timeout = inject_timeout
        self.call_count = 0
        self.history: list = []

    async def execute(
        self,
        request: RequestContext,
        model_spec: ModelSpec,
        replica_id: str,
        prompt_text: str,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.history.append({
            "request_id": request.request_id,
            "privacy_level": request.privacy_level.value,
            "model_id": model_spec.model_id,
            "replica_id": replica_id,
            "time": time.time(),
        })

        if self.inject_timeout:
            raise TimeoutError("Simulated provider upstream timeout")

        if self.inject_failure_status:
            raise RuntimeError(f"Simulated HTTP {self.inject_failure_status} error from {self.provider_name}")

        start_time = time.perf_counter()
        # Simulated network latency
        await asyncio.sleep(self.base_latency_ms / 1000.0)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        p_tokens = count_tokens(prompt_text)
        out_tokens = request.estimated_output_tokens or 60

        response_content = (
            f"[{self.provider_name} Model {model_spec.model_id} via {replica_id}]: "
            f"Response for task {request.task_type}."
        )

        return {
            "content": response_content,
            "prompt_tokens": p_tokens,
            "output_tokens": out_tokens,
            "latency_ms": round(latency_ms, 2),
            "replica_id": replica_id,
            "provider": self.provider_name,
            "source": "mock_external",
        }
