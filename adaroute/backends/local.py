"""Local open-model backend for AdaRoute (Hardened Week-3 Version).

Connects to local Ollama runtime (e.g. qwen2.5:1.5b, deepseek-r1:1.5b).
Supports explicit execution modes:
- "real": strictly requires real local Ollama; fails closed (no fallback) if unavailable.
- "simulation": explicit simulated test mode.
- "hybrid": tries real Ollama, falls back with transparent labeling.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from adaroute.schemas import ModelSpec, RequestContext
from adaroute.workloads.tokenizer import count_tokens

from .base import BaseBackend


class LocalBackendUnavailableError(Exception):
    """Raised when real Ollama instance is unreachable in real execution mode."""


class LocalBackend(BaseBackend):
    def __init__(
        self,
        ollama_host: str | None = None,
        timeout_sec: float = 30.0,
        execution_mode: str | None = None,
    ):
        self.ollama_host = (
            ollama_host or os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        ).rstrip("/")
        self.timeout_sec = timeout_sec
        self.execution_mode = (execution_mode or os.getenv("EXECUTION_MODE", "hybrid")).lower()

    async def ping(self) -> bool:
        """Pings local Ollama server to verify availability."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.ollama_host}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    async def execute(
        self,
        request: RequestContext,
        model_spec: ModelSpec,
        replica_id: str,
        prompt_text: str,
        messages: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()

        # If strict simulation mode requested, skip real network attempt
        if self.execution_mode == "simulation":
            return await self._execute_simulation(request, model_spec, replica_id, prompt_text, start_time)

        # Resolve model name for Ollama
        target_model = model_spec.model_id
        if target_model in ("qwen-2.5-local", "qwen2.5-local"):
            target_model = "qwen2.5:1.5b"
        elif target_model in ("deepseek-r1-local", "deepseek-local"):
            target_model = "deepseek-r1:1.5b"

        # Structured messages preserving native conversational roles
        chat_messages = messages or [{"role": "user", "content": prompt_text}]

        # Attempt real Ollama chat completion
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                payload: dict[str, Any] = {
                    "model": target_model,
                    "messages": chat_messages,
                    "stream": False,
                    "options": {},
                }
                if request.max_tokens:
                    payload["options"]["num_predict"] = request.max_tokens
                if request.temperature is not None:
                    payload["options"]["temperature"] = request.temperature

                res = await client.post(
                    f"{self.ollama_host}/api/chat",
                    json=payload,
                )

                if res.status_code == 200:
                    data = res.json()
                    message = data.get("message", {})
                    response_text = message.get("content", "")
                    latency_ms = (time.perf_counter() - start_time) * 1000.0

                    # Extract native metrics from Ollama response
                    prompt_eval_count = data.get("prompt_eval_count")
                    eval_count = data.get("eval_count")

                    p_tokens = prompt_eval_count if prompt_eval_count is not None else count_tokens(prompt_text)
                    out_tokens = eval_count if eval_count is not None else max(1, count_tokens(response_text))

                    return {
                        "content": response_text,
                        "prompt_tokens": p_tokens,
                        "output_tokens": out_tokens,
                        "latency_ms": round(latency_ms, 2),
                        "replica_id": replica_id,
                        "model": target_model,
                        "provider": "local_ollama",
                        "source": "ollama_real",
                        "hardware": "cpu",
                        "model_digest": data.get("model_digest") or f"sha256:{hash(target_model)}",
                        "total_duration_ns": data.get("total_duration", 0),
                    }
                else:
                    if self.execution_mode == "real":
                        raise LocalBackendUnavailableError(f"Ollama returned HTTP {res.status_code}: {res.text}")
        except Exception as e:
            if self.execution_mode == "real":
                raise LocalBackendUnavailableError(f"Real Ollama instance at {self.ollama_host} is unreachable: {e!s}")

        # Fallback to simulation engine in hybrid or offline mode
        return await self._execute_simulation(request, model_spec, replica_id, prompt_text, start_time)

    async def _execute_simulation(
        self,
        request: RequestContext,
        model_spec: ModelSpec,
        replica_id: str,
        prompt_text: str,
        start_time: float,
    ) -> dict[str, Any]:
        await asyncio.sleep(0.01)  # 10ms simulated local CPU compute
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        p_tokens = count_tokens(prompt_text)
        out_tokens = min(request.max_tokens or 50, request.estimated_output_tokens or 50)

        response_content = (
            f"Simulated response from [Local Model {model_spec.model_id} via {replica_id}]: "
            f"Successfully processed {request.task_type} request with privacy={request.privacy_level.value}."
        )

        return {
            "content": response_content,
            "prompt_tokens": p_tokens,
            "output_tokens": max(1, out_tokens),
            "latency_ms": round(latency_ms, 2),
            "replica_id": replica_id,
            "model": model_spec.model_id,
            "provider": model_spec.provider,
            "source": "simulated_local_cpu",
            "hardware": "simulated_cpu",
        }
