"""vLLM Provider Backend Adapter for AdaRoute.

Connects to a self-hosted vLLM server exposing the OpenAI-compatible
/v1/chat/completions endpoint (default port 7000 inside Docker, exposed
as 7001 on the host via docker-compose).

Execution modes:
- "real"       : strictly requires a reachable vLLM endpoint; fails closed.
- "simulation" : explicit simulated/test mode — no network calls.
- "hybrid"     : tries real vLLM, transparent simulation fallback if unreachable.

Environment variables (checked in order):
  VLLM_BASE_URL   – e.g. http://vllm:7000/v1
  VLLM_HOST       – base host without path (appended with /v1 automatically)
  VLLM_API_KEY    – optional Bearer token for secured vLLM deployments
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


class VLLMBackendUnavailableError(Exception):
    """Raised when the vLLM endpoint is unreachable in real execution mode."""


class VLLMBackend(BaseBackend):
    """Backend adapter that routes inference to a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_sec: float = 60.0,
        execution_mode: str | None = None,
    ) -> None:
        # Resolve base URL: env > constructor arg > Docker-compose service default
        raw_url = (base_url or os.getenv("VLLM_BASE_URL") or os.getenv("VLLM_HOST") or "http://vllm:7000").rstrip("/")
        # Normalise: always end with /v1
        self.base_url = raw_url if raw_url.endswith("/v1") else f"{raw_url}/v1"
        self.timeout_sec = timeout_sec
        self.execution_mode = (execution_mode or os.getenv("EXECUTION_MODE", "hybrid")).lower()

        # Optional bearer token for secured vLLM deployments
        self._api_key: str | None = os.getenv("VLLM_API_KEY")

    # ------------------------------------------------------------------
    # Health probe
    # ------------------------------------------------------------------

    async def ping(self, endpoint: str | None = None) -> bool:
        """Returns True when the vLLM server responds to its /v1/models probe."""
        try:
            base = endpoint.rstrip("/") if endpoint else self.base_url
            if not base.endswith("/v1"):
                base = f"{base}/v1"
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(
                    f"{base}/models",
                    headers=self._auth_headers(),
                )
                return res.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        request: RequestContext,
        model_spec: ModelSpec | None = None,
        replica_id: str = "none",
        prompt_text: str = "",
        messages: list[dict[str, str]] | None = None,
        candidate: Any = None,
        endpoint: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()
        eff_mode = (getattr(request, "execution_mode", None) or self.execution_mode).lower()

        if eff_mode == "simulation":
            return await self._simulate(request, model_spec, replica_id, prompt_text, start_time)

        # ---- Resolve endpoint & model name ----
        eff_base = self._resolve_base_url(endpoint, candidate, model_spec)
        target_model = self._resolve_model_name(candidate, model_spec)

        chat_messages = messages or [{"role": "user", "content": prompt_text}]
        p_tokens = (
            count_tokens(prompt_text) if prompt_text else sum(count_tokens(m.get("content", "")) for m in chat_messages)
        )

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": chat_messages,
            "temperature": getattr(request, "temperature", 0.7),
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                res = await client.post(
                    f"{eff_base}/chat/completions",
                    json=payload,
                    headers=self._auth_headers(candidate, model_spec),
                )
                if res.status_code != 200:
                    raise VLLMBackendUnavailableError(
                        f"vLLM endpoint {eff_base} returned HTTP {res.status_code}: {res.text}"
                    )

                data = res.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                prompt_tok = usage.get("prompt_tokens", p_tokens)
                completion_tok = usage.get("completion_tokens", count_tokens(content))
                lat_ms = (time.perf_counter() - start_time) * 1000.0

                in_p = getattr(candidate, "input_price_per_1k", 0.0) or getattr(model_spec, "input_price_per_1k", 0.0)
                out_p = getattr(candidate, "output_price_per_1k", 0.0) or getattr(
                    model_spec, "output_price_per_1k", 0.0
                )
                cost = (prompt_tok / 1000.0) * in_p + (completion_tok / 1000.0) * out_p

                return {
                    "content": content,
                    "prompt_tokens": prompt_tok,
                    "output_tokens": completion_tok,
                    "latency_ms": round(lat_ms, 2),
                    "replica_id": replica_id,
                    "model": target_model,
                    "model_id": target_model,
                    "provider": "vllm",
                    "source": "vllm_real",
                    "hardware": "gpu",
                    "actual_cost": cost,
                    "cost_state": "allocated_estimate",
                    "raw_response": data,
                }

        except VLLMBackendUnavailableError:
            if self.execution_mode == "real":
                raise
        except Exception as exc:
            if self.execution_mode == "real":
                raise VLLMBackendUnavailableError(f"Failed to connect to vLLM at {eff_base}: {exc}") from exc

        # Hybrid fallback: transparent simulation
        return await self._simulate(request, model_spec, replica_id, prompt_text, start_time)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auth_headers(
        self,
        candidate: Any = None,
        model_spec: Any = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        # 1. Explicit env var VLLM_API_KEY
        token = self._api_key
        # 2. Credential reference on candidate / model spec
        if not token:
            cred_ref = getattr(candidate, "credential_reference", None) or getattr(model_spec, "api_key_env", None)
            if cred_ref:
                token = os.getenv(cred_ref)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _resolve_base_url(
        self,
        endpoint: str | None,
        candidate: Any,
        model_spec: Any,
    ) -> str:
        raw = (
            endpoint
            or (candidate.endpoint if candidate else None)
            or (getattr(model_spec, "base_url", None) if model_spec else None)
            or self.base_url
        )
        raw = raw.rstrip("/")
        return raw if raw.endswith("/v1") else f"{raw}/v1"

    @staticmethod
    def _resolve_model_name(candidate: Any, model_spec: Any) -> str:
        return (
            getattr(candidate, "provider_model_name", None)
            or getattr(candidate, "model_id", None)
            or (getattr(model_spec, "model_id", None) if model_spec else None)
            or "unsloth/Qwen3-0.6B-GGUF:Q4_K_M"
        )

    async def _simulate(
        self,
        request: RequestContext,
        model_spec: ModelSpec | None,
        replica_id: str,
        prompt_text: str,
        start_time: float,
    ) -> dict[str, Any]:
        """Lightweight deterministic simulation — no real network calls."""
        await asyncio.sleep(0.01)  # 10 ms simulated GPU inference
        lat_ms = (time.perf_counter() - start_time) * 1000.0
        p_tokens = count_tokens(prompt_text)
        out_tokens = min(request.max_tokens or 50, request.estimated_output_tokens or 50)
        model_id = getattr(model_spec, "model_id", "vllm-model") if model_spec else "vllm-model"
        provider = getattr(model_spec, "provider", "vllm") if model_spec else "vllm"

        response_content = (
            f"[Simulated vLLM] {model_id} via {replica_id}: "
            f"Processed {getattr(request, 'task_type', 'general')} request "
            f"(privacy={getattr(getattr(request, 'privacy_level', None), 'value', 'public')})."
        )
        return {
            "content": response_content,
            "prompt_tokens": p_tokens,
            "output_tokens": max(1, out_tokens),
            "latency_ms": round(lat_ms, 2),
            "replica_id": replica_id,
            "model": model_id,
            "model_id": model_id,
            "provider": provider,
            "source": "simulated_vllm",
            "hardware": "simulated_gpu",
            "actual_cost": 0.0,
            "cost_state": "allocated_estimate",
        }
