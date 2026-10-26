"""OpenAI-Compatible / vLLM Provider Backend Adapter (Week-4 Revamp).

Connects to self-hosted vLLM, TensorRT-LLM, or any OpenAI-compatible server at the
selected deployment/replica endpoint using standard /v1/chat/completions.
Reads Authorization token from candidate's credential_reference if configured.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from adaroute.schemas import RequestContext
from adaroute.workloads.tokenizer import count_tokens

from .base import BackendResult, BaseBackend


class OpenAICompatibleUnavailableError(Exception):
    """Raised when an OpenAI-compatible / vLLM endpoint is unreachable."""


class OpenAICompatibleBackend(BaseBackend):
    def __init__(self, default_endpoint: str = "http://localhost:8000/v1", timeout_sec: float = 30.0):
        self.default_endpoint = default_endpoint.rstrip("/")
        self.timeout_sec = timeout_sec

    async def execute(
        self,
        request: RequestContext,
        model_spec: Any = None,
        replica_id: str = "none",
        prompt_text: str = "",
        messages: list[dict[str, str]] | None = None,
        candidate: Any = None,
        endpoint: str | None = None,
        **kwargs: Any,
    ) -> BackendResult:
        start_time = time.perf_counter()

        # Resolve endpoint from replica, candidate, or default
        eff_endpoint = endpoint or (candidate.endpoint if candidate else None) or (
            getattr(model_spec, "base_url", None) if model_spec else None
        ) or self.default_endpoint
        eff_endpoint = eff_endpoint.rstrip("/")
        if not eff_endpoint.endswith("/v1"):
            eff_endpoint = f"{eff_endpoint}/v1"

        target_model = (
            getattr(candidate, "provider_model_name", None)
            or getattr(candidate, "model_id", None)
            or getattr(model_spec, "model_id", "vllm-model")
        )

        headers: dict[str, str] = {"Content-Type": "application/json"}
        cred_ref = getattr(candidate, "credential_reference", None) or getattr(model_spec, "api_key_env", None)
        if cred_ref:
            token = os.getenv(cred_ref)
            if token:
                headers["Authorization"] = f"Bearer {token}"

        chat_messages = messages or [{"role": "user", "content": prompt_text}]
        p_tokens = count_tokens(prompt_text) if prompt_text else sum(count_tokens(m.get("content", "")) for m in chat_messages)

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": chat_messages,
            "temperature": getattr(request, "temperature", 0.7),
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                res = await client.post(f"{eff_endpoint}/chat/completions", json=payload, headers=headers)
                if res.status_code != 200:
                    raise OpenAICompatibleUnavailableError(
                        f"OpenAI-compatible endpoint {eff_endpoint} returned HTTP {res.status_code}: {res.text}"
                    )
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                prompt_tok = usage.get("prompt_tokens", p_tokens)
                completion_tok = usage.get("completion_tokens", count_tokens(content))
                lat_ms = (time.perf_counter() - start_time) * 1000.0

                in_p = getattr(candidate, "input_price_per_1k", 0.0) or getattr(model_spec, "input_price_per_1k", 0.0)
                out_p = getattr(candidate, "output_price_per_1k", 0.0) or getattr(model_spec, "output_price_per_1k", 0.0)
                cost = (prompt_tok / 1000.0) * in_p + (completion_tok / 1000.0) * out_p

                return BackendResult(
                    content=content,
                    prompt_tokens=prompt_tok,
                    output_tokens=completion_tok,
                    latency_ms=lat_ms,
                    replica_id=replica_id,
                    provider="openai_compatible",
                    model_id=target_model,
                    deployment_id=getattr(candidate, "deployment_id", "vllm-deploy"),
                    actual_cost=cost,
                    cost_state="allocated_estimate",
                    raw_response=data,
                )
        except Exception as e:
            if isinstance(e, OpenAICompatibleUnavailableError):
                raise
            raise OpenAICompatibleUnavailableError(f"Failed to connect to OpenAI-compatible endpoint at {eff_endpoint}: {e}") from e
