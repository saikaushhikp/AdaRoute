"""LiteLLM Gateway Backend for External Cloud Providers.
Handles external API calls via LiteLLM to Groq, Cohere, etc., using authentic API keys.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import litellm
from dotenv import load_dotenv

from adaroute.schemas import ModelSpec, RequestContext
from adaroute.workloads.tokenizer import count_tokens

from .base import BaseBackend

load_dotenv()


class ExternalBackendUnavailableError(Exception):
    """Raised when external provider API fails or keys are missing in real mode."""


class LiteLLMBackend(BaseBackend):
    def __init__(self, execution_mode: str | None = None):
        self.execution_mode = (execution_mode or os.getenv("EXECUTION_MODE", "hybrid")).lower()

    async def execute(
        self,
        request: RequestContext,
        model_spec: ModelSpec,
        replica_id: str,
        prompt_text: str,
        messages: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()

        # If strict simulation mode requested, skip external network call
        if self.execution_mode == "simulation":
            return await self._execute_simulation(request, model_spec, replica_id, prompt_text, start_time)

        # Ensure API keys are loaded
        load_dotenv()

        target_model = model_spec.litellm_model or model_spec.model_id
        # Map generic external mock aliases to available configured models if requested
        if target_model in ("mock/gpt-4o", "gpt-4o-external", "mock-external-flagship"):
            if os.getenv("GROQ_API_KEY"):
                target_model = "groq/qwen/qwen3.8-27b"
            elif os.getenv("OPENAI_API_KEY"):
                target_model = "gpt-4o"

        try:
            chat_messages = messages or [{"role": "user", "content": prompt_text}]
            timeout_sec = float(os.getenv("EXTERNAL_TIMEOUT_SEC", "12.0"))
            kwargs: dict[str, Any] = {
                "model": target_model,
                "messages": chat_messages,
                "max_tokens": request.max_tokens or 256,
                "timeout": timeout_sec,
            }
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature

            # Dynamically resolve credential reference from model_spec declarative configuration
            api_key = None
            if model_spec.api_key_env:
                api_key = os.getenv(model_spec.api_key_env)

            # Generic fallback credential discovery if not explicitly referenced
            if not api_key:
                p_upper = model_spec.provider.upper()
                api_key = (
                    os.getenv(f"{p_upper}_API_KEY")
                    or os.getenv(f"{p_upper}_CHAT_API_KEY")
                    or os.getenv("GOOGLE_API_KEY")
                    or os.getenv("GEMINI_API_KEY")
                )

            if api_key:
                kwargs["api_key"] = api_key
                if model_spec.api_key_env:
                    os.environ[model_spec.api_key_env] = api_key
            else:
                if self.execution_mode in ("real", "hybrid"):
                    raise ExternalBackendUnavailableError(
                        f"API key missing for {model_spec.model_id} (expected {model_spec.api_key_env or 'API key'})."
                    )
                return await self._execute_simulation(request, model_spec, replica_id, prompt_text, start_time)

            resp = await litellm.acompletion(**kwargs)

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            choice = resp.choices[0]
            content = choice.message.content or ""

            usage = getattr(resp, "usage", None)
            p_tokens = getattr(usage, "prompt_tokens", None) or count_tokens(prompt_text)
            out_tokens = getattr(usage, "completion_tokens", None) or count_tokens(content)

            # Compute actual cost via litellm or model spec pricing
            actual_cost = 0.0
            try:
                actual_cost = litellm.completion_cost(completion_response=resp)
            except Exception:
                actual_cost = (p_tokens / 1000.0) * model_spec.input_price_per_1k + (
                    out_tokens / 1000.0
                ) * model_spec.output_price_per_1k

            return {
                "content": content,
                "prompt_tokens": p_tokens,
                "output_tokens": max(1, out_tokens),
                "latency_ms": round(latency_ms, 2),
                "replica_id": replica_id,
                "model": target_model,
                "provider": model_spec.provider,
                "source": "litellm_external_real",
                "actual_cost": actual_cost,
                "response_id": getattr(resp, "id", None),
            }
        except Exception as e:
            if self.execution_mode in ("real", "hybrid"):
                # Strict failover: raise error so router tries next candidate model
                raise ExternalBackendUnavailableError(f"LiteLLM call to {target_model} failed: {e!s}") from e

            # In pure simulation mode: fallback to simulation engine
            return await self._execute_simulation(
                request,
                model_spec,
                replica_id,
                prompt_text,
                start_time,
                fallback_reason=str(e),
            )

    async def _execute_simulation(
        self,
        request: RequestContext,
        model_spec: ModelSpec,
        replica_id: str,
        prompt_text: str,
        start_time: float,
        fallback_reason: str | None = None,
    ) -> dict[str, Any]:
        await asyncio.sleep(0.02)  # 20ms simulated external network delay
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        p_tokens = count_tokens(prompt_text)
        out_tokens = min(request.max_tokens or 60, request.estimated_output_tokens or 60)

        source_label = "simulated_external_cloud_fallback" if fallback_reason else "simulated_external_cloud"
        response_content = (
            f"Simulated cloud response from [Provider {model_spec.provider} / {model_spec.model_id} via {replica_id}]: "
            f"Successfully processed public request."
        )

        cost = (p_tokens / 1000.0) * model_spec.input_price_per_1k + (
            out_tokens / 1000.0
        ) * model_spec.output_price_per_1k

        return {
            "content": response_content,
            "prompt_tokens": p_tokens,
            "output_tokens": max(1, out_tokens),
            "latency_ms": round(latency_ms, 2),
            "replica_id": replica_id,
            "model": model_spec.model_id,
            "provider": model_spec.provider,
            "source": source_label,
            "actual_cost": cost,
            "fallback_reason": fallback_reason,
        }
