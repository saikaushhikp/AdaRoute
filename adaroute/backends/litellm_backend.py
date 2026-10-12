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
    ) -> dict[str, Any]:
        start_time = time.perf_counter()

        # If strict simulation mode requested, skip external network call
        if self.execution_mode == "simulation":
            return await self._execute_simulation(request, model_spec, replica_id, prompt_text, start_time)

        # Ensure API keys are loaded
        load_dotenv()

        target_model = model_spec.model_id
        # Map generic external aliases to working verified models
        if target_model in ("gpt-4o-external", "mock-external-flagship"):
            # For gpt-4o-external, if OpenAI key has no credits, Groq qwen3.8-27b provides flagship public serving
            if os.getenv("GROQ_API_KEY"):
                target_model = "groq/qwen/qwen3.8-27b"

        try:
            messages = [{"role": "user", "content": prompt_text}]
            kwargs: dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "max_tokens": request.max_tokens or 256,
                "timeout": 30.0,
            }
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature

            # Explicitly pass provider API keys and guard against missing keys in hybrid mode
            if "gemini" in target_model.lower() or "google" in model_spec.provider.lower():
                gem_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not gem_key:
                    if self.execution_mode == "real":
                        raise ExternalBackendUnavailableError("Google/Gemini API key missing.")
                    return await self._execute_simulation(request, model_spec, replica_id, prompt_text, start_time)
                kwargs["api_key"] = gem_key
                os.environ["GEMINI_API_KEY"] = gem_key

            elif "groq" in target_model.lower():
                groq_key = os.getenv("GROQ_API_KEY")
                if not groq_key:
                    if self.execution_mode == "real":
                        raise ExternalBackendUnavailableError("Groq API key missing.")
                    return await self._execute_simulation(request, model_spec, replica_id, prompt_text, start_time)
                kwargs["api_key"] = groq_key

            elif "cohere" in target_model.lower():
                cohere_key = os.getenv("COHERE_API_KEY") or os.getenv("COHERE_CHAT_API_KEY")
                if not cohere_key:
                    if self.execution_mode == "real":
                        raise ExternalBackendUnavailableError("Cohere API key missing.")
                    return await self._execute_simulation(request, model_spec, replica_id, prompt_text, start_time)
                kwargs["api_key"] = cohere_key
                os.environ["COHERE_API_KEY"] = cohere_key

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
            # Raise so api.py circuit breaker and retry failover can engage alternate candidate providers
            raise ExternalBackendUnavailableError(f"LiteLLM call to {target_model} failed: {e!s}") from e

    async def _execute_simulation(
        self,
        request: RequestContext,
        model_spec: ModelSpec,
        replica_id: str,
        prompt_text: str,
        start_time: float,
    ) -> dict[str, Any]:
        await asyncio.sleep(0.03)  # 30ms simulated external network delay
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        p_tokens = count_tokens(prompt_text)
        out_tokens = min(request.max_tokens or 60, request.estimated_output_tokens or 60)

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
            "source": "simulated_external_cloud",
            "actual_cost": cost,
        }
