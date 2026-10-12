"""LiteLLM Gateway Adapter for AdaRoute.

Connects AdaRoute's two-stage policy engine (Eligibility, Model Selection,
and Dynamic Workload Replica Scheduling) with LiteLLM's provider multiplexer
and client execution pipeline.
"""

from __future__ import annotations

import os
from typing import Any

import litellm

from adaroute.policy.eligibility import EligibilityChecker
from adaroute.policy.model_selection import (
    AdaptiveModelPolicy,
    CheapestModelPolicy,
    StaticRulePolicy,
    StrongestModelPolicy,
)
from adaroute.policy.registry import ModelRegistry, default_registry
from adaroute.policy.replica_selection import (
    LeastLoadedScheduler,
    RoundRobinScheduler,
    SLOAwareScheduler,
)
from adaroute.schemas import ChatCompletionRequest, ModelSpec, RequestContext
from adaroute.workloads.estimator import OutputLengthEstimator
from adaroute.workloads.tokenizer import count_tokens


class AdaRouteLiteLLMRouter:
    """Plugs AdaRoute's modular policies into LiteLLM's execution ecosystem."""

    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or default_registry
        self.eligibility = EligibilityChecker(self.registry)
        self.estimator = OutputLengthEstimator()

        self.model_policies = {
            "adaptive": AdaptiveModelPolicy(self.registry),
            "strongest": StrongestModelPolicy(self.registry),
            "cheapest": CheapestModelPolicy(self.registry),
            "static": StaticRulePolicy(self.registry),
        }

        self.replica_schedulers = {
            "slo_aware": SLOAwareScheduler(self.registry),
            "least_loaded": LeastLoadedScheduler(self.registry),
            "round_robin": RoundRobinScheduler(self.registry),
        }

    def route_request(self, request: ChatCompletionRequest) -> tuple[ModelSpec | None, str | None, float]:
        """Stage 1: Evaluates hard constraints and applies requested model policy."""
        prompt_text = request.get_full_prompt()
        p_tokens = count_tokens(prompt_text)
        est_tokens = self.estimator.estimate(
            task_type=request.model,
            prompt_tokens=p_tokens,
            max_tokens=request.max_tokens,
        )

        ctx = RequestContext(
            task_type=request.model,
            prompt_tokens=p_tokens,
            estimated_output_tokens=est_tokens,
            quality_requirement=request.quality_requirement,
            latency_slo=request.latency_slo,
            privacy_level=request.privacy_level,
            budget=request.budget,
            tenant_id=request.tenant_id or "default",
            user_id=request.user_id or "anonymous",
            no_cache=request.no_cache,
            max_tokens=request.max_tokens,
            temperature=request.temperature or 0.7,
        )

        policy = self.model_policies.get(request.policy, self.model_policies["adaptive"])
        selected_spec, est_cost, err_msg = policy.select_model(ctx)
        return selected_spec, err_msg, est_cost

    async def execute_with_litellm(
        self,
        model_spec: ModelSpec,
        request: ChatCompletionRequest,
    ) -> Any:
        """Dispatches selected model call through LiteLLM's native async API."""
        target_model = model_spec.model_id
        if target_model in ("qwen2.5:1.5b", "qwen-2.5-local"):
            # LiteLLM connects to local Ollama via ollama/ prefix
            target_model = "ollama/qwen2.5:1.5b"
        elif target_model in ("deepseek-r1:1.5b", "deepseek-r1-local"):
            target_model = "ollama/deepseek-r1:1.5b"
        elif target_model in ("gpt-4o-external", "mock-external-flagship"):
            if os.getenv("GROQ_API_KEY"):
                target_model = "groq/qwen/qwen3.8-27b"

        messages = [
            {"role": m.role, "content": m.content}
            for m in (request.messages or [{"role": "user", "content": request.get_full_prompt()}])
        ]

        kwargs: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "max_tokens": request.max_tokens or 256,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        return await litellm.acompletion(**kwargs)
