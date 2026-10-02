from __future__ import annotations

from adaroute.policy.model_selection import (
    AdaptiveModelPolicy,
    StaticRulePolicy,
)
from adaroute.policy.model_selection import (
    CheapestModelPolicy as ModularCheapestPolicy,
)
from adaroute.policy.model_selection import (
    StrongestModelPolicy as ModularStrongestPolicy,
)
from adaroute.schemas import BackendState, RequestContext


class PolicyEngine:
    def __init__(self, replicas: list[BackendState]):
        self.replicas = replicas

    def route(self, request: RequestContext) -> BackendState | None:
        pass


class StrongestModelPolicy(PolicyEngine):
    """Legacy interface backed by real capability quality ordering."""

    CAPABILITY_RANKS = {
        "gpt-4o": 100,
        "gpt-4": 95,
        "llama3": 70,
        "llama-3-8b": 70,
        "qwen-2.5-local": 75,
    }

    def route(self, request: RequestContext) -> BackendState | None:
        healthy = [r for r in self.replicas if r.availability]
        if not healthy:
            return None
        # Sort by capability rank descending
        healthy.sort(key=lambda r: self.CAPABILITY_RANKS.get(r.model, 50), reverse=True)
        return healthy[0]


class CheapestModelPolicy(PolicyEngine):
    """Legacy interface backed by real pricing ordering."""

    COST_RANKS = {
        "llama3": 0.001,
        "llama-3-8b": 0.001,
        "qwen-2.5-local": 0.001,
        "gpt-4": 0.03,
        "gpt-4o": 0.03,
    }

    def route(self, request: RequestContext) -> BackendState | None:
        healthy = [r for r in self.replicas if r.availability]
        if not healthy:
            return None
        # Sort by cost ascending
        healthy.sort(key=lambda r: self.COST_RANKS.get(r.model, 1.0))
        return healthy[0]


__all__ = [
    "AdaptiveModelPolicy",
    "CheapestModelPolicy",
    "ModularCheapestPolicy",
    "ModularStrongestPolicy",
    "PolicyEngine",
    "StaticRulePolicy",
    "StrongestModelPolicy",
]
