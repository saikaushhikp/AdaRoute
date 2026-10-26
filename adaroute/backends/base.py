"""Base Protocol and Result Contracts for AdaRoute Provider Adapters (Week-4 Revamp)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from adaroute.schemas import RequestContext


@dataclass
class BackendResult:
    """Standardized response produced by any provider backend adapter."""

    content: str
    prompt_tokens: int
    output_tokens: int
    latency_ms: float
    replica_id: str
    provider: str
    model_id: str
    deployment_id: str = "none"
    actual_cost: float | None = None
    cost_state: str = "allocated_estimate"  # "invoiced_real", "allocated_estimate", "unknown_unbilled"
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "replica_id": self.replica_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "deployment_id": self.deployment_id,
            "actual_cost": self.actual_cost,
            "cost_state": self.cost_state,
        }

    def __getitem__(self, item: str) -> Any:
        # Compatibility with legacy dictionary access
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class BaseBackend(ABC):
    """Abstract base provider adapter."""

    @abstractmethod
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
    ) -> dict[str, Any] | BackendResult:
        """Execute inference on the provider backend."""
