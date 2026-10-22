"""Base abstract class for all AdaRoute inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from adaroute.schemas import ModelSpec, RequestContext


class BaseBackend(ABC):
    @abstractmethod
    async def execute(
        self,
        request: RequestContext,
        model_spec: ModelSpec,
        replica_id: str,
        prompt_text: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute request on the specified replica.

        Returns dictionary containing:
        - content: str
        - prompt_tokens: int
        - output_tokens: int
        - latency_ms: float
        - replica_id: str
        - provider: str
        """
