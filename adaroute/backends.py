from __future__ import annotations

import asyncio
import random
from typing import Any

from adaroute.backends.base import BaseBackend
from adaroute.backends.local import LocalBackend
from adaroute.backends.mock import MockBackend
from adaroute.schemas import RequestContext


class LegacyMockBackend:
    def __init__(self, replica_id: str, latency_ms: float = 100):
        self.replica_id = replica_id
        self.latency_ms = latency_ms

    async def generate(self, request: RequestContext) -> dict[str, Any]:
        await asyncio.sleep(self.latency_ms / 1000.0)
        actual_output = max(1, int(random.gauss(request.estimated_output_tokens, 10)))
        return {"replica_id": self.replica_id, "output_tokens": actual_output}


__all__ = ["BaseBackend", "LegacyMockBackend", "LocalBackend", "MockBackend"]
