import asyncio
import random

from adaroute.schemas import RequestContext


class MockBackend:
    def __init__(self, replica_id: str, latency_ms: float = 100):
        self.replica_id = replica_id
        self.latency_ms = latency_ms

    async def generate(self, request: RequestContext):
        # Simulate processing time
        await asyncio.sleep(self.latency_ms / 1000.0)
        # Simulate varying output length based on estimate
        actual_output = max(1, int(random.gauss(request.estimated_output_tokens, 10)))
        return {"replica_id": self.replica_id, "output_tokens": actual_output}
