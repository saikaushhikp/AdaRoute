from __future__ import annotations

from adaroute.schemas import BackendState, RequestContext


class DeterministicRouter:
    def __init__(self, backends: list[BackendState]):
        self.backends = backends

    def route(self, request: RequestContext) -> BackendState | None:
        # Filter out unavailable backends
        available = [b for b in self.backends if b.availability]
        if not available:
            return None
            
        # Hard Constraint: Privacy
        # If privacy is high/internal, do not route to external providers
        if request.privacy_level in ["internal", "high"]:
            available = [b for b in available if "external" not in b.provider]
            
        # Hard Constraint: Budget
        # For this mock, assume 'gpt-4o' costs 0.05 and 'llama-3' costs 0.001
        cost_map = {"gpt-4o": 0.05, "llama-3-8b": 0.001}
        available = [b for b in available if cost_map.get(b.model, 1.0) <= request.budget]

        if not available:
            return None

        # Deterministic Choice: Choose the one with the lowest queue depth
        available.sort(key=lambda x: x.queue_depth)
        
        return available[0]
