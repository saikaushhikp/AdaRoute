from __future__ import annotations

from adaroute.schemas import BackendState, PrivacyLevel, RequestContext


class DeterministicRouter:
    def __init__(self, backends: list[BackendState]):
        self.backends = backends

    def route(self, request: RequestContext) -> BackendState | None:
        # Filter out unavailable backends
        available = [b for b in self.backends if b.availability]
        if not available:
            return None

        # P0 FIX: Canonical Privacy Enforcement (Fail Closed!)
        # Any privacy level other than PUBLIC must NEVER route to external providers.
        # This covers INTERNAL_ONLY, CONFIDENTIAL, RESTRICTED.
        if request.privacy_level != PrivacyLevel.PUBLIC:
            available = [b for b in available if "external" not in b.provider.lower()]

        # Hard Constraint: Budget
        cost_map = {
            "gpt-4o": 0.05,
            "gpt-4o-external": 0.05,
            "llama-3": 0.001,
            "llama-3-8b": 0.001,
            "qwen-2.5-local": 0.001,
        }
        available = [b for b in available if cost_map.get(b.model, 0.001) <= request.budget]

        if not available:
            return None

        # Deterministic Choice: Choose candidate with lowest queue depth
        available.sort(key=lambda x: x.queue_depth)

        return available[0]
