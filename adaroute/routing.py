from __future__ import annotations

from adaroute.schemas import BackendState, RequestContext


class PolicyEngine:
    def __init__(self, replicas: list[BackendState]):
        self.replicas = replicas

    def route(self, request: RequestContext) -> BackendState | None:
        pass


class StrongestModelPolicy(PolicyEngine):
    def route(self, request: RequestContext) -> BackendState | None:
        # Sort by capability (assuming 'model' encodes capability for simplicity)
        healthy = [r for r in self.replicas if r.availability]
        if not healthy:
            return None
        return healthy[0]  # Simplistic for prototype


class CheapestModelPolicy(PolicyEngine):
    def route(self, request: RequestContext) -> BackendState | None:
        healthy = [r for r in self.replicas if r.availability]
        if not healthy:
            return None
        # Simplistic: return first healthy
        return healthy[0]
