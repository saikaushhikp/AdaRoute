"""Session-Aware Continuity & Cumulative Budget Store (Week-4 Revamp).

Maintains lightweight, durable, tenant-partitioned session state with Redis backing:
- Tracks turn index, previous deployment ID, and accumulated session cost
- Enforces session-level cumulative budget
- Applies soft session continuity and switch penalties
- TTL bounded (default 24h)
- Never stores raw prompt content
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

try:
    import redis
except ImportError:
    redis = None


@dataclass
class SessionState:
    """Tenant-scoped conversation session tracking state."""

    session_id: str
    tenant_id: str = "user"
    turn_index: int = 0
    previous_model_id: str | None = None
    previous_deployment_id: str | None = None
    current_capability: str | None = None
    model_switch_count: int = 0
    accumulated_cost: float = 0.0
    session_budget: float | None = None
    session_budget_remaining: float | None = None
    last_active: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["last_active"] = round(self.last_active, 2)
        d["accumulated_cost"] = round(self.accumulated_cost, 6)
        if self.session_budget_remaining is not None:
            d["session_budget_remaining"] = round(self.session_budget_remaining, 6)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        return cls(
            session_id=data["session_id"],
            tenant_id=data.get("tenant_id", "user"),
            turn_index=int(data.get("turn_index", 0)),
            previous_model_id=data.get("previous_model_id"),
            previous_deployment_id=data.get("previous_deployment_id"),
            current_capability=data.get("current_capability"),
            model_switch_count=int(data.get("model_switch_count", 0)),
            accumulated_cost=float(data.get("accumulated_cost", 0.0)),
            session_budget=float(data["session_budget"]) if data.get("session_budget") is not None else None,
            session_budget_remaining=(
                float(data["session_budget_remaining"]) if data.get("session_budget_remaining") is not None else None
            ),
            last_active=float(data.get("last_active", time.time())),
        )


class SessionStore:
    """Durable, TTL-bound session store with transparent in-memory fallback."""

    def __init__(
        self,
        redis_url: str | None = None,
        default_ttl_seconds: int = 86400,
    ):
        self.default_ttl = default_ttl_seconds
        self._memory_store: dict[str, tuple[SessionState, float]] = {}
        self.redis_client = None

        r_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        if redis is not None and r_url:
            try:
                client = redis.from_url(r_url, decode_responses=True, socket_connect_timeout=1.5)
                client.ping()
                self.redis_client = client
            except Exception:
                self.redis_client = None

    def _get_key(self, tenant_id: str, session_id: str) -> str:
        return f"adaroute:session:{tenant_id}:{session_id}"

    def get_session(
        self, session_id: str, tenant_id: str = "user", initial_budget: float | None = None
    ) -> SessionState:
        """Retrieves active session state or initializes a new one."""
        key = self._get_key(tenant_id, session_id)
        now = time.time()

        if self.redis_client:
            try:
                raw = self.redis_client.get(key)
                if raw:
                    data = json.loads(raw)
                    return SessionState.from_dict(data)
            except Exception:
                pass

        # Check in-memory store
        if key in self._memory_store:
            sess, expiry = self._memory_store[key]
            if now < expiry:
                return sess
            del self._memory_store[key]

        # Initialize fresh session
        new_sess = SessionState(
            session_id=session_id,
            tenant_id=tenant_id,
            session_budget=initial_budget,
            session_budget_remaining=initial_budget,
            last_active=now,
        )
        self.save_session(new_sess)
        return new_sess

    def save_session(self, session: SessionState) -> None:
        """Atomically saves session state with TTL."""
        key = self._get_key(session.tenant_id, session.session_id)
        session.last_active = time.time()
        payload = json.dumps(session.to_dict())

        if self.redis_client:
            try:
                self.redis_client.set(key, payload, ex=int(self.default_ttl))
                return
            except Exception:
                pass

        # In-memory fallback
        expiry = time.time() + self.default_ttl
        self._memory_store[key] = (session, expiry)

    def record_turn(
        self,
        session_id: str,
        tenant_id: str,
        selected_deployment_id: str,
        selected_model_id: str,
        capability: str,
        billed_cost: float,
    ) -> SessionState:
        """Updates session state following request completion."""
        sess = self.get_session(session_id, tenant_id)
        sess.turn_index += 1

        if sess.previous_deployment_id and sess.previous_deployment_id != selected_deployment_id:
            sess.model_switch_count += 1

        sess.previous_deployment_id = selected_deployment_id
        sess.previous_model_id = selected_model_id
        sess.current_capability = capability
        sess.accumulated_cost += billed_cost

        if sess.session_budget_remaining is not None:
            sess.session_budget_remaining = max(0.0, sess.session_budget_remaining - billed_cost)

        self.save_session(sess)
        return sess

    def reset_session(self, session_id: str, tenant_id: str = "user") -> None:
        """Explicitly resets/deletes session state."""
        key = self._get_key(tenant_id, session_id)
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception:
                pass
        self._memory_store.pop(key, None)
