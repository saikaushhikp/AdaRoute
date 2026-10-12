"""Circuit Breaker Pattern for Backend Protection (Hardened Week-3 Version).

Prevents retry storms by transitioning through CLOSED -> OPEN -> HALF_OPEN states.
Enforces single-probe throttling in HALF_OPEN state.
"""

from __future__ import annotations

import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, requests rejected immediately
    HALF_OPEN = "half_open"  # Testing canary requests to assess recovery


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_sec: float = 10.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self._states: dict[str, CircuitState] = {}
        self._failure_counts: dict[str, int] = {}
        self._last_state_change: dict[str, float] = {}
        self._half_open_probes: dict[str, int] = {}

    def reset(self, resource_id: str | None = None) -> None:
        if resource_id:
            self._states.pop(resource_id, None)
            self._failure_counts.pop(resource_id, None)
            self._last_state_change.pop(resource_id, None)
            self._half_open_probes.pop(resource_id, None)
        else:
            self._states.clear()
            self._failure_counts.clear()
            self._last_state_change.clear()
            self._half_open_probes.clear()

    def clear(self) -> None:
        self.reset()

    def get_state(self, resource_id: str) -> CircuitState:
        state = self._states.get(resource_id, CircuitState.CLOSED)
        if state == CircuitState.OPEN:
            elapsed = time.time() - self._last_state_change.get(resource_id, 0.0)
            if elapsed >= self.recovery_timeout_sec:
                # Transition to HALF_OPEN
                self._states[resource_id] = CircuitState.HALF_OPEN
                self._last_state_change[resource_id] = time.time()
                self._half_open_probes[resource_id] = 0
                return CircuitState.HALF_OPEN
        return state

    def allow_request(self, resource_id: str) -> bool:
        """Check if request is allowed to pass to backend."""
        state = self.get_state(resource_id)
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            # Single-probe throttling: allow only 1 concurrent probe request
            probes = self._half_open_probes.get(resource_id, 0)
            if probes < 1:
                self._half_open_probes[resource_id] = probes + 1
                return True
            return False
        return False  # OPEN rejects traffic immediately

    def record_success(self, resource_id: str) -> None:
        self._failure_counts[resource_id] = 0
        self._half_open_probes[resource_id] = 0
        if self._states.get(resource_id) in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            self._states[resource_id] = CircuitState.CLOSED
            self._last_state_change[resource_id] = time.time()

    def record_failure(self, resource_id: str) -> None:
        count = self._failure_counts.get(resource_id, 0) + 1
        self._failure_counts[resource_id] = count
        self._half_open_probes[resource_id] = 0
        if count >= self.failure_threshold:
            self._states[resource_id] = CircuitState.OPEN
            self._last_state_change[resource_id] = time.time()
