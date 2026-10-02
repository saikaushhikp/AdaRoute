"""Circuit Breaker Pattern for Backend Protection.

Prevents retry storms by transitioning through CLOSED -> OPEN -> HALF_OPEN states.
"""

import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing fast, requests rejected immediately
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

    def get_state(self, resource_id: str) -> CircuitState:
        state = self._states.get(resource_id, CircuitState.CLOSED)
        if state == CircuitState.OPEN:
            elapsed = time.time() - self._last_state_change.get(resource_id, 0.0)
            if elapsed >= self.recovery_timeout_sec:
                # Transition to HALF_OPEN
                self._states[resource_id] = CircuitState.HALF_OPEN
                self._last_state_change[resource_id] = time.time()
                return CircuitState.HALF_OPEN
        return state

    def allow_request(self, resource_id: str) -> bool:
        """Check if request is allowed to pass to backend."""
        state = self.get_state(resource_id)
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return True  # Allow canary request
        return False  # OPEN rejects traffic immediately

    def record_success(self, resource_id: str):
        self._failure_counts[resource_id] = 0
        if self._states.get(resource_id) in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            self._states[resource_id] = CircuitState.CLOSED
            self._last_state_change[resource_id] = time.time()

    def record_failure(self, resource_id: str):
        count = self._failure_counts.get(resource_id, 0) + 1
        self._failure_counts[resource_id] = count
        if count >= self.failure_threshold:
            self._states[resource_id] = CircuitState.OPEN
            self._last_state_change[resource_id] = time.time()
