"""Bounded Retry and Failover Controller.

Limits maximum attempts, distinguishes transient from permanent errors,
and governs policy-safe fallbacks.
"""

import asyncio


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 2,
        initial_backoff_sec: float = 0.05,
        backoff_multiplier: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.initial_backoff_sec = initial_backoff_sec
        self.backoff_multiplier = backoff_multiplier

    def is_retryable(self, exc: Exception) -> bool:
        """Only retry transient timeouts, connection drops, or 5xx failures."""
        if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)):
            return True
        msg = str(exc).lower()
        return bool(any(code in msg for code in ("500", "502", "503", "504", "timeout", "temporarily unavailable")))
