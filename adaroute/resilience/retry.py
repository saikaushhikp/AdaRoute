"""Bounded Retry and Failover Controller (Hardened Week-3 Version).

Limits maximum attempts, distinguishes transient from permanent errors,
implements exponential backoff and 429 Retry-After handling.
"""

from __future__ import annotations

import asyncio


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 2,
        initial_backoff_sec: float = 0.05,
        backoff_multiplier: float = 2.0,
        max_total_timeout_sec: float = 15.0,
    ):
        self.max_attempts = max_attempts
        self.initial_backoff_sec = initial_backoff_sec
        self.backoff_multiplier = backoff_multiplier
        self.max_total_timeout_sec = max_total_timeout_sec

    def is_retryable(self, exc: Exception) -> bool:
        """Retry transient timeouts, connection drops, 429 rate limits, or 5xx failures."""
        if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)):
            return True
        msg = str(exc).lower()
        retryable_indicators = (
            "500",
            "502",
            "503",
            "504",
            "429",
            "rate limit",
            "too many requests",
            "timeout",
            "temporarily unavailable",
            "connection refused",
            "connection reset",
        )
        return any(ind in msg for ind in retryable_indicators)

    def get_backoff(self, attempt: int, retry_after: float | None = None) -> float:
        """Computes exponential backoff sleep or uses Retry-After header."""
        if retry_after is not None and retry_after > 0:
            return min(retry_after, 5.0)  # Bound maximum wait to 5s
        return self.initial_backoff_sec * (self.backoff_multiplier ** (attempt - 1))
