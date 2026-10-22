"""Durable Sanitized Outbox Ledger for AdaRoute Telemetry.

Append-only, durable JSONL ledger recording all request and attempt transactions with:
1. Secret and credential sanitization (Bearer, Google API, OpenAI, Anthropic, etc.)
2. Explicit cost states: 'invoiced_real', 'allocated_estimate', 'unknown_unbilled'
3. Transactional durability to disk with thread-safe append
4. Full correlation across multi-attempt execution lifecycles
"""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Secret patterns for scrubbing credentials and proprietary identifiers
SECRET_PATTERNS = [
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"AIza[0-9A-Za-z\-_]{30,}"),
    re.compile(r"sk-ant-[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"gsk_[a-zA-Z0-9]{20,}"),
    re.compile(r"AQ\.[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"key=[a-zA-Z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"api[_-]?key[:=]\s*['\"]?[a-zA-Z0-9_\-\.]+['\"]?", re.IGNORECASE),
    re.compile(r"API_KEY_[A-Za-z0-9_]+"),
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),
]


def sanitize_secrets(text: str | None) -> str:
    """Scrub Bearer tokens, Google/OpenAI API keys, and sensitive tokens from text."""
    if text is None:
        return ""
    sanitized = str(text)
    for pat in SECRET_PATTERNS:
        sanitized = pat.sub("[REDACTED_SECRET]", sanitized)
    return sanitized


class LedgerEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entry_type: str = "request"  # "request" | "attempt"
    request_id: str
    attempt_number: int | None = None
    timestamp: float = Field(default_factory=time.time)
    logical_model: str = "general"
    selected_model: str | None = None
    selected_replica: str | None = None
    provider: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    cost_state: str = "allocated_estimate"  # "invoiced_real", "allocated_estimate", "unknown_unbilled"
    status: str = "success"  # "success", "failed", "rejected", "cached"
    execution_mode: str = "simulated"  # "real", "simulated", "hybrid", "mock"
    is_simulated: bool = False
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DurableLedger:
    """Thread-safe transactional outbox ledger appending records to disk (JSONL)."""

    def __init__(self, ledger_path: str | Path = "data/ledger.jsonl"):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record_entry(self, entry: LedgerEntry) -> LedgerEntry:
        """Sanitizes text and durably writes a single ledger entry to JSONL."""
        if entry.error:
            entry.error = sanitize_secrets(entry.error)

        # Invariant: Failed attempts with unknown cost must never pretend to have invoiced token cost
        if entry.status == "failed" and (entry.cost_state == "unknown_unbilled" or entry.cost == 0.0):
            entry.cost = 0.0
            entry.cost_state = "unknown_unbilled"

        line = entry.model_dump_json() + "\n"
        with self._lock, open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        return entry

    def record_request(
        self,
        request_id: str,
        logical_model: str = "general",
        selected_model: str | None = None,
        selected_replica: str | None = None,
        provider: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        cost: float = 0.0,
        cost_state: str = "allocated_estimate",
        status: str = "success",
        execution_mode: str = "simulated",
        is_simulated: bool = False,
        error: str | None = None,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            entry_type="request",
            request_id=request_id,
            timestamp=timestamp if timestamp is not None else time.time(),
            logical_model=logical_model,
            selected_model=selected_model,
            selected_replica=selected_replica,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost=cost,
            cost_state=cost_state,
            status=status,
            execution_mode=execution_mode,
            is_simulated=is_simulated,
            error=sanitize_secrets(error) if error else None,
            metadata=metadata or {},
        )
        return self.record_entry(entry)

    def record_attempt(
        self,
        request_id: str,
        attempt_number: int,
        replica_id: str,
        model_id: str,
        provider: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        cost: float = 0.0,
        cost_state: str = "allocated_estimate",
        status: str = "success",
        error: str | None = None,
        execution_mode: str = "simulated",
        is_simulated: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        if status == "failed" and (cost_state == "unknown_unbilled" or cost == 0.0):
            cost = 0.0
            cost_state = "unknown_unbilled"

        entry = LedgerEntry(
            entry_type="attempt",
            request_id=request_id,
            attempt_number=attempt_number,
            timestamp=time.time(),
            logical_model="general",
            selected_model=model_id,
            selected_replica=replica_id,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost=cost,
            cost_state=cost_state,
            status=status,
            execution_mode=execution_mode,
            is_simulated=is_simulated,
            error=sanitize_secrets(error) if error else None,
            metadata=metadata or {},
        )
        return self.record_entry(entry)

    def get_entries(self, request_id: str | None = None, entry_type: str | None = None) -> list[LedgerEntry]:
        """Reads ledger records from disk, optionally filtered by request_id."""
        if not self.ledger_path.exists():
            return []
        entries = []
        with self._lock, open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = LedgerEntry.model_validate_json(line)
                    if request_id and e.request_id != request_id:
                        continue
                    if entry_type and e.entry_type != entry_type:
                        continue
                    entries.append(e)
                except Exception:  # noqa: S112
                    continue
        return entries

    def clear(self) -> None:
        """Clears the ledger file (used in test setups)."""
        with self._lock:
            if self.ledger_path.exists():
                with open(self.ledger_path, "w", encoding="utf-8") as f:
                    f.truncate(0)


default_ledger = DurableLedger()
