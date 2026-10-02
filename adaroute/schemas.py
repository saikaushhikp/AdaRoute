from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PrivacyLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL_ONLY = "internal_only"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    @classmethod
    def from_str(cls, value: str) -> PrivacyLevel:
        val = value.strip().lower()
        if val in ("internal", "internal_only"):
            return cls.INTERNAL_ONLY
        elif val == "public":
            return cls.PUBLIC
        elif val == "confidential":
            return cls.CONFIDENTIAL
        elif val == "restricted":
            return cls.RESTRICTED
        raise ValueError(
            f"Invalid privacy level '{value}'. Valid canonical options are: "
            f"{[e.value for e in cls]}"
        )


class RequestContext(BaseModel):
    task_type: str = "general"
    prompt_tokens: int = 0
    estimated_output_tokens: int = 50
    quality_requirement: str = "medium"
    latency_slo: float = 2.0
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC
    budget: float = 1.0
    tenant_id: str = "default"
    request_id: str | None = None
    user_id: str | None = "anonymous"
    no_cache: bool = False
    max_tokens: int | None = None

    @field_validator("privacy_level", mode="before")
    @classmethod
    def validate_privacy(cls, v: Any) -> PrivacyLevel:
        if isinstance(v, PrivacyLevel):
            return v
        if isinstance(v, str):
            return PrivacyLevel.from_str(v)
        raise ValueError(f"Invalid privacy level type: {type(v)}")

    @field_validator("budget")
    @classmethod
    def validate_budget(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Budget must be non-negative")
        return v


class BackendState(BaseModel):
    provider: str
    model: str
    replica_id: str
    queue_depth: int = 0
    estimated_workload: float = 0.0
    latency: float = 0.1
    memory_pressure: float = 0.0
    failure_rate: float = 0.0
    availability: bool = True


class ReplicaState(BaseModel):
    replica_id: str
    model_id: str
    provider: str
    active_requests: int = 0
    workload_tokens: int = 0  # Dynamic W_q = sum(prompt_tokens + estimated_output)
    recent_latency_ms: float = 50.0
    memory_pressure: float = 0.1
    last_heartbeat: float = Field(default_factory=time.time)
    consecutive_failures: int = 0
    availability: bool = True
    is_degraded: bool = False

    def is_alive(self, max_staleness_sec: float = 3600.0) -> bool:
        return not (not self.availability or self.consecutive_failures >= 3)


class ModelSpec(BaseModel):
    model_id: str
    logical_capabilities: list[str]
    provider: str
    quality_scores: dict[str, float] = Field(default_factory=dict)
    input_price_per_1k: float = 0.001
    output_price_per_1k: float = 0.002
    allowed_privacy: set[PrivacyLevel] = Field(default_factory=lambda: {PrivacyLevel.PUBLIC})
    replicas: list[str] = Field(default_factory=list)
    max_context: int = 4096
    version: str = "v1"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "general"  # Logical capability alias (e.g., 'general', 'coding', 'reasoning')
    messages: list[ChatMessage] = Field(default_factory=list)
    prompt: str | None = None
    max_tokens: int | None = None
    temperature: float = 0.7
    budget: float | None = 1.0
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC
    quality_requirement: str = "medium"
    latency_slo: float | None = 2.0
    tenant_id: str = "default"
    user_id: str | None = "anonymous"
    no_cache: bool = False

    @field_validator("privacy_level", mode="before")
    @classmethod
    def validate_privacy(cls, v: Any) -> PrivacyLevel:
        if isinstance(v, PrivacyLevel):
            return v
        if isinstance(v, str):
            return PrivacyLevel.from_str(v)
        raise ValueError(f"Invalid privacy level: {v}")

    def get_full_prompt(self) -> str:
        if self.prompt:
            return self.prompt
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class RoutingMetadata(BaseModel):
    request_id: str
    logical_model: str
    selected_model: str
    selected_replica: str
    provider: str
    latency_ms: float
    attempts: int = 1
    cache_hit: bool = False
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    slo_attained: bool = True
    policy_name: str = "adaptive"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    logical_model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    routing_metadata: RoutingMetadata
    content: str | None = None
    replica_id: str | None = None
    provider: str | None = None
