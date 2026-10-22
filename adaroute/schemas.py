from __future__ import annotations

import re
import time
import unicodedata
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def normalize_prompt_text(text: str) -> str:
    """Normalizes prompt text canonically (NFKC, line endings, collapsed spaces).

    Preserves code indentation (leading spaces/tabs) for code blocks while
    canonicalizing whitespace variations for prose.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Detect code blocks or syntax where leading indentation is semantically significant
    has_code = "```" in text or any(
        line.lstrip().startswith(("def ", "class ", "import ", "from ", "return ", "if __name__"))
        for line in text.split("\n")
    )

    if has_code:
        # Preserve indentation for code blocks while stripping trailing spaces
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    # Standard prose: collapse multiple whitespace into single space, normalize case,
    # and strip trailing sentence punctuation (? . ! , : ;) to match real-world query variations.
    lines = []
    for line in text.split("\n"):
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        cleaned = re.sub(r"[\?\.\!,;:]+$", "", cleaned).strip()
        if cleaned:
            lines.append(cleaned.lower())
    return "\n".join(lines).strip()


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
        raise ValueError(f"Invalid privacy level '{value}'. Valid canonical options are: {[e.value for e in cls]}")


class RequestContext(BaseModel):
    task_type: str = "general"
    prompt_tokens: int = 0
    estimated_output_tokens: int = 50
    quality_requirement: str = "medium"
    latency_slo: float = 2.0
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC
    budget: float = 1.0
    tenant_id: str = "user"
    request_id: str | None = None
    user_id: str | None = "user"
    no_cache: bool = False
    max_tokens: int | None = None
    temperature: float = 0.7
    is_authenticated: bool = True
    user_private: bool = False
    execution_mode: str = "hybrid"
    policy_name: str = "adaptive"

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

    @field_validator("latency_slo")
    @classmethod
    def validate_slo(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Latency SLO must be strictly positive")
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
    endpoint: str | None = None
    weight: float = 1.0
    active_requests: int = 0
    workload_tokens: int = 0  # Dynamic W_q = sum(prompt_tokens + estimated_output)
    recent_latency_ms: float = 50.0
    memory_pressure: float = 0.1
    last_heartbeat: float = Field(default_factory=time.time)
    consecutive_failures: int = 0
    availability: bool = True
    is_degraded: bool = False

    def is_alive(self, max_staleness_sec: float = 300.0) -> bool:
        if not self.availability or self.consecutive_failures >= 3:
            return False
        if max_staleness_sec > 0 and (time.time() - self.last_heartbeat) > max_staleness_sec:
            return False
        return True


class ModelSpec(BaseModel):
    model_id: str
    logical_capabilities: list[str]
    provider: str
    execution_type: str = "local"  # "local" | "external" | "mock"
    litellm_model: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    quality_scores: dict[str, float] = Field(default_factory=dict)
    input_price_per_1k: float = 0.001
    output_price_per_1k: float = 0.002
    allowed_privacy: set[PrivacyLevel] = Field(default_factory=lambda: {PrivacyLevel.PUBLIC})
    replicas: list[str] = Field(default_factory=list)
    max_context: int = 4096
    version: str = "v1"
    is_external: bool = False

    @property
    def is_cloud(self) -> bool:
        """Indicates whether this model is cloud-hosted (skips replica scheduling)."""
        if self.execution_type in ("external", "cloud", "mock-external"):
            return True
        if self.is_external:
            return True
        if self.execution_type in ("local", "mock-local"):
            return False
        return self.provider in ("groq", "cohere", "gemini", "google", "litellm", "mock-external")

    def calculate_max_cost(self, prompt_tokens: int, max_tokens: int | None = None) -> float:
        """Calculates hard worst-case cost based on prompt tokens and max_tokens ceiling."""
        out_tokens = max_tokens if max_tokens is not None and max_tokens > 0 else 256
        return (prompt_tokens / 1000.0) * self.input_price_per_1k + (out_tokens / 1000.0) * self.output_price_per_1k


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        role = v.strip().lower()
        if role not in ("system", "user", "assistant", "function", "tool"):
            raise ValueError(f"Invalid role: '{v}'")
        return role

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if v is None:
            raise ValueError("Message content cannot be None")
        return v


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Forbid unvalidated unknown fields

    model: str = "general"
    task_type: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    prompt: str | None = None
    max_tokens: int | None = Field(default=256, gt=0)
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    budget: float = Field(default=1.0, ge=0.0)
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC
    quality_requirement: str = "medium"
    latency_slo: float = Field(default=2.0, gt=0.0)
    tenant_id: str | None = "user"
    user_id: str | None = "user"
    no_cache: bool = False
    policy: str = "adaptive"
    scheduler: str = "slo_aware"
    user_private: bool = False
    execution_mode: str | None = Field(default=None)

    @field_validator("stream")
    @classmethod
    def validate_stream(cls, v: bool) -> bool:
        if v:
            raise ValueError("Streaming is not supported in current AdaRoute release.")
        return v

    @field_validator("privacy_level", mode="before")
    @classmethod
    def validate_privacy(cls, v: Any) -> PrivacyLevel:
        if isinstance(v, PrivacyLevel):
            return v
        if isinstance(v, str):
            return PrivacyLevel.from_str(v)
        raise ValueError(f"Invalid privacy level: {v}")

    @field_validator("quality_requirement")
    @classmethod
    def validate_quality(cls, v: str) -> str:
        val = v.strip().lower()
        if val not in ("low", "medium", "high"):
            raise ValueError(f"Invalid quality requirement: '{v}'. Must be low, medium, or high.")
        return val

    @field_validator("policy")
    @classmethod
    def validate_policy(cls, v: str) -> str:
        val = v.strip().lower()
        valid_policies = (
            "adaptive",
            "strongest",
            "cheapest",
            "static",
            "latency_first",
            "cost_first",
            "quality_first",
        )
        if val not in valid_policies:
            raise ValueError(f"Invalid routing policy: '{v}'. Must be one of {valid_policies}")
        return val

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, v: str) -> str:
        val = v.strip().lower()
        if val not in ("slo_aware", "least_loaded", "round_robin"):
            raise ValueError(f"Invalid replica scheduler: '{v}'.")
        return val

    @model_validator(mode="after")
    def validate_prompt_or_messages(self) -> ChatCompletionRequest:
        has_prompt = bool(self.prompt and self.prompt.strip())
        has_non_empty_msg = any(bool(m.content and m.content.strip()) for m in self.messages)
        if not has_prompt and not has_non_empty_msg:
            raise ValueError("Request must contain non-empty prompt or at least one non-empty message.")
        return self

    def to_messages_list(self) -> list[dict[str, str]]:
        """Returns structured message dictionaries preserving native conversation roles."""
        valid_msgs = [m for m in self.messages if m.content and m.content.strip()]
        if valid_msgs:
            return [{"role": m.role.strip().lower(), "content": m.content} for m in valid_msgs]
        if self.prompt and self.prompt.strip():
            return [{"role": "user", "content": self.prompt}]
        return []

    def get_full_prompt(self) -> str:
        """Returns canonical normalized representation of prompt or messages."""
        if self.prompt and self.prompt.strip():
            return normalize_prompt_text(self.prompt)

        valid_msgs = [m for m in self.messages if m.content and m.content.strip()]
        if not valid_msgs:
            return ""

        # Single user message: treat identically to standalone prompt text
        if len(valid_msgs) == 1 and valid_msgs[0].role.strip().lower() == "user":
            return normalize_prompt_text(valid_msgs[0].content)

        # Multi-turn messages: canonical normalized format
        return "\n".join(f"{m.role.strip().lower()}: {normalize_prompt_text(m.content)}" for m in valid_msgs)

    def get_latest_user_query(self) -> str:
        """Extracts and normalizes the latest user query turn."""
        if self.prompt and self.prompt.strip():
            return normalize_prompt_text(self.prompt)
        for m in reversed(self.messages):
            if m.role.strip().lower() == "user" and m.content and m.content.strip():
                return normalize_prompt_text(m.content)
        return self.get_full_prompt()


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AttemptRecord(BaseModel):
    attempt_number: int
    replica_id: str
    model_id: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    cost_state: str = "allocated_estimate"  # "invoiced_real", "allocated_estimate", "unknown_unbilled"
    status: str = "success"  # "success" | "failed"
    error_message: str | None = None
    execution_mode: str = "simulated"
    is_simulated: bool = False


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
    scheduler_name: str = "slo_aware"
    attempt_history: list[AttemptRecord] = Field(default_factory=list)


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
