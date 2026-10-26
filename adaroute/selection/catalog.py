"""Declarative Catalog & System-Agnostic Architecture (Week-4 Revamp).

Separates:
- CapabilityProfile: task/domain capability attributes and quality provenance
- ModelDefinition: architecture, family, context window, output limit
- ProviderDefinition: provider backend adapter, credentials reference, protocol
- DeploymentDefinition: deployed model instance, endpoint, replicas, pricing, privacy
- ReplicaDefinition: physical serving endpoints and weights
- RuntimeSnapshot: dynamic telemetry snapshot (latency, queue depth, health)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from adaroute.schemas import PrivacyLevel


@dataclass
class CapabilityProfile:
    """Represents a model's support and quality profile for a specific capability/task."""

    name: str
    supported: bool = True
    quality_value: float = 0.80
    quality_source: str = "configured"  # "configured" | "observed" | "benchmark"
    quality_confidence: float = 0.50
    reasoning_preferred: bool = False
    reasoning_required: bool = False
    modality: list[str] = field(default_factory=lambda: ["text"])
    context_requirements: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.quality_value <= 1.0):
            raise ValueError(f"Quality value for capability '{self.name}' must be in [0, 1], got {self.quality_value}")
        if not (0.0 <= self.quality_confidence <= 1.0):
            raise ValueError(f"Quality confidence for '{self.name}' must be in [0, 1], got {self.quality_confidence}")


@dataclass
class ModelDefinition:
    """Represents logical model architecture and capability characteristics."""

    model_id: str
    family: str = "generic"
    capability_profiles: dict[str, CapabilityProfile] = field(default_factory=dict)
    context_window: int = 4096
    output_limit: int = 2048
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.context_window <= 0:
            raise ValueError(f"Model context_window must be positive, got {self.context_window}")
        if self.output_limit <= 0:
            raise ValueError(f"Model output_limit must be positive, got {self.output_limit}")

    def get_capability_quality(self, capability: str) -> float:
        cap = self.capability_profiles.get(capability.lower())
        if cap and cap.supported:
            return cap.quality_value
        gen = self.capability_profiles.get("general")
        if gen and gen.supported:
            return gen.quality_value
        return 0.70

    def get_capability_profile(self, capability: str) -> CapabilityProfile | None:
        return self.capability_profiles.get(capability.lower()) or self.capability_profiles.get("general")


@dataclass
class ProviderDefinition:
    """Represents a model serving provider and execution adapter."""

    provider_id: str
    adapter_kind: str  # "ollama" | "openai_compatible" | "vllm" | "litellm" | "mock"
    execution_type: str = "local"  # "local" | "external" | "mock"
    credential_reference: str | None = None  # Env var name e.g. "GOOGLE_API_KEY", NEVER secret value
    default_endpoint: str | None = None
    supported_protocol: str = "http"
    regions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.credential_reference:
            # Must look like an env var name, not an actual secret token
            if any(s in self.credential_reference for s in ("sk-", "AIza", "Bearer", " ")):
                raise ValueError(
                    f"credential_reference must be an environment variable name, got potential secret: {self.credential_reference}"
                )


@dataclass
class ReplicaDefinition:
    """Represents an individual physical serving replica of a deployment."""

    replica_id: str
    endpoint: str
    weight: float = 1.0
    availability: bool = True

    def __post_init__(self) -> None:
        if self.weight <= 0.0:
            raise ValueError(f"Replica weight must be positive, got {self.weight}")


@dataclass
class DeploymentDefinition:
    """Represents a deployed instance of a model via a provider."""

    deployment_id: str
    model_id: str
    provider_id: str
    endpoint: str | None = None
    input_price_per_1k: float = 0.0
    output_price_per_1k: float = 0.0
    allowed_privacy: set[PrivacyLevel] = field(default_factory=lambda: {PrivacyLevel.PUBLIC})
    capacity: int = 10
    replicas: list[ReplicaDefinition] = field(default_factory=list)
    enabled: bool = True
    provider_model_name: str | None = None  # e.g. "gemini-1.5-flash" or "qwen2.5:1.5b"

    def __post_init__(self) -> None:
        if self.input_price_per_1k < 0.0:
            raise ValueError(f"input_price_per_1k must be non-negative, got {self.input_price_per_1k}")
        if self.output_price_per_1k < 0.0:
            raise ValueError(f"output_price_per_1k must be non-negative, got {self.output_price_per_1k}")


@dataclass
class RuntimeSnapshot:
    """Dynamic operational telemetry snapshot for a deployment/replica."""

    availability: bool = True
    queue_depth: int = 0
    active_requests: int = 0
    workload_tokens: int = 0
    p50_latency_ms: float = 45.0
    p95_latency_ms: float = 150.0
    failure_rate: float = 0.0
    cooldown_until: float = 0.0
    memory_pressure: float = 0.0
    last_heartbeat: float = 0.0
    sample_count: int = 0

    def is_in_cooldown(self, current_time: float) -> bool:
        return self.cooldown_until > current_time


class SystemCatalog:
    """Authoritative deployment and model catalog.

    Validates:
    - Provider references exist.
    - Model references exist.
    - Deployment IDs are unique.
    - Replica IDs are unique within a deployment.
    - Prices are non-negative.
    - Quality scores are in [0, 1].
    - Context windows and output limits are positive.
    - No undeclared models appear in strict production mode.
    """

    def __init__(self, strict_catalog: bool = True):
        self.strict_catalog = strict_catalog
        self.providers: dict[str, ProviderDefinition] = {}
        self.models: dict[str, ModelDefinition] = {}
        self.deployments: dict[str, DeploymentDefinition] = {}
        self.runtime_snapshots: dict[str, RuntimeSnapshot] = {}

    def register_provider(self, provider: ProviderDefinition) -> None:
        if provider.provider_id in self.providers:
            raise ValueError(f"Duplicate provider_id: '{provider.provider_id}'")
        self.providers[provider.provider_id] = provider

    def register_model(self, model: ModelDefinition) -> None:
        if model.model_id in self.models:
            raise ValueError(f"Duplicate model_id: '{model.model_id}'")
        self.models[model.model_id] = model

    def register_deployment(self, deployment: DeploymentDefinition) -> None:
        if deployment.deployment_id in self.deployments:
            raise ValueError(f"Duplicate deployment_id: '{deployment.deployment_id}'")
        if deployment.provider_id not in self.providers:
            raise ValueError(
                f"Deployment '{deployment.deployment_id}' references unknown provider '{deployment.provider_id}'"
            )
        if deployment.model_id not in self.models:
            raise ValueError(
                f"Deployment '{deployment.deployment_id}' references unknown model '{deployment.model_id}'"
            )

        # Validate unique replica IDs within this deployment
        rep_ids = set()
        for r in deployment.replicas:
            if r.replica_id in rep_ids:
                raise ValueError(
                    f"Duplicate replica_id '{r.replica_id}' in deployment '{deployment.deployment_id}'"
                )
            rep_ids.add(r.replica_id)

        self.deployments[deployment.deployment_id] = deployment
        if deployment.deployment_id not in self.runtime_snapshots:
            self.runtime_snapshots[deployment.deployment_id] = RuntimeSnapshot()

    def get_deployment(self, deployment_id: str) -> DeploymentDefinition | None:
        return self.deployments.get(deployment_id)

    def get_model(self, model_id: str) -> ModelDefinition | None:
        return self.models.get(model_id)

    def get_provider(self, provider_id: str) -> ProviderDefinition | None:
        return self.providers.get(provider_id)

    def get_snapshot(self, deployment_id: str) -> RuntimeSnapshot:
        if deployment_id not in self.runtime_snapshots:
            self.runtime_snapshots[deployment_id] = RuntimeSnapshot()
        return self.runtime_snapshots[deployment_id]

    def update_snapshot(self, deployment_id: str, **kwargs: Any) -> None:
        snap = self.get_snapshot(deployment_id)
        for k, v in kwargs.items():
            if hasattr(snap, k):
                setattr(snap, k, v)

    def get_deployments_for_capability(self, capability: str) -> list[DeploymentDefinition]:
        """Returns all enabled deployments whose underlying model advertises this capability."""
        cap_clean = capability.lower().strip()
        matched: list[DeploymentDefinition] = []
        for dep in self.deployments.values():
            if not dep.enabled:
                continue
            model = self.models.get(dep.model_id)
            if not model:
                continue
            # Match either specific capability or model_id alias match
            if dep.model_id.lower() == cap_clean or dep.deployment_id.lower() == cap_clean:
                matched.append(dep)
                continue
            cap_prof = model.capability_profiles.get(cap_clean)
            if cap_prof and cap_prof.supported:
                matched.append(dep)
            elif cap_clean == "general" and "general" in model.capability_profiles:
                if model.capability_profiles["general"].supported:
                    matched.append(dep)
        return matched

    @classmethod
    def from_dict(cls, data: dict[str, Any], strict_catalog: bool = True) -> SystemCatalog:
        """Parses declarative configuration into a validated SystemCatalog."""
        catalog = cls(strict_catalog=strict_catalog)

        # 1. Parse Providers
        providers_data = data.get("providers", {})
        if isinstance(providers_data, dict):
            for pid, pdata in providers_data.items():
                pdef = ProviderDefinition(
                    provider_id=pid,
                    adapter_kind=pdata.get("adapter", "ollama"),
                    execution_type=pdata.get("execution_type", "local"),
                    credential_reference=pdata.get("credential_ref") or pdata.get("api_key_env"),
                    default_endpoint=pdata.get("default_endpoint") or pdata.get("endpoint"),
                    supported_protocol=pdata.get("protocol", "http"),
                    regions=pdata.get("regions", []),
                )
                catalog.register_provider(pdef)

        # 2. Parse Models
        models_data = data.get("models", [])
        if isinstance(models_data, list):
            for m in models_data:
                mid = m["model_id"]
                # Parse capabilities
                cap_profiles: dict[str, CapabilityProfile] = {}
                raw_caps = m.get("capabilities", {})
                if isinstance(raw_caps, dict):
                    for cname, cdata in raw_caps.items():
                        if isinstance(cdata, dict):
                            q_info = cdata.get("quality", {})
                            q_val = q_info.get("value", 0.80) if isinstance(q_info, dict) else float(q_info)
                            q_src = q_info.get("source", "configured") if isinstance(q_info, dict) else "configured"
                            q_conf = (
                                q_info.get("confidence", 0.50) if isinstance(q_info, dict) else 0.50
                            )
                            cap_profiles[cname.lower()] = CapabilityProfile(
                                name=cname.lower(),
                                supported=cdata.get("supported", True),
                                quality_value=q_val,
                                quality_source=q_src,
                                quality_confidence=q_conf,
                                reasoning_preferred=cdata.get("reasoning_preferred", False),
                                reasoning_required=cdata.get("reasoning_required", False),
                                modality=cdata.get("modality", ["text"]),
                                context_requirements=cdata.get("context_requirements", 0),
                            )
                elif "logical_aliases" in m:
                    # Legacy/compatibility format: expand aliases to capability profiles
                    base_q = float(m.get("quality", 0.80))
                    for alias in m.get("logical_aliases", ["general"]):
                        cap_profiles[alias.lower()] = CapabilityProfile(
                            name=alias.lower(),
                            supported=True,
                            quality_value=base_q,
                            quality_source="configured",
                            quality_confidence=0.50,
                        )

                mdef = ModelDefinition(
                    model_id=mid,
                    family=m.get("family", mid.split(":")[0] if ":" in mid else "generic"),
                    capability_profiles=cap_profiles,
                    context_window=int(m.get("max_context") or m.get("context_window") or 4096),
                    output_limit=int(m.get("output_limit", 2048)),
                    version=str(m.get("version", "1.0.0")),
                )
                catalog.register_model(mdef)

        # 3. Parse Deployments
        deployments_data = data.get("deployments", [])
        if isinstance(deployments_data, list) and deployments_data:
            for d in deployments_data:
                reps: list[ReplicaDefinition] = []
                for r in d.get("replicas", []):
                    reps.append(
                        ReplicaDefinition(
                            replica_id=r["replica_id"],
                            endpoint=r["endpoint"],
                            weight=float(r.get("weight", 1.0)),
                            availability=r.get("availability", True),
                        )
                    )
                pricing = d.get("pricing", {})
                in_price = float(pricing.get("input_per_1k") or d.get("cost_per_1k_input") or 0.0)
                out_price = float(pricing.get("output_per_1k") or d.get("cost_per_1k_output") or 0.0)

                priv_data = d.get("privacy", {})
                if isinstance(priv_data, dict):
                    allowed_priv = {PrivacyLevel.from_str(p) for p in priv_data.get("allowed", ["public"])}
                elif isinstance(priv_data, list):
                    allowed_priv = {PrivacyLevel.from_str(p) for p in priv_data}
                else:
                    allowed_priv = {PrivacyLevel.PUBLIC}

                ddef = DeploymentDefinition(
                    deployment_id=d["deployment_id"],
                    model_id=d["model_id"],
                    provider_id=d["provider_id"],
                    endpoint=d.get("endpoint"),
                    input_price_per_1k=in_price,
                    output_price_per_1k=out_price,
                    allowed_privacy=allowed_priv,
                    capacity=int(d.get("capacity", 10)),
                    replicas=reps,
                    enabled=d.get("enabled", True),
                    provider_model_name=d.get("provider_model_name"),
                )
                catalog.register_deployment(ddef)
        else:
            # Compatibility auto-generation of deployments from legacy models list
            for m in models_data:
                mid = m["model_id"]
                pid = m.get("provider", "local")
                if pid not in catalog.providers:
                    # Synthesize provider definition if missing
                    exec_type = m.get("execution_type", "local")
                    adapter = "litellm" if exec_type == "external" else ("mock" if exec_type == "mock" else "ollama")
                    catalog.register_provider(
                        ProviderDefinition(
                            provider_id=pid,
                            adapter_kind=adapter,
                            execution_type=exec_type,
                            credential_reference=m.get("api_key_env"),
                            default_endpoint=m.get("base_url"),
                        )
                    )

                reps = [
                    ReplicaDefinition(
                        replica_id=r["replica_id"],
                        endpoint=r.get("endpoint", "http://localhost:11434"),
                        weight=float(r.get("weight", 1.0)),
                        availability=r.get("availability", True),
                    )
                    for r in m.get("replicas", [])
                ]
                priv_set = {PrivacyLevel.from_str(pl) for pl in m.get("privacy_levels", ["public"])}
                ddef = DeploymentDefinition(
                    deployment_id=f"deploy-{mid}",
                    model_id=mid,
                    provider_id=pid,
                    endpoint=m.get("base_url"),
                    input_price_per_1k=float(m.get("cost_per_1k_input", 0.0)),
                    output_price_per_1k=float(m.get("cost_per_1k_output", 0.0)),
                    allowed_privacy=priv_set,
                    replicas=reps,
                    enabled=True,
                    provider_model_name=m.get("litellm_model") or mid,
                )
                catalog.register_deployment(ddef)

        return catalog

    @classmethod
    def from_yaml_file(cls, path: Path | str, strict_catalog: bool = True) -> SystemCatalog:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Catalog file not found: {path}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data, strict_catalog=strict_catalog)
