"""Model Registry for AdaRoute (Hardened Week-3 Version).

Provides versioned capability, pricing, quality, and privacy allow-lists
for all supported models and their replicas. Supports loading from config.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from adaroute.schemas import ModelSpec, PrivacyLevel, ReplicaState
from adaroute.selection import SystemCatalog, UnifiedModelSelector


class ModelRegistry:
    def __init__(self, config_path: str | Path | None = None):
        self._models: dict[str, ModelSpec] = {}
        self._replicas: dict[str, ReplicaState] = {}
        self.raw_config: dict[str, Any] = {}
        self.routing_config: dict[str, Any] = {}
        self.estimation_config: dict[str, Any] = {}
        self.registry_config: dict[str, Any] = {}

        if config_path:
            self._config_path = Path(config_path)
        else:
            # Compose GPU profiles select an isolated catalog so their logical
            # aliases always match the model launched by vLLM.  The default
            # config remains unchanged for CPU/local development.
            configured_path = os.getenv("ADAROUTE_CONFIG")
            cand = Path(configured_path) if configured_path else Path("config.yaml")
            if cand.exists():
                self._config_path = cand
            else:
                self._config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"

        if self._config_path.exists():
            self.load_from_yaml(self._config_path)
        else:
            self._init_defaults()

    def load_from_yaml(self, path: Path | str) -> None:
        """Loads models and replicas dynamically from config.yaml."""
        p = Path(path)
        if not p.exists():
            self._init_defaults()
            return

        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        self.raw_config = cfg
        self.routing_config = cfg.get("routing", {})
        self.estimation_config = cfg.get("estimation", {})
        self.registry_config = cfg.get("registry", {})

        models_data = cfg.get("models", [])
        strict_cat = cfg.get("gateway", {}).get("strict_catalog", False)
        for m in models_data:
            privacy_set = {PrivacyLevel.from_str(pl) for pl in m.get("privacy_levels", ["public"])}
            exec_type = m.get("execution_type", "local")
            is_ext = exec_type == "external" or m.get("provider") in (
                "groq",
                "cohere",
                "gemini",
                "google",
                "openai",
                "mock-external",
                "litellm",
            )
            # Support explicit per-capability quality profiles or legacy scalar quality
            quality_scores: dict[str, float] = {}
            raw_caps = m.get("capabilities", {})
            if isinstance(raw_caps, dict) and raw_caps:
                for cname, cdata in raw_caps.items():
                    if isinstance(cdata, dict):
                        q_info = cdata.get("quality", {})
                        q_val = q_info.get("value", 0.80) if isinstance(q_info, dict) else float(q_info)
                        quality_scores[cname.lower()] = q_val
                    else:
                        quality_scores[cname.lower()] = float(cdata)
            base_q = float(m.get("quality", 0.80))
            for alias in m.get("logical_aliases", ["general"]):
                if alias.lower() not in quality_scores:
                    quality_scores[alias.lower()] = base_q
            if "general" not in quality_scores:
                quality_scores["general"] = base_q

            spec = ModelSpec(
                model_id=m["model_id"],
                logical_capabilities=m.get("logical_aliases", ["general"]),
                provider=m.get("provider", "local"),
                execution_type=exec_type,
                litellm_model=m.get("litellm_model"),
                api_key_env=m.get("api_key_env"),
                base_url=m.get("base_url"),
                quality_scores=quality_scores,
                input_price_per_1k=m.get("cost_per_1k_input", 0.0),
                output_price_per_1k=m.get("cost_per_1k_output", 0.0),
                allowed_privacy=privacy_set,
                replicas=[r["replica_id"] for r in m.get("replicas", [])],
                max_context=m.get("max_context", 4096),
                version="3.0.0",
                is_external=is_ext,
            )
            self.register_model(spec)

            for r in m.get("replicas", []):
                rep_state = ReplicaState(
                    replica_id=r["replica_id"],
                    model_id=m["model_id"],
                    provider=m.get("provider", "local"),
                    endpoint=r.get("endpoint"),
                    weight=float(r.get("weight", 1.0)),
                    availability=r.get("availability", True),
                    recent_latency_ms=45.0,
                )
                self.register_replica(rep_state)

        # Conditionally ensure legacy aliases exist for backward compatibility with W1/W2 tests
        if self.registry_config.get("compatibility_models", True) and not strict_cat:
            self._ensure_backward_compatibility()

        try:
            self.catalog = SystemCatalog.from_dict(cfg, strict_catalog=strict_cat)
            # Synchronize any compatibility models in _models to catalog
            if self.registry_config.get("compatibility_models", True) and not strict_cat:
                for spec in self._models.values():
                    if spec.model_id not in self.catalog.models:
                        self._sync_spec_to_catalog(spec)
            self.selector = UnifiedModelSelector(catalog=self.catalog, config=cfg)
        except Exception as e:
            import logging

            logging.getLogger("adaroute.registry").exception("Catalog/Selector initialization error")
            self.catalog = None
            self.selector = None
            self._selector_init_error = str(e)

    def get_policy_weights(self, policy_name: str) -> dict[str, float] | None:
        """Retrieves objective weights for a named policy from declarative config."""
        policies = self.routing_config.get("policies", {})
        policy_cfg = policies.get(policy_name)
        if policy_cfg and "weights" in policy_cfg:
            return dict(policy_cfg["weights"])
        return None

    def get_static_task_map(self) -> dict[str, str]:
        """Retrieves static task routing map from declarative config."""
        return dict(self.routing_config.get("static_task_map", {}))

    def get_estimation_config(self) -> dict[str, Any]:
        """Retrieves estimation configuration section."""
        return dict(self.estimation_config)

    def _ensure_backward_compatibility(self) -> None:
        if "gpt-4o-external" not in self._models:
            self.register_model(
                ModelSpec(
                    model_id="gpt-4o-external",
                    logical_capabilities=["general", "qa", "summarization", "coding", "reasoning"],
                    provider="mock-external",
                    quality_scores={
                        "qa": 0.95,
                        "factual_qa": 0.96,
                        "summarization": 0.92,
                        "coding": 0.94,
                        "reasoning": 0.95,
                        "general": 0.94,
                        "creative": 0.92,
                    },
                    input_price_per_1k=0.005,
                    output_price_per_1k=0.015,
                    allowed_privacy={PrivacyLevel.PUBLIC},
                    replicas=["external-r1"],
                    max_context=16384,
                    version="2024-08",
                    is_external=True,
                )
            )
            self.register_replica(
                ReplicaState(
                    replica_id="external-r1",
                    model_id="gpt-4o-external",
                    provider="mock-external",
                    recent_latency_ms=250.0,
                )
            )

        if "llama-3-8b" not in self._models:
            self.register_model(
                ModelSpec(
                    model_id="llama-3-8b",
                    logical_capabilities=["general", "qa", "summarization", "coding", "creative"],
                    provider="mock-local",
                    quality_scores={
                        "qa": 0.81,
                        "summarization": 0.79,
                        "coding": 0.77,
                        "reasoning": 0.74,
                        "general": 0.79,
                        "creative": 0.77,
                    },
                    input_price_per_1k=0.0005,
                    output_price_per_1k=0.0015,
                    allowed_privacy={
                        PrivacyLevel.PUBLIC,
                        PrivacyLevel.INTERNAL_ONLY,
                        PrivacyLevel.CONFIDENTIAL,
                        PrivacyLevel.RESTRICTED,
                    },
                    replicas=["replica-1"],
                    max_context=8192,
                    version="1.0.0",
                )
            )
            self.register_replica(
                ReplicaState(
                    replica_id="replica-1",
                    model_id="llama-3-8b",
                    provider="mock-local",
                    recent_latency_ms=40.0,
                )
            )

    def _init_defaults(self) -> None:
        # Default local model: qwen2.5:1.5b with 2 replicas
        local_model = ModelSpec(
            model_id="qwen2.5:1.5b",
            logical_capabilities=["general", "qa", "summarization", "coding", "fast"],
            provider="local",
            quality_scores={
                "qa": 0.82,
                "factual_qa": 0.82,
                "summarization": 0.80,
                "coding": 0.78,
                "reasoning": 0.75,
                "general": 0.80,
                "creative": 0.78,
            },
            input_price_per_1k=0.0,
            output_price_per_1k=0.0,
            allowed_privacy={
                PrivacyLevel.PUBLIC,
                PrivacyLevel.INTERNAL_ONLY,
                PrivacyLevel.CONFIDENTIAL,
                PrivacyLevel.RESTRICTED,
            },
            replicas=["local-r1", "local-r2"],
            max_context=32768,
            version="1.0.0",
        )
        self.register_model(local_model)
        self.register_replica(
            ReplicaState(replica_id="local-r1", model_id="qwen2.5:1.5b", provider="local", recent_latency_ms=45.0)
        )
        self.register_replica(
            ReplicaState(replica_id="local-r2", model_id="qwen2.5:1.5b", provider="local", recent_latency_ms=50.0)
        )

        # Local reasoning model: deepseek-r1:1.5b
        deepseek_model = ModelSpec(
            model_id="deepseek-r1:1.5b",
            logical_capabilities=["reasoning", "complex"],
            provider="local",
            quality_scores={
                "reasoning": 0.88,
                "complex": 0.88,
                "general": 0.84,
                "coding": 0.82,
            },
            input_price_per_1k=0.0,
            output_price_per_1k=0.0,
            allowed_privacy={
                PrivacyLevel.PUBLIC,
                PrivacyLevel.INTERNAL_ONLY,
                PrivacyLevel.CONFIDENTIAL,
                PrivacyLevel.RESTRICTED,
            },
            replicas=["local-deepseek-r1"],
            max_context=131072,
            version="1.0.0",
        )
        self.register_model(deepseek_model)
        self.register_replica(
            ReplicaState(
                replica_id="local-deepseek-r1", model_id="deepseek-r1:1.5b", provider="local", recent_latency_ms=85.0
            )
        )

        self._ensure_backward_compatibility()

    def _sync_spec_to_catalog(self, spec: ModelSpec) -> None:
        if not hasattr(self, "catalog") or not self.catalog:
            return
        from adaroute.selection.catalog import (
            CapabilityProfile,
            DeploymentDefinition,
            ModelDefinition,
            ProviderDefinition,
            ReplicaDefinition,
        )

        pid = spec.provider or "local"
        if pid not in self.catalog.providers:
            exec_type = spec.execution_type or (
                "external" if spec.is_external else ("mock" if "mock" in pid else "local")
            )
            adapter = (
                "litellm"
                if spec.is_external
                else ("mock" if "mock" in pid else ("vllm" if pid in ("vllm", "local_vllm", "local") else "ollama"))
            )
            self.catalog.register_provider(
                ProviderDefinition(
                    provider_id=pid,
                    adapter_kind=adapter,
                    execution_type=exec_type,
                    credential_reference=spec.api_key_env,
                    default_endpoint=spec.base_url,
                )
            )

        if spec.model_id not in self.catalog.models:
            cap_profiles = {
                c.lower(): CapabilityProfile(
                    name=c.lower(),
                    supported=True,
                    quality_value=spec.quality_scores.get(c.lower(), 0.80),
                )
                for c in spec.logical_capabilities
            }
            if "general" not in cap_profiles:
                cap_profiles["general"] = CapabilityProfile(name="general", supported=True, quality_value=0.80)
            self.catalog.register_model(
                ModelDefinition(
                    model_id=spec.model_id,
                    capability_profiles=cap_profiles,
                    context_window=spec.max_context or 4096,
                )
            )

        dep_id = f"deploy-{spec.model_id}"
        if dep_id not in self.catalog.deployments:
            reps = [
                ReplicaDefinition(
                    replica_id=rid,
                    endpoint=spec.base_url or "http://localhost:7000",
                    availability=True,
                )
                for rid in spec.replicas
            ]
            self.catalog.register_deployment(
                DeploymentDefinition(
                    deployment_id=dep_id,
                    model_id=spec.model_id,
                    provider_id=pid,
                    endpoint=spec.base_url,
                    input_price_per_1k=spec.input_price_per_1k,
                    output_price_per_1k=spec.output_price_per_1k,
                    allowed_privacy=spec.allowed_privacy,
                    replicas=reps,
                    enabled=True,
                    provider_model_name=spec.litellm_model or spec.model_id,
                )
            )

    def register_model(self, spec: ModelSpec) -> None:
        self._models[spec.model_id] = spec
        self._sync_spec_to_catalog(spec)

    def register_replica(self, replica: ReplicaState) -> None:
        self._replicas[replica.replica_id] = replica
        if hasattr(self, "catalog") and self.catalog:
            from adaroute.selection.catalog import ReplicaDefinition

            # Find matching deployment
            dep = self.catalog.get_deployment(f"deploy-{replica.model_id}")
            if not dep:
                for d in self.catalog.deployments.values():
                    if d.model_id == replica.model_id:
                        dep = d
                        break
            if dep and not any(r.replica_id == replica.replica_id for r in dep.replicas):
                dep.replicas.append(
                    ReplicaDefinition(
                        replica_id=replica.replica_id,
                        endpoint=replica.endpoint or "",
                        weight=replica.weight,
                        availability=replica.availability,
                    )
                )

    def get_model(self, model_id: str) -> ModelSpec | None:
        return self._models.get(model_id)

    def get_replica(self, replica_id: str) -> ReplicaState | None:
        return self._replicas.get(replica_id)

    def get_all_models(self) -> list[ModelSpec]:
        return list(self._models.values())

    def get_all_replicas(self) -> list[ReplicaState]:
        return list(self._replicas.values())

    def get_replicas_for_model(self, model_id: str) -> list[ReplicaState]:
        return [r for r in self._replicas.values() if r.model_id == model_id]

    def is_valid_capability_or_model(self, alias_or_id: str) -> bool:
        """Checks if the given alias is a known logical capability or concrete model ID."""
        target = alias_or_id.lower().strip()
        if any(m_id.lower() == target for m_id in self._models):
            return True
        for m in self._models.values():
            if any(c.lower() == target for c in m.logical_capabilities):
                return True
        if hasattr(self, "catalog") and self.catalog:
            if self.catalog.get_deployment(alias_or_id) or self.catalog.get_model(alias_or_id):
                return True
            for mdef in self.catalog.models.values():
                if target in [c.lower() for c in mdef.capability_profiles]:
                    return True
        return False

    def get_models_for_capability(self, capability: str) -> list[ModelSpec]:
        """Resolves models matching a specific capability alias or concrete model ID."""
        cap = capability.lower().strip()
        # Direct model_id match
        for m_id, spec in self._models.items():
            if m_id.lower() == cap:
                return [spec]
        # Logical capability matches (strict: no silent fallback to general)
        matches = [m for m in self._models.values() if cap in [c.lower() for c in m.logical_capabilities]]
        return matches


default_registry = ModelRegistry()
