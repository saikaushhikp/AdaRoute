"""Model Registry for AdaRoute (Hardened Week-3 Version).

Provides versioned capability, pricing, quality, and privacy allow-lists
for all supported models and their replicas. Supports loading from config.yaml.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from adaroute.schemas import ModelSpec, PrivacyLevel, ReplicaState


class ModelRegistry:
    def __init__(self, config_path: str | Path | None = None):
        self._models: dict[str, ModelSpec] = {}
        self._replicas: dict[str, ReplicaState] = {}
        self._config_path = Path(config_path) if config_path else Path("config.yaml")
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
            cfg = yaml.safe_load(f)

        models_data = cfg.get("models", [])
        for m in models_data:
            privacy_set = {PrivacyLevel.from_str(pl) for pl in m.get("privacy_levels", ["public"])}
            exec_type = m.get("execution_type", "local")
            is_ext = exec_type == "external" or m.get("provider") in (
                "groq",
                "cohere",
                "gemini",
                "google",
                "mock-external",
                "litellm",
            )
            spec = ModelSpec(
                model_id=m["model_id"],
                logical_capabilities=m.get("logical_aliases", ["general"]),
                provider=m.get("provider", "local"),
                quality_scores={
                    "general": m.get("quality", 0.80),
                    "qa": m.get("quality", 0.80),
                    "coding": m.get("quality", 0.80),
                    "reasoning": m.get("quality", 0.80),
                    "summarization": m.get("quality", 0.80),
                },
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
                    availability=r.get("availability", True),
                    recent_latency_ms=45.0,
                )
                self.register_replica(rep_state)

        # Ensure legacy aliases exist for backward compatibility with W1/W2 tests
        self._ensure_backward_compatibility()

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

    def register_model(self, spec: ModelSpec) -> None:
        self._models[spec.model_id] = spec

    def register_replica(self, replica: ReplicaState) -> None:
        self._replicas[replica.replica_id] = replica

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

    def get_models_for_capability(self, capability: str) -> list[ModelSpec]:
        cap = capability.lower().strip()
        matches = [m for m in self._models.values() if cap in [c.lower() for c in m.logical_capabilities]]
        if not matches:
            matches = [m for m in self._models.values() if "general" in [c.lower() for c in m.logical_capabilities]]
        return matches


default_registry = ModelRegistry()
