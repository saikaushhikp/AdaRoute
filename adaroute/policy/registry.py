"""Model Registry for AdaRoute.

Provides versioned capability, pricing, quality, and privacy allow-lists
for all supported models and their replicas.
"""
from __future__ import annotations

from adaroute.schemas import ModelSpec, PrivacyLevel, ReplicaState


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, ModelSpec] = {}
        self._replicas: dict[str, ReplicaState] = {}
        self._init_defaults()

    def _init_defaults(self):
        # 1. Local Open Model (Llama-3-8B / Qwen2.5:1.5B class)
        # Allows all privacy levels (Internal, Confidential, Restricted, Public)
        local_model = ModelSpec(
            model_id="qwen-2.5-local",
            logical_capabilities=["general", "qa", "summarization", "coding", "reasoning"],
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
            input_price_per_1k=0.0005,
            output_price_per_1k=0.0015,
            allowed_privacy={
                PrivacyLevel.PUBLIC,
                PrivacyLevel.INTERNAL_ONLY,
                PrivacyLevel.CONFIDENTIAL,
                PrivacyLevel.RESTRICTED,
            },
            replicas=["local-r1", "local-r2"],
            max_context=8192,
            version="1.0.0",
        )
        self.register_model(local_model)

        # 2 Replicas of the Local Model to evaluate within-model load balancing
        self.register_replica(
            ReplicaState(
                replica_id="local-r1",
                model_id="qwen-2.5-local",
                provider="local",
                recent_latency_ms=45.0,
            )
        )
        self.register_replica(
            ReplicaState(
                replica_id="local-r2",
                model_id="qwen-2.5-local",
                provider="local",
                recent_latency_ms=50.0,
            )
        )

        # Legacy / Alias model llama-3-8b for backward compatibility with W1 integration tests
        llama_model = ModelSpec(
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
        self.register_model(llama_model)
        self.register_replica(
            ReplicaState(
                replica_id="replica-1",
                model_id="llama-3-8b",
                provider="mock-local",
                recent_latency_ms=40.0,
            )
        )

        # 2. External Cloud / Mock Provider (GPT-4o / High-Capability class)
        # STRICT PRIVACY RULE: Only allows PUBLIC requests!
        external_model = ModelSpec(
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
        )
        self.register_model(external_model)

        self.register_replica(
            ReplicaState(
                replica_id="external-r1",
                model_id="gpt-4o-external",
                provider="mock-external",
                recent_latency_ms=250.0,
            )
        )

    def register_model(self, spec: ModelSpec):
        self._models[spec.model_id] = spec

    def register_replica(self, replica: ReplicaState):
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
            # Fallback to general capability models
            matches = [m for m in self._models.values() if "general" in [c.lower() for c in m.logical_capabilities]]
        return matches


default_registry = ModelRegistry()
