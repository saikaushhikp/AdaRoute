"""Candidate Deployment Representation.

Combines deployment, model definition, provider definition, and dynamic runtime
snapshot into a concrete, ranked candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaroute.schemas import PrivacyLevel

from .catalog import (
    CapabilityProfile,
    DeploymentDefinition,
    ModelDefinition,
    ProviderDefinition,
    RuntimeSnapshot,
    SystemCatalog,
)


@dataclass
class CandidateDeployment:
    """A fully contextualized candidate deployment ready for eligibility filtering and ranking."""

    deployment: DeploymentDefinition
    model: ModelDefinition
    provider: ProviderDefinition
    runtime: RuntimeSnapshot

    @property
    def deployment_id(self) -> str:
        return self.deployment.deployment_id

    @property
    def model_id(self) -> str:
        return self.model.model_id

    @property
    def provider_id(self) -> str:
        return self.provider.provider_id

    @property
    def adapter_kind(self) -> str:
        return self.provider.adapter_kind

    @property
    def execution_type(self) -> str:
        return self.provider.execution_type

    @property
    def is_cloud(self) -> bool:
        return self.execution_type == "external" or self.provider.adapter_kind == "litellm"

    @property
    def is_local(self) -> bool:
        return self.execution_type == "local"

    @property
    def is_mock(self) -> bool:
        return self.execution_type == "mock" or self.provider.adapter_kind == "mock"

    @property
    def input_price_per_1k(self) -> float:
        return self.deployment.input_price_per_1k

    @property
    def output_price_per_1k(self) -> float:
        return self.deployment.output_price_per_1k

    @property
    def context_window(self) -> int:
        return self.model.context_window

    @property
    def output_limit(self) -> int:
        return self.model.output_limit

    @property
    def allowed_privacy(self) -> set[PrivacyLevel]:
        return self.deployment.allowed_privacy

    @property
    def replicas(self) -> list[Any]:
        return self.deployment.replicas

    @property
    def endpoint(self) -> str | None:
        return self.deployment.endpoint or self.provider.default_endpoint

    @property
    def credential_reference(self) -> str | None:
        return self.provider.credential_reference

    @property
    def provider_model_name(self) -> str:
        return self.deployment.provider_model_name or self.model.model_id

    def get_capability_quality(self, capability: str) -> float:
        return self.model.get_capability_quality(capability)

    def get_capability_profile(self, capability: str) -> CapabilityProfile | None:
        return self.model.get_capability_profile(capability)

    def calculate_expected_cost(self, prompt_tokens: int, expected_tokens: int) -> float:
        return (prompt_tokens / 1000.0) * self.input_price_per_1k + (
            expected_tokens / 1000.0
        ) * self.output_price_per_1k

    def calculate_max_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        return (prompt_tokens / 1000.0) * self.input_price_per_1k + (
            max_tokens / 1000.0
        ) * self.output_price_per_1k


class CandidateResolver:
    """Resolves matching candidate deployments from a SystemCatalog for a given request."""

    def __init__(self, catalog: SystemCatalog):
        self.catalog = catalog

    def resolve_candidates(
        self, capability_or_model: str, target_deployment_id: str | None = None
    ) -> list[CandidateDeployment]:
        """Resolves all eligible CandidateDeployment objects matching the capability or model."""
        if target_deployment_id:
            dep = self.catalog.get_deployment(target_deployment_id)
            if dep and dep.enabled:
                mdef = self.catalog.get_model(dep.model_id)
                pdef = self.catalog.get_provider(dep.provider_id)
                if mdef and pdef:
                    snap = self.catalog.get_snapshot(dep.deployment_id)
                    return [CandidateDeployment(dep, mdef, pdef, snap)]
            return []

        matched_deployments = self.catalog.get_deployments_for_capability(capability_or_model)
        candidates: list[CandidateDeployment] = []
        for dep in matched_deployments:
            mdef = self.catalog.get_model(dep.model_id)
            pdef = self.catalog.get_provider(dep.provider_id)
            if not mdef or not pdef:
                continue
            snap = self.catalog.get_snapshot(dep.deployment_id)
            candidates.append(CandidateDeployment(dep, mdef, pdef, snap))

        return candidates
