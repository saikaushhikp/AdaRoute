"""Immutable Candidate Features Snapshot with Full Metric Provenance (Week-4 Revamp).

Every feature carries:
- value: float | int | bool | str
- source: "configured" | "observed" | "estimated" | "simulated" | "provider"
- timestamp: float
- sample_count: int
- confidence: float
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .candidates import CandidateDeployment
from .estimator import OutputEstimate
from .signals import SemanticSignals


@dataclass(frozen=True)
class ProvenanceValue:
    """Individual feature value with traceable origin, confidence, and timestamp."""

    value: float
    source: str  # "configured" | "observed" | "estimated" | "simulated" | "provider"
    timestamp: float = field(default_factory=time.time)
    sample_count: int = 1
    confidence: float = 0.80

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 6) if isinstance(self.value, float) else self.value,
            "source": self.source,
            "timestamp": round(self.timestamp, 2),
            "sample_count": self.sample_count,
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class CandidateFeatures:
    """Immutable snapshot of candidate features used for Pareto pruning and scoring."""

    candidate_id: str
    model_id: str
    deployment_id: str
    provider_id: str

    # Metrics with full provenance
    cost_expected: ProvenanceValue
    cost_reserved: ProvenanceValue
    cost_worst_case: ProvenanceValue
    quality: ProvenanceValue
    latency_p50: ProvenanceValue
    latency_p95: ProvenanceValue
    failure_rate: ProvenanceValue
    session_continuity: ProvenanceValue
    semantic_fit: ProvenanceValue
    uncertainty: ProvenanceValue

    # Hardware & topology characteristics
    is_cloud: bool
    is_local: bool
    is_mock: bool
    context_window: int
    queue_depth: int

    def get_metric_value(self, metric_name: str) -> float | None:
        """Retrieves raw metric value by canonical name."""
        name_clean = metric_name.lower().strip()
        mapping = {
            "cost": self.cost_expected.value,
            "cost_expected": self.cost_expected.value,
            "cost_reserved": self.cost_reserved.value,
            "cost_worst_case": self.cost_worst_case.value,
            "quality": self.quality.value,
            "quality_loss": 1.0 - self.quality.value,
            "latency": self.latency_p50.value,
            "latency_p50": self.latency_p50.value,
            "latency_p95": self.latency_p95.value,
            "reliability": 1.0 - self.failure_rate.value,
            "failure": self.failure_rate.value,
            "failure_rate": self.failure_rate.value,
            "session_switch": 1.0 - self.session_continuity.value,
            "session_continuity": self.session_continuity.value,
            "semantic_fit": self.semantic_fit.value,
            "uncertainty": self.uncertainty.value,
            "queue_depth": float(self.queue_depth),
        }
        return mapping.get(name_clean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_id": self.model_id,
            "deployment_id": self.deployment_id,
            "provider_id": self.provider_id,
            "cost_expected": self.cost_expected.to_dict(),
            "cost_reserved": self.cost_reserved.to_dict(),
            "cost_worst_case": self.cost_worst_case.to_dict(),
            "quality": self.quality.to_dict(),
            "latency_p50": self.latency_p50.to_dict(),
            "failure_rate": self.failure_rate.to_dict(),
            "session_continuity": self.session_continuity.to_dict(),
            "semantic_fit": self.semantic_fit.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "is_cloud": self.is_cloud,
            "queue_depth": self.queue_depth,
        }


class FeatureSnapshotBuilder:
    """Builds immutable CandidateFeatures snapshots from candidate, signals, estimate, and session."""

    @classmethod
    def build(
        cls,
        candidate: CandidateDeployment,
        capability: str,
        prompt_tokens: int,
        estimate: OutputEstimate,
        max_tokens: int | None = None,
        signals: SemanticSignals | None = None,
        previous_deployment_id: str | None = None,
    ) -> CandidateFeatures:
        now = time.time()
        # 1. Cost Calculations
        in_p = candidate.input_price_per_1k
        out_p = candidate.output_price_per_1k

        cost_exp = (prompt_tokens / 1000.0) * in_p + (estimate.expected_tokens / 1000.0) * out_p
        cost_res = (prompt_tokens / 1000.0) * in_p + (estimate.reservation_tokens / 1000.0) * out_p

        eff_max_tokens = max_tokens if (max_tokens and max_tokens > 0) else estimate.reservation_tokens
        cost_worst = (prompt_tokens / 1000.0) * in_p + (eff_max_tokens / 1000.0) * out_p

        cost_source = "simulated" if candidate.is_mock else ("estimated" if cost_exp > 0.0 else "configured")

        prov_cost_exp = ProvenanceValue(
            value=cost_exp,
            source=cost_source,
            timestamp=now,
            sample_count=estimate.sample_count,
            confidence=estimate.confidence,
        )
        prov_cost_res = ProvenanceValue(
            value=cost_res,
            source=cost_source,
            timestamp=now,
            sample_count=estimate.sample_count,
            confidence=estimate.confidence,
        )
        prov_cost_worst = ProvenanceValue(
            value=cost_worst,
            source=cost_source,
            timestamp=now,
            sample_count=estimate.sample_count,
            confidence=0.99 if max_tokens else estimate.confidence,
        )

        # 2. Quality by Capability
        cap_prof = candidate.get_capability_profile(capability)
        q_val = cap_prof.quality_value if cap_prof else candidate.get_capability_quality(capability)
        q_src = cap_prof.quality_source if cap_prof else "configured"
        q_conf = cap_prof.quality_confidence if cap_prof else 0.50

        prov_quality = ProvenanceValue(
            value=q_val,
            source=q_src,
            timestamp=now,
            sample_count=1,
            confidence=q_conf,
        )

        # 3. Latency
        snap = candidate.runtime
        prov_lat_p50 = ProvenanceValue(
            value=snap.p50_latency_ms,
            source="observed" if snap.sample_count > 0 else "configured",
            timestamp=snap.last_heartbeat or now,
            sample_count=snap.sample_count,
            confidence=min(0.95, 0.40 + 0.1 * snap.sample_count),
        )
        prov_lat_p95 = ProvenanceValue(
            value=snap.p95_latency_ms,
            source="observed" if snap.sample_count > 0 else "configured",
            timestamp=snap.last_heartbeat or now,
            sample_count=snap.sample_count,
            confidence=min(0.95, 0.40 + 0.1 * snap.sample_count),
        )

        # 4. Failure Rate / Reliability
        prov_failure = ProvenanceValue(
            value=snap.failure_rate,
            source="observed" if snap.sample_count > 0 else "configured",
            timestamp=snap.last_heartbeat or now,
            sample_count=snap.sample_count,
            confidence=min(0.95, 0.50 + 0.05 * snap.sample_count),
        )

        # 5. Session Continuity
        is_same_dep = bool(previous_deployment_id and previous_deployment_id == candidate.deployment_id)
        prov_continuity = ProvenanceValue(
            value=1.0 if is_same_dep else 0.0,
            source="observed",
            timestamp=now,
            sample_count=1,
            confidence=1.0,
        )

        # 6. Semantic Fit
        sem_fit = 0.50
        if signals:
            if signals.reasoning_mode == "reasoning_required":
                # Prefer models with high quality on reasoning
                sem_fit = candidate.get_capability_quality("reasoning")
            elif signals.has_code:
                sem_fit = candidate.get_capability_quality("coding")
            else:
                sem_fit = candidate.get_capability_quality("general")

        prov_semantic = ProvenanceValue(
            value=sem_fit,
            source=signals.source if signals else "configured",
            timestamp=now,
            sample_count=1,
            confidence=0.75,
        )

        # 7. Uncertainty Penalty
        # High uncertainty if quality confidence is low or sample count is zero
        unc = max(0.0, 1.0 - q_conf) * 0.5 + (0.5 if snap.sample_count == 0 else 0.0)
        prov_unc = ProvenanceValue(
            value=unc,
            source="estimated",
            timestamp=now,
            sample_count=1,
            confidence=0.80,
        )

        return CandidateFeatures(
            candidate_id=candidate.deployment_id,
            model_id=candidate.model_id,
            deployment_id=candidate.deployment_id,
            provider_id=candidate.provider_id,
            cost_expected=prov_cost_exp,
            cost_reserved=prov_cost_res,
            cost_worst_case=prov_cost_worst,
            quality=prov_quality,
            latency_p50=prov_lat_p50,
            latency_p95=prov_lat_p95,
            failure_rate=prov_failure,
            session_continuity=prov_continuity,
            semantic_fit=prov_semantic,
            uncertainty=prov_unc,
            is_cloud=candidate.is_cloud,
            is_local=candidate.is_local,
            is_mock=candidate.is_mock,
            context_window=candidate.context_window,
            queue_depth=snap.queue_depth,
        )
