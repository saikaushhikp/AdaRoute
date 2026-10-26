"""Robust, Stable Feature Normalization (Week-4 Revamp).

Provides bounded domain normalization to prevent score instability across candidate sets:
- Zero-division safe
- Stable with single-candidate sets
- Configured reference bounds with fallback
- Records used normalization bounds in decision metadata for reproducibility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NormalizationDomain:
    """Configured reference domain bounds for a specific objective metric."""

    min_val: float
    max_val: float
    direction: str = "minimize"  # "minimize" or "maximize"

    def normalize(self, raw_value: float | None, default_val: float = 0.5) -> float:
        if raw_value is None:
            return default_val
        if self.max_val <= self.min_val:
            return 0.0

        # Clamp into domain bounds
        clamped = max(self.min_val, min(self.max_val, raw_value))
        norm = (clamped - self.min_val) / (self.max_val - self.min_val)

        # Convert to minimization scale: 0.0 is best, 1.0 is worst
        if self.direction == "maximize":
            return 1.0 - norm
        return norm


class NormalizationEngine:
    """Manages metric normalization across candidate sets."""

    # Default domain bounds for common LLM gateway metrics
    DEFAULT_DOMAINS = {
        "cost": NormalizationDomain(min_val=0.0, max_val=0.015, direction="minimize"),
        "cost_expected": NormalizationDomain(min_val=0.0, max_val=0.015, direction="minimize"),
        "latency": NormalizationDomain(min_val=20.0, max_val=3000.0, direction="minimize"),
        "latency_p50": NormalizationDomain(min_val=20.0, max_val=3000.0, direction="minimize"),
        "latency_p95": NormalizationDomain(min_val=50.0, max_val=5000.0, direction="minimize"),
        "quality": NormalizationDomain(min_val=0.50, max_val=1.0, direction="maximize"),
        "quality_loss": NormalizationDomain(min_val=0.0, max_val=0.50, direction="minimize"),
        "reliability": NormalizationDomain(min_val=0.50, max_val=1.0, direction="maximize"),
        "failure_rate": NormalizationDomain(min_val=0.0, max_val=1.0, direction="minimize"),
        "failure": NormalizationDomain(min_val=0.0, max_val=1.0, direction="minimize"),
        "session_switch": NormalizationDomain(min_val=0.0, max_val=1.0, direction="minimize"),
        "uncertainty": NormalizationDomain(min_val=0.0, max_val=1.0, direction="minimize"),
        "queue_depth": NormalizationDomain(min_val=0.0, max_val=20.0, direction="minimize"),
    }

    def __init__(self, custom_domains: dict[str, dict[str, Any]] | None = None):
        self.domains: dict[str, NormalizationDomain] = dict(self.DEFAULT_DOMAINS)
        if custom_domains:
            for mname, mdata in custom_domains.items():
                min_v = float(mdata.get("min", 0.0))
                max_v = float(mdata.get("max", 1.0))
                dir_v = str(mdata.get("direction", "minimize"))
                self.domains[mname.lower()] = NormalizationDomain(min_val=min_v, max_val=max_v, direction=dir_v)

    def normalize_feature(
        self, metric_name: str, raw_value: float | None, default_val: float = 0.5
    ) -> tuple[float, dict[str, Any]]:
        """Normalizes a raw feature value to [0.0, 1.0] (where 0.0 is best / minimized).

        Returns (normalized_value, bounds_dict).
        """
        dom = self.domains.get(
            metric_name.lower(),
            NormalizationDomain(min_val=0.0, max_val=1.0, direction="minimize"),
        )
        norm_val = dom.normalize(raw_value, default_val=default_val)
        bounds_record = {
            "min": dom.min_val,
            "max": dom.max_val,
            "direction": dom.direction,
        }
        return norm_val, bounds_record
