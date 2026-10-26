"""Deployment-Aware Output Token Estimator (Week-4 Revamp).

Estimates expected and reservation completion tokens per:
(deployment_id, capability, prompt_length_bucket)

Follows strict precedence:
1. Explicit request max_tokens (clamps expected and reservation)
2. Deployment / capability configured prior
3. Bounded online quantile (only with >= minimum_observations)
4. Conservative fallback prior

Returns OutputEstimate with expected_tokens, reservation_tokens, confidence,
strategy, and sample_count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class OutputEstimate:
    """Structured output token estimate distinguishing ranking expectation from queue reservation."""

    expected_tokens: int
    reservation_tokens: int
    confidence: float
    strategy: str  # "explicit_max", "online_quantile", "configured_prior", "fallback"
    sample_count: int

    def __post_init__(self) -> None:
        self.expected_tokens = max(self.expected_tokens, 1)
        self.reservation_tokens = max(self.reservation_tokens, self.expected_tokens)


class DeploymentOutputEstimator:
    """Deployment-aware output token estimator."""

    def __init__(
        self,
        default_prior: int = 80,
        task_priors: dict[str, int] | None = None,
        deployment_priors: dict[str, dict[str, int]] | None = None,
        expected_quantile: float = 0.75,
        reservation_quantile: float = 0.95,
        minimum_observations: int = 5,
        max_token_ratio: float = 0.15,
        max_history_per_key: int = 100,
    ):
        self.default_prior = default_prior
        self.task_priors = {k.lower(): v for k, v in (task_priors or {}).items()}
        self.deployment_priors = {
            dep.lower(): {k.lower(): v for k, v in priors.items()}
            for dep, priors in (deployment_priors or {}).items()
        }
        self.expected_quantile = expected_quantile
        self.reservation_quantile = reservation_quantile
        self.minimum_observations = minimum_observations
        self.max_token_ratio = max_token_ratio
        self.max_history_per_key = max_history_per_key

        # Key: (deployment_id, capability, prompt_length_bucket) -> list of actual output tokens
        self.history: dict[tuple[str, str, str], list[int]] = {}

    @classmethod
    def get_prompt_bucket(cls, prompt_tokens: int) -> str:
        if prompt_tokens < 100:
            return "short"
        if prompt_tokens <= 500:
            return "medium"
        return "long"

    def estimate(
        self,
        deployment_id: str,
        capability: str,
        prompt_tokens: int,
        max_tokens: int | None = None,
    ) -> OutputEstimate:
        """Computes expected and reservation token counts for a deployment and capability."""
        dep_clean = deployment_id.lower()
        cap_clean = capability.lower()
        bucket = self.get_prompt_bucket(prompt_tokens)
        obs_key = (dep_clean, cap_clean, bucket)

        history_samples = self.history.get(obs_key, [])
        sample_count = len(history_samples)

        # 1. Configured prior lookup
        # Check deployment-specific prior first, then task prior, then default prior
        dep_specific = self.deployment_priors.get(dep_clean, {}).get(cap_clean)
        prior_val = dep_specific or self.task_priors.get(cap_clean) or self.default_prior

        # Adjust prior for long prompts if configured
        if bucket == "long":
            prior_val = int(prior_val * 1.25)

        strategy = "configured_prior"
        confidence = 0.50
        expected = prior_val
        reservation = math.ceil(prior_val * 1.35)

        # 2. Check if we have sufficient observations for online quantile
        if sample_count >= self.minimum_observations:
            sorted_samples = sorted(history_samples)
            # Calculate expected quantile
            exp_idx = math.ceil(self.expected_quantile * sample_count) - 1
            exp_idx = max(0, min(exp_idx, sample_count - 1))
            expected = sorted_samples[exp_idx]

            # Calculate reservation quantile
            res_idx = math.ceil(self.reservation_quantile * sample_count) - 1
            res_idx = max(0, min(res_idx, sample_count - 1))
            reservation = sorted_samples[res_idx]

            strategy = "online_quantile"
            confidence = min(0.95, 0.50 + (sample_count / (self.minimum_observations * 5.0)) * 0.45)

        # 3. Explicit request max_tokens precedence
        if max_tokens is not None and max_tokens > 0:
            if expected > max_tokens:
                expected = max_tokens
                strategy = "explicit_max"
                confidence = 0.99
            reservation = min(reservation, max_tokens)

        # Ensure reservation is at least expected
        reservation = max(expected, reservation)

        return OutputEstimate(
            expected_tokens=expected,
            reservation_tokens=reservation,
            confidence=confidence,
            strategy=strategy,
            sample_count=sample_count,
        )

    def record_observation(
        self,
        deployment_id: str,
        capability: str,
        prompt_tokens: int,
        actual_output_tokens: int,
    ) -> None:
        """Records an observed completion token length for future quantile adaptation."""
        if actual_output_tokens <= 0:
            return
        bucket = self.get_prompt_bucket(prompt_tokens)
        obs_key = (deployment_id.lower(), capability.lower(), bucket)
        if obs_key not in self.history:
            self.history[obs_key] = []
        samples = self.history[obs_key]
        samples.append(actual_output_tokens)
        if len(samples) > self.max_history_per_key:
            samples.pop(0)

    def reset_observations(self) -> None:
        """Resets all observed history."""
        self.history.clear()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DeploymentOutputEstimator:
        est_cfg = config.get("estimation", {})
        default_prior = int(est_cfg.get("default_prior", 80))
        task_priors = est_cfg.get("task_priors", {})
        deployment_priors = est_cfg.get("deployment_priors", {})
        expected_q = float(est_cfg.get("historical_quantile") or est_cfg.get("expected_quantile", 0.75))
        reservation_q = float(est_cfg.get("reservation_quantile", 0.95))
        min_obs = int(est_cfg.get("minimum_observations", 5))
        max_ratio = float(est_cfg.get("max_token_ratio", 0.15))

        return cls(
            default_prior=default_prior,
            task_priors=task_priors,
            deployment_priors=deployment_priors,
            expected_quantile=expected_q,
            reservation_quantile=reservation_q,
            minimum_observations=min_obs,
            max_token_ratio=max_ratio,
        )
