"""Automated Online Estimator Calibration via Bayesian Linear Regression (Week-5 Milestone M11).

Replaces heuristic token estimation with online recursive Bayesian linear regression:
- Conjugate Gaussian-Gaussian prior-posterior updates (Bishop PRML §3.3)
- Feature vector: x = [1, prompt_tokens / 100.0, is_code, is_reasoning]
- Sequential parameter updates without full dataset recomputation:
    Sigma_N^-1 = Sigma_{N-1}^-1 + beta * x_n * x_n^T
    mu_N = Sigma_N * (Sigma_{N-1}^-1 * mu_{N-1} + beta * y_n * x_n)
- Computes analytical predictive distribution:
    mu_* = mu_N^T * x_*
    sigma_*^2 = 1/beta + x_*^T * Sigma_N * x_*
- Derives 95% Upper Credible Bound for provably safe worst-case reservation:
    reservation_tokens = mu_* + z_0.95 * sigma_*  (z_0.95 = 1.645)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BayesianEstimate:
    """Analytical predictive distribution output for token demand."""

    expected_tokens: int
    reservation_tokens: int
    predictive_std: float
    confidence: float
    sample_count: int
    strategy: str = "bayesian_online"

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_tokens": self.expected_tokens,
            "reservation_tokens": self.reservation_tokens,
            "predictive_std": round(self.predictive_std, 2),
            "confidence": round(self.confidence, 4),
            "sample_count": self.sample_count,
            "strategy": self.strategy,
        }


class DomainBayesianRegressor:
    """Individual Bayesian Linear Regression model for a specific task capability."""

    def __init__(
        self,
        domain_name: str,
        prior_mean: list[float] | None = None,
        prior_precision_alpha: float = 0.01,
        noise_precision_beta: float = 0.05,
    ):
        self.domain_name = domain_name
        self.dim = 4  # [bias, prompt_tok/100, is_code, is_reasoning]
        self.alpha = prior_precision_alpha  # Prior precision
        self.beta = noise_precision_beta  # Measurement noise precision

        # Prior mean: [baseline_intercept, prompt_slope, code_offset, reasoning_offset]
        self.mu = prior_mean or [80.0, 5.0, 30.0, 40.0]

        # Prior precision matrix Lambda_0 = alpha * I_4
        self.lambda_mat = [[self.alpha if i == j else 0.0 for j in range(self.dim)] for i in range(self.dim)]
        self.sample_count = 0

    @classmethod
    def _invert_4x4(cls, mat: list[list[float]]) -> list[list[float]]:
        """Inverts a 4x4 symmetric positive-definite matrix via Gaussian elimination."""
        n = 4
        # Augment with identity
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(mat)]
        for i in range(n):
            # Pivot
            pivot = aug[i][i]
            if abs(pivot) < 1e-12:
                pivot = 1e-6
            for j in range(2 * n):
                aug[i][j] /= pivot
            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * n):
                        aug[k][j] -= factor * aug[i][j]
        return [row[n:] for row in aug]

    def _extract_features(
        self, prompt_tokens: int, is_code: bool = False, is_reasoning: bool = False
    ) -> list[float]:
        return [
            1.0,
            float(prompt_tokens) / 100.0,
            1.0 if is_code else 0.0,
            1.0 if is_reasoning else 0.0,
        ]

    def update(
        self,
        prompt_tokens: int,
        actual_output_tokens: int,
        is_code: bool = False,
        is_reasoning: bool = False,
    ) -> None:
        """Sequential online recursive Bayesian update (Bishop PRML §3.3)."""
        x = self._extract_features(prompt_tokens, is_code, is_reasoning)
        y = float(actual_output_tokens)

        # 1. Update precision matrix: Lambda_{N} = Lambda_{N-1} + beta * x * x^T
        for i in range(self.dim):
            for j in range(self.dim):
                self.lambda_mat[i][j] += self.beta * x[i] * x[j]

        # 2. Invert precision matrix to get covariance Sigma_N
        sigma = self._invert_4x4(self.lambda_mat)

        # 3. Update mean vector:
        # mu_N = Sigma_N * (Lambda_{N-1} * mu_{N-1} + beta * y * x)
        # Note: Lambda_{N-1} * mu_{N-1} + beta * y * x = Lambda_N * mu_N in sequential form.
        # Track rhs vector directly:
        if not hasattr(self, "_rhs"):
            # Initial rhs_0 = Lambda_0 * mu_0
            self._rhs = [sum(self.lambda_mat[i][j] * self.mu[j] for j in range(self.dim)) for i in range(self.dim)]
        for i in range(self.dim):
            self._rhs[i] += self.beta * y * x[i]

        new_mu = [sum(sigma[i][j] * self._rhs[j] for j in range(self.dim)) for i in range(self.dim)]
        self.mu = new_mu
        self.sample_count += 1

    def predict(
        self,
        prompt_tokens: int,
        is_code: bool = False,
        is_reasoning: bool = False,
        max_tokens: int | None = None,
    ) -> BayesianEstimate:
        """Calculates expected output, predictive standard deviation, and 95% upper credible bound."""
        x = self._extract_features(prompt_tokens, is_code, is_reasoning)
        sigma = self._invert_4x4(self.lambda_mat)

        # Predictive mean mu_* = mu^T * x
        mu_star = sum(self.mu[i] * x[i] for i in range(self.dim))
        mu_star = max(10.0, mu_star)

        # Predictive variance sigma_*^2 = 1/beta + x^T * Sigma * x
        quad_form = sum(x[i] * sum(sigma[i][j] * x[j] for j in range(self.dim)) for i in range(self.dim))
        var_star = (1.0 / self.beta) + max(0.0, quad_form)
        std_star = math.sqrt(var_star)

        # 95% Upper Credible Bound: mu_* + 1.645 * sigma_*
        z_95 = 1.645
        reservation = mu_star + (z_95 * std_star)

        expected_int = round(mu_star)
        reservation_int = math.ceil(reservation)

        # Clamp with explicit max_tokens if provided
        if max_tokens is not None and max_tokens > 0:
            expected_int = min(expected_int, max_tokens)
            reservation_int = min(reservation_int, max_tokens)

        reservation_int = max(expected_int, reservation_int)

        # Confidence metric derived from variance shrinkage
        conf = 1.0 / (1.0 + (var_star / 500.0))
        conf = max(0.10, min(0.99, conf))

        return BayesianEstimate(
            expected_tokens=expected_int,
            reservation_tokens=reservation_int,
            predictive_std=std_star,
            confidence=conf,
            sample_count=self.sample_count,
            strategy="bayesian_online",
        )


class BayesianLinearEstimator:
    """Multi-domain Bayesian token estimator calibrated online via feedback loops."""

    DEFAULT_DOMAIN_PRIORS = {
        "qa": [45.0, 2.0, 0.0, 5.0],
        "general": [75.0, 4.0, 10.0, 15.0],
        "summarization": [160.0, 15.0, 0.0, 10.0],
        "coding": [280.0, 12.0, 40.0, 10.0],
        "reasoning": [220.0, 10.0, 10.0, 60.0],
        "creative": [250.0, 18.0, 0.0, 20.0],
    }

    def __init__(self, persistence_path: str | Path | None = None):
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.regressors: dict[str, DomainBayesianRegressor] = {}
        self._init_domains()

    def _init_domains(self) -> None:
        for domain, prior in self.DEFAULT_DOMAIN_PRIORS.items():
            self.regressors[domain] = DomainBayesianRegressor(
                domain_name=domain,
                prior_mean=list(prior),
            )

    def get_regressor(self, task_type: str) -> DomainBayesianRegressor:
        key = task_type.lower().strip()
        if key not in self.regressors:
            self.regressors[key] = DomainBayesianRegressor(
                domain_name=key,
                prior_mean=[80.0, 5.0, 20.0, 20.0],
            )
        return self.regressors[key]

    def estimate(
        self,
        task_type: str,
        prompt_tokens: int,
        is_code: bool = False,
        is_reasoning: bool = False,
        max_tokens: int | None = None,
    ) -> BayesianEstimate:
        reg = self.get_regressor(task_type)
        return reg.predict(
            prompt_tokens=prompt_tokens,
            is_code=is_code,
            is_reasoning=is_reasoning,
            max_tokens=max_tokens,
        )

    def record_actual(
        self,
        task_type: str,
        prompt_tokens: int,
        actual_output_tokens: int,
        is_code: bool = False,
        is_reasoning: bool = False,
    ) -> None:
        """Feedback loop: updates Bayesian posterior weights sequentially."""
        if actual_output_tokens <= 0:
            return
        reg = self.get_regressor(task_type)
        reg.update(
            prompt_tokens=prompt_tokens,
            actual_output_tokens=actual_output_tokens,
            is_code=is_code,
            is_reasoning=is_reasoning,
        )
        if self.persistence_path:
            self.save(self.persistence_path)

    def save(self, path: Path | str) -> None:
        data: dict[str, Any] = {}
        for dname, reg in self.regressors.items():
            data[dname] = {
                "mu": reg.mu,
                "lambda_mat": reg.lambda_mat,
                "sample_count": reg.sample_count,
            }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path | str) -> None:
        p = Path(path)
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for dname, saved in data.items():
            reg = self.get_regressor(dname)
            reg.mu = saved["mu"]
            reg.lambda_mat = saved["lambda_mat"]
            reg.sample_count = saved["sample_count"]
