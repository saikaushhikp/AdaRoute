"""Pre-inference output token length estimator (System-Agnostic, Config-Driven).

Never leaks actual future generation length into the routing decision.
Configurable via config.yaml, with adaptive online Bayesian/quantile updates.
Maintains running error statistics (MAE, signed error, per-task metrics).
"""

from __future__ import annotations

from typing import Any


class OutputEstimate:
    def __init__(
        self,
        expected_tokens: int,
        upper_bound_tokens: int,
        confidence: float,
        strategy: str,
    ):
        self.expected_tokens = expected_tokens
        self.upper_bound_tokens = upper_bound_tokens
        self.confidence = confidence
        self.strategy = strategy

    def __int__(self) -> int:
        return self.expected_tokens


class OutputLengthEstimator:
    # Default baseline priors if no configuration provided
    DEFAULT_TASK_PRIORS = {
        "qa": 50,
        "factual_qa": 45,
        "summarization": 160,
        "coding": 280,
        "reasoning": 220,
        "structured_extraction": 90,
        "creative": 250,
        "general": 80,
    }

    def __init__(
        self,
        task_priors: dict[str, int] | None = None,
        max_token_ratio: float = 0.15,
        prompt_expansion_threshold: int = 500,
        prompt_expansion_ratio: float = 1.25,
        default_prior: int = 80,
        historical_quantile: float = 0.75,
    ):
        self.task_priors = dict(self.DEFAULT_TASK_PRIORS)
        if task_priors:
            self.task_priors.update({k.lower().strip(): int(v) for k, v in task_priors.items()})

        self.max_token_ratio = float(max_token_ratio)
        self.prompt_expansion_threshold = int(prompt_expansion_threshold)
        self.prompt_expansion_ratio = float(prompt_expansion_ratio)
        self.default_prior = int(default_prior)
        self.historical_quantile = float(historical_quantile)

        self.error_history: list[float] = []
        self.signed_error_history: list[float] = []
        self.task_errors: dict[str, list[float]] = {}
        self.task_actuals: dict[str, list[int]] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> OutputLengthEstimator:
        """Instantiates OutputLengthEstimator driven by deployment config dictionary."""
        cfg = config or {}
        est_cfg = cfg.get("estimation", {})
        return cls(
            task_priors=est_cfg.get("task_priors"),
            max_token_ratio=est_cfg.get("max_token_ratio", 0.15),
            prompt_expansion_threshold=est_cfg.get("prompt_expansion_threshold", 500),
            prompt_expansion_ratio=est_cfg.get("prompt_expansion_ratio", 1.25),
            default_prior=est_cfg.get("default_prior", 80),
            historical_quantile=est_cfg.get("historical_quantile", 0.75),
        )

    def estimate_detailed(
        self,
        task_type: str,
        prompt_tokens: int,
        max_tokens: int | None = None,
        model_id: str | None = None,
    ) -> OutputEstimate:
        """Returns structured OutputEstimate with expected and upper-bound tokens."""
        task = task_type.lower().strip()
        strategy = "config_prior"

        # 1. Check if sufficient online historical observations exist for empirical quantile
        actuals = self.task_actuals.get(task, [])
        if len(actuals) >= 5:
            sorted_actuals = sorted(actuals)
            idx = int(len(sorted_actuals) * self.historical_quantile)
            idx = min(idx, len(sorted_actuals) - 1)
            base_estimate = sorted_actuals[idx]
            strategy = "historical_quantile"
        else:
            base_estimate = self.task_priors.get(task, self.default_prior)

        # 2. Max tokens constraint
        if max_tokens is not None and max_tokens > 0:
            if max_tokens <= base_estimate:
                expected = max_tokens
            else:
                expected = min(max_tokens, int(base_estimate + self.max_token_ratio * (max_tokens - base_estimate)))
            upper_bound = max_tokens
        else:
            expected = base_estimate
            upper_bound = int(base_estimate * 2.0)

        # 3. Prompt expansion adjustment
        if prompt_tokens > self.prompt_expansion_threshold and task in ("summarization", "reasoning"):
            expected = int(expected * self.prompt_expansion_ratio)
            upper_bound = max(upper_bound, int(expected * 1.5))

        expected = max(1, expected)
        upper_bound = max(expected, upper_bound)
        confidence = 0.85 if strategy == "historical_quantile" else 0.70

        return OutputEstimate(
            expected_tokens=expected,
            upper_bound_tokens=upper_bound,
            confidence=confidence,
            strategy=strategy,
        )

    def estimate(
        self,
        task_type: str,
        prompt_tokens: int,
        max_tokens: int | None = None,
        model_id: str | None = None,
    ) -> int:
        """Estimate generation length before inference (Canonical API)."""
        detailed = self.estimate_detailed(task_type, prompt_tokens, max_tokens, model_id)
        return detailed.expected_tokens

    def record_observation(self, task_type: str, estimated: int, actual: int) -> float:
        """Record ground truth actual output, updating online historical actuals and error metrics."""
        err = abs(estimated - actual)
        signed_err = estimated - actual
        self.error_history.append(err)
        self.signed_error_history.append(signed_err)

        task = task_type.lower().strip()
        if task not in self.task_errors:
            self.task_errors[task] = []
        self.task_errors[task].append(err)

        if task not in self.task_actuals:
            self.task_actuals[task] = []
        self.task_actuals[task].append(actual)

        return err

    def get_stats(self) -> dict[str, float]:
        """Compute MAE, mean signed error, and per-task error."""
        if not self.error_history:
            return {"mae": 0.0, "mean_signed_error": 0.0, "samples": 0}

        mae = sum(self.error_history) / len(self.error_history)
        mean_signed = sum(self.signed_error_history) / len(self.signed_error_history)

        per_task_mae = {f"mae_{t}": sum(errs) / len(errs) for t, errs in self.task_errors.items() if errs}

        return {
            "mae": round(mae, 2),
            "mean_signed_error": round(mean_signed, 2),
            "samples": len(self.error_history),
            **per_task_mae,
        }
