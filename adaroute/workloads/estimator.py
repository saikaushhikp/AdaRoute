"""Pre-inference output token length estimator.

Never leaks actual future generation length into the routing decision.
Maintains running error statistics (MAE, signed error, per-task metrics).
"""
from __future__ import annotations


class OutputLengthEstimator:
    # Baseline prior medians per task type (calibrated from development traces)
    TASK_PRIORS = {
        "qa": 50,
        "factual_qa": 45,
        "summarization": 160,
        "coding": 280,
        "reasoning": 220,
        "structured_extraction": 90,
        "creative": 250,
        "general": 80,
    }

    def __init__(self):
        self.error_history: list[float] = []
        self.signed_error_history: list[float] = []
        self.task_errors: dict[str, list[float]] = {}

    def estimate(
        self,
        task_type: str,
        prompt_tokens: int,
        max_tokens: int | None = None,
        model_id: str | None = None,
    ) -> int:
        """Estimate generation length before inference."""
        task = task_type.lower().strip()
        base_estimate = self.TASK_PRIORS.get(task, self.TASK_PRIORS["general"])

        if max_tokens is not None and max_tokens > 0:
            if max_tokens <= base_estimate:
                return max_tokens
            # If max_tokens is set large, generation typically consumes a fraction
            estimate = min(max_tokens, int(base_estimate + 0.15 * (max_tokens - base_estimate)))
        else:
            estimate = base_estimate

        # Slight adjustment based on prompt length (longer prompts for reasoning/summarization
        # often correlate with proportional output, capped at 1.5x)
        if prompt_tokens > 500 and task in ("summarization", "reasoning"):
            estimate = int(estimate * 1.25)

        return max(1, estimate)

    def record_observation(self, task_type: str, estimated: int, actual: int) -> float:
        """Record ground truth actual output and track error metrics."""
        err = abs(estimated - actual)
        signed_err = estimated - actual
        self.error_history.append(err)
        self.signed_error_history.append(signed_err)

        task = task_type.lower().strip()
        if task not in self.task_errors:
            self.task_errors[task] = []
        self.task_errors[task].append(err)
        return err

    def get_stats(self) -> dict[str, float]:
        """Compute MAE, mean signed error, and per-task error."""
        if not self.error_history:
            return {"mae": 0.0, "mean_signed_error": 0.0, "samples": 0}

        mae = sum(self.error_history) / len(self.error_history)
        mean_signed = sum(self.signed_error_history) / len(self.signed_error_history)

        per_task_mae = {
            f"mae_{t}": sum(errs) / len(errs) for t, errs in self.task_errors.items() if errs
        }

        return {
            "mae": round(mae, 2),
            "mean_signed_error": round(mean_signed, 2),
            "samples": len(self.error_history),
            **per_task_mae,
        }
