"""Tests for Week-5 Commitment 3: Automated Online Estimator Calibration via Bayesian Linear Regression (Milestone M11).

Verifies:
- Domain prior initialization (coding vs QA vs reasoning priors)
- Sequential online Bayesian updating (mean tracks empirical output tokens)
- Variance shrinkage (predictive variance shrinks with sample count)
- 95% Upper Credible Bound reservation (reservation >= expected always)
- Explicit max_tokens clamping
- Disk persistence and reload across restarts
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from adaroute.workloads.bayesian_estimator import (
    BayesianLinearEstimator,
    DomainBayesianRegressor,
)


def test_tc_m11_01_domain_priors_differentiation():
    estimator = BayesianLinearEstimator()
    est_qa = estimator.estimate("qa", prompt_tokens=50)
    est_coding = estimator.estimate("coding", prompt_tokens=50, is_code=True)
    est_reasoning = estimator.estimate("reasoning", prompt_tokens=50, is_reasoning=True)

    # Coding and reasoning priors must naturally exceed QA prior
    assert est_coding.expected_tokens > est_qa.expected_tokens
    assert est_reasoning.expected_tokens > est_qa.expected_tokens
    assert est_coding.reservation_tokens >= est_coding.expected_tokens
    assert est_qa.reservation_tokens >= est_qa.expected_tokens


def test_tc_m11_02_online_bayesian_updating_and_variance_shrinkage():
    regressor = DomainBayesianRegressor(
        domain_name="general",
        prior_mean=[100.0, 5.0, 0.0, 0.0],
        prior_precision_alpha=0.01,
        noise_precision_beta=0.1,
    )

    # Initial prediction before observations
    pred_init = regressor.predict(prompt_tokens=50)
    init_std = pred_init.predictive_std
    init_expected = pred_init.expected_tokens

    # Feed 15 sequential observations with higher completion tokens (~250 tokens)
    for _ in range(15):
        regressor.update(prompt_tokens=50, actual_output_tokens=250)

    pred_updated = regressor.predict(prompt_tokens=50)

    # 1. Expected tokens shifted towards empirical data (250)
    assert pred_updated.expected_tokens > init_expected
    assert pred_updated.expected_tokens >= 180

    # 2. Predictive variance shrank (standard deviation decreased)
    assert pred_updated.predictive_std < init_std

    # 3. Confidence increased
    assert pred_updated.confidence > pred_init.confidence

    # 4. Sample count tracked
    assert pred_updated.sample_count == 15


def test_tc_m11_03_95_percent_upper_credible_bound_reservation():
    estimator = BayesianLinearEstimator()
    est = estimator.estimate("general", prompt_tokens=100)

    # Invariant: reservation tokens (95% UCB) must be strictly >= expected tokens
    assert est.reservation_tokens >= est.expected_tokens
    # And difference must be bounded by ~1.645 * std
    diff = est.reservation_tokens - est.expected_tokens
    assert diff >= int(1.645 * est.predictive_std) - 1


def test_tc_m11_04_explicit_max_tokens_clamping():
    estimator = BayesianLinearEstimator()
    # If client requests max_tokens = 40, expected and reservation must clamp to 40
    est = estimator.estimate("coding", prompt_tokens=50, is_code=True, max_tokens=40)
    assert est.expected_tokens == 40
    assert est.reservation_tokens == 40


def test_tc_m11_05_persistence_and_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file = Path(tmpdir) / "bayesian_weights.json"
        est1 = BayesianLinearEstimator(persistence_path=save_file)

        # Train with 10 observations
        for _ in range(10):
            est1.record_actual("qa", prompt_tokens=30, actual_output_tokens=90)

        pred1 = est1.estimate("qa", prompt_tokens=30)
        assert save_file.exists()

        # Reload in fresh instance
        est2 = BayesianLinearEstimator(persistence_path=save_file)
        est2.load(save_file)
        pred2 = est2.estimate("qa", prompt_tokens=30)

        assert pred1.expected_tokens == pred2.expected_tokens
        assert pred1.reservation_tokens == pred2.reservation_tokens
        assert pred1.sample_count == pred2.sample_count
