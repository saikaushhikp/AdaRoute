import os

from adaroute.experiments.runner import BenchmarkRunner
from adaroute.workloads.estimator import OutputLengthEstimator
from adaroute.workloads.generator import WorkloadGenerator


def test_tc_m8_01_independent_workload_and_estimator_error_tracking():
    """[TC_M8_01] Milestone M8: Decoupled ground truth generation allows non-circular estimation error tracking."""
    generator = WorkloadGenerator(seed=123)
    trace = generator.generate_trace(count=20)
    assert len(trace) == 20

    estimator = OutputLengthEstimator()
    errors = []

    for req in trace:
        est = estimator.estimate(req.task_type, req.prompt_tokens)
        # Ground truth actual_output_tokens was sampled independently!
        err = estimator.record_observation(req.task_type, est, req.actual_output_tokens)
        errors.append(err)

    stats = estimator.get_stats()
    assert stats["samples"] == 20
    assert stats["mae"] > 0.0  # Real non-zero error demonstrating non-circular evaluation


def test_tc_m8_02_reproducible_benchmark_execution(tmp_path):
    """[TC_M8_02] Milestone M8: Fixed seed produces 100% reproducible benchmark results saved to disk."""
    runner1 = BenchmarkRunner(seed=99)
    res1 = runner1.run_all(num_requests=10, output_dir=str(tmp_path / "run1"))

    runner2 = BenchmarkRunner(seed=99)
    res2 = runner2.run_all(num_requests=10, output_dir=str(tmp_path / "run2"))

    assert len(res1) == 6
    assert len(res2) == 6

    # Invariant: Fixed seed produces identical metrics
    for r1, r2 in zip(res1, res2):
        assert r1["model_policy"] == r2["model_policy"]
        assert r1["replica_scheduler"] == r2["replica_scheduler"]
        assert r1["total_cost_usd"] == r2["total_cost_usd"]
        assert r1["avg_quality_score"] == r2["avg_quality_score"]
        assert r1["slo_attainment_pct"] == r2["slo_attainment_pct"]

    # Verify JSON artifacts exist
    assert os.path.exists(tmp_path / "run1" / "benchmark_summary.json")
    assert os.path.exists(tmp_path / "run1" / "workload_trace.jsonl")
