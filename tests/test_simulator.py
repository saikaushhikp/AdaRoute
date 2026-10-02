import unittest

from llm_serving_lab.policies import (
    AdaptiveModel,
    CheapestModel,
    LeastLoadedReplica,
    Model,
    Replica,
    ReplicaState,
    StrongestModel,
)
from llm_serving_lab.simulator import run_simulation
from llm_serving_lab.workloads import Request, generate_workload


class WorkloadTests(unittest.TestCase):
    def test_generation_is_deterministic_and_heterogeneous(self):
        first = generate_workload(40, 60, 7, "heterogeneous")
        second = generate_workload(40, 60, 7, "heterogeneous")
        self.assertEqual(first, second)
        self.assertGreater(len({item.prompt_tokens for item in first}), 5)
        self.assertGreater(len({item.expected_output_tokens for item in first}), 5)


class PolicyBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.request = Request("r1", 0, "chat", 100, 40, 0.7, 10, "standard", 1)
        self.models = [
            Model("fast", 0.75, 0.5, 1000, 100, 2048, frozenset({"standard"}), 0.99),
            Model("strong", 0.95, 4, 500, 40, 8192, frozenset({"standard"}), 0.99),
        ]

    def test_model_selector_returns_logical_model(self):
        self.assertEqual(StrongestModel().select(self.request, self.models).model_id, "strong")
        self.assertIn(AdaptiveModel().select(self.request, self.models).model_id, {"fast", "strong"})

    def test_replica_selector_uses_only_supplied_instances(self):
        replicas = [
            Replica("busy", "fast", 1, 2, 0.99),
            Replica("idle", "fast", 0.8, 2, 0.99),
        ]
        states = {
            "busy": ReplicaState(available_at=8, active=2),
            "idle": ReplicaState(),
        }
        chosen = LeastLoadedReplica().select(self.request, replicas, states, 0)
        self.assertEqual(chosen.replica_id, "idle")

    def test_simulation_is_reproducible(self):
        trace = [self.request]
        replica = [Replica("fast-a", "fast", 1, 2, 1)]
        policy = type("FixedPolicy", (), {"name": "test", "model_selector": CheapestModel(), "replica_selector": LeastLoadedReplica()})()
        first = run_simulation(trace, self.models, replica, policy, 3, 10, False)
        second = run_simulation(trace, self.models, replica, policy, 3, 10, False)
        self.assertEqual(first.summary, second.summary)
        self.assertEqual(first.requests, second.requests)

    def test_replica_capacity_runs_requests_in_parallel(self):
        trace = [
            self.request,
            Request("r2", 0, "chat", 100, 40, 0.7, 10, "standard", 1),
        ]
        replica = [Replica("fast-a", "fast", 1, 2, 1)]
        policy = type("FixedPolicy", (), {"name": "test", "model_selector": CheapestModel(), "replica_selector": LeastLoadedReplica()})()
        result = run_simulation(trace, self.models, replica, policy, 3, 10, False)
        self.assertAlmostEqual(
            result.requests[0].latency_seconds,
            result.requests[1].latency_seconds,
        )


if __name__ == "__main__":
    unittest.main()