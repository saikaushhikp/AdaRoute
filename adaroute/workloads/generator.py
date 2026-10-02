"""Independent Workload Generator for AdaRoute.

Generates ground-truth demand (prompt length, actual output demand, privacy,
budget) completely independent of the router's pre-inference estimator.
"""
from __future__ import annotations

import random

from pydantic import BaseModel

from adaroute.schemas import PrivacyLevel

from .tokenizer import count_tokens


class SyntheticRequest(BaseModel):
    request_id: str
    task_type: str
    prompt: str
    prompt_tokens: int
    actual_output_tokens: int  # Ground truth hidden from router
    privacy_level: PrivacyLevel
    budget: float
    quality_requirement: str
    latency_slo: float
    tenant_id: str = "tenant-1"
    arrival_offset_ms: float = 0.0


class WorkloadGenerator:
    TASK_PROMPTS = {
        "qa": [
            "What is the capital of Australia and what is its population?",
            "Explain the difference between TCP and UDP protocols in two sentences.",
            "Who proposed the theory of general relativity and in what year?",
            "What is the primary function of ribosomes inside biological cells?",
        ],
        "coding": [
            "Write a Python function to perform topological sort on a directed acyclic graph.",
            "Implement a thread-safe LRU cache in Python with get and put methods.",
            "Write an asynchronous rate limiter using token bucket algorithm in Python.",
            "Create a FastAPI dependency that verifies JWT bearer tokens with expiration checks.",
        ],
        "summarization": [
            "Summarize the following 10-page quarterly earnings call transcript highlighting revenue growth, operating margin compression, and forward-looking guidance for cloud infrastructure services across North America and APAC regions.",
            "Provide a concise executive summary of the distributed systems consensus paper discussing Raft leader election, log replication, and safety guarantees under network partitions.",
        ],
        "reasoning": [
            "Solve this math problem: A train leaves Station A at 60 mph. Another leaves Station B at 80 mph. If distance is 280 miles, when do they collide?",
            "Analyze the game-theoretic Nash equilibrium for a duopoly price competition under asymmetric marginal costs.",
        ],
        "creative": [
            "Compose a sci-fi short story about an autonomous AI gateway achieving consciousness inside a liquid-cooled data center.",
            "Write a persuasive blog post advocating for open-source AI models in academic computing infrastructure.",
        ],
    }

    TASK_DISTRIBUTIONS = {
        "qa": (45, 15),           # (mean, std) for actual output tokens
        "coding": (310, 60),
        "summarization": (175, 40),
        "reasoning": (230, 50),
        "creative": (270, 70),
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_trace(
        self,
        count: int = 50,
        mix: dict[str, float] | None = None,
        privacy_ratio: float = 0.3,
        budget_range: tuple = (0.005, 0.08),
    ) -> list[SyntheticRequest]:
        """Generate an independent trace with ground truth attributes."""
        self.rng.seed(self.seed)
        mix = mix or {"qa": 0.3, "coding": 0.3, "summarization": 0.2, "reasoning": 0.2}

        tasks = list(mix.keys())
        weights = list(mix.values())

        requests = []
        current_time_ms = 0.0

        for i in range(count):
            task = self.rng.choices(tasks, weights=weights, k=1)[0]
            prompt_template = self.rng.choice(self.TASK_PROMPTS[task])

            # Optionally extend prompt to simulate varied input lengths
            multiplier = self.rng.choice([1, 1, 2, 4 if task == "summarization" else 1])
            full_prompt = (prompt_template + " ") * multiplier
            p_tokens = count_tokens(full_prompt)

            # Ground-truth output tokens sampled independently
            mean_out, std_out = self.TASK_DISTRIBUTIONS[task]
            actual_out = max(5, int(self.rng.gauss(mean_out, std_out)))

            # Privacy assignment
            if self.rng.random() < privacy_ratio:
                privacy = self.rng.choice([PrivacyLevel.INTERNAL_ONLY, PrivacyLevel.CONFIDENTIAL])
            else:
                privacy = PrivacyLevel.PUBLIC

            budget = round(self.rng.uniform(*budget_range), 4)
            quality = self.rng.choice(["medium", "high", "high" if task == "coding" else "medium"])
            slo = self.rng.choice([1.5, 2.0, 3.0])

            # Burst / arrival interval (Poisson process)
            inter_arrival = self.rng.expovariate(1.0 / 100.0)  # ~10 req/s mean
            current_time_ms += inter_arrival

            req = SyntheticRequest(
                request_id=f"req-{i+1:04d}",
                task_type=task,
                prompt=full_prompt,
                prompt_tokens=p_tokens,
                actual_output_tokens=actual_out,
                privacy_level=privacy,
                budget=budget,
                quality_requirement=quality,
                latency_slo=slo,
                tenant_id=f"tenant-{(i % 3) + 1}",
                arrival_offset_ms=round(current_time_ms, 2),
            )
            requests.append(req)

        return requests

    def save_trace(self, filepath: str, requests: list[SyntheticRequest]):
        with open(filepath, "w") as f:
            f.writelines(r.model_dump_json() + "\n" for r in requests)
