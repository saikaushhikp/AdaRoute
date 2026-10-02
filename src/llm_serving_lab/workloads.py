"""Deterministic synthetic workload generation."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_time: float
    task: str
    prompt_tokens: int
    expected_output_tokens: int
    minimum_quality: float
    latency_target_seconds: float
    privacy: str
    max_cost: float


def generate_workload(
    count: int, duration_seconds: float, seed: int, scenario: str
) -> list[Request]:
    """Generate a fixed-seed request trace with scenario-specific arrival patterns."""
    if count < 0 or duration_seconds <= 0:
        raise ValueError("count must be nonnegative and duration_seconds must be positive")
    if scenario not in {"bursty", "heterogeneous", "fault_injected"}:
        raise ValueError(f"unknown scenario: {scenario}")

    rng = random.Random(seed)
    requests: list[Request] = []
    for index in range(count):
        fraction = (index + rng.random()) / max(count, 1)
        if scenario == "bursty":
            # Concentrate arrivals into recurring short bursts separated by quiet gaps.
            phase = fraction * 6
            arrival = (int(phase) * 0.72) + (phase % 1) ** 2 * 0.72
            arrival *= duration_seconds / 4.32
        else:
            arrival = fraction * duration_seconds

        task = rng.choices(
            ["chat", "summarization", "code", "reasoning"],
            weights=[42, 24, 20, 14],
            k=1,
        )[0]
        if scenario == "heterogeneous":
            prompt = int(rng.lognormvariate(6.1, 1.0))
            output = int(rng.lognormvariate(4.0, 0.8))
        else:
            prompt = rng.choice([96, 256, 768, 2048, 6144, 12000])
            output = rng.choice([48, 96, 192, 384, 768])
        prompt = max(32, min(prompt, 48_000))
        output = max(16, min(output, 2_000))

        requests.append(
            Request(
                request_id=f"req-{index:05d}",
                arrival_time=arrival,
                task=task,
                prompt_tokens=prompt,
                expected_output_tokens=output,
                minimum_quality=rng.choice([0.68, 0.76, 0.84, 0.92]),
                latency_target_seconds=rng.choice([2.0, 4.0, 8.0, 16.0]),
                privacy=rng.choices(
                    ["standard", "sensitive", "local_only"],
                    weights=[82, 15, 3],
                    k=1,
                )[0],
                max_cost=rng.choice([0.0005, 0.003, 0.02, 0.08]),
            )
        )
    return sorted(requests, key=lambda request: (request.arrival_time, request.request_id))