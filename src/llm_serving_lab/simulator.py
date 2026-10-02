"""Discrete-event simulator for policy comparisons."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from .policies import Model, Policy, Replica, ReplicaState
from .workloads import Request


@dataclass
class RequestResult:
    policy: str
    request_id: str
    model_id: str | None
    replica_id: str | None
    task: str
    prompt_tokens: int
    output_tokens: int
    arrival_time: float
    ttft_seconds: float | None
    tpot_seconds: float | None
    latency_seconds: float | None
    estimated_cost: float
    quality: float
    success: bool
    retried: bool
    slo_met: bool


@dataclass
class SimulationResult:
    policy: str
    requests: list[RequestResult]
    summary: dict[str, float | int | str]


def run_simulation(
    requests: list[Request], models: list[Model], replicas: list[Replica], policy: Policy,
    seed: int, duration_seconds: float, faults_enabled: bool,
) -> SimulationResult:
    model_by_id = {model.model_id: model for model in models}
    states = {replica.replica_id: ReplicaState() for replica in replicas}
    rng = random.Random(seed)
    results: list[RequestResult] = []
    busy_time = {replica.replica_id: 0.0 for replica in replicas}
    completion_times: list[float] = []
    pending_completions = {replica.replica_id: [] for replica in replicas}

    for request in requests:
        feasible = [
            model for model in models
            if model.quality >= request.minimum_quality
            and request.privacy in model.privacy_levels
            and request.prompt_tokens + request.expected_output_tokens <= model.max_context_tokens
            and ((request.prompt_tokens + request.expected_output_tokens) * model.cost_per_million_tokens / 1_000_000) <= request.max_cost
        ]
        if not feasible:
            results.append(_failed_result(policy.name, request))
            continue

        model = policy.model_selector.select(request, feasible)
        model_replicas = [replica for replica in replicas if replica.model_id == model.model_id]
        for item in model_replicas:
            pending_completions[item.replica_id] = [
                finish for finish in pending_completions[item.replica_id] if finish > request.arrival_time
            ]
            current = states[item.replica_id]
            lane_times = current.lane_available_at or (0.0,) * max(item.capacity, 1)
            states[item.replica_id] = ReplicaState(
                available_at=min(lane_times),
                active=len(pending_completions[item.replica_id]),
                completions=current.completions,
                recent_latency=current.recent_latency,
                healthy=True,
                lane_available_at=lane_times,
            )
        eligible = model_replicas
        if not eligible:
            results.append(_failed_result(policy.name, request, model))
            continue

        replica = policy.replica_selector.select(request, eligible, states, request.arrival_time)
        retried = False
        faulted = faults_enabled and _faulted(replica.replica_id, request.arrival_time)
        if faulted:
            alternatives = [item for item in eligible if item.replica_id != replica.replica_id]
            if alternatives:
                replica = policy.replica_selector.select(request, alternatives, states, request.arrival_time)
                retried = True
            else:
                results.append(_failed_result(policy.name, request, model, replica, retried=True))
                continue

        if rng.random() > min(model.reliability, replica.reliability):
            results.append(_failed_result(policy.name, request, model, replica))
            continue

        state = states[replica.replica_id]
        lane_times = list(state.lane_available_at or (0.0,) * max(replica.capacity, 1))
        lane_index = min(range(len(lane_times)), key=lambda index: max(request.arrival_time, lane_times[index]))
        start = max(request.arrival_time, lane_times[lane_index])
        queue_wait = start - request.arrival_time
        prefill = request.prompt_tokens / (model.prefill_tokens_per_second * replica.speed_factor)
        tpot = 1 / (model.decode_tokens_per_second * replica.speed_factor)
        service = prefill + request.expected_output_tokens * tpot
        completion = start + service
        ttft = queue_wait + prefill + tpot
        latency = completion - request.arrival_time
        token_cost = (request.prompt_tokens + request.expected_output_tokens) * model.cost_per_million_tokens / 1_000_000
        states[replica.replica_id] = ReplicaState(
            available_at=min(lane_times[:lane_index] + [completion] + lane_times[lane_index + 1:]),
            active=state.active + 1,
            completions=state.completions + 1,
            recent_latency=(state.recent_latency * 0.7) + (latency * 0.3),
            healthy=state.healthy,
            lane_available_at=tuple(lane_times[:lane_index] + [completion] + lane_times[lane_index + 1:]),
        )
        pending_completions[replica.replica_id].append(completion)
        busy_time[replica.replica_id] += service
        completion_times.append(completion)
        results.append(
            RequestResult(
                policy=policy.name, request_id=request.request_id, model_id=model.model_id,
                replica_id=replica.replica_id, task=request.task, prompt_tokens=request.prompt_tokens,
                output_tokens=request.expected_output_tokens, arrival_time=request.arrival_time,
                ttft_seconds=ttft, tpot_seconds=tpot, latency_seconds=latency,
                estimated_cost=token_cost, quality=model.quality, success=True, retried=retried,
                slo_met=latency <= request.latency_target_seconds,
            )
        )

    summary = summarize(policy.name, results, busy_time, duration_seconds, completion_times)
    return SimulationResult(policy.name, results, summary)


def _faulted(replica_id: str, arrival_time: float) -> bool:
    # Repeatable 12-second outages offset by replica; availability returns automatically.
    offset = sum(ord(character) for character in replica_id) % 31
    phase = (arrival_time + offset) % 73
    return 31 <= phase < 43


def _failed_result(policy: str, request: Request, model: Model | None = None, replica: Replica | None = None, retried: bool = False) -> RequestResult:
    return RequestResult(
        policy=policy, request_id=request.request_id, model_id=model.model_id if model else None,
        replica_id=replica.replica_id if replica else None, task=request.task,
        prompt_tokens=request.prompt_tokens, output_tokens=request.expected_output_tokens,
        arrival_time=request.arrival_time, ttft_seconds=None, tpot_seconds=None,
        latency_seconds=None, estimated_cost=0.0, quality=model.quality if model else 0.0,
        success=False, retried=retried, slo_met=False,
    )


def summarize(
    policy: str, results: list[RequestResult], busy_time: dict[str, float], duration: float,
    completion_times: list[float],
) -> dict[str, float | int | str]:
    successful = [result for result in results if result.success]
    latencies = sorted(result.latency_seconds for result in successful if result.latency_seconds is not None)
    ttfts = [result.ttft_seconds for result in successful if result.ttft_seconds is not None]
    tpots = [result.tpot_seconds for result in successful if result.tpot_seconds is not None]
    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        return values[max(0, math.ceil(fraction * len(values)) - 1)]

    end = max([duration, *completion_times])
    total = len(results)
    return {
        "policy": policy,
        "requests": total,
        "successes": len(successful),
        "failures": total - len(successful),
        "retries": sum(result.retried for result in results),
        "throughput_rps": len(successful) / max(end, 1e-9),
        "mean_ttft_seconds": sum(ttfts) / len(ttfts) if ttfts else 0.0,
        "mean_tpot_seconds": sum(tpots) / len(tpots) if tpots else 0.0,
        "p95_latency_seconds": percentile(latencies, 0.95),
        "p99_latency_seconds": percentile(latencies, 0.99),
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "estimated_cost": sum(result.estimated_cost for result in results),
        "mean_quality": sum(result.quality for result in successful) / len(successful) if successful else 0.0,
        "slo_attainment": sum(result.slo_met for result in successful) / total if total else 0.0,
        "utilization": sum(busy_time.values()) / max(end * len(busy_time), 1e-9),
    }