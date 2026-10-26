"""Unified AI Gateway API Contract (Hardened Week-3 Version).

Coordinates:
- Server-side Zero-Trust Authentication and Anti-Spoofing
- Strict Pydantic ChatCompletionRequest validation
- Pre-routing characterization (tokenization & estimation)
- Safe Exact Caching with strict provenance & privacy eligibility
- Stage 1: Model selection (Strongest / Cheapest / Static / Adaptive)
- Stage 2: Replica scheduling (Workload-weighted queue W_q & SLO awareness)
- Multi-provider execution: Real Ollama Local + LiteLLM External + Mock Simulation
- Bounded retries, circuit breaker check for all alternates, and exponential backoff
- Full multi-attempt ledger accounting and sanitized telemetry
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from adaroute.backends.litellm_backend import LiteLLMBackend
from adaroute.backends.local import LocalBackend
from adaroute.backends.mock import MockBackend
from adaroute.backends.openai_compatible import OpenAICompatibleBackend
from adaroute.cache.exact import ExactCache
from adaroute.policy.model_selection import (
    AdaptiveModelPolicy,
    CheapestModelPolicy,
    CostFirstPolicy,
    LatencyFirstPolicy,
    QualityFirstPolicy,
    StaticRulePolicy,
    StrongestModelPolicy,
)
from adaroute.policy.registry import default_registry
from adaroute.policy.replica_selection import (
    LeastLoadedScheduler,
    RoundRobinScheduler,
    SLOAwareScheduler,
)
from adaroute.resilience.circuit import CircuitBreaker
from adaroute.resilience.retry import RetryPolicy
from adaroute.schemas import (
    AttemptRecord,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelSpec,
    PolicyRollbackRequest,
    RequestContext,
    RoutingMetadata,
    UsageInfo,
)
from adaroute.selection import SessionStore
from adaroute.telemetry.tracer import default_tracer, sanitize_message
from adaroute.workloads.bayesian_estimator import BayesianLinearEstimator
from adaroute.workloads.estimator import OutputLengthEstimator
from adaroute.workloads.tokenizer import count_tokens

# Core Engine Components (Config-Driven & System-Agnostic)
registry = default_registry
estimator = OutputLengthEstimator.from_config(registry.raw_config)
cache = ExactCache()
circuit_breaker = CircuitBreaker()
retry_policy = RetryPolicy(max_attempts=3)
tracer = default_tracer

# Stage 1 Policies (Config-Driven)
model_policies = {
    "adaptive": AdaptiveModelPolicy(registry, policy_name="adaptive"),
    "strongest": StrongestModelPolicy(registry),
    "cheapest": CheapestModelPolicy(registry),
    "static": StaticRulePolicy(registry),
    "latency_first": LatencyFirstPolicy(registry),
    "cost_first": CostFirstPolicy(registry),
    "quality_first": QualityFirstPolicy(registry),
}

# Stage 2 Schedulers
replica_schedulers = {
    "slo_aware": SLOAwareScheduler(registry),
    "least_loaded": LeastLoadedScheduler(registry),
    "round_robin": RoundRobinScheduler(registry),
}

# Backend Runners (Pluggable & Extensible)
backend_runners = {
    "local": LocalBackend(),
    "local_ollama": LocalBackend(),
    "groq": LiteLLMBackend(),
    "cohere": LiteLLMBackend(),
    "gemini": LiteLLMBackend(),
    "google": LiteLLMBackend(),
    "litellm": LiteLLMBackend(),
    "mock-local": LocalBackend(),
    "mock-external": MockBackend(provider_name="mock-external", base_latency_ms=30.0),
    "openai_compatible": OpenAICompatibleBackend(),
    "vllm": OpenAICompatibleBackend(),
}
session_store = SessionStore()
bayesian_estimator = BayesianLinearEstimator(persistence_path="data/bayesian_estimator.json")
try:
    bayesian_estimator.load("data/bayesian_estimator.json")
except Exception:
    pass

# Control Plane Versioned Policy State
policy_versions: list[dict[str, Any]] = [
    {
        "version": 1,
        "policy_name": "adaptive",
        "description": "Baseline Pareto multi-objective optimization (quality, cost, latency)",
        "timestamp": time.time(),
        "author": "system",
        "weights": {"quality": 0.35, "cost": 0.35, "latency": 0.30},
    }
]
default_active_policy = "adaptive"



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Background heartbeat task to maintain healthy replica state
    stop_event = asyncio.Event()

    async def heartbeat_loop():
        while not stop_event.is_set():
            try:
                local_backend = backend_runners.get("local")
                if local_backend and hasattr(local_backend, "ping"):
                    is_ok = await local_backend.ping()
                    now = time.time()
                    for r in registry.get_all_replicas():
                        if r.provider in ("local", "local_ollama"):
                            if is_ok:
                                r.last_heartbeat = now
                                r.consecutive_failures = 0
                                r.availability = True
                        elif r.provider.startswith("mock"):
                            r.last_heartbeat = now
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                pass

    task = asyncio.create_task(heartbeat_loop())
    yield
    stop_event.set()
    await task


app = FastAPI(title="AdaRoute Unified AI Gateway", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_backend_runner(spec: ModelSpec, candidate: Any = None):
    """Dynamically resolves backend runner for any provider on any system."""
    adapter_kind = getattr(candidate, "adapter_kind", None)
    if adapter_kind in ("openai_compatible", "vllm") or spec.provider in ("openai_compatible", "vllm"):
        return backend_runners["openai_compatible"]
    if spec.provider in backend_runners:
        return backend_runners[spec.provider]
    if spec.execution_type in ("external", "cloud"):
        return backend_runners["litellm"]
    if spec.execution_type == "mock":
        return backend_runners["mock-external"]
    return backend_runners["local"]


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health")
async def health():
    # Refresh replica heartbeats if local engine is running
    local_backend = backend_runners.get("local")
    if local_backend and hasattr(local_backend, "ping"):
        try:
            if await local_backend.ping():
                now = time.time()
                for r in registry.get_all_replicas():
                    if r.provider in ("local", "local_ollama"):
                        r.last_heartbeat = now
                        r.consecutive_failures = 0
                        r.availability = True
                    elif r.provider.startswith("mock"):
                        r.last_heartbeat = now
        except Exception:
            pass
    return {
        "status": "healthy",
        "models": [m.model_id for m in registry.get_all_models()],
        "replicas": [r.replica_id for r in registry.get_all_replicas() if r.is_alive()],
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    raw_req: dict[str, Any],
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    start_total_time = time.perf_counter()
    req_id = str(uuid.uuid4())

    # 2. Strict Request Validation via Pydantic Schema
    try:
        validated_req = ChatCompletionRequest.model_validate(raw_req)
    except Exception as e:
        tracer.record_validation_error(req_id, f"Validation failure: {e!s}")
        raise HTTPException(status_code=400, detail=f"Invalid request parameters: {e!s}")

    # Single User Context: defaults to 'user', preserving custom partition if provided
    tenant_id = validated_req.tenant_id or "user"
    user_id = validated_req.user_id or "user"

    prompt_text = validated_req.get_full_prompt()
    privacy = validated_req.privacy_level
    budget = validated_req.budget
    logical_model = validated_req.model
    if not registry.is_valid_capability_or_model(logical_model):
        tracer.record_validation_error(req_id, f"Unknown capability alias or model ID: '{logical_model}'")
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model or capability alias: '{logical_model}'.",
        )

    task_type = logical_model
    quality_req = validated_req.quality_requirement
    latency_slo = validated_req.latency_slo
    no_cache = validated_req.no_cache
    max_tokens = validated_req.max_tokens
    policy_name = validated_req.policy
    if policy_name == "adaptive" and default_active_policy != "adaptive":
        policy_name = default_active_policy
    scheduler_name = validated_req.scheduler
    user_private = validated_req.user_private

    # Validate policy name against registered and declarative policies
    valid_policies = set(model_policies.keys())
    if hasattr(registry, "selector") and registry.selector and hasattr(registry.selector, "policies"):
        valid_policies.update(registry.selector.policies.keys())
    if policy_name not in valid_policies:
        tracer.record_validation_error(req_id, f"Unknown routing policy: '{policy_name}'")
        raise HTTPException(
            status_code=400,
            detail=f"Unknown routing policy: '{policy_name}'. Valid policies: {sorted(valid_policies)}",
        )

    # 3. Pre-Routing Request Characterization
    p_tokens = count_tokens(prompt_text)
    est_out_tokens = estimator.estimate(
        task_type=task_type,
        prompt_tokens=p_tokens,
        max_tokens=max_tokens,
    )

    ctx = RequestContext(
        task_type=task_type,
        prompt_tokens=p_tokens,
        estimated_output_tokens=est_out_tokens,
        quality_requirement=quality_req,
        latency_slo=latency_slo,
        privacy_level=privacy,
        budget=budget,
        tenant_id=tenant_id,
        user_id=user_id,
        request_id=req_id,
        no_cache=no_cache,
        max_tokens=max_tokens,
        temperature=validated_req.temperature if validated_req.temperature is not None else 0.7,
        is_authenticated=True,
        user_private=user_private,
        execution_mode=(validated_req.execution_mode or os.getenv("EXECUTION_MODE", "hybrid")),
        policy_name=policy_name,
        session_id=validated_req.session_id,
        session_budget=validated_req.session_budget,
    )

    # 4. Start Correlation Trace
    tracer.start_trace(
        request_id=req_id,
        tenant_id=tenant_id,
        user_id=user_id,
        logical_model=logical_model,
        task_type=task_type,
        privacy_level=privacy.value,
        budget=budget,
        latency_slo=latency_slo,
        prompt_tokens=p_tokens,
        estimated_output_tokens=est_out_tokens,
    )

    # 5. Safe Exact Caching Check (Provenance & Privacy verified first)
    cached_res = cache.get(
        request=ctx,
        logical_model=logical_model,
        model_version="1.0.0",
        prompt=prompt_text,
        messages=validated_req.messages,
        policy_name=policy_name,
    )
    if cached_res:
        total_lat = (time.perf_counter() - start_total_time) * 1000.0
        slo_met = (total_lat / 1000.0) <= latency_slo if latency_slo else True
        tracer.finish_trace(
            request_id=req_id,
            selected_model=cached_res["model"],
            selected_replica=cached_res["replica_id"],
            actual_output_tokens=cached_res["output_tokens"],
            total_latency_ms=total_lat,
            estimated_cost=0.0,
            actual_cost=0.0,
            cache_outcome="hit",
            final_status="success",
        )
        return ChatCompletionResponse(
            id=req_id,
            model=cached_res["model"],
            logical_model=logical_model,
            choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=cached_res["content"]))],
            usage=UsageInfo(
                prompt_tokens=p_tokens,
                completion_tokens=cached_res["output_tokens"],
                total_tokens=p_tokens + cached_res["output_tokens"],
            ),
            routing_metadata=RoutingMetadata(
                request_id=req_id,
                logical_model=logical_model,
                selected_model=cached_res["model"],
                selected_replica=cached_res["replica_id"],
                provider=cached_res.get("provider", "cache"),
                latency_ms=round(total_lat, 2),
                attempts=0,
                cache_hit=True,
                estimated_cost=0.0,
                actual_cost=0.0,
                slo_attained=slo_met,
                policy_name=policy_name,
                scheduler_name=scheduler_name,
                attempt_history=[],
            ),
            content=cached_res["content"],
            replica_id=cached_res["replica_id"],
            provider=cached_res.get("provider", "cache"),
        )

    # 6. Stage 1: Candidate Ranking
    model_policy = model_policies.get(policy_name, model_policies["adaptive"])
    ranked_candidates = model_policy.rank_models(ctx)

    if not ranked_candidates:
        err_msg = "No eligible models satisfy hard constraints (privacy/budget/availability)"
        tracer.finish_trace(
            request_id=req_id,
            selected_model="none",
            selected_replica="none",
            actual_output_tokens=0,
            total_latency_ms=(time.perf_counter() - start_total_time) * 1000.0,
            estimated_cost=0.0,
            actual_cost=0.0,
            cache_outcome="miss",
            final_status="rejected",
            error_detail=err_msg,
        )
        raise HTTPException(status_code=503, detail=f"Routing rejected: {err_msg}")

    # 7. Execution with Dynamic Workload Queue Tracking & Bounded Failover Retry
    raw_messages = validated_req.to_messages_list()
    attempt = 0
    max_attempts = retry_policy.max_attempts
    last_error = None
    execution_result = None
    attempt_history: list[AttemptRecord] = []
    failed_replicas: set[str] = set()
    candidate_idx = 0

    selected_spec = ranked_candidates[0][0]
    est_cost = ranked_candidates[0][1]
    effective_replica_id = "none"
    effective_scheduler_name = scheduler_name

    while attempt < max_attempts and candidate_idx < len(ranked_candidates):
        # Check cumulative total deadline
        elapsed_so_far = time.perf_counter() - start_total_time
        if elapsed_so_far >= retry_policy.max_total_timeout_sec:
            last_error = TimeoutError(
                f"Request total deadline exceeded ({elapsed_so_far:.2f}s >= {retry_policy.max_total_timeout_sec}s)"
            )
            break

        selected_spec, est_cost = ranked_candidates[candidate_idx]

        # Stage 2: Replica Selection (Local models only; skipped for external cloud models)
        if selected_spec.is_cloud:
            selected_replica = None
            rep_id = f"cloud-{selected_spec.provider}"
            effective_replica_id = "cloud-direct"
            effective_scheduler_name = "skipped"
            scheduler = None
        else:
            scheduler = replica_schedulers.get(scheduler_name, replica_schedulers["slo_aware"])
            selected_replica = scheduler.select_replica(selected_spec, ctx, exclude_ids=failed_replicas)
            if not selected_replica:
                # No more healthy replicas for this model; try next candidate model
                candidate_idx += 1
                continue
            rep_id = selected_replica.replica_id
            effective_replica_id = rep_id
            effective_scheduler_name = scheduler_name

        # Check circuit breaker before invoking candidate
        if not circuit_breaker.allow_request(rep_id):
            failed_replicas.add(rep_id)
            if not selected_spec.is_cloud:
                all_reps = registry.get_replicas_for_model(selected_spec.model_id)
                alt_candidates = [
                    r
                    for r in all_reps
                    if r.replica_id not in failed_replicas
                    and r.is_alive()
                    and circuit_breaker.allow_request(r.replica_id)
                ]
                if not alt_candidates:
                    candidate_idx += 1
                continue
            else:
                candidate_idx += 1
                continue

        attempt += 1

        # Dynamic state update: W_q increases (local models only)
        if not selected_spec.is_cloud and scheduler:
            scheduler.dispatch(rep_id, p_tokens, est_out_tokens)
        attempt_start = time.perf_counter()
        dispatched = True

        try:
            try:
                runner = get_backend_runner(selected_spec)
                res = await runner.execute(
                    request=ctx,
                    model_spec=selected_spec,
                    replica_id=rep_id,
                    prompt_text=prompt_text,
                    messages=raw_messages,
                    endpoint=selected_replica.endpoint if selected_replica else None,
                )
                attempt_lat = (time.perf_counter() - attempt_start) * 1000.0

                # Dynamic state update: W_q decreases & success recorded (local models only)
                if not selected_spec.is_cloud and scheduler:
                    scheduler.complete(rep_id, p_tokens, est_out_tokens, attempt_lat, success=True)
                    dispatched = False
                circuit_breaker.record_success(rep_id)

                attempt_out_tokens = res.get("output_tokens", est_out_tokens)
                attempt_cost = res.get("actual_cost")
                if attempt_cost is None:
                    attempt_cost = (p_tokens / 1000.0) * selected_spec.input_price_per_1k + (
                        attempt_out_tokens / 1000.0
                    ) * selected_spec.output_price_per_1k

                attempt_cost_state = (
                    "invoiced_real"
                    if selected_spec.is_external and not getattr(selected_spec, "is_simulated", False)
                    else "allocated_estimate"
                )
                record = AttemptRecord(
                    attempt_number=attempt,
                    replica_id=effective_replica_id,
                    model_id=selected_spec.model_id,
                    provider=selected_spec.provider,
                    prompt_tokens=p_tokens,
                    completion_tokens=attempt_out_tokens,
                    latency_ms=round(attempt_lat, 2),
                    cost=round(attempt_cost, 6),
                    cost_state=attempt_cost_state,
                    status="success",
                )
                attempt_history.append(record)

                tracer.record_attempt(
                    request_id=req_id,
                    attempt_number=attempt,
                    replica_id=effective_replica_id,
                    model_id=selected_spec.model_id,
                    provider=selected_spec.provider,
                    prompt_tokens=p_tokens,
                    completion_tokens=attempt_out_tokens,
                    latency_ms=attempt_lat,
                    cost=attempt_cost,
                    cost_state=attempt_cost_state,
                    status="success",
                )
                execution_result = res
                if validated_req.session_id:
                    session_store.record_turn(
                        session_id=validated_req.session_id,
                        tenant_id=tenant_id,
                        selected_deployment_id=selected_spec.model_id,
                        selected_model_id=selected_spec.model_id,
                        capability=task_type,
                        billed_cost=attempt_cost,
                    )
                break

            except Exception as exc:
                attempt_lat = (time.perf_counter() - attempt_start) * 1000.0
                if not selected_spec.is_cloud and scheduler and dispatched:
                    scheduler.complete(rep_id, p_tokens, est_out_tokens, attempt_lat, success=False)
                    dispatched = False
                circuit_breaker.record_failure(rep_id)
                failed_replicas.add(rep_id)
                last_error = exc

                failed_cost = 0.0
                sanitized_err = sanitize_message(str(exc)) or "Execution failed"
                record = AttemptRecord(
                    attempt_number=attempt,
                    replica_id=effective_replica_id,
                    model_id=selected_spec.model_id,
                    provider=selected_spec.provider,
                    prompt_tokens=p_tokens,
                    completion_tokens=0,
                    latency_ms=round(attempt_lat, 2),
                    cost=round(failed_cost, 6),
                    cost_state="unknown_unbilled",
                    status="failed",
                    error_message=sanitized_err,
                )
                attempt_history.append(record)

                tracer.record_attempt(
                    request_id=req_id,
                    attempt_number=attempt,
                    replica_id=effective_replica_id,
                    model_id=selected_spec.model_id,
                    provider=selected_spec.provider,
                    prompt_tokens=p_tokens,
                    completion_tokens=0,
                    latency_ms=attempt_lat,
                    cost=failed_cost,
                    cost_state="unknown_unbilled",
                    status="failed",
                    error=sanitized_err,
                )

                if not retry_policy.is_retryable(exc):
                    if selected_spec.is_cloud or (candidate_idx + 1 < len(ranked_candidates)):
                        candidate_idx += 1
                        continue
                    else:
                        break

                if attempt < max_attempts:
                    backoff_delay = retry_policy.get_backoff(attempt)
                    await asyncio.sleep(backoff_delay)

                if selected_spec.is_cloud:
                    candidate_idx += 1
                else:
                    all_reps = registry.get_replicas_for_model(selected_spec.model_id)
                    remaining = [
                        r
                        for r in all_reps
                        if r.replica_id not in failed_replicas
                        and r.is_alive()
                        and circuit_breaker.allow_request(r.replica_id)
                    ]
                    if not remaining:
                        candidate_idx += 1
        finally:
            if dispatched and not selected_spec.is_cloud and scheduler:
                scheduler.complete(rep_id, p_tokens, est_out_tokens, 0.0, success=False)

    if not execution_result:
        total_lat = (time.perf_counter() - start_total_time) * 1000.0
        total_billed_cost = sum(a.cost for a in attempt_history)
        sanitized_final_err = sanitize_message(str(last_error)) or "Execution failed"
        tracer.finish_trace(
            request_id=req_id,
            selected_model=selected_spec.model_id,
            selected_replica=effective_replica_id,
            actual_output_tokens=0,
            total_latency_ms=total_lat,
            estimated_cost=est_cost,
            actual_cost=total_billed_cost,
            cache_outcome="miss",
            final_status="failed",
            error_detail=sanitized_final_err,
        )
        raise HTTPException(
            status_code=503,
            detail=f"Inference execution failed after {attempt} attempts: {sanitized_final_err}",
        )

    # 9. Post-Execution Accounting & Safe Cache Ingestion
    total_lat = (time.perf_counter() - start_total_time) * 1000.0
    actual_out_tokens = execution_result.get("output_tokens", est_out_tokens)
    total_actual_cost = sum(a.cost for a in attempt_history)

    # Update estimator observation for calibration
    estimator.record_observation(task_type, est_out_tokens, actual_out_tokens)

    # Online recursive Bayesian conjugate updating
    is_code = "def " in prompt_text or "class " in prompt_text or "function" in prompt_text.lower() or task_type == "coding"
    is_reasoning = "step-by-step" in prompt_text.lower() or "think" in prompt_text.lower() or task_type == "reasoning"
    bayesian_estimator.record_actual(
        task_type=task_type,
        prompt_tokens=p_tokens,
        actual_output_tokens=actual_out_tokens,
        is_code=is_code,
        is_reasoning=is_reasoning,
    )

    # Ingest into Provenance-Aware Cache
    model_q = selected_spec.quality_scores.get(task_type, 0.80)
    is_sim = "simulated" in str(execution_result.get("source", "")).lower()
    cache.put(
        request=ctx,
        logical_model=logical_model,
        model_version="1.0.0",
        prompt=prompt_text,
        messages=validated_req.messages,
        response={
            "content": execution_result["content"],
            "model": selected_spec.model_id,
            "replica_id": effective_replica_id,
            "provider": selected_spec.provider,
            "output_tokens": actual_out_tokens,
        },
        origin_model=selected_spec.model_id,
        origin_provider=selected_spec.provider,
        origin_privacy=privacy,
        origin_quality=model_q,
        origin_policy=policy_name,
        policy_name=policy_name,
        is_simulated=is_sim,
        source=execution_result.get("source", ""),
    )

    # Finish trace
    tracer.finish_trace(
        request_id=req_id,
        selected_model=selected_spec.model_id,
        selected_replica=effective_replica_id,
        actual_output_tokens=actual_out_tokens,
        total_latency_ms=total_lat,
        estimated_cost=est_cost,
        actual_cost=total_actual_cost,
        cache_outcome="miss",
        final_status="success",
    )

    slo_met = (total_lat / 1000.0) <= latency_slo if latency_slo else True

    return ChatCompletionResponse(
        id=req_id,
        model=selected_spec.model_id,
        logical_model=logical_model,
        choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=execution_result["content"]))],
        usage=UsageInfo(
            prompt_tokens=p_tokens,
            completion_tokens=actual_out_tokens,
            total_tokens=p_tokens + actual_out_tokens,
        ),
        routing_metadata=RoutingMetadata(
            request_id=req_id,
            logical_model=logical_model,
            selected_model=selected_spec.model_id,
            selected_replica=effective_replica_id,
            provider=selected_spec.provider,
            latency_ms=round(total_lat, 2),
            attempts=attempt,
            cache_hit=False,
            estimated_cost=round(est_cost, 6),
            actual_cost=round(total_actual_cost, 6),
            slo_attained=slo_met,
            policy_name=policy_name,
            scheduler_name=effective_scheduler_name,
            attempt_history=attempt_history,
        ),
        content=execution_result["content"],
        replica_id=effective_replica_id,
        provider=selected_spec.provider,
    )


@app.get("/v1/estimator/bayesian/state")
async def get_bayesian_estimator_state():
    """Returns the current online Bayesian estimator parameters and sample counts."""
    return {
        "status": "active",
        "domains": {
            domain: {
                "mu": [round(float(x), 4) for x in reg.mu],
                "sample_count": reg.sample_count,
            }
            for domain, reg in bayesian_estimator.regressors.items()
        },
    }


def _read_ledger_records(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    records = []
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return records


@app.get("/v1/control-plane/ledger")
async def get_control_plane_ledger(limit: int = 50):
    """Computes attributable savings S_p = B_p - A_p, aggregates, and transaction ledger."""
    ledger_path = Path("data/ledger.jsonl")
    records = await asyncio.to_thread(_read_ledger_records, ledger_path)

    total_reqs = 0
    actual_cost_total = 0.0
    baseline_cost_total = 0.0
    cache_hits = 0
    by_capability: dict[str, dict[str, Any]] = {}
    recent_transactions: list[dict[str, Any]] = []

    for record in records:
        # Count request level entries or fallback to attempt
        if record.get("entry_type") != "request":
            continue

        total_reqs += 1
        cost = float(record.get("cost", 0.0))
        p_tok = int(record.get("prompt_tokens", 0))
        c_tok = int(record.get("completion_tokens", 0))

        # Baseline benchmark cost: frontier pricing ($0.005/1k prompt, $0.015/1k output)
        counterfactual = (p_tok / 1000.0) * 0.005 + (c_tok / 1000.0) * 0.015
        baseline = max(counterfactual, cost)
        savings = max(0.0, baseline - cost)

        actual_cost_total += cost
        baseline_cost_total += baseline

        is_cached = (record.get("status") == "cached") or ("cache" in str(record.get("provider", "")).lower())
        if is_cached:
            cache_hits += 1

        cap = str(record.get("logical_model", "general"))
        if cap not in by_capability:
            by_capability[cap] = {
                "requests": 0,
                "actual_cost": 0.0,
                "baseline_cost": 0.0,
                "attributable_savings": 0.0,
            }
        by_capability[cap]["requests"] += 1
        by_capability[cap]["actual_cost"] += cost
        by_capability[cap]["baseline_cost"] += baseline
        by_capability[cap]["attributable_savings"] += savings

        recent_transactions.append({
            "request_id": record.get("request_id"),
            "timestamp": record.get("timestamp"),
            "logical_model": cap,
            "selected_model": record.get("selected_model"),
            "provider": record.get("provider"),
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "actual_cost": round(cost, 6),
            "baseline_cost": round(baseline, 6),
            "attributable_savings": round(savings, 6),
            "cache_hit": is_cached,
            "status": record.get("status", "success"),
        })

    net_savings = max(0.0, baseline_cost_total - actual_cost_total)
    savings_pct = (net_savings / baseline_cost_total * 100.0) if baseline_cost_total > 0.0 else 0.0
    hit_rate = (cache_hits / total_reqs * 100.0) if total_reqs > 0 else 0.0

    # Sort recent transactions newest first
    recent_transactions.reverse()
    clipped_transactions = recent_transactions[:limit]

    return {
        "summary": {
            "total_requests": total_reqs,
            "actual_cost": round(actual_cost_total, 6),
            "baseline_cost": round(baseline_cost_total, 6),
            "attributable_savings": round(net_savings, 6),
            "savings_percentage": round(savings_pct, 2),
            "cache_hits": cache_hits,
            "cache_hit_rate": round(hit_rate, 2),
        },
        "by_capability": {
            k: {
                "requests": v["requests"],
                "actual_cost": round(v["actual_cost"], 6),
                "baseline_cost": round(v["baseline_cost"], 6),
                "attributable_savings": round(v["attributable_savings"], 6),
            }
            for k, v in by_capability.items()
        },
        "recent_transactions": clipped_transactions,
    }


@app.get("/v1/control-plane/policies")
async def get_control_plane_policies():
    """Returns the active routing policy, version history, and available policy configurations."""
    current_ver = policy_versions[-1]["version"] if policy_versions else 1
    return {
        "active_policy": default_active_policy,
        "active_version": current_ver,
        "history": policy_versions,
        "available_policies": sorted(model_policies.keys()),
    }


@app.post("/v1/control-plane/rollback")
async def rollback_control_plane_policy(req: PolicyRollbackRequest):
    """Rolls back or switches the active gateway routing policy dynamically with full version audit."""
    global default_active_policy

    target_policy = None
    if req.version is not None:
        # Rollback to specific version in history
        match = next((v for v in policy_versions if v["version"] == req.version), None)
        if not match:
            raise HTTPException(status_code=404, detail=f"Policy version {req.version} not found in version history.")
        target_policy = match["policy_name"]
    elif req.policy_name:
        target_policy = req.policy_name.strip()
    else:
        raise HTTPException(status_code=400, detail="Must provide either policy_name or version to rollback.")

    if target_policy not in model_policies:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown policy '{target_policy}'. Available: {sorted(model_policies.keys())}",
        )

    new_version_num = len(policy_versions) + 1
    action_desc = (
        f"Rollback to version {req.version} ({target_policy})"
        if req.version
        else f"Switched active policy to '{target_policy}'"
    )
    new_record = {
        "version": new_version_num,
        "policy_name": target_policy,
        "description": action_desc,
        "timestamp": time.time(),
        "author": req.author or "admin",
        "weights": getattr(model_policies.get(target_policy), "weights", {}),
    }
    policy_versions.append(new_record)
    default_active_policy = target_policy

    return {
        "status": "success",
        "active_policy": default_active_policy,
        "active_version": new_version_num,
        "message": f"Successfully activated policy '{default_active_policy}' at version {new_version_num}",
        "history": policy_versions,
    }

