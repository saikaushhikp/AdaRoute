"""Unified AI Gateway API Contract (M1 / M2 / M3 / M4 / M5 / M6 / M7).

Exposes standardized /v1/chat/completions endpoint that coordinates:
- Pre-routing characterization (tokenization & estimation)
- Safe Exact Caching with tenant isolation
- Stage 1: Model selection (Strongest / Cheapest / Static / Adaptive)
- Stage 2: Replica scheduling (Workload-weighted queue W_q & SLO awareness)
- Bounded retries and privacy-preserving fallback
- Full request/attempt correlation tracing
"""

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from adaroute.backends.local import LocalBackend
from adaroute.backends.mock import MockBackend
from adaroute.cache.exact import ExactCache
from adaroute.policy.model_selection import (
    AdaptiveModelPolicy,
    CheapestModelPolicy,
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
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    PrivacyLevel,
    RequestContext,
    RoutingMetadata,
    UsageInfo,
)
from adaroute.telemetry.tracer import default_tracer
from adaroute.workloads.estimator import OutputLengthEstimator
from adaroute.workloads.tokenizer import count_tokens

app = FastAPI(title="AdaRoute Unified AI Gateway", version="2.0.0")

# Core Engine Components
registry = default_registry
estimator = OutputLengthEstimator()
cache = ExactCache()
circuit_breaker = CircuitBreaker()
retry_policy = RetryPolicy(max_attempts=2)
tracer = default_tracer

# Stage 1 Policies
model_policies = {
    "adaptive": AdaptiveModelPolicy(registry),
    "strongest": StrongestModelPolicy(registry),
    "cheapest": CheapestModelPolicy(registry),
    "static": StaticRulePolicy(registry),
}

# Stage 2 Schedulers
replica_schedulers = {
    "slo_aware": SLOAwareScheduler(registry),
    "least_loaded": LeastLoadedScheduler(registry),
    "round_robin": RoundRobinScheduler(registry),
}

# Backend Runners
backend_runners = {
    "local": LocalBackend(),
    "mock-local": LocalBackend(),
    "mock-external": MockBackend(provider_name="mock-external", base_latency_ms=30.0),
}


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models": [m.model_id for m in registry.get_all_models()],
        "replicas": [r.replica_id for r in registry.get_all_replicas() if r.is_alive()],
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(raw_req: dict[str, Any]):
    start_total_time = time.perf_counter()
    req_id = str(uuid.uuid4())

    # 1. Parse & Validate Request (Handles standard & simplified inputs)
    try:
        # Extract prompt text
        prompt_text = raw_req.get("prompt")
        messages = raw_req.get("messages", [])
        if not prompt_text and messages:
            prompt_text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
        elif not prompt_text:
            prompt_text = "Hello"

        # Canonicalize privacy level (Fail closed!)
        raw_privacy = raw_req.get("privacy_level", "public")
        privacy = PrivacyLevel.from_str(str(raw_privacy))

        budget = float(raw_req.get("budget", 1.0))
        if budget < 0:
            raise ValueError("Budget must be non-negative")

        logical_model = raw_req.get("model", "general")
        task_type = raw_req.get("task_type", logical_model)
        quality_req = raw_req.get("quality_requirement", "medium")
        latency_slo = float(raw_req["latency_slo"]) if "latency_slo" in raw_req and raw_req["latency_slo"] is not None else 2.0
        tenant_id = str(raw_req.get("tenant_id", "default"))
        user_id = str(raw_req.get("user_id", "anonymous"))
        no_cache = bool(raw_req.get("no_cache", False))
        max_tokens = int(raw_req["max_tokens"]) if "max_tokens" in raw_req and raw_req["max_tokens"] is not None else None

        policy_name = raw_req.get("policy", "adaptive")
        scheduler_name = raw_req.get("scheduler", "slo_aware")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid request format: {e!s}")

    # 2. Pre-Routing Request Characterization
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
    )

    # 3. Observability Start Trace
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

    # 4. Safe Exact Caching Check (Authorization & Privacy evaluated first)
    cached_res = cache.get(
        request=ctx,
        logical_model=logical_model,
        model_version="1.0.0",
        prompt=prompt_text,
    )
    if cached_res:
        total_lat = (time.perf_counter() - start_total_time) * 1000.0
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
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=cached_res["content"])
                )
            ],
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
                attempts=1,
                cache_hit=True,
                estimated_cost=0.0,
                actual_cost=0.0,
                slo_attained=True,
                policy_name=policy_name,
            ),
            content=cached_res["content"],
            replica_id=cached_res["replica_id"],
            provider=cached_res.get("provider", "cache"),
        )

    # 5. Stage 1: Model Selection
    model_policy = model_policies.get(policy_name, model_policies["adaptive"])
    selected_spec, est_cost, err_msg = model_policy.select_model(ctx)

    if not selected_spec:
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
        )
        raise HTTPException(status_code=503, detail=f"Routing rejected: {err_msg}")

    # 6. Stage 2: Replica Selection
    scheduler = replica_schedulers.get(scheduler_name, replica_schedulers["slo_aware"])
    selected_replica = scheduler.select_replica(selected_spec, ctx)

    if not selected_replica:
        raise HTTPException(status_code=503, detail="No healthy replicas available for selected model")

    # 7. Execution with Dynamic Queue Tracking & Bounded Retry
    attempt = 0
    max_attempts = retry_policy.max_attempts
    last_error = None
    execution_result = None

    while attempt < max_attempts:
        attempt += 1
        rep_id = selected_replica.replica_id

        # Check circuit breaker
        if not circuit_breaker.allow_request(rep_id):
            # Try alternate replica if available
            alt_replicas = [
                r for r in registry.get_replicas_for_model(selected_spec.model_id)
                if r.replica_id != rep_id and r.is_alive()
            ]
            if alt_replicas:
                selected_replica = alt_replicas[0]
                rep_id = selected_replica.replica_id
            else:
                break

        # Dynamic state update: W_q increases
        scheduler.dispatch(rep_id, p_tokens, est_out_tokens)
        attempt_start = time.perf_counter()

        try:
            runner = backend_runners.get(selected_spec.provider, backend_runners["mock-external"])
            res = await runner.execute(
                request=ctx,
                model_spec=selected_spec,
                replica_id=rep_id,
                prompt_text=prompt_text,
            )
            attempt_lat = (time.perf_counter() - attempt_start) * 1000.0

            # Dynamic state update: W_q decreases & success recorded
            scheduler.complete(rep_id, p_tokens, est_out_tokens, attempt_lat, success=True)
            circuit_breaker.record_success(rep_id)

            tracer.record_attempt(
                request_id=req_id,
                attempt_number=attempt,
                replica_id=rep_id,
                model_id=selected_spec.model_id,
                provider=selected_spec.provider,
                latency_ms=attempt_lat,
                status="success",
            )
            execution_result = res
            break

        except Exception as exc:
            attempt_lat = (time.perf_counter() - attempt_start) * 1000.0
            scheduler.complete(rep_id, p_tokens, est_out_tokens, attempt_lat, success=False)
            circuit_breaker.record_failure(rep_id)
            last_error = exc

            tracer.record_attempt(
                request_id=req_id,
                attempt_number=attempt,
                replica_id=rep_id,
                model_id=selected_spec.model_id,
                provider=selected_spec.provider,
                latency_ms=attempt_lat,
                status="failed",
                error=str(exc),
            )

            if not retry_policy.is_retryable(exc):
                break

            # Switch to alternate replica for retry if possible
            alt_replicas = [
                r for r in registry.get_replicas_for_model(selected_spec.model_id)
                if r.replica_id != rep_id and r.is_alive()
            ]
            if alt_replicas:
                selected_replica = alt_replicas[0]

    if not execution_result:
        total_lat = (time.perf_counter() - start_total_time) * 1000.0
        tracer.finish_trace(
            request_id=req_id,
            selected_model=selected_spec.model_id,
            selected_replica=selected_replica.replica_id,
            actual_output_tokens=0,
            total_latency_ms=total_lat,
            estimated_cost=est_cost,
            actual_cost=0.0,
            cache_outcome="miss",
            final_status="failed",
        )
        raise HTTPException(
            status_code=503,
            detail=f"Inference execution failed after {attempt} attempts: {last_error!s}",
        )

    # 8. Post-Execution Accounting & Cache Write
    total_lat = (time.perf_counter() - start_total_time) * 1000.0
    actual_out_tokens = execution_result["output_tokens"]
    actual_cost = (
        (p_tokens / 1000.0) * selected_spec.input_price_per_1k
        + (actual_out_tokens / 1000.0) * selected_spec.output_price_per_1k
    )

    # Record observation in estimator for continuous calibration
    estimator.record_observation(task_type, est_out_tokens, actual_out_tokens)

    # Store in Safe ExactCache
    cache.put(
        request=ctx,
        logical_model=logical_model,
        model_version="1.0.0",
        prompt=prompt_text,
        response={
            "content": execution_result["content"],
            "model": selected_spec.model_id,
            "replica_id": selected_replica.replica_id,
            "provider": selected_spec.provider,
            "output_tokens": actual_out_tokens,
        },
    )

    # Observability Finish Trace
    tracer.finish_trace(
        request_id=req_id,
        selected_model=selected_spec.model_id,
        selected_replica=selected_replica.replica_id,
        actual_output_tokens=actual_out_tokens,
        total_latency_ms=total_lat,
        estimated_cost=est_cost,
        actual_cost=actual_cost,
        cache_outcome="miss",
        final_status="success",
    )

    slo_met = (total_lat / 1000.0) <= latency_slo if latency_slo else True

    return ChatCompletionResponse(
        id=req_id,
        model=selected_spec.model_id,
        logical_model=logical_model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=execution_result["content"])
            )
        ],
        usage=UsageInfo(
            prompt_tokens=p_tokens,
            completion_tokens=actual_out_tokens,
            total_tokens=p_tokens + actual_out_tokens,
        ),
        routing_metadata=RoutingMetadata(
            request_id=req_id,
            logical_model=logical_model,
            selected_model=selected_spec.model_id,
            selected_replica=selected_replica.replica_id,
            provider=selected_spec.provider,
            latency_ms=round(total_lat, 2),
            attempts=attempt,
            cache_hit=False,
            estimated_cost=round(est_cost, 6),
            actual_cost=round(actual_cost, 6),
            slo_attained=slo_met,
            policy_name=policy_name,
        ),
        content=execution_result["content"],
        replica_id=selected_replica.replica_id,
        provider=selected_spec.provider,
    )
