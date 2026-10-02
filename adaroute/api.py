from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from adaroute.experiments.baseline_routing import DeterministicRouter
from adaroute.experiments.mock_provider import MockProvider
from adaroute.schemas import BackendState, RequestContext

app = FastAPI(title="AdaRoute Unified API")

# Initialize mock backend states (registry)
mock_backends_registry = [
    BackendState(provider="mock-local", model="llama-3-8b", replica_id="replica-1", queue_depth=0, estimated_workload=0.0, latency=0.05, memory_pressure=0.2, failure_rate=0.0, availability=True),
    BackendState(provider="mock-external", model="gpt-4o", replica_id="replica-2", queue_depth=5, estimated_workload=10.0, latency=0.5, memory_pressure=0.5, failure_rate=0.0, availability=True),
]

router = DeterministicRouter(mock_backends_registry)

# Initialize physical mock providers
providers = {
    "replica-1": MockProvider(replica_id="replica-1", base_latency_ms=50),
    "replica-2": MockProvider(replica_id="replica-2", base_latency_ms=500),
}

class UnifiedRequest(BaseModel):
    prompt: str
    task_type: str = "general"
    privacy_level: str = "public"
    budget: float = 1.0

class UnifiedResponse(BaseModel):
    content: str
    replica_id: str
    provider: str
    model: str
    tokens_used: int

@app.post("/v1/chat/completions", response_model=UnifiedResponse)
async def chat_completions(req: UnifiedRequest):
    # 1. Request Characterization (Simple heuristics for M1)
    prompt_tokens = len(req.prompt.split())
    
    ctx = RequestContext(
        task_type=req.task_type,
        prompt_tokens=prompt_tokens,
        estimated_output_tokens=50,  # Baseline estimation
        quality_requirement="medium",
        latency_slo=2.0,
        privacy_level=req.privacy_level,
        budget=req.budget,
        tenant_id="default-tenant"
    )

    # 2. Routing Decision
    selected_backend = router.route(ctx)
    if not selected_backend:
        raise HTTPException(status_code=503, detail="No suitable backend found for constraints")

    # 3. Provider Execution (Unified Mock Provider)
    provider = providers.get(selected_backend.replica_id)
    if not provider:
        raise HTTPException(status_code=500, detail="Backend implementation not found")
        
    result = await provider.execute(ctx)
    
    return UnifiedResponse(
        content=result["content"],
        replica_id=selected_backend.replica_id,
        provider=selected_backend.provider,
        model=selected_backend.model,
        tokens_used=result["output_tokens"]
    )
