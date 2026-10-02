from pydantic import BaseModel


class RequestContext(BaseModel):
    task_type: str
    prompt_tokens: int
    estimated_output_tokens: int
    quality_requirement: str
    latency_slo: float
    privacy_level: str
    budget: float
    tenant_id: str


class BackendState(BaseModel):
    provider: str
    model: str
    replica_id: str
    queue_depth: int
    estimated_workload: float
    latency: float
    memory_pressure: float
    failure_rate: float
    availability: bool
