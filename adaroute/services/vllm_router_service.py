"""AdaRoute Bundled Heuristic Router Service (vLLM Semantic Router Preview Protocol).

Exposes OpenAI-compatible preview routing endpoint compliant with vllm-sr v0.4.0 preview protocol:
- POST /api/v1/routing/preview: Evaluates messages and classifies into model tier using bundled heuristic
- GET /health: Health check identifying as bundled heuristic
- GET /api/v1/models: Virtual route catalog
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from adaroute.selection.vllm_router import classify_semantic_intent

logger = logging.getLogger("adaroute.vllm_router_service")

router_app = FastAPI(title="AdaRoute Bundled Heuristic Router Daemon", version="0.4.0")


class PreviewRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    preview_context: dict[str, Any] | None = None


@router_app.post("/api/v1/routing/preview")
async def routing_preview(req: PreviewRequest):
    prompt_text = ""
    for m in reversed(req.messages):
        role = str(m.get("role", "user")).lower()
        if role == "user":
            prompt_text = str(m.get("content", ""))
            break
    if not prompt_text and req.messages:
        prompt_text = str(req.messages[-1].get("content", ""))

    token_est = len(prompt_text.split())
    target_model, intent, conf, reason = classify_semantic_intent(prompt_text, token_est)

    return {
        "selected_model": target_model,
        "selection_status": "ok",
        "selection_reason": reason,
        "selection_method": "bundled_heuristic_classifier",
        "router_type": "bundled_heuristic",
        "service": "adaroute-bundled-heuristic",
        "decision_result": {
            "algorithm": "multi_factor",
            "confidence": conf,
            "intent": intent,
            "reason": reason,
        },
    }


@router_app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "adaroute-bundled-heuristic-router",
        "router_type": "bundled_heuristic",
        "version": "0.4.0",
        "timestamp": time.time(),
    }


@router_app.get("/api/v1/models")
@router_app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "vllm-sr/auto", "object": "model", "owned_by": "vllm-sr"},
            {"id": "vllm-sr/fast", "object": "model", "owned_by": "vllm-sr"},
            {"id": "vllm-sr/coding", "object": "model", "owned_by": "vllm-sr"},
            {"id": "vllm-sr/reasoning", "object": "model", "owned_by": "vllm-sr"},
        ],
    }
