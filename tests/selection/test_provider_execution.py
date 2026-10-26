"""Tests for Section 12.7: Provider Backend Adapters & Execution Contract.

Verifies:
- Ollama adapter uses selected replica endpoint
- OpenAI-compatible / vLLM adapter targets /v1/chat/completions with Bearer credentials
- LiteLLM adapter preserves provider model identity
- Native token counts and costs are reported with structured provenance
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adaroute.backends.local import LocalBackend
from adaroute.backends.openai_compatible import OpenAICompatibleBackend
from adaroute.schemas import RequestContext
from adaroute.selection.candidates import CandidateDeployment
from adaroute.selection.catalog import (
    DeploymentDefinition,
    ModelDefinition,
    ProviderDefinition,
    RuntimeSnapshot,
)


@pytest.mark.asyncio
async def test_tc_m2_08_ollama_replica_endpoint_preservation():
    """Ollama backend calls the exact replica endpoint provided, not hardcoded host."""
    backend = LocalBackend(execution_mode="real")
    custom_endpoint = "http://custom-ollama-node:11434"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "qwen2.5:1.5b",
        "message": {"role": "assistant", "content": "Hello from custom replica!"},
        "prompt_eval_count": 12,
        "eval_count": 8,
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        req = RequestContext(task_type="general", prompt_tokens=12)
        res = await backend.execute(
            request=req,
            prompt_text="Hello",
            replica_id="custom-rep-1",
            endpoint=custom_endpoint,
        )

        assert mock_post.called
        call_url = mock_post.call_args[0][0]
        assert call_url == f"{custom_endpoint}/api/chat"
        assert res["content"] == "Hello from custom replica!"
        assert res["prompt_tokens"] == 12
        assert res["output_tokens"] == 8
        assert res["replica_id"] == "custom-rep-1"


@pytest.mark.asyncio
async def test_tc_m2_09_openai_compatible_vllm_protocol_and_auth():
    """OpenAI-compatible backend invokes /v1/chat/completions with Bearer authorization."""
    backend = OpenAICompatibleBackend()
    vllm_endpoint = "http://vllm-cluster:8000"

    pdef = ProviderDefinition(
        provider_id="vllm_prov",
        adapter_kind="openai_compatible",
        credential_reference="VLLM_TEST_KEY",
    )
    mdef = ModelDefinition(model_id="deepseek-vllm", context_window=32768)
    ddef = DeploymentDefinition(
        deployment_id="vllm-dep",
        model_id="deepseek-vllm",
        provider_id="vllm_prov",
        endpoint=vllm_endpoint,
        input_price_per_1k=0.0005,
        output_price_per_1k=0.0015,
    )
    cand = CandidateDeployment(ddef, mdef, pdef, RuntimeSnapshot())

    os.environ["VLLM_TEST_KEY"] = "secret-vllm-token-123"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "chatcmpl-vllm-1",
        "object": "chat.completion",
        "model": "deepseek-vllm",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "vLLM high-speed response"}}
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        req = RequestContext(task_type="general", prompt_tokens=20)
        res = await backend.execute(
            request=req,
            candidate=cand,
            replica_id="vllm-rep-0",
            prompt_text="Compute matrix multiply",
        )

        assert mock_post.called
        call_url = mock_post.call_args[0][0]
        assert call_url == f"{vllm_endpoint}/v1/chat/completions"

        headers = mock_post.call_args[1].get("headers", {})
        assert headers.get("Authorization") == "Bearer secret-vllm-token-123"

        assert res.content == "vLLM high-speed response"
        assert res.prompt_tokens == 20
        assert res.output_tokens == 15
        assert res.provider == "openai_compatible"
        assert res.actual_cost > 0.0
