"""Unit tests for system-agnostic multi-model vLLM launcher and router."""

from pathlib import Path

from scripts.vllm_multi_server import (
    build_lookup_table,
    build_models_response,
    infer_dtype,
    infer_tokenizer,
    is_gguf_model,
    load_local_models_from_catalog,
    resolve_target_model,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_infer_tokenizer():
    """Verify tokenizer inference handles GGUF quantization tags and standard repos."""
    assert infer_tokenizer("unsloth/Qwen3-0.6B-GGUF:Q4_K_M") == "Qwen/Qwen3-0.6B"
    assert infer_tokenizer("unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M") == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert infer_tokenizer("unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M") == "unsloth/Llama-3.2-1B-Instruct"
    assert infer_tokenizer("unsloth/gemma-3-1b-it-GGUF:Q4_K_M") == "unsloth/gemma-3-1b-it"
    assert infer_tokenizer("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    assert infer_tokenizer("custom/model:Q8_0", configured_tokenizer="explicit/tok") == "explicit/tok"


def test_infer_dtype():
    """Verify dtype inference respects Gemma bfloat16 requirements and explicit overrides."""
    # Gemma models must use bfloat16 due to numerical instability in half
    assert infer_dtype("unsloth/gemma-3-1b-it-GGUF:Q4_K_M", is_gguf=True) == "bfloat16"
    assert infer_dtype("google/gemma-2-2b-it", is_gguf=False) == "bfloat16"

    # GGUF models default to half
    assert infer_dtype("unsloth/Qwen3-0.6B-GGUF:Q4_K_M", is_gguf=True) == "half"
    assert infer_dtype("unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", is_gguf=True) == "half"

    # Full precision models default to bfloat16
    assert infer_dtype("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", is_gguf=False) == "bfloat16"

    # Explicit configuration takes precedence
    assert infer_dtype("any/model", configured_dtype="float32", is_gguf=True) == "float32"
    assert infer_dtype("any/model", configured_dtype="bfloat16", is_gguf=True) == "bfloat16"


def test_is_gguf_model():
    """Verify detection of GGUF models vs full-precision models."""
    assert is_gguf_model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M") is True
    assert is_gguf_model("Qwen/Qwen2.5-0.5B-Instruct") is False
    assert is_gguf_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") is False


def test_load_models_from_config_gpu():
    """Verify parsing of config.gpu.yaml discovers both local GGUF models."""
    config_path = REPO_ROOT / "config.gpu.yaml"
    models = load_local_models_from_catalog(config_path)

    assert len(models) == 2
    model_ids = [m.model_id for m in models]
    assert "unsloth/Qwen3-0.6B-GGUF:Q4_K_M" in model_ids
    assert "unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M" in model_ids

    # Port allocation
    assert models[0].port == 7001
    assert models[1].port == 7002

    # GGUF detection
    assert models[0].is_gguf is True
    assert models[0].dtype == "half"
    assert models[1].is_gguf is True
    assert models[1].dtype == "half"


def test_load_models_from_config_cpu():
    """Verify parsing of config.yaml discovers CPU/standard local models."""
    config_path = REPO_ROOT / "config.yaml"
    models = load_local_models_from_catalog(config_path)

    assert len(models) == 2
    model_ids = [m.model_id for m in models]
    assert "Qwen/Qwen2.5-0.5B-Instruct" in model_ids
    assert "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" in model_ids

    # Full precision detection
    assert models[0].is_gguf is False
    assert models[0].dtype == "bfloat16"


def test_dynamic_routing_resolution():
    """Verify request resolution matches by exact ID, aliases, and keywords."""
    config_path = REPO_ROOT / "config.gpu.yaml"
    models = load_local_models_from_catalog(config_path)
    lookup = build_lookup_table(models)

    # Exact model_id match
    assert resolve_target_model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M", models, lookup).port == 7001
    assert resolve_target_model("unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M", models, lookup).port == 7002

    # Alias matching
    assert resolve_target_model("reasoning", models, lookup).port == 7001
    assert resolve_target_model("coding", models, lookup).port == 7002
    assert resolve_target_model("complex", models, lookup).port == 7002

    # Family keyword matching
    assert resolve_target_model("coder", models, lookup).port == 7002
    assert resolve_target_model("qwen3", models, lookup).port == 7001

    # Fallback to default
    assert resolve_target_model(None, models, lookup).port == 7001
    assert resolve_target_model("unknown-model", models, lookup).port == 7001


def test_openai_models_response_generation():
    """Verify /v1/models response structure contains all local models."""
    config_path = REPO_ROOT / "config.gpu.yaml"
    models = load_local_models_from_catalog(config_path)
    resp = build_models_response(models)

    assert resp["object"] == "list"
    ids = [item["id"] for item in resp["data"]]
    assert "unsloth/Qwen3-0.6B-GGUF:Q4_K_M" in ids
    assert "unsloth/Qwen2.5-Coder-0.5B-Instruct-128K-GGUF:Q4_K_M" in ids
    assert "coding" in ids
    assert "reasoning" in ids
