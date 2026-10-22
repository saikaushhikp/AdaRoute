#!/usr/bin/env python3
"""Install dependencies, cache Qwen AWQ, and serve it through vLLM.

The server exposes vLLM's OpenAI-compatible API. The model is downloaded to
the Hugging Face cache on first run, so later runs do not download it again.

Authentication:
    export HF_TOKEN=hf_...
    python vLLM_testing/serve_qwen_vllm.py

Alternatively, put ``HF_TOKEN=hf_...`` in the repository's ignored ``.env``
file. The token is never printed or included in the vLLM subprocess command.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
CPU_FALLBACK_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def load_hf_token() -> str | None:
    """Load HF_TOKEN from the environment, optionally using the local .env."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    repository_env = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=repository_env)
    return os.environ.get("HF_TOKEN") or None


def detect_device() -> str:
    """Return the vLLM device backend available on this machine."""
    requested = os.environ.get("VLLM_DEVICE")
    if requested:
        return requested
    if shutil.which("nvidia-smi"):
        return "cuda"
    if shutil.which("rocminfo") or shutil.which("rocm-smi"):
        return "rocm"
    return "cpu"


def install_vllm(device: str) -> None:
    """Install vLLM using the documented package index for the detected device."""
    if sys.version_info < (3, 11):
        raise RuntimeError("vLLM requires Python 3.11 or newer.")

    command = [sys.executable, "-m", "pip", "install", "vllm"]
    environment = os.environ.copy()
    if device == "cpu":
        # The CPU wheels use PyTorch's CPU index and the vLLM CPU build target.
        command.extend(["--extra-index-url", "https://download.pytorch.org/whl/cpu"])
        environment["VLLM_TARGET_DEVICE"] = "cpu"
    elif device == "rocm":
        environment["VLLM_TARGET_DEVICE"] = "rocm"

    print(f"vLLM is not installed; installing for {device}...", flush=True)
    subprocess.run(command, check=True, env=environment)


def ensure_vllm(device: str) -> None:
    if importlib.util.find_spec("vllm") is None:
        install_vllm(device)


def has_cpu_native_backend() -> bool:
    """Return whether the installed vLLM package registered its CPU ops."""
    try:
        import torch
        from vllm import _custom_ops  # noqa: F401

        return hasattr(torch.ops._C, "init_cpu_memory_env")
    except (ImportError, OSError, RuntimeError):
        return False


def has_cpu_awq_backend() -> bool:
    """Return whether vLLM exposes the CPU kernel required by AWQ."""
    try:
        import torch
        from vllm import _custom_ops  # noqa: F401

        return hasattr(torch.ops._C, "cpu_gemm_wna16")
    except (ImportError, OSError, RuntimeError):
        return False


def ensure_model(model_id: str, hf_token: str | None) -> str:
    """Download the model if needed and return its local cache path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "vLLM did not install huggingface_hub. Reinstall vLLM and try again."
        ) from error

    if hf_token:
        print("Using HF_TOKEN for authenticated Hugging Face download.", flush=True)
    else:
        print(
            "HF_TOKEN is not set; Hugging Face may rate-limit this download. "
            "Set HF_TOKEN as an environment variable or in .env.",
            flush=True,
        )
    print(f"Checking Hugging Face cache for {model_id}...", flush=True)
    return snapshot_download(repo_id=model_id, token=hf_token)


def build_command(
    args: argparse.Namespace, model_path: str, quantization: str | None
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
    ]
    if quantization:
        command.extend(["--quantization", quantization])
    if args.gpu_memory_utilization is not None:
        command.extend(
            ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        )
    if args.enforce_eager:
        command.append("--enforce-eager")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--served-model-name", default=MODEL_ID)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="Hugging Face model ID or a local model directory.",
    )
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="Fail instead of using the unquantized Qwen model on unsupported CPUs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = detect_device()
    ensure_vllm(device)
    if device == "cpu" and not has_cpu_native_backend():
        raise RuntimeError(
            "No CUDA or ROCm accelerator was detected. "
            f"{MODEL_ID} requires vLLM's native CPU wheel for a CPU run, "
            "but that wheel is not installed. Reinstall vLLM using the "
            "official CPU wheel before retrying."
        )
    hf_token = load_hf_token()

    model_path = args.model
    quantization = "awq" if model_path == MODEL_ID else None
    if (
        device == "cpu"
        and quantization == "awq"
        and not has_cpu_awq_backend()
    ):
        if args.no_cpu_fallback:
            raise RuntimeError(
                "This CPU has no vLLM cpu_gemm_wna16 AWQ kernel. "
                "Use a CPU with the required ISA, a CUDA/ROCm host, or omit "
                "--no-cpu-fallback to serve the unquantized Qwen model."
            )
        print(
            f"CPU AWQ is unavailable; falling back to {CPU_FALLBACK_MODEL_ID}. "
            "Use --no-cpu-fallback to require AWQ.",
            flush=True,
        )
        model_path = CPU_FALLBACK_MODEL_ID
        quantization = None
    if "/" in model_path and not Path(model_path).exists():
        model_path = ensure_model(model_path, hf_token)

    if device == "cpu":
        print(
            "Detected CPU-only execution. AWQ support and usable performance "
            "depend on the installed vLLM CPU backend; a clear backend error "
            "will be reported if this quantized model is unsupported.",
            flush=True,
        )

    command = build_command(args, model_path, quantization)
    if device == "cpu" and args.gpu_memory_utilization is None:
        command.extend(["--gpu-memory-utilization", "0.25"])
    print("Starting vLLM OpenAI-compatible server:", " ".join(command), flush=True)
    server_environment = os.environ.copy()
    if device in {"cpu", "rocm"}:
        # vLLM initializes its platform before parsing --device. Set the
        # backend explicitly so CPU-only and ROCm hosts do not remain "auto".
        server_environment["VLLM_TARGET_DEVICE"] = device
    subprocess.run(command, check=True, env=server_environment)


if __name__ == "__main__":
    main()
