#!/usr/bin/env python3
"""System-Agnostic Multi-Model vLLM Launcher & Dynamic Router for AdaRoute.

This script enables a single vLLM container to dynamically host, manage, and
route across ANY number of local models defined in an AdaRoute configuration file
(e.g., config.gpu.yaml, config.yaml, or a user-provided custom config).

Features:
  1. Declarative Discovery: Reads the catalog from ADAROUTE_CONFIG or CLI arg,
     dynamically discovering all local models (provider == 'vllm' or 'local').
  2. Automatic Resource Slicing: Distributes GPU VRAM and assigns internal ports
     (7001, 7002, ...) dynamically based on the number of configured local models.
  3. Tokenizer & Quantization Inference: Detects GGUF variants, assigns base
     tokenizers, and chooses appropriate dtypes (half for GGUF, bfloat16 for full).
  4. Dynamic OpenAI-Compatible Routing:
     - Listens on 0.0.0.0:7000.
     - Routes requests by exact model_id, logical aliases, or model name keywords.
     - Automatically normalizes the payload model field to the engine's served name.
     - Provides unified /health and /v1/models endpoints aggregating all backends.
  5. Process Supervision: Monitors sub-engines and cleanly handles termination
     signals (SIGTERM, SIGINT).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Default port for the unified gateway entry point inside the container
GATEWAY_PORT = int(os.getenv("VLLM_GATEWAY_PORT", "7000"))
BASE_INTERNAL_PORT = 7001


@dataclass
class LocalModelConfig:
    """Resolved runtime configuration for a local model engine."""

    model_id: str
    aliases: list[str] = field(default_factory=list)
    tokenizer: str = ""
    port: int = 7001
    dtype: str = "half"
    max_model_len: int = 8192
    gpu_mem_util: float = 0.05
    is_gguf: bool = False
    process: subprocess.Popen | None = None
    last_healthy: bool = False
    last_check_time: float = 0.0
    restart_count: int = 0
    status: str = "starting"  # "starting", "healthy", "crashed", "stopped"


# Global server state
_local_models: list[LocalModelConfig] = []
_model_lookup: dict[str, LocalModelConfig] = {}
_default_model: LocalModelConfig | None = None
_server_shutting_down = False


def find_config_file(explicit_path: str | None = None) -> Path:
    """Find the catalog configuration file from explicit arg, env, or defaults."""
    candidates: list[str] = []
    if explicit_path:
        candidates.append(explicit_path)

    env_config = os.getenv("ADAROUTE_CONFIG")
    if env_config:
        candidates.append(env_config)

    candidates.extend(
        [
            "/app/config.gpu.yaml",
            "config.gpu.yaml",
            "/app/config.yaml",
            "config.yaml",
        ]
    )

    for c in candidates:
        p = Path(c).resolve()
        if p.exists() and p.is_file():
            return p

    raise FileNotFoundError(
        f"Could not find a valid AdaRoute configuration file among candidates: {candidates}. "
        "Set ADAROUTE_CONFIG or pass --config <path>."
    )


def infer_tokenizer(model_id: str, configured_tokenizer: str | None = None) -> str:
    """Infers base model Hugging Face tokenizer from model_id if not configured."""
    if configured_tokenizer:
        return configured_tokenizer

    # Clean quantization tags like :Q4_K_M or :Q8_0
    clean = model_id.split(":")[0]

    # Strip common GGUF suffixes to obtain base model repo
    for suffix in ("-GGUF", "_GGUF", "-gguf", "_gguf"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
            break

    # Strip context length tags like -128K, -64K, -32K, -16K (e.g. unsloth/Qwen2.5-Coder-0.5B-Instruct-128K)
    clean = re.sub(r"[-_](128|64|32|16)k\b", "", clean, flags=re.IGNORECASE)

    # Known repository redirects where tokenizer lives under upstream org
    if clean == "unsloth/Qwen3-0.6B":
        return "Qwen/Qwen3-0.6B"
    if clean.startswith("unsloth/Qwen"):
        # e.g. unsloth/Qwen2.5-Coder-0.5B-Instruct -> Qwen/Qwen2.5-Coder-0.5B-Instruct
        return clean.replace("unsloth/", "Qwen/")

    return clean


def infer_dtype(model_id: str, configured_dtype: str | None = None, is_gguf: bool = False) -> str:
    """Selects the optimal torch dtype for the model architecture.

    Rules:
    1. Explicitly configured dtype in catalog YAML or env var always takes precedence.
    2. Gemma models strictly require bfloat16 (or float32) in vLLM due to numerical instability in float16.
    3. GGUF models generally use 'half' (weights are dequantized into float16 internally).
    4. Full-precision models default to 'bfloat16' for stability on modern GPUs.
    """
    if configured_dtype:
        return configured_dtype.strip().lower()

    env_dtype = os.getenv("VLLM_DTYPE")
    if env_dtype:
        return env_dtype.strip().lower()

    lower_id = model_id.lower()

    # Gemma architectures (e.g. gemma, gemma2, gemma3, gemma3_text) forbid float16 in vLLM
    if "gemma" in lower_id:
        return "bfloat16"

    if is_gguf:
        return "half"

    return "bfloat16"


def is_gguf_model(model_id: str) -> bool:
    """Detect if a model is packaged in GGUF format."""
    lower = model_id.lower()
    return bool(re.search(r"(:q\d|gguf|\.gguf)", lower))


def load_local_models_from_catalog(config_path: Path) -> list[LocalModelConfig]:
    """Parses catalog YAML and extracts local models with dynamic resource allocation."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required. Run 'pip install pyyaml'.")

    print(f"[vLLM Multi-Server] Reading model catalog from: {config_path}", flush=True)
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_models = data.get("models", [])
    if not isinstance(raw_models, list):
        print(f"[vLLM Multi-Server] Warning: 'models' section is not a list in {config_path}", flush=True)
        raw_models = []

    # Filter for local models targeting vLLM
    local_entries: list[dict[str, Any]] = []
    for m in raw_models:
        if not isinstance(m, dict):
            continue
        provider = str(m.get("provider", "")).lower()
        exec_type = str(m.get("execution_type", "")).lower()
        base_url = str(m.get("base_url", "")).lower()

        # Matches local vLLM models
        is_local_vllm = (provider in ("vllm", "local", "local_vllm") and exec_type in ("local", "")) or (
            "vllm:" in base_url or "localhost:7000" in base_url
        )

        if is_local_vllm and "model_id" in m:
            local_entries.append(m)

    # Fallback to env var if no local models in config file
    if not local_entries:
        fallback_id = os.getenv("VLLM_MODEL", "unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
        print(
            f"[vLLM Multi-Server] No local vLLM models found in {config_path}. "
            f"Using fallback model from environment: {fallback_id}",
            flush=True,
        )
        local_entries.append(
            {
                "model_id": fallback_id,
                "logical_aliases": ["general", "default"],
                "max_context": 8192,
            }
        )

    num_models = len(local_entries)

    # Dynamic GPU memory slicing:
    # If VLLM_GPU_MEM_UTIL is set explicitly, use it as per-model cap.
    # Otherwise divide safe total budget (default 0.50 for multi-model) across models.
    env_mem = os.getenv("VLLM_GPU_MEM_UTIL")
    if env_mem:
        try:
            per_model_mem = float(env_mem)
        except ValueError:
            per_model_mem = 0.20
    else:
        # Safe allocation per model to fit multiple models concurrently on GPU
        # e.g. 2 models -> 0.25 each (total 50% VRAM)
        per_model_mem = min(0.35, max(0.08, round(0.50 / max(num_models, 1), 2)))

    resolved: list[LocalModelConfig] = []
    for idx, entry in enumerate(local_entries):
        mid = str(entry["model_id"]).strip()
        aliases = [str(a).strip() for a in entry.get("logical_aliases", []) if a]
        gguf = is_gguf_model(mid)
        tok = infer_tokenizer(mid, entry.get("tokenizer"))

        # Context length capped for safe memory utilization in concurrent mode
        raw_ctx = int(entry.get("max_model_len") or entry.get("max_context") or entry.get("context_window") or 4096)
        safe_ctx = min(raw_ctx, int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")))

        # Dtype selection
        dtype = infer_dtype(mid, entry.get("dtype"), gguf)

        # Per-model memory override or dynamic slice
        model_mem = entry.get("gpu_memory_utilization") or entry.get("gpu_mem_util")
        if model_mem is not None:
            try:
                alloc_mem = float(model_mem)
            except ValueError:
                alloc_mem = per_model_mem
        else:
            alloc_mem = per_model_mem

        port = BASE_INTERNAL_PORT + idx

        cfg = LocalModelConfig(
            model_id=mid,
            aliases=aliases,
            tokenizer=tok,
            port=port,
            dtype=dtype,
            max_model_len=safe_ctx,
            gpu_mem_util=alloc_mem,
            is_gguf=gguf,
        )
        resolved.append(cfg)

    return resolved


def build_lookup_table(models: list[LocalModelConfig]) -> dict[str, LocalModelConfig]:
    """Builds a case-insensitive lookup table matching model_id, aliases, and keywords."""
    lookup: dict[str, LocalModelConfig] = {}
    for m in models:
        # 1. Exact model_id
        lookup[m.model_id.lower()] = m
        # 2. Base model name without repo prefix (e.g., 'Qwen3-0.6B-GGUF:Q4_K_M')
        if "/" in m.model_id:
            lookup[m.model_id.split("/")[-1].lower()] = m
        # 3. Clean name without quantization (e.g., 'unsloth/qwen3-0.6b-gguf')
        clean = m.model_id.split(":")[0].lower()
        lookup[clean] = m
        if "/" in clean:
            lookup[clean.split("/")[-1]] = m
        # 4. Aliases
        for alias in m.aliases:
            a_lower = alias.lower()
            if a_lower not in lookup:
                lookup[a_lower] = m
    return lookup


def resolve_target_model(
    requested_model: str | None,
    models: list[LocalModelConfig] | None = None,
    lookup: dict[str, LocalModelConfig] | None = None,
) -> LocalModelConfig:
    """Matches an incoming requested model string to the best local model config."""
    active_models = models if models is not None else _local_models
    active_lookup = lookup if lookup is not None else _model_lookup
    default_m = _default_model or (active_models[0] if active_models else None)

    if not requested_model:
        if default_m is None:
            raise RuntimeError("No local models configured to handle request.")
        return default_m

    req_clean = requested_model.strip().lower()

    # 1. Exact lookup
    if req_clean in active_lookup:
        return active_lookup[req_clean]

    # 2. Substring matching for model families (e.g. 'llama', 'qwen', 'deepseek', 'mistral')
    for m in active_models:
        m_lower = m.model_id.lower()
        keywords = re.findall(r"[a-z0-9]+", m_lower)
        for kw in keywords:
            if len(kw) >= 4 and kw in req_clean:
                return m

    # 3. Fallback to default
    if default_m is None:
        raise RuntimeError("No local models configured to handle request.")
    return default_m


def is_backend_alive(port: int) -> bool:
    """Checks if an internal vLLM instance is responding to /health."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def check_engine_health(cfg: LocalModelConfig) -> str:
    """Checks the live status of an engine process and its HTTP endpoint."""
    if cfg.process is not None:
        poll = cfg.process.poll()
        if poll is not None:
            cfg.status = f"crashed (exit code {poll})"
            cfg.last_healthy = False
            return cfg.status

    if is_backend_alive(cfg.port):
        cfg.status = "healthy"
        cfg.last_healthy = True
        cfg.last_check_time = time.time()
        return "healthy"

    if cfg.process is not None and cfg.process.poll() is None:
        cfg.status = "starting"
        cfg.last_healthy = False
        return "starting"

    cfg.status = "stopped"
    cfg.last_healthy = False
    return "stopped"


def start_watchdog_thread() -> None:
    """Starts a daemon thread that supervises engine processes and performs auto-restart."""

    def _watchdog_loop() -> None:
        while not _server_shutting_down:
            for cfg in _local_models:
                status = check_engine_health(cfg)
                if "crashed" in status and cfg.restart_count < 3 and not _server_shutting_down:
                    cfg.restart_count += 1
                    print(
                        f"[vLLM Multi-Server] WARNING: Engine for '{cfg.model_id}' died ({status})! "
                        f"Attempting auto-restart #{cfg.restart_count}/3...",
                        flush=True,
                    )
                    try:
                        cfg.process = launch_engine(cfg)
                        cfg.status = "starting"
                    except Exception as e:
                        print(f"[vLLM Multi-Server] ERROR: Failed to restart '{cfg.model_id}': {e}", flush=True)
            time.sleep(3.0)

    t = threading.Thread(target=_watchdog_loop, daemon=True, name="vllm-engine-watchdog")
    t.start()


def build_models_response(models: list[LocalModelConfig]) -> dict[str, Any]:
    """Generates an OpenAI-compatible /v1/models response for available local models."""
    data: list[dict[str, Any]] = []
    now = int(time.time())
    for m in models:
        # Exclude models whose processes have crashed and failed to restart
        if "crashed" in getattr(m, "status", ""):
            continue
        data.append(
            {
                "id": m.model_id,
                "object": "model",
                "created": now,
                "owned_by": "vllm",
                "root": m.model_id,
                "parent": None,
                "permission": [],
            }
        )
        # Expose primary aliases as models for standard SDKs
        for alias in m.aliases:
            data.append(
                {
                    "id": alias,
                    "object": "model",
                    "created": now,
                    "owned_by": "vllm-alias",
                    "root": m.model_id,
                    "parent": None,
                    "permission": [],
                }
            )
    return {"object": "list", "data": data}


class SystemAgnosticProxyHandler(BaseHTTPRequestHandler):
    """Threaded HTTP handler routing OpenAI API requests to matching local vLLM engines."""

    def log_message(self, format: str, *args: object) -> None:
        # Suppress standard access logs to avoid terminal clutter
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            parsed_url = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            filter_model = query_params.get("model", [None])[0]

            health_status: dict[str, Any] = {}
            any_healthy = False
            all_healthy = True

            for m in _local_models:
                status = check_engine_health(m)
                health_status[m.model_id] = {
                    "status": status,
                    "port": m.port,
                    "dtype": m.dtype,
                    "restarts": m.restart_count,
                }
                if status == "healthy":
                    any_healthy = True
                else:
                    all_healthy = False

            if filter_model:
                target_m = resolve_target_model(filter_model)
                is_ok = target_m.status == "healthy"
                code = 200 if is_ok else 503
                payload = {
                    "model": target_m.model_id,
                    "status": target_m.status,
                    "port": target_m.port,
                    "dtype": target_m.dtype,
                }
            else:
                code = 200 if any_healthy else 503
                payload = {
                    "status": "healthy" if all_healthy else ("degraded" if any_healthy else "starting"),
                    "backends": health_status,
                }

            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        if self.path in ("/v1/models", "/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(build_models_response(_local_models)).encode("utf-8"))
            return

        # Default GET: forward to primary model backend
        target = _default_model or _local_models[0]
        self._proxy_request(target, method="GET")

    def do_POST(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b""

        # Extract requested model from JSON payload
        requested_model = None
        parsed_payload: dict[str, Any] | None = None
        if body:
            try:
                parsed_payload = json.loads(body.decode("utf-8"))
                if isinstance(parsed_payload, dict):
                    requested_model = parsed_payload.get("model")
            except Exception:
                parsed_payload = None

        target = resolve_target_model(requested_model)

        # Ensure target engine is monitored / attempt recovery if crashed
        target_status = check_engine_health(target)
        if "crashed" in target_status and target.restart_count < 3 and not _server_shutting_down:
            target.restart_count += 1
            print(
                f"[vLLM Multi-Server] On-demand restart #{target.restart_count} for '{target.model_id}'...",
                flush=True,
            )
            try:
                target.process = launch_engine(target)
            except Exception as e:
                print(f"[vLLM Multi-Server] Failed on-demand restart: {e}", flush=True)

        # Normalize the "model" field to match target engine's served model name
        if parsed_payload is not None and "model" in parsed_payload:
            parsed_payload["model"] = target.model_id
            body = json.dumps(parsed_payload).encode("utf-8")

        self._proxy_request(target, method="POST", body=body)

    def _proxy_request(self, target: LocalModelConfig, method: str, body: bytes | None = None) -> None:
        target_url = f"http://127.0.0.1:{target.port}{self.path}"
        req = urllib.request.Request(target_url, data=body, method=method)

        # Forward headers, adjusting Content-Length and Host
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length"):
                req.add_header(k, v)
        if body is not None:
            req.add_header("Content-Length", str(len(body)))

        try:
            with urllib.request.urlopen(req, timeout=180.0) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding",):
                        self.send_header(k, v)
                self.end_headers()
                # Streams both regular and chunked/SSE responses
                shutil.copyfileobj(resp, self.wfile)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding",):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            target_status = getattr(target, "status", "unknown")
            err_resp = {
                "error": {
                    "message": (
                        f"vLLM engine for '{target.model_id}' (port {target.port}) unavailable: {e}. "
                        f"Engine status: {target_status} (dtype: {target.dtype})"
                    ),
                    "type": "backend_unavailable",
                    "code": 503,
                }
            }
            self.wfile.write(json.dumps(err_resp).encode("utf-8"))


def launch_engine(cfg: LocalModelConfig) -> subprocess.Popen:
    """Launches a single vLLM server subprocess on its allocated internal port."""
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        cfg.model_id,
        "--served-model-name",
        cfg.model_id,
        *cfg.aliases,
        "--port",
        str(cfg.port),
        "--host",
        "127.0.0.1",
        "--dtype",
        cfg.dtype,
        "--max-model-len",
        str(cfg.max_model_len),
        "--gpu-memory-utilization",
        str(cfg.gpu_mem_util),
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--max-num-seqs",
        "64",
        "--enable-prefix-caching",
        "--enforce-eager",
    ]

    if cfg.tokenizer:
        cmd.extend(["--tokenizer", cfg.tokenizer])

    print(
        f"[vLLM Multi-Server] Launching engine for '{cfg.model_id}' on 127.0.0.1:{cfg.port} "
        f"(mem: {cfg.gpu_mem_util}, dtype: {cfg.dtype}, tokenizer: {cfg.tokenizer})...",
        flush=True,
    )
    return subprocess.Popen(cmd)


def shutdown(signum: int = 0, frame: object = None) -> None:
    """Graceful termination of all child processes and router."""
    global _server_shutting_down
    if _server_shutting_down:
        return
    _server_shutting_down = True
    print("\n[vLLM Multi-Server] Shutting down all vLLM engines...", flush=True)

    for m in _local_models:
        if m.process and m.process.poll() is None:
            try:
                m.process.terminate()
            except Exception:
                pass

    # Wait up to 5s for clean process exit
    deadline = time.time() + 5.0
    for m in _local_models:
        if m.process:
            while time.time() < deadline and m.process.poll() is None:
                time.sleep(0.1)
            if m.process.poll() is None:
                try:
                    m.process.kill()
                except Exception:
                    pass

    print("[vLLM Multi-Server] Shutdown complete.", flush=True)
    sys.exit(0)


def print_discovered_summary(models: list[LocalModelConfig], config_path: Path) -> None:
    """Displays a formatted summary of discovered local models and their allocations."""
    print("=" * 70)
    print("AdaRoute System-Agnostic vLLM Multi-Engine Router")
    print(f"Catalog Source: {config_path}")
    print(f"External Gateway Port: {GATEWAY_PORT}")
    print(f"Local Models Configured: {len(models)}")
    print("-" * 70)
    for idx, m in enumerate(models, 1):
        print(f"[{idx}] {m.model_id}")
        print(f"    - Internal Endpoint:  http://127.0.0.1:{m.port}")
        print(f"    - Base Tokenizer:     {m.tokenizer}")
        print(f"    - GPU Memory Util:    {m.gpu_mem_util} (dtype: {m.dtype})")
        print(f"    - Max Context:        {m.max_model_len}")
        print(f"    - Logical Aliases:    {', '.join(m.aliases) if m.aliases else 'None'}")
    print("=" * 70, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdaRoute System-Agnostic vLLM Multi-Engine Server")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to AdaRoute catalog YAML file (defaults to ADAROUTE_CONFIG or config.gpu.yaml).",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=GATEWAY_PORT,
        help=f"Port for the unified reverse proxy (default: {GATEWAY_PORT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate catalog discovery without launching vLLM engines.",
    )
    return parser.parse_args()


def main() -> None:
    global _local_models, _model_lookup, _default_model, GATEWAY_PORT
    args = parse_args()
    GATEWAY_PORT = args.port

    # 1. Discover and parse catalog configuration
    config_file = find_config_file(args.config)
    _local_models = load_local_models_from_catalog(config_file)
    if not _local_models:
        print("[vLLM Multi-Server] Fatal: No local models could be configured. Exiting.", file=sys.stderr)
        sys.exit(1)

    _default_model = _local_models[0]
    _model_lookup = build_lookup_table(_local_models)

    # 2. Display summary
    print_discovered_summary(_local_models, config_file)

    if args.dry_run:
        print("[vLLM Multi-Server] Dry-run complete. Exiting.", flush=True)
        sys.exit(0)

    # 3. Register signal handlers
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # 4. Launch vLLM engines for each model
    for m in _local_models:
        m.process = launch_engine(m)
        time.sleep(1.0)  # Slight stagger to avoid GPU initialization race

    # 5. Start engine watchdog thread
    start_watchdog_thread()

    # 6. Start unified reverse proxy
    server_address = ("0.0.0.0", GATEWAY_PORT)
    httpd = ThreadingHTTPServer(server_address, SystemAgnosticProxyHandler)
    print(
        f"\n[vLLM Multi-Server] Unified router listening on http://0.0.0.0:{GATEWAY_PORT}. "
        "Ready to accept gateway traffic.",
        flush=True,
    )

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
