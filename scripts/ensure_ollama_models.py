"""Ensure all locally configured models are present in the Ollama service."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
import yaml

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11436").rstrip("/")
CONFIG_PATH = Path(os.getenv("ADAROUTE_CONFIG", "/app/config.yaml"))
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)


def configured_local_models() -> list[str]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    models = config.get("models", [])
    return list(
        dict.fromkeys(
            model["model_id"] for model in models if model.get("provider") == "local" and model.get("model_id")
        )
    )


def wait_for_ollama(client: httpx.Client) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            response = client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            return
        except (httpx.HTTPError, OSError):
            time.sleep(2)
    raise RuntimeError(f"Ollama did not become ready at {OLLAMA_URL}")


def installed_models(client: httpx.Client) -> set[str]:
    response = client.get(f"{OLLAMA_URL}/api/tags")
    response.raise_for_status()
    return {model["name"] for model in response.json().get("models", [])}


def pull_model(client: httpx.Client, model: str) -> None:
    print(f"Pulling Ollama model {model}...", flush=True)
    with client.stream("POST", f"{OLLAMA_URL}/api/pull", json={"name": model, "stream": True}) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                payload = yaml.safe_load(line) or {}
                if payload.get("error"):
                    raise RuntimeError(f"Ollama failed to pull {model}: {payload['error']}")
                status = payload.get("status")
                if status:
                    print(f"  {status}", flush=True)
    print(f"Installed Ollama model {model}", flush=True)


def main() -> int:
    models = configured_local_models()
    if not models:
        raise RuntimeError(f"No local models found in {CONFIG_PATH}")

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        wait_for_ollama(client)
        installed = installed_models(client)
        for model in models:
            if model not in installed:
                pull_model(client, model)
            else:
                print(f"Ollama model already installed: {model}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, httpx.HTTPError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"Failed to prepare Ollama models: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
