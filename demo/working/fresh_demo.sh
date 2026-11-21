#!/usr/bin/env bash
# AdaRoute Fresh Demo Lifecycle Automation
# Tears down existing containers, spins up clean GPU containers,
# waits for replica initialization, and executes the verification suite.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================================================="
echo "  AdaRoute Fresh Demonstration Bootstrap"
echo "=========================================================================="

echo "[1/4] Tearing down running containers to clear all in-memory & cache state..."
docker compose down -v 2>/dev/null || docker compose down

echo "[2/4] Starting fresh GPU services (NVIDIA RTX PRO 4000 Profile)..."
just gpu-up

echo "[3/4] Polling gateway cluster readiness (waiting for all 3 vLLM engines)..."
for i in {1..30}; do
  if curl -sf http://localhost:7000/health | grep -q '"status":"healthy"'; then
    echo "  -> Gateway and all engine replicas are HEALTHY!"
    break
  fi
  echo "  -> Waiting for vLLM replicas to profile and reach healthy state... (${i}/30)"
  sleep 4
done

echo "[4/4] Executing master 15-step demonstration verification..."
python3 demo/working/run_automated_demo.py

echo "=========================================================================="
echo "  Fresh Demonstration Environment is READY!"
echo "  To present live, launch:"
echo "    just cli"
echo "=========================================================================="
