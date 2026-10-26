"""Tests for Auditable Control Plane and Policy Rollback (Week-5 Milestone M12)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adaroute.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_tc_m12_01_bayesian_estimator_state_endpoint(client):
    """Verifies that the online Bayesian estimator state endpoint exposes calibrated domains."""
    res = client.get("/v1/estimator/bayesian/state")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "active"
    assert "domains" in data
    assert "general" in data["domains"]
    assert "coding" in data["domains"]
    assert len(data["domains"]["general"]["mu"]) == 4


def test_tc_m12_02_control_plane_ledger_calculation(client, tmp_path, monkeypatch):
    """Verifies that the control plane ledger calculates attributable savings S_p = B_p - A_p."""
    fake_ledger = tmp_path / "ledger.jsonl"
    lines = [
        # Request 1: actual_cost = 0.0001, prompt=100, completion=50
        # Counterfactual B_p = (100/1000)*0.005 + (50/1000)*0.015 = 0.0005 + 0.00075 = 0.00125
        # Savings = 0.00125 - 0.0001 = 0.00115
        {
            "entry_type": "request",
            "request_id": "r1",
            "timestamp": 1000.0,
            "logical_model": "general",
            "selected_model": "qwen2.5:1.5b",
            "provider": "local",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost": 0.0001,
            "status": "success",
        },
        # Request 2: cached response (cost = 0.0, prompt=100, completion=50)
        # Counterfactual B_p = 0.00125, Savings = 0.00125
        {
            "entry_type": "request",
            "request_id": "r2",
            "timestamp": 1005.0,
            "logical_model": "general",
            "selected_model": "qwen2.5:1.5b",
            "provider": "cache",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost": 0.0,
            "status": "cached",
        },
    ]
    with open(fake_ledger, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item) + "\n" for item in lines)

    monkeypatch.setattr("adaroute.api.Path", lambda p: fake_ledger if "ledger.jsonl" in str(p) else Path(p))

    res = client.get("/v1/control-plane/ledger?limit=10")
    assert res.status_code == 200
    body = res.json()

    summary = body["summary"]
    assert summary["total_requests"] == 2
    assert summary["cache_hits"] == 1
    assert summary["cache_hit_rate"] == 50.0
    assert summary["actual_cost"] == 0.0001
    assert summary["baseline_cost"] == pytest.approx(0.0025, rel=1e-3)
    assert summary["attributable_savings"] == pytest.approx(0.0024, rel=1e-3)
    assert summary["savings_percentage"] > 90.0

    assert "general" in body["by_capability"]
    assert len(body["recent_transactions"]) == 2
    # Verify newest first sorting
    assert body["recent_transactions"][0]["request_id"] == "r2"


def test_tc_m12_03_control_plane_policies_and_rollback(client):
    """Verifies querying active policy, rolling back dynamically, and version tracking."""
    # 1. Get initial policy state
    res = client.get("/v1/control-plane/policies")
    assert res.status_code == 200
    data = res.json()
    assert data["active_policy"] in data["available_policies"]
    initial_ver = data["active_version"]

    # 2. Roll back to 'cost_first'
    rb_res = client.post(
        "/v1/control-plane/rollback",
        json={"policy_name": "cost_first", "author": "devops_lead"},
    )
    assert rb_res.status_code == 200
    rb_data = rb_res.json()
    assert rb_data["status"] == "success"
    assert rb_data["active_policy"] == "cost_first"
    assert rb_data["active_version"] == initial_ver + 1

    # 3. Roll back to version 1 by version ID
    rb_v1_res = client.post(
        "/v1/control-plane/rollback",
        json={"version": 1, "author": "admin"},
    )
    assert rb_v1_res.status_code == 200
    rb_v1_data = rb_v1_res.json()
    assert rb_v1_data["active_policy"] == "adaptive"
    assert rb_v1_data["active_version"] == initial_ver + 2


def test_tc_m12_04_control_plane_rollback_invalid_policy(client):
    """Rejects unknown policies with HTTP 400."""
    res = client.post(
        "/v1/control-plane/rollback",
        json={"policy_name": "non_existent_policy"},
    )
    assert res.status_code == 400
    assert "Unknown policy" in res.json()["detail"]


def test_tc_m12_05_control_plane_rollback_missing_version(client):
    """Rejects non-existent versions with HTTP 404."""
    res = client.post(
        "/v1/control-plane/rollback",
        json={"version": 9999},
    )
    assert res.status_code == 404
    assert "not found in version history" in res.json()["detail"]
