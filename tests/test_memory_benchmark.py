import uuid

import pytest
from fastapi.testclient import TestClient

import api.main as main_mod
from api.main import app
from memory.benchmark_harness import run_memory_benchmark
from memory.benchmark_store import load_latest, save_latest


@pytest.fixture(autouse=True)
def _disable_mcp_startup(monkeypatch):
    async def _noop_init_mcp():
        return None

    monkeypatch.setattr(main_mod, "_init_mcp", _noop_init_mcp)


@pytest.mark.asyncio
async def test_memory_benchmark_harness_returns_expected_metric_shape():
    owner_id = f"benchmark-owner-{uuid.uuid4().hex[:8]}"
    payload = await run_memory_benchmark(owner_id)

    assert payload["owner_id"] == owner_id
    assert "overall_score" in payload
    assert 0.0 <= payload["overall_score"] <= 1.0
    assert "metrics" in payload
    assert "deterministic_extraction" in payload["metrics"]
    assert "direct_fact_guard" in payload["metrics"]
    assert "semantic_retrieval" in payload["metrics"]


def test_memory_benchmark_store_round_trip():
    owner_id = f"benchmark-store-{uuid.uuid4().hex[:8]}"
    sample = {"owner_id": owner_id, "overall_score": 0.77}
    save_latest(owner_id, sample)
    loaded = load_latest(owner_id)
    assert loaded is not None
    assert loaded["overall_score"] == 0.77


def test_memory_benchmark_api_run_and_latest():
    owner_id = f"benchmark-api-{uuid.uuid4().hex[:8]}"
    with TestClient(app) as client:
        run_resp = client.post("/memory/benchmark/run", json={"owner_id": owner_id})
        assert run_resp.status_code == 200
        run_payload = run_resp.json()
        assert run_payload["owner_id"] == owner_id
        assert "metrics" in run_payload

        latest_resp = client.get("/memory/benchmark/latest", params={"owner_id": owner_id})
        assert latest_resp.status_code == 200
        latest_payload = latest_resp.json()
        assert latest_payload["result"] is not None
        assert latest_payload["result"]["owner_id"] == owner_id
