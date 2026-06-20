from fastapi.testclient import TestClient

from api.harness_lab import get_harness_lab_catalog, run_harness_core_benchmark
from api.main import app


def test_harness_lab_catalog_exposes_modes_wedges_and_references():
    catalog = get_harness_lab_catalog()

    assert catalog["title"] == "Harness Lab"
    assert catalog["benchmark"]["deterministic"] is True
    assert any(mode["id"] == "plain_model" for mode in catalog["modes"])
    assert any(mode["id"] == "shovs_enforced" for mode in catalog["modes"])
    assert any(wedge["id"] == "run_ledger" for wedge in catalog["wedges"])
    assert any(ref["id"] == "agenttrace" for ref in catalog["papers"])


def test_harness_lab_benchmark_is_deterministic_and_passes():
    result = run_harness_core_benchmark()

    assert result["suite"] == "agent_harness_core"
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert {item["scenario"] for item in result["results"]} == {
        "response_guard_tool_json",
        "source_collection_contract",
        "source_collection_drift_negative_control",
        "ledger_orphan_tool_result_rejected",
        "memory_replacement_rollback",
    }


def test_harness_lab_api_endpoints():
    client = TestClient(app)

    catalog = client.get("/api/harness-lab/catalog")
    assert catalog.status_code == 404

    catalog = client.get("/harness-lab/catalog")
    assert catalog.status_code == 200
    assert catalog.json()["benchmark"]["endpoint"] == "/harness-lab/benchmark/run"
    assert catalog.json()["benchmark"]["frontend_endpoint"] == "/api/harness-lab/benchmark/run"

    run = client.post("/harness-lab/benchmark/run", json={})
    assert run.status_code == 200
    assert run.json()["passed"] is True
