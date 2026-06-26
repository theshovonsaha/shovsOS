from fastapi.testclient import TestClient

import pytest

from api.harness_lab import (
    get_harness_lab_catalog,
    run_harness_core_benchmark,
    run_mode_comparison,
    run_runtime_harness_tests,
)
from api.main import app


def test_harness_lab_catalog_exposes_modes_wedges_and_references():
    catalog = get_harness_lab_catalog()

    assert catalog["title"] == "Harness Lab"
    assert catalog["benchmark"]["deterministic"] is True
    assert any(mode["id"] == "plain_model" for mode in catalog["modes"])
    assert any(mode["id"] == "shovs_enforced" for mode in catalog["modes"])
    assert any(wedge["id"] == "run_ledger" for wedge in catalog["wedges"])
    assert any(ref["id"] == "agenttrace" for ref in catalog["papers"])
    assert catalog["comparison"]["endpoint"] == "/harness-lab/compare/run"
    assert catalog["runtime"]["endpoint"] == "/harness-lab/runtime/run"


def test_harness_lab_benchmark_is_deterministic_and_passes():
    result = run_harness_core_benchmark()

    assert result["suite"] == "agent_harness_core"
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert {item["scenario"] for item in result["results"]} == {
        "response_guard_tool_json",
        "source_collection_contract",
        "source_collection_drift_negative_control",
        "policy_trace_recovery_contract",
        "ledger_orphan_tool_result_rejected",
        "memory_replacement_rollback",
    }


def test_harness_lab_api_endpoints():
    client = TestClient(app)

    catalog = client.get("/api/harness-lab/catalog")
    assert catalog.status_code == 200
    assert catalog.json()["runtime"]["frontend_endpoint"] == "/api/harness-lab/runtime/run"

    catalog = client.get("/harness-lab/catalog")
    assert catalog.status_code == 200
    assert catalog.json()["benchmark"]["endpoint"] == "/harness-lab/benchmark/run"
    assert catalog.json()["benchmark"]["frontend_endpoint"] == "/api/harness-lab/benchmark/run"

    run = client.post("/harness-lab/benchmark/run", json={})
    assert run.status_code == 200
    assert run.json()["passed"] is True

    compare = client.post(
        "/harness-lab/compare/run",
        json={"objective": "Search top 3 stocks today, then search each, fetch 3 URLs each."},
    )
    assert compare.status_code == 200
    payload = compare.json()
    assert payload["suite"] == "shovs_mode_comparison"
    assert payload["deterministic"] is True
    assert [item["mode"] for item in payload["results"]] == [
        "plain_model",
        "model_plus_tools",
        "shovs_react",
        "shovs_plan_execute",
        "shovs_graph_harness",
    ]
    assert payload["results"][0]["passed"] is False
    assert payload["results"][1]["passed"] is False
    assert payload["results"][2]["passed"] is False
    assert payload["results"][3]["passed"] is True
    assert payload["results"][4]["passed"] is True
    assert payload["results"][3]["control_policy"] == "plan_execute"
    assert payload["results"][4]["control_policy"] == "graph_harness"
    assert payload["results"][3]["policy_eval"]["passed"] is True
    assert payload["results"][4]["policy_eval"]["passed"] is True

    runtime = client.post("/harness-lab/runtime/run", json={})
    assert runtime.status_code == 200
    runtime_payload = runtime.json()
    assert runtime_payload["suite"] == "shovs_runtime_harness"
    assert runtime_payload["passed"] is True
    assert runtime_payload["deterministic"] is True
    assert runtime_payload["live_model_required"] is False

    runtime_api = client.post("/api/harness-lab/runtime/run", json={})
    assert runtime_api.status_code == 200
    assert runtime_api.json()["passed"] is True


def test_mode_comparison_exposes_real_state_differences():
    result = run_mode_comparison("Find top 3 sushi places in Toronto, search each, fetch 3 URLs each.")

    by_mode = {item["mode"]: item for item in result["results"]}
    assert by_mode["shovs_plan_execute"]["state_eval"]["state"]["entities"] == [
        "SUSHI MASAKI SAITO",
        "YASU",
        "SUSHI KAJI",
    ]
    assert by_mode["plain_model"]["issues"] == [
        "no_tool_evidence",
        "no_ledger_authority",
        "no_source_contract",
    ]
    assert "forbidden_query_drift" in by_mode["model_plus_tools"]["issues"]
    assert "mutable_plan_drift_risk" in by_mode["shovs_react"]["issues"]
    assert by_mode["shovs_plan_execute"]["score"] == 1.0
    assert by_mode["shovs_plan_execute"]["trace_summary"]["completion_gate"] == "passed"
    assert by_mode["shovs_plan_execute"]["policy_eval"]["state"]["ledger_modes"] == ["ledger_enforced"]
    assert by_mode["shovs_graph_harness"]["control_policy"] == "graph_harness"
    assert result["contract"]["workflow_shape"] == "source_collection"
    assert result["pass_graph"]["workflow_shape"] == "source_collection"


def test_mode_comparison_uses_shopping_entities_for_comparison_tasks():
    result = run_mode_comparison("Compare top 3 stores for a product, search each, fetch 3 URLs each.")
    by_mode = {item["mode"]: item for item in result["results"]}

    assert by_mode["shovs_plan_execute"]["state_eval"]["state"]["entities"] == [
        "COSTCO",
        "WALMART",
        "BEST BUY",
    ]
    assert by_mode["shovs_plan_execute"]["passed"] is True
    assert by_mode["model_plus_tools"]["state_eval"]["state"]["forbidden_hits"]


@pytest.mark.asyncio
async def test_runtime_harness_executes_real_run_engine_probe():
    result = await run_runtime_harness_tests()
    by_scenario = {item["scenario"]: item for item in result["results"]}

    assert result["passed"] is True
    assert by_scenario["runtime_image_generation_tool_loop"]["passed"] is True
    assert by_scenario["runtime_image_generation_tool_loop"]["evidence"]["event_types"] == [
        "session",
        "conversation_tension",
        "plan",
        "tool_call",
        "tool_result",
        "token",
        "done",
    ]
    assert (
        by_scenario["runtime_image_generation_tool_loop"]["evidence"]["tool_result_payload"]["type"]
        == "image_generation_result"
    )
    assert by_scenario["runtime_ledger_rejects_fake_tool_claim"]["passed"] is True
    assert by_scenario["runtime_policy_gate_recovers_entity_drift"]["passed"] is True
    assert result["coverage"]["failure_modes"] == ["tool_not_called", "fake_tool_claim", "entity_drift"]
