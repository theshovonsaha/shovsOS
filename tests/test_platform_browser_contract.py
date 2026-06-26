from fastapi.testclient import TestClient

from api.main import app


def test_browser_facing_api_prefix_contract_is_available():
    client = TestClient(app)

    runtime = client.get("/api/runtime/health")
    tools = client.get("/api/tools")
    harness = client.get("/api/harness-lab/catalog")
    workflow = client.get("/api/workflow-lab/catalog")

    assert runtime.status_code == 200
    assert runtime.json()["features"]["api_prefix_compatibility"] is True
    assert runtime.json()["tools"]["image_generate"] is True

    assert tools.status_code == 200
    assert any(item["name"] == "image_generate" for item in tools.json()["tools"])

    assert harness.status_code == 200
    assert harness.json()["runtime"]["frontend_endpoint"] == "/api/harness-lab/runtime/run"

    assert workflow.status_code == 200
    assert any(item["id"] == "vision_inspection_v1" for item in workflow.json()["workflows"])


def test_browser_facing_runtime_harness_contract_is_actionable():
    client = TestClient(app)

    response = client.post("/api/harness-lab/runtime/run", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["passed"] is True
    assert payload["coverage"]["frontend_contract"] == [
        "tool_call",
        "tool_result",
        "image_generation_result",
        "done",
    ]
    assert payload["operator_guidance"]
