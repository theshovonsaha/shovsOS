import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import app
from api.workflow_lab import (
    WorkflowRunStore,
    build_custom_workflow_definition,
    execute_live_workflow_run,
    get_workflow_definition,
    make_workflow_lab_router,
    start_workflow_run,
    workflow_catalog,
)


def test_workflow_lab_catalog_exposes_contracts_and_guides():
    catalog = workflow_catalog()

    assert catalog["title"] == "Workflow Lab"
    assert catalog["api"]["start"] == "POST /workflow-lab/workflows/{workflow_id}/runs"
    assert any(item["id"] == "shopping_comparison_v1" for item in catalog["workflows"])
    assert any(item["id"] == "vision_inspection_v1" for item in catalog["workflows"])
    shopping = next(item for item in catalog["workflows"] if item["id"] == "shopping_comparison_v1")
    assert shopping["runtime"]["policy"] == "plan_execute"
    assert shopping["runtime"]["ledger_mode"] == "ledger_enforced"
    assert "web_search" in shopping["tools"]
    assert shopping["memory_policy"]["provenance_required"] is True
    assert catalog["legal_usage"]


def test_workflow_definition_has_visible_steps_and_output_schema():
    definition = get_workflow_definition("local_recommendation_v1").to_dict()

    assert [step["id"] for step in definition["steps"]] == [
        "intake",
        "context",
        "plan",
        "tools",
        "evidence",
        "memory",
        "verify",
        "response",
    ]
    assert definition["output_schema"]["type"] == "ranked_local_options"
    assert "source_urls" in definition["output_schema"]["fields"]


def test_vision_workflow_contract_declares_image_inputs():
    definition = get_workflow_definition("vision_inspection_v1").to_dict()

    assert definition["runtime"]["policy"] == "react"
    assert definition["input_schema"]["images"] == "optional base64[]"
    assert definition["output_schema"]["type"] == "vision_inspection"
    assert "vision" in definition["tags"]


def test_start_workflow_run_returns_status_events_and_result_contract():
    record = start_workflow_run(
        "shopping_comparison_v1",
        {
            "input": {
                "product": "air purifier",
                "location": "Toronto",
                "stores": ["Costco", "Walmart", "Best Buy"],
            },
            "owner_id": "owner-test",
        },
    )

    assert record["status"] == "completed"
    assert record["definition"]["id"] == "shopping_comparison_v1"
    assert record["events"][0]["phase"] == "intake"
    assert any(event["phase"] == "verify" for event in record["events"])
    assert record["result"]["output_schema"]["type"] == "comparison_table"
    assert record["result"]["api_contract"]["events"].endswith("/events")


def test_workflow_lab_api_run_lifecycle():
    client = TestClient(app)

    catalog = client.get("/workflow-lab/catalog")
    assert catalog.status_code == 200
    api_catalog = client.get("/api/workflow-lab/catalog")
    assert api_catalog.status_code == 200
    assert api_catalog.json()["title"] == "Workflow Lab"

    created = client.post(
        "/workflow-lab/workflows/research_brief_v1/runs",
        json={"input": {"query": "agent workflow traceability", "source_count": 3}},
    )
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    status = client.get(f"/workflow-lab/runs/{run_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["event_count"] >= 6

    events = client.get(f"/workflow-lab/runs/{run_id}/events")
    assert events.status_code == 200
    assert events.json()["events"][0]["event_type"] == "workflow_input"

    result = client.get(f"/workflow-lab/runs/{run_id}/result")
    assert result.status_code == 200
    assert result.json()["trace_summary"]["ledger_mode"] == "ledger_enforced"

    api_result = client.get(f"/api/workflow-lab/runs/{run_id}/result")
    assert api_result.status_code == 200
    assert api_result.json()["run_id"] == run_id


def test_workflow_lab_unknown_workflow_returns_404():
    client = TestClient(app)

    assert client.get("/workflow-lab/workflows/missing").status_code == 404
    assert client.post("/workflow-lab/workflows/missing/runs", json={}).status_code == 404


def test_workflow_runs_are_durable(tmp_path):
    db_path = tmp_path / "workflow_runs.db"
    store = WorkflowRunStore(str(db_path))
    record = start_workflow_run(
        "local_recommendation_v1",
        {"input": {"category": "sushi", "location": "Toronto"}},
        store=store,
    )

    reloaded = WorkflowRunStore(str(db_path)).get_run(record["run_id"])

    assert reloaded is not None
    assert reloaded["status"] == "completed"
    assert reloaded["workflow_id"] == "local_recommendation_v1"
    assert len(reloaded["events"]) >= 5
    assert reloaded["result"]["output_schema"]["type"] == "ranked_local_options"


def test_custom_workflow_definition_can_be_created_and_run(tmp_path):
    store = WorkflowRunStore(str(tmp_path / "workflow_runs.db"))
    definition = build_custom_workflow_definition(
        {
            "label": "Receipt Checker",
            "description": "Check receipt photos and summarize questionable charges.",
            "policy": "react",
            "tools": ["query_memory"],
            "input_fields": ["question", "image_count"],
            "output_fields": ["answer", "visible_details", "uncertainties"],
        }
    )
    store.save_workflow_definition(definition, owner_id="owner-custom")

    catalog = workflow_catalog(store=store, owner_id="owner-custom")
    assert any(item["id"] == definition.id for item in catalog["workflows"])

    record = start_workflow_run(
        definition.id,
        {"input": {"question": "What is on this receipt?", "image_count": 1}},
        store=store,
    )

    assert record["status"] == "completed"
    assert record["definition"]["id"] == definition.id
    assert record["result"]["output_schema"]["fields"] == ["answer", "visible_details", "uncertainties"]


def test_workflow_lab_api_creates_custom_workflow_and_runs_it(tmp_path):
    store = WorkflowRunStore(str(tmp_path / "workflow_runs.db"))
    test_app = FastAPI()
    test_app.include_router(make_workflow_lab_router(store=store, default_model="fake:model"))
    client = TestClient(test_app)

    created = client.post(
        "/workflow-lab/workflows",
        json={
            "label": "Store Price Watch",
            "description": "Compare a product across stores and return a compact table.",
            "tools": ["web_search", "web_fetch"],
            "input_fields": ["product", "location", "stores"],
            "output_fields": ["best_option", "prices", "source_urls"],
        },
    )

    assert created.status_code == 200
    workflow = created.json()["workflow"]
    assert workflow["id"].startswith("store_price_watch")
    assert workflow["custom"] is True or workflow["id"]

    catalog = client.get("/workflow-lab/catalog")
    assert any(item["id"] == workflow["id"] for item in catalog.json()["workflows"])

    run = client.post(
        f"/workflow-lab/workflows/{workflow['id']}/runs",
        json={"input": {"product": "chair", "location": "Toronto", "stores": "Walmart"}},
    )
    assert run.status_code == 200
    status = client.get(run.json()["status_url"])
    assert status.status_code == 200
    assert status.json()["status"] == "completed"


class FakeRunEngine:
    def __init__(self):
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield {"type": "session", "session_id": request.session_id, "run_id": "engine-run-1"}
        yield {"type": "tool_call", "tool": "web_search", "arguments": {"query": "test"}}
        yield {"type": "tool_result", "tool": "web_search", "success": True}
        yield {"type": "token", "content": "Live workflow answer."}
        yield {"type": "done", "run_id": "engine-run-1", "session_id": request.session_id}


def test_live_workflow_execution_records_run_engine_events(tmp_path):
    store = WorkflowRunStore(str(tmp_path / "workflow_runs.db"))
    definition = get_workflow_definition("research_brief_v1")
    fake_engine = FakeRunEngine()
    record = store.create_run(
        definition=definition,
        input_payload={"query": "agent traces"},
        mode="live_run_engine",
        status="queued",
    )

    asyncio.run(
        execute_live_workflow_run(
            run_id=record["run_id"],
            store=store,
            run_engine=fake_engine,
            definition=definition,
            input_payload={"query": "agent traces"},
            model="fake:model",
        )
    )
    loaded = store.get_run(record["run_id"])

    assert loaded is not None
    assert loaded["status"] == "completed"
    assert any(event["event_type"] == "run_engine_tool_call" for event in loaded["events"])
    assert loaded["result"]["run_engine_run_id"] == "engine-run-1"
    assert loaded["result"]["summary"] == "Live workflow answer."
    assert fake_engine.requests[0].images is None


def test_live_vision_workflow_forwards_images_without_storing_raw_blobs(tmp_path):
    store = WorkflowRunStore(str(tmp_path / "workflow_runs.db"))
    definition = get_workflow_definition("vision_inspection_v1")
    fake_engine = FakeRunEngine()
    record = store.create_run(
        definition=definition,
        input_payload={"question": "What is visible?", "image_count": 1},
        mode="live_run_engine",
        status="queued",
    )

    asyncio.run(
        execute_live_workflow_run(
            run_id=record["run_id"],
            store=store,
            run_engine=fake_engine,
            definition=definition,
            input_payload={"question": "What is visible?", "image_count": 1},
            model="ollama:llava",
            images=["base64-image"],
        )
    )
    loaded = store.get_run(record["run_id"])

    assert loaded is not None
    assert loaded["status"] == "completed"
    assert loaded["input"] == {"question": "What is visible?", "image_count": 1}
    assert fake_engine.requests[0].images == ["base64-image"]
    assert fake_engine.requests[0].control_policy == "react"


def test_live_workflow_router_accepts_async_mode(tmp_path):
    store = WorkflowRunStore(str(tmp_path / "workflow_runs.db"))
    test_app = FastAPI()
    test_app.include_router(make_workflow_lab_router(run_engine=FakeRunEngine(), store=store, default_model="fake:model"))
    client = TestClient(test_app)

    created = client.post(
        "/workflow-lab/workflows/research_brief_v1/runs",
        json={"mode": "live_run_engine", "input": {"query": "agent traces"}},
    )

    assert created.status_code == 200
    run_id = created.json()["run_id"]
    status = client.get(f"/workflow-lab/runs/{run_id}")
    assert status.status_code == 200
    assert status.json()["status"] in {"queued", "running", "completed"}
