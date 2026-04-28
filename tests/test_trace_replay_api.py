import uuid

import pytest
from fastapi.testclient import TestClient

import api.main as main_mod
from api.main import app
from config.logger import get_logger, log
from config.trace_store import get_trace_store
from orchestration.run_store import get_run_store


@pytest.fixture(autouse=True)
def _disable_mcp_startup(monkeypatch):
    async def _noop_init_mcp():
        return None

    monkeypatch.setattr(main_mod, "_init_mcp", _noop_init_mcp)


def test_trace_run_replay_assembles_run_state():
    owner_id = f"trace-replay-owner-{uuid.uuid4().hex[:8]}"
    run_id = f"trace-replay-run-{uuid.uuid4().hex[:8]}"
    session_id = f"trace-replay-session-{uuid.uuid4().hex[:8]}"

    run_store = get_run_store()
    trace_store = get_trace_store()

    run_store.start_run(
        run_id=run_id,
        session_id=session_id,
        agent_id="default",
        model="llama3.2",
        owner_id=owner_id,
    )
    run_store.save_checkpoint(
        run_id=run_id,
        phase="response",
        tool_turn=1,
        status="complete",
        strategy="Use exact-domain evidence first.",
        notes="Promote verified source snippets into the final answer.",
        tools=["web_fetch"],
        tool_results=[{"tool_name": "web_fetch", "success": True}],
    )
    run_store.save_pass(
        run_id=run_id,
        phase="response",
        tool_turn=1,
        status="complete",
        objective="Investigate wigglebudget.com.",
        strategy="Use exact-domain evidence first.",
        notes="Response should stay grounded in fetched evidence.",
        selected_tools=["web_fetch"],
        compiled_context={
            "phase": "response",
            "included": [
                {
                    "item_id": "working_evidence",
                    "kind": "evidence",
                    "trace_id": "evidence:working_evidence",
                    "content": "Wiggle Budget homepage says it helps project balances and compare debt payoff plans.",
                    "provenance": {"selected_count": 1, "exact_target_matches": 1},
                }
            ],
        },
        response_preview="Wiggle Budget focuses on future-balance planning.",
        input_tokens=1200,
        output_tokens=180,
        total_tokens=1380,
        estimated_cost_usd=0.0125,
    )
    run_store.save_artifact(
        run_id=run_id,
        session_id=session_id,
        owner_id=owner_id,
        artifact_type="assistant_response",
        label="assistant-response.md",
        preview="Wiggle Budget focuses on future-balance planning.",
        size_bytes=56,
    )
    run_store.save_eval(
        run_id=run_id,
        session_id=session_id,
        owner_id=owner_id,
        eval_type="verification",
        phase="response",
        passed=True,
        detail="Evidence support present.",
    )
    run_store.finish_run(run_id, status="completed")

    trace_store.append_event(
        "default",
        session_id,
        "phase_context",
        {
            "phase": "response",
            "included": [
                {
                    "item_id": "working_evidence",
                    "kind": "evidence",
                    "trace_id": "evidence:working_evidence",
                    "content": "Wiggle Budget homepage says it helps project balances and compare debt payoff plans.",
                    "provenance": {"selected_count": 1},
                }
            ],
        },
        run_id=run_id,
        owner_id=owner_id,
    )

    with TestClient(app) as client:
        response = client.get(
            f"/logs/traces/run/{run_id}",
            params={"owner_id": owner_id, "trace_limit": 40},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["found"] is True
    assert payload["run"]["run_id"] == run_id
    assert payload["summary"]["checkpoint_count"] == 1
    assert payload["summary"]["artifact_count"] == 1
    assert payload["summary"]["eval_count"] == 1
    assert payload["summary"]["evidence_count"] >= 1
    assert payload["summary"]["total_tokens"] == 1380
    assert payload["summary"]["estimated_cost_usd"] == pytest.approx(0.0125)
    assert payload["latest_pass"]["objective"] == "Investigate wigglebudget.com."
    assert payload["passes"][0]["input_tokens"] == 1200
    assert payload["passes"][0]["cumulative_cost_usd"] == pytest.approx(0.0125)
    assert payload["evidence"][0]["item_id"] == "working_evidence"
    assert payload["evidence"][0]["provenance"]["selected_count"] == 1
    assert payload["artifacts"][0]["artifact_type"] == "assistant_response"


def test_trace_run_replay_requires_owner_match():
    owner_id = f"trace-owner-{uuid.uuid4().hex[:8]}"
    run_id = f"trace-run-{uuid.uuid4().hex[:8]}"
    session_id = f"trace-session-{uuid.uuid4().hex[:8]}"

    run_store = get_run_store()
    run_store.start_run(
        run_id=run_id,
        session_id=session_id,
        agent_id="default",
        model="llama3.2",
        owner_id=owner_id,
    )
    run_store.finish_run(run_id, status="completed")

    with TestClient(app) as client:
        response = client.get(
            f"/logs/traces/run/{run_id}",
            params={"owner_id": "wrong-owner"},
        )

    assert response.status_code == 404


def test_trace_event_requires_matching_owner():
    owner_id = f"trace-event-owner-{uuid.uuid4().hex[:8]}"
    session_id = f"trace-event-session-{uuid.uuid4().hex[:8]}"
    trace_store = get_trace_store()
    event = trace_store.append_event(
        "default",
        session_id,
        "phase_context",
        {"phase": "response", "content": "compiled context"},
        owner_id=owner_id,
    )

    with TestClient(app) as client:
        denied = client.get(
            f"/logs/traces/event/{event['id']}",
            params={"owner_id": "wrong-owner"},
        )
        allowed = client.get(
            f"/logs/traces/event/{event['id']}",
            params={"owner_id": owner_id},
        )

    assert denied.status_code == 200
    assert denied.json()["found"] is False
    assert allowed.status_code == 200
    assert allowed.json()["found"] is True


def test_recent_logs_require_matching_owner_and_normalize_categories():
    owner_id = f"log-owner-{uuid.uuid4().hex[:8]}"
    other_owner = f"log-owner-{uuid.uuid4().hex[:8]}"
    logger = get_logger()
    baseline_ts = logger.recent(limit=1)[0].ts if logger.recent(limit=1) else 0.0

    log(
        "mcp",
        "session-alpha",
        "Connected MCP server",
        level="info",
        owner_id=owner_id,
    )
    log(
        "orch",
        "session-beta",
        "Planner selected tool sequence",
        level="info",
        owner_id=other_owner,
    )

    with TestClient(app) as client:
        response = client.get(
            "/logs/recent",
            params={"owner_id": owner_id, "limit": 20},
        )

    assert response.status_code == 200
    payload = response.json()
    matching = [
        item
        for item in payload["logs"]
        if item["ts"] >= baseline_ts and item.get("owner_id") == owner_id
    ]
    assert matching
    assert all(item["owner_id"] == owner_id for item in matching)
    assert any(item["category"] == "tool" for item in matching)
    assert any(item["meta"].get("source_category") == "mcp" for item in matching)
    assert all(item["message"] != "Planner selected tool sequence" for item in matching)
