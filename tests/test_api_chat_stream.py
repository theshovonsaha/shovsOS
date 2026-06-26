import pytest
from fastapi.testclient import TestClient

from api.main import app, profile_manager, run_engine
from orchestration.agent_profiles import AgentProfile


@pytest.mark.asyncio
async def test_chat_stream_first_turn_new_agent_does_not_crash_on_missing_session_id():
    profile = AgentProfile(
        id="stream_new_agent_test",
        owner_id="test-owner",
        name="Stream New Agent Test",
        model="llama3.2",
        tools=["web_search"],
    )
    profile_manager.create(profile)

    client = TestClient(app)
    response = client.post(
        "/chat/stream",
        data={
            "message": "hello",
            "agent_id": "stream_new_agent_test",
            "owner_id": "test-owner",
            "model": "llama3.2",
        },
    )

    body = response.text
    assert response.status_code == 200
    assert "local variable 'session_id' referenced before assignment" not in body


def test_chat_stream_ignores_extra_runtime_path_field(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_stream(request):
        seen["request"] = request
        yield {"type": "session", "session_id": request.session_id, "run_id": "run-managed-1"}
        yield {"type": "token", "content": "managed"}
        yield {"type": "done", "session_id": request.session_id, "run_id": "run-managed-1"}

    monkeypatch.setattr(run_engine, "stream", fake_stream)

    client = TestClient(app)
    response = client.post(
        "/chat/stream",
        data={
            "message": "hello",
            "owner_id": "runtime-gate-owner",
            "runtime_path": "legacy",
        },
    )

    assert response.status_code == 200
    assert seen.get("request") is not None
    assert "managed" in response.text


def test_api_prefixed_chat_stream_reaches_run_engine(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_stream(request):
        seen["request"] = request
        yield {"type": "session", "session_id": request.session_id, "run_id": "run-api-prefix-1"}
        yield {"type": "token", "content": "api-prefixed"}
        yield {"type": "done", "session_id": request.session_id, "run_id": "run-api-prefix-1"}

    monkeypatch.setattr(run_engine, "stream", fake_stream)

    client = TestClient(app)
    response = client.post(
        "/api/chat/stream",
        data={
            "message": "hello through frontend path",
            "owner_id": "api-prefix-owner",
            "model": "llama3.2",
        },
    )

    assert response.status_code == 200
    assert seen.get("request") is not None
    assert "api-prefixed" in response.text


def test_runtime_health_reports_frontend_relevant_features():
    client = TestClient(app)

    response = client.get("/api/runtime/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["runtime"] == "run_engine"
    assert payload["tools"]["image_generate"] is True
    assert payload["features"]["harness_lab"] is True
    assert payload["features"]["api_prefix_compatibility"] is True


def test_chat_stream_ignores_unknown_runtime_path_field(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_stream(request):
        seen["request"] = request
        yield {"type": "session", "session_id": request.session_id, "run_id": "run-managed-1"}
        yield {"type": "token", "content": "managed"}
        yield {"type": "done", "session_id": request.session_id, "run_id": "run-managed-1"}

    monkeypatch.setattr(run_engine, "stream", fake_stream)

    client = TestClient(app)
    response = client.post(
        "/chat/stream",
        data={
            "message": "hello",
            "owner_id": "runtime-gate-owner",
            "runtime_path": "mystery-path",
        },
    )

    assert response.status_code == 200
    assert seen.get("request") is not None
    assert "managed" in response.text
