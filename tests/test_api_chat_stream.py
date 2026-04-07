import pytest
from fastapi.testclient import TestClient

from api.main import app, agent_manager, profile_manager, run_engine
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


def test_chat_stream_legacy_runtime_path_is_compatibility_gated(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_stream(request):
        seen["request"] = request
        yield {"type": "session", "session_id": request.session_id, "run_id": "run-managed-1"}
        yield {"type": "token", "content": "managed"}
        yield {"type": "done", "session_id": request.session_id, "run_id": "run-managed-1"}

    def _should_not_use_legacy(*args, **kwargs):
        raise AssertionError("legacy runtime should be gated off by default")

    monkeypatch.setenv("ALLOW_LEGACY_CHAT_RUNTIME", "false")
    monkeypatch.setattr(run_engine, "stream", fake_stream)
    monkeypatch.setattr(agent_manager, "get_agent_instance", _should_not_use_legacy)

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
    assert "Legacy runtime path is disabled" in response.text
    assert "managed" in response.text


def test_chat_stream_legacy_runtime_path_can_be_enabled(monkeypatch):
    class FakeLegacyAgent:
        async def chat_stream(self, **kwargs):
            yield {"type": "session", "session_id": kwargs.get("session_id"), "run_id": "legacy-run-1"}
            yield {"type": "token", "content": "legacy"}
            yield {"type": "done", "session_id": kwargs.get("session_id"), "run_id": "legacy-run-1"}

    def fake_get_agent_instance(agent_id: str, owner_id: str | None = None):
        return FakeLegacyAgent()

    async def _should_not_use_managed(_request):
        raise AssertionError("managed runtime should not be used when legacy compatibility is enabled")
        yield {}

    monkeypatch.setenv("ALLOW_LEGACY_CHAT_RUNTIME", "true")
    monkeypatch.setattr(agent_manager, "get_agent_instance", fake_get_agent_instance)
    monkeypatch.setattr(run_engine, "stream", _should_not_use_managed)

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
    assert "legacy" in response.text
