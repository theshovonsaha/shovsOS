from fastapi.testclient import TestClient

from api.main import app, consumer_run_engine, consumer_session_manager, profile_manager


def test_consumer_profile_is_seeded_with_plain_language_defaults():
    with TestClient(app):
        profile = profile_manager.get("consumer")

        assert profile is not None
        assert profile.id == "consumer"
        assert profile.default_use_planner is True
        assert profile.default_loop_mode == "auto"
        assert "query_memory" in profile.tools


def test_consumer_session_route_creates_consumer_scoped_session():
    client = TestClient(app)
    owner_id = "consumer-route-owner"

    response = client.post(
        "/consumer/session",
        json={"owner_id": owner_id, "model": "groq:moonshotai/kimi-k2-instruct"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_id"] == "consumer"

    session = consumer_session_manager.get(payload["id"], owner_id=owner_id)
    assert session is not None
    assert session.agent_id == "consumer"
    assert payload["context_mode"] == "v2"
    assert session.context_mode == "v2"


def test_consumer_chat_stream_uses_managed_run_engine_contract(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_stream(request):
        seen["request"] = request
        yield {"type": "session", "session_id": request.session_id, "run_id": "run-1"}
        yield {"type": "plan", "strategy": "Use web search", "tools": ["web_search"], "confidence": 0.9}
        yield {"type": "tool_call", "tool_name": "web_search", "arguments": {"query": "hello"}}
        yield {"type": "tool_result", "tool_name": "web_search", "success": True, "content": "ok"}
        yield {"type": "token", "content": "Hello from consumer."}
        yield {"type": "done", "session_id": request.session_id, "run_id": "run-1"}

    monkeypatch.setattr(consumer_run_engine, "stream", fake_stream)

    client = TestClient(app)
    response = client.post(
        "/consumer/chat/stream",
        data={"message": "hello", "owner_id": "consumer-chat-owner"},
    )

    assert response.status_code == 200
    request = seen["request"]
    assert request.agent_id == "consumer"
    assert request.owner_id == "consumer-chat-owner"
    assert request.use_planner is True
    assert request.context_mode == "v2"
    assert "query_memory" in request.allowed_tools
    assert "Hello from consumer." in response.text
    assert '"phase": "working"' in response.text
    assert '"type": "done"' in response.text
