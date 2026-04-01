from fastapi.testclient import TestClient

from api.main import app, consumer_agent_manager, consumer_session_manager, profile_manager


def test_consumer_profile_is_seeded_with_plain_language_defaults():
    with TestClient(app):
        profile = profile_manager.get("consumer")

        assert profile is not None
        assert profile.id == "consumer"
        assert profile.default_use_planner is False
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


def test_consumer_chat_stream_uses_consumer_agent_contract(monkeypatch):
    seen: dict[str, object] = {}

    class FakeAgent:
        async def chat_stream(self, **kwargs):
            seen.update(kwargs)
            yield {"type": "session", "session_id": "consumer-session", "run_id": "run-1"}
            yield {"type": "token", "content": "Hello from consumer."}

    def fake_get_agent_instance(agent_id: str, owner_id: str | None = None):
        seen["requested_agent_id"] = agent_id
        seen["owner_id"] = owner_id
        return FakeAgent()

    monkeypatch.setattr(consumer_agent_manager, "get_agent_instance", fake_get_agent_instance)

    client = TestClient(app)
    response = client.post(
        "/consumer/chat/stream",
        data={"message": "hello", "owner_id": "consumer-chat-owner"},
    )

    assert response.status_code == 200
    assert seen["requested_agent_id"] == "consumer"
    assert seen["agent_id"] == "consumer"
    assert seen["use_planner"] is False
    assert seen["loop_mode"] == "single"
    assert "consumer-session" in response.text
    assert "Hello from consumer." in response.text
