import pytest
from fastapi.testclient import TestClient

from api.main import app, profile_manager
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
