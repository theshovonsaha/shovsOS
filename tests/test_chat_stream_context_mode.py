import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_chat_stream_updates_existing_session_context_mode_and_forwards_request(monkeypatch):
    from api.main import app, run_engine, session_manager

    owner_id = "context-mode-owner"
    session = session_manager.create(
        model="llama3.2",
        system_prompt="You are Shovs.",
        agent_id="default",
        owner_id=owner_id,
    )
    session_manager.set_context_mode(session.id, "v1")

    captured = {}

    async def fake_stream(request):
        captured["context_mode"] = request.context_mode
        yield {"type": "session", "session_id": request.session_id, "run_id": "test-run"}
        yield {"type": "done", "session_id": request.session_id, "run_id": "test-run"}

    monkeypatch.setattr(run_engine, "stream", fake_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/chat/stream",
            data={
                "message": "hello",
                "session_id": session.id,
                "agent_id": "default",
                "model": "llama3.2",
                "context_mode": "v3",
                "owner_id": owner_id,
            },
        )

    assert response.status_code == 200
    assert captured["context_mode"] == "v3"
    stored = session_manager.get(session.id, owner_id=owner_id)
    assert stored is not None
    assert stored.context_mode == "v3"
