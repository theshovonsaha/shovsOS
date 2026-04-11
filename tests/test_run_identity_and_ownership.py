import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from config.trace_store import get_trace_store
from engine.core import AgentCore
from orchestration.agent_profiles import AgentProfile, ProfileManager
from orchestration.run_store import get_run_store
from orchestration.session_manager import SessionManager
from plugins.tool_registry import ToolRegistry


class AsyncIter:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


@pytest.mark.asyncio
async def test_chat_stream_persists_run_identity_and_trace_filters():
    adapter = MagicMock()
    adapter.stream = MagicMock(side_effect=lambda *a, **k: AsyncIter(["Plain response."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    session_manager = SessionManager()
    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=session_manager,
        tool_registry=ToolRegistry(),
        orchestrator=None,
    )

    owner_id = f"owner_{uuid.uuid4().hex[:8]}"
    session_id = f"run_identity_{uuid.uuid4().hex[:8]}"
    events = [event async for event in core.chat_stream("hello there", session_id=session_id, owner_id=owner_id, use_planner=False)]

    session_event = next(event for event in events if event["type"] == "session")
    run_id = session_event["run_id"]

    stored = get_run_store().get(run_id)
    assert stored is not None
    assert stored.session_id == session_id
    assert stored.owner_id == owner_id
    assert stored.status == "completed"
    artifacts = get_run_store().list_artifacts(run_id)
    assert any(item.artifact_type == "assistant_response" for item in artifacts)

    trace_store = get_trace_store()
    run_events = trace_store.list_events(limit=40, run_id=run_id, owner_id=owner_id)
    assert run_events
    assert all(event["run_id"] == run_id for event in run_events)
    assert all(event["owner_id"] == owner_id for event in run_events)


def test_owner_scoped_sessions_and_profiles():
    owner_a = "tenant_a"
    owner_b = "tenant_b"

    sessions = SessionManager(max_sessions=10, db_path=f"/tmp/sessions_{uuid.uuid4().hex}.db")
    session_a = sessions.create(model="llama3.2", system_prompt="a", owner_id=owner_a)
    session_b = sessions.create(model="llama3.2", system_prompt="b", owner_id=owner_b)

    listed_a = sessions.list_sessions(owner_id=owner_a)
    assert {item["id"] for item in listed_a} == {session_a.id}
    assert sessions.get(session_a.id, owner_id=owner_a) is not None
    assert sessions.get(session_b.id, owner_id=owner_a) is None

    profiles = ProfileManager(db_path=f"/tmp/agents_{uuid.uuid4().hex}.db")
    custom = AgentProfile(
        id="custom-owner-profile",
        owner_id=owner_a,
        name="Owner A profile",
        description="Scoped profile",
        tools=["web_search"],
    )
    profiles.create(custom)

    visible_to_owner = {profile.id for profile in profiles.list_all(owner_id=owner_a)}
    visible_to_other = {profile.id for profile in profiles.list_all(owner_id=owner_b)}

    assert "custom-owner-profile" in visible_to_owner
    assert "custom-owner-profile" not in visible_to_other
    assert profiles.get("custom-owner-profile", owner_id=owner_a) is not None
    assert profiles.get("custom-owner-profile", owner_id=owner_b) is None


@pytest.mark.asyncio
async def test_api_requires_owner_and_denies_cross_owner_session_access():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        missing = await ac.post("/sessions", json={"agent_id": "default", "model": "llama3.2"})
        assert missing.status_code == 400

        create = await ac.post(
            "/sessions",
            json={"agent_id": "default", "model": "llama3.2", "owner_id": "owner-a"},
        )
        assert create.status_code == 200
        session_id = create.json()["id"]

        allowed = await ac.get(f"/sessions/{session_id}", params={"owner_id": "owner-a"})
        denied = await ac.get(f"/sessions/{session_id}", params={"owner_id": "owner-b"})

        assert allowed.status_code == 200
        assert allowed.json()["agent_id"] == "default"
        assert denied.status_code == 404

        await ac.delete(f"/sessions/{session_id}", params={"owner_id": "owner-a"})
