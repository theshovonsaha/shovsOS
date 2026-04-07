from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestration.run_store import RunStore
from orchestration.session_manager import SessionManager
from plugins.tool_registry import Tool, ToolRegistry
from run_engine.engine import RunEngine
from run_engine.types import RunEngineRequest


class AsyncIter:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


class FakeTraceStore:
    def __init__(self):
        self.events = []

    def append_event(self, agent_id, session_id, event_type, data, **kwargs):
        event = {
            "agent_id": agent_id,
            "session_id": session_id,
            "event_type": event_type,
            "data": data,
            **kwargs,
        }
        self.events.append(event)
        return event


@pytest.mark.asyncio
async def test_run_engine_uses_resolved_objective_across_planning_and_pass_records(tmp_path):
    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value='{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\\"query\\": \\"wigglebudget.com\\"}"}}]}'
    )
    adapter.stream = MagicMock(return_value=AsyncIter(["Done."]))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(
        return_value={
            "strategy": "Search the target site first.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Need fresh evidence."}],
            "confidence": 0.9,
        }
    )
    orchestrator.observe_with_context = AsyncMock(
        return_value={
            "status": "finalize",
            "strategy": "Enough evidence.",
            "tools": [],
            "notes": "Answer directly.",
            "confidence": 0.8,
        }
    )
    orchestrator.verify_with_context = AsyncMock(
        return_value={"supported": True, "issues": [], "confidence": 0.95}
    )

    registry = ToolRegistry()

    async def web_search(query: str, **kwargs):
        return {"type": "web_search_results", "query": query, "results": [{"title": "wigglebudget"}]}

    registry.register(
        Tool(
            name="web_search",
            description="Search the web.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=web_search,
            response_format="json",
        )
    )

    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    runs = RunStore(db_path=str(tmp_path / "runs.db"))

    session = sessions.create(
        model="llama3.2",
        system_prompt="You are Shovs.",
        agent_id="default",
        session_id="objective-lane-session",
        owner_id="objective-owner",
    )
    sessions.append_message(session.id, "user", "research wigglebudget.com")
    sessions.append_message(session.id, "assistant", "I will gather evidence.")

    engine = RunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=registry,
        run_store=runs,
        trace_store=FakeTraceStore(),
        orchestrator=orchestrator,
        context_engine=None,
        graph=None,
    )

    request = RunEngineRequest(
        session_id=session.id,
        owner_id="objective-owner",
        agent_id="default",
        user_message="try again",
        model="llama3.2",
        system_prompt="You are Shovs.",
        allowed_tools=("web_search",),
        use_planner=True,
    )

    events = [event async for event in engine.stream(request)]

    assert any(event["type"] == "done" for event in events)
    assert orchestrator.plan_with_context.await_args.kwargs["query"] == "research wigglebudget.com"
    assert orchestrator.observe_with_context.await_args.kwargs["query"] == "research wigglebudget.com"
    assert orchestrator.verify_with_context.await_args.kwargs["query"] == "research wigglebudget.com"

    run_id = next(event["run_id"] for event in events if event["type"] == "session")
    pass_records = runs.list_passes(run_id)
    assert pass_records
    assert all(record.objective == "research wigglebudget.com" for record in pass_records if record.phase in {"planning", "response", "verification"})
