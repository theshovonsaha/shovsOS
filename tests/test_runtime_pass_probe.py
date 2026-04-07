from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.semantic_graph import SemanticGraph
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


def _dump(label: str, value) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


@pytest.mark.asyncio
async def test_run_engine_probe_dumps_prompt_passes_and_evals(tmp_path):
    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value='{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\\"query\\": \\"top 3 stocks today\\"}"}}]}'
    )
    adapter.stream = MagicMock(return_value=AsyncIter(["Here are the current stock leaders based on the gathered search results."]))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(
        return_value={
            "strategy": "Search current market leaders first.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Need current market data."}],
            "confidence": 0.96,
        }
    )
    orchestrator.observe_with_context = AsyncMock(
        return_value={
            "status": "finalize",
            "strategy": "Enough evidence gathered from search.",
            "tools": [],
            "notes": "Answer from the web results and close the turn.",
            "confidence": 0.88,
        }
    )
    orchestrator.verify_with_context = AsyncMock(
        return_value={"supported": True, "issues": [], "confidence": 0.98}
    )

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = "Session first message: hi"
    context_engine.compress_exchange = AsyncMock(
        return_value=(
            "compressed memory block",
            [
                {
                    "subject": "User",
                    "predicate": "asked_about",
                    "object": "stocks",
                    "fact": "User asked about stocks",
                }
            ],
            [],
        )
    )

    registry = ToolRegistry()

    async def web_search(query: str, **kwargs):
        return {
            "type": "web_search_results",
            "query": query,
            "results": [
                {"title": "Stock 1", "url": "https://example.com/1", "snippet": "Top stock one"},
                {"title": "Stock 2", "url": "https://example.com/2", "snippet": "Top stock two"},
                {"title": "Stock 3", "url": "https://example.com/3", "snippet": "Top stock three"},
            ],
        }

    registry.register(
        Tool(
            name="web_search",
            description="Search the web.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"],
            },
            handler=web_search,
            response_format="json",
        )
    )

    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    runs = RunStore(db_path=str(tmp_path / "runs.db"))
    traces = FakeTraceStore()
    graph = SemanticGraph(db_path=str(tmp_path / "memory.db"))

    engine = RunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=registry,
        run_store=runs,
        trace_store=traces,
        orchestrator=orchestrator,
        context_engine=context_engine,
        graph=graph,
    )

    request = RunEngineRequest(
        session_id="probe-pass-session",
        owner_id="owner-probe",
        agent_id="default",
        user_message="web search top 3 stocks today",
        model="llama3.2",
        system_prompt="You are Shovs.",
        allowed_tools=("web_search",),
        use_planner=True,
    )

    events = [event async for event in engine.stream(request)]

    run_id = next(event["run_id"] for event in events if event["type"] == "session")
    pass_records = runs.list_passes(run_id)
    checkpoints = runs.list_checkpoints(run_id)
    evals = runs.list_evals(run_id)

    actor_messages = adapter.complete.await_args.kwargs["messages"]
    final_messages = adapter.stream.call_args.kwargs["messages"]

    _dump("ACTOR PROMPT MESSAGES", actor_messages)
    _dump("FINAL RESPONSE MESSAGES", final_messages)
    _dump(
        "PASS RECORDS",
        [
            {
                "phase": record.phase,
                "tool_turn": record.tool_turn,
                "status": record.status,
                "objective": record.objective,
                "strategy": record.strategy,
                "notes": record.notes,
                "selected_tools": record.selected_tools,
                "tool_results": record.tool_results,
                "response_preview": record.response_preview,
                "compiled_context_keys": sorted((record.compiled_context or {}).keys()),
            }
            for record in pass_records
        ],
    )
    _dump(
        "CHECKPOINTS",
        [
            {
                "phase": checkpoint.phase,
                "tool_turn": checkpoint.tool_turn,
                "status": checkpoint.status,
                "strategy": checkpoint.strategy,
                "notes": checkpoint.notes,
                "tools": checkpoint.tools,
                "tool_results": checkpoint.tool_results,
            }
            for checkpoint in checkpoints
        ],
    )
    _dump(
        "EVALS",
        [
            {
                "eval_type": item.eval_type,
                "phase": item.phase,
                "passed": item.passed,
                "score": item.score,
                "detail": item.detail,
                "metadata": item.metadata,
            }
            for item in evals
        ],
    )
    _dump(
        "TRACE EVENTS",
        [
            {
                "event_type": event["event_type"],
                "data": event["data"],
            }
            for event in traces.events
            if event["event_type"]
            in {
                "run_phase",
                "phase_context",
                "tool_call",
                "tool_result",
                "manager_observation",
                "verification_result",
            }
        ],
    )

    assert any(event.get("type") == "tool_call" and event.get("tool_name") == "web_search" for event in events)
    assert any(record.phase == "planning" and record.selected_tools == ["web_search"] for record in pass_records)
    assert any(record.phase == "acting" and record.tool_results for record in pass_records)
    assert any(record.phase == "observation" and record.status == "finalize" for record in pass_records)
    assert any(record.phase == "verification" and record.status == "verified" for record in pass_records)
    assert any(item.eval_type == "response_support" and item.passed for item in evals)