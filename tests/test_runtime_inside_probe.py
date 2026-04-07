from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.trace_store import get_trace_store
from engine.core import AgentCore
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


def _register_probe_tools(registry: ToolRegistry) -> None:
    registry.register(
        Tool(
            name="web_search",
            description="Search the web.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=AsyncMock(
                return_value=json.dumps(
                    {
                        "type": "web_search_results",
                        "results": [
                            {"title": "Stock 1", "url": "https://example.com/1", "snippet": "Top stock one"},
                            {"title": "Stock 2", "url": "https://example.com/2", "snippet": "Top stock two"},
                            {"title": "Stock 3", "url": "https://example.com/3", "snippet": "Top stock three"},
                        ],
                    }
                )
            ),
            response_format="json",
        )
    )
    registry.register(
        Tool(
            name="rag_search",
            description="Search session memory.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=AsyncMock(return_value=json.dumps({"type": "rag_search_results", "results": []})),
            response_format="json",
        )
    )
    registry.register(
        Tool(
            name="file_create",
            description="Create a file.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            },
            handler=AsyncMock(return_value="created"),
        )
    )


class InsideRunEngineAdapter:
    def __init__(self):
        self.actor_messages: list[list[dict[str, str]]] = []
        self.final_messages: list[list[dict[str, str]]] = []

    async def complete(self, *, messages, **kwargs):
        self.actor_messages.append(messages)
        joined = "\n\n".join(str(item.get("content") or "") for item in messages)
        if (
            "allowed tools below are available in this runtime right now" in joined.lower()
            and "top 3 stocks today" in joined.lower()
            and "web_search" in joined
        ):
            return '{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\\"query\\": \\\"top 3 stocks today\\\"}"}}]}'
        return ""

    async def stream(self, *, messages, **kwargs):
        self.final_messages.append(messages)
        last_user = next(
            (str(item.get("content") or "") for item in reversed(messages) if item.get("role") == "user"),
            "",
        )
        if "top 3 stocks today" in last_user.lower():
            yield "Here are the top stock results from the gathered search evidence."
            return
        yield "Hello."


class InsideLegacyAdapter:
    def __init__(self):
        self.prompt_messages: list[list[dict[str, str]]] = []
        self.turn_index = 0

    async def stream(self, *, messages, **kwargs):
        self.prompt_messages.append(messages)
        joined = "\n\n".join(str(item.get("content") or "") for item in messages)
        if self.turn_index == 0:
            self.turn_index += 1
            yield "Hello."
            return
        if self.turn_index == 1:
            self.turn_index += 1
            if (
                "tool reality" in joined.lower()
                and "current info rule" in joined.lower()
                and "top 3 stocks today" in joined.lower()
                and "web_search" in joined
            ):
                yield '{"tool":"web_search","arguments":{"query":"top 3 stocks today"}}'
                return
            yield "I cannot browse the web from here."
            return
        yield "Here are the top stock results from the gathered search evidence."


def _session_events(session_id: str, limit: int = 120) -> list[dict]:
    store = get_trace_store()
    index_events = store.list_events(limit=limit, session_id=session_id)
    return [store.get_event(item["id"]) or item for item in reversed(index_events)]


def _event_data(events: list[dict], event_type: str) -> list[dict]:
    return [event.get("data") or {} for event in events if event.get("event_type") == event_type]


@pytest.mark.asyncio
async def test_run_engine_inside_probe_captures_prompt_and_uses_web_search_after_greeting(tmp_path):
    adapter = InsideRunEngineAdapter()
    traces = FakeTraceStore()

    async def _plan_with_context(query: str, **kwargs):
        normalized = query.strip().lower()
        if normalized == "hi":
            return {"strategy": "No tools needed for a trivial conversational turn.", "tools": [], "confidence": 0.9}
        return {
            "strategy": "Search current stock leaders first.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Need current market data."}],
            "confidence": 0.95,
        }

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(side_effect=_plan_with_context)
    orchestrator.observe_with_context = AsyncMock(
        return_value={
            "status": "finalize",
            "strategy": "Enough evidence.",
            "tools": [],
            "notes": "Use the gathered search evidence in the final answer.",
            "confidence": 0.86,
        }
    )
    orchestrator.verify_with_context = AsyncMock(
        return_value={"supported": True, "issues": [], "confidence": 0.97}
    )

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))

    registry = ToolRegistry()
    _register_probe_tools(registry)

    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=registry,
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=traces,
        orchestrator=orchestrator,
        context_engine=context_engine,
    )

    session_id = "inside-run-engine-session"
    _ = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id=session_id,
                owner_id="owner",
                agent_id="default",
                user_message="hi",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("web_search", "rag_search", "file_create"),
                use_planner=True,
            )
        )
    ]
    second_events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id=session_id,
                owner_id="owner",
                agent_id="default",
                user_message="web search top 3 stocks today",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("web_search", "rag_search", "file_create"),
                use_planner=True,
            )
        )
    ]

    assert any(event.get("type") == "tool_call" and event.get("tool_name") == "web_search" for event in second_events)
    assert not any(event.get("tool_name") in {"file_create", "rag_search"} for event in second_events if event.get("type") == "tool_call")

    actor_prompt = "\n\n".join(str(item.get("content") or "") for item in adapter.actor_messages[-1])
    assert "allowed tools below are available in this runtime right now" in actor_prompt.lower()
    assert "session first message: hi" in actor_prompt.lower()
    assert "top 3 stocks today" in actor_prompt.lower()

    second_run_id = next(event["run_id"] for event in second_events if event["type"] == "session")
    second_phase_contexts = [
        event["data"] for event in traces.events if event.get("run_id") == second_run_id and event.get("event_type") == "phase_context"
    ]
    assert any(item["phase"] == "planning" for item in second_phase_contexts)
    assert any(
        item["phase"] == "response" and any(entry.get("item_id") == "session_anchor" for entry in item.get("included", []))
        for item in second_phase_contexts
    )


@pytest.mark.asyncio
async def test_legacy_inside_probe_captures_prompt_and_uses_web_search_after_greeting():
    adapter = InsideLegacyAdapter()

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    registry = ToolRegistry()
    _register_probe_tools(registry)

    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=registry,
        orchestrator=None,
    )

    session_id = f"inside_legacy_{uuid.uuid4().hex[:8]}"
    with patch("engine.core.VectorEngine") as mock_ve:
        ve = MagicMock()
        ve.query = AsyncMock(return_value=[])
        ve.index = AsyncMock(return_value=None)
        mock_ve.return_value = ve

        _ = [event async for event in core.chat_stream("hi", session_id=session_id, use_planner=False)]
        second_events = [
            event
            async for event in core.chat_stream(
                "top 3 stocks today",
                session_id=session_id,
                use_planner=False,
                forced_tools=["web_search", "rag_search", "file_create"],
                max_tool_calls=1,
            )
        ]

    assert any(event.get("type") == "tool_call" and event.get("tool_name") == "web_search" for event in second_events)
    assert not any(event.get("tool_name") in {"file_create", "rag_search"} for event in second_events if event.get("type") == "tool_call")

    trace_events = _session_events(session_id)
    prompts = _event_data(trace_events, "llm_prompt")
    second_prompt = prompts[-2]
    first_pass_messages = second_prompt["messages"]
    joined_prompt = "\n\n".join(str(item.get("content") or "") for item in first_pass_messages)
    assert "tool reality" in joined_prompt.lower()
    assert "current info rule" in joined_prompt.lower()
    assert "top 3 stocks today" in joined_prompt.lower()
    assert any("Hello." in str(item.get("content") or "") for item in first_pass_messages if item.get("role") == "assistant")