from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


class PromptSensitiveRunEngineAdapter:
    def __init__(self):
        self.actor_messages = []
        self.final_messages = []

    async def complete(self, *, messages, **kwargs):
        self.actor_messages.append(messages)
        joined = "\n\n".join(str(item.get("content") or "") for item in messages)
        if (
            "allowed tools below are available in this runtime right now" in joined.lower()
            and "web_search" in joined
            and "top 3 stocks today" in joined.lower()
        ):
            return '{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\\"query\\": \\\"top 3 stocks today\\\"}"}}]}'
        return "I cannot browse the web from here."

    async def stream(self, *, messages, **kwargs):
        self.final_messages.append(messages)
        yield "Here are the top stock results from the gathered search evidence."


class PromptSensitiveLegacyAdapter:
    def __init__(self):
        self.prompt_messages = []
        self.turn_index = 0

    async def stream(self, *, messages, **kwargs):
        self.prompt_messages.append(messages)
        joined = "\n\n".join(str(item.get("content") or "") for item in messages)
        if self.turn_index == 0:
            self.turn_index += 1
            if (
                "tool reality" in joined.lower()
                and "current info rule" in joined.lower()
                and "web_search" in joined
                and "top 3 stocks today" in joined.lower()
            ):
                yield '{"tool":"web_search","arguments":{"query":"top 3 stocks today"}}'
                return
            yield "I do not have access to web search in this environment."
            return
        yield "Here are the top stock results from the gathered search evidence."


@pytest.mark.asyncio
async def test_run_engine_prompt_introspection_prefers_web_search_for_current_stock_query(tmp_path):
    adapter = PromptSensitiveRunEngineAdapter()

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(
        return_value={
            "strategy": "Search current stock leaders first.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Need current market data."}],
            "confidence": 0.92,
        }
    )
    orchestrator.observe_with_context = AsyncMock(
        return_value={
            "status": "finalize",
            "strategy": "Enough evidence.",
            "tools": [],
            "notes": "Use the live search results to answer directly.",
            "confidence": 0.85,
        }
    )
    orchestrator.verify_with_context = AsyncMock(
        return_value={"supported": True, "issues": [], "confidence": 0.97}
    )

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))

    registry = ToolRegistry()

    async def web_search(query: str, **kwargs):
        return json.dumps(
            {
                "type": "web_search_results",
                "query": query,
                "results": [
                    {"title": "Stock 1", "url": "https://example.com/1", "snippet": "Top stock one"},
                    {"title": "Stock 2", "url": "https://example.com/2", "snippet": "Top stock two"},
                    {"title": "Stock 3", "url": "https://example.com/3", "snippet": "Top stock three"},
                ],
            }
        )

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
    registry.register(
        Tool(
            name="rag_search",
            description="Search session memory.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=AsyncMock(return_value="{}"),
            response_format="json",
        )
    )
    registry.register(
        Tool(
            name="file_create",
            description="Create a file.",
            parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
            handler=AsyncMock(return_value="created"),
        )
    )

    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=registry,
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=FakeTraceStore(),
        orchestrator=orchestrator,
        context_engine=context_engine,
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="inside-run-engine",
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

    assert any(event.get("type") == "tool_call" and event.get("tool_name") == "web_search" for event in events)
    assert not any(event.get("tool_name") == "file_create" for event in events if event.get("type") == "tool_call")
    actor_prompt = "\n\n".join(str(item.get("content") or "") for item in adapter.actor_messages[0])
    assert "allowed tools below are available in this runtime right now" in actor_prompt.lower()
    assert "web_search" in actor_prompt
    reminder_prompt = "\n\n".join(str(item.get("content") or "") for item in adapter.final_messages[0])
    assert "do not claim browsing is unavailable" in reminder_prompt.lower()


@pytest.mark.asyncio
async def test_legacy_prompt_introspection_prefers_web_search_for_current_stock_query(tmp_path):
    adapter = PromptSensitiveLegacyAdapter()

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="web_search",
            description="Search the web.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            handler=AsyncMock(
                return_value='{"type":"web_search_results","results":[{"title":"Stock 1","url":"https://example.com/1"}]}'
            ),
        )
    )
    registry.register(
        Tool(
            name="rag_search",
            description="Search session memory.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=AsyncMock(return_value="{}"),
        )
    )
    registry.register(
        Tool(
            name="file_create",
            description="Create a file.",
            parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
            handler=AsyncMock(return_value="created"),
        )
    )

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

        events = [
            event
            async for event in core.chat_stream(
                "top 3 stocks today",
                session_id=session_id,
                use_planner=False,
                forced_tools=["web_search", "rag_search", "file_create"],
                max_tool_calls=1,
            )
        ]

    assert any(event.get("type") == "tool_call" and event.get("tool_name") == "web_search" for event in events)
    assert not any(event.get("tool_name") == "file_create" for event in events if event.get("type") == "tool_call")
    first_prompt = "\n\n".join(str(item.get("content") or "") for item in adapter.prompt_messages[0])
    assert "tool reality" in first_prompt.lower()
    assert "current info rule" in first_prompt.lower()
    assert "top 3 stocks today" in first_prompt.lower()
