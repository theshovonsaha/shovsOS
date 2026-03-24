from unittest.mock import AsyncMock, MagicMock

import pytest

from config.trace_store import get_trace_store
from engine.core import AgentCore
from engine.manifest_parser import Manifest, ManifestParser
from memory.semantic_graph import SemanticGraph
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
async def test_chat_stream_blocks_ungrounded_compression_fact_from_deterministic_memory(tmp_path):
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncIter(["Plain response."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=(
        "ctx",
        [{
            "subject": "stock",
            "predicate": "symbol",
            "object": "HYDR",
            "fact": "stock symbol HYDR",
            "key": "stock symbol",
        }],
        [],
    ))
    context_engine.set_adapter = MagicMock()

    session_manager = SessionManager(db_path=str(tmp_path / "sessions.db"))
    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=session_manager,
        tool_registry=ToolRegistry(),
        orchestrator=None,
    )
    core.graph = SemanticGraph(db_path=str(tmp_path / "memory.db"))

    owner_id = "owner-state-integrity"
    session_id = "state-integrity-1"
    events = [
        event
        async for event in core.chat_stream(
            "research the stock HydroGraph Clean Power and tell me if it will go up",
            session_id=session_id,
            owner_id=owner_id,
            use_planner=False,
        )
    ]

    assert any(event["type"] == "done" for event in events)
    assert core.graph.get_current_facts(session_id, owner_id=owner_id) == []
    session = session_manager.get(session_id, owner_id=owner_id)
    assert session is not None
    assert "Candidate: stock symbol HYDR" in session.candidate_context

    run_id = next(event["run_id"] for event in events if event["type"] == "session")
    trace_events = get_trace_store().list_events(limit=20, run_id=run_id, owner_id=owner_id)
    filters = [event for event in trace_events if event["event_type"] == "memory_fact_filter"]
    assert filters
    assert filters[-1]["data"]["blocked_count"] == 1
    checkpoint = get_run_store().latest_checkpoint(run_id)
    assert checkpoint is not None
    assert "stock symbol HYDR" in (checkpoint.candidate_facts or [])


@pytest.mark.asyncio
async def test_chat_stream_allows_user_grounded_fact_into_deterministic_memory(tmp_path):
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncIter(["Plain response."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=(
        "ctx",
        [{
            "subject": "User",
            "predicate": "preferred_name",
            "object": "Shovon",
            "fact": "User preferred_name Shovon",
            "key": "User preferred_name",
        }],
        [],
    ))
    context_engine.set_adapter = MagicMock()

    session_manager = SessionManager(db_path=str(tmp_path / "sessions.db"))
    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=session_manager,
        tool_registry=ToolRegistry(),
        orchestrator=None,
    )
    core.graph = SemanticGraph(db_path=str(tmp_path / "memory.db"))

    owner_id = "owner-state-integrity"
    session_id = "state-integrity-2"
    events = [
        event
        async for event in core.chat_stream(
            "call me Shovon from now on",
            session_id=session_id,
            owner_id=owner_id,
            use_planner=False,
        )
    ]

    assert any(event["type"] == "done" for event in events)
    assert ("User", "preferred_name", "Shovon") in core.graph.get_current_facts(session_id, owner_id=owner_id)


@pytest.mark.asyncio
async def test_manifest_parser_blocks_ungrounded_fact_and_allows_grounded_fact(monkeypatch):
    fake_graph = MagicMock()
    fake_graph.get_current_facts.return_value = set()
    fake_graph.add_triplet = AsyncMock()

    class FakeSemanticGraph:
        def __new__(cls):
            return fake_graph

    monkeypatch.setattr("memory.semantic_graph.SemanticGraph", FakeSemanticGraph)

    parser = ManifestParser()

    await parser.store(
        Manifest(fact=("stock", "symbol", "HYDR")),
        session_id="manifest-session",
        turn=1,
        user_message="research HydroGraph Clean Power",
        grounding_text="",
    )
    fake_graph.add_temporal_fact.assert_not_called()

    await parser.store(
        Manifest(fact=("User", "preferred_name", "Shovon")),
        session_id="manifest-session",
        turn=2,
        user_message="call me Shovon",
        grounding_text="",
    )
    fake_graph.add_temporal_fact.assert_called_once()
