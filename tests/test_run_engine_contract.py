import json
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engine.context_schema import ContextItem, ContextKind, ContextPhase
from memory.semantic_graph import SemanticGraph
from orchestration.run_store import RunStore
from orchestration.session_manager import SessionManager
from plugins.tool_registry import Tool, ToolRegistry
from plugins.tools import register_tools
from run_engine.context_packets import PacketBuildInputs, build_phase_packet
from run_engine.engine import RunEngine, _extract_urls_from_tool_results
from run_engine.tool_contract import canonical_tool_result
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
async def test_run_engine_multi_turn_profile_converges_current_facts_over_long_chat(tmp_path):
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncIter(["Acknowledged."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    runs = RunStore(db_path=str(tmp_path / "runs.db"))
    traces = FakeTraceStore()
    graph = SemanticGraph(db_path=str(tmp_path / "memory.db"))

    engine = RunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=ToolRegistry(),
        run_store=runs,
        trace_store=traces,
        orchestrator=None,
        context_engine=context_engine,
        graph=graph,
    )

    session_id = "run-engine-transcript"
    owner_id = "owner-transcript"
    transcript = [
        "Hi, I'm Shovon. I work as a System Development Specialist at City of Toronto.",
        "I'm building an open source project called shovsOS.",
        "I use Cursor as my main editor right now.",
        "My current project budget is $5k.",
        "I'm based in Toronto, but planning to move to Berlin in June.",
        "I have 5 years experience at IBM and City of Toronto.",
        "Actually, update: I switched from Cursor to VS Code last week.",
        "Correction: my budget is $3k, not $5k.",
        "I'm not moving to Berlin anymore, staying in Toronto.",
        "My role is now focused on AI integration, not just enterprise apps.",
        "Actually I might increase budget to $4k next month, but keep $3k for now.",
        "Note that as a candidate, not a fact yet.",
    ]

    for message in transcript:
        request = RunEngineRequest(
            session_id=session_id,
            owner_id=owner_id,
            agent_id="default",
            user_message=message,
            model="llama3.2",
            system_prompt="You are Shovs.",
            allowed_tools=(),
            use_planner=False,
        )
        events = [event async for event in engine.stream(request)]
        assert any(event["type"] == "done" for event in events)

    current_facts = set(graph.get_current_facts(session_id, owner_id=owner_id))

    assert ("User", "preferred_name", "Shovon") in current_facts
    assert ("User", "professional_role", "System Development Specialist") in current_facts
    assert ("User", "current_employer", "City of Toronto") in current_facts
    assert ("User", "current_project", "shovsOS") in current_facts
    assert ("User", "preferred_editor", "VS Code") in current_facts
    assert ("Task", "budget_limit", "$3k") in current_facts
    assert ("User", "location", "Toronto") in current_facts
    assert ("User", "professional_focus", "AI integration") in current_facts

    timeline = graph.list_temporal_facts(session_id, owner_id=owner_id, limit=80)
    superseded = {(item["predicate"], item["object"]) for item in timeline if item["status"] == "superseded"}
    assert ("preferred_editor", "Cursor") in superseded
    assert ("budget_limit", "$5k") in superseded
    assert ("location", "Toronto,") not in current_facts

    phase_context_events = [event for event in traces.events if event["event_type"] == "phase_context"]
    assert phase_context_events
    assert any("Memory Authority" in event["data"].get("content", "") for event in phase_context_events)
    assert any("Contradiction Policy:" in event["data"].get("content", "") for event in phase_context_events)


@pytest.mark.asyncio
async def test_run_engine_trivial_greeting_bypasses_cognitive_lanes_with_history(tmp_path):
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncIter(["The model should not be called."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    session = sessions.create(
        model="llama3.2",
        system_prompt="You are Shovs.",
        agent_id="default",
        session_id="trivial-hi-history",
        owner_id="owner-trivial",
    )
    sessions.append_message(session.id, "user", "hi")
    sessions.append_message(session.id, "assistant", "Hi.")
    sessions.append_message(session.id, "user", "hi")
    sessions.append_message(session.id, "assistant", "Hi.")

    traces = FakeTraceStore()
    engine = RunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=traces,
        orchestrator=None,
        context_engine=context_engine,
        graph=SemanticGraph(db_path=str(tmp_path / "memory.db")),
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id=session.id,
                owner_id="owner-trivial",
                agent_id="default",
                user_message="hi",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("web_search", "query_memory"),
                use_planner=True,
            )
        )
    ]

    assert "".join(event.get("content", "") for event in events if event.get("type") == "token") == "Hi."
    assert any(event.get("type") == "done" for event in events)
    adapter.stream.assert_not_called()
    assert not any(event.get("type") == "conversation_tension" for event in events)
    assert not any(event["event_type"] == "conversation_tension" for event in traces.events)
    assert not any(event["event_type"] == "deterministic_fact_extractor" for event in traces.events)
    assert any(
        event["event_type"] == "route_decision"
        and event["data"].get("route_type") == "trivial_chat"
        for event in traces.events
    )


@pytest.mark.asyncio
async def test_run_engine_executes_plan_tool_and_streams_response(tmp_path):
    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value='{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\\"query\\": \\"find run engine\\"}"}}]}'
    )
    adapter.stream = MagicMock(return_value=AsyncIter(["Final ", "answer."]))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(
        return_value={
            "strategy": "Search first, then answer.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Need evidence."}],
            "confidence": 0.9,
        }
    )
    orchestrator.observe_with_context = AsyncMock(
        return_value={
            "status": "finalize",
            "strategy": "Enough evidence.",
            "tools": [],
            "notes": "Use the gathered web evidence in the final answer.",
            "confidence": 0.8,
        }
    )
    orchestrator.verify_with_context = AsyncMock(
        return_value={"supported": True, "issues": [], "confidence": 0.95}
    )

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(
        return_value=(
            "ctx",
            [
                {
                    "subject": "General",
                    "predicate": "working_on",
                    "object": "Secret roadmap",
                    "fact": "General working_on Secret roadmap",
                }
            ],
            [],
        )
    )

    registry = ToolRegistry()

    async def web_search(query: str, **kwargs):
        return {"type": "web_search_results", "results": [{"title": "Run Engine", "url": "https://example.com"}], "query": query}

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
    graph.add_temporal_fact("run-engine-1", "User", "preferred_name", "Shovon", turn=1, owner_id="owner-run-engine")

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
    # The runtime resolves its compression engine through the governor; pin the
    # mock there so compress_exchange's stubbed return value (including the
    # blockable "General working_on …" fact) actually drives the assertions.
    engine._context_governor._v3_engine = context_engine

    request = RunEngineRequest(
        session_id="run-engine-1",
        owner_id="owner-run-engine",
        agent_id="default",
        user_message="Call me Alex. Research run engine and summarize it.",
        model="llama3.2",
        system_prompt="You are Shovs.",
        allowed_tools=("web_search",),
        use_planner=True,
    )

    events = [event async for event in engine.stream(request)]

    event_types = [event["type"] for event in events]
    assert "session" in event_types
    assert "conversation_tension" in event_types
    assert "plan" in event_types
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert "done" in event_types
    assert "".join(event.get("content", "") for event in events if event["type"] == "token") == "Final answer."
    streamed_tool_call = next(event for event in events if event["type"] == "tool_call")
    streamed_tool_result = next(event for event in events if event["type"] == "tool_result")
    assert streamed_tool_call["tool_call_id"]
    assert streamed_tool_result["tool_call_id"] == streamed_tool_call["tool_call_id"]
    assert streamed_tool_result["tool_result_id"]

    run_id = next(event["run_id"] for event in events if event["type"] == "session")
    checkpoint = runs.latest_checkpoint(run_id)
    assert checkpoint is not None
    assert checkpoint.phase in {"response", "observation", "acting", "memory_commit"}

    pass_records = runs.list_passes(run_id)
    checkpoints = runs.list_checkpoints(run_id)
    assert {record.phase for record in pass_records} >= {"planning", "acting", "observation", "response", "verification"}
    assert any(record.phase == "planning" and record.selected_tools == ["web_search"] for record in pass_records)
    assert any(record.phase == "acting" and record.tool_results for record in pass_records)
    assert any(record.phase == "observation" and record.status == "finalize" for record in pass_records)
    assert any(record.phase == "verification" and record.status == "verified" for record in pass_records)
    assert any(
        record.phase == "response"
        and record.status == "complete"
        and "Manager status: finalize" in record.notes
        and record.response_preview == "Final answer."
        for record in pass_records
    )
    assert any(
        checkpoint.phase == "response"
        and checkpoint.status == "complete"
        and checkpoint.strategy == "Enough evidence."
        and "Manager status: finalize" in checkpoint.notes
        and "Notes: Use the gathered web evidence in the final answer." in checkpoint.notes
        for checkpoint in checkpoints
    )
    acting_record = next(record for record in pass_records if record.phase == "acting" and record.tool_results)
    assert acting_record.tool_results[0]["status"] in {"ok", "failed"}
    assert "preview" in acting_record.tool_results[0]

    assert orchestrator.plan_with_context.await_args.kwargs["compiled_context"]
    assert orchestrator.verify_with_context.await_args.kwargs["compiled_context"]
    final_messages = adapter.stream.call_args.kwargs["messages"]
    reminder_messages = [
        message["content"]
        for message in final_messages
        if message.get("role") == "system" and "<system-reminder>" in message.get("content", "")
    ]
    assert reminder_messages
    assert "Use the controller handoff below to close the turn." in reminder_messages[0]
    assert "Manager status: finalize" in reminder_messages[0]
    assert "Notes: Use the gathered web evidence in the final answer." in reminder_messages[0]

    session = sessions.get("run-engine-1", owner_id="owner-run-engine")
    assert session is not None
    assert session.full_history[-1]["role"] == "assistant"
    assert session.compressed_context == "ctx"
    assert getattr(session, "candidate_signals", [])
    assert session.candidate_signals[0]["text"] == "General working_on Secret roadmap"
    assert "Candidate: General working_on Secret roadmap" in getattr(session, "candidate_context", "")

    current_facts = graph.get_current_facts("run-engine-1", owner_id="owner-run-engine")
    assert ("User", "preferred_name", "Alex") in current_facts

    assert any(event["event_type"] == "tool_result" for event in traces.events)
    assert any(event["event_type"] == "tool_call" for event in traces.events)
    assert any(event["event_type"] == "run_ledger" for event in traces.events)
    assert any(event["event_type"] == "phase_packet" for event in traces.events)
    tool_result_events = [event for event in traces.events if event["event_type"] == "tool_result"]
    assert any(event["data"].get("status") == "ok" for event in tool_result_events)
    assert any(event["data"].get("tool_call_id") == streamed_tool_call["tool_call_id"] for event in tool_result_events)
    tool_call_events = [event for event in traces.events if event["event_type"] == "tool_call"]
    assert any(event["data"].get("arguments_summary") for event in tool_call_events)
    assert any(event["data"].get("tool") == "web_search" for event in tool_call_events)
    assert any(event["data"].get("tool_call_id") == streamed_tool_call["tool_call_id"] for event in tool_call_events)
    run_ledger_events = [event for event in traces.events if event["event_type"] == "run_ledger"]
    assert any(event["data"]["summary"]["tool_calls"] >= 1 for event in run_ledger_events)
    assert any(event["data"]["summary"]["tool_results"] >= 1 for event in run_ledger_events)
    phase_packet_events = [event for event in traces.events if event["event_type"] == "phase_packet"]
    assert any(event["data"].get("run_ledger", {}).get("version") == "run-ledger-v1" for event in phase_packet_events)
    assert any(event["event_type"] == "deterministic_fact_extractor" for event in traces.events)
    assert any(event["event_type"] == "conversation_tension" for event in traces.events)
    assert any(event["event_type"] == "phase_context" for event in traces.events)
    assert any(event["event_type"] == "manager_observation" for event in traces.events)
    assert any(event["event_type"] == "verification_result" for event in traces.events)
    assert any(event["event_type"] == "memory_fact_filter" for event in traces.events)
    memory_commit_plans = [event for event in traces.events if event["event_type"] == "memory_commit_plan"]
    assert memory_commit_plans
    assert memory_commit_plans[-1]["data"]["conversation_tension"]["dynamic"] is True
    assert memory_commit_plans[-1]["data"]["conversation_tension"]["storage_action"] == "void_previous_store_current"
    assert "conversation_tension" in memory_commit_plans[-1]["data"]["phase_packet"]["included_ids"]
    assert "meta_context" in memory_commit_plans[-1]["data"]["phase_packet"]["excluded_ids"]
    compiled_events = [event for event in traces.events if event["event_type"] == "compiled_context"]
    phase_context_events = [event for event in traces.events if event["event_type"] == "phase_context"]
    assert phase_context_events
    assert any(event["data"].get("phase") == "memory_commit" for event in phase_context_events)
    assert not compiled_events
    assert all(event["data"].get("trace_scope") == "phase_packet" for event in phase_context_events)
    assert all(event["data"].get("canonical_event") == "phase_context" for event in phase_context_events)
    assert all("content" in event["data"] for event in phase_context_events)


@pytest.mark.asyncio
async def test_run_engine_direct_fact_turn_skips_planner_and_tools_when_deterministic_facts_suffice(tmp_path):
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncIter(["You use VS Code."]))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock()
    orchestrator.observe_with_context = AsyncMock()
    orchestrator.verify_with_context = AsyncMock(return_value={"supported": True, "issues": [], "confidence": 0.99})

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    runs = RunStore(db_path=str(tmp_path / "runs.db"))
    traces = FakeTraceStore()
    graph = SemanticGraph(db_path=str(tmp_path / "memory.db"))
    graph.add_temporal_fact("run-engine-fact", "User", "preferred_editor", "VS Code", turn=1, owner_id="owner-fact")

    engine = RunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=ToolRegistry(),
        run_store=runs,
        trace_store=traces,
        orchestrator=orchestrator,
        context_engine=context_engine,
        graph=graph,
    )

    request = RunEngineRequest(
        session_id="run-engine-fact",
        owner_id="owner-fact",
        agent_id="default",
        user_message="What editor do I use now?",
        model="llama3.2",
        system_prompt="You are Shovs.",
        allowed_tools=("query_memory", "web_search"),
        use_planner=True,
    )

    events = [event async for event in engine.stream(request)]

    assert "".join(event.get("content", "") for event in events if event["type"] == "token") == "You use VS Code."
    assert not any(event["type"] == "tool_call" for event in events)
    orchestrator.plan_with_context.assert_not_awaited()
    orchestrator.observe_with_context.assert_not_awaited()
    # Direct-fact-memory-only routes intentionally skip LLM verification —
    # the answer is sourced from the deterministic fact graph (pre-grounded),
    # so there is nothing to ground-check against tool evidence.
    orchestrator.verify_with_context.assert_not_awaited()

    phase_context_events = [event for event in traces.events if event["event_type"] == "phase_context"]
    assert any("deterministic_only" in str(event["data"].get("content", "")) for event in phase_context_events)
    assert any(
        event["data"]["phase"] == "response"
        and any(item["item_id"] == "memory_authority" for item in event["data"]["included"])
        for event in phase_context_events
    )
    assert any(
        any(item["item_id"] == "deterministic_facts" for item in event["data"]["included"])
        for event in phase_context_events
    )


@pytest.mark.asyncio
async def test_run_engine_simple_chat_skips_planner_tools_and_memory_commit(tmp_path):
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncIter(["Hi."]))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock()
    orchestrator.observe_with_context = AsyncMock()
    orchestrator.verify_with_context = AsyncMock(return_value={"supported": True, "issues": [], "confidence": 0.99})

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = "stale source workflow context"
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    traces = FakeTraceStore()
    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=traces,
        orchestrator=orchestrator,
        context_engine=context_engine,
        graph=SemanticGraph(db_path=str(tmp_path / "memory.db")),
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="simple-chat-runtime",
                owner_id="owner-simple-chat",
                agent_id="default",
                user_message="hi again",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("web_search", "web_fetch"),
                use_planner=True,
            )
        )
    ]

    assert any(event["type"] == "done" for event in events)
    assert not any(event["type"] == "tool_call" for event in events)
    orchestrator.plan_with_context.assert_not_awaited()
    context_engine.compress_exchange.assert_not_awaited()
    assert any(
        event["event_type"] == "memory_commit_skipped"
        and event["data"].get("reason") == "low_value_social_turn"
        for event in traces.events
    )


@pytest.mark.asyncio
async def test_run_engine_async_memory_commit_does_not_block_done_event(tmp_path):
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncIter(["Saved that preference."]))

    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_compress_exchange(**_kwargs):
        started.set()
        await release.wait()
        return ("- User prefers concise answers", [], [])

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(side_effect=delayed_compress_exchange)

    traces = FakeTraceStore()
    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=traces,
        orchestrator=None,
        context_engine=context_engine,
        graph=SemanticGraph(db_path=str(tmp_path / "memory.db")),
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="async-memory-runtime",
                owner_id="owner-async-memory",
                agent_id="default",
                user_message="Remember that I prefer concise answers.",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=(),
                use_planner=False,
                memory_commit_mode="async",
            )
        )
    ]

    event_types = [event["type"] for event in events]
    assert event_types.index("memory_commit_deferred") < event_types.index("done")
    assert any(
        event["event_type"] == "memory_commit_deferred"
        for event in traces.events
    )

    release.set()
    pending = list(engine._background_memory_tasks)
    if pending:
        await asyncio.gather(*pending)
    assert context_engine.compress_exchange.await_count == 1


@pytest.mark.asyncio
async def test_run_engine_adaptive_memory_skips_llm_compression_for_ordinary_turn(tmp_path):
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncIter(["Tokyo."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    traces = FakeTraceStore()
    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=traces,
        orchestrator=None,
        context_engine=context_engine,
        graph=SemanticGraph(db_path=str(tmp_path / "memory.db")),
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="adaptive-no-compress",
                owner_id="owner-adaptive-no-compress",
                agent_id="default",
                user_message="What is the capital of Japan?",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=(),
                use_planner=False,
                memory_commit_mode="sync",
            )
        )
    ]

    assert any(event["type"] == "done" for event in events)
    context_engine.compress_exchange.assert_not_awaited()
    assert any(
        event["event_type"] == "llm_compression_skipped"
        and event["data"].get("reason") == "adaptive_gate"
        for event in traces.events
    )


@pytest.mark.asyncio
async def test_run_engine_blocks_memory_commit_on_unsupported_verification(tmp_path):
    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value='{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\\"query\\": \\\"find run engine\\\"}"}}]}'
    )
    adapter.stream = MagicMock(return_value=AsyncIter(["Unsupported answer."]))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(
        return_value={
            "strategy": "Search first, then answer.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Need evidence."}],
            "confidence": 0.9,
        }
    )
    orchestrator.observe_with_context = AsyncMock(
        return_value={
            "status": "finalize",
            "strategy": "Enough evidence.",
            "tools": [],
            "notes": "Answer directly now.",
            "confidence": 0.8,
        }
    )
    orchestrator.verify_with_context = AsyncMock(
        return_value={"supported": False, "issues": ["Claim is not grounded."], "confidence": 0.91}
    )

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    registry = ToolRegistry()

    async def web_search(query: str, **kwargs):
        return {"type": "web_search_results", "results": [{"title": "Run Engine", "url": "https://example.com"}], "query": query}

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

    engine = RunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=registry,
        run_store=runs,
        trace_store=traces,
        orchestrator=orchestrator,
        context_engine=context_engine,
        graph=SemanticGraph(db_path=str(tmp_path / "memory.db")),
    )

    request = RunEngineRequest(
        session_id="run-engine-verify-block",
        owner_id="owner-run-engine",
        agent_id="default",
        user_message="Research run engine and summarize it.",
        model="llama3.2",
        system_prompt="You are Shovs.",
        allowed_tools=("web_search",),
        use_planner=True,
    )

    events = [event async for event in engine.stream(request)]
    run_id = next(event["run_id"] for event in events if event["type"] == "session")

    assert any(event["type"] == "verification_warning" for event in events)
    context_engine.compress_exchange.assert_not_awaited()
    assert any(
        checkpoint.phase == "memory_commit" and checkpoint.status == "blocked_verification"
        for checkpoint in runs.list_checkpoints(run_id)
    )
    assert any(
        event["event_type"] == "memory_commit_skipped"
        and event["data"].get("reason") == "verification_unsupported"
        for event in traces.events
    )


@pytest.mark.asyncio
async def test_run_engine_persists_continuation_state_on_needs_replan(tmp_path):
    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value='{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\\"query\\": \\\"example pricing\\\"}"}}]}'
    )
    adapter.stream = MagicMock(return_value=AsyncIter(["I do not have the pricing page yet."]))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(
        return_value={
            "strategy": "Search first, then fetch pricing.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Need first lead."}],
            "plan_steps": [
                {"id": "step_1", "description": "Search for pricing page", "tool": "web_search", "status": "pending"},
                {"id": "step_2", "description": "Fetch first-party pricing page", "tool": "web_fetch", "status": "pending"},
            ],
            "confidence": 0.9,
        }
    )
    orchestrator.observe_with_context = AsyncMock(
        return_value={
            "status": "finalize",
            "strategy": "Search was not enough.",
            "tools": [],
            "notes": "Need pricing page before final answer.",
            "confidence": 0.7,
        }
    )
    orchestrator.verify_with_context = AsyncMock(
        return_value={
            "supported": False,
            "verdict": "needs_replan",
            "target_state": "gather",
            "missing_evidence": ["first-party pricing page"],
            "issues": ["Pricing page was not fetched."],
            "confidence": 0.2,
        }
    )

    registry = ToolRegistry()

    async def web_search(query: str, **kwargs):
        return {"type": "web_search_results", "results": [{"title": "Pricing", "url": "https://example.com/pricing"}], "query": query}

    registry.register(
        Tool(
            name="web_search",
            description="Search the web.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            handler=web_search,
            response_format="json",
        )
    )
    registry.register(
        Tool(
            name="web_fetch",
            description="Fetch a URL.",
            parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            handler=AsyncMock(return_value='{"type":"web_fetch_result","content":"pricing"}'),
            response_format="json",
        )
    )

    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    engine = RunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=registry,
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=FakeTraceStore(),
        orchestrator=orchestrator,
        context_engine=MagicMock(compress_exchange=AsyncMock(return_value=("ctx", [], []))),
        graph=SemanticGraph(db_path=str(tmp_path / "memory.db")),
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="run-engine-needs-replan",
                owner_id="owner-run-engine",
                agent_id="default",
                user_message="Research example.com pricing.",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("web_search", "web_fetch"),
                use_planner=True,
                max_tool_calls=1,
            )
        )
    ]

    assert any(event["type"] == "replan_recommended" for event in events)
    session = sessions.get("run-engine-needs-replan", owner_id="owner-run-engine")
    assert session is not None
    assert session.continuation_state["reason"] == "verification_needs_replan"
    assert session.continuation_state["target_state"] == "gather"
    assert "first-party pricing page" in session.continuation_state["missing_slots"]
    assert session.continuation_state["pending_steps"]


@pytest.mark.asyncio
async def test_scripted_model_resume_run_uses_continuation_state_across_phases(tmp_path):
    from orchestration.orchestrator import AgenticOrchestrator

    class ScriptedModel:
        def __init__(self):
            self.complete_prompts = []
            self.stream_prompts = []

        async def complete(self, *, messages, **kwargs):
            joined = "\n\n".join(str(item.get("content") or "") for item in messages)
            self.complete_prompts.append(joined)
            if "[Shovs Orchestrator]" in joined:
                assert "Continuation State" in joined
                assert "https://example.com/pricing" in joined
                return (
                    '{"strategy":"Resume by fetching the pricing page.",'
                    '"tools":[{"name":"web_fetch","priority":"high","reason":"Continuation requires exact pricing evidence.","target_argument_clue":"https://example.com/pricing"}],'
                    '"plan_steps":[{"id":"step_1","description":"Fetch exact pricing page","tool":"web_fetch","status":"pending","risk":"read_only"}],'
                    '"risk_tier":"read_only","confidence":0.95}'
                )
            if "You are the Shovs Run Engine actor" in joined:
                assert "Continuation State" in joined
                assert "Planner argument hints" in joined
                return '{"tool_calls":[{"function":{"name":"web_fetch","arguments":"{\\"url\\":\\"https://example.com/pricing\\"}"}}]}'
            if "[Shovs Verification Layer]" in joined:
                assert "Example pricing page" in joined
                return '{"supported":true,"verdict":"supported","issues":[],"confidence":0.98}'
            return "{}"

        async def stream(self, *, messages, **kwargs):
            self.stream_prompts.append("\n\n".join(str(item.get("content") or "") for item in messages))
            yield "The pricing page shows the Starter plan is $10/month."

    adapter = ScriptedModel()
    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    session = sessions.create(
        model="llama3.2",
        system_prompt="You are Shovs.",
        agent_id="default",
        session_id="scripted-resume",
        owner_id="owner-run-engine",
    )
    sessions.update_continuation_state(
        session.id,
        {
            "reason": "verification_needs_replan",
            "objective": "Research example.com pricing.",
            "next_action": "Fetch https://example.com/pricing",
            "pending_steps": [
                {"id": "step_1", "description": "Fetch exact pricing page", "tool": "web_fetch", "status": "pending"}
            ],
            "missing_slots": ["first-party pricing page"],
            "evidence_summary": ["- web_search [ok]: Found https://example.com/pricing"],
        },
        owner_id="owner-run-engine",
    )

    registry = ToolRegistry()

    async def web_fetch(url: str, **kwargs):
        return {
            "type": "web_fetch_result",
            "url": url,
            "title": "Example pricing page",
            "content": "Example pricing page " + ("lists Starter at $10/month. " * 12),
        }

    registry.register(
        Tool(
            name="web_fetch",
            description="Fetch a URL.",
            parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            handler=web_fetch,
            response_format="json",
        )
    )

    traces = FakeTraceStore()
    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))
    context_engine.set_adapter = MagicMock()

    engine = RunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=registry,
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=traces,
        orchestrator=AgenticOrchestrator(adapter=adapter),
        context_engine=context_engine,
        graph=SemanticGraph(db_path=str(tmp_path / "memory.db")),
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id=session.id,
                owner_id="owner-run-engine",
                agent_id="default",
                user_message="continue",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("web_fetch",),
                use_planner=True,
            )
        )
    ]

    assert any(event.get("type") == "tool_call" and event.get("tool_name") == "web_fetch" for event in events)
    tool_call = next(event for event in events if event.get("type") == "tool_call")
    assert tool_call["arguments"] == {"url": "https://example.com/pricing"}
    refreshed = sessions.get(session.id, owner_id="owner-run-engine")
    assert refreshed is not None
    assert refreshed.continuation_state == {}
    phase_contexts = [event["data"] for event in traces.events if event["event_type"] == "phase_context"]
    assert any(
        item["item_id"] == "continuation_state"
        for packet in phase_contexts
        for item in packet.get("included", [])
    )


@pytest.mark.asyncio
async def test_run_engine_observation_finalize_does_not_emit_followup_plan(tmp_path):
    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value='{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\\"query\\": \\\"find run engine\\\"}"}}]}'
    )
    adapter.stream = MagicMock(return_value=AsyncIter(["Final answer."]))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(
        return_value={
            "strategy": "Search first.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Need evidence."}],
            "confidence": 0.9,
        }
    )
    orchestrator.observe_with_context = AsyncMock(
        return_value={
            "status": "finalize",
            "strategy": "Enough evidence gathered.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Do not actually continue."}],
            "notes": "Answer directly now.",
            "confidence": 0.83,
        }
    )
    orchestrator.verify_with_context = AsyncMock(
        return_value={"supported": True, "issues": [], "confidence": 0.95}
    )

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    registry = ToolRegistry()

    async def web_search(query: str, **kwargs):
        return {"type": "web_search_results", "results": [{"title": "Run Engine", "url": "https://example.com"}], "query": query}

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

    engine = RunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=registry,
        run_store=runs,
        trace_store=traces,
        orchestrator=orchestrator,
        context_engine=context_engine,
        graph=SemanticGraph(db_path=str(tmp_path / "memory.db")),
    )

    request = RunEngineRequest(
        session_id="run-engine-observation-normalize",
        owner_id="owner-run-engine",
        agent_id="default",
        user_message="Research run engine and summarize it.",
        model="llama3.2",
        system_prompt="You are Shovs.",
        allowed_tools=("web_search",),
        use_planner=True,
    )

    events = [event async for event in engine.stream(request)]
    plans = [event for event in events if event["type"] == "plan"]
    run_id = next(event["run_id"] for event in events if event["type"] == "session")
    pass_records = runs.list_passes(run_id)

    assert len(plans) == 1
    assert plans[0]["tools"] == ["web_search"]
    assert any(
        record.phase == "observation" and record.status == "finalize" and record.selected_tools == []
        for record in pass_records
    )
    assert any(
        event["event_type"] == "manager_observation"
        and event["data"].get("status") == "finalize"
        and event["data"].get("raw_status") == "finalize"
        and event["data"].get("tools") == []
        for event in traces.events
    )


@pytest.mark.asyncio
async def test_run_engine_reuses_duplicate_web_result_without_second_network_call(tmp_path):
    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value='{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\\"query\\": \\\"find run engine\\\"}"}}]}'
    )
    adapter.stream = MagicMock(return_value=AsyncIter(["Final answer."]))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(
        return_value={
            "strategy": "Search first.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Need evidence."}],
            "confidence": 0.9,
        }
    )
    orchestrator.observe_with_context = AsyncMock(
        side_effect=[
            {
                "status": "continue",
                "strategy": "Repeat search to simulate actor drift.",
                "tools": [{"name": "web_search", "priority": "high", "reason": "Actor repeats same query."}],
                "notes": "Continue once.",
                "confidence": 0.6,
            },
            {
                "status": "finalize",
                "strategy": "Cached evidence is enough.",
                "tools": [],
                "notes": "Answer from cached result.",
                "confidence": 0.9,
            },
        ]
    )
    orchestrator.verify_with_context = AsyncMock(
        return_value={"supported": True, "issues": [], "confidence": 0.95}
    )

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    call_count = 0

    async def web_search(query: str, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "type": "web_search_results",
            "query": query,
            "results": [{"title": "Run Engine", "url": "https://example.com/run-engine"}],
        }

    registry = ToolRegistry()
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

    runs = RunStore(db_path=str(tmp_path / "runs.db"))
    traces = FakeTraceStore()
    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=registry,
        run_store=runs,
        trace_store=traces,
        orchestrator=orchestrator,
        context_engine=context_engine,
        graph=SemanticGraph(db_path=str(tmp_path / "memory.db")),
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="run-engine-duplicate-web-cache",
                owner_id="owner-run-engine",
                agent_id="default",
                user_message="Research run engine and summarize it.",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("web_search",),
                use_planner=True,
                max_tool_calls=3,
            )
        )
    ]

    tool_results = [event for event in events if event.get("type") == "tool_result"]
    assert call_count == 1
    assert len(tool_results) >= 2
    assert tool_results[0]["status"] == "ok"
    assert tool_results[1]["status"] == "cached"
    assert tool_results[1]["cached_duplicate"] is True
    assert not any(
        event["event_type"] == "tool_result"
        and event["data"].get("status") == "failed"
        and "Duplicate web_search" in str(event["data"].get("content_preview") or "")
        for event in traces.events
    )


@pytest.mark.asyncio
async def test_run_engine_stops_when_duplicate_recovery_has_no_new_action(tmp_path):
    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value='{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\\"query\\": \\\"find run engine\\\"}"}}]}'
    )
    adapter.stream = MagicMock(return_value=AsyncIter(["Partial answer."]))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(
        return_value={
            "strategy": "Search first.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Need evidence."}],
            "confidence": 0.9,
        }
    )
    orchestrator.observe_with_context = AsyncMock(
        return_value={
            "status": "continue",
            "strategy": "Repeat search again.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Repeat same query."}],
            "notes": "Continue loop.",
            "confidence": 0.5,
        }
    )
    orchestrator.verify_with_context = AsyncMock(
        return_value={"supported": False, "issues": ["duplicate_loop"], "confidence": 0.3}
    )

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    call_count = 0

    async def web_search(query: str, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "type": "web_search_results",
            "query": query,
            "results": [{"title": "Run Engine", "url": "https://example.com/run-engine"}],
        }

    registry = ToolRegistry()
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

    traces = FakeTraceStore()
    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=registry,
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=traces,
        orchestrator=orchestrator,
        context_engine=context_engine,
        graph=SemanticGraph(db_path=str(tmp_path / "memory.db")),
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="run-engine-duplicate-stagnation",
                owner_id="owner-run-engine",
                agent_id="default",
                user_message="Research run engine and summarize it.",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("web_search",),
                use_planner=True,
                max_tool_calls=4,
            )
        )
    ]

    assert call_count == 1
    assert any(event.get("type") == "loop_stagnation" for event in events)
    assert any(event["event_type"] == "loop_stagnation" for event in traces.events)
    assert orchestrator.observe_with_context.await_count == 2


def test_build_phase_packet_includes_guidance_and_memory_items():
    context_engine = MagicMock()
    context_engine.build_context_items.return_value = [
        ContextItem(
            item_id="memory_fact",
            kind=ContextKind.MEMORY,
            title="Relevant Memory",
            content="Stored fact for this session.",
            source="context_engine",
            priority=40,
            trace_id="memory:fact",
        )
    ]

    session = SimpleNamespace(
        first_message="Original user goal.",
        sliding_window=[
            {"role": "user", "content": "Research the run engine."},
            {"role": "assistant", "content": "I will gather evidence."},
        ],
    )
    request = RunEngineRequest(
        session_id="packet-test",
        owner_id="owner-1",
        agent_id="default",
        user_message="Summarize the run engine.",
        model="llama3.2",
        system_prompt="You are Shovs.",
        allowed_tools=("web_search",),
    )

    packet = build_phase_packet(
        context_engine=context_engine,
        inputs=PacketBuildInputs(
            request=request,
            session=session,
            phase=ContextPhase.ACTING,
            system_prompt=request.system_prompt,
            current_context="compressed context",
            allowed_tools=[{"name": "web_search", "description": "Search the web."}],
            tool_results=[{"tool_name": "web_search", "success": True, "content": "Result payload"}],
            tool_turn=1,
            strategy="Search first, then synthesize.",
            notes="Focus on verified sources.",
            observation_status="continue",
            observation_tools=["web_search"],
        ),
    )

    assert "Manager status: continue" in packet.content
    assert "Preferred next tools: web_search" in packet.content
    assert "Strategy: Search first, then synthesize." in packet.content
    assert "Notes: Focus on verified sources." in packet.content
    assert "Stored fact for this session." in packet.content


def test_build_phase_packet_includes_runtime_attention_lane():
    session = SimpleNamespace(
        first_message="Original user goal.",
        sliding_window=[],
    )
    request = RunEngineRequest(
        session_id="packet-attention",
        owner_id="owner-1",
        agent_id="default",
        user_message="Search ROKU and fetch exact sources.",
        model="llama3.2",
        system_prompt="You are Shovs.",
        allowed_tools=("web_search", "web_fetch"),
    )
    from run_engine.ledger import RunLedger

    ledger = RunLedger(
        run_id="run-attention",
        session_id="packet-attention",
        turn_id="turn-1",
        objective=request.user_message,
        allowed_tools=["web_search", "web_fetch"],
    )
    ledger.set_plan([
        {"id": "step_1", "description": "Search ROKU news", "tool": "web_search", "status": "pending"},
    ])

    packet = build_phase_packet(
        context_engine=None,
        inputs=PacketBuildInputs(
            request=request,
            session=session,
            phase=ContextPhase.ACTING,
            system_prompt=request.system_prompt,
            current_context="",
            allowed_tools=[{"name": "web_search", "description": "Search the web."}],
            tool_results=[],
            run_ledger=ledger,
        ),
    )

    assert "Runtime Attention:" in packet.content
    assert "plan:step_1" in packet.content
    attention_items = [
        item for item in packet.trace["included"]
        if item["item_id"] == "runtime_attention"
    ]
    assert attention_items
    assert attention_items[0]["provenance"]["policy"] == "ledger_phase_weighted"


def test_build_final_messages_includes_controller_reminder(tmp_path):
    engine = RunEngine(
        adapter=MagicMock(),
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=FakeTraceStore(),
    )
    session = SimpleNamespace(
        sliding_window=[
            {"role": "user", "content": "Research the run engine."},
            {"role": "assistant", "content": "I will gather evidence."},
        ]
    )

    messages = engine._build_final_messages(
        session=session,
        system_prompt="You are Shovs.",
        user_message="Summarize the run engine.",
        effective_objective="Summarize the run engine.",
        tool_results=[{"tool_name": "web_search", "success": True, "content": "Result payload"}],
        context_block="Observation State:\nManager status: finalize",
        controller_summary="Manager status: finalize\nStrategy: Enough evidence.\nNotes: Answer directly now.",
    )

    reminder_messages = [
        message["content"]
        for message in messages
        if message.get("role") == "system" and "<system-reminder>" in message.get("content", "")
    ]
    assert reminder_messages
    assert "Use the controller handoff below to close the turn." in reminder_messages[0]
    assert "Manager status: finalize" in reminder_messages[0]
    assert "Strategy: Enough evidence." in reminder_messages[0]
    assert "Prioritize verified exact-domain fetch evidence" not in reminder_messages[0]
    assert "Evidence focus:" in reminder_messages[0]
    assert "- web_search [ok]: Result payload" in reminder_messages[0]


def test_build_final_messages_prioritizes_exact_domain_fetch_evidence(tmp_path):
    engine = RunEngine(
        adapter=MagicMock(),
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=FakeTraceStore(),
    )
    session = SimpleNamespace(sliding_window=[])

    messages = engine._build_final_messages(
        session=session,
        system_prompt="You are Shovs.",
        user_message="Check whether example.com is active and summarize the result.",
        effective_objective="Check whether example.com is active and summarize the result.",
        tool_results=[
            {
                "tool_name": "web_search",
                "success": True,
                "content": "Search result mentions several domains.",
                "arguments": {"query": "example.com active"},
            },
            {
                "tool_name": "web_fetch",
                "success": True,
                "content": "Fetched https://example.com and confirmed example.com is active.",
                "arguments": {"url": "https://example.com"},
            },
        ],
        context_block="Observation State:\nManager status: finalize",
        controller_summary="Manager status: finalize\nStrategy: Enough evidence.\nNotes: Answer directly now.",
    )

    reminder_messages = [
        message["content"]
        for message in messages
        if message.get("role") == "system" and "<system-reminder>" in message.get("content", "")
    ]
    assert reminder_messages
    assert "Prioritize verified exact-domain fetch evidence over noisier search results" in reminder_messages[0]
    assert "- web_fetch [ok]: Fetched https://example.com and confirmed example.com is active." in reminder_messages[0]


def test_build_final_messages_prefers_substantive_evidence_over_bookkeeping(tmp_path):
    engine = RunEngine(
        adapter=MagicMock(),
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=FakeTraceStore(),
    )
    session = SimpleNamespace(sliding_window=[])

    messages = engine._build_final_messages(
        session=session,
        system_prompt="You are Shovs.",
        user_message="Summarize the run engine.",
        effective_objective="Summarize the run engine.",
        tool_results=[
            {
                "tool_name": "todo_update",
                "success": True,
                "content": "Marked task as in progress.",
                "arguments": {"tasks": [{"title": "Research run engine", "status": "in-progress"}]},
            },
            {
                "tool_name": "query_memory",
                "success": True,
                "content": "Found prior context about run engine notes.",
                "arguments": {"query": "run engine"},
            },
            {
                "tool_name": "web_search",
                "success": True,
                "content": "Run engine search results confirm the main architecture and execution flow.",
                "arguments": {"query": "run engine architecture"},
            },
        ],
        context_block="Observation State:\nManager status: finalize",
        controller_summary="Manager status: finalize\nStrategy: Enough evidence.\nNotes: Answer directly now.",
    )

    reminder_messages = [
        message["content"]
        for message in messages
        if message.get("role") == "system" and "<system-reminder>" in message.get("content", "")
    ]
    assert reminder_messages
    assert "Evidence focus:" in reminder_messages[0]
    assert "- web_search [ok]: Run engine search results confirm the main architecture and execution flow." in reminder_messages[0]
    assert "todo_update" not in reminder_messages[0]
    assert "query_memory" not in reminder_messages[0]


def test_canonical_tool_result_normalizes_preview_and_status():
    summary = canonical_tool_result(
        {
            "tool_name": "web_search",
            "success": True,
            "content": "x" * 300,
            "arguments": {"query": "run engine"},
        },
        preview_chars=32,
    )

    assert summary == {
        "tool": "web_search",
        "tool_name": "web_search",
        "success": True,
        "status": "ok",
        "preview": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx...",
        "arguments": {"query": "run engine"},
        "arguments_summary": "query: run engine",
    }


@pytest.mark.asyncio
async def test_run_engine_bootstraps_tool_call_when_planner_is_disabled(tmp_path):
    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value="No structured tool call.")
    adapter.stream = MagicMock(return_value=AsyncIter(["Done."]))

    registry = ToolRegistry()

    async def web_search(query: str, **kwargs):
        return {"type": "web_search_results", "query": query, "results": [{"title": "A"}]}

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

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))

    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=registry,
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=FakeTraceStore(),
        orchestrator=None,
        context_engine=context_engine,
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="run-engine-bootstrap-no-planner",
                owner_id="owner-1",
                agent_id="default",
                user_message="search latest stocks today",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("web_search",),
                use_planner=False,
            )
        )
    ]

    assert any(event.get("type") == "tool_call" and event.get("tool_name") == "web_search" for event in events)
    assert any(event.get("type") == "tool_result" and event.get("tool_name") == "web_search" for event in events)


@pytest.mark.asyncio
async def test_run_engine_tool_fallback_survives_actor_completion_error(tmp_path):
    adapter = MagicMock()
    adapter.complete = AsyncMock(side_effect=RuntimeError("actor completion failed"))
    adapter.stream = MagicMock(return_value=AsyncIter(["Done."]))

    registry = ToolRegistry()

    async def web_search(query: str, **kwargs):
        return {"type": "web_search_results", "query": query, "results": [{"title": "A"}]}

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

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))

    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=registry,
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=FakeTraceStore(),
        orchestrator=None,
        context_engine=context_engine,
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="run-engine-fallback-on-actor-error",
                owner_id="owner-1",
                agent_id="default",
                user_message="search for run engine docs",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("web_search",),
                use_planner=False,
            )
        )
    ]

    assert any(event.get("type") == "tool_call" and event.get("tool_name") == "web_search" for event in events)
    assert any(event.get("type") == "tool_result" and event.get("tool_name") == "web_search" for event in events)


@pytest.mark.asyncio
async def test_run_engine_honors_max_tool_calls_limit(tmp_path):
    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value='{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\\"query\\": \\"latest run engine changes\\"}"}}]}'
    )
    adapter.stream = MagicMock(return_value=AsyncIter(["Done."]))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(
        return_value={
            "strategy": "Collect evidence.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Need current info."}],
            "confidence": 0.9,
        }
    )
    orchestrator.observe_with_context = AsyncMock(
        return_value={
            "status": "continue",
            "strategy": "Keep searching.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Need one more pass."}],
            "notes": "Continue tool loop.",
            "confidence": 0.7,
        }
    )
    orchestrator.verify_with_context = AsyncMock(
        return_value={"supported": True, "issues": [], "confidence": 0.9}
    )

    registry = ToolRegistry()
    call_counter = {"count": 0}

    async def web_search(query: str, **kwargs):
        call_counter["count"] += 1
        return {"type": "web_search_results", "query": query, "results": [{"title": "A"}]}

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

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    runs = RunStore(db_path=str(tmp_path / "runs.db"))
    engine = RunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=registry,
        run_store=runs,
        trace_store=FakeTraceStore(),
        orchestrator=orchestrator,
        context_engine=context_engine,
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="run-engine-max-tool-calls",
                owner_id="owner-1",
                agent_id="default",
                user_message="find the latest run engine updates",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("web_search",),
                use_planner=True,
                max_tool_calls=1,
                max_turns=5,
            )
        )
    ]

    tool_call_events = [event for event in events if event.get("type") == "tool_call"]
    assert len(tool_call_events) == 1
    assert call_counter["count"] == 1
    assert orchestrator.observe_with_context.await_count == 1


@pytest.mark.asyncio
async def test_run_engine_rejects_model_fabricated_source_select_payloads(tmp_path):
    """Messy frontend-like run: model tries to smuggle fake source state.

    This simulates a fast Gemini/Flash-style actor output: the model skips
    ledger result IDs and passes invented search_payloads directly to
    source_select. The managed runtime must not let deterministic tools treat
    that as real evidence.
    """

    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value=json.dumps({
            "tool": "source_select",
            "arguments": {
                "search_payloads": [
                    {
                        "query": "top 3 sushi restaurants in Toronto",
                        "results": [
                            {"title": "Fake sushi page", "url": "https://www.example.com/toronto-sushi"}
                        ],
                    }
                ],
                "per_entity": 3,
            },
        })
    )
    adapter.stream = MagicMock(return_value=AsyncIter(["I could not verify the requested sources."]))

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(
        return_value={
            "strategy": "Select URLs from prior search evidence.",
            "tools": [{"name": "source_select", "priority": "high", "reason": "Select fetched sources."}],
            "confidence": 0.8,
        }
    )
    orchestrator.observe_with_context = AsyncMock(
        return_value={
            "status": "finalize",
            "strategy": "Stop after rejected source selection.",
            "tools": [],
            "notes": "The selection payload was not ledger-backed.",
            "confidence": 0.5,
        }
    )
    orchestrator.verify_with_context = AsyncMock(
        return_value={"supported": True, "issues": [], "confidence": 0.8}
    )

    registry = ToolRegistry()
    register_tools(registry, "source_select")

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))

    traces = FakeTraceStore()
    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=registry,
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=traces,
        orchestrator=orchestrator,
        context_engine=context_engine,
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="messy-source-select",
                owner_id="owner-1",
                agent_id="default",
                user_message=(
                    "top 3 sushi places in toronto, then search each, fetch 3 URLs each "
                    "to get intel and then give a summary tldr table only"
                ),
                model="gemini-3-flash-test",
                system_prompt="You are Shovs.",
                allowed_tools=("source_select",),
                use_planner=True,
                max_tool_calls=1,
            )
        )
    ]

    tool_results = [event for event in events if event.get("type") == "tool_result"]
    assert tool_results
    assert tool_results[0]["success"] is False
    assert "rejected_untrusted_payload" in tool_results[0]["content"]
    assert not any("https://www.example.com/toronto-sushi" in str(event) and event.get("success") is True for event in events)


def test_engine_url_extraction_ignores_truncated_display_urls():
    urls = _extract_urls_from_tool_results(
        [
            {
                "tool_name": "web_search",
                "success": True,
                "content": json.dumps({
                    "type": "web_search_results",
                    "results": [
                        {"url": "https://fi..."},
                        {"url": "https://news.example/roku"},
                    ],
                }),
            }
        ]
    )

    assert urls == ["https://news.example/roku"]


@pytest.mark.asyncio
async def test_run_engine_injects_search_routing_and_embed_model_context(tmp_path):
    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value='{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\\"query\\": \\"search run engine docs\\"}"}}]}'
    )
    adapter.stream = MagicMock(return_value=AsyncIter(["Done."]))

    registry = ToolRegistry()
    seen: dict[str, object] = {}

    async def web_search(query: str, backend: str = "", search_engine: str = "", **kwargs):
        seen["query"] = query
        seen["backend"] = backend
        seen["search_engine"] = search_engine
        seen["embed_model"] = kwargs.get("_embed_model")
        return {"type": "web_search_results", "query": query, "results": [{"title": "A"}]}

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

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))

    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=registry,
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=FakeTraceStore(),
        orchestrator=None,
        context_engine=context_engine,
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="run-engine-routing-injection",
                owner_id="owner-1",
                agent_id="default",
                user_message="search run engine docs",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("web_search",),
                use_planner=False,
                search_backend="searxng",
                search_engine="brave",
                embed_model="openai:text-embedding-3-small",
            )
        )
    ]

    tool_call_events = [event for event in events if event.get("type") == "tool_call"]
    assert tool_call_events
    assert tool_call_events[0]["arguments"]["backend"] == "searxng"
    assert tool_call_events[0]["arguments"]["search_engine"] == "brave"
    assert seen["backend"] == "searxng"
    assert seen["search_engine"] == "brave"
    assert seen["embed_model"] == "openai:text-embedding-3-small"


@pytest.mark.asyncio
async def test_run_engine_passes_provider_qualified_model_to_context_compression(tmp_path):
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncIter(["Acknowledged."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=FakeTraceStore(),
        orchestrator=None,
        context_engine=context_engine,
    )
    engine._resolve_adapter = MagicMock(return_value=adapter)

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="provider-qualified-compression",
                owner_id="owner-model-route",
                agent_id="default",
                user_message="Remember that my test route uses Groq.",
                model="groq:llama-3.3-70b-versatile",
                system_prompt="You are Shovs.",
                allowed_tools=(),
                use_planner=False,
            )
        )
    ]

    assert any(event["type"] == "done" for event in events)
    assert context_engine.compress_exchange.await_args.kwargs["model"] == "groq:llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_run_engine_rejects_images_on_nonvision_model_without_fallback(tmp_path, monkeypatch):
    from config.config import cfg

    monkeypatch.setattr(cfg, "VISION_MODEL", "", raising=False)
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncIter(["Should not run."]))

    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=FakeTraceStore(),
        orchestrator=None,
        context_engine=None,
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="vision-required-no-fallback",
                owner_id="owner-vision",
                agent_id="default",
                user_message="Describe this image.",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=(),
                use_planner=False,
                images=["base64-image"],
            )
        )
    ]

    assert events == [
        {
            "type": "error",
            "message": (
                "Image input requires a vision-capable model. Select a vision model "
                "or set VISION_MODEL to a provider-prefixed vision model such as "
                "gemini:gemini-1.5-flash, openai:gpt-4o-mini, or ollama:llava."
            ),
        }
    ]
    adapter.stream.assert_not_called()


@pytest.mark.asyncio
async def test_run_engine_routes_images_to_configured_vision_model(tmp_path, monkeypatch):
    from config.config import cfg

    monkeypatch.setattr(cfg, "VISION_MODEL", "llava:latest", raising=False)
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncIter(["Image answer."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    traces = FakeTraceStore()
    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=traces,
        orchestrator=None,
        context_engine=context_engine,
    )
    engine._resolve_adapter = MagicMock(return_value=adapter)

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="vision-fallback-model",
                owner_id="owner-vision",
                agent_id="default",
                user_message="Describe this image.",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=(),
                use_planner=False,
                images=["base64-image"],
            )
        )
    ]

    assert any(event["type"] == "done" for event in events)
    assert adapter.stream.call_args.kwargs["model"] == "llava:latest"
    assert adapter.stream.call_args.kwargs["images"] == ["base64-image"]
    assert any(
        event["event_type"] == "model_capability_violation"
        and event["data"]["fallback"] == "llava:latest"
        for event in traces.events
    )


@pytest.mark.asyncio
async def test_run_engine_skill_discovery_uses_allowed_tools_not_missing_tools_attr(tmp_path):
    workspace = tmp_path / "workspace"
    skill_dir = workspace / ".agent" / "skills" / "research"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: Research\n"
        "description: Research helper\n"
        "triggers: research\n"
        "requirements: web_search\n"
        "---\n"
        "Use web_search for current research.\n",
        encoding="utf-8",
    )

    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncIter(["Acknowledged."]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=FakeTraceStore(),
        orchestrator=None,
        context_engine=context_engine,
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="skill-discovery-request-tools-regression",
                owner_id="owner-skill-route",
                agent_id="default",
                user_message="research this topic",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("web_search",),
                use_planner=False,
                workspace_path=str(workspace),
            )
        )
    ]

    assert any(event["type"] == "done" for event in events)


@pytest.mark.asyncio
async def test_run_engine_retries_response_stream_once_on_rate_limit(tmp_path, monkeypatch):
    from llm.base_adapter import RateLimitError

    async def rate_limited_stream():
        raise RateLimitError("Gemini Rate Limit: retry in 0.01s")
        yield ""

    adapter = MagicMock()
    adapter.stream = MagicMock(side_effect=[rate_limited_stream(), AsyncIter(["Recovered."])])
    monkeypatch.setattr("run_engine.engine.asyncio.sleep", AsyncMock())

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("ctx", [], []))

    engine = RunEngine(
        adapter=adapter,
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=FakeTraceStore(),
        orchestrator=None,
        context_engine=context_engine,
    )

    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="rate-limit-retry",
                owner_id="owner-rate-limit",
                agent_id="default",
                user_message="Say hello.",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=(),
                use_planner=False,
            )
        )
    ]

    assert any(event.get("type") == "activity_short" and "Rate limit hit" in event.get("text", "") for event in events)
    assert "".join(event.get("content", "") for event in events if event.get("type") == "token") == "Recovered."
    assert adapter.stream.call_count == 2
