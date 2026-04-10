from unittest.mock import MagicMock

import pytest

from memory.semantic_graph import SemanticGraph
from orchestration.session_manager import SessionManager
from plugins.tool_registry import ToolCall, ToolRegistry

from engine.conversation_tension import analyze_conversation_tension
from run_engine.memory_pipeline import (
    MemoryCommitPlan,
    apply_memory_commit,
    build_deterministic_memory_commit,
    build_grounding_text,
    merge_candidate_context,
    plan_memory_commit,
)
from run_engine.tool_selection import build_actor_request_content, extract_tool_call, fallback_tool_call, summarize_arguments


def test_extract_tool_call_parses_tool_calls_payload():
    registry = ToolRegistry()
    raw = '{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\\"query\\": \\"run engine\\"}"}}]}'

    call = extract_tool_call(raw, registry)

    assert isinstance(call, ToolCall)
    assert call.tool_name == "web_search"
    assert call.arguments == {"query": "run engine"}


def test_fallback_tool_call_builds_url_and_search_arguments():
    search_call = fallback_tool_call("web_search", "find run engine docs")
    fetch_call = fallback_tool_call("web_fetch", "please fetch https://example.com/docs.")
    memory_call = fallback_tool_call("query_memory", "what did i say about my editor?")

    assert search_call is not None
    assert search_call.arguments == {"query": "find run engine docs"}
    assert fetch_call is not None
    assert fetch_call.arguments == {"url": "https://example.com/docs"}
    assert memory_call is not None
    assert memory_call.arguments == {"topic": "what did i say about my editor?"}


def test_build_actor_request_content_includes_recent_result_previews():
    content = build_actor_request_content(
        user_message="Research the run engine.",
        effective_objective="Research the run engine.",
        session_first_message="Initial goal.",
        allowed_tools=["web_search"],
        tool_results=[{"tool_name": "web_search", "success": True, "content": "Found documentation."}],
        context_block="working context",
        clip_text=lambda text, _: text,
    )

    assert "Allowed tools: web_search" in content
    assert "Found documentation." in content
    assert "Context block:\nworking context" in content


def test_summarize_arguments_clips_long_values():
    summary = summarize_arguments({"query": "x" * 80, "limit": 5})

    assert "query:" in summary
    assert "..." in summary
    assert "limit: 5" in summary


def test_build_grounding_text_can_filter_failed_results():
    tool_results = [
        {"tool_name": "one", "success": False, "content": "bad"},
        {"tool_name": "two", "success": True, "content": "good"},
    ]

    assert build_grounding_text(tool_results, successful_only=False) == "bad\ngood"
    assert build_grounding_text(tool_results, successful_only=True) == "good"


def test_plan_memory_commit_merges_and_blocks_candidate_records():
    plan = plan_memory_commit(
        context_result=(
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
        ),
        user_message="Call me Alex.",
        tool_results=[],
        deterministic_keyed_facts=[
            {
                "subject": "User",
                "predicate": "preferred_name",
                "object": "Alex",
                "fact": "User preferred_name Alex",
            }
        ],
        deterministic_voids=[],
        current_facts=[],
        existing_candidate_signals=[],
        existing_candidate_context="",
    )

    assert plan.new_context == "ctx"
    assert (plan.merged_facts or [])[0]["predicate"] == "preferred_name"
    assert len(plan.blocked_keyed_facts or []) == 1
    assert len(plan.candidate_signals or []) == 1
    assert "Candidate: General working_on Secret roadmap" in plan.candidate_context


def test_merge_candidate_context_deduplicates_lines():
    existing = "- Candidate: General working_on Secret roadmap (reason=not_grounded)"
    merged = merge_candidate_context(
        existing,
        [
            {
                "subject": "General",
                "predicate": "working_on",
                "object": "Secret roadmap",
                "grounding_reason": "not_grounded",
            }
        ],
    )

    assert merged.count("Secret roadmap") == 1


def test_build_deterministic_memory_commit_preserves_existing_candidate_state():
    plan = build_deterministic_memory_commit(
        deterministic_keyed_facts=[
            {
                "subject": "User",
                "predicate": "preferred_name",
                "object": "Alex",
                "fact": "User preferred_name Alex",
                "key": "User preferred_name",
            }
        ],
        deterministic_voids=[{"subject": "User", "predicate": "nickname"}],
        existing_candidate_signals=[{"text": "General working_on Secret roadmap", "reason": "not_grounded"}],
        existing_candidate_context="- Candidate: General working_on Secret roadmap (reason=not_grounded)",
    )

    assert len(plan.merged_facts or []) == 1
    assert len(plan.merged_voids or []) == 1
    assert plan.blocked_keyed_facts == []
    assert plan.candidate_signals == [{"text": "General working_on Secret roadmap", "reason": "not_grounded"}]
    assert "Candidate: General working_on Secret roadmap" in plan.candidate_context


def test_build_deterministic_memory_commit_can_merge_stance_signals():
    plan = build_deterministic_memory_commit(
        deterministic_keyed_facts=[],
        deterministic_voids=[],
        existing_candidate_signals=[],
        existing_candidate_context="",
        user_message="I prefer direct contradiction over polite smoothing.",
        new_candidate_signals=[
            {
                "signal_type": "stance",
                "topic": "direct contradiction polite smoothing",
                "position": "direct contradiction over polite smoothing",
                "confidence": "asserted",
                "turn_index": 4,
                "raw_text": "I prefer direct contradiction over polite smoothing.",
                "superseded": False,
                "source": "stance_extractor",
                "reason": "stance_asserted",
                "text": "Stance [direct contradiction polite smoothing]: direct contradiction over polite smoothing",
            }
        ],
    )

    assert plan.candidate_signal_updates is True
    assert len(plan.candidate_signals or []) == 1
    assert plan.candidate_signals[0]["signal_type"] == "stance"
    assert "Stance [direct contradiction polite smoothing]" in plan.candidate_context


@pytest.mark.asyncio
async def test_apply_memory_commit_updates_session_graph_and_vector_index(tmp_path, monkeypatch):
    indexed_calls = []

    class FakeVectorEngine:
        def __init__(self, session_id, agent_id="default", owner_id=None):
            self.session_id = session_id
            self.agent_id = agent_id
            self.owner_id = owner_id

        async def index(self, key: str, anchor: str, metadata=None):
            indexed_calls.append({"key": key, "anchor": anchor, "metadata": dict(metadata or {})})

    monkeypatch.setattr("run_engine.memory_pipeline.VectorEngine", FakeVectorEngine)

    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    session = sessions.create(
        model="llama3.2",
        system_prompt="",
        agent_id="default",
        session_id="phase2-commit-1",
        owner_id="owner-phase2",
    )
    graph = SemanticGraph(db_path=str(tmp_path / "memory.db"))

    outcome = await apply_memory_commit(
        sessions=sessions,
        session_id=session.id,
        owner_id="owner-phase2",
        agent_id="default",
        turn=2,
        run_id="run-123",
        user_message="Call me Alex.",
        assistant_response="Understood.",
        graph=graph,
        current_context="",
        plan=MemoryCommitPlan(
            new_context="ctx",
            merged_facts=[
                {
                    "subject": "User",
                    "predicate": "preferred_name",
                    "object": "Alex",
                    "fact": "User preferred_name Alex",
                }
            ],
            merged_voids=[{"subject": "User", "predicate": "nickname"}],
            blocked_keyed_facts=[
                {
                    "subject": "General",
                    "predicate": "working_on",
                    "object": "Secret roadmap",
                    "fact": "General working_on Secret roadmap",
                }
            ],
            candidate_signals=[{"text": "General working_on Secret roadmap", "reason": "not_grounded"}],
            candidate_context="- Candidate: General working_on Secret roadmap (reason=not_grounded)",
        ),
    )

    updated = sessions.get(session.id, owner_id="owner-phase2")
    assert updated is not None
    assert updated.compressed_context == "ctx"
    assert updated.candidate_signals == [{"text": "General working_on Secret roadmap", "reason": "not_grounded"}]
    assert "Candidate: General working_on Secret roadmap" in updated.candidate_context
    assert ("User", "preferred_name", "Alex") in graph.get_current_facts(session.id, owner_id="owner-phase2")
    assert outcome.indexed_fact_keys == ["User preferred_name"]
    assert indexed_calls
    assert indexed_calls[0]["key"] == "User preferred_name"


def test_build_phase_packet_can_include_conversation_tension():
    from types import SimpleNamespace

    from engine.context_schema import ContextPhase
    from run_engine.context_packets import PacketBuildInputs, build_phase_packet
    from run_engine.types import RunEngineRequest

    tension = analyze_conversation_tension(
        user_message="Actually, I live in Berlin.",
        current_facts=[("User", "location", "Toronto")],
        deterministic_keyed_facts=[{"subject": "User", "predicate": "location", "object": "Berlin"}],
        session_history=[{"role": "user", "content": "I live in Toronto."}],
    )
    packet = build_phase_packet(
        context_engine=None,
        inputs=PacketBuildInputs(
            request=RunEngineRequest(
                session_id="packet-tension",
                owner_id="owner-1",
                agent_id="default",
                user_message="Actually, I live in Berlin.",
                model="llama3.2",
                system_prompt="You are Shovs.",
            ),
            session=SimpleNamespace(first_message="I live in Toronto.", sliding_window=[]),
            phase=ContextPhase.RESPONSE,
            system_prompt="You are Shovs.",
            effective_objective="Actually, I live in Berlin.",
            current_context="",
            allowed_tools=[],
            tool_results=[],
            conversation_tension=tension,
        ),
    )

    assert "Conversation Tension" in packet.content
    assert "Drift:" in packet.content
    assert packet.trace["runtime_path"] == "run_engine"
    assert packet.trace["content"] == packet.content
    assert any(item["item_id"] == "conversation_tension" for item in packet.trace["included"])


def test_build_phase_packet_can_include_shared_session_and_memory_lanes():
    from types import SimpleNamespace

    from engine.context_schema import ContextPhase
    from run_engine.context_packets import PacketBuildInputs, build_phase_packet
    from run_engine.types import RunEngineRequest

    packet = build_phase_packet(
        context_engine=None,
        inputs=PacketBuildInputs(
            request=RunEngineRequest(
                session_id="packet-shared-lanes",
                owner_id="owner-1",
                agent_id="default",
                user_message="Summarize what you know.",
                model="llama3.2",
                system_prompt="You are Shovs.",
            ),
            session=SimpleNamespace(
                first_message="My name is Shovon.",
                message_count=4,
                candidate_context="- Candidate: User location Berlin (reason=not_grounded)",
                sliding_window=[{"role": "user", "content": "My name is Shovon."}],
                full_history=[
                    {"role": "user", "content": "Earlier constraint one."},
                    {"role": "assistant", "content": "Earlier answer one."},
                    {"role": "user", "content": "My name is Shovon."},
                ],
            ),
            phase=ContextPhase.RESPONSE,
            system_prompt="You are Shovs.",
            effective_objective="Summarize what you know.",
            current_context="",
            allowed_tools=[],
            tool_results=[{"tool_name": "web_search", "success": True, "content": "Found one result."}],
            tool_turn=1,
            strategy="Use stored facts first.",
            notes="Preserve contradiction state.",
            current_facts=[("User", "preferred_name", "Shovon")],
        ),
    )

    assert "Session Anchor" in packet.content
    assert "Meta Context" in packet.content
    assert "Epistemic Posture:" in packet.content
    assert "Deterministic Facts" in packet.content
    assert "Candidate Signals" in packet.content
    assert "Working Evidence" in packet.content
    assert "Curated evidence most relevant to the current objective." in packet.content
    meta_item = next(item for item in packet.trace["included"] if item["item_id"] == "meta_context")
    assert meta_item["kind"] == "meta"
    assert meta_item["provenance"]["known_fact_count"] == 1
    assert meta_item["provenance"]["candidate_count"] == 1
    evidence_item = next(item for item in packet.trace["included"] if item["item_id"] == "working_evidence")
    assert evidence_item["kind"] == "evidence"
    assert evidence_item["provenance"]["selected_count"] == 1


def test_build_phase_packet_meta_context_can_define_minimum_probe():
    from types import SimpleNamespace

    from engine.context_schema import ContextPhase
    from run_engine.context_packets import PacketBuildInputs, build_phase_packet
    from run_engine.types import RunEngineRequest

    packet = build_phase_packet(
        context_engine=None,
        inputs=PacketBuildInputs(
            request=RunEngineRequest(
                session_id="packet-meta-probe",
                owner_id="owner-1",
                agent_id="default",
                user_message="Research wigglebudget.com",
                model="llama3.2",
                system_prompt="You are Shovs.",
            ),
            session=SimpleNamespace(first_message="Research wigglebudget.com", sliding_window=[]),
            phase=ContextPhase.PLANNING,
            system_prompt="You are Shovs.",
            effective_objective="Research wigglebudget.com",
            current_context="",
            allowed_tools=[{"name": "web_fetch", "description": "Fetch exact page"}, {"name": "web_search", "description": "Search the web"}],
            tool_results=[],
        ),
    )

    assert "Minimum Next Probe:" in packet.content
    assert "Fetch the exact target directly: wigglebudget.com." in packet.content
    meta_item = next(item for item in packet.trace["included"] if item["item_id"] == "meta_context")
    assert meta_item["kind"] == "meta"
    assert meta_item["provenance"]["evidence_count"] == 0


def test_build_phase_packet_prefers_resolved_working_objective_when_present():
    from types import SimpleNamespace

    from engine.context_schema import ContextPhase
    from run_engine.context_packets import PacketBuildInputs, build_phase_packet
    from run_engine.types import RunEngineRequest

    packet = build_phase_packet(
        context_engine=None,
        inputs=PacketBuildInputs(
            request=RunEngineRequest(
                session_id="packet-objective",
                owner_id="owner-1",
                agent_id="default",
                user_message="try again",
                model="llama3.2",
                system_prompt="You are Shovs.",
            ),
            session=SimpleNamespace(first_message="research wigglebudget.com", sliding_window=[]),
            phase=ContextPhase.PLANNING,
            system_prompt="You are Shovs.",
            effective_objective="research wigglebudget.com",
            current_context="",
            allowed_tools=[],
            tool_results=[],
        ),
    )

    assert "Resolved working objective:" in packet.content
    assert "research wigglebudget.com" in packet.content
    assert "Current user turn:" in packet.content
    assert any(item["item_id"] == "current_objective" for item in packet.trace["included"])
    assert packet.trace["trace_scope"] == "phase_packet"
    assert packet.trace["canonical_event"] == "phase_context"


def test_build_phase_packet_uses_shared_context_engine_memory_fallback_shape():
    from types import SimpleNamespace

    from engine.context_schema import ContextPhase
    from run_engine.context_packets import PacketBuildInputs, build_phase_packet
    from run_engine.types import RunEngineRequest

    context_engine = MagicMock()
    context_engine.build_context_items.return_value = []
    context_engine.build_context_block.return_value = "--- Session Memory ---\nkept fact\n--- End Session Memory ---"

    packet = build_phase_packet(
        context_engine=context_engine,
        inputs=PacketBuildInputs(
            request=RunEngineRequest(
                session_id="packet-context-engine-fallback",
                owner_id="owner-1",
                agent_id="default",
                user_message="Summarize the retained memory.",
                model="llama3.2",
                system_prompt="You are Shovs.",
            ),
            session=SimpleNamespace(sliding_window=[]),
            phase=ContextPhase.RESPONSE,
            system_prompt="You are Shovs.",
            current_context="kept fact",
            allowed_tools=[],
            tool_results=[],
        ),
    )

    memory_item = next(item for item in packet.trace["included"] if item["item_id"] == "context_engine_memory")
    assert memory_item["trace_id"] == "memory:context_engine"
    assert memory_item["source"] == "context_engine"
