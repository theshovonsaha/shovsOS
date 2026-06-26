import json
from unittest.mock import MagicMock

import pytest

from memory.semantic_graph import SemanticGraph
from orchestration.session_manager import SessionManager
from plugins.tool_registry import ToolCall, ToolRegistry

from engine.conversation_tension import analyze_conversation_tension
from orchestration.run_store import RunStore
from run_engine.control_policies import resolve_control_policy
from run_engine.engine import RunEngine
from run_engine.ledger import RunLedger, recovery_policy_for
from run_engine.memory_pipeline import (
    MemoryCommitPlan,
    apply_memory_commit,
    build_deterministic_memory_commit,
    build_grounding_text,
    merge_candidate_context,
    plan_memory_commit,
)
from run_engine.tool_selection import build_actor_request_content, extract_tool_call, fallback_tool_call, summarize_arguments
from run_engine.search_query import compile_web_search_query


class _TraceCapture:
    def __init__(self):
        self.events = []

    def append_event(self, agent_id, session_id, event_type, data, **kwargs):
        self.events.append({
            "agent_id": agent_id,
            "session_id": session_id,
            "event_type": event_type,
            "data": data,
            **kwargs,
        })


def test_continuation_gate_resumes_ambiguous_followup():
    from run_engine.engine import _assess_continuation_state

    decision = _assess_continuation_state(
        user_message="continue",
        continuation_state={
            "objective": "Find a storage bin under $20 near Toronto.",
            "pending_steps": [{"id": "s1", "tool": "shopping_advice", "status": "pending"}],
        },
        ambiguous_followup=True,
    )

    assert decision["action"] == "resume"
    assert decision["effective_objective"] == "Find a storage bin under $20 near Toronto."
    assert decision["pending_step_count"] == 1


def test_recovery_policy_classes_are_typed():
    policy = recovery_policy_for("entity_drift")

    assert policy.max_retries == 1
    assert "web_search" in policy.allowed_recovery_tools
    assert policy.persist_continuation is True


def test_engine_policy_gate_recovers_bad_plan_execute_tool_call_in_enforced_mode(tmp_path):
    traces = _TraceCapture()
    engine = RunEngine(
        adapter=MagicMock(),
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=traces,
    )
    ledger = RunLedger(
        run_id="run-policy-enforced",
        session_id="session-policy-enforced",
        turn_id="turn-1",
        objective="Search ROKU, TBN, SENEA separately and fetch sources",
        allowed_tools=["web_search", "web_fetch"],
        ledger_mode="ledger_enforced",
    )
    ledger.set_control_policy(resolve_control_policy(ledger.objective, requested="plan_execute"))
    ledger.lock_entities(["ROKU", "TBN", "SENEA"])
    ledger.set_source_contract({
        "next_tool": "web_search",
        "next_arguments": {"query": "ROKU stock news June 13 2026"},
        "forbid_unlocked_entity_drift": True,
    })

    recovered, event = engine._gate_tool_call_with_ledger(
        request=MagicMock(session_id="session-policy-enforced", agent_id="agent", owner_id="owner"),
        run_id="run-policy-enforced",
        ledger=ledger,
        tool_call=ToolCall(
            "web_search",
            {"query": "EPAM stock news June 13 2026"},
            '{"tool":"web_search"}',
        ),
        tool_turn=1,
    )

    assert recovered is not None
    assert recovered.tool_name == "web_search"
    assert recovered.arguments == {"query": "ROKU stock news June 13 2026"}
    assert event["type"] == "recovery_started"
    assert event["recovery_policy"]["recovery_class"] == "entity_drift"
    assert any(item["event_type"] == "policy_violation" for item in traces.events)
    assert any(item["event_type"] == "recovery_started" for item in traces.events)


def test_engine_policy_gate_records_but_allows_bad_call_in_shadow_mode(tmp_path):
    traces = _TraceCapture()
    engine = RunEngine(
        adapter=MagicMock(),
        sessions=SessionManager(db_path=str(tmp_path / "sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=traces,
    )
    ledger = RunLedger(
        run_id="run-policy-shadow",
        session_id="session-policy-shadow",
        turn_id="turn-1",
        objective="Search ROKU separately and fetch sources",
        allowed_tools=["web_search"],
        ledger_mode="shadow",
    )
    ledger.set_control_policy(resolve_control_policy(ledger.objective, requested="plan_execute"))
    ledger.lock_entities(["ROKU"])
    ledger.set_source_contract({
        "next_tool": "web_search",
        "next_arguments": {"query": "ROKU stock news June 13 2026"},
        "forbid_unlocked_entity_drift": True,
    })
    original = ToolCall(
        "web_search",
        {"query": "EPAM stock news June 13 2026"},
        '{"tool":"web_search"}',
    )

    allowed, event = engine._gate_tool_call_with_ledger(
        request=MagicMock(session_id="session-policy-shadow", agent_id="agent", owner_id="owner"),
        run_id="run-policy-shadow",
        ledger=ledger,
        tool_call=original,
        tool_turn=1,
    )

    assert allowed is original
    assert event["type"] == "policy_violation"
    assert event["ledger_mode"] == "shadow"
    assert ledger.policy_violations[0]["recovery_policy"]["persist_continuation"] is True


def test_continuation_gate_extends_related_additive_turn():
    from run_engine.engine import _assess_continuation_state

    decision = _assess_continuation_state(
        user_message="also check Walmart for the storage bin",
        continuation_state={
            "objective": "Find a storage bin under $20 near Toronto at Canadian Tire.",
            "pending_steps": [{"id": "s1", "tool": "shopping_advice", "status": "pending"}],
        },
        ambiguous_followup=False,
    )

    assert decision["action"] == "resume_with_update"
    assert "Current turn update" in decision["effective_objective"]
    assert decision["overlap"] > 0


def test_continuation_gate_supersedes_unrelated_new_task():
    from run_engine.engine import _assess_continuation_state

    decision = _assess_continuation_state(
        user_message="Write a Python script to rename files.",
        continuation_state={
            "objective": "Find a storage bin under $20 near Toronto.",
            "pending_steps": [{"id": "s1", "tool": "shopping_advice", "status": "pending"}],
        },
        ambiguous_followup=False,
    )

    assert decision["action"] == "supersede"
    assert decision["effective_objective"] == "Write a Python script to rename files."


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


def test_compile_web_search_query_removes_workflow_control_text():
    query = compile_web_search_query(
        "Search top 3 stocks today with major jumps web search those 3 stocks separately "
        "and capture the 3 relevant for each results web fetch all 9 urls that was found "
        "one by one analyze report for each and write a tldr summary table."
    )

    assert query == "top 3 stocks today with major jumps"


def test_fallback_web_search_does_not_fire_full_source_workflow_as_query():
    call = fallback_tool_call(
        "web_search",
        "Search top 3 sushi places in Toronto, then search each, fetch 3 URLs each, and write a TLDR table.",
    )

    assert call is not None
    assert call.arguments == {"query": "top 3 sushi places in Toronto"}


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


def test_build_actor_request_content_includes_capability_cards():
    content = build_actor_request_content(
        user_message="Find a storage bin near Toronto.",
        effective_objective="Find a storage bin near Toronto.",
        session_first_message="",
        allowed_tools=["shopping_advice"],
        tool_results=[],
        context_block="",
        clip_text=lambda text, _: text,
        capability_context="Capability: Local Store Shopping Advisor\nOutput:\n- answer_patch.comparison_table",
    )

    assert "Capability cards for available workflows:" in content
    assert "Local Store Shopping Advisor" in content
    assert "answer_patch.comparison_table" in content


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
    tension_item = next(item for item in packet.trace["included"] if item["item_id"] == "conversation_tension")
    assert tension_item["provenance"]["dynamic"] is True
    assert tension_item["provenance"]["conflict_count"] == 1
    assert tension_item["provenance"]["storage_action"] == "void_previous_store_current"

    memory_packet = build_phase_packet(
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
            phase=ContextPhase.MEMORY_COMMIT,
            system_prompt="You are Shovs.",
            effective_objective="Actually, I live in Berlin.",
            current_context="",
            allowed_tools=[],
            tool_results=[],
            conversation_tension=tension,
        ),
    )

    assert "Conversation Tension" in memory_packet.content
    assert "Storage Impact: void_previous_store_current" in memory_packet.content
    assert "Meta Context" not in memory_packet.content


def test_build_phase_packet_includes_capability_cards():
    from types import SimpleNamespace

    from engine.context_schema import ContextPhase
    from run_engine.context_packets import PacketBuildInputs, build_phase_packet
    from run_engine.types import RunEngineRequest

    packet = build_phase_packet(
        context_engine=None,
        inputs=PacketBuildInputs(
            request=RunEngineRequest(
                session_id="packet-capability",
                owner_id="owner-1",
                agent_id="shopping-advisor",
                user_message="Find a storage bin near Toronto.",
                model="llama3.2",
                system_prompt="You are Shovs.",
            ),
            session=SimpleNamespace(first_message="", sliding_window=[]),
            phase=ContextPhase.ACTING,
            system_prompt="You are Shovs.",
            effective_objective="Find a storage bin near Toronto.",
            current_context="",
            allowed_tools=[{"name": "shopping_advice", "description": "Verified shopping advice."}],
            tool_results=[],
            capability_context="Capability: Local Store Shopping Advisor\nOutput:\n- answer_patch.comparison_table",
        ),
    )

    assert "Capability Cards" in packet.content
    assert "Local Store Shopping Advisor" in packet.content
    item = next(item for item in packet.trace["included"] if item["item_id"] == "capability_cards")
    assert item["source"] == "capability_registry"


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
    assert "Memory Authority" in packet.content
    assert "Deterministic Facts" in packet.content
    assert "Candidate Signals" not in packet.content
    assert "Working Evidence" in packet.content
    assert "Curated evidence most relevant to the current objective." in packet.content
    meta_item = next(item for item in packet.trace["included"] if item["item_id"] == "meta_context")
    assert meta_item["kind"] == "meta"
    assert meta_item["provenance"]["known_fact_count"] == 1
    assert meta_item["provenance"]["candidate_count"] == 1
    assert meta_item["provenance"]["memory_mode"] == "deterministic_plus_evidence"
    assert "tool_economy" in meta_item["provenance"]
    assert "contradiction_policy" in meta_item["provenance"]
    evidence_item = next(item for item in packet.trace["included"] if item["item_id"] == "working_evidence")
    assert evidence_item["kind"] == "evidence"
    assert evidence_item["provenance"]["selected_count"] == 1
    assert packet.trace["governed_memory"]["candidate_source"] == "legacy_candidate_context"


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


def test_failed_or_empty_tool_results_do_not_count_as_substantive_evidence():
    from run_engine.evidence_lane import build_working_evidence_snapshot, is_substantive_tool_result
    from run_engine.meta_context import build_meta_context_snapshot

    failed_search = {
        "tool_name": "web_search",
        "success": False,
        "content": "Network timeout",
        "arguments": {"query": "latest shovs os research"},
    }
    empty_search = {
        "tool_name": "web_search",
        "success": True,
        "content": '{"type":"web_search_results","query":"latest shovs os research","results":[]}',
        "arguments": {"query": "latest shovs os research"},
    }

    assert is_substantive_tool_result(failed_search) is False
    assert is_substantive_tool_result(empty_search) is False

    snapshot = build_working_evidence_snapshot(
        [failed_search, empty_search],
        user_message="Research latest shovs os research",
    )
    assert snapshot.substantive_count == 0

    meta = build_meta_context_snapshot(
        objective="Research latest shovs os research",
        allowed_tools=[{"name": "web_search"}],
        evidence_snapshot=snapshot,
    )
    assert "No substantive evidence" in meta.verification_posture
    assert meta.memory_mode == "evidence_first"


def test_memory_commit_packet_excludes_epistemic_posture_meta_context():
    from types import SimpleNamespace

    from engine.context_schema import ContextPhase
    from run_engine.context_packets import PacketBuildInputs, build_phase_packet
    from run_engine.types import RunEngineRequest

    packet = build_phase_packet(
        context_engine=None,
        inputs=PacketBuildInputs(
            request=RunEngineRequest(
                session_id="packet-memory-commit",
                owner_id="owner-1",
                agent_id="default",
                user_message="Remember that my editor is Cursor.",
                model="llama3.2",
                system_prompt="You are Shovs.",
            ),
            session=SimpleNamespace(
                first_message="Remember that my editor is Cursor.",
                candidate_context="- Candidate: User preferred_editor Cursor (reason=asserted)",
                sliding_window=[],
                full_history=[],
            ),
            phase=ContextPhase.MEMORY_COMMIT,
            system_prompt="You are Shovs.",
            effective_objective="Remember that my editor is Cursor.",
            current_context="",
            allowed_tools=[{"name": "query_memory", "description": "Recall memory"}],
            tool_results=[
                {
                    "tool_name": "web_search",
                    "success": True,
                    "content": "External evidence that should remain evidence, not posture.",
                }
            ],
            current_facts=[("User", "preferred_editor", "Cursor")],
        ),
    )

    included = [item["item_id"] for item in packet.trace["included"]]
    assert "meta_context" not in included
    assert "Meta Context" not in packet.content
    assert "Epistemic Posture:" not in packet.content
    assert "Deterministic Facts" in packet.content
    assert "Memory Authority" in packet.content


def test_phase_packet_includes_durable_continuation_state():
    from types import SimpleNamespace

    from engine.context_schema import ContextPhase
    from run_engine.context_packets import PacketBuildInputs, build_phase_packet
    from run_engine.types import RunEngineRequest

    packet = build_phase_packet(
        context_engine=None,
        inputs=PacketBuildInputs(
            request=RunEngineRequest(
                session_id="packet-continuation",
                owner_id="owner-1",
                agent_id="default",
                user_message="continue",
                model="llama3.2",
                system_prompt="You are Shovs.",
            ),
            session=SimpleNamespace(
                first_message="Research example.com pricing.",
                sliding_window=[],
                full_history=[],
                continuation_state={
                    "reason": "pending_plan_steps",
                    "objective": "Research example.com pricing.",
                    "next_action": "Fetch the pricing page.",
                    "pending_steps": [
                        {
                            "id": "step_2",
                            "description": "Fetch first-party pricing evidence",
                            "tool": "web_fetch",
                            "status": "pending",
                        }
                    ],
                    "missing_slots": ["first-party pricing page"],
                    "evidence_summary": ["- web_search [ok]: Found example.com homepage"],
                },
            ),
            phase=ContextPhase.PLANNING,
            system_prompt="You are Shovs.",
            effective_objective="Research example.com pricing.",
            current_context="",
            allowed_tools=[{"name": "web_fetch", "description": "Fetch exact page"}],
            tool_results=[],
        ),
    )

    assert "Continuation State" in packet.content
    assert "Prior run did not fully close" in packet.content
    assert "first-party pricing page" in packet.content
    item = next(item for item in packet.trace["included"] if item["item_id"] == "continuation_state")
    assert item["provenance"]["reason"] == "pending_plan_steps"
    assert item["provenance"]["pending_step_count"] == 1


def test_tool_action_claim_guard_requires_matching_read_tool_evidence():
    from engine.side_effect_guard import check_side_effect_claims

    unsupported = check_side_effect_claims(
        "I searched the web and found three sources.",
        tool_results=[],
    )
    assert unsupported["supported"] is False
    assert "web_search" in unsupported["claims"]

    supported = check_side_effect_claims(
        "I searched the web and found three sources.",
        tool_results=[{"tool_name": "web_search", "success": True, "content": "results"}],
    )
    assert supported["supported"] is True


def test_tool_action_claim_guard_requires_matching_fetch_tool_evidence():
    from engine.side_effect_guard import check_side_effect_claims

    unsupported = check_side_effect_claims(
        "I fetched the page and read the pricing details.",
        tool_results=[{"tool_name": "web_search", "success": True, "content": "result"}],
    )
    assert unsupported["supported"] is False
    assert "web_fetch" in unsupported["claims"]

    supported = check_side_effect_claims(
        "I fetched the page and read the pricing details.",
        tool_results=[{"tool_name": "web_fetch", "success": True, "content": "pricing"}],
    )
    assert supported["supported"] is True


@pytest.mark.asyncio
async def test_tool_actor_fallback_can_fetch_url_from_context_block(tmp_path):
    from unittest.mock import AsyncMock, MagicMock

    from orchestration.run_store import RunStore
    from orchestration.session_manager import SessionManager
    from plugins.tool_registry import Tool, ToolRegistry
    from run_engine.engine import RunEngine
    from run_engine.types import RunEngineRequest

    adapter = MagicMock()
    adapter.complete = AsyncMock(side_effect=RuntimeError("model failed"))

    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    session = sessions.create(
        model="llama3.2",
        system_prompt="",
        agent_id="default",
        session_id="fallback-context-url",
        owner_id="owner-1",
    )
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="web_fetch",
            description="Fetch URL",
            parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            handler=AsyncMock(return_value="{}"),
        )
    )
    engine = RunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=registry,
        run_store=RunStore(db_path=str(tmp_path / "runs.db")),
        trace_store=MagicMock(append_event=MagicMock()),
    )

    call = await engine._select_tool_call(
        adapter=adapter,
        model="llama3.2",
        request=RunEngineRequest(
            session_id=session.id,
            owner_id="owner-1",
            agent_id="default",
            user_message="continue",
            model="llama3.2",
            system_prompt="",
        ),
        session=session,
        allowed_tools=["web_fetch"],
        tool_results=[],
        context_block="Continuation State\nNext required action: fetch https://example.com/pricing",
    )

    assert call is not None
    assert call.tool_name == "web_fetch"
    assert call.arguments == {"url": "https://example.com/pricing"}


def test_build_phase_packet_direct_fact_turn_prefers_deterministic_only_mode():
    from types import SimpleNamespace

    from engine.context_schema import ContextPhase
    from run_engine.context_packets import PacketBuildInputs, build_phase_packet
    from run_engine.types import RunEngineRequest

    packet = build_phase_packet(
        context_engine=None,
        inputs=PacketBuildInputs(
            request=RunEngineRequest(
                session_id="packet-direct-fact",
                owner_id="owner-1",
                agent_id="default",
                user_message="What editor do I use now?",
                model="llama3.2",
                system_prompt="You are Shovs.",
            ),
            session=SimpleNamespace(
                first_message="I use VS Code.",
                candidate_context="- Candidate: User editor Cursor (reason=stale)",
                sliding_window=[],
                full_history=[{"role": "user", "content": "I used Cursor before."}],
            ),
            phase=ContextPhase.RESPONSE,
            system_prompt="You are Shovs.",
            effective_objective="What editor do I use now?",
            current_context="",
            allowed_tools=[{"name": "query_memory", "description": "Recall memory"}],
            tool_results=[],
            current_facts=[("User", "preferred_editor", "VS Code")],
            direct_fact_memory_only=True,
        ),
    )

    assert "Memory Mode:" in packet.content
    assert "- deterministic_only" in packet.content
    assert "Historical Context" not in packet.content
    assert "Candidate Signals" not in packet.content


def test_build_phase_packet_prefers_structured_candidate_signals_over_stale_candidate_text():
    from types import SimpleNamespace

    from engine.context_schema import ContextPhase
    from run_engine.context_packets import PacketBuildInputs, build_phase_packet
    from run_engine.types import RunEngineRequest

    packet = build_phase_packet(
        context_engine=None,
        inputs=PacketBuildInputs(
            request=RunEngineRequest(
                session_id="packet-structured-candidates",
                owner_id="owner-1",
                agent_id="default",
                user_message="What candidate signals are active?",
                model="llama3.2",
                system_prompt="You are Shovs.",
            ),
            session=SimpleNamespace(
                first_message="Track my constraints.",
                candidate_context="- Candidate: stale legacy text (reason=old)",
                candidate_signals=[
                    {
                        "text": "User may increase budget to $4k next month",
                        "reason": "candidate_budget_change",
                    }
                ],
                sliding_window=[],
                full_history=[],
            ),
            phase=ContextPhase.PLANNING,
            system_prompt="You are Shovs.",
            effective_objective="What candidate signals are active?",
            current_context="",
            allowed_tools=[],
            tool_results=[],
        ),
    )

    assert "User may increase budget to $4k next month" in packet.content
    assert "stale legacy text" not in packet.content
    assert packet.trace["governed_memory"]["candidate_source"] == "structured_candidate_signals"


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


# ── Slice 1: plan_steps spine ─────────────────────────────────────────────


def test_normalize_plan_steps_keeps_well_formed_entries():
    from orchestration.orchestrator import normalize_plan_steps

    raw = [
        {"id": "step_1", "description": "fetch the page", "tool": "web_fetch",
         "status": "pending", "risk": "read_only"},
        {"id": "step_2", "description": "summarize", "tool": None,
         "status": "pending", "risk": "read_only"},
    ]
    steps = normalize_plan_steps(raw, tools=[{"name": "web_fetch"}])
    assert len(steps) == 2
    assert steps[0]["status"] == "pending"
    assert steps[0]["tool"] == "web_fetch"
    assert steps[1]["tool"] is None


def test_normalize_plan_steps_drops_malformed_and_caps_at_six():
    from orchestration.orchestrator import normalize_plan_steps

    raw = [{"id": f"s{i}", "description": f"d{i}", "tool": "bash"} for i in range(10)]
    raw.append({"description": ""})  # empty desc → dropped
    raw.append("not a dict")  # non-dict → dropped
    steps = normalize_plan_steps(raw, tools=[{"name": "bash"}])
    assert len(steps) == 6  # cap


def test_normalize_plan_steps_forces_pending_status():
    from orchestration.orchestrator import normalize_plan_steps

    # Planner tries to declare a step already done — runtime owns status.
    raw = [{"id": "s1", "description": "x", "tool": "bash", "status": "done"}]
    steps = normalize_plan_steps(raw, tools=[{"name": "bash"}])
    assert steps[0]["status"] == "pending"


def test_normalize_plan_steps_normalizes_invalid_risk():
    from orchestration.orchestrator import normalize_plan_steps

    raw = [{"id": "s1", "description": "x", "tool": "bash", "risk": "nuclear"}]
    steps = normalize_plan_steps(raw, tools=[{"name": "bash"}])
    assert steps[0]["risk"] == "read_only"


def test_normalize_plan_steps_empty_input_returns_empty():
    from orchestration.orchestrator import normalize_plan_steps

    assert normalize_plan_steps([], []) == []
    assert normalize_plan_steps(None, []) == []


def test_update_plan_step_marks_first_pending_match_done():
    from run_engine.engine import _update_plan_step

    steps = [
        {"id": "s1", "tool": "web_fetch", "status": "pending"},
        {"id": "s2", "tool": "web_fetch", "status": "pending"},
    ]
    updated = _update_plan_step(steps, "web_fetch", success=True)
    assert updated == "s1"
    assert steps[0]["status"] == "done"
    assert steps[1]["status"] == "pending"


def test_update_plan_step_marks_failed_on_failure():
    from run_engine.engine import _update_plan_step

    steps = [{"id": "s1", "tool": "bash", "status": "pending"}]
    _update_plan_step(steps, "bash", success=False)
    assert steps[0]["status"] == "failed"


def test_update_plan_step_marks_failed_on_hard_failure():
    from run_engine.engine import _update_plan_step

    steps = [{"id": "s1", "tool": "bash", "status": "pending"}]
    _update_plan_step(steps, "bash", success=True, hard_failure=True)
    assert steps[0]["status"] == "failed"


def test_update_plan_step_no_match_returns_none():
    from run_engine.engine import _update_plan_step

    steps = [{"id": "s1", "tool": "bash", "status": "done"}]
    assert _update_plan_step(steps, "web_fetch", success=True) is None
    assert _update_plan_step([], "bash", success=True) is None


def test_normalize_plan_steps_truncates_long_description():
    from orchestration.orchestrator import normalize_plan_steps

    raw = [{"id": "s1", "description": "x" * 500, "tool": "bash"}]
    steps = normalize_plan_steps(raw, tools=[{"name": "bash"}])
    assert len(steps[0]["description"]) == 240


# ── Slice 2: finalize gate ────────────────────────────────────────────────


def test_finalize_gate_passes_with_no_plan_steps():
    from run_engine.engine import _should_finalize

    should, reason = _should_finalize("finalize", [], 2, 10, "s1")
    assert should is True
    assert reason == "no_plan_steps"

    should, reason = _should_finalize("continue", [], 2, 10, "s1")
    assert should is False  # backward compat: respect observation status
    assert reason == "no_plan_steps"


def test_finalize_gate_clean_finalize_when_all_done():
    from run_engine.engine import _should_finalize

    steps = [
        {"id": "s1", "tool": "bash", "status": "done"},
        {"id": "s2", "tool": "web_search", "status": "done"},
    ]
    should, reason = _should_finalize("finalize", steps, 2, 10, "s1")
    assert should is True
    assert reason == "clean_finalize"


def test_finalize_gate_blocks_on_pending_steps():
    from run_engine.engine import _should_finalize

    steps = [
        {"id": "s1", "tool": "bash", "status": "done"},
        {"id": "s2", "tool": "web_search", "status": "pending"},
    ]
    should, reason = _should_finalize("finalize", steps, 2, 10, "s1")
    assert should is False
    assert reason.startswith("pending_steps")
    assert "s2" in reason


def test_finalize_gate_forces_finalize_at_max_calls_with_pending_flag():
    from run_engine.engine import _should_finalize

    steps = [{"id": "s1", "tool": "bash", "status": "pending"}]
    should, reason = _should_finalize("continue", steps, 10, 10, "s1")
    assert should is True
    assert "max_tool_calls_reached" in reason
    assert "1_steps_pending" in reason


def test_finalize_gate_blocks_on_open_todos_when_steps_done():
    from run_engine.engine import _should_finalize

    class FakeTracker:
        def has_active_tasks(self, sid):
            return True

    steps = [{"id": "s1", "tool": "bash", "status": "done"}]
    should, reason = _should_finalize(
        "finalize", steps, 2, 10, "s1", task_tracker=FakeTracker()
    )
    assert should is False
    assert reason == "open_todos"


def test_finalize_gate_swallows_tracker_errors():
    from run_engine.engine import _should_finalize

    class BrokenTracker:
        def has_active_tasks(self, sid):
            raise RuntimeError("boom")

    steps = [{"id": "s1", "tool": "bash", "status": "done"}]
    # Tracker error must not deadlock the loop — treated as no open todos.
    should, reason = _should_finalize(
        "finalize", steps, 2, 10, "s1", task_tracker=BrokenTracker()
    )
    assert should is True
    assert reason == "clean_finalize"


def test_finalize_gate_allows_no_max_cap():
    from run_engine.engine import _should_finalize

    steps = [{"id": "s1", "tool": "bash", "status": "done"}]
    # max_tool_calls=None means no cap; should still finalize cleanly.
    should, reason = _should_finalize("finalize", steps, 999, None, "s1")
    assert should is True
    assert reason == "clean_finalize"


def test_stock_source_workflow_forces_separate_ticker_searches_before_fetches():
    from run_engine.engine import _stock_source_workflow_override

    objective = (
        "Search top 3 stocks today with major jumps web search those 3 stocks separately "
        "and capture the 3 relevant for each results web fetch all 9 urls one by one."
    )
    broad_result = {
        "tool_name": "web_search",
        "success": True,
        "arguments": {"query": "top 3 stocks with major price jumps today"},
        "content": json.dumps({
            "results": [
                {"title": "Morningstar market movers", "url": "https://www.morningstar.com/markets/movers"},
                {"title": "EPAM jumps on earnings", "url": "https://news.example/epam"},
                {"title": "ARKO stock rises today", "url": "https://news.example/arko"},
            ]
        }),
    }

    override = _stock_source_workflow_override(
        objective=objective,
        allowed_tools=[{"name": "web_search"}, {"name": "web_fetch"}],
        tool_results=[broad_result],
    )

    assert override["status"] == "partial"
    assert override["selected_tools"] == ["web_fetch"]
    assert override["argument_clues"]["web_fetch"] == "https://www.morningstar.com/markets/movers"


def test_stock_source_workflow_locks_tickers_from_fetched_movers_table_not_noisy_search():
    from run_engine.engine import _stock_source_workflow_override

    objective = (
        "Search top 3 stocks today with major jumps web search those 3 stocks separately "
        "and capture the 3 relevant for each results web fetch all 9 urls one by one."
    )
    noisy_search = {
        "tool_name": "web_search",
        "success": True,
        "arguments": {"query": "top 3 stocks with major price jumps today June 13 2026"},
        "content": json.dumps({
            "results": [
                {"title": "EPAM stock major jump today", "url": "https://news.example/epam"},
                {"title": "US stock market update PM", "url": "https://news.example/us-pm"},
                {"title": "ARKO stock headlines", "url": "https://news.example/arko"},
            ]
        }),
    }
    movers_fetch = {
        "tool_name": "web_fetch",
        "success": True,
        "arguments": {"url": "https://www.morningstar.com/markets/movers"},
        "content": """
## Gainers
| 1-Day Chart | Stock | Price | Volume/Average |
| --- | --- | --- | --- |
| | [Roku Inc](https://www.morningstar.com/stocks/xnas/roku/quote) ROKU | $143.66 +24.02 (20.08%) | 15M 3M |
| | [Tamboran Resources Corp](https://www.morningstar.com/stocks/xnys/tbn/quote) TBN | $40.37 +6.71 (19.93%) | 509,942 222,049 |
| | [Seneca Foods Corp](https://www.morningstar.com/stocks/xnas/senea/quote) SENEA | $175.37 +26.16 (17.53%) | 538,088 133,799 |
## Losers
""",
    }

    override = _stock_source_workflow_override(
        objective=objective,
        allowed_tools=[{"name": "web_search"}, {"name": "web_fetch"}],
        tool_results=[noisy_search, movers_fetch],
    )

    assert override["status"] == "partial"
    assert override["selected_tools"] == ["web_search"]
    assert override["tickers"] == ["ROKU", "TBN", "SENEA"]
    assert override["argument_clues"]["web_search"] == "ROKU stock news June 13 2026"
    assert "EPAM" not in override["argument_clues"]["web_search"]


def test_stock_source_workflow_locks_tickers_from_tradingview_compact_rows():
    from run_engine.engine import _stock_source_workflow_override

    objective = (
        "Search top 3 stocks today with major jumps web search those 3 stocks separately "
        "and capture the 3 relevant for each results web fetch all 9 urls one by one."
    )
    tradingview_fetch = {
        "tool_name": "web_fetch",
        "success": True,
        "arguments": {"url": "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/"},
        "content": json.dumps({
            "type": "web_fetch_result",
            "url": "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers",
            "content": """
US stocks that increased the most in price
MEIMethode Electronics, Inc. D · +21.18%, 14.02 USD
DOMODomo, Inc. D · +21.00%, 2.42 USD
GEGGreat Elm Group, Inc. D · +19.43%, 2.09 USD
""",
        }),
    }

    override = _stock_source_workflow_override(
        objective=objective,
        allowed_tools=[{"name": "web_search"}, {"name": "web_fetch"}],
        tool_results=[tradingview_fetch],
    )

    assert override["status"] == "partial"
    assert override["selected_tools"] == ["web_search"]
    assert override["tickers"] == ["MEI", "DOMO", "GEG"]
    assert override["argument_clues"]["web_search"] == "MEI stock news June 13 2026"


def test_stock_source_workflow_locks_tickers_from_real_tradingview_fetch_payload():
    from run_engine.engine import _stock_source_workflow_override

    objective = (
        "Search top 3 stocks today with major jumps web search those 3 stocks separately "
        "and capture the 3 relevant for each results web fetch all 9 urls one by one."
    )
    tradingview_fetch = {
        "tool_name": "web_fetch",
        "success": True,
        "arguments": {"url": "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers"},
        "content": json.dumps({
            "type": "web_fetch_result",
            "url": "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers",
            "final_url": "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/",
            "host": "www.tradingview.com",
            "backend": "httpx-html",
            "content": """
Top Gaining US Stocks — TradingView
US stocks that increased the most in price
Symbol Chg % Price Vol Rel vol Mkt cap P/E EPS dil TTM Sector Analyst rating
LNKS Linkers Industries Limited
+67.50% 2.68 USD 69.88 M 51.24 4.75 M USD — −28.06 USD Electronic technology No rating
CAST FreeCast, Inc.
+56.70% 8.07 USD 113.02 M 2.77 333.63 M USD Technology services Strong buy
BFLY Butterfly Network, Inc.
+55.87% 8.90 USD 60.48 M 9.79 2.33 B USD Health technology Strong buy
ATPC Agape ATP Corporation
+42.12% 3.88 USD 72.45 M
""",
            "truncated": True,
            "total_length": 14899,
            "title": "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers",
            "status_code": 200,
        }),
    }

    override = _stock_source_workflow_override(
        objective=objective,
        allowed_tools=[{"name": "web_search"}, {"name": "web_fetch"}],
        tool_results=[tradingview_fetch],
    )

    assert override["status"] == "partial"
    assert override["selected_tools"] == ["web_search"]
    assert override["tickers"] == ["LNKS", "CAST", "BFLY"]
    assert override["argument_clues"]["web_search"] == "LNKS stock news June 13 2026"


def test_stock_source_workflow_does_not_refetch_trailing_slash_movers_url():
    from run_engine.engine import _stock_source_workflow_override

    objective = (
        "Search top 3 stocks today with major jumps web search those 3 stocks separately "
        "and capture the 3 relevant for each results web fetch all 9 urls one by one."
    )
    search_result = {
        "tool_name": "web_search",
        "success": True,
        "arguments": {"query": "top 3 stocks today with major jumps"},
        "content": json.dumps({
            "type": "web_search_results",
            "results": [
                {
                    "title": "Top Gaining US Stocks - TradingView",
                    "url": "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers",
                    "snippet": "MEIMethode Electronics, Inc. D · +21.18%",
                }
            ],
        }),
    }
    fetched_result = {
        "tool_name": "web_fetch",
        "success": True,
        "arguments": {"url": "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/"},
        "content": "US stocks that increased the most in price\nMEIMethode Electronics, Inc. D · +21.18%\nDOMODomo, Inc. D · +21.00%\nGEGGreat Elm Group, Inc. D · +19.43%",
    }

    override = _stock_source_workflow_override(
        objective=objective,
        allowed_tools=[{"name": "web_search"}, {"name": "web_fetch"}],
        tool_results=[search_result, fetched_result],
    )

    assert override["selected_tools"] == ["web_search"]
    assert override["strategy"] == "Search MEI separately before fetching sources."
    assert override["argument_clues"]["web_search"] == "MEI stock news June 13 2026"


def test_stock_source_workflow_fetches_collected_urls_before_finalizing():
    from run_engine.engine import _stock_source_workflow_override

    objective = (
        "Search top 3 stocks today with major jumps web search those 3 stocks separately "
        "and capture the 3 relevant for each results web fetch all 9 urls one by one."
    )
    tool_results = []
    tool_results.append({
        "tool_name": "web_fetch",
        "success": True,
        "arguments": {"url": "https://www.morningstar.com/markets/movers"},
        "content": """
## Gainers
| 1-Day Chart | Stock | Price | Volume/Average |
| --- | --- | --- | --- |
| | [Materialise NV](https://www.morningstar.com/stocks/xnas/mtls/quote) MTLS | $10.00 +2.00 (25.00%) | 1M 2M |
| | [Luxfer Holdings](https://www.morningstar.com/stocks/xnys/lxfr/quote) LXFR | $11.00 +2.00 (22.00%) | 1M 2M |
| | [ARKO Corp](https://www.morningstar.com/stocks/xnas/arko/quote) ARKO | $12.00 +2.00 (20.00%) | 1M 2M |
## Losers
""",
    })
    for ticker in ("MTLS", "LXFR", "ARKO"):
        tool_results.append({
            "tool_name": "web_search",
            "success": True,
            "arguments": {"query": f"{ticker} stock news June 13 2026"},
            "content": json.dumps({
                "results": [
                    {"title": f"{ticker} source A", "url": f"https://news.example/{ticker.lower()}-a"},
                    {"title": f"{ticker} source B", "url": f"https://news.example/{ticker.lower()}-b"},
                    {"title": f"{ticker} source C", "url": f"https://news.example/{ticker.lower()}-c"},
                ]
            }),
        })

    override = _stock_source_workflow_override(
        objective=objective,
        allowed_tools=[{"name": "web_search"}, {"name": "web_fetch"}],
        tool_results=tool_results,
    )

    assert override["status"] == "partial"
    assert override["selected_tools"] == ["web_fetch"]
    assert override["argument_clues"]["web_fetch"].startswith("https://news.example/")


def test_stock_source_workflow_gets_enough_default_tool_turns():
    from run_engine.engine import _default_tool_turn_budget, _source_collection_contract_from_objective

    objective = (
        "Search top 3 stocks today with major jumps web search those 3 stocks separately "
        "and capture the 3 relevant for each results web fetch all 9 urls one by one."
    )

    assert _source_collection_contract_from_objective(objective) == {
        "entity_count": 3,
        "urls_per_entity": 3,
        "total_urls": 9,
    }
    assert _default_tool_turn_budget(objective, None) >= 14
    assert _default_tool_turn_budget("hello", None) == 3
    assert _default_tool_turn_budget(objective, 5) == 5


def test_source_collection_contract_is_topic_agnostic():
    from run_engine.engine import _source_collection_contract_from_objective

    objective = (
        "Find top 4 budget laptops, search each separately, collect 2 review URLs "
        "for each, web fetch all 8 URLs, then summarize."
    )

    assert _source_collection_contract_from_objective(objective) == {
        "entity_count": 4,
        "urls_per_entity": 2,
        "total_urls": 8,
    }


# ── Slice 3: deterministic extractor canonical-predicate coverage ────────


def test_extractor_handles_call_me_phrase_with_canonical_name():
    from engine.deterministic_facts import extract_user_stated_fact_updates

    facts, voids = extract_user_stated_fact_updates("Call me Shovon")
    assert any(f["predicate"] == "preferred_name" and "shovon" in f["object"].lower() for f in facts)


def test_extractor_handles_environment_correction():
    from engine.deterministic_facts import extract_user_stated_fact_updates

    facts, _ = extract_user_stated_fact_updates("Use prod not dev for this run")
    assert any(f["predicate"] == "environment_mode" and f["object"].lower() in {"prod", "production"} for f in facts)


def test_extractor_handles_location_move():
    from engine.deterministic_facts import extract_user_stated_fact_updates

    facts, _ = extract_user_stated_fact_updates("I moved to Berlin last week")
    assert any(f["predicate"] == "location" and "berlin" in f["object"].lower() for f in facts)


def test_extractor_handles_budget_declaration():
    from engine.deterministic_facts import extract_user_stated_fact_updates

    facts, _ = extract_user_stated_fact_updates("My budget is $500 for this experiment")
    assert any(f["predicate"] == "budget_limit" and "500" in f["object"] for f in facts)


def test_extractor_handles_explicit_revocation_when_fact_present():
    from engine.deterministic_facts import extract_user_stated_fact_updates

    current = [("User", "preferred_editor", "Cursor")]
    _, voids = extract_user_stated_fact_updates(
        "Forget my editor preference",
        current_facts=current,
    )
    assert any(v.get("predicate") == "preferred_editor" for v in voids)


def test_extractor_returns_empty_for_irrelevant_message():
    from engine.deterministic_facts import extract_user_stated_fact_updates

    facts, voids = extract_user_stated_fact_updates("What's the weather today?")
    assert facts == []
    assert voids == []


def test_extractor_canonical_predicate_normalization():
    """Predicates emitted by the extractor must match direct_fact_policy.py exactly."""
    from engine.deterministic_facts import extract_user_stated_fact_updates
    from engine.direct_fact_policy import normalize_memory_predicate

    facts, _ = extract_user_stated_fact_updates(
        "Call me Shovon, I work at Anthropic, and I use Cursor"
    )
    for fact in facts:
        # Each emitted predicate must already be canonical (idempotent under normalize).
        assert normalize_memory_predicate(fact["predicate"]) == fact["predicate"], (
            f"Non-canonical predicate emitted: {fact['predicate']!r}"
        )


# ── Slice 4: tool result shaping ──────────────────────────────────────────


def test_shape_tool_result_short_bash_unchanged():
    from engine.tool_contract import shape_tool_result_for_actor

    short = "ok\nall good"
    shaped, summary = shape_tool_result_for_actor("bash", short, success=True)
    assert shaped == short
    assert summary.startswith("[bash ok]")


def test_shape_tool_result_long_bash_truncates_with_key_lines():
    from engine.tool_contract import shape_tool_result_for_actor

    body = "first line ok\n" + ("filler line\n" * 200) + "ERROR: something broke at /var/lib/foo\n" + ("more filler\n" * 200) + "tail line"
    shaped, summary = shape_tool_result_for_actor("bash", body, success=False)
    assert "[KEY_LINES]" in shaped
    assert "ERROR" in shaped
    assert "chars omitted" in shaped
    assert summary.startswith("[bash fail]")
    assert len(shaped) < len(body)


def test_shape_tool_result_hard_failure_pinned_first_and_untruncated():
    from engine.tool_contract import shape_tool_result_for_actor

    body = "x" * 5000 + "\nHARD_FAILURE: write target missing\n" + "y" * 5000
    shaped, summary = shape_tool_result_for_actor("bash", body, success=False)
    assert shaped.startswith("[HARD_FAILURE — do not claim this action succeeded]")
    assert "HARD_FAILURE" in summary
    # Body must be preserved verbatim — no truncation on HARD_FAILURE
    assert len(shaped) >= len(body)


def test_shape_tool_result_web_search_summary_block():
    from engine.tool_contract import shape_tool_result_for_actor

    body = (
        "First Result Title\n"
        "Some snippet text\n"
        "https://example.com/1\n"
        "Second Result Title\n"
        "Another snippet\n"
        "https://example.com/2\n"
    )
    shaped, summary = shape_tool_result_for_actor("web_search", body, success=True)
    assert "[SUMMARY]" in shaped
    assert "First Result Title" in shaped
    # Original content still present after the summary block
    assert "Second Result Title" in shaped
    assert summary.startswith("[web_search ok]")


def test_shape_tool_result_empty_content_returns_empty_summary():
    from engine.tool_contract import shape_tool_result_for_actor

    shaped, summary = shape_tool_result_for_actor("bash", "", success=True)
    assert shaped == ""
    assert summary == ""


def test_shape_tool_result_unknown_tool_unchanged_with_summary():
    from engine.tool_contract import shape_tool_result_for_actor

    body = "weather is sunny in Berlin"
    shaped, summary = shape_tool_result_for_actor("weather_fetch", body, success=True)
    assert shaped == body
    assert "weather_fetch" in summary
    assert "sunny" in summary.lower()


# ── Slice 4: 3-way verifier verdict ──────────────────────────────────────


def _verify_with_payload(payload_text, *, route="direct_fact"):
    """Helper: run verify_with_context against a mocked adapter that returns payload_text."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from orchestration.orchestrator import AgenticOrchestrator

    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value=payload_text)
    orch = AgenticOrchestrator(adapter=adapter)
    return asyncio.run(
        orch.verify_with_context(
            query="What is the price?",
            response="It is $50.",
            tool_results=[{"tool_name": "web_search", "success": True, "content": "price page snippet"}],
            model="any",
            route_type=route,
        )
    )


def test_verifier_returns_supported_verdict_when_supported_true():
    payload = '{"supported": true, "issues": [], "confidence": 0.9}'
    result = _verify_with_payload(payload)
    assert result["supported"] is True
    assert result["verdict"] == "supported"
    assert result["target_state"] == ""


def test_verifier_infers_needs_redraft_for_legacy_unsupported():
    """Old verifier output with no verdict field defaults to needs_redraft."""
    payload = '{"supported": false, "issues": ["missing caveat"], "confidence": 0.6}'
    result = _verify_with_payload(payload)
    assert result["supported"] is False
    assert result["verdict"] == "needs_redraft"


def test_verifier_explicit_needs_replan_with_target_state():
    payload = (
        '{"supported": false, "verdict": "needs_replan", '
        '"target_state": "gather", '
        '"missing_evidence": ["pricing page not fetched"], '
        '"issues": ["claim unsupported"], "confidence": 0.2}'
    )
    result = _verify_with_payload(payload)
    assert result["verdict"] == "needs_replan"
    assert result["target_state"] == "gather"
    assert "pricing page not fetched" in result["missing_evidence"]


def test_verifier_low_confidence_escalates_to_needs_replan_on_grounded_route():
    """Low-confidence redraft on a grounded route should auto-escalate to replan."""
    payload = '{"supported": false, "issues": ["unclear claim"], "confidence": 0.15}'
    result = _verify_with_payload(payload, route="direct_fact")
    assert result["verdict"] == "needs_replan"
    assert result["target_state"] == "gather"


def test_verifier_no_evidence_on_grounded_route_returns_needs_replan():
    """No tool results on a grounded route → fail-closed needs_replan, not redraft."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from orchestration.orchestrator import AgenticOrchestrator

    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value='{}')
    orch = AgenticOrchestrator(adapter=adapter)
    result = asyncio.run(
        orch.verify_with_context(
            query="What is the price?",
            response="It is $50.",
            tool_results=[],
            model="any",
            route_type="url_fetch",
        )
    )
    assert result["verdict"] == "needs_replan"
    assert result["target_state"] == "gather"
    assert result.get("fail_closed") is True


def test_verifier_invalid_target_state_falls_back_to_gather():
    payload = (
        '{"supported": false, "verdict": "needs_replan", '
        '"target_state": "invalid_state_name", '
        '"issues": ["x"], "confidence": 0.2}'
    )
    result = _verify_with_payload(payload)
    assert result["verdict"] == "needs_replan"
    assert result["target_state"] == "gather"


# ── Slice 5: spatial-aware planning packet ───────────────────────────────


def test_packet_emits_spatial_drawers_item_when_provided():
    """The PLANNING packet must include a spatial_drawers ContextItem when
    spatial_drawers is provided. ACTING packets must NOT include it (planning-only)."""
    from types import SimpleNamespace
    from engine.context_schema import ContextPhase
    from run_engine.context_packets import PacketBuildInputs, build_phase_packet
    from run_engine.types import RunEngineRequest

    drawers = [
        {"locus_id": "european_travel", "hop": 0, "score": 1.0,
         "content": "# European Travel Plan\n## Current facts\n- Trip — starts: 2026-06-01"},
        {"locus_id": "passports", "hop": 1, "score": 0.85,
         "content": "# Passports\n## Current facts\n- Passport — expires: 2027-09-15"},
    ]

    inputs = PacketBuildInputs(
        request=RunEngineRequest(
            session_id="s1", owner_id="u1", agent_id="default",
            user_message="What's my travel plan?",
            model="llama3.2", system_prompt="You are Shovs.",
        ),
        session=SimpleNamespace(sliding_window=[]),
        phase=ContextPhase.PLANNING,
        system_prompt="You are Shovs.",
        current_context="",
        allowed_tools=[],
        tool_results=[],
        spatial_drawers=drawers,
    )
    packet = build_phase_packet(context_engine=None, inputs=inputs)
    included = [item["item_id"] for item in packet.trace["included"]]
    assert "spatial_drawers" in included
    # Content must reference the primary locus and the neighbor
    assert "european_travel" in packet.content
    assert "passports" in packet.content


def test_packet_omits_spatial_drawers_in_acting_phase():
    """spatial_drawers ContextItem has phase_visibility=PLANNING; ACTING packet excludes it."""
    from types import SimpleNamespace
    from engine.context_schema import ContextPhase
    from run_engine.context_packets import PacketBuildInputs, build_phase_packet
    from run_engine.types import RunEngineRequest

    drawers = [
        {"locus_id": "x", "hop": 0, "score": 1.0, "content": "# X\nfact"}
    ]
    inputs = PacketBuildInputs(
        request=RunEngineRequest(
            session_id="s1", owner_id="u1", agent_id="default",
            user_message="hi", model="llama3.2", system_prompt="",
        ),
        session=SimpleNamespace(sliding_window=[]),
        phase=ContextPhase.ACTING,
        system_prompt="",
        current_context="",
        allowed_tools=[],
        tool_results=[],
        spatial_drawers=drawers,
    )
    packet = build_phase_packet(context_engine=None, inputs=inputs)
    included = [item["item_id"] for item in packet.trace["included"]]
    excluded = [item["item_id"] for item in packet.trace.get("excluded", [])]
    # Either excluded explicitly or absent entirely from the included list
    assert "spatial_drawers" not in included


def test_packet_truncates_long_drawer_content():
    from types import SimpleNamespace
    from engine.context_schema import ContextPhase
    from run_engine.context_packets import PacketBuildInputs, build_phase_packet
    from run_engine.types import RunEngineRequest

    big_content = "# X\n" + ("filler line\n" * 500)
    drawers = [{"locus_id": "x", "hop": 0, "score": 1.0, "content": big_content}]
    inputs = PacketBuildInputs(
        request=RunEngineRequest(
            session_id="s1", owner_id="u1", agent_id="default",
            user_message="q", model="llama3.2", system_prompt="",
        ),
        session=SimpleNamespace(sliding_window=[]),
        phase=ContextPhase.PLANNING,
        system_prompt="",
        current_context="",
        allowed_tools=[],
        tool_results=[],
        spatial_drawers=drawers,
    )
    packet = build_phase_packet(context_engine=None, inputs=inputs)
    # Drawer content must be capped (shows truncation marker)
    assert "truncated" in packet.content.lower() or "use shovs_memory_query" in packet.content


# ── Slice 6: dispatch gate ───────────────────────────────────────────────


def test_dispatch_gate_blocks_destructive_without_auth_verb():
    """The advisory's max_tier=destructive + clear=False is the precondition
    for the dispatch gate. Verifies the upstream check produces that signal."""
    from engine.side_effect_guard import check_plan_for_side_effects

    advisory = check_plan_for_side_effects(
        user_message="explain this code",  # no auth verb
        selected_tools=["bash"],
    )
    assert advisory["max_tier"] == "destructive"
    assert advisory["clear"] is False
    assert any("destructive" in w.lower() for w in advisory["warnings"])


def test_dispatch_gate_passes_destructive_with_explicit_auth_verb():
    from engine.side_effect_guard import check_plan_for_side_effects

    advisory = check_plan_for_side_effects(
        user_message="run the migration script",  # 'run' is an auth verb
        selected_tools=["bash"],
    )
    assert advisory["max_tier"] == "destructive"
    assert advisory["clear"] is True


def test_dispatch_gate_passes_write_with_implicit_auth():
    """'fix the bug' implies write authorization; should pass without warning."""
    from engine.side_effect_guard import check_plan_for_side_effects

    advisory = check_plan_for_side_effects(
        user_message="fix the auth bug in login.py",
        selected_tools=["file_str_replace"],
    )
    assert advisory["max_tier"] == "write"
    assert advisory["clear"] is True


def test_dispatch_gate_silent_for_read_only():
    from engine.side_effect_guard import check_plan_for_side_effects

    advisory = check_plan_for_side_effects(
        user_message="what is the weather",
        selected_tools=["web_search"],
    )
    assert advisory["max_tier"] == "read_only"
    assert advisory["clear"] is True
    assert advisory["warnings"] == []
