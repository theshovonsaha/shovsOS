from types import SimpleNamespace
from unittest.mock import MagicMock

from engine.context_item_builders import (
    AVAILABLE_TOOLS_PHASES,
    CANDIDATE_CONTEXT_PHASES,
    CONVERSATION_TENSION_PHASES,
    DETERMINISTIC_FACT_PHASES,
    LOOP_CONTRACT_PHASES,
    MEMORY_AUTHORITY_PHASES,
    SESSION_ANCHOR_PHASES,
    WORKING_EVIDENCE_PHASES,
    build_available_tools_item,
    build_candidate_context_item,
    build_conversation_tension_item,
    build_deterministic_facts_item,
    build_loop_contract_item,
    build_memory_authority_item,
    build_session_anchor_item,
    build_working_evidence_item,
)
from engine.context_schema import ContextPhase
from engine.core import AgentCore
from orchestration.run_store import LoopCheckpoint
from run_engine.context_packets import PacketBuildInputs, build_phase_packet
from run_engine.types import RunEngineRequest


def test_shared_context_item_builders_expose_canonical_phase_visibility():
    loop_contract = build_loop_contract_item(
        source="runtime",
        trace_id="test:loop_contract",
    )
    session_anchor = build_session_anchor_item(
        first_message="hello",
        message_count=4,
        source="session",
        trace_id="test:session_anchor",
    )
    facts = build_deterministic_facts_item(
        facts=[("User", "preferred_name", "Alex")],
        source="memory",
        trace_id="test:facts",
    )
    candidate = build_candidate_context_item(
        candidate_context="Candidate: User may prefer Alex",
        source="session",
        trace_id="test:candidate",
    )
    tension = build_conversation_tension_item(
        content="Drift: preferred name differs.",
        source="runtime",
        trace_id="test:tension",
    )
    authority = build_memory_authority_item(
        correction_turn=True,
        direct_fact_memory_only=False,
        source="runtime",
        trace_id="test:authority",
    )
    evidence = build_working_evidence_item(
        content="- web_fetch [ok]: homepage",
        source="runtime",
        trace_id="test:evidence",
    )
    tools = build_available_tools_item(
        content="- web_search: Search the web",
        source="tools",
        trace_id="test:tools",
    )

    assert loop_contract is not None
    assert session_anchor is not None
    assert facts is not None
    assert candidate is not None
    assert tension is not None
    assert authority is not None
    assert evidence is not None
    assert tools is not None

    assert loop_contract.phase_visibility == LOOP_CONTRACT_PHASES
    assert session_anchor.phase_visibility == SESSION_ANCHOR_PHASES
    assert facts.phase_visibility == DETERMINISTIC_FACT_PHASES
    assert candidate.phase_visibility == CANDIDATE_CONTEXT_PHASES
    assert tension.phase_visibility == CONVERSATION_TENSION_PHASES
    assert authority.phase_visibility == MEMORY_AUTHORITY_PHASES
    assert evidence.phase_visibility == WORKING_EVIDENCE_PHASES
    assert tools.phase_visibility == AVAILABLE_TOOLS_PHASES


def test_candidate_context_is_hidden_in_response_phase():
    candidate = build_candidate_context_item(
        candidate_context="Candidate: User may be in Berlin",
        source="session",
        trace_id="test:candidate",
    )

    assert candidate is not None
    assert ContextPhase.RESPONSE not in candidate.phase_visibility


def test_managed_and_legacy_context_surfaces_share_canonical_item_titles():
    checkpoint = LoopCheckpoint(
        checkpoint_id=1,
        run_id="run-1",
        phase="observation",
        tool_turn=1,
        status="continue",
        strategy="Use the fetched homepage as evidence.",
        notes="Prefer exact-domain evidence.",
        tool_results=[
            {
                "tool_name": "web_fetch",
                "success": True,
                "content": '{"type":"web_fetch_result","url":"https://example.com","title":"Example","content":"Example body"}',
                "arguments": {"url": "https://example.com"},
            }
        ],
    )

    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    agent.ctx_eng.build_context_items.return_value = []
    agent.ctx_eng.build_context_block.return_value = ""
    agent.tools.build_tools_block.return_value = "--- Available Tools ---\n- web_search: Search the web\n--- End Available Tools ---"

    legacy_compiled = agent._compile_phase_context(
        phase=ContextPhase.ACTING,
        system_prompt="You are Shovs.",
        context="",
        user_message="Investigate example.com and summarize it.",
        first_message="Investigate example.com and summarize it.",
        message_count=4,
        current_facts=[("User", "preferred_name", "Alex")],
        candidate_context="Candidate: User may be in Berlin",
        loop_checkpoint=checkpoint,
        ctx_engine=agent.ctx_eng,
        conversation_tension=None,
    )

    request = RunEngineRequest(
        session_id="session-1",
        owner_id="owner-1",
        agent_id="default",
        user_message="Investigate example.com and summarize it.",
        model="llama3.2",
        system_prompt="You are Shovs.",
        allowed_tools=("web_search",),
    )
    session = SimpleNamespace(
        first_message="Investigate example.com and summarize it.",
        message_count=4,
        candidate_context="Candidate: User may be in Berlin",
        sliding_window=[],
        full_history=[],
    )
    context_engine = MagicMock()
    context_engine.build_context_items.return_value = []

    managed_packet = build_phase_packet(
        context_engine=context_engine,
        inputs=PacketBuildInputs(
            request=request,
            session=session,
            phase=ContextPhase.ACTING,
            system_prompt="You are Shovs.",
            current_context="",
            allowed_tools=[{"name": "web_search", "description": "Search the web"}],
            tool_results=list(checkpoint.tool_results or []),
            current_facts=[("User", "preferred_name", "Alex")],
            effective_objective="Investigate example.com and summarize it.",
            tool_turn=1,
            observation_status="continue",
            observation_tools=["web_search"],
            strategy="Use the fetched homepage as evidence.",
            notes="Prefer exact-domain evidence.",
        ),
    )

    legacy_items = {item["item_id"]: item for item in legacy_compiled["included"]}
    managed_items = {item["item_id"]: item for item in managed_packet.trace["included"]}

    shared_ids = {
        "runtime_metadata",
        "core_instruction",
        "loop_contract",
        "session_anchor",
        "deterministic_facts",
        "candidate_context",
        "working_evidence",
        "available_tools",
    }

    for item_id in shared_ids:
        assert item_id in legacy_items, f"legacy missing {item_id}"
        assert item_id in managed_items, f"managed missing {item_id}"
        assert legacy_items[item_id]["title"] == managed_items[item_id]["title"]
