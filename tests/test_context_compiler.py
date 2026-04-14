from unittest.mock import MagicMock

from engine.conversation_tension import analyze_conversation_tension
from engine.context_compiler import compile_context_items
from engine.context_schema import ContextItem, ContextKind, ContextPhase
from engine.core import AgentCore, _classify_model_profile, SYSTEM_PROMPT_CHAR_BUDGET
from orchestration.run_store import LoopCheckpoint


def test_context_compiler_filters_items_by_phase_policy():
    items = [
        ContextItem(
            item_id="instruction",
            kind=ContextKind.INSTRUCTION,
            title="Instruction",
            content="Follow the system rules.",
            source="system",
            priority=10,
        ),
        ContextItem(
            item_id="working",
            kind=ContextKind.WORKING,
            title="Working State",
            content="Scratchpad state.",
            source="session",
            priority=20,
        ),
        ContextItem(
            item_id="evidence",
            kind=ContextKind.EVIDENCE,
            title="Working Evidence",
            content="Fetched source evidence.",
            source="run_engine",
            priority=25,
        ),
        ContextItem(
            item_id="meta",
            kind=ContextKind.META,
            title="Meta Context",
            content="Known vs candidate, falsifier, minimum probe.",
            source="run_engine",
            priority=27,
        ),
        ContextItem(
            item_id="tools",
            kind=ContextKind.ENVIRONMENT,
            title="Tools",
            content="Tool docs",
            source="tool_registry",
            priority=30,
        ),
    ]

    compiled = compile_context_items(
        items,
        phase=ContextPhase.PLANNING,
        char_budget=2000,
        truncate_section=lambda content, budget: content[:budget],
    )

    assert "Instruction" in compiled.content
    assert "Meta Context" in compiled.content
    assert "Working Evidence" in compiled.content
    assert "Tools" in compiled.content
    assert "Working State" not in compiled.content
    excluded = {record.item_id: record.reason for record in compiled.excluded}
    assert excluded["working"] == "kind_not_allowed"


def test_build_messages_records_typed_context_compilation_summary():
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    agent.ctx_eng.build_context_block.return_value = "--- Session Memory ---\n- user likes blue\n--- End Session Memory ---"
    agent.tools.build_tools_block.return_value = "--- Available Tools ---\nTool docs\n--- End Tools ---"

    messages = agent._build_messages(
        system_prompt="You are a helpful assistant.",
        context="- user likes blue",
        sliding_window=[],
        user_message="current message",
        first_message="hello there",
        message_count=4,
        unified_hits=[{"key": "Color Pref", "anchor": "User: I like blue", "metadata": {"fact": "likes blue"}}],
        ctx_engine=agent.ctx_eng,
    )

    trace = agent._last_context_compilation

    assert messages[0]["role"] == "system"
    assert trace["phase"] == "acting"
    assert trace["runtime_path"] == "legacy"
    assert trace["content"] == messages[0]["content"]
    assert trace["trace_scope"] == "message_prompt"
    assert trace["canonical_event"] == "compiled_context"
    assert "instruction" in trace["summary"]["included_kinds"]
    assert "memory" in trace["summary"]["included_kinds"]
    assert "environment" in trace["summary"]["included_kinds"]
    assert trace["history"]["retained_count"] == 0


def test_classify_model_profile_marks_small_local_runner():
    class OpenAIAdapter:
        def __init__(self):
            self.base_url = "http://127.0.0.1:1234/v1"

    profile = _classify_model_profile(OpenAIAdapter(), "lmstudio:qwen2.5-coder-3b-instruct-mlx")
    assert profile == "small_local"


def test_build_messages_uses_smaller_budget_for_small_local_profile():
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    agent.ctx_eng.build_context_block.return_value = "--- Session Memory ---\n" + ("m" * 8000)
    agent.tools.build_tools_block.return_value = "--- Available Tools ---\n" + ("t" * 4000)

    _ = agent._build_messages(
        system_prompt="You are a helpful assistant.",
        context="- user likes blue",
        sliding_window=[{"role": "assistant", "content": "a" * 5000}],
        user_message="current message",
        first_message="hello there",
        message_count=4,
        unified_hits=[],
        ctx_engine=agent.ctx_eng,
        model_profile="small_local",
    )

    trace = agent._last_context_compilation
    assert trace["model_profile"] == "small_local"
    assert trace["char_budget"] < SYSTEM_PROMPT_CHAR_BUDGET
    assert trace["history"]["max_chars_per_window_msg"] < 4000


def test_agent_core_phase_context_can_include_conversation_tension():
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    agent.ctx_eng.build_context_block.return_value = ""
    agent.tools.build_tools_block.return_value = ""
    tension = analyze_conversation_tension(
        user_message="Actually, call me Alex.",
        current_facts=[("User", "preferred_name", "Shovon")],
        deterministic_keyed_facts=[{"subject": "User", "predicate": "preferred_name", "object": "Alex"}],
        session_history=[{"role": "user", "content": "Call me Shovon."}],
    )

    compiled = agent._compile_phase_context(
        phase=ContextPhase.RESPONSE,
        system_prompt="You are Shovs.",
        context="",
        user_message="Actually, call me Alex.",
        current_facts=[("User", "preferred_name", "Shovon")],
        candidate_context="",
        ctx_engine=agent.ctx_eng,
        conversation_tension=tension,
    )

    assert "Conversation Tension" in compiled["content"]
    assert "Drift:" in compiled["content"]
    assert compiled["trace_scope"] == "phase_packet"
    assert compiled["canonical_event"] == "phase_context"
    assert any(item["item_id"] == "conversation_tension" for item in compiled["included"])


def test_agent_core_phase_context_can_include_working_evidence():
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    agent.ctx_eng.build_context_block.return_value = ""
    agent.tools.build_tools_block.return_value = ""
    checkpoint = LoopCheckpoint(
        checkpoint_id=1,
        run_id="run-1",
        phase="observation",
        tool_turn=1,
        status="continue",
        strategy="Use the fetched homepage as primary evidence.",
        notes="Prefer exact-domain evidence.",
        tool_results=[
            {
                "tool_name": "web_search",
                "success": True,
                "content": '{"type":"web_search_results","query":"wigglebudget.com review","results":[{"title":"Noisy review"}]}',
                "arguments": {"query": "wigglebudget.com review"},
            },
            {
                "tool_name": "web_fetch",
                "success": True,
                "content": '{"type":"web_fetch_result","url":"https://wigglebudget.com/","title":"Wiggle Budget","content":"Wiggle Budget is a personal finance app."}',
                "arguments": {"url": "https://wigglebudget.com/"},
            },
        ],
    )

    compiled = agent._compile_phase_context(
        phase=ContextPhase.RESPONSE,
        system_prompt="You are Shovs.",
        context="",
        user_message="Investigate wigglebudget.com and summarize it.",
        loop_checkpoint=checkpoint,
        ctx_engine=agent.ctx_eng,
    )

    assert "Working Evidence" in compiled["content"]
    assert "exact-target" in compiled["content"]
    evidence_item = next(item for item in compiled["included"] if item["item_id"] == "working_evidence")
    assert evidence_item["kind"] == "evidence"
    assert evidence_item["provenance"]["selected_count"] >= 1


def test_agent_core_build_messages_can_include_conversation_tension():
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    agent.ctx_eng.build_context_block.return_value = ""
    agent.tools.build_tools_block.return_value = ""
    tension = analyze_conversation_tension(
        user_message="I live in Berlin now.",
        current_facts=[("User", "location", "Toronto")],
        deterministic_keyed_facts=[{"subject": "User", "predicate": "location", "object": "Berlin"}],
        session_history=[{"role": "user", "content": "I live in Toronto."}],
    )

    messages = agent._build_messages(
        system_prompt="You are Shovs.",
        context="",
        sliding_window=[],
        user_message="I live in Berlin now.",
        current_facts=[("User", "location", "Toronto")],
        candidate_context="",
        ctx_engine=agent.ctx_eng,
        conversation_tension=tension,
    )

    assert "Conversation Tension" in messages[0]["content"]
    assert "Should Challenge: yes" in messages[0]["content"]
    assert any(item["item_id"] == "conversation_tension" for item in agent._last_context_compilation["included"])


def test_agent_core_uses_shared_context_engine_memory_fallback_shape():
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    agent.ctx_eng.build_context_items.return_value = []
    agent.ctx_eng.build_context_block.return_value = "--- Session Memory ---\nkept fact\n--- End Session Memory ---"
    agent.tools.build_tools_block.return_value = ""

    agent._build_messages(
        system_prompt="You are Shovs.",
        context="kept fact",
        sliding_window=[],
        user_message="current message",
        ctx_engine=agent.ctx_eng,
    )

    memory_item = next(
        item for item in agent._last_context_compilation["included"] if item["item_id"] == "context_governor_v1_memory"
    )
    assert memory_item["trace_id"] == "memory:context_engine:governor:v1"
    assert memory_item["source"] == "context_governor"
