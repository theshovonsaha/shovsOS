from unittest.mock import MagicMock

from engine.context_compiler import compile_context_items
from engine.context_schema import ContextItem, ContextKind, ContextPhase
from engine.core import AgentCore, _classify_model_profile, SYSTEM_PROMPT_CHAR_BUDGET


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
