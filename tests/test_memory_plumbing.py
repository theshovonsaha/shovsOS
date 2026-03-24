import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from engine import core as core_mod
from engine.core import AgentCore, DEFAULT_SYSTEM_PROMPT, _classify_route, _enforce_total_budget
from engine.context_engine_v2 import ContextEngineV2
from orchestration.orchestrator import AgenticOrchestrator


@pytest.mark.asyncio
async def test_orchestrator_only_injects_query_memory_on_memory_signal():
    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value='{"tools": []}')
    orch = AgenticOrchestrator(adapter=adapter)

    tools_list = [
        {"name": "query_memory", "description": "Query prior conversation context"},
        {"name": "web_search", "description": "Search web"},
    ]

    soft_ack = await orch.plan_with_context(
        query="yes, do it",
        tools_list=tools_list,
        model="llama3.2",
        session_has_history=True,
        current_fact_count=3,
    )
    assert all(t["name"] != "query_memory" for t in soft_ack["tools"])

    explicit_memory = await orch.plan_with_context(
        query="as i mentioned earlier, my name is Shovon",
        tools_list=tools_list,
        model="llama3.2",
        session_has_history=True,
        current_fact_count=3,
    )
    assert any(t["name"] == "query_memory" for t in explicit_memory["tools"])


def test_total_budget_preserves_tool_result_messages():
    messages = [
        {"role": "system", "content": "s" * 1800},
        {"role": "assistant", "content": "<SYSTEM_TOOL_RESULT name=\"web_search\">result</SYSTEM_TOOL_RESULT>"},
        {"role": "assistant", "content": "a" * 1800},
        {"role": "user", "content": "latest prompt"},
    ]

    original_get_encoding = core_mod._get_token_encoding
    original_limit = core_mod.TOKEN_SAFETY_LIMITS.get("tiny-test")
    try:
        core_mod._get_token_encoding = lambda: None
        core_mod.TOKEN_SAFETY_LIMITS["tiny-test"] = 200
        trimmed = _enforce_total_budget(messages, "tiny-test")
    finally:
        core_mod._get_token_encoding = original_get_encoding
        if original_limit is None:
            del core_mod.TOKEN_SAFETY_LIMITS["tiny-test"]
        else:
            core_mod.TOKEN_SAFETY_LIMITS["tiny-test"] = original_limit

    assert any("SYSTEM_TOOL_RESULT" in m.get("content", "") for m in trimmed)
    assert not any(m.get("role") == "assistant" and m.get("content", "") == "a" * 1800 for m in trimmed)


def test_strip_internal_execution_chatter_removes_echoed_system_packets():
    raw = (
        "<SYSTEM_EVIDENCE_PACKET>Tool: web_search</SYSTEM_EVIDENCE_PACKET>\n"
        "<system_observation>Recent tool digest</system_observation>\n"
        "Execution budget for additional tool steps has been reached.\n"
        "Final concise answer."
    )

    cleaned = AgentCore._strip_internal_execution_chatter(raw, has_tool_results=True)

    assert "SYSTEM_EVIDENCE_PACKET" not in cleaned
    assert "system_observation" not in cleaned
    assert "Final concise answer." in cleaned


def test_build_messages_deduplicates_fact_covered_anchor():
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    agent.ctx_eng.build_context_block.return_value = ""
    agent.tools.build_tools_block.return_value = ""

    messages = agent._build_messages(
        system_prompt="sys",
        context="",
        sliding_window=[],
        user_message="hello",
        current_facts=[("User", "likes", "blue")],
        unified_hits=[
            {"key": "Color Pref", "anchor": "User likes blue"},
            {"key": "Weather Note", "anchor": "User lives in Toronto"},
        ],
        ctx_engine=agent.ctx_eng,
    )

    system_content = messages[0]["content"]
    assert "Weather Note" in system_content
    assert "Color Pref" not in system_content


def test_context_v2_protected_modules_are_not_voided():
    ctx = ContextEngineV2(adapter=MagicMock())
    ctx._modules = {
        "User preferred name": {"content": "User preferred name is Shovon", "goals": {"session"}, "hit_count": 1, "created_turn": 1, "protected": True},
        "Temp detail": {"content": "One-time detail", "goals": {"session"}, "hit_count": 1, "created_turn": 1, "protected": False},
    }

    void_records = ctx._apply_voids(["User preferred name", "Temp detail"])

    assert "User preferred name" in ctx._modules
    assert "Temp detail" not in ctx._modules
    assert void_records == [{"subject": "Temp detail", "predicate": "requires"}]


@pytest.mark.asyncio
async def test_chat_stream_uses_planner_fallback_model_by_default():
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=type("AsyncIter", (), {
        "__aiter__": lambda self: self,
        "__anext__": AsyncMock(side_effect=["hello", StopAsyncIteration]),
    })())

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(return_value={
        "strategy": "Use web_search",
        "tools": [],
        "force_memory": False,
        "confidence": 0.4,
    })
    orchestrator.set_adapter = MagicMock()

    session_mgr = MagicMock()
    session = MagicMock()
    session.id = "planner-fallback"
    session.agent_id = "default"
    session.model = "llama3.2"
    session.system_prompt = "sys"
    session.compressed_context = ""
    session.sliding_window = []
    session.first_message = None
    session.message_count = 0
    session.lock = asyncio.Lock()
    session_mgr.get_or_create.return_value = session

    tools = MagicMock()
    tools.list_tools.return_value = [{"name": "web_search", "description": "search"}]
    tools.has_tools.return_value = False
    tools.build_tools_block.return_value = ""

    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=session_mgr,
        tool_registry=tools,
        orchestrator=orchestrator,
    )

    events = [event async for event in core.chat_stream("research the market and summarize")]

    assert any(event["type"] == "done" for event in events)
    assert orchestrator.plan_with_context.await_args.kwargs["model"] == core_mod.DEFAULT_PLANNER_FALLBACK_MODEL


@pytest.mark.asyncio
async def test_chat_stream_resolves_context_engine_per_request_without_mutating_shared_state():
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=type("AsyncIter", (), {
        "__aiter__": lambda self: self,
        "__anext__": AsyncMock(side_effect=["hello", StopAsyncIteration]),
    })())

    original_ctx = MagicMock()
    original_ctx.build_context_block.return_value = ""
    original_ctx.compress_exchange = AsyncMock(return_value=("", [], []))
    original_ctx.set_adapter = MagicMock()

    resolved_ctx = MagicMock()
    resolved_ctx.build_context_block.return_value = ""
    resolved_ctx.compress_exchange = AsyncMock(return_value=("", [], []))
    resolved_ctx.set_adapter = MagicMock()

    session_mgr = MagicMock()
    session = MagicMock()
    session.id = "ctx-local"
    session.agent_id = "default"
    session.model = "llama3.2"
    session.system_prompt = "sys"
    session.compressed_context = ""
    session.sliding_window = []
    session.first_message = None
    session.message_count = 0
    session.context_mode = "v2"
    session.lock = asyncio.Lock()
    session_mgr.get_or_create.return_value = session
    session_mgr.append_message.return_value = True
    session_mgr.update_context = MagicMock()

    tools = MagicMock()
    tools.has_tools.return_value = False
    tools.build_tools_block.return_value = ""

    core = AgentCore(adapter, original_ctx, session_mgr, tools)
    core._resolve_context_engine = MagicMock(return_value=resolved_ctx)

    events = [event async for event in core.chat_stream("please summarize the current state")]
    assert any(event["type"] == "done" for event in events)
    assert core.ctx_eng is original_ctx
    resolved_ctx.set_adapter.assert_called_once()
    resolved_ctx.compress_exchange.assert_awaited()


def test_build_messages_enforces_system_prompt_budget():
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    agent.ctx_eng.build_context_block.return_value = "--- Session Memory ---\n" + ("m" * 4000)
    agent.tools.build_tools_block.return_value = "--- Tools ---\n" + ("t" * 5000)

    messages = agent._build_messages(
        system_prompt="sys" + ("s" * 3000),
        context="ctx",
        sliding_window=[],
        user_message="hello",
        unified_hits=[{"key": "Huge", "anchor": "a" * 5000, "metadata": {}}],
        route_type="open_ended",
        ctx_engine=agent.ctx_eng,
    )

    assert len(messages[0]["content"]) <= core_mod.SYSTEM_PROMPT_CHAR_BUDGET + 200


def test_short_greeting_with_address_classifies_as_trivial_chat():
    route = _classify_route(
        "hey bro",
        session_has_history=False,
        current_fact_count=0,
        active_task_count=0,
    )
    assert route == "trivial_chat"


def test_build_messages_adds_plaintext_guard_for_conversational_turn():
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    agent.ctx_eng.build_context_block.return_value = ""
    agent.tools.build_tools_block.return_value = ""

    messages = agent._build_messages(
        system_prompt="sys",
        context="",
        sliding_window=[],
        user_message="hey bro",
        route_type="trivial_chat",
        ctx_engine=agent.ctx_eng,
    )

    assert "Conversational Response Guard" in messages[0]["content"]
    assert "Respond in normal plain text" in messages[0]["content"]


def test_default_system_prompt_matches_language_os_runtime_contract():
    assert "Language OS assistant" in DEFAULT_SYSTEM_PROMPT
    assert "output ONLY one valid JSON tool call" in DEFAULT_SYSTEM_PROMPT
    assert "Do not silently rename or broaden the target" in DEFAULT_SYSTEM_PROMPT
    assert "Shovs Agent Platform" not in DEFAULT_SYSTEM_PROMPT
    assert "You may chain multiple tool calls" not in DEFAULT_SYSTEM_PROMPT
    assert "Wrap visual data in ```html" not in DEFAULT_SYSTEM_PROMPT


def test_build_messages_adds_entity_fidelity_and_loop_contract_for_nontrivial_turn():
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    agent.ctx_eng.build_context_block.return_value = ""
    agent.tools.build_tools_block.return_value = ""

    messages = agent._build_messages(
        system_prompt="sys",
        context="",
        sliding_window=[],
        user_message="research wigglebudget.com and compare it",
        route_type="multi_step",
        ctx_engine=agent.ctx_eng,
    )

    assert "Entity Fidelity" in messages[0]["content"]
    assert "Preserve the user's exact entities" in messages[0]["content"]
    assert "Loop Contract" in messages[0]["content"]
    assert "either emit one valid JSON tool call or answer the user directly" in messages[0]["content"]


def test_build_messages_injects_profile_bootstrap_docs(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("Agent behavior: act like a precise researcher.", encoding="utf-8")
    (workspace / "IDENTITY.md").write_text("Identity: OpenClaw is an example agent on the platform.", encoding="utf-8")

    agent = AgentCore(
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        workspace_path=str(workspace),
        bootstrap_files=["AGENTS.md", "IDENTITY.md", "MISSING.md"],
        bootstrap_max_chars=1200,
    )
    agent.ctx_eng.build_context_block.return_value = ""
    agent.tools.build_tools_block.return_value = ""

    messages = agent._build_messages(
        system_prompt="sys",
        context="",
        sliding_window=[],
        user_message="research this",
        route_type="multi_step",
        ctx_engine=agent.ctx_eng,
    )

    assert "Agent Bootstrap" in messages[0]["content"]
    assert "AGENTS.md" in messages[0]["content"]
    assert "IDENTITY.md" in messages[0]["content"]
    assert "act like a precise researcher" in messages[0]["content"]
    assert "OpenClaw is an example agent on the platform" in messages[0]["content"]
