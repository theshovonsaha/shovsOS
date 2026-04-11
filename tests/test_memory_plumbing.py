import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from engine.compression_fact_policy import finalize_compression_fact_records
from engine import core as core_mod
from engine.direct_fact_policy import should_answer_direct_fact_from_memory
from engine.deterministic_facts import extract_user_stated_fact_updates, is_redundant_user_alias_text
from engine.core import AgentCore, DEFAULT_SYSTEM_PROMPT, _classify_route, _enforce_total_budget
from engine.context_engine_v2 import ContextEngineV2
from orchestration.orchestrator import AgenticOrchestrator
from plugins.tool_registry import Tool, ToolRegistry


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


@pytest.mark.asyncio
async def test_orchestrator_memory_recall_route_queries_memory_even_without_active_history():
    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value='{"tools": []}')
    orch = AgenticOrchestrator(adapter=adapter)

    result = await orch.plan_with_context(
        query="Do you remember me?",
        tools_list=[
            {"name": "query_memory", "description": "Query prior conversation context"},
            {"name": "web_search", "description": "Search web"},
        ],
        model="llama3.2",
        session_has_history=False,
        current_fact_count=0,
    )

    assert result["route_type"] == "memory_recall"
    assert result["should_plan"] is False
    assert [tool["name"] for tool in result["tools"]] == ["query_memory"]
    adapter.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_returns_no_tools_for_conversational_greeting():
    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value='{"tools": [{"name": "web_search"}]}')
    orch = AgenticOrchestrator(adapter=adapter)

    tools_list = [
        {"name": "web_search", "description": "Search web"},
        {"name": "weather_fetch", "description": "Get weather"},
    ]

    result = await orch.plan_with_context(
        query="Hello, how are you today?",
        tools_list=tools_list,
        model="llama3.2",
        session_has_history=False,
        current_fact_count=0,
    )

    assert result["route_type"] == "trivial_chat"
    assert result["tools"] == []


@pytest.mark.asyncio
async def test_orchestrator_research_report_prefers_evidence_before_file_creation():
    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value='{"strategy":"Create report","tools":[{"name":"file_create","priority":"high","reason":"Need report file"}]}')
    orch = AgenticOrchestrator(adapter=adapter)

    result = await orch.plan_with_context(
        query="Research OpenClaw properly and write a report",
        tools_list=[
            {"name": "web_search", "description": "Search web"},
            {"name": "file_create", "description": "Create file"},
        ],
        model="llama3.2",
        session_has_history=False,
        current_fact_count=0,
    )

    tool_names = [tool["name"] for tool in result["tools"]]
    assert tool_names[0] == "web_search"
    assert "file_create" not in tool_names


@pytest.mark.asyncio
async def test_orchestrator_gather_intel_falls_back_to_web_search():
    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value='{"tools": []}')
    orch = AgenticOrchestrator(adapter=adapter)

    result = await orch.plan_with_context(
        query="What do you know about OpenClaw, can you gather intel?",
        tools_list=[
            {"name": "web_search", "description": "Search web"},
            {"name": "file_create", "description": "Create file"},
        ],
        model="llama3.2",
        session_has_history=False,
        current_fact_count=0,
    )

    assert result["route_type"] == "multi_step"
    assert [tool["name"] for tool in result["tools"]] == ["web_search"]


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


def test_direct_fact_memory_guard_prefers_trusted_facts_over_tools():
    assert should_answer_direct_fact_from_memory(
        "What is my preferred name and current location?",
        [
            ("User", "preferred_name", "Shovon"),
            ("User", "location", "Toronto"),
        ],
    ) is True

    assert should_answer_direct_fact_from_memory(
        "What is my preferred name and current location?",
        [("User", "preferred_name", "Shovon")],
    ) is False


def test_deterministic_fact_extractor_supports_more_user_preferences():
    facts, voids = extract_user_stated_fact_updates(
        "My timezone is EST. I use Cursor. My package manager is pnpm. "
        "My primary language is TypeScript. I prefer concise responses. I am on Linux. My pronouns are he/him."
    )

    assert voids == []
    pairs = {(item["predicate"], item["object"]) for item in facts}
    assert ("timezone", "EST") in pairs
    assert ("preferred_editor", "Cursor") in pairs
    assert ("package_manager", "pnpm") in pairs
    assert ("primary_language", "TypeScript") in pairs
    assert ("response_verbosity", "concise") in pairs
    assert ("operating_system", "Linux") in pairs
    assert ("pronouns", "he/him") in pairs


def test_deterministic_fact_extractor_captures_task_state_and_constraints():
    facts, voids = extract_user_stated_fact_updates(
        "Use staging environment mode. Keep scope to engine/ and tests/. "
        "Keep it under two hours. Do not use web_search. Follow up tomorrow on packaging."
    )

    assert voids == []
    pairs = {(item["predicate"], item["object"]) for item in facts}
    assert ("environment_mode", "staging") in pairs
    assert ("scope_boundary", "engine/") in pairs
    assert ("budget_limit", "two hours") in pairs
    assert ("task_constraint", "Do not use web_search") in pairs
    assert ("followup_directive", "Follow up tomorrow on packaging") in pairs


def test_direct_fact_memory_guard_supports_preference_queries_beyond_name_and_location():
    current_facts = [
        ("User", "timezone", "EST"),
        ("User", "preferred_editor", "Cursor"),
        ("User", "package_manager", "pnpm"),
        ("User", "primary_language", "TypeScript"),
        ("User", "response_verbosity", "concise"),
        ("User", "operating_system", "Linux"),
        ("User", "pronouns", "he/him"),
    ]

    assert should_answer_direct_fact_from_memory("What is my timezone?", current_facts) is True
    assert should_answer_direct_fact_from_memory("Which editor do I use?", current_facts) is True
    assert should_answer_direct_fact_from_memory("What package manager do I use?", current_facts) is True
    assert should_answer_direct_fact_from_memory("What is my primary language?", current_facts) is True
    assert should_answer_direct_fact_from_memory("What response style do I prefer?", current_facts) is True
    assert should_answer_direct_fact_from_memory("What operating system do I use?", current_facts) is True
    assert should_answer_direct_fact_from_memory("What are my pronouns?", current_facts) is True


def test_direct_fact_memory_guard_supports_task_state_queries():
    current_facts = [
        ("Task", "environment_mode", "staging"),
        ("Task", "scope_boundary", "engine/"),
        ("Task", "budget_limit", "two hours"),
        ("Task", "task_constraint", "Do not use web_search"),
        ("Task", "followup_directive", "Follow up tomorrow on packaging"),
    ]

    assert should_answer_direct_fact_from_memory("Which environment did I ask for?", current_facts) is True
    assert should_answer_direct_fact_from_memory("What scope did I set?", current_facts) is True
    assert should_answer_direct_fact_from_memory("What budget did I set?", current_facts) is True
    assert should_answer_direct_fact_from_memory("What constraints did I set?", current_facts) is True
    assert should_answer_direct_fact_from_memory("What follow up directive did I set?", current_facts) is True


def test_finalize_compression_fact_records_blocks_alias_noise_after_grounding():
    allowed, blocked = finalize_compression_fact_records(
        [
            {"subject": "Shovon", "predicate": "lives_in", "object": "Vancouver", "fact": "Shovon lives in Vancouver"},
            {"subject": "Tool", "predicate": "source", "object": "SEC filing", "fact": "Tool source SEC filing"},
        ],
        user_message="My name is Shovon and I live in Vancouver.",
        grounding_text="Company facts from SEC filing",
        deterministic_facts=[
            {"subject": "User", "predicate": "preferred_name", "object": "Shovon"},
            {"subject": "User", "predicate": "location", "object": "Vancouver"},
        ],
        current_facts=[("User", "preferred_name", "Shovon"), ("User", "location", "Vancouver")],
    )

    assert any(item["subject"] == "Tool" for item in allowed)
    assert any(item["fact"] == "Shovon lives in Vancouver" for item in blocked)


def test_retrieval_alias_text_filter_blocks_named_subject_redundancy():
    assert is_redundant_user_alias_text(
        "Shovon lives in Vancouver",
        current_facts=[
            ("User", "preferred_name", "Shovon"),
            ("User", "location", "Vancouver"),
        ],
    ) is True

    assert is_redundant_user_alias_text(
        "Tool source SEC filing",
        current_facts=[
            ("User", "preferred_name", "Shovon"),
            ("User", "location", "Vancouver"),
        ],
    ) is False


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


@pytest.mark.asyncio
async def test_agent_loop_blocks_store_memory_on_deterministic_fact_turn():
    class AsyncIter:
        def __init__(self, chunks):
            self._iter = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    adapter = MagicMock()
    adapter.stream = MagicMock(side_effect=[
        AsyncIter(['{"tool":"store_memory","arguments":{"subject":"User","predicate":"location","object_":"Berlin"}}']),
        AsyncIter(["Your current location is Berlin."]),
    ])

    registry = ToolRegistry()
    blocked_handler = AsyncMock(return_value="should not run")
    registry.register(
        Tool(
            name="store_memory",
            description="Store memory",
            parameters={"type": "object", "properties": {"subject": {"type": "string"}}},
            handler=blocked_handler,
        )
    )

    core = AgentCore(
        adapter=adapter,
        context_engine=MagicMock(),
        session_manager=MagicMock(),
        tool_registry=registry,
        orchestrator=None,
    )

    events = [
        event
        async for event in core._agent_loop(
            model="llama3.2",
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "I moved to Berlin."}],
            adapter=adapter,
            session_id="blocked-store-memory",
            route_type="open_ended",
            user_message="I moved to Berlin.",
            blocked_tools={"store_memory"},
        )
    ]

    assert not any(event.get("type") == "tool_call" and event.get("tool_name") == "store_memory" for event in events)
    assert blocked_handler.await_count == 0
    assert any(event.get("type") == "token" for event in events)


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
