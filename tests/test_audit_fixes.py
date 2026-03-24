"""
Audit Verification Tests
--------------------------
Tests for the critical fixes found during the full system audit.
Covers: dynamic adapter propagation, Gemini/Anthropic role merging,
config centralization, and system prompt alignment.
"""

import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from httpx import AsyncClient, ASGITransport


# ── 1. Gemini Consecutive Role Merging ──────────────────────────────────────

def test_gemini_merges_consecutive_same_role():
    """Gemini API rejects consecutive same-role messages. Verify merge."""
    from llm.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter.__new__(GeminiAdapter)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user",   "content": "Hello there."},
        {"role": "user",   "content": "Follow-up question."},
        {"role": "assistant", "content": "Response 1"},
        {"role": "assistant", "content": "Response 2"},
        {"role": "user",   "content": "Final question"},
    ]

    with patch.dict("sys.modules", {"google.genai": MagicMock(), "google": MagicMock()}):
        # Mock the types module that _convert_messages imports internally
        import sys
        mock_genai = MagicMock()
        mock_genai.types.Content = lambda role, parts: {"role": role, "parts": parts}
        mock_genai.types.Part = lambda text: text
        sys.modules["google.genai"] = mock_genai
        sys.modules["google.genai.types"] = mock_genai.types
        sys.modules["google"] = MagicMock()
        sys.modules["google.genai"] = mock_genai

        try:
            result = adapter._convert_messages(messages)
        finally:
            # Cleanup mocks
            for k in ["google.genai", "google.genai.types", "google"]:
                sys.modules.pop(k, None)

    roles = [r["role"] for r in result]

    # No consecutive duplicates
    for i in range(1, len(roles)):
        assert roles[i] != roles[i-1], f"Consecutive same role at index {i}: {roles}"


# ── 2. Anthropic Consecutive Role Merging ───────────────────────────────────

def test_anthropic_merges_consecutive_roles():
    """Anthropic requires strictly alternating user/assistant."""
    from llm.anthropic_adapter import AnthropicAdapter

    adapter = AnthropicAdapter.__new__(AnthropicAdapter)

    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Q1"},
        {"role": "user", "content": "Q2"},    # consecutive user
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q3"},
    ]

    msgs, system = adapter._prepare_messages(messages, images=None)

    assert system == "System prompt."

    # No consecutive same-role messages
    roles = [m["role"] for m in msgs]
    for i in range(1, len(roles)):
        assert roles[i] != roles[i-1], f"Consecutive same role at {i}: {roles}"

    # First message must be 'user'
    assert msgs[0]["role"] == "user"

    # Q1 and Q2 should be merged
    assert "Q1" in msgs[0]["content"] and "Q2" in msgs[0]["content"]


def test_anthropic_ensures_user_first():
    """If all system messages are extracted, first msg must still be user."""
    from llm.anthropic_adapter import AnthropicAdapter

    adapter = AnthropicAdapter.__new__(AnthropicAdapter)

    messages = [
        {"role": "system", "content": "System"},
        {"role": "assistant", "content": "Hi there"},  # assistant first after system removed
        {"role": "user", "content": "Hello"},
    ]

    msgs, system = adapter._prepare_messages(messages, images=None)
    assert msgs[0]["role"] == "user", "First message must be user for Anthropic"


def test_anthropic_merges_multiple_system_messages():
    """Multiple system messages should be concatenated into one."""
    from llm.anthropic_adapter import AnthropicAdapter

    adapter = AnthropicAdapter.__new__(AnthropicAdapter)

    messages = [
        {"role": "system", "content": "Part 1."},
        {"role": "system", "content": "Part 2."},
        {"role": "user", "content": "Hello"},
    ]

    msgs, system = adapter._prepare_messages(messages, images=None)
    assert "Part 1." in system and "Part 2." in system


# ── 3. Context Engine set_adapter ───────────────────────────────────────────

def test_context_engine_set_adapter():
    """ContextEngine.set_adapter should swap the underlying adapter."""
    from engine.context_engine import ContextEngine

    mock_adapter_1 = MagicMock()
    mock_adapter_2 = MagicMock()

    ce = ContextEngine(adapter=mock_adapter_1)
    assert ce.adapter is mock_adapter_1

    ce.set_adapter(mock_adapter_2)
    assert ce.adapter is mock_adapter_2


# ── 4. Orchestrator set_adapter ─────────────────────────────────────────────

def test_orchestrator_set_adapter():
    """Orchestrator.set_adapter should swap the underlying adapter."""
    from orchestration.orchestrator import AgenticOrchestrator

    mock_adapter_1 = MagicMock()
    mock_adapter_2 = MagicMock()

    orch = AgenticOrchestrator(adapter=mock_adapter_1)
    assert orch.adapter is mock_adapter_1

    orch.set_adapter(mock_adapter_2)
    assert orch.adapter is mock_adapter_2


# ── 5. Config Centralization ────────────────────────────────────────────────

def test_config_has_anthropic_key():
    """ANTHROPIC_API_KEY must be in centralized config."""
    from config.config import Config
    assert hasattr(Config, "ANTHROPIC_API_KEY")


def test_session_manager_reads_from_config():
    """session_manager should use cfg.SLIDING_WINDOW_SIZE, not a hardcoded value."""
    from orchestration import session_manager
    from config.config import cfg
    assert session_manager.SLIDING_WINDOW_SIZE == cfg.SLIDING_WINDOW_SIZE


@pytest.mark.asyncio
async def test_tool_registry_rejects_malformed_json_payloads():
    from plugins.tool_registry import ToolRegistry, Tool, ToolCall

    registry = ToolRegistry()
    registry.register(Tool(
        name="web_search",
        description="search",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        handler=AsyncMock(return_value="backend temporarily unavailable"),
        response_format="json",
    ))

    result = await registry.execute(ToolCall(
        tool_name="web_search",
        arguments={"query": "test"},
        raw_json='{"tool":"web_search","arguments":{"query":"test"}}',
    ))

    assert result.success is False
    assert "tool_output_validation_error" in result.content


@pytest.mark.asyncio
async def test_web_search_returns_json_error_payload_for_invalid_backend():
    from plugins.tools_web import _web_search

    raw = await _web_search("wigglebudget", backend="not-a-backend")
    payload = json.loads(raw)

    assert payload["type"] == "web_search_error"
    assert "unknown backend" in payload["error"]


@pytest.mark.asyncio
async def test_web_fetch_returns_json_error_payload_for_invalid_url():
    from plugins.tools_web import _web_fetch

    raw = await _web_fetch("file:///etc/passwd")
    payload = json.loads(raw)

    assert payload["type"] == "web_fetch_error"
    assert "only http/https are allowed" in payload["error"]


# ── 6. Strip Tool JSON edge cases ──────────────────────────────────────────

def test_strip_tool_json_no_separator():
    """Tool JSON immediately followed by text (no space) must be fully stripped."""
    from engine.core import AgentCore

    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    # This was the exact failing case from the previous test suite
    raw = '{"tool": "web_search", "arguments": {"query": "test"}}Here is the answer about test results.'
    cleaned = agent._strip_tool_json(raw)
    assert '{"tool"' not in cleaned
    assert cleaned == "Here is the answer about test results."


def test_strip_tool_json_nested_braces():
    """Tool JSON with nested JSON in arguments must be fully stripped."""
    from engine.core import AgentCore

    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    raw = '{"tool": "file_create", "arguments": {"path": "t.json", "content": "data"}}Output follows.'
    cleaned = agent._strip_tool_json(raw)
    assert '{"tool"' not in cleaned
    assert "Output" in cleaned


def test_strip_tool_json_multiple_calls():
    """Multiple tool calls should all be stripped."""
    from engine.core import AgentCore

    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    raw = '{"tool": "web_search", "arguments": {"query": "test"}} Let me search. {"tool": "web_fetch", "arguments": {"url": "https://example.com"}} Here are the results.'
    cleaned = agent._strip_tool_json(raw)
    assert '{"tool"' not in cleaned
    assert "results" in cleaned


# ── 7. Dynamic Adapter Propagation ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_adapter_propagation_to_subsystems():
    """core.py must propagate dynamic adapter to ContextEngine and Orchestrator."""
    from engine.core import AgentCore

    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncMock().__aiter__())

    ctx_eng = MagicMock()
    ctx_eng.compress_exchange = AsyncMock(return_value=("ctx", [], []))
    ctx_eng.build_context_block.return_value = ""
    ctx_eng.set_adapter = MagicMock()

    orch = MagicMock()
    orch.plan = AsyncMock(return_value=[])
    orch.set_adapter = MagicMock()

    session_mgr = MagicMock()
    session = MagicMock()
    session.id = "test-prop"
    session.agent_id = "default"
    session.model = "llama3.2"
    session.system_prompt = "test"
    session.compressed_context = ""
    session.sliding_window = []
    session.first_message = None
    session.message_count = 0
    session.lock = asyncio.Lock()
    session_mgr.get_or_create.return_value = session
    session_mgr.append_message.return_value = True
    session_mgr.update_model = MagicMock()
    session_mgr.update_context = MagicMock()

    tools = MagicMock()
    tools.has_tools.return_value = False
    tools.build_tools_block.return_value = ""

    mock_new_adapter = MagicMock()
    mock_new_adapter.stream = MagicMock(return_value=AsyncMock().__aiter__())

    with patch("engine.core.VectorEngine") as mock_ve, \
         patch("engine.core.create_adapter", return_value=mock_new_adapter), \
         patch("engine.core.SemanticGraph"):

        mock_ve_inst = MagicMock()
        mock_ve_inst.query = AsyncMock(return_value=[])
        mock_ve_inst.index = AsyncMock()
        mock_ve.return_value = mock_ve_inst

        agent = AgentCore(adapter, ctx_eng, session_mgr, tools, orchestrator=orch)

        events = []
        async for ev in agent.chat_stream("hello", model="groq/llama-3.3-70b-versatile"):
            events.append(ev)

        # ContextEngine and Orchestrator should have received the new adapter
        ctx_eng.set_adapter.assert_called_once_with(mock_new_adapter)
        orch.set_adapter.assert_called_once_with(mock_new_adapter)


# ── 8. System Prompt Branding ───────────────────────────────────────────────

def test_system_prompt_branding():
    """System prompts should reference 'Shovs', not 'Antigravity'."""
    from engine.core import DEFAULT_SYSTEM_PROMPT
    from orchestration.agent_profiles import PLATINUM_SYSTEM_PROMPT

    assert "Shovs" in DEFAULT_SYSTEM_PROMPT, "Default prompt must use Shovs branding"
    assert "Antigravity" not in DEFAULT_SYSTEM_PROMPT, "Old branding should be removed"
    assert "Shovs Agent Platform" not in DEFAULT_SYSTEM_PROMPT, "Old platform framing should be removed"
    assert "Shovs" in PLATINUM_SYSTEM_PROMPT, "Profile prompt must use Shovs branding"


# ── 9. Delegation reliability regressions ──────────────────────────────────

def test_default_profile_includes_delegate_tool():
    """Default agent must include delegate_to_agent so planner/tool loop can execute delegation."""
    from orchestration.agent_profiles import ProfileManager
    pm = ProfileManager()
    default = pm.get("default")
    assert default is not None
    assert "delegate_to_agent" in default.tools


@pytest.mark.asyncio
async def test_delegate_tool_accepts_legacy_arguments():
    """
    delegate_to_agent tool should accept legacy payload shape:
    {"agent": "...", "function": "...", "args": [...]}
    and normalize it into a concrete task.
    """
    import plugins.tools as tools_mod

    mock_manager = MagicMock()
    mock_manager.run_agent_task = AsyncMock(return_value="delegated")
    with patch.object(tools_mod, "_agent_manager", mock_manager):
        result = await tools_mod._delegate_to_agent(
            agent="analyst",
            function="write_report",
            args=["aapl_report.md", "Unable to find stock price"],
            _session_id="parent_session",
        )

    assert result == "delegated"
    mock_manager.run_agent_task.assert_awaited_once()
    awaited = mock_manager.run_agent_task.await_args
    # run_agent_task(agent_id, task, parent_id=...)
    assert awaited.args[0] == "analyst"
    assert "aapl_report.md" in awaited.args[1]
    assert awaited.kwargs["parent_id"] == "parent_session"


@pytest.mark.asyncio
async def test_delegation_inherits_parent_adapter_when_model_is_bare():
    """
    If parent session model is bare (no provider prefix), delegation should keep
    the manager's active adapter instead of resolving from model text.
    """
    from orchestration.agent_manager import AgentManager
    from orchestration.agent_profiles import ProfileManager, AgentProfile
    from orchestration.session_manager import SessionManager
    from engine.context_engine import ContextEngine
    from plugins.tool_registry import ToolRegistry, Tool

    parent_adapter = MagicMock()
    sessions = SessionManager()
    profiles = ProfileManager()
    registry = ToolRegistry()

    async def _fake_file_create(path: str, content: str = ""):
        return f"created {path}"

    registry.register(Tool(
        name="file_create",
        description="create file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        handler=_fake_file_create,
    ))

    profiles.create(AgentProfile(
        id="writer_test",
        name="Writer Test",
        model="llama3.2",
        tools=["file_create"],
    ))

    manager = AgentManager(
        profiles=profiles,
        sessions=sessions,
        context_engine=ContextEngine(adapter=parent_adapter),
        adapter=parent_adapter,
        global_registry=registry,
        orchestrator=None,
    )

    parent_sid = "parent_bare_model"
    sessions.get_or_create(
        session_id=parent_sid,
        model="llama-3.3-70b-versatile",  # bare model id (no provider prefix)
        system_prompt="parent",
        agent_id="default",
    )

    captured = {}

    class FakeAgentCore:
        def __init__(self, *args, **kwargs):
            captured["adapter"] = kwargs.get("adapter")
        async def chat_stream(self, **kwargs):
            yield {"type": "token", "content": "delegated-ok"}

    with patch("engine.core.AgentCore", FakeAgentCore), \
         patch("llm.adapter_factory.create_adapter") as mock_create_adapter:
        result = await manager.run_agent_task(
            agent_id="writer_test",
            task="create a file",
            parent_id=parent_sid,
        )

    assert result == "delegated-ok"
    assert captured.get("adapter") is parent_adapter
    mock_create_adapter.assert_not_called()


@pytest.mark.asyncio
async def test_create_session_endpoint_supports_context_mode():
    """Frontend toggle should work before first message by creating a session with mode."""
    from api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        owner_id = "audit-owner"
        create = await ac.post("/sessions", json={
            "agent_id": "default",
            "model": "llama3.2",
            "context_mode": "v2",
            "owner_id": owner_id,
        })
        assert create.status_code == 200
        payload = create.json()
        sid = payload["id"]
        assert payload["context_mode"] == "v2"

        get_s = await ac.get(f"/sessions/{sid}", params={"owner_id": owner_id})
        assert get_s.status_code == 200
        assert get_s.json().get("context_mode") == "v2"

        await ac.delete(f"/sessions/{sid}", params={"owner_id": owner_id})
