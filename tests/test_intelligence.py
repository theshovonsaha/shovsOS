"""
Intelligence Tests
-------------------
Tests that verify the core intelligence loop works correctly.
These test the critical bugs found via live log analysis.
"""

import pytest
import asyncio
import re
from engine.core import AgentCore, _ev
from unittest.mock import AsyncMock, MagicMock, patch


class AsyncIter:
    """Helper: yields items as an async iterator."""
    def __init__(self, items):
        self.items = items
    async def __aiter__(self):
        for item in self.items:
            yield item


# ── Bug 1+4: Tool-call JSON must NOT appear in stored response ─────────────

@pytest.mark.asyncio
async def test_tool_json_stripped_from_stored_response():
    """
    CRITICAL: When the model emits tool-call JSON followed by prose,
    the stored assistant response must NOT contain the raw JSON.
    This was the #1 intelligence bug — corrupt session history.
    """
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    
    # Simulate what the model actually outputs
    raw_response = '{"tool": "web_search", "arguments": {"query": "test"}}Here is the answer about test results.'
    cleaned = agent._strip_tool_json(raw_response)
    
    assert '{"tool"' not in cleaned, "Tool JSON should be stripped"
    assert "Here is the answer" in cleaned, "Prose should remain"
    assert cleaned == "Here is the answer about test results."


@pytest.mark.asyncio
async def test_strip_tool_json_preserves_clean_text():
    """If no tool JSON is present, text should pass through unchanged."""
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    
    clean_text = "The weather in Toronto is -5°C with light snow."
    result = agent._strip_tool_json(clean_text)
    assert result == clean_text


@pytest.mark.asyncio
async def test_strip_tool_json_handles_nested_args():
    """Tool calls with complex nested arguments should be fully stripped."""
    agent = AgentCore(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    
    raw = '{"tool": "bash", "arguments": {"command": "ls -la"}}Output of the command was...'
    cleaned = agent._strip_tool_json(raw)
    assert '{"tool"' not in cleaned
    assert "Output" in cleaned


# ── Bug 2+6: Model override must actually work ──────────────────────────────

@pytest.mark.asyncio
async def test_model_override_updates_session():
    """
    When user switches model mid-session, the session's model 
    should be updated and the new model should be used for streaming.
    """
    # Setup mocks
    adapter = MagicMock()
    adapter.stream = MagicMock(return_value=AsyncIter(["Hello!"]))
    
    session_mgr = MagicMock()
    session = MagicMock()
    session.id = "test-session"
    session.agent_id = "default"
    session.model = "llama3.2"  # Original model
    session.system_prompt = "You are a test."
    session.compressed_context = ""
    session.sliding_window = []
    session.first_message = "hi"
    session.message_count = 2
    session.lock = asyncio.Lock()
    
    session_mgr.get_or_create.return_value = session
    session_mgr.append_message.return_value = False
    session_mgr.update_model = MagicMock()
    session_mgr.update_context = MagicMock()
    
    ctx_eng = MagicMock()
    ctx_eng.compress_exchange = AsyncMock(return_value=("- test context", [], []))
    ctx_eng.build_context_block.return_value = ""
    ctx_eng.set_adapter = MagicMock()  # New: hot-swap support
    
    tools = MagicMock()
    tools.has_tools.return_value = False
    tools.build_tools_block.return_value = ""
    
    # Patch VectorEngine and create_adapter to avoid side effects and ensure mock usage
    with patch("engine.core.VectorEngine") as mock_ve, \
         patch("engine.core.create_adapter") as mock_create_adapter:
        
        mock_create_adapter.return_value = adapter
        mock_ve_instance = MagicMock()
        mock_ve_instance.query = AsyncMock(return_value=[])
        mock_ve.return_value = mock_ve_instance
        
        agent = AgentCore(adapter, ctx_eng, session_mgr, tools, default_model="llama3.2")
        
        # User sends message with a DIFFERENT model
        events = []
        async for ev in agent.chat_stream(
            user_message="test",
            session_id="test-session",
            model="qwen2.5:32b-instruct",  # Switched model!
        ):
            events.append(ev)
        
        # Verify model was updated
        session_mgr.update_model.assert_called_once_with("test-session", "qwen2.5:32b-instruct")
        
        # Verify the LLM was called with the NEW model
        adapter.stream.assert_called()
        call_kwargs = adapter.stream.call_args
        assert call_kwargs.kwargs.get("model") == "qwen2.5:32b-instruct" or \
               (call_kwargs.args and call_kwargs.args[0] == "qwen2.5:32b-instruct") or \
               "qwen2.5:32b-instruct" in str(call_kwargs), \
               f"LLM should be called with new model, got: {call_kwargs}"


# ── Bug 3: Agent re-instantiation (caching) ────────────────────────────────

def test_agent_manager_caches_instances():
    """AgentManager should return the same instance for the same owner/agent/revision tuple."""
    from orchestration.agent_manager import AgentManager
    
    profiles = MagicMock()
    profile = MagicMock()
    profile.model = "llama3.2"
    profile.tools = []
    profile.system_prompt = "test"
    profile.name = "Test"
    profiles.get.return_value = profile
    
    mgr = AgentManager(
        profiles=profiles,
        sessions=MagicMock(),
        context_engine=MagicMock(),
        adapter=MagicMock(),
        global_registry=MagicMock(),
    )
    
    instance1 = mgr.get_agent_instance("default")
    instance2 = mgr.get_agent_instance("default")
    
    assert instance1 is instance2, "Should return cached instance"
    assert profiles.get.call_count >= 1, "Profile lookup should occur"


def test_agent_manager_invalidate_cache():
    """Invalidating cache should force re-creation."""
    from orchestration.agent_manager import AgentManager
    
    profiles = MagicMock()
    profile = MagicMock()
    profile.model = "llama3.2"
    profile.tools = []
    profile.system_prompt = "test"
    profile.name = "Test"
    profiles.get.return_value = profile
    
    mgr = AgentManager(
        profiles=profiles,
        sessions=MagicMock(),
        context_engine=MagicMock(),
        adapter=MagicMock(),
        global_registry=MagicMock(),
    )
    
    instance1 = mgr.get_agent_instance("default")
    mgr.invalidate_cache("default")
    instance2 = mgr.get_agent_instance("default")
    
    assert instance1 is not instance2, "After invalidation, should create new instance"


def test_agent_manager_separates_cache_by_owner_and_revision():
    """Same agent id across owners or revisions must not share one cached core."""
    from orchestration.agent_manager import AgentManager

    profiles = MagicMock()
    owner_a_profile = MagicMock()
    owner_a_profile.model = "llama3.2"
    owner_a_profile.tools = []
    owner_a_profile.system_prompt = "test-a"
    owner_a_profile.name = "OwnerA"
    owner_a_profile.revision = 1

    owner_b_profile = MagicMock()
    owner_b_profile.model = "llama3.2"
    owner_b_profile.tools = []
    owner_b_profile.system_prompt = "test-b"
    owner_b_profile.name = "OwnerB"
    owner_b_profile.revision = 1

    revised_profile = MagicMock()
    revised_profile.model = "llama3.2"
    revised_profile.tools = []
    revised_profile.system_prompt = "test-a-v2"
    revised_profile.name = "OwnerA"
    revised_profile.revision = 2

    state = {"owner-a": owner_a_profile, "owner-b": owner_b_profile}

    def _get_profile(profile_id, owner_id=None):
        if owner_id == "owner-a" and state["owner-a"] is revised_profile:
            return revised_profile
        return state.get(owner_id, owner_a_profile)

    profiles.get.side_effect = _get_profile

    mgr = AgentManager(
        profiles=profiles,
        sessions=MagicMock(),
        context_engine=MagicMock(),
        adapter=MagicMock(),
        global_registry=MagicMock(),
    )

    instance_a1 = mgr.get_agent_instance("default", owner_id="owner-a")
    instance_a2 = mgr.get_agent_instance("default", owner_id="owner-a")
    instance_b = mgr.get_agent_instance("default", owner_id="owner-b")

    assert instance_a1 is instance_a2
    assert instance_a1 is not instance_b

    state["owner-a"] = revised_profile
    instance_a_rev2 = mgr.get_agent_instance("default", owner_id="owner-a")
    assert instance_a_rev2 is not instance_a1


def test_agent_manager_routes_through_runtime_adapter():
    """Profiles can select a runtime adapter without changing the manager API."""
    from orchestration.agent_manager import AgentManager
    from orchestration.agent_runtime import AgentRuntimeCapabilities

    class DummyInstance:
        async def chat_stream(self, *args, **kwargs):
            if False:
                yield {}

    class FakeRuntimeAdapter:
        runtime_kind = "fake"

        def __init__(self):
            self.seen = []

        def get_capabilities(self, context=None):
            return AgentRuntimeCapabilities(
                supports_managed_loop=False,
                supports_checkpoints=True,
                supported_loop_modes=("single",),
                preferred_loop_mode="single",
                notes=("fake runtime",),
            )

        def build_instance(self, context):
            self.seen.append(("build", context.agent_id, context.owner_id, context.profile.runtime_kind))
            return DummyInstance()

        async def run_task(self, context, task, *, parent_id=None, parent_run_id=None):
            self.seen.append(("task", task, parent_id, parent_run_id))
            return "fake-result"

    profiles = MagicMock()
    profile = MagicMock()
    profile.model = "llama3.2"
    profile.tools = []
    profile.system_prompt = "test"
    profile.name = "Fake Runtime"
    profile.revision = 1
    profile.runtime_kind = "fake"
    profiles.get.return_value = profile

    mgr = AgentManager(
        profiles=profiles,
        sessions=MagicMock(),
        context_engine=MagicMock(),
        adapter=MagicMock(),
        global_registry=MagicMock(),
    )
    fake = FakeRuntimeAdapter()
    mgr.register_runtime_adapter(fake)

    instance = mgr.get_agent_instance("default", owner_id="owner-a")

    assert isinstance(instance, DummyInstance)
    assert fake.seen[0] == ("build", "default", "owner-a", "fake")
    assert getattr(instance, "_runtime_kind") == "fake"
    assert getattr(instance, "_runtime_capabilities").preferred_loop_mode == "single"


def test_agent_manager_defaults_to_managed_runtime_when_profile_runtime_missing():
    from orchestration.agent_manager import AgentManager

    profiles = MagicMock()
    profile = MagicMock()
    profile.model = "llama3.2"
    profile.tools = []
    profile.system_prompt = "test"
    profile.name = "Runtime Fallback"
    profile.revision = 1
    profile.runtime_kind = ""
    profiles.get.return_value = profile

    mgr = AgentManager(
        profiles=profiles,
        sessions=MagicMock(),
        context_engine=MagicMock(),
        adapter=MagicMock(),
        global_registry=MagicMock(),
    )

    runtime = mgr.get_runtime_adapter(profile.runtime_kind)

    assert getattr(runtime, "runtime_kind", None) == "managed"


@pytest.mark.asyncio
async def test_agent_manager_runtime_run_task_uses_registered_adapter():
    from orchestration.agent_manager import AgentManager
    from orchestration.agent_runtime import AgentRuntimeCapabilities

    class FakeRuntimeAdapter:
        runtime_kind = "fake"

        def build_instance(self, context):
            raise AssertionError("build_instance should not be used for run_task")

        def get_capabilities(self, context=None):
            return AgentRuntimeCapabilities(
                supports_managed_loop=False,
                supports_checkpoints=True,
                supports_run_artifacts=False,
                supports_run_evals=False,
                supported_loop_modes=("single",),
                preferred_loop_mode="single",
            )

        async def run_task(self, context, task, *, parent_id=None, parent_run_id=None, loop_mode=None, run_metadata=None):
            assert context.agent_id == "default"
            assert context.owner_id == "owner-a"
            assert context.profile.runtime_kind == "fake"
            assert parent_id == "parent-session"
            assert parent_run_id == "run-parent"
            assert loop_mode == "single"
            assert run_metadata["runtime_kind"] == "fake"
            assert run_metadata["runtime_capabilities"]["preferred_loop_mode"] == "single"
            return f"handled:{task}"

    profiles = MagicMock()
    profile = MagicMock()
    profile.model = "llama3.2"
    profile.tools = []
    profile.system_prompt = "test"
    profile.name = "Fake Runtime"
    profile.revision = 1
    profile.runtime_kind = "fake"
    profiles.get.return_value = profile

    mgr = AgentManager(
        profiles=profiles,
        sessions=MagicMock(),
        context_engine=MagicMock(),
        adapter=MagicMock(),
        global_registry=MagicMock(),
    )
    mgr.register_runtime_adapter(FakeRuntimeAdapter())

    result = await mgr.run_agent_task(
        "default",
        "do the thing",
        parent_id="parent-session",
        parent_run_id="run-parent",
        owner_id="owner-a",
    )

    assert result == "handled:do the thing"


@pytest.mark.asyncio
async def test_agent_manager_runtime_override_selects_managed_adapter():
    from orchestration.agent_manager import AgentManager

    profiles = MagicMock()
    profile = MagicMock()
    profile.model = "llama3.2"
    profile.tools = []
    profile.system_prompt = "test"
    profile.name = "Default Runtime"
    profile.revision = 1
    profile.runtime_kind = "native"
    profiles.get.return_value = profile

    mgr = AgentManager(
        profiles=profiles,
        sessions=MagicMock(),
        context_engine=MagicMock(),
        adapter=MagicMock(),
        global_registry=MagicMock(),
    )

    managed_runtime = mgr.get_runtime_adapter("managed")
    managed_runtime.run_task = AsyncMock(return_value="managed-result")

    result = await mgr.run_agent_task(
        "default",
        "delegate task",
        parent_id="parent-session",
        owner_id="owner-a",
        runtime_kind_override="managed",
    )

    assert result == "managed-result"
    managed_runtime.run_task.assert_awaited()


# ── Context Assembly Order ──────────────────────────────────────────────────

def test_context_assembly_order():
    """
    System prompt must be assembled in the correct order:
    1. System prompt (agent personality)
    2. Session Anchor (first message)
    3. Historical Context (RAG)
    4. Session Memory (compressed)
    5. Tools schema
    """
    agent = AgentCore(
        adapter=MagicMock(),
        context_engine=MagicMock(),
        session_manager=MagicMock(),
        tool_registry=MagicMock(),
    )
    
    # Mock context engine
    agent.ctx_eng.build_context_block.return_value = "--- Session Memory ---\n- user likes blue"
    agent.tools.build_tools_block.return_value = "--- Tools ---\n[{\"name\": \"test\"}]"
    
    messages = agent._build_messages(
        system_prompt="You are a helpful assistant.",
        context="- user likes blue",
        sliding_window=[{"role": "user", "content": "previous message"}],
        user_message="current message",
        first_message="hello there",
        message_count=4,
        unified_hits=[{"key": "Color Pref", "anchor": "User: I like blue", "metadata": {"fact": "likes blue"}}],
        ctx_engine=agent.ctx_eng,
    )
    
    # System message should contain all parts in order
    system_content = messages[0]["content"]
    
    assert "You are a helpful assistant" in system_content
    assert "Session Anchor" in system_content
    assert "hello there" in system_content
    assert "Historical Context" in system_content
    assert "Color Pref" in system_content
    assert "Session Memory" in system_content
    
    # Order verification: anchor before historical, historical before memory
    anchor_pos = system_content.index("Session Anchor")
    hist_pos = system_content.index("Historical Context")
    assert anchor_pos < hist_pos, "Session Anchor should come before Historical Context"
    
    # Last message should be the current user message
    assert messages[-1]["content"] == "current message"
    assert messages[-1]["role"] == "user"


# ── Session Model Persistence ───────────────────────────────────────────────

def test_session_update_model():
    """update_model should persist the model change to the session."""
    from orchestration.session_manager import SessionManager
    import tempfile, os
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    try:
        sm = SessionManager(db_path=db_path)
        session = sm.create(model="llama3.2", system_prompt="test")
        
        assert session.model == "llama3.2"
        
        sm.update_model(session.id, "qwen2.5:32b-instruct")
        
        # Reload from DB to verify persistence
        sm2 = SessionManager(db_path=db_path)
        reloaded = sm2.get(session.id)
        
        assert reloaded is not None
        assert reloaded.model == "qwen2.5:32b-instruct"
    finally:
        os.unlink(db_path)
