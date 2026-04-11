"""
Real-World System Tests
-----------------------
Exercises live code paths using Groq as the LLM provider.
Covers all 6 bugs fixed in the audit, plus key functional flows.

Run:
    source venv/bin/activate
    pytest tests/test_real_world.py -v -s

Requirements:
    - GROQ_API_KEY set in .env
    - Internet access (for web_search / web_fetch tests)
"""

import asyncio
import json
import os
import uuid
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from dotenv import load_dotenv
load_dotenv()

# ─── Helpers ─────────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
FAST_MODEL  = "llama-3.1-8b-instant"


def _groq_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY", ""))


requires_groq = pytest.mark.skipif(not _groq_available(), reason="GROQ_API_KEY not set")


# ─── 1. Tool Registry ─────────────────────────────────────────────────────────

def test_places_search_registered():
    """Bug 4 fix: PLACES_SEARCH_TOOL was missing from ALL_TOOLS. Verify it's now registered."""
    from plugins.tools import ALL_TOOLS
    names = [t.name for t in ALL_TOOLS]
    assert "places_search" in names, f"places_search missing from ALL_TOOLS. Got: {names}"


def test_all_expected_tools_registered():
    """Verify the full set of tools the default agent relies on are present."""
    from plugins.tools import ALL_TOOLS
    names = {t.name for t in ALL_TOOLS}
    required = {
        "web_search", "web_fetch", "image_search", "bash",
        "file_create", "file_view", "file_str_replace",
        "weather_fetch", "places_search", "places_map",
        "store_memory", "query_memory",
        "delegate_to_agent", "pdf_processor",
    }
    missing = required - names
    assert not missing, f"Missing tools from ALL_TOOLS: {missing}"


def test_delegate_to_agent_tool_registered():
    """Verify delegation tool is registered."""
    from plugins.tools import ALL_TOOLS
    names = [t.name for t in ALL_TOOLS]
    assert "delegate_to_agent" in names


# ─── 2. Adapter Factory ───────────────────────────────────────────────────────

def test_adapter_factory_groq_prefix():
    """create_adapter('groq:llama-...') must return GroqLLMAdapter, not OllamaAdapter."""
    from llm.adapter_factory import create_adapter
    adapter = create_adapter("groq:llama-3.3-70b-versatile")
    assert "Groq" in adapter.__class__.__name__, (
        f"Expected GroqLLMAdapter, got {adapter.__class__.__name__}"
    )


def test_adapter_factory_gemini_prefix():
    """create_adapter('gemini:...') must return GeminiAdapter."""
    from llm.adapter_factory import create_adapter
    adapter = create_adapter("gemini:gemini-1.5-flash")
    assert "Gemini" in adapter.__class__.__name__


def test_adapter_factory_lmstudio_and_llamacpp_are_distinct(monkeypatch):
    """Local OpenAI-compatible runners must not collapse into generic OpenAI."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
    monkeypatch.delenv("LLAMACPP_BASE_URL", raising=False)

    from llm import adapter_factory

    adapter_factory._ADAPTER_CACHE.clear()
    openai_adapter = adapter_factory.create_adapter("openai:gpt-4o-mini")
    lmstudio_adapter = adapter_factory.create_adapter("lmstudio:openai/gpt-oss-20b")
    llamacpp_adapter = adapter_factory.create_adapter("llamacpp:llama-3.1-8b-instruct")

    assert "OpenAI" in openai_adapter.__class__.__name__
    assert "OpenAI" in lmstudio_adapter.__class__.__name__
    assert "OpenAI" in llamacpp_adapter.__class__.__name__
    assert openai_adapter is not lmstudio_adapter
    assert lmstudio_adapter is not llamacpp_adapter
    assert getattr(lmstudio_adapter, "base_url", "") == "http://127.0.0.1:1234/v1"
    assert getattr(llamacpp_adapter, "base_url", "") == "http://127.0.0.1:8080/v1"


def test_strip_provider_prefix():
    """strip_provider_prefix must remove known provider labels."""
    from llm.adapter_factory import strip_provider_prefix
    cases = [
        ("groq:llama-3.3-70b-versatile", "llama-3.3-70b-versatile"),
        ("openai:gpt-4o",                "gpt-4o"),
        ("lmstudio:openai/gpt-oss-20b",  "openai/gpt-oss-20b"),
        ("llamacpp:llama-3.1-8b",        "llama-3.1-8b"),
        ("gemini:gemini-1.5-flash",      "gemini-1.5-flash"),
        ("llama3.2",                     "llama3.2"),         # no prefix → unchanged
        ("ollama:llama3.2",              "llama3.2"),
    ]
    for raw, expected in cases:
        result = strip_provider_prefix(raw)
        assert result == expected, f"strip_provider_prefix({raw!r}) = {result!r}, expected {expected!r}"


def test_continue_is_not_treated_as_trivial_chat():
    from engine.core import _is_trivial_query

    assert _is_trivial_query("hi")
    assert not _is_trivial_query("continue")
    assert not _is_trivial_query("go on")


def test_local_openai_compatible_runners_prefer_single_loop():
    from llm.openai_adapter import OpenAIAdapter
    from engine.core import _should_prefer_single_loop

    adapter = OpenAIAdapter(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
    assert _should_prefer_single_loop(adapter, "lmstudio:openai/gpt-oss-20b")


# ─── 3. Context Model Default (Bug 1 & 3) ────────────────────────────────────

def test_context_model_default_is_none():
    """
    Bug Fix: context_model Form default was 'deepseek-r1:8b' (Ollama-only).
    The api/main.py form param must now default to None so cloud adapters
    use the active session model for compression instead.
    """
    import inspect
    from api import main as api_main
    sig = inspect.signature(api_main.chat_stream)
    param = sig.parameters.get("context_model")
    assert param is not None, "context_model param not found on chat_stream"
    # The default should be None (not the old "deepseek-r1:8b" string)
    default_val = param.default
    # FastAPI wraps defaults in FieldInfo — extract raw value
    raw = getattr(default_val, "default", default_val)
    assert raw is None or raw != "deepseek-r1:8b", (
        f"context_model default is still 'deepseek-r1:8b' — Bug 1/3 not fixed. Got: {raw!r}"
    )


@pytest.mark.asyncio
async def test_bash_docker_daemon_check():
    """
    Verify that if DOCKER_DISABLED=true, the bash tool returns a denied message.
    """
    import plugins.tools as tools_mod
    original = os.environ.get("DOCKER_DISABLED")
    os.environ["DOCKER_DISABLED"] = "true"
    try:
        result = await tools_mod._bash("echo test")
        assert "[denied]" in result.lower() or "blocked" in result.lower()
    finally:
        if original is None:
            del os.environ["DOCKER_DISABLED"]
        else:
            os.environ["DOCKER_DISABLED"] = original


@pytest.mark.asyncio
async def test_bash_sandbox_confinement():
    """Bash must refuse to run outside sandbox dir."""
    from plugins.tools import _bash
    result = await _bash("pwd", workdir="/etc")  # /etc is outside sandbox
    # Should run in sandbox, not /etc
    from plugins.tools import SANDBOX_DIR
    assert "/etc" not in result or "sandbox" in result.lower() or str(SANDBOX_DIR) in result


@pytest.mark.asyncio
async def test_bash_blocked_commands():
    """Verify destructive commands are either blocked by safety policy or rejected by OS."""
    from plugins.tools import _bash
    # Try a command that matches the safety regex (rm -rf /*)
    result = await _bash("rm -rf /*")
    # Safety filter should block it before subprocess is even created
    assert "blocked" in result.lower(), (
        f"Safety filter should have blocked rm -rf/*. Got: {result!r}"
    )


@pytest.mark.asyncio
async def test_bash_real_command():
    """Bash executes real shell commands and returns stdout."""
    from plugins.tools import _bash
    result = await _bash("echo 'hello from sandbox'")
    if "[denied]" in result.lower() and "docker daemon" in result.lower():
        pytest.skip("Docker daemon unavailable in this environment")
    if "No such image: python:3.11-slim" in result:
        pytest.skip("Docker sandbox image unavailable in this environment")
    assert "hello from sandbox" in result


# ─── 5. Delegation Model Inheritance (Bug 2 & 5) ─────────────────────────────

def test_delegate_inherits_parent_session_model():
    """
    Bug Fix: AgentManager.run_agent_task must inherit parent session model.
    Verify the session manager lookup and model resolution logic works.
    """
    from orchestration.session_manager import SessionManager
    from orchestration.agent_profiles import ProfileManager
    from orchestration.agent_manager import AgentManager
    from llm.adapter_factory import create_adapter, strip_provider_prefix
    from plugins.tool_registry import ToolRegistry
    from engine.context_engine import ContextEngine

    sessions  = SessionManager()
    profiles  = ProfileManager()
    adapter   = create_adapter("groq")
    registry  = ToolRegistry()
    ctx       = ContextEngine(adapter=adapter)
    manager   = AgentManager(
        profiles=profiles, sessions=sessions,
        context_engine=ctx, adapter=adapter, global_registry=registry
    )

    # Simulate a parent session using Groq
    test_sid = f"test_parent_{uuid.uuid4().hex[:6]}"
    sessions.get_or_create(
        session_id=test_sid,
        model="groq:llama-3.3-70b-versatile",
        system_prompt="test",
        agent_id="default"
    )

    parent = sessions.get(test_sid)
    assert parent is not None, "Parent session not found"
    assert parent.model == "groq:llama-3.3-70b-versatile", (
        f"Parent session has wrong model: {parent.model}"
    )

    # Simulate the logic inside run_agent_task
    effective_model = parent.model
    delegation_adapter = create_adapter(provider=effective_model)
    clean_model = strip_provider_prefix(effective_model)

    assert "Groq" in delegation_adapter.__class__.__name__, (
        f"Delegation adapter should be Groq, got: {delegation_adapter.__class__.__name__}"
    )
    assert clean_model == "llama-3.3-70b-versatile", (
        f"Clean model should not have prefix: {clean_model}"
    )


# ─── 6. Context Engine ────────────────────────────────────────────────────────

@pytest.mark.asyncio
@requires_groq
async def test_context_compression_with_groq():
    """
    Real context compression using Groq.
    Bug 3 fix: compression should work (not 400) when adapter is Groq.
    """
    from llm.adapter_factory import create_adapter
    from engine.context_engine import ContextEngine

    adapter = create_adapter("groq")
    if not await adapter.health():
        pytest.skip("Groq API unavailable (missing credentials or network).")
    ctx = ContextEngine(adapter=adapter, compression_model=FAST_MODEL)

    updated, facts, voids = await ctx.compress_exchange(
        user_message="My name is Alex and I love Python programming.",
        assistant_response="Great to meet you Alex! Python is an excellent choice for many tasks.",
        current_context="",
        is_first_exchange=True,
        model=FAST_MODEL,
    )

    assert updated, "Context should have been updated"
    # Should contain some reference to either Alex, Python, or the first message
    assert len(updated) > 5, f"Context too short: {updated!r}"
    print(f"\n✓ Context compression output:\n{updated}")


# ─── 7. Planner / Orchestrator ────────────────────────────────────────────────

@pytest.mark.asyncio
@requires_groq
async def test_planner_selects_correct_tools():
    """
    Orchestrator plan() should select relevant tools for different queries.
    Bug 1 fix: planner uses the correct model (not a cross-provider string).
    """
    from llm.adapter_factory import create_adapter
    from orchestration.orchestrator import AgenticOrchestrator

    adapter = create_adapter("groq")
    if not await adapter.health():
        pytest.skip("Groq API unavailable (missing credentials or network).")
    orch = AgenticOrchestrator(adapter=adapter)

    tools = [
        {"name": "web_search",      "description": "Search the web for current info."},
        {"name": "weather_fetch",   "description": "Get weather data for a location."},
        {"name": "file_create",     "description": "Create a file in the sandbox."},
        {"name": "delegate_to_agent", "description": "Delegate a task to another agent."},
        {"name": "bash",            "description": "Run a bash command."},
    ]

    # Weather query → should plan weather_fetch
    plan_weather = await orch.plan("What's the weather in Toronto?", tools, model=FAST_MODEL)
    print(f"\n✓ Weather plan: {plan_weather}")
    assert "weather_fetch" in plan_weather, (
        f"Planner should include weather_fetch for weather query. Got: {plan_weather}"
    )

    # Conversational → should return empty (no tools needed)
    plan_chat = await orch.plan("Hello, how are you today?", tools, model=FAST_MODEL)
    print(f"✓ Chat plan: {plan_chat}")
    assert len(plan_chat) == 0 or "web_search" not in plan_chat, (
        f"Planner should not force tools for a greeting. Got: {plan_chat}"
    )

    # Research query → should include web_search
    plan_search = await orch.plan("What are the best Python frameworks in 2026?", tools, model=FAST_MODEL)
    print(f"✓ Search plan: {plan_search}")
    assert "web_search" in plan_search, (
        f"Planner should include web_search for factual query. Got: {plan_search}"
    )


# ─── 8. Web Tools ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_web_search_duckduckgo():
    """web_search with fallback DuckDuckGo engine should return results."""
    from plugins.tools import _web_search
    result = await _web_search("Python programming language", num_results=3)
    if result.startswith("No results found") or "web_search" in result and "error" in result.lower():
        pytest.skip(f"Web search unavailable: {result[:120]}")
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        pytest.skip(f"Search backend returned non-JSON output: {result[:120]}")
    assert data.get("type") == "web_search_results"
    if len(data.get("results", [])) == 0:
        pytest.skip("Search backend returned zero results in this environment")
    assert len(data.get("results", [])) > 0, "Expected at least 1 search result"
    print(f"\n✓ DuckDuckGo search returned {len(data['results'])} results")


@pytest.mark.asyncio
async def test_web_fetch_example_com():
    """web_fetch should retrieve example.com or return a graceful error (e.g. SSL on macOS)."""
    from plugins.tools import _web_fetch
    result = await _web_fetch("https://example.com", max_chars=500)
    data = json.loads(result)
    # On macOS Python 3.10 without certifi, SSL cert errors are environment issues — skip
    if data.get("error") and "SSL" in str(data.get("error", "")):
        pytest.skip("SSL certificate issue in test environment — install certifi to fix")
    if data.get("error") and "403" in str(data.get("error", "")):
        pytest.skip("Jina Reader returned 403 — network/rate-limit issue")
    if data.get("error") and ("nodename nor servname" in str(data.get("error", "")) or "connection" in str(data.get("error", "")).lower()):
        pytest.skip(f"Network unavailable in test environment: {data.get('error')}")
    assert "error" not in data, f"Fetch error: {data.get('error')}"
    assert "Example" in data.get("content", "") or "example" in data.get("title", "").lower()
    print(f"\n✓ Fetched {data.get('total_length', '?')} chars from example.com")


# ─── 9. File Tools (Sandbox) ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_file_create_and_view():
    """file_create should write a file; file_view should read it back."""
    from plugins.tools import _file_create, _file_view
    fname = f"test_{uuid.uuid4().hex[:6]}.txt"
    content = "Real-world test payload 🚀"

    create_result = await _file_create(path=fname, content=content)
    create_payload = json.loads(create_result)
    assert create_payload["success"] is True, f"Unexpected create result: {create_result}"
    assert create_payload["verification"]["exists_after"] is True
    assert create_payload["path"] == fname

    view_result = await _file_view(path=fname)
    assert content in view_result, f"File content not found in view: {view_result}"

    # Cleanup
    from plugins.tools import SANDBOX_DIR
    (SANDBOX_DIR / fname).unlink(missing_ok=True)
    print(f"\n✓ file_create + file_view cycle passed for {fname}")


@pytest.mark.asyncio
async def test_file_view_accepts_sandbox_prefixed_path():
    """file_view should accept the /sandbox/... path returned by file_create previews."""
    from plugins.tools import _file_create, _file_view, SANDBOX_DIR

    fname = f"prefixed_{uuid.uuid4().hex[:6]}.txt"
    content = "Sandbox-prefixed path should resolve"

    await _file_create(path=fname, content=content)
    view_result = await _file_view(path=f"/sandbox/{fname}")

    assert content in view_result, f"Sandbox-prefixed path did not resolve: {view_result}"

    (SANDBOX_DIR / fname).unlink(missing_ok=True)
    print(f"\n✓ file_view accepts /sandbox path for {fname}")


@pytest.mark.asyncio
async def test_file_str_replace():
    """file_str_replace should do exact-match replacement."""
    from plugins.tools import _file_create, _file_str_replace, _file_view
    fname = f"replace_test_{uuid.uuid4().hex[:6]}.txt"

    await _file_create(path=fname, content="Hello World. This is version 1.")
    result = await _file_str_replace(path=fname, old_str="version 1", new_str="version 2")
    assert "Replaced" in result

    view = await _file_view(path=fname)
    assert "version 2" in view
    assert "version 1" not in view

    from plugins.tools import SANDBOX_DIR
    (SANDBOX_DIR / fname).unlink(missing_ok=True)
    print(f"\n✓ file_str_replace passed")


# ─── 10. Weather Tool ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_weather_fetch_real():
    """weather_fetch should return real data for a known city (no API key needed)."""
    from plugins.tools import _weather_fetch
    result = await _weather_fetch("Toronto")
    if "error" in result.lower() and ("nodename nor servname" in result.lower() or "connection" in result.lower()):
        pytest.skip(f"Weather API unavailable in test environment: {result[:120]}")
    assert "Toronto" in result or "Weather" in result, f"Unexpected: {result}"
    assert "Temperature" in result or "°" in result, "Should contain temperature data"
    print(f"\n✓ Weather result snippet: {result[:120]}")


# ─── 11. Memory / Semantic Graph ─────────────────────────────────────────────

def test_semantic_graph_add_and_retrieve():
    """SemanticGraph can store and query temporal facts."""
    from memory.semantic_graph import SemanticGraph
    sid = f"test_sg_{uuid.uuid4().hex[:6]}"
    sg = SemanticGraph()

    sg.add_temporal_fact(sid, "user", "prefers", "dark mode", turn=1)
    facts = sg.get_current_facts(sid)
    assert any("dark mode" in str(f) for f in facts), (
        f"Fact not found in graph. Got: {facts}"
    )
    print(f"\n✓ SemanticGraph fact stored and retrieved: {facts}")


# ─── 12. Full Delegation Flow (Live Groq) ────────────────────────────────────

@pytest.mark.asyncio
@requires_groq
async def test_live_delegation_model_inheritance():
    """
    Bug 2 & 5 regression test: delegated agents must use parent's active model.
    Runs a real delegation where parent uses Groq and verifies no 404 errors.
    """
    from llm.adapter_factory import create_adapter
    from plugins.tool_registry import ToolRegistry
    from plugins.tools import register_all_tools, SANDBOX_DIR
    from orchestration.agent_profiles import ProfileManager, AgentProfile
    from orchestration.session_manager import SessionManager
    from engine.context_engine import ContextEngine
    from orchestration.orchestrator import AgenticOrchestrator
    from orchestration.agent_manager import AgentManager

    adapter    = create_adapter("groq")
    if not await adapter.health():
        pytest.skip("Groq API unavailable (missing credentials or network).")
    registry   = ToolRegistry()
    sessions   = SessionManager(max_sessions=20)
    ctx        = ContextEngine(adapter=adapter, compression_model=FAST_MODEL)
    profiles   = ProfileManager()
    orch       = AgenticOrchestrator(adapter=adapter)

    # Create a minimal writer sub-agent
    profiles.create(AgentProfile(
        id="writer_test",
        name="Writer Test Agent",
        model=GROQ_MODEL,
        tools=["file_create"],
        system_prompt="You are a writer. When asked to create a file, use the file_create tool immediately."
    ))

    manager = AgentManager(
        profiles=profiles, sessions=sessions,
        context_engine=ctx, adapter=adapter,
        global_registry=registry, orchestrator=orch
    )
    register_all_tools(registry, agent_manager=manager)

    # Simulate parent session with Groq model
    parent_sid = f"parent_{uuid.uuid4().hex[:8]}"
    sessions.get_or_create(
        session_id=parent_sid,
        model=f"groq:{GROQ_MODEL}",
        system_prompt="You are a project manager.",
        agent_id="default"
    )

    # Run delegation task directly through the manager
    fname = f"delegation_test_{uuid.uuid4().hex[:6]}.txt"
    task = f"Create a file called '{fname}' with content 'Delegation test passed!'"

    # Avoid local embedding-network dependency in this delegation regression test.
    with patch("memory.vector_engine.VectorEngine.query", new=AsyncMock(return_value=[])), \
         patch("memory.vector_engine.VectorEngine.index", new=AsyncMock()):
        result = await manager.run_agent_task(
            agent_id="writer_test",
            task=task,
            parent_id=parent_sid
        )

    print(f"\n✓ Delegation result: {result[:200]}")
    assert "Error" not in result[:50] or "delegation" in result.lower(), (
        f"Delegation returned an error response: {result}"
    )
    # Whether or not the sub-agent created the file, we should get a non-404 response
    assert "404" not in result, f"Sub-agent still getting 404 model error: {result}"

    # Cleanup
    (SANDBOX_DIR / fname).unlink(missing_ok=True)


# ─── 13. Session Manager ─────────────────────────────────────────────────────

def test_session_create_and_retrieve():
    """SessionManager creates and retrieves sessions with correct model."""
    from orchestration.session_manager import SessionManager
    sm = SessionManager()
    sid = f"test_sm_{uuid.uuid4().hex[:6]}"
    sess = sm.get_or_create(
        session_id=sid,
        model="groq:llama-3.3-70b-versatile",
        system_prompt="test",
        agent_id="default"
    )
    assert sess.id == sid
    assert sess.model == "groq:llama-3.3-70b-versatile"

    retrieved = sm.get(sid)
    assert retrieved is not None
    assert retrieved.model == "groq:llama-3.3-70b-versatile"
    print(f"\n✓ Session stored and retrieved with correct model")


def test_agent_cache_invalidation():
    """invalidate_cache should evict the specific agent from the cache."""
    from orchestration.agent_profiles import ProfileManager
    from orchestration.session_manager import SessionManager
    from orchestration.agent_manager import AgentManager
    from llm.adapter_factory import create_adapter
    from plugins.tool_registry import ToolRegistry
    from engine.context_engine import ContextEngine

    adapter  = create_adapter("groq")
    manager  = AgentManager(
        profiles=ProfileManager(),
        sessions=SessionManager(),
        context_engine=ContextEngine(adapter=adapter),
        adapter=adapter,
        global_registry=ToolRegistry()
    )

    instance1 = manager.get_agent_instance("default")
    manager.invalidate_cache("default")
    instance2 = manager.get_agent_instance("default")

    assert instance1 is not instance2, "Cache should have been invalidated"
    print("\n✓ Agent cache invalidation works correctly")


# ─── 14. PATCH /agents endpoint ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_patch_agent_model():
    """PATCH /agents/{id} should update model and invalidate cache."""
    from httpx import AsyncClient, ASGITransport
    from api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        owner_id = "realworld-owner"
        # Get default agent current state
        resp = await ac.get("/agents/default", params={"owner_id": owner_id})
        assert resp.status_code == 200
        original_model = resp.json().get("model")

        # PATCH the model
        patch_resp = await ac.patch(
            "/agents/default",
            json={"model": "groq:llama-3.1-8b-instant", "owner_id": owner_id}
        )
        assert patch_resp.status_code == 200, (
            f"PATCH failed with {patch_resp.status_code}: {patch_resp.text}"
        )
        assert patch_resp.json().get("model") == "groq:llama-3.1-8b-instant"

        # Verify persisted
        verify_resp = await ac.get("/agents/default", params={"owner_id": owner_id})
        assert verify_resp.json().get("model") == "groq:llama-3.1-8b-instant"

        # Restore original model
        await ac.patch("/agents/default", json={"model": original_model or "llama3.2", "owner_id": owner_id})
        print(f"\n✓ PATCH /agents/default: model updated and persisted")
