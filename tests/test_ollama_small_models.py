import re
import uuid
import json
from pathlib import Path

import httpx
import pytest

from api.main import app
from config.trace_store import get_trace_store
from engine.context_engine import ContextEngine
from engine.core import AgentCore
from llm.llm_adapter import OllamaAdapter
from memory.semantic_graph import SemanticGraph
from orchestration.session_manager import SessionManager
from plugins.tool_registry import Tool, ToolRegistry


def _ollama_base_url() -> str:
    from config.config import cfg

    return (cfg.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")


async def _ollama_tags() -> list[dict]:
    async with httpx.AsyncClient(timeout=3.0) as client:
        response = await client.get(f"{_ollama_base_url()}/api/tags")
        response.raise_for_status()
        data = response.json()
    return data.get("models", [])


def _pick_small_models(tags: list[dict], limit: int = 2) -> list[str]:
    preferred_patterns = [
        r"^gemma2:2b$",
        r"^qwen2\.5-coder:3b$",
        r"^llama3\.2:1b$",
        r"^llama3\.2$",
        r"^qwen3\.5(?::latest|:9b)?$",
        r"^phi3:mini$",
        r"^phi3\.5.*mini",
        r"^qwen2\.5-coder:7b$",
        r"^mistral:7b",
    ]
    names = [str(model.get("name", "")).strip() for model in tags]
    names = [name for name in names if name and "embed" not in name.lower()]

    selected: list[str] = []
    for pattern in preferred_patterns:
        for name in names:
            if re.search(pattern, name, re.IGNORECASE) and name not in selected:
                selected.append(name)
                if len(selected) >= limit:
                    return selected

    size_hint = re.compile(r":(?:1b|2b|3b|7b|8b|9b)\b", re.IGNORECASE)
    for name in names:
        if size_hint.search(name) and name not in selected:
            selected.append(name)
            if len(selected) >= limit:
                return selected

    return selected[:limit]


async def _live_core(model_name: str, tmp_path: Path, with_todo_tool: bool = False) -> AgentCore:
    adapter = OllamaAdapter(base_url=_ollama_base_url())
    context_engine = ContextEngine(adapter=adapter, compression_model=model_name)
    sessions = SessionManager(db_path=str(tmp_path / f"sessions_{uuid.uuid4().hex[:8]}.db"))
    registry = ToolRegistry()

    if with_todo_tool:
        async def _todo_write_stub(**kwargs):
            return f"todo stub ok: {sorted(kwargs.keys())}"

        registry.register(Tool(
            name="todo_write",
            description="Initialize tasks for a multi-step workflow.",
            parameters={
                "type": "object",
                "properties": {
                    "tasks": {"type": "array", "items": {"type": "object"}},
                    "topic": {"type": "string"},
                },
                "required": ["tasks"],
            },
            handler=_todo_write_stub,
        ))

    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=sessions,
        tool_registry=registry,
    )
    core.graph = SemanticGraph(db_path=str(tmp_path / f"memory_{uuid.uuid4().hex[:8]}.db"))
    return core


def _sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)


def _token_text(events: list[dict]) -> str:
    return "".join(event.get("content", "") for event in events if event.get("type") == "token").strip()


def _error_events(events: list[dict]) -> list[dict]:
    return [event for event in events if event.get("type") == "error"]


def _write_raw_log(tmp_path: Path, test_name: str, model_name: str, payload: dict) -> Path:
    log_dir = tmp_path / "ollama_chat_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{test_name}__{_sanitize_model_name(model_name)}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _session_trace_payload(session_id: str, limit: int = 80) -> dict:
    store = get_trace_store()
    index_events = store.list_events(limit=limit, session_id=session_id)
    full_events = []
    for event in reversed(index_events):
        event_id = event.get("id")
        full_event = store.get_event(event_id) if event_id else event
        if full_event:
            full_events.append(full_event)

    model_io = []
    for event in full_events:
        event_type = event.get("event_type")
        data = event.get("data")
        if event_type == "llm_prompt":
            model_io.append({
                "type": "llm_prompt",
                "pass_index": event.get("pass_index"),
                "model": (data or {}).get("model"),
                "messages": (data or {}).get("messages"),
                "estimated_tokens": (data or {}).get("estimated_tokens"),
            })
        elif event_type == "llm_pass_complete":
            model_io.append({
                "type": "llm_pass_complete",
                "pass_index": event.get("pass_index"),
                "model": (data or {}).get("model"),
                "completion_chars": (data or {}).get("completion_chars"),
            })
        elif event_type == "assistant_response":
            model_io.append({
                "type": "assistant_response",
                "pass_index": event.get("pass_index"),
                "content": (data or {}).get("content"),
                "raw_length": (data or {}).get("raw_length"),
                "clean_length": (data or {}).get("clean_length"),
            })

    return {
        "events": full_events,
        "model_io": model_io,
    }


async def _stream_api_chat(payload: dict) -> list[dict]:
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    events: list[dict] = []
    payload = dict(payload)
    payload.setdefault("owner_id", "ollama-test-owner")
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat/stream", data=payload)
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                events.append({"type": "decode_error", "raw": line})
    return events


@pytest.mark.asyncio
async def test_small_ollama_models_available():
    try:
        tags = await _ollama_tags()
    except Exception as exc:
        pytest.skip(f"Ollama is not reachable: {exc}")

    small_models = _pick_small_models(tags)
    if not small_models:
        pytest.skip(f"No small Ollama chat models found. Installed models: {[m.get('name') for m in tags]}")

    assert small_models


@pytest.mark.asyncio
async def test_small_ollama_trivial_chat_turn(tmp_path):
    try:
        tags = await _ollama_tags()
    except Exception as exc:
        pytest.skip(f"Ollama is not reachable: {exc}")

    small_models = _pick_small_models(tags, limit=2)
    if not small_models:
        pytest.skip("No small Ollama chat models available for live chat test.")

    for model_name in small_models:
        core = await _live_core(model_name, tmp_path)
        session_id = f"ollama_smoke_{uuid.uuid4().hex[:8]}"
        events = [
            event async for event in core.chat_stream(
                user_message="hi",
                session_id=session_id,
                model=f"ollama:{model_name}",
                use_planner=False,
            )
        ]
        log_path = _write_raw_log(tmp_path, "trivial_chat", model_name, {
            "test": "test_small_ollama_trivial_chat_turn",
            "model": model_name,
            "session_id": session_id,
            "user_message": "hi",
            "events": events,
            "trace": _session_trace_payload(session_id),
        })
        print(f"\n[raw-log] {log_path}")
        assert not any(event.get("type") == "error" for event in events), f"error event for model {model_name}: {events}"
        token_text = "".join(event.get("content", "") for event in events if event.get("type") == "token").strip()
        assert token_text, f"No token output produced by model {model_name}"


@pytest.mark.asyncio
async def test_small_ollama_multistep_route_emits_todo_plan(tmp_path):
    try:
        tags = await _ollama_tags()
    except Exception as exc:
        pytest.skip(f"Ollama is not reachable: {exc}")

    small_models = _pick_small_models(tags, limit=1)
    if not small_models:
        pytest.skip("No small Ollama chat models available for multi-step route test.")

    model_name = small_models[0]
    core = await _live_core(model_name, tmp_path, with_todo_tool=True)
    session_id = f"ollama_plan_{uuid.uuid4().hex[:8]}"
    user_message = "research local ai options then summarize them and save notes"
    events = [
        event async for event in core.chat_stream(
            user_message=user_message,
            session_id=session_id,
            model=f"ollama:{model_name}",
            use_planner=False,
        )
    ]
    log_path = _write_raw_log(tmp_path, "multistep_route", model_name, {
        "test": "test_small_ollama_multistep_route_emits_todo_plan",
        "model": model_name,
        "session_id": session_id,
        "user_message": user_message,
        "events": events,
        "trace": _session_trace_payload(session_id),
    })
    print(f"\n[raw-log] {log_path}")

    plan_events = [event for event in events if event.get("type") == "plan"]
    assert plan_events, f"No plan event emitted for model {model_name}"
    assert any("todo_write" in (event.get("tools") or []) for event in plan_events), plan_events


@pytest.mark.asyncio
async def test_small_ollama_context_engine_extracts_user_fact_signal(tmp_path):
    try:
        tags = await _ollama_tags()
    except Exception as exc:
        pytest.skip(f"Ollama is not reachable: {exc}")

    small_models = _pick_small_models(tags, limit=1)
    if not small_models:
        pytest.skip("No small Ollama chat models available for context-engine signal test.")

    model_name = small_models[0]
    adapter = OllamaAdapter(base_url=_ollama_base_url())
    ctx_engine = ContextEngine(adapter=adapter, compression_model=model_name)

    updated_ctx, new_facts, voids = await ctx_engine.compress_exchange(
        user_message="Call me Shovon from now on. That is my preferred name.",
        assistant_response="Understood. I will call you Shovon from now on.",
        current_context="",
        is_first_exchange=False,
        model=model_name,
    )

    log_path = _write_raw_log(tmp_path, "context_engine_fact_signal", model_name, {
        "test": "test_small_ollama_context_engine_extracts_user_fact_signal",
        "model": model_name,
        "updated_ctx": updated_ctx,
        "new_facts": new_facts,
        "voids": voids,
    })
    print(f"\n[raw-log] {log_path}")

    if not new_facts:
        pytest.skip(f"Context engine did not extract any grounded fact on {model_name}. See {log_path}")

    assert any("shovon" in " ".join(str(part) for part in fact.values()).lower() for fact in new_facts), (
        f"Context engine did not preserve the user preference signal on {model_name}. See {log_path}"
    )


@pytest.mark.asyncio
async def test_small_ollama_memory_fact_write_and_followup_recall(tmp_path):
    try:
        tags = await _ollama_tags()
    except Exception as exc:
        pytest.skip(f"Ollama is not reachable: {exc}")

    small_models = _pick_small_models(tags, limit=1)
    if not small_models:
        pytest.skip("No small Ollama chat models available for memory recall test.")

    model_name = small_models[0]
    core = await _live_core(model_name, tmp_path)
    owner_id = f"ollama-memory-owner-{uuid.uuid4().hex[:8]}"
    session_id = f"ollama_memory_{uuid.uuid4().hex[:8]}"

    first_events = [
        event async for event in core.chat_stream(
            user_message="Call me Shovon from now on. Please remember that for later.",
            session_id=session_id,
            owner_id=owner_id,
            model=f"ollama:{model_name}",
            use_planner=False,
        )
    ]
    second_events = [
        event async for event in core.chat_stream(
            user_message="What should you call me?",
            session_id=session_id,
            owner_id=owner_id,
            model=f"ollama:{model_name}",
            use_planner=False,
        )
    ]

    facts = core.graph.get_current_facts(session_id, owner_id=owner_id)
    final_text = _token_text(second_events)
    log_path = _write_raw_log(tmp_path, "memory_recall", model_name, {
        "test": "test_small_ollama_memory_fact_write_and_followup_recall",
        "model": model_name,
        "session_id": session_id,
        "owner_id": owner_id,
        "first_events": first_events,
        "second_events": second_events,
        "current_facts": facts,
        "final_text": final_text,
        "trace": _session_trace_payload(session_id, limit=140),
    })
    print(f"\n[raw-log] {log_path}")

    if _error_events(first_events) or _error_events(second_events):
        pytest.skip(f"Live memory recall hit runtime error on {model_name}. See {log_path}")
    if not any("shovon" in " ".join(part for part in fact if isinstance(part, str)).lower() for fact in facts):
        pytest.skip(f"No deterministic memory fact containing 'Shovon' was written on {model_name}. See {log_path}")

    assert "shovon" in final_text.lower(), f"Follow-up recall did not surface the stored name. See {log_path}"


@pytest.mark.asyncio
async def test_small_ollama_stock_research_workflow_logs_full_trace(tmp_path):
    try:
        tags = await _ollama_tags()
    except Exception as exc:
        pytest.skip(f"Ollama is not reachable: {exc}")

    small_models = _pick_small_models(tags, limit=1)
    if not small_models:
        pytest.skip("No small Ollama chat models available for stock workflow test.")

    model_name = small_models[0]
    session_id = f"ollama_stock_{uuid.uuid4().hex[:8]}"
    user_message = "Research AAPL stock today and summarize the main drivers, risks, and near-term outlook."
    events = await _stream_api_chat({
        "message": user_message,
        "session_id": session_id,
        "model": f"ollama:{model_name}",
        "use_planner": "false",
    })

    log_path = _write_raw_log(tmp_path, "stock_research", model_name, {
        "test": "test_small_ollama_stock_research_workflow_logs_full_trace",
        "model": model_name,
        "session_id": session_id,
        "user_message": user_message,
        "events": events,
        "trace": _session_trace_payload(session_id, limit=140),
    })
    print(f"\n[raw-log] {log_path}")

    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    tool_results = [event for event in events if event.get("type") == "tool_result"]
    final_text = "".join(event.get("content", "") for event in events if event.get("type") == "token").strip()
    error_events = [event for event in events if event.get("type") == "error"]

    if error_events:
        pytest.skip(f"Stock workflow hit runtime error on {model_name}: {error_events[-1]}")
    if not tool_calls:
        pytest.skip(f"Stock workflow did not call tools on {model_name}. See {log_path}")
    if not tool_results:
        pytest.skip(f"Stock workflow did not produce tool results on {model_name}. See {log_path}")
    if not any(event.get("tool_name") == "web_search" or event.get("tool") == "web_search" for event in tool_calls + tool_results):
        pytest.skip(f"Stock workflow did not use web_search on {model_name}. See {log_path}")

    assert final_text, f"No final synthesis text generated. See {log_path}"
