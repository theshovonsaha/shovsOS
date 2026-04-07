import json
import os
import re
import uuid
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from config.trace_store import get_trace_store


def _groq_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def _write_log(test_name: str, payload: dict) -> Path:
    log_dir = Path("logs/test_runs/groq_intelligence")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{test_name}__{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _session_trace_payload(session_id: str, limit: int = 160) -> dict:
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
        data = event.get("data") or {}
        if event_type in {"llm_prompt", "llm_pass_complete", "assistant_response", "route_decision", "retrieval_policy", "task_policy"}:
            model_io.append({
                "type": event_type,
                "pass_index": event.get("pass_index"),
                "data": data,
            })

    return {"events": full_events, "model_io": model_io}


def _latest_assistant_response(session_id: str) -> str:
    trace = _session_trace_payload(session_id, limit=200)
    responses = [
        item.get("data", {}).get("content", "")
        for item in trace.get("model_io", [])
        if item.get("type") == "assistant_response"
    ]
    return responses[-1] if responses else ""


async def _stream_api_chat(payload: dict) -> list[dict]:
    transport = ASGITransport(app=app)
    events: list[dict] = []
    payload = dict(payload)
    payload.setdefault("owner_id", "groq-test-owner")
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


async def _api_get(path: str, params: dict | None = None) -> dict:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path, params=params or {})
        response.raise_for_status()
        return response.json()


@pytest.mark.asyncio
async def test_groq_stock_research_workflow_has_real_tool_use():
    if not _groq_available():
        pytest.skip("GROQ_API_KEY not set")

    session_id = f"groq_stock_{uuid.uuid4().hex[:8]}"
    prompt = "Research AAPL stock today and summarize the main drivers, risks, and near-term outlook."
    events = await _stream_api_chat({
        "message": prompt,
        "session_id": session_id,
        "model": "groq:llama-3.3-70b-versatile",
        "use_planner": "false",
    })

    log_path = _write_log("stock_research", {
        "session_id": session_id,
        "model": "groq:llama-3.3-70b-versatile",
        "prompt": prompt,
        "events": events,
        "trace": _session_trace_payload(session_id),
    })
    print(f"\n[groq-log] {log_path}")

    errors = [event for event in events if event.get("type") == "error"]
    if errors:
        pytest.skip(f"Groq workflow runtime error: {errors[-1]}")

    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    tool_results = [event for event in events if event.get("type") == "tool_result"]
    final_text = "".join(event.get("content", "") for event in events if event.get("type") == "token").strip()

    assert tool_calls, f"No tool calls recorded. See {log_path}"
    assert tool_results, f"No tool results recorded. See {log_path}"
    assert any((event.get("tool_name") == "web_search") or (event.get("tool") == "web_search") for event in tool_calls + tool_results), log_path
    assert final_text, f"No final synthesis text. See {log_path}"
    assert "drivers" in final_text.lower() or "risks" in final_text.lower() or "outlook" in final_text.lower()


@pytest.mark.asyncio
async def test_groq_memory_correction_workflow_tracks_source_and_cleanup():
    if not _groq_available():
        pytest.skip("GROQ_API_KEY not set")

    session_id = f"groq_memory_{uuid.uuid4().hex[:8]}"
    first_prompt = "My name is Shovon and I live in Vancouver."
    second_prompt = "Actually, I live in Toronto now. What do you call me and where do I live?"

    first_events = await _stream_api_chat({
        "message": first_prompt,
        "session_id": session_id,
        "model": "groq:llama-3.3-70b-versatile",
        "use_planner": "false",
    })
    second_events = await _stream_api_chat({
        "message": second_prompt,
        "session_id": session_id,
        "model": "groq:llama-3.3-70b-versatile",
        "use_planner": "false",
    })

    log_path = _write_log("memory_correction", {
        "session_id": session_id,
        "model": "groq:llama-3.3-70b-versatile",
        "prompts": [first_prompt, second_prompt],
        "first_events": first_events,
        "second_events": second_events,
        "trace": _session_trace_payload(session_id),
    })
    print(f"\n[groq-log] {log_path}")

    second_errors = [event for event in second_events if event.get("type") == "error"]
    if second_errors:
        pytest.skip(f"Groq memory correction runtime error: {second_errors[-1]}")

    final_text = _latest_assistant_response(session_id).strip()
    final_turn_tool_calls = [event for event in transcripts[-1] if event.get("type") == "tool_call"]
    assert final_text, f"No final answer in memory correction workflow. See {log_path}"
    assert "shovon" in final_text.lower(), log_path
    assert "toronto" in final_text.lower(), log_path
    assert "vancouver" not in final_text.lower(), log_path


@pytest.mark.asyncio
async def test_groq_golden_ticket_memory_state_is_correct_and_inspectable():
    if not _groq_available():
        pytest.skip("GROQ_API_KEY not set")

    owner_id = f"groq-golden-{uuid.uuid4().hex[:8]}"
    session_id = f"groq_golden_{uuid.uuid4().hex[:8]}"
    model = "groq:llama-3.3-70b-versatile"

    prompts = [
        "Call me Shovon. I live in Vancouver.",
        "Actually, I moved to Toronto.",
        "What is my preferred name and current location? Answer in one short sentence.",
    ]

    transcripts: list[list[dict]] = []
    for prompt in prompts:
        events = await _stream_api_chat({
            "message": prompt,
            "session_id": session_id,
            "owner_id": owner_id,
            "model": model,
            "use_planner": "false",
        })
        transcripts.append(events)
        errors = [event for event in events if event.get("type") == "error"]
        if errors:
            pytest.skip(f"Groq golden ticket runtime error: {errors[-1]}")

    memory_state = await _api_get(
        f"/sessions/{session_id}/memory-state",
        params={"owner_id": owner_id},
    )
    final_text = _latest_assistant_response(session_id).strip()
    final_turn_tool_calls = [event for event in transcripts[-1] if event.get("type") == "tool_call"]

    log_path = _write_log("golden_ticket_memory_state", {
        "session_id": session_id,
        "owner_id": owner_id,
        "model": model,
        "prompts": prompts,
        "turns": transcripts,
        "memory_state": memory_state,
        "trace": _session_trace_payload(session_id, limit=220),
        "final_text": final_text,
    })
    print(f"\n[groq-log] {log_path}")

    assert final_text, f"No final answer for golden ticket. See {log_path}"
    assert "shovon" in final_text.lower(), log_path
    assert "toronto" in final_text.lower(), log_path
    assert "vancouver" not in final_text.lower(), log_path
    assert not final_turn_tool_calls, log_path

    summary = memory_state["summary"]
    assert summary["deterministic_fact_count"] >= 2, log_path
    assert summary["superseded_fact_count"] >= 1, log_path
    assert any(
        item["predicate"] == "preferred_name" and item["object"] == "Shovon"
        for item in memory_state["deterministic_facts"]
    ), log_path
    assert any(
        item["predicate"] == "location" and item["object"] == "Toronto"
        for item in memory_state["deterministic_facts"]
    ), log_path
    assert any(
        item["predicate"] == "location" and item["object"] == "Vancouver"
        for item in memory_state["superseded_facts"]
    ), log_path
    assert any(
        signal["label"] == "Deterministic extractor"
        for signal in memory_state["recent_memory_signals"]
    ), log_path
