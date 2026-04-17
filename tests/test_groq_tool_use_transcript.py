import json
import os
import uuid
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from config.trace_store import get_trace_store


TRANSCRIPT = [
    "Hi, I'm Shovon. I work as a System Development Specialist at City of Toronto.",
    "I'm building an open source project called shovsOS.",
    "I use Cursor as my main editor right now.",
    "My current project budget is $5k.",
    "I'm based in Toronto, but planning to move to Berlin in June.",
    "I have 5 years experience at IBM and City of Toronto.",
    "I prefer TypeScript and Python for backend work.",
    "My goal is to make agents that don't hallucinate after long chats.",
    "Can you remember these details for later?",
    "Summarize what you know about me so far.",
    "Actually, update: I switched from Cursor to VS Code last week.",
    "Correction: my budget is $3k, not $5k.",
    "I'm not moving to Berlin anymore, staying in Toronto.",
    "My role is now focused on AI integration, not just enterprise apps.",
    "Please void the old editor preference.",
    "What editor do I use now?",
    "Where am I based?",
    "What's my budget?",
    "Good, thanks for updating.",
    "Let's talk about something else for a bit.",
    "Explain the difference between RAG and fine-tuning in one paragraph.",
    "How would you optimize a slow Postgres query with 5M rows?",
    "What's a good pattern for WebSocket reconnection in React?",
    "Tell me a quick joke about AI.",
    "Summarize the latest trends in local LLMs.",
    "How do I set up Ollama on macOS?",
    "What's the best way to structure a FastAPI project?",
    "Explain vector databases like I'm 12.",
    "Give me three tips for government data compliance.",
    "What's the difference between deterministic and probabilistic memory?",
    "How does Docker layer caching work?",
    "Recommend a good TypeScript linter setup.",
    "What's the weather usually like in Toronto in April?",
    "Explain phase-aware context in simple terms.",
    "Thanks, back to my project.",
    "Wait, did I say I moved to Berlin? I think I changed my mind.",
    "Remind me what city I told you originally?",
    "And what city am I in now?",
    "Did I ever say I use Cursor or VS Code? I'm confused.",
    "What did I say my budget was first, and what is it now?",
    "Actually I might increase budget to $4k next month, but keep $3k for now.",
    "Note that as a candidate, not a fact yet.",
    "What's my current confirmed budget?",
    "Do you remember my employer?",
    "What project am I building?",
    "Summarize my current setup: name, employer, location, editor, budget, project.",
    "List the facts you have hardened as deterministic.",
    "What candidate signals are you tracking?",
    "What did I originally say about Berlin, and when did I correct it?",
    "If I ask you in a new session tomorrow, will you remember my editor and location?",
]


def _groq_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def _write_log(test_name: str, payload: dict) -> Path:
    log_dir = Path("logs/test_runs/groq_transcript")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{test_name}__{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _session_trace_payload(session_id: str, limit: int = 400) -> dict:
    store = get_trace_store()
    index_events = store.list_events(limit=limit, session_id=session_id)
    full_events = []
    for event in reversed(index_events):
        event_id = event.get("id")
        full_event = store.get_event(event_id) if event_id else event
        if full_event:
            full_events.append(full_event)
    return {"events": full_events}


async def _stream_api_chat(payload: dict) -> list[dict]:
    transport = ASGITransport(app=app)
    events: list[dict] = []
    payload = dict(payload)
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
async def test_groq_tool_use_long_transcript_preserves_current_profile_and_logs_authority():
    if not _groq_available():
        pytest.skip("GROQ_API_KEY not set")

    owner_id = f"groq-transcript-{uuid.uuid4().hex[:8]}"
    session_id = f"groq_transcript_{uuid.uuid4().hex[:8]}"
    model = "groq:llama3-groq-tool-use"

    turns: list[dict] = []
    for message in TRANSCRIPT:
        events = await _stream_api_chat(
            {
                "message": message,
                "session_id": session_id,
                "owner_id": owner_id,
                "model": model,
            }
        )
        turns.append({"message": message, "events": events})
        errors = [event for event in events if event.get("type") == "error"]
        if errors:
            pytest.skip(f"Groq transcript runtime error: {errors[-1]}")

    memory_state = await _api_get(
        f"/sessions/{session_id}/memory-state",
        params={"owner_id": owner_id},
    )
    trace = _session_trace_payload(session_id)
    log_path = _write_log(
        "tool_use_long_transcript",
        {
            "session_id": session_id,
            "owner_id": owner_id,
            "model": model,
            "turns": turns,
            "memory_state": memory_state,
            "trace": trace,
        },
    )
    print(f"\n[groq-log] {log_path}")

    deterministic = {
        (item["predicate"], item["object"])
        for item in memory_state.get("deterministic_facts", [])
        if item.get("status") == "current"
    }
    superseded = {
        (item["predicate"], item["object"])
        for item in memory_state.get("superseded_facts", [])
    }

    assert ("preferred_name", "Shovon") in deterministic, log_path
    assert ("preferred_editor", "VS Code") in deterministic, log_path
    assert ("location", "Toronto") in deterministic, log_path
    assert ("budget_limit", "$3k") in deterministic, log_path
    assert ("current_project", "shovsOS") in deterministic, log_path
    assert ("current_employer", "City of Toronto") in deterministic, log_path
    assert ("professional_role", "System Development Specialist") in deterministic, log_path
    assert ("professional_focus", "AI integration") in deterministic, log_path

    assert ("preferred_editor", "Cursor") in superseded, log_path
    assert ("budget_limit", "$5k") in superseded, log_path

    phase_contexts = [
        event for event in trace.get("events", [])
        if event.get("event_type") == "phase_context"
    ]
    assert phase_contexts, log_path
    assert any("Memory Authority" in str(event.get("data", {}).get("content", "")) for event in phase_contexts), log_path
    assert any("Contradiction Policy:" in str(event.get("data", {}).get("content", "")) for event in phase_contexts), log_path
    assert any("Tool Economy:" in str(event.get("data", {}).get("content", "")) for event in phase_contexts), log_path
