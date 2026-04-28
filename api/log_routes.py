"""
Add these routes to main.py to expose the internal log stream.

1. Add to imports at top of main.py:
   from config.logger import get_logger, log

2. Add after app = FastAPI(...):
   from api.log_routes import setup_log_routes
   setup_log_routes(app)

3. In chat_stream(), after agent_instance is created, add:
   log("agent", "system", f"Request: agent={agent_id} model={model or 'default'}")
"""

from dataclasses import asdict
import json
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional

from config.trace_store import get_trace_store
from api.owner import require_owner_id
from orchestration.run_store import get_run_store


def _clip_text(value: object, max_chars: int = 420) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _collect_evidence_entries(
    context_trace: object,
    *,
    source: str,
    phase: Optional[str] = None,
    tool_turn: Optional[int] = None,
    pass_id: Optional[int] = None,
) -> list[dict]:
    if not isinstance(context_trace, dict):
        return []

    entries: list[dict] = []
    included = context_trace.get("included")
    if not isinstance(included, list):
        return entries

    for item in included:
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "evidence" and item.get("item_id") != "working_evidence":
            continue

        provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
        entries.append(
            {
                "source": source,
                "phase": phase or context_trace.get("phase") or "",
                "tool_turn": tool_turn,
                "pass_id": pass_id,
                "item_id": item.get("item_id") or "working_evidence",
                "trace_id": item.get("trace_id"),
                "provenance": provenance,
                "summary": _clip_text(item.get("content") or item.get("summary") or ""),
            }
        )
    return entries


def _build_run_replay_payload(*, run_id: str, owner_id: str, trace_limit: int = 160) -> dict:
    run_store = get_run_store()
    trace_store = get_trace_store()

    run = run_store.get(run_id)
    if run is None or run.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="run not found")

    checkpoints = run_store.list_checkpoints(run_id)
    passes = run_store.list_passes(run_id)
    usage = run_store.summarize_usage(run_id)
    artifacts = run_store.list_artifacts(run_id)
    evals = run_store.list_evals(run_id)
    trace_events = trace_store.list_events(
        limit=max(20, min(500, trace_limit)),
        run_id=run_id,
        owner_id=owner_id,
    )

    evidence_entries: list[dict] = []
    for record in passes:
        evidence_entries.extend(
            _collect_evidence_entries(
                record.compiled_context,
                source="run_pass",
                phase=record.phase,
                tool_turn=record.tool_turn,
                pass_id=record.pass_id,
            )
        )

    for event in trace_events:
        if event.get("event_type") != "phase_context":
            continue
        evidence_entries.extend(
            _collect_evidence_entries(
                event.get("data"),
                source="trace_event",
                phase=(event.get("data") or {}).get("phase") if isinstance(event.get("data"), dict) else None,
                tool_turn=(event.get("data") or {}).get("tool_turn") if isinstance(event.get("data"), dict) else None,
            )
        )

    deduped_evidence: list[dict] = []
    seen_evidence: set[tuple] = set()
    for entry in evidence_entries:
        key = (
            entry.get("source"),
            entry.get("phase"),
            entry.get("tool_turn"),
            entry.get("pass_id"),
            entry.get("trace_id"),
            entry.get("summary"),
        )
        if key in seen_evidence:
            continue
        seen_evidence.add(key)
        deduped_evidence.append(entry)

    return {
        "found": True,
        "run": asdict(run),
        "summary": {
            "checkpoint_count": len(checkpoints),
            "pass_count": len(passes),
            "artifact_count": len(artifacts),
            "eval_count": len(evals),
            "trace_event_count": len(trace_events),
            "evidence_count": len(deduped_evidence),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "estimated_cost_usd": usage["estimated_cost_usd"],
        },
        "latest_checkpoint": asdict(checkpoints[-1]) if checkpoints else None,
        "latest_pass": asdict(passes[-1]) if passes else None,
        "checkpoints": [asdict(item) for item in checkpoints],
        "passes": [asdict(item) for item in passes],
        "artifacts": [asdict(item) for item in artifacts],
        "evals": [asdict(item) for item in evals],
        "evidence": deduped_evidence,
        "trace_events": trace_events,
    }


def setup_log_routes(app: FastAPI):
    from config.logger import get_logger
    trace_store = get_trace_store()
    _require_owner_id = require_owner_id

    @app.get("/logs/stream")
    async def log_stream(
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        owner_id: Optional[str] = None,
    ):
        """
        SSE stream of internal log entries.
        Optional filters: ?session_id=xxx&category=tool&owner_id=...
        """
        logger = get_logger()
        normalized_owner_id = _require_owner_id(owner_id) if owner_id is not None else None

        async def generate():
            # Send recent history first so panel isn't blank on connect
            for entry in logger.recent(
                limit=80,
                session_id=session_id,
                category=category,
                owner_id=normalized_owner_id,
            ):
                yield entry.to_sse()

            # Then stream live
            q = logger.subscribe()
            try:
                async for chunk in logger.stream(q):
                    # Parse and filter
                    try:
                        data = json.loads(chunk.replace("data: ", "").strip())
                        if session_id and data.get("session") not in (session_id, "system"):
                            continue
                        if category and data.get("category") != category:
                            continue
                        if normalized_owner_id and data.get("owner_id") != normalized_owner_id:
                            continue
                    except Exception:
                        pass
                    yield chunk
            finally:
                logger.unsubscribe(q)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    @app.get("/logs/recent")
    async def log_recent(
        limit: int = 100,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        owner_id: Optional[str] = None,
    ):
        """REST endpoint for recent logs — useful for initial load."""
        logger = get_logger()
        normalized_owner_id = _require_owner_id(owner_id) if owner_id is not None else None
        entries = logger.recent(
            limit=limit,
            session_id=session_id,
            category=category,
            owner_id=normalized_owner_id,
        )
        return {"logs": [asdict(e) for e in entries]}

    @app.get("/logs/traces/recent")
    async def trace_recent(
        limit: int = 120,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        event_type: Optional[str] = None,
        before_ts: Optional[float] = None,
    ):
        """
        Paginated trace index for monitor UIs.
        Returns compact summaries with optional payload references.
        """
        owner_id = _require_owner_id(owner_id)
        safe_limit = max(1, min(500, limit))
        normalized_type = None if not event_type or event_type == "all" else event_type
        events = trace_store.list_events(
            limit=safe_limit,
            session_id=session_id,
            run_id=run_id,
            owner_id=owner_id,
            event_type=normalized_type,
            before_ts=before_ts,
        )
        return {
            "events": events,
            "count": len(events),
            "next_before_ts": events[-1].get("ts") if events else None,
        }

    @app.get("/logs/traces/event/{event_id}")
    async def trace_event(event_id: str, owner_id: Optional[str] = None):
        """Fetch one trace event including full payload when available."""
        owner_id = _require_owner_id(owner_id)
        event = trace_store.get_event(event_id)
        if not event or event.get("owner_id") != owner_id:
            return {"found": False, "event": None}
        return {"found": True, "event": event}

    @app.get("/logs/traces/stats")
    async def trace_stats(
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        window: int = 400,
    ):
        """High-level trace metrics for dashboard widgets."""
        owner_id = _require_owner_id(owner_id)
        safe_window = max(10, min(2000, window))
        stats = trace_store.stats(session_id=session_id, run_id=run_id, owner_id=owner_id, window=safe_window)
        return stats

    @app.get("/logs/traces/run/{run_id}")
    async def trace_run_replay(
        run_id: str,
        owner_id: Optional[str] = None,
        trace_limit: int = 160,
    ):
        """Assemble one replayable run object from checkpoints, passes, artifacts, evals, and traces."""
        owner_id = _require_owner_id(owner_id)
        return _build_run_replay_payload(run_id=run_id, owner_id=owner_id, trace_limit=trace_limit)
