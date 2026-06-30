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


def _latest_event(trace_events: list[dict], event_types: set[str]) -> Optional[dict]:
    for event in trace_events:
        if event.get("event_type") in event_types:
            return event
    return None


def _event_summary(event: Optional[dict], fallback: str = "not recorded") -> str:
    if not event:
        return fallback
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    preview = event.get("preview")
    for key in ("strategy", "notes", "content_preview", "summary", "message", "reason"):
        if data.get(key):
            return _clip_text(data.get(key), 180)
    if data.get("tool_name"):
        status = "failed" if data.get("success") is False else "recorded"
        return _clip_text(f"{data.get('tool_name')} {status}", 180)
    return _clip_text(preview or event.get("event_type") or fallback, 180)


def _extract_language_kernel_run_map(trace_events: list[dict]) -> Optional[dict]:
    for event in trace_events:
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        ledger = data.get("run_ledger") if isinstance(data.get("run_ledger"), dict) else data
        kernel = ledger.get("language_kernel") if isinstance(ledger.get("language_kernel"), dict) else data.get("language_kernel")
        if not isinstance(kernel, dict):
            continue
        run_map = kernel.get("ui_run_map")
        if not isinstance(run_map, dict):
            continue
        sections = run_map.get("sections")
        if not isinstance(sections, list):
            continue
        clean_sections = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            section_id = str(section.get("id") or section.get("label") or "").strip()
            label = str(section.get("label") or section.get("id") or "").strip()
            if not section_id or not label:
                continue
            clean_sections.append({
                "id": section_id,
                "label": label,
                "status": str(section.get("status") or "not_recorded"),
                "count": int(section.get("count") or 0),
                "event_id": event.get("id"),
                "event_type": event.get("event_type"),
            })
        if clean_sections:
            return {
                "version": str(run_map.get("version") or kernel.get("version") or ""),
                "next_focus": str(run_map.get("next_focus") or ""),
                "sections": clean_sections,
            }
    return None


def _kernel_section_summary(section: dict) -> str:
    section_id = str(section.get("id") or "")
    status = str(section.get("status") or "not_recorded").replace("_", " ")
    count = int(section.get("count") or 0)
    if section_id == "objective":
        return "Current run goal is captured." if count else "No objective recorded yet."
    if section_id == "plan":
        return f"{count} plan step(s) recorded." if count else "Plan not recorded yet."
    if section_id == "tools":
        return f"{count} tool call(s) linked." if count else "No tool calls yet."
    if section_id == "evidence":
        return f"{count} evidence item(s) selected." if count else "Evidence not selected yet."
    if section_id == "memory":
        return f"{count} memory write(s) recorded." if count else "No memory write recorded."
    if section_id == "verification":
        return f"{count or 1} issue(s) need attention." if status == "blocked" else f"Verification is {status}."
    if section_id == "response":
        return "Final response is blocked." if status == "blocked" else "Final response is allowed."
    return f"{section.get('label') or section_id} is {status}."


def _build_operator_story(
    *,
    run,
    checkpoints: list,
    passes: list,
    artifacts: list,
    evals: list,
    evidence: list[dict],
    trace_events: list[dict],
    usage: dict,
) -> dict:
    latest_plan = _latest_event(trace_events, {"plan", "plan_steps", "continuation_gate"})
    latest_policy = _latest_event(trace_events, {"control_policy", "policy_selected", "policy_violation"})
    latest_graph = _latest_event(trace_events, {"pass_graph_execution", "pass_node_started", "pass_node_completed", "pass_node_failed"})
    latest_context = _latest_event(trace_events, {"phase_packet", "phase_context", "compiled_context"})
    latest_tool_result = _latest_event(trace_events, {"tool_result"})
    latest_tool_call = _latest_event(trace_events, {"tool_call"})
    latest_recovery = _latest_event(trace_events, {"recovery_started", "completion_gate"})
    latest_verification = _latest_event(trace_events, {"verification_warning", "verification_result"})
    latest_memory = _latest_event(trace_events, {"memory_commit_plan", "memory_write_policy", "memory_commit_skipped"})
    latest_response = _latest_event(trace_events, {"assistant_response"})
    kernel_run_map = _extract_language_kernel_run_map(trace_events)

    def lane(
        lane_id: str,
        label: str,
        event: Optional[dict],
        *,
        summary: str = "",
        count: int = 0,
        status: str = "idle",
    ) -> dict:
        event_data = event.get("data") if event and isinstance(event.get("data"), dict) else {}
        lane_status = status
        if event:
            if event_data.get("success") is False or "warning" in str(event.get("event_type") or ""):
                lane_status = "attention"
            elif event_data.get("supported") is False:
                lane_status = "attention"
            else:
                lane_status = "done"
        return {
            "id": lane_id,
            "label": label,
            "status": lane_status,
            "count": count,
            "event_id": event.get("id") if event else None,
            "event_type": event.get("event_type") if event else None,
            "phase": event_data.get("phase") or "",
            "summary": summary or _event_summary(event),
        }

    base_story = {
        "status": run.status,
        "objective": _clip_text(
            getattr(passes[-1], "objective", "")
            if passes
            else getattr(checkpoints[-1], "strategy", "")
            if checkpoints
            else "",
            260,
        ),
        "cost": {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "estimated_cost_usd": usage["estimated_cost_usd"],
        },
        "next_best_action": _event_summary(
            _latest_event(trace_events, {"continuation_state_updated", "replan_recommended", "verification_warning"}),
            "No follow-up action recorded.",
        ),
        "artifact_count": len(artifacts),
    }
    if kernel_run_map:
        base_story["source"] = "language_kernel"
        base_story["next_best_action"] = (
            f"Next focus: {kernel_run_map['next_focus']}"
            if kernel_run_map.get("next_focus")
            else base_story["next_best_action"]
        )
        base_story["lanes"] = [
            {
                "id": section["id"],
                "label": section["label"],
                "status": section["status"],
                "count": section["count"],
                "event_id": section.get("event_id"),
                "event_type": section.get("event_type"),
                "phase": section["id"],
                "summary": _kernel_section_summary(section),
            }
            for section in kernel_run_map["sections"]
        ]
        return base_story
    base_story["source"] = "trace_timeline"
    base_story["lanes"] = [
        lane("plan", "Plan", latest_plan, count=len([p for p in passes if str(p.phase).lower() == "planning"])),
        lane("policy", "Policy", latest_policy, count=len([e for e in trace_events if e.get("event_type") in {"control_policy", "policy_selected", "policy_violation"}])),
        lane("graph", "Graph", latest_graph, count=len([e for e in trace_events if e.get("event_type") in {"pass_graph_execution", "pass_node_started", "pass_node_completed", "pass_node_failed"}])),
        lane("context", "Context", latest_context, count=len([e for e in trace_events if e.get("event_type") in {"phase_packet", "phase_context", "compiled_context"}])),
        lane("tool", "Tools", latest_tool_result or latest_tool_call, count=len([e for e in trace_events if e.get("event_type") in {"tool_call", "tool_result"}])),
        lane("evidence", "Evidence", None, summary=(evidence[0].get("summary") if evidence else "No evidence selected yet."), count=len(evidence), status="done" if evidence else "idle"),
        lane("recovery", "Recovery", latest_recovery, count=len([e for e in trace_events if e.get("event_type") in {"recovery_started", "completion_gate"}])),
        lane("memory", "Memory", latest_memory, count=len([e for e in trace_events if "memory" in str(e.get("event_type") or "")])),
        lane("verify", "Verify", latest_verification, count=len(evals)),
        lane("response", "Response", latest_response, count=len([e for e in trace_events if e.get("event_type") == "assistant_response"])),
    ]
    return base_story


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

    operator_story = _build_operator_story(
        run=run,
        checkpoints=checkpoints,
        passes=passes,
        artifacts=artifacts,
        evals=evals,
        evidence=deduped_evidence,
        trace_events=trace_events,
        usage=usage,
    )

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
        "operator_story": operator_story,
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
