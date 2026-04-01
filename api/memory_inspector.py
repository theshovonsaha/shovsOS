from __future__ import annotations

from typing import Callable, Optional

from config.trace_store import TraceStore, get_trace_store
from memory.semantic_graph import SemanticGraph


MEMORY_SIGNAL_TYPES = {
    "deterministic_fact_extractor",
    "memory_fact_filter",
    "facts_indexed",
    "memory_write_policy",
}


def _candidate_signals(candidate_context: str) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    for raw_line in (candidate_context or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cleaned = line[2:] if line.startswith("- ") else line
        reason = "under_review"
        text = cleaned
        if "(reason=" in cleaned and cleaned.endswith(")"):
            text, _, reason_part = cleaned.rpartition("(reason=")
            text = text.rstrip()
            reason = reason_part[:-1].strip() or reason
        if text.lower().startswith("candidate:"):
            text = text[len("candidate:"):].strip()
        signals.append({"text": text, "reason": reason})
    return signals


def _summarize_memory_signal(event: dict) -> Optional[dict[str, object]]:
    event_type = str(event.get("event_type") or "")
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    if event_type == "deterministic_fact_extractor":
        fact_count = int(data.get("fact_count") or 0)
        void_count = int(data.get("void_count") or 0)
        if not fact_count and not void_count:
            return None
        return {
            "event_type": event_type,
            "label": "Deterministic extractor",
            "summary": f"Captured {fact_count} trusted fact{'s' if fact_count != 1 else ''} and voided {void_count} prior fact{'s' if void_count != 1 else ''}.",
            "facts": list(data.get("facts") or [])[:6],
            "voids": list(data.get("voids") or [])[:6],
            "created_at": event.get("iso_ts"),
        }
    if event_type == "memory_fact_filter":
        blocked_count = int(data.get("blocked_count") or 0)
        if blocked_count <= 0:
            return None
        blocked = list(data.get("blocked") or [])[:6]
        return {
            "event_type": event_type,
            "label": "Candidate fact filter",
            "summary": f"Held back {blocked_count} low-confidence fact{'s' if blocked_count != 1 else ''} for review instead of treating them as true.",
            "blocked": blocked,
            "created_at": event.get("iso_ts"),
        }
    if event_type == "facts_indexed":
        facts = list(data.get("facts") or [])[:6]
        if not facts:
            return None
        return {
            "event_type": event_type,
            "label": "Fact indexing",
            "summary": f"Indexed {len(facts)} accepted fact{'s' if len(facts) != 1 else ''} for retrieval.",
            "facts": facts,
            "created_at": event.get("iso_ts"),
        }
    if event_type == "memory_write_policy":
        should_write = bool(data.get("should_write_memory"))
        should_compress = bool(data.get("should_compress"))
        return {
            "event_type": event_type,
            "label": "Memory write policy",
            "summary": (
                f"Write memory={'yes' if should_write else 'no'}, "
                f"compress={'yes' if should_compress else 'no'} for this turn."
            ),
            "created_at": event.get("iso_ts"),
        }
    return None


def build_session_memory_payload(
    *,
    session,
    owner_id: str,
    context_preview: Callable[[str], list[str]],
    graph: Optional[SemanticGraph] = None,
    trace_store: Optional[TraceStore] = None,
) -> dict:
    graph = graph or SemanticGraph()
    trace_store = trace_store or get_trace_store()

    timeline = graph.list_temporal_facts(session.id, owner_id=owner_id, limit=40)
    current_facts = [item for item in timeline if item.get("status") == "current"]
    superseded_facts = [item for item in timeline if item.get("status") == "superseded"]
    candidate_signals = _candidate_signals(getattr(session, "candidate_context", "") or "")

    recent_events = trace_store.list_events(
        limit=160,
        session_id=session.id,
        owner_id=owner_id,
    )
    memory_signals: list[dict[str, object]] = []
    for event in recent_events:
        if str(event.get("event_type") or "") not in MEMORY_SIGNAL_TYPES:
            continue
        summary = _summarize_memory_signal(event)
        if summary:
            memory_signals.append(summary)
        if len(memory_signals) >= 8:
            break

    compressed_preview = context_preview(getattr(session, "compressed_context", "") or "")

    return {
        "session_id": session.id,
        "agent_id": getattr(session, "agent_id", "default"),
        "model": getattr(session, "model", ""),
        "context_mode": getattr(session, "context_mode", "v1"),
        "message_count": int(getattr(session, "message_count", 0) or 0),
        "summary": {
            "deterministic_fact_count": len(current_facts),
            "superseded_fact_count": len(superseded_facts),
            "candidate_signal_count": len(candidate_signals),
            "context_line_count": len(compressed_preview),
            "memory_signal_count": len(memory_signals),
        },
        "deterministic_facts": current_facts,
        "superseded_facts": superseded_facts,
        "candidate_signals": candidate_signals,
        "context_preview": compressed_preview,
        "recent_memory_signals": memory_signals,
        "explanation": [
            "Trusted facts are treated as true and override older memory.",
            "Superseded facts stay visible for audit, but they are no longer active.",
            "Candidate signals are stored separately until the system has stronger grounding.",
        ],
    }
