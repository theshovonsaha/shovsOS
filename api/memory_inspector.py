from __future__ import annotations

from typing import Callable, Optional

from config.trace_store import TraceStore, get_trace_store
from engine.candidate_signals import parse_candidate_context, render_candidate_signals
from memory.semantic_graph import SemanticGraph


MEMORY_SIGNAL_TYPES = {
    "deterministic_fact_extractor",
    "memory_fact_filter",
    "facts_indexed",
    "memory_write_policy",
    "stance_signals_extracted",
    "evidence_disputed",
}


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
    if event_type == "stance_signals_extracted":
        signals = list(data.get("signals") or [])[:6]
        if not signals:
            return None
        return {
            "event_type": event_type,
            "label": "Stance extractor",
            "summary": f"Captured {len(signals)} stance candidate{'s' if len(signals) != 1 else ''} for later drift checks.",
            "signals": signals,
            "created_at": event.get("iso_ts"),
        }
    if event_type == "evidence_disputed":
        disputed = list(data.get("disputed") or data.get("facts") or [])[:6]
        return {
            "event_type": event_type,
            "label": "Evidence dispute",
            "summary": f"Demoted {len(disputed)} fact{'s' if len(disputed) != 1 else ''} because fresh evidence disputed stored memory.",
            "disputed": disputed,
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
    if hasattr(graph, "get_current_fact_records"):
        current_facts = graph.get_current_fact_records(session.id, owner_id=owner_id, limit=200)
    else:
        current_facts = [item for item in timeline if item.get("status") == "current"]
    superseded_facts = [item for item in timeline if item.get("status") == "superseded"]
    typed_memory_counts: dict[str, int] = {}
    for item in current_facts:
        memory_type = str(item.get("memory_type") or "fact").strip() or "fact"
        typed_memory_counts[memory_type] = typed_memory_counts.get(memory_type, 0) + 1
    exact_policy_memory = [item for item in current_facts if str(item.get("memory_type") or "") == "policy"]
    exact_preference_memory = [item for item in current_facts if str(item.get("memory_type") or "") == "preference"]
    candidate_signals = list(getattr(session, "candidate_signals", []) or [])
    candidate_signal_source = "structured"
    if not candidate_signals:
        candidate_signals = parse_candidate_context(getattr(session, "candidate_context", "") or "")
        candidate_signal_source = "legacy_text"
    stance_signals = [item for item in candidate_signals if str(item.get("signal_type") or "") == "stance"]
    disputed_signals = [
        item for item in candidate_signals
        if str(item.get("reason") or "") == "evidence_disputed"
        or str(item.get("signal_type") or "") == "disputed_fact"
    ]
    conflict_traced = [
        item for item in timeline
        if bool(item.get("conflict_trace")) or bool(item.get("prior_value_disputed"))
    ]
    candidate_context_preview = render_candidate_signals(candidate_signals) if candidate_signals else ""

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
            "stance_signal_count": len(stance_signals),
            "disputed_fact_count": len(disputed_signals),
            "conflict_traced_count": len(conflict_traced),
            "context_line_count": len(compressed_preview),
            "memory_signal_count": len(memory_signals),
            "candidate_signal_source": candidate_signal_source,
            "typed_memory_counts": typed_memory_counts,
            "policy_memory_count": len(exact_policy_memory),
            "preference_memory_count": len(exact_preference_memory),
        },
        "deterministic_facts": current_facts,
        "superseded_facts": superseded_facts,
        "exact_policy_memory": exact_policy_memory,
        "exact_preference_memory": exact_preference_memory,
        "candidate_signals": candidate_signals,
        "stance_signals": stance_signals,
        "disputed_facts": disputed_signals,
        "conflict_traced_facts": conflict_traced,
        "candidate_context_preview": candidate_context_preview,
        "context_preview": compressed_preview,
        "recent_memory_signals": memory_signals,
        "explanation": [
            "Trusted facts are treated as true and override older memory.",
            "Superseded facts stay visible for audit, but they are no longer active.",
            "Candidate signals are stored separately until the system has stronger grounding.",
            "Candidate text is generated from structured candidate signals; legacy text parsing is compatibility-only.",
            "Stance signals track durable user positions and can trigger drift checks without becoming hard facts automatically.",
            "Disputed facts were stored as truth until fresh tool evidence contradicted them; they are demoted to candidates pending verification.",
            "Conflict-traced facts were stored despite an unresolved contradiction; the conflict provenance travels with them.",
            "Policy and preference facts are exact memory lanes; they are injected directly instead of relying on semantic similarity.",
        ],
    }
