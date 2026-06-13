from __future__ import annotations

from typing import Callable, Optional

from engine.candidate_signals import parse_candidate_context
from memory.semantic_graph import SemanticGraph


def build_memory_payload(
    *,
    session,
    owner_id: str,
    context_preview: Callable[[str], list[str]],
    graph: Optional[SemanticGraph] = None,
    timeline_limit: int = 40,
) -> dict:
    """Build the full inspectable memory state for one session.

    Correctness notes:
      - `deterministic_facts` comes from get_current_facts(), NOT from a
        truncated timeline slice. The timeline limit only bounds the
        history view; it must never make active facts disappear from
        the summary counts.
      - `graph` should be passed explicitly. Default-constructing one
        opens the default db_path, which silently diverges from a
        facade configured with a custom path. We keep the fallback for
        backward compatibility but flag it in the payload so a mismatch
        is visible instead of mysterious.
    """
    graph_was_defaulted = graph is None
    graph = graph or SemanticGraph()

    # ── Current truth: authoritative source, no limit applied ──
    if hasattr(graph, "get_current_fact_records"):
        current_facts = graph.get_current_fact_records(session.id, owner_id=owner_id, limit=200)
    else:
        current_triples = graph.get_current_facts(session.id, owner_id=owner_id)
        current_facts = [
            {"subject": s, "predicate": p, "object": o, "status": "current", "memory_type": "fact"}
            for (s, p, o) in current_triples
        ]

    # ── History: bounded view for audit, newest first ──
    timeline = graph.list_temporal_facts(session.id, owner_id=owner_id, limit=timeline_limit)
    superseded_facts = [item for item in timeline if item.get("status") == "superseded"]
    typed_memory_counts: dict[str, int] = {}
    for item in current_facts:
        memory_type = str(item.get("memory_type") or "fact").strip() or "fact"
        typed_memory_counts[memory_type] = typed_memory_counts.get(memory_type, 0) + 1
    exact_policy_memory = [item for item in current_facts if str(item.get("memory_type") or "") == "policy"]
    exact_preference_memory = [item for item in current_facts if str(item.get("memory_type") or "") == "preference"]

    # ── Candidate layer ──
    candidate_signals = list(getattr(session, "candidate_signals", []) or [])
    if not candidate_signals:
        candidate_signals = parse_candidate_context(getattr(session, "candidate_context", "") or "")
    stance_signals = [
        item for item in candidate_signals
        if str(item.get("signal_type") or "") == "stance"
    ]
    # Facts demoted by the evidence-vs-memory tension policy: stored truth
    # that fresh tool evidence disputed. Surfaced as a distinct lane because
    # "the system caught its own stale memory" is the wedge's core demo.
    disputed_signals = [
        item for item in candidate_signals
        if str(item.get("reason") or "") == "evidence_disputed"
        or str(item.get("signal_type") or "") == "disputed_fact"
    ]
    # Facts stored with conflict provenance (store_current_with_conflict_trace).
    conflict_traced = [
        item for item in timeline
        if bool(item.get("conflict_trace")) or bool(item.get("prior_value_disputed"))
    ]

    compressed_preview = context_preview(getattr(session, "compressed_context", "") or "")

    return {
        "session_id": session.id,
        "agent_id": getattr(session, "agent_id", "default"),
        "model": getattr(session, "model", ""),
        "context_mode": getattr(session, "context_mode", "v1"),
        "message_count": int(getattr(session, "message_count", 0) or 0),
        "graph_defaulted": graph_was_defaulted,
        "summary": {
            "deterministic_fact_count": len(current_facts),
            "superseded_fact_count": len(superseded_facts),
            "candidate_signal_count": len(candidate_signals),
            "stance_signal_count": len(stance_signals),
            "disputed_fact_count": len(disputed_signals),
            "conflict_traced_count": len(conflict_traced),
            "context_line_count": len(compressed_preview),
            "timeline_truncated": len(timeline) >= timeline_limit,
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
        "context_preview": compressed_preview,
        "explanation": [
            "Trusted facts are treated as true and override older memory.",
            "Superseded facts stay visible for audit, but they are no longer active.",
            "Candidate signals are stored separately until the system has stronger grounding.",
            "Stance signals track durable user positions and can trigger drift checks without becoming hard facts automatically.",
            "Disputed facts were stored as truth until fresh tool evidence contradicted them; they are demoted to candidates pending verification.",
            "Conflict-traced facts were stored despite an unresolved contradiction; the conflict provenance travels with them.",
            "Policy and preference facts are exact memory lanes; they are injected directly instead of relying on semantic similarity.",
        ],
    }
