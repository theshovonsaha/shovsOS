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
) -> dict:
    graph = graph or SemanticGraph()

    timeline = graph.list_temporal_facts(session.id, owner_id=owner_id, limit=40)
    current_facts = [item for item in timeline if item.get("status") == "current"]
    superseded_facts = [item for item in timeline if item.get("status") == "superseded"]
    candidate_signals = list(getattr(session, "candidate_signals", []) or [])
    if not candidate_signals:
        candidate_signals = parse_candidate_context(getattr(session, "candidate_context", "") or "")
    stance_signals = [item for item in candidate_signals if str(item.get("signal_type") or "") == "stance"]
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
            "context_line_count": len(compressed_preview),
        },
        "deterministic_facts": current_facts,
        "superseded_facts": superseded_facts,
        "candidate_signals": candidate_signals,
        "stance_signals": stance_signals,
        "context_preview": compressed_preview,
        "recent_memory_signals": [],
        "explanation": [
            "Trusted facts are treated as true and override older memory.",
            "Superseded facts stay visible for audit, but they are no longer active.",
            "Candidate signals are stored separately until the system has stronger grounding.",
            "Stance signals track durable user positions and can trigger drift checks without becoming hard facts automatically.",
        ],
    }
