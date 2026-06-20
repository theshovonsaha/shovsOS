from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from engine.candidate_signals import has_correction_signal, merge_candidate_signals, render_candidate_signals
from engine.compression_fact_policy import finalize_compression_fact_records
from engine.deterministic_facts import merge_fact_records, merge_void_records
from memory.vector_engine import VectorEngine

try:  # soft import so the pipeline stays usable in isolation/tests
    from engine.conversation_tension import ConversationTension
except Exception:  # pragma: no cover
    ConversationTension = None  # type: ignore[assignment]


# storage_action values that deterministically void the prior conflicting fact.
_VOIDING_ACTIONS = frozenset({
    "void_previous_store_current",
    "supersede_prior_stance",
})
# storage_action values that keep both but stamp conflict provenance.
_TRACE_ACTIONS = frozenset({
    "store_current_with_conflict_trace",
})
# storage_action values that demote disputed stored facts to candidates.
_DEMOTE_ACTIONS = frozenset({
    "demote_to_candidate_pending_verification",
})


@dataclass
class MemoryCommitPlan:
    new_context: str = ""
    merged_facts: list[dict[str, Any]] | None = None
    merged_voids: list[dict[str, Any]] | None = None
    blocked_keyed_facts: list[dict[str, Any]] | None = None
    candidate_signals: list[dict[str, str]] | None = None
    candidate_context: str = ""
    candidate_signal_updates: bool = False
    # Enforcement provenance — inspectable in memory_commit_plan traces.
    tension_storage_action: str = "none"
    tension_voids: list[dict[str, Any]] = field(default_factory=list)
    tension_demotions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MemoryCommitOutcome:
    context_lines: int = 0
    merged_facts: list[dict[str, Any]] | None = None
    merged_voids: list[dict[str, Any]] | None = None
    blocked_keyed_facts: list[dict[str, Any]] | None = None
    candidate_signals: list[dict[str, str]] | None = None
    candidate_context: str = ""
    indexed_fact_keys: list[str] | None = None
    graph_error: str = ""
    index_error: str = ""


def build_grounding_text(
    tool_results: list[dict[str, Any]],
    *,
    successful_only: bool,
    limit: int = 4,
    separator: str = "\n",
) -> str:
    items = tool_results[-limit:]
    if successful_only:
        items = [item for item in items if item.get("success")]
    return separator.join(str(item.get("content") or "") for item in items)


def derive_tension_enforcement(
    conversation_tension: Optional["ConversationTension"],
    deterministic_keyed_facts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Translate the tension lane's storage_action into deterministic
    storage operations. This is the enforcement wire: detection without
    this function is advisory prose; with it, the policy is structural.

    Returns (tension_voids, tension_demotions, action).
    - tension_voids: (subject, predicate) pairs to void in the graph,
      derived from the EXACT conflicts the analyzer found — no LLM,
      no [VOIDS:] marker dependency.
    - tension_demotions: evidence-disputed stored facts to re-file as
      candidate signals instead of deterministic truth.
    """
    if conversation_tension is None:
        return [], [], "none"
    action = str(getattr(conversation_tension, "storage_action", "") or "none")

    tension_voids: list[dict[str, Any]] = []
    if action in _VOIDING_ACTIONS:
        for conflict in getattr(conversation_tension, "conflicting_facts", ()) or ():
            subject = str(conflict.get("subject") or "").strip()
            predicate = str(conflict.get("predicate") or "").strip()
            if subject and predicate:
                tension_voids.append({
                    "subject": subject,
                    "predicate": predicate,
                    "void_source": "tension_policy",
                    "policy": action,
                })

    if action in _TRACE_ACTIONS:
        conflict_keys = {
            (str(c.get("subject") or "").strip().lower(),
             str(c.get("predicate") or "").strip().lower())
            for c in getattr(conversation_tension, "conflicting_facts", ()) or ()
        }
        for fact in deterministic_keyed_facts:
            fk = (str(fact.get("subject") or "").strip().lower(),
                  str(fact.get("predicate") or "").strip().lower())
            if fk in conflict_keys:
                fact["conflict_trace"] = True
                fact["prior_value_disputed"] = True

    tension_demotions: list[dict[str, Any]] = []
    if action in _DEMOTE_ACTIONS:
        for ec in getattr(conversation_tension, "evidence_conflicts", ()) or ():
            subject = str(ec.get("subject") or "").strip()
            predicate = str(ec.get("predicate") or "").strip()
            stored = str(ec.get("stored") or "").strip()
            if subject and predicate:
                tension_demotions.append({
                    "text": f"{subject} {predicate} {stored}".strip(),
                    "reason": "evidence_disputed",
                    "source": "tension_policy",
                    "signal_type": "disputed_fact",
                })
                # Void the deterministic version — the fact survives only
                # as a candidate until re-verified.
                tension_voids.append({
                    "subject": subject,
                    "predicate": predicate,
                    "void_source": "tension_policy",
                    "policy": action,
                })

    return tension_voids, tension_demotions, action


def plan_memory_commit(
    *,
    context_result: tuple[Any, ...],
    user_message: str,
    tool_results: list[dict[str, Any]],
    deterministic_keyed_facts: list[dict[str, Any]],
    deterministic_voids: list[dict[str, Any]],
    current_facts: Optional[Iterable[tuple[str, str, str]]],
    existing_candidate_signals: Optional[list[dict[str, str]]],
    existing_candidate_context: str,
    conversation_tension: Optional["ConversationTension"] = None,
    new_candidate_signals: Optional[list[dict[str, Any]]] = None,
    current_turn: Optional[int] = None,
) -> MemoryCommitPlan:
    new_context = context_result[0] if context_result else ""
    compression_keyed_facts = list(context_result[1] or []) if len(context_result) > 1 else []
    compression_voids = list(context_result[2] or []) if len(context_result) > 2 else []
    compression_keyed_facts, blocked_keyed_facts = finalize_compression_fact_records(
        compression_keyed_facts,
        user_message=user_message,
        grounding_text=build_grounding_text(tool_results, successful_only=True),
        deterministic_facts=deterministic_keyed_facts,
        current_facts=current_facts,
    )

    # ── Enforcement wire: tension policy → deterministic storage ops ──
    tension_voids, tension_demotions, tension_action = derive_tension_enforcement(
        conversation_tension,
        deterministic_keyed_facts,
    )

    candidate_signals = merge_candidate_signals(
        existing_candidate_signals or [],
        blocked_keyed_facts,
        extra_signals=[*(new_candidate_signals or []), *tension_demotions],
        supersede_matching_stances=has_correction_signal(user_message),
        current_turn=current_turn,
    )
    rendered_candidate_context = render_candidate_signals(candidate_signals) if candidate_signals else merge_candidate_context(existing_candidate_context, blocked_keyed_facts)
    return MemoryCommitPlan(
        new_context=str(new_context or ""),
        merged_facts=merge_fact_records(deterministic_keyed_facts, compression_keyed_facts),
        merged_voids=merge_void_records(deterministic_voids, compression_voids, tension_voids),
        blocked_keyed_facts=blocked_keyed_facts,
        candidate_signals=candidate_signals,
        candidate_context=rendered_candidate_context,
        candidate_signal_updates=(
            candidate_signals != list(existing_candidate_signals or [])
            or rendered_candidate_context != str(existing_candidate_context or "")
        ),
        tension_storage_action=tension_action,
        tension_voids=tension_voids,
        tension_demotions=tension_demotions,
    )


def build_deterministic_memory_commit(
    *,
    deterministic_keyed_facts: list[dict[str, Any]],
    deterministic_voids: list[dict[str, Any]],
    existing_candidate_signals: Optional[list[dict[str, str]]],
    existing_candidate_context: str,
    user_message: str = "",
    conversation_tension: Optional["ConversationTension"] = None,
    new_candidate_signals: Optional[list[dict[str, Any]]] = None,
    current_turn: Optional[int] = None,
) -> MemoryCommitPlan:
    tension_voids, tension_demotions, tension_action = derive_tension_enforcement(
        conversation_tension,
        deterministic_keyed_facts,
    )
    candidate_signals = merge_candidate_signals(
        list(existing_candidate_signals or []),
        [],
        extra_signals=[*(new_candidate_signals or []), *tension_demotions],
        supersede_matching_stances=has_correction_signal(user_message),
        current_turn=current_turn,
    )
    rendered_candidate_context = render_candidate_signals(candidate_signals) if candidate_signals else str(existing_candidate_context or "")
    return MemoryCommitPlan(
        new_context="",
        merged_facts=merge_fact_records(deterministic_keyed_facts),
        merged_voids=merge_void_records(deterministic_voids, tension_voids),
        blocked_keyed_facts=[],
        candidate_signals=candidate_signals,
        candidate_context=rendered_candidate_context,
        candidate_signal_updates=(
            candidate_signals != list(existing_candidate_signals or [])
            or rendered_candidate_context != str(existing_candidate_context or "")
        ),
        tension_storage_action=tension_action,
        tension_voids=tension_voids,
        tension_demotions=tension_demotions,
    )


async def apply_memory_commit(
    *,
    sessions,
    session_id: str,
    owner_id: Optional[str],
    agent_id: str,
    turn: int,
    run_id: str,
    user_message: str,
    assistant_response: str,
    graph,
    plan: MemoryCommitPlan,
    current_context: str,
    planned_locus_id: str = "",
) -> MemoryCommitOutcome:
    next_context = str(plan.new_context or current_context or "")
    if plan.new_context:
        sessions.update_context(session_id, plan.new_context)

    blocked_keyed_facts = list(plan.blocked_keyed_facts or [])
    candidate_signals = list(plan.candidate_signals or [])
    candidate_context = str(plan.candidate_context or "")
    if plan.candidate_signal_updates or blocked_keyed_facts:
        if candidate_signals:
            sessions.update_candidate_signals(session_id, candidate_signals)
        elif candidate_context:
            sessions.update_candidate_context(session_id, candidate_context)
        else:
            sessions.update_candidate_signals(session_id, [])

    merged_voids = list(plan.merged_voids or [])
    merged_facts = list(plan.merged_facts or [])
    graph_error = ""
    if graph is not None and (merged_facts or merged_voids):
        try:
            graph_facts: list[dict] = []
            for item in merged_facts:
                subject = str(item.get("subject") or "").strip()
                predicate = str(item.get("predicate") or "").strip()
                if not subject or not predicate or subject == "General":
                    continue
                item_locus = str(item.get("locus_id") or "").strip() or (planned_locus_id or None)
                graph_facts.append(
                    {
                        **item,
                        "subject": subject,
                        "predicate": predicate,
                        "object": str(item.get("object") or ""),
                        "run_id": run_id,
                        "locus_id": item_locus,
                    }
                )
            graph.replace_temporal_facts(
                session_id,
                facts=graph_facts,
                voids=merged_voids,
                turn=turn,
                owner_id=owner_id,
                run_id=run_id,
                locus_id=planned_locus_id or None,
            )
            for item in graph_facts:
                subject = str(item.get("subject") or "").strip()
                predicate = str(item.get("predicate") or "").strip()
                try:
                    from plugins.hook_registry import hooks
                    hooks.emit_sync(
                        "memory_stored",
                        {
                            "subject": subject,
                            "predicate": predicate,
                            "object": str(item.get("object") or ""),
                            "turn": turn,
                            "owner_id": owner_id,
                            "conflict_trace": bool(item.get("conflict_trace")),
                        },
                        run_id=run_id,
                        session_id=session_id,
                    )
                except Exception:
                    pass
        except Exception as exc:
            graph_error = str(exc)

        if hasattr(graph, "compile_locus_drawer"):
            touched_loci: set[str] = set()
            for item in merged_facts:
                lid = str(item.get("locus_id") or "").strip()
                if lid:
                    touched_loci.add(lid)
            if planned_locus_id:
                touched_loci.add(str(planned_locus_id).strip())
            for lid in touched_loci:
                if not lid:
                    continue
                try:
                    graph.compile_locus_drawer(lid, owner_id=owner_id)
                except Exception:
                    pass

    indexed_fact_keys: list[str] = []
    index_error = ""
    if merged_facts:
        try:
            vector_engine = VectorEngine(session_id, agent_id=agent_id, owner_id=owner_id)
            anchor_text = f"User: {user_message}\nAssistant: {assistant_response}"
            for item in merged_facts:
                key = str(
                    item.get("key")
                    or " ".join(
                        part
                        for part in (
                            item.get("subject"),
                            item.get("predicate"),
                        )
                        if part
                    )
                    or item.get("fact")
                    or ""
                ).strip()
                fact = str(item.get("fact") or "").strip()
                if not key or not fact:
                    continue
                await vector_engine.index(
                    key=key,
                    anchor=anchor_text,
                    metadata={"fact": fact, "run_id": run_id, "owner_id": owner_id or ""},
                )
                indexed_fact_keys.append(key)
        except Exception as exc:
            index_error = str(exc)

    return MemoryCommitOutcome(
        context_lines=_count_context_lines(next_context),
        merged_facts=merged_facts,
        merged_voids=merged_voids,
        blocked_keyed_facts=blocked_keyed_facts,
        candidate_signals=candidate_signals,
        candidate_context=candidate_context,
        indexed_fact_keys=indexed_fact_keys,
        graph_error=graph_error,
        index_error=index_error,
    )


def merge_candidate_context(existing: str, blocked_records: list[dict[str, Any]]) -> str:
    lines = [line.strip() for line in (existing or "").splitlines() if line.strip()]
    seen = {line.lower() for line in lines}
    for record in blocked_records or []:
        fact = str(
            record.get("fact")
            or " ".join(
                part
                for part in (
                    record.get("subject"),
                    record.get("predicate"),
                    record.get("object"),
                )
                if part
            )
        ).strip()
        if not fact:
            continue
        reason = str(record.get("grounding_reason") or "candidate")
        line = f"- Candidate: {fact} (reason={reason})"
        if line.lower() in seen:
            continue
        lines.append(line)
        seen.add(line.lower())
    return "\n".join(lines[-12:])


def _count_context_lines(context: str) -> int:
    return len([line for line in str(context or "").splitlines() if line.strip()])
