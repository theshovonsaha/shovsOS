from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from engine.candidate_signals import has_correction_signal, merge_candidate_signals, render_candidate_signals
from engine.compression_fact_policy import finalize_compression_fact_records
from engine.deterministic_facts import merge_fact_records, merge_void_records
from memory.vector_engine import VectorEngine


@dataclass
class MemoryCommitPlan:
    new_context: str = ""
    merged_facts: list[dict[str, Any]] | None = None
    merged_voids: list[dict[str, Any]] | None = None
    blocked_keyed_facts: list[dict[str, Any]] | None = None
    candidate_signals: list[dict[str, str]] | None = None
    candidate_context: str = ""
    candidate_signal_updates: bool = False


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
    candidate_signals = merge_candidate_signals(
        existing_candidate_signals or [],
        blocked_keyed_facts,
        extra_signals=new_candidate_signals,
        supersede_matching_stances=has_correction_signal(user_message),
        current_turn=current_turn,
    )
    rendered_candidate_context = render_candidate_signals(candidate_signals) if candidate_signals else merge_candidate_context(existing_candidate_context, blocked_keyed_facts)
    return MemoryCommitPlan(
        new_context=str(new_context or ""),
        merged_facts=merge_fact_records(deterministic_keyed_facts, compression_keyed_facts),
        merged_voids=merge_void_records(deterministic_voids, compression_voids),
        blocked_keyed_facts=blocked_keyed_facts,
        candidate_signals=candidate_signals,
        candidate_context=rendered_candidate_context,
        candidate_signal_updates=(
            candidate_signals != list(existing_candidate_signals or [])
            or rendered_candidate_context != str(existing_candidate_context or "")
        ),
    )


def build_deterministic_memory_commit(
    *,
    deterministic_keyed_facts: list[dict[str, Any]],
    deterministic_voids: list[dict[str, Any]],
    existing_candidate_signals: Optional[list[dict[str, str]]],
    existing_candidate_context: str,
    user_message: str = "",
    new_candidate_signals: Optional[list[dict[str, Any]]] = None,
    current_turn: Optional[int] = None,
) -> MemoryCommitPlan:
    candidate_signals = merge_candidate_signals(
        list(existing_candidate_signals or []),
        [],
        extra_signals=new_candidate_signals,
        supersede_matching_stances=has_correction_signal(user_message),
        current_turn=current_turn,
    )
    rendered_candidate_context = render_candidate_signals(candidate_signals) if candidate_signals else str(existing_candidate_context or "")
    return MemoryCommitPlan(
        new_context="",
        merged_facts=merge_fact_records(deterministic_keyed_facts),
        merged_voids=merge_void_records(deterministic_voids),
        blocked_keyed_facts=[],
        candidate_signals=candidate_signals,
        candidate_context=rendered_candidate_context,
        candidate_signal_updates=(
            candidate_signals != list(existing_candidate_signals or [])
            or rendered_candidate_context != str(existing_candidate_context or "")
        ),
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
            for void in merged_voids:
                subject = str(void.get("subject") or "").strip()
                predicate = str(void.get("predicate") or "").strip()
                if not subject or not predicate:
                    continue
                graph.void_temporal_fact(
                    session_id,
                    subject,
                    predicate,
                    turn,
                    owner_id=owner_id,
                )
            for item in merged_facts:
                subject = str(item.get("subject") or "").strip()
                predicate = str(item.get("predicate") or "").strip()
                if not subject or not predicate or subject == "General":
                    continue
                graph.add_temporal_fact(
                    session_id,
                    subject,
                    predicate,
                    str(item.get("object") or ""),
                    turn,
                    owner_id=owner_id,
                    run_id=run_id,
                )
        except Exception as exc:
            graph_error = str(exc)

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
