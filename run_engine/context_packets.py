from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from engine.candidate_signals import parse_candidate_context, render_candidate_signals
from engine.context_compiler import CompiledPhaseContext, compile_context_items
from engine.context_item_builders import (
    build_available_tools_item,
    build_candidate_context_item,
    build_conversation_tension_item,
    build_core_instruction_item,
    build_deterministic_facts_item,
    build_historical_context_item,
    build_loop_contract_item,
    build_memory_authority_item,
    build_runtime_metadata_item,
    build_session_anchor_item,
    build_working_evidence_item,
)
from engine.conversation_tension import (
    ConversationTension,
    conversation_tension_audit_payload,
    render_conversation_tension,
)
from engine.context_schema import ContextItem, ContextKind, ContextPhase
from run_engine.evidence_lane import build_working_evidence_block, build_working_evidence_snapshot
from run_engine.meta_context import build_meta_context_block, build_meta_context_snapshot
from run_engine.tool_contract import format_tool_result_line
from run_engine.types import CompiledPassPacket, RunEngineRequest
from memory.task_tracker import get_session_task_tracker


DEFAULT_PHASE_CHAR_BUDGETS: dict[ContextPhase, int] = {
    ContextPhase.PLANNING: 21000,
    ContextPhase.ACTING: 28000,
    ContextPhase.RESPONSE: 31000,
    ContextPhase.VERIFICATION: 27000,
    ContextPhase.MEMORY_COMMIT: 16000,
}


@dataclass(frozen=True)
class PacketBuildInputs:
    request: RunEngineRequest
    session: Any
    phase: ContextPhase
    system_prompt: str
    current_context: str
    allowed_tools: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    effective_objective: str = ""
    tool_turn: int = 0
    strategy: str = ""
    notes: str = ""
    observation_status: str = ""
    observation_tools: Optional[list[str]] = None
    final_response: str = ""
    current_facts: Optional[list[tuple[str, str, str]]] = None
    conversation_tension: Optional[ConversationTension] = None
    active_skill_context: str = ""
    active_skill_name: str = ""
    capability_context: str = ""
    code_intent_note: str = ""
    execution_risk_tier: str = ""
    correction_turn: bool = False
    direct_fact_memory_only: bool = False
    available_loci: Optional[list[dict[str, str]]] = None
    planned_locus_id: str = ""
    # Per-turn plan_steps from the planner. Updated in place by the runtime
    # as tools complete. Visible to ACTING/RESPONSE so the actor can see
    # remaining work and the response phase can flag partial completion.
    plan_steps: Optional[list[dict[str, Any]]] = None
    # Spatial drawers (Slice 5): pre-fetched locus drawers (primary + 1-hop
    # neighbors) for the planning phase. Each entry: {locus_id, hop, score,
    # content}. Visible to PLANNING only — planner reads them to decide
    # which loci to query; actor uses tools, not raw drawer dumps.
    spatial_drawers: Optional[list[dict[str, Any]]] = None
    # Canonical run ledger snapshot. In shadow mode this is an additional
    # consistency lane; in enforced mode it becomes the state contract phases
    # must obey. Passed as an object to avoid serializing the full trace into
    # every prompt.
    run_ledger: Optional[Any] = None


def build_phase_packet(
    *,
    context_engine: Optional[object],
    context_governor: Optional[object] = None,
    inputs: PacketBuildInputs,
) -> CompiledPassPacket:
    items: list[ContextItem] = []
    request = inputs.request
    session = inputs.session

    items.append(
        build_runtime_metadata_item(
            source="run_engine",
            trace_id="run_engine:runtime_metadata",
            phase=inputs.phase,
            tool_turn=inputs.tool_turn,
            provenance={"phase": inputs.phase.value},
        )
    )

    instruction_item = build_core_instruction_item(
        content=inputs.system_prompt,
        source="system_prompt",
        trace_id="run_engine:core_instruction",
    )
    if instruction_item is not None:
        items.append(instruction_item)

    if inputs.run_ledger is not None and hasattr(inputs.run_ledger, "render_for_phase"):
        try:
            ledger_content = str(inputs.run_ledger.render_for_phase(inputs.phase) or "").strip()
            ledger_packet = inputs.run_ledger.to_phase_packet(inputs.phase) if hasattr(inputs.run_ledger, "to_phase_packet") else {}
        except Exception:
            ledger_content = ""
            ledger_packet = {}
        if ledger_content:
            items.append(
                ContextItem(
                    item_id="canonical_run_ledger",
                    kind=ContextKind.RUNTIME,
                    title="Canonical Run Ledger",
                    content=ledger_content,
                    source="run_ledger",
                    priority=18,
                    max_chars=1800,
                    trace_id="run_engine:run_ledger",
                    provenance={
                        "version": str(ledger_packet.get("version") or ""),
                        "ledger_mode": str(ledger_packet.get("ledger_mode") or ""),
                        "tool_call_count": int((ledger_packet.get("summary") or {}).get("tool_call_count") or 0),
                        "tool_result_count": int((ledger_packet.get("summary") or {}).get("tool_result_count") or 0),
                        "evidence_count": int((ledger_packet.get("summary") or {}).get("evidence_count") or 0),
                        "pending_step_count": int((ledger_packet.get("summary") or {}).get("pending_step_count") or 0),
                    },
                    phase_visibility=frozenset({
                        ContextPhase.PLANNING,
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.MEMORY_COMMIT,
                        ContextPhase.VERIFICATION,
                    }),
                )
            )

    items.append(
        ContextItem(
            item_id="current_objective",
            kind=ContextKind.OBJECTIVE,
            title="Current Objective",
            content=(
                (
                    "Resolved working objective:\n"
                    f"{(inputs.effective_objective or request.user_message).strip()}\n\n"
                    "Current user turn:\n"
                    f"{request.user_message.strip()}\n\n"
                    "Use the resolved working objective as the operative intent for planning, tool selection, and response synthesis."
                )
                if (inputs.effective_objective or "").strip()
                and inputs.effective_objective.strip() != request.user_message.strip()
                else request.user_message.strip()
            ),
            source="user_message",
            priority=30,
            max_chars=900,
            trace_id="run_engine:current_objective",
        )
    )

    if inputs.active_skill_context.strip():
        skill_name = inputs.active_skill_name or "unknown"
        items.append(
            ContextItem(
                item_id="active_skill",
                kind=ContextKind.INSTRUCTION,
                title=f"Active Skill: {skill_name}",
                content=inputs.active_skill_context.strip(),
                source="skill_loader",
                priority=25,
                max_chars=1400,
                ttl_turns=1,
                trace_id=f"skill_loader:{skill_name}",
                provenance={
                    "skill_name": skill_name,
                    "chars": len(inputs.active_skill_context.strip()),
                },
                phase_visibility=frozenset({
                    ContextPhase.PLANNING,
                    ContextPhase.ACTING,
                }),
            )
        )

    governed_memory = None
    if context_governor is not None and hasattr(context_governor, "build_memory_surface"):
        try:
            governed_memory = context_governor.build_memory_surface(
                engine=context_engine,
                session=session,
                context=inputs.current_context,
                current_facts=inputs.current_facts,
                trace_prefix="memory:context_engine",
                correction_turn=inputs.correction_turn,
                direct_fact_memory_only=inputs.direct_fact_memory_only,
            )
        except Exception:
            governed_memory = None
    if governed_memory is None:
        candidate_signals = list(getattr(session, "candidate_signals", []) or [])
        candidate_context = str(getattr(session, "candidate_context", "") or "").strip()
        if candidate_signals:
            candidate_context = render_candidate_signals(candidate_signals)
            candidate_source = "structured_candidate_signals"
        else:
            candidate_source = "legacy_candidate_context" if candidate_context else "none"
            if not candidate_context:
                candidate_signals = parse_candidate_context(candidate_context)
        from engine.context_memory_items import build_context_engine_memory_items

        governed_memory = {
            "candidate_context": candidate_context,
            "historical_context": _build_historical_context(inputs),
            "memory_items": build_context_engine_memory_items(
                context_engine,
                inputs.current_context,
                context_governor=context_governor,
                current_facts=inputs.current_facts,
                fallback_trace_id="memory:context_engine",
                fallback_source="context_engine",
                fallback_provenance={"engine": context_engine.__class__.__name__} if context_engine is not None else None,
            ),
            "provenance": {
                "mode": "fallback",
                "candidate_source": candidate_source,
                "candidate_count": len(candidate_signals),
                "historical_segments": len([
                    seg for seg in _build_historical_context(inputs).split("\n\n---\n") if seg.strip()
                ]),
                "memory_item_count": len(
                    build_context_engine_memory_items(
                        context_engine,
                        inputs.current_context,
                        context_governor=context_governor,
                        current_facts=inputs.current_facts,
                        fallback_trace_id="memory:context_engine",
                        fallback_source="context_engine",
                        fallback_provenance={"engine": context_engine.__class__.__name__} if context_engine is not None else None,
                    )
                ),
                "direct_fact_memory_only": bool(inputs.direct_fact_memory_only),
                "correction_turn": bool(inputs.correction_turn),
            },
        }
    candidate_context = (
        str((getattr(governed_memory, "candidate_context", None) if not isinstance(governed_memory, dict) else governed_memory.get("candidate_context")) or "").strip()
    )
    evidence_objective = inputs.effective_objective or inputs.request.user_message
    evidence_snapshot = build_working_evidence_snapshot(
        inputs.tool_results,
        user_message=evidence_objective,
        max_results=3,
    )
    meta_snapshot = build_meta_context_snapshot(
        objective=evidence_objective,
        allowed_tools=inputs.allowed_tools,
        current_facts=inputs.current_facts,
        candidate_context=candidate_context,
        evidence_snapshot=evidence_snapshot,
        conversation_tension=inputs.conversation_tension,
        observation_status=inputs.observation_status,
    )
    meta_context = build_meta_context_block(meta_snapshot)
    if meta_context:
        items.append(
            ContextItem(
                item_id="meta_context",
                kind=ContextKind.META,
                title="Meta Context",
                content=meta_context,
                source="run_engine",
                priority=31,
                max_chars=1200,
                trace_id="run_engine:meta_context",
                provenance={
                    "known_fact_count": meta_snapshot.known_fact_count,
                    "candidate_count": meta_snapshot.candidate_count,
                    "evidence_count": meta_snapshot.evidence_count,
                    "exact_match_count": meta_snapshot.exact_match_count,
                    "substantive_evidence_count": meta_snapshot.substantive_evidence_count,
                    "memory_mode": meta_snapshot.memory_mode,
                    "tool_economy": meta_snapshot.tool_economy,
                    "contradiction_policy": meta_snapshot.contradiction_policy,
                },
                phase_visibility=frozenset({
                    ContextPhase.PLANNING,
                    ContextPhase.ACTING,
                    ContextPhase.RESPONSE,
                    ContextPhase.VERIFICATION,
                }),
            )
        )

    risk_note = (inputs.execution_risk_tier or "").strip()
    items.append(
        build_loop_contract_item(
            source="run_engine",
            trace_id="run_engine:loop_contract",
            extra_note=risk_note,
            provenance={"phase": inputs.phase.value},
        )
    )

    session_anchor_item = build_session_anchor_item(
        first_message=str(getattr(session, "first_message", "") or ""),
        message_count=int(getattr(session, "message_count", 0) or 0),
        source="session_manager",
        trace_id="run_engine:session_anchor",
    )
    if session_anchor_item is not None:
        items.append(session_anchor_item)

    items.append(
        build_memory_authority_item(
            correction_turn=inputs.correction_turn,
            direct_fact_memory_only=inputs.direct_fact_memory_only,
            source="run_engine",
            trace_id="run_engine:memory_authority",
            provenance={"phase": inputs.phase.value},
        )
    )

    deterministic_facts_item = build_deterministic_facts_item(
        facts=inputs.current_facts or [],
        source="semantic_graph",
        trace_id="run_engine:deterministic_facts",
    )
    if deterministic_facts_item is not None:
        items.append(deterministic_facts_item)

    # ── L0 identity brief ────────────────────────────────────────────────────
    # Owner-scoped current facts pulled across every session. Provides cross-
    # session continuity on a fresh chat where session-scoped current_facts is
    # empty. Filtered to facts not already in this turn's deterministic set so
    # we don't duplicate.
    graph = getattr(context_governor, "graph", None)
    owner_id_for_brief = getattr(inputs.request, "owner_id", None)
    if graph is not None and owner_id_for_brief and hasattr(graph, "get_owner_current_facts"):
        try:
            owner_facts = graph.get_owner_current_facts(owner_id_for_brief, limit=40) or []
        except Exception:
            owner_facts = []
        session_keys = {
            ((s or "").strip().lower(), (p or "").strip().lower())
            for s, p, _ in (inputs.current_facts or [])
        }
        identity_lines: list[str] = []
        for subject, predicate, object_ in owner_facts:
            key = ((subject or "").strip().lower(), (predicate or "").strip().lower())
            if not key[0] or not key[1] or key in session_keys:
                continue
            identity_lines.append(f"- {subject} — {predicate}: {object_}")
        if identity_lines:
            items.append(
                ContextItem(
                    item_id="identity_brief",
                    kind=ContextKind.MEMORY,
                    title="Owner Identity Brief",
                    content=(
                        "Cross-session facts known about the owner (do not contradict without explicit correction):\n"
                        + "\n".join(identity_lines)
                    ),
                    source="semantic_graph",
                    priority=34,
                    max_chars=1400,
                    trace_id="run_engine:identity_brief",
                    provenance={
                        "fact_count": len(identity_lines),
                        "owner_id": owner_id_for_brief,
                    },
                    # Cross-session identity is only useful for planning the
                    # turn and shaping the final response. Acting and verify
                    # work off the deterministic + working evidence lanes;
                    # paying ~1.4KB on every tool turn would be wasted.
                    phase_visibility=frozenset({
                        ContextPhase.PLANNING,
                        ContextPhase.RESPONSE,
                    }),
                )
            )

    candidate_context_item = build_candidate_context_item(
        candidate_context=candidate_context,
        source=(
            str(
                (
                    getattr(governed_memory, "provenance", None)
                    if not isinstance(governed_memory, dict)
                    else governed_memory.get("provenance", {})
                ).get("candidate_source")
                or "session_manager"
            )
        ),
        trace_id="run_engine:candidate_context",
    )
    if candidate_context_item is not None:
        items.append(candidate_context_item)

    if inputs.allowed_tools:
        tool_lines = [
            f"- {tool.get('name')}: {tool.get('description', '')}"
            for tool in inputs.allowed_tools
            if isinstance(tool, dict) and tool.get("name")
        ]
        available_tools_item = build_available_tools_item(
            content="\n".join(tool_lines),
            source="tool_registry",
            trace_id="run_engine:available_tools",
            provenance={"tool_count": len(tool_lines)},
        )
        if available_tools_item is not None:
            items.append(available_tools_item)

    if inputs.capability_context.strip():
        items.append(
            ContextItem(
                item_id="capability_cards",
                kind=ContextKind.RUNTIME,
                title="Capability Cards",
                content=inputs.capability_context.strip(),
                source="capability_registry",
                priority=24,
                max_chars=1800,
                trace_id="run_engine:capability_cards",
                phase_visibility=frozenset({
                    ContextPhase.PLANNING,
                    ContextPhase.ACTING,
                    ContextPhase.RESPONSE,
                    ContextPhase.VERIFICATION,
                }),
                provenance={"chars": len(inputs.capability_context.strip())},
            )
        )

    if inputs.strategy.strip() or inputs.notes.strip():
        guidance_parts = []
        if inputs.strategy.strip():
            guidance_parts.append(f"Strategy: {inputs.strategy.strip()}")
        if inputs.notes.strip():
            guidance_parts.append(f"Notes: {inputs.notes.strip()}")
        items.append(
            ContextItem(
                item_id="phase_guidance",
                kind=ContextKind.WORKING,
                title="Phase Guidance",
                content="\n".join(guidance_parts),
                source="orchestrator",
                priority=40,
                max_chars=700,
                trace_id="run_engine:phase_guidance",
                phase_visibility=frozenset({
                    ContextPhase.ACTING,
                    ContextPhase.RESPONSE,
                    ContextPhase.VERIFICATION,
                    ContextPhase.MEMORY_COMMIT,
                }),
                )
            )

    continuation_state = getattr(session, "continuation_state", {}) or {}
    if isinstance(continuation_state, dict) and continuation_state:
        continuation_content = _build_continuation_state(continuation_state)
        if continuation_content:
            items.append(
                ContextItem(
                    item_id="continuation_state",
                    kind=ContextKind.WORKING,
                    title="Continuation State",
                    content=continuation_content,
                    source="session_manager",
                    priority=33,
                    max_chars=1400,
                    trace_id="run_engine:continuation_state",
                    provenance={
                        "reason": str(continuation_state.get("reason") or ""),
                        "pending_step_count": len(continuation_state.get("pending_steps") or []),
                        "missing_slot_count": len(continuation_state.get("missing_slots") or []),
                        "source_run_id": str(continuation_state.get("run_id") or ""),
                    },
                    phase_visibility=frozenset({
                        ContextPhase.PLANNING,
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                    }),
                )
            )

    code_intent_note = (inputs.code_intent_note or "").strip()
    if code_intent_note:
        guidance_items = [item for item in items if item.item_id == "phase_guidance"]
        if guidance_items:
            original = guidance_items[0]
            items.remove(original)
            items.append(
                ContextItem(
                    item_id="phase_guidance",
                    kind=original.kind,
                    title=original.title,
                    content=original.content + "\nCode Intent: " + code_intent_note,
                    source=original.source,
                    priority=original.priority,
                    max_chars=original.max_chars + 300,
                    trace_id=original.trace_id,
                    phase_visibility=original.phase_visibility,
                )
            )
        else:
            items.append(
                ContextItem(
                    item_id="phase_guidance",
                    kind=ContextKind.WORKING,
                    title="Phase Guidance",
                    content="Code Intent: " + code_intent_note,
                    source="code_intent",
                    priority=40,
                    max_chars=500,
                    trace_id="run_engine:phase_guidance",
                    phase_visibility=frozenset({
                        ContextPhase.PLANNING,
                        ContextPhase.ACTING,
                    }),
                )
            )

    # ── Memory Palace loci context (PLANNING + ACTING only) ──────────────────
    if inputs.available_loci:
        loci_lines = []
        for locus in inputs.available_loci:
            lid = str(locus.get("id", "")).strip()
            lname = str(locus.get("name", "")).strip()
            ldesc = str(locus.get("description", "")).strip()
            if not lid:
                continue
            line = f"- {lid}: {lname}"
            if ldesc:
                line += f" — {ldesc}"
            if inputs.planned_locus_id and lid == inputs.planned_locus_id:
                line += " [TARGETED]"
            loci_lines.append(line)
        if loci_lines:
            loci_content = "Named loci in Memory Palace:\n" + "\n".join(loci_lines)
            if inputs.planned_locus_id:
                loci_content += f"\n\nActive locus: {inputs.planned_locus_id} — use shovs_memory_query with locus_id=\"{inputs.planned_locus_id}\" to query it."
            items.append(
                ContextItem(
                    item_id="memory_palace_loci",
                    kind=ContextKind.MEMORY,
                    title="Memory Palace",
                    content=loci_content,
                    source="semantic_graph",
                    priority=38,
                    max_chars=600,
                    trace_id="run_engine:memory_palace_loci",
                    provenance={
                        "loci_count": len(loci_lines),
                        "planned_locus_id": inputs.planned_locus_id or "",
                    },
                    phase_visibility=frozenset({
                        ContextPhase.PLANNING,
                        ContextPhase.ACTING,
                    }),
                )
            )

    # ── Spatial drawers (PLANNING only) ─────────────────────────────────────
    # Pre-fetched locus drawers for the targeted locus + 1-hop neighbors.
    # Lets the planner reason about workspace context before tool selection.
    # Planning-only on purpose: the actor uses tools (shovs_memory_query) to
    # query specific drawers rather than getting raw drawer dumps in the
    # acting context.
    if inputs.spatial_drawers:
        drawer_lines: list[str] = []
        primary_id = ""
        for entry in inputs.spatial_drawers[:4]:
            lid = str(entry.get("locus_id", "")).strip()
            hop = int(entry.get("hop", 0))
            content = str(entry.get("content", "")).strip()
            if not lid or not content:
                continue
            if hop == 0:
                primary_id = lid
            label = f"## Locus: {lid}" if hop == 0 else f"## Neighbor (hop {hop}): {lid}"
            # Cap each drawer to keep total budget bounded.
            snippet = content[:1200]
            if len(content) > 1200:
                snippet = snippet.rstrip() + "\n…(truncated; use shovs_memory_query for full)"
            drawer_lines.append(f"{label}\n{snippet}")
        if drawer_lines:
            spatial_content = (
                "Pre-fetched workspace drawers for this query. The targeted locus and its "
                "neighbors. Use shovs_memory_query with locus_id when you need specific facts.\n\n"
                + "\n\n".join(drawer_lines)
            )
            items.append(
                ContextItem(
                    item_id="spatial_drawers",
                    kind=ContextKind.MEMORY,
                    title="Spatial Drawers (planning hint)",
                    content=spatial_content,
                    source="semantic_graph",
                    priority=37,
                    max_chars=4800,
                    trace_id="run_engine:spatial_drawers",
                    provenance={
                        "primary_locus": primary_id,
                        "drawer_count": len(drawer_lines),
                    },
                    phase_visibility=frozenset({ContextPhase.PLANNING}),
                )
            )

    # ── Plan steps lane (ACTING + RESPONSE) ─────────────────────────────────
    # Per-turn plan_steps from the planner. The runtime updates statuses
    # in-place after each tool turn. Visible to ACTING so the actor sees
    # remaining work, and RESPONSE so the response phase can flag partial
    # completion honestly when steps remain pending.
    if inputs.plan_steps:
        plan_lines = []
        for step in inputs.plan_steps:
            sid = str(step.get("id", "")).strip()
            sdesc = str(step.get("description", "")).strip()
            sstatus = str(step.get("status", "pending")).strip()
            stool = str(step.get("tool") or "").strip()
            tool_part = f" [{stool}]" if stool else ""
            plan_lines.append(f"- {sid} ({sstatus}){tool_part}: {sdesc}")
        if plan_lines:
            items.append(
                ContextItem(
                    item_id="plan_steps",
                    kind=ContextKind.WORKING,
                    title="Plan Steps",
                    content="Per-turn plan (status reflects current run progress):\n" + "\n".join(plan_lines),
                    source="planner",
                    priority=39,
                    max_chars=900,
                    trace_id="run_engine:plan_steps",
                    provenance={
                        "step_count": len(plan_lines),
                        "pending": sum(1 for s in inputs.plan_steps if s.get("status") == "pending"),
                        "done": sum(1 for s in inputs.plan_steps if s.get("status") == "done"),
                        "failed": sum(1 for s in inputs.plan_steps if s.get("status") == "failed"),
                    },
                    phase_visibility=frozenset({
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                    }),
                )
            )

    # ── Active task list injection (ACTING + RESPONSE only) ──────────────────
    # When the agent wrote a todo_write plan, surface the current task state
    # in the acting context so the actor knows to call todo_update as tasks
    # start and complete — without needing to call todo_read first.
    _session_id_for_tasks = str(getattr(inputs.session, "id", "") or "").strip()
    if _session_id_for_tasks:
        try:
            _tracker = get_session_task_tracker()
            if _tracker.has_active_tasks(_session_id_for_tasks):
                _task_state = _tracker.render(_session_id_for_tasks)
                items.append(
                    ContextItem(
                        item_id="active_task_list",
                        kind=ContextKind.WORKING,
                        title="Active Task List",
                        content=(
                            _task_state
                            + "\n\nCall todo_update(task_id, 'in_progress') before starting a task "
                            "and todo_update(task_id, 'completed') immediately after it finishes."
                        ),
                        source="task_tracker",
                        priority=41,
                        max_chars=700,
                        trace_id="run_engine:active_task_list",
                        provenance={"session_id": _session_id_for_tasks},
                        phase_visibility=frozenset({
                            ContextPhase.ACTING,
                            ContextPhase.RESPONSE,
                        }),
                    )
                )
        except Exception:
            pass  # Never block packet build for task state

    tension_content = render_conversation_tension(inputs.conversation_tension or ConversationTension())
    tension_item = build_conversation_tension_item(
        content=tension_content,
        source="run_engine",
        trace_id="run_engine:conversation_tension",
        provenance=conversation_tension_audit_payload(inputs.conversation_tension),
    )
    if tension_item is not None:
        items.append(tension_item)

    observation_state = _build_observation_state(inputs)
    if observation_state:
        items.append(
            ContextItem(
                item_id="observation_state",
                kind=ContextKind.WORKING,
                title="Observation State",
                content=observation_state,
                source="run_engine",
                priority=43,
                max_chars=1000,
                trace_id="run_engine:observation_state",
                provenance={
                    "tool_turn": inputs.tool_turn,
                    "tool_result_count": len(inputs.tool_results or []),
                },
                phase_visibility=frozenset({
                    ContextPhase.ACTING,
                    ContextPhase.RESPONSE,
                    ContextPhase.MEMORY_COMMIT,
                    ContextPhase.VERIFICATION,
                }),
            )
        )

    working_evidence = build_working_evidence_block(
        inputs.tool_results,
        user_message=evidence_objective,
        max_results=3,
        preview_chars=180,
    )
    working_evidence_item = build_working_evidence_item(
        content=working_evidence,
        source="run_engine",
        trace_id="run_engine:working_evidence",
        provenance={
            "tool_result_count": len(inputs.tool_results or []),
            "selected_count": len(evidence_snapshot.selected),
            "substantive_count": evidence_snapshot.substantive_count,
            "exact_match_count": evidence_snapshot.exact_match_count,
            "objective": evidence_objective.strip(),
        },
    )
    if working_evidence_item is not None:
        items.append(working_evidence_item)

    working_state = _build_working_state(inputs)
    if working_state:
        items.append(
            ContextItem(
                item_id="working_state",
                kind=ContextKind.WORKING,
                title="Working State",
                content=working_state,
                source="run_engine",
                priority=45,
                max_chars=2200,
                trace_id="run_engine:working_state",
            )
        )

    historical_context = (
        str((getattr(governed_memory, "historical_context", None) if not isinstance(governed_memory, dict) else governed_memory.get("historical_context")) or "")
    )
    historical_context_item = build_historical_context_item(
        content=historical_context,
        source="context_governor" if governed_memory is not None else "session_history",
        trace_id="run_engine:historical_context",
        provenance=(
            dict(
                (
                    getattr(governed_memory, "provenance", None)
                    if not isinstance(governed_memory, dict)
                    else governed_memory.get("provenance", {})
                )
                or {}
            )
        ),
    )
    if historical_context_item is not None:
        items.append(historical_context_item)

    items.extend(
        list(
            (
                getattr(governed_memory, "memory_items", None)
                if not isinstance(governed_memory, dict)
                else governed_memory.get("memory_items", [])
            )
            or []
        )
    )

    compiled = compile_context_items(
        items,
        phase=inputs.phase,
        char_budget=DEFAULT_PHASE_CHAR_BUDGETS[inputs.phase],
        truncate_section=_truncate_section,
    )
    history = list(getattr(session, "sliding_window", []) or [])[-4:]
    retained_history = 0
    truncated_history = 0
    max_chars_per_window_msg = 180
    for entry in history:
        role = str(entry.get("role") or "")
        content = str(entry.get("content") or "")
        if role in {"user", "assistant"} and content.strip():
            retained_history += 1
            if len(content.strip()) > max_chars_per_window_msg:
                truncated_history += 1
    trace = compiled.to_trace_payload()
    trace["content"] = compiled.content
    trace["runtime_path"] = "run_engine"
    trace["trace_scope"] = "phase_packet"
    trace["canonical_event"] = "phase_context"
    trace["packet_contract_version"] = "phase-packet-v1"
    trace["model"] = request.model
    trace["history"] = {
        "retained_count": retained_history,
        "truncated_count": truncated_history,
        "input_count": len(getattr(session, "sliding_window", []) or []),
        "max_chars_per_window_msg": max_chars_per_window_msg,
    }
    trace["governed_memory"] = dict(
        (
            getattr(governed_memory, "provenance", None)
            if not isinstance(governed_memory, dict)
            else governed_memory.get("provenance", {})
        )
        or {}
    )
    return CompiledPassPacket(
        phase=inputs.phase,
        content=compiled.content,
        trace=trace,
    )


def _build_working_state(inputs: PacketBuildInputs) -> str:
    blocks: list[str] = []

    history = list(getattr(inputs.session, "sliding_window", []) or [])[-4:]
    history_lines = []
    for entry in history:
        role = str(entry.get("role") or "")
        content = str(entry.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            history_lines.append(f"- {role}: {_clip(content, 180)}")
    if history_lines:
        blocks.append("Recent Transcript:\n" + "\n".join(history_lines))

    if inputs.final_response.strip():
        blocks.append("Draft Response:\n" + _clip(inputs.final_response.strip(), 700))

    return "\n\n".join(blocks)


def _build_continuation_state(state: dict[str, Any]) -> str:
    objective = str(state.get("objective") or "").strip()
    reason = str(state.get("reason") or "").strip()
    next_action = str(state.get("next_action") or "").strip()
    strategy = str(state.get("strategy") or "").strip()
    notes = str(state.get("notes") or "").strip()
    pending_steps = [item for item in (state.get("pending_steps") or []) if isinstance(item, dict)]
    missing_slots = [str(item).strip() for item in (state.get("missing_slots") or []) if str(item).strip()]
    evidence = [str(item).strip() for item in (state.get("evidence_summary") or []) if str(item).strip()]
    issues = [str(item).strip() for item in (state.get("issues") or []) if str(item).strip()]

    lines = [
        "Prior run did not fully close. Treat this as the durable handoff for resuming the workflow.",
    ]
    if reason:
        lines.append(f"Reason: {reason}")
    if objective:
        lines.append(f"Unfinished objective: {objective}")
    if next_action:
        lines.append(f"Next required action: {next_action}")
    if strategy:
        lines.append(f"Prior strategy: {strategy}")
    if notes:
        lines.append(f"Prior notes: {notes}")
    if missing_slots:
        lines.append("Missing data:")
        lines.extend(f"- {item}" for item in missing_slots[:6])
    if pending_steps:
        lines.append("Pending steps:")
        for step in pending_steps[:6]:
            sid = str(step.get("id") or "").strip()
            desc = str(step.get("description") or "").strip()
            tool = str(step.get("tool") or "").strip()
            label = f"- {sid}: " if sid else "- "
            if tool:
                label += f"[{tool}] "
            label += desc
            lines.append(label.rstrip())
    if evidence:
        lines.append("Evidence already gathered:")
        lines.extend(f"- {item}" for item in evidence[:4])
    if issues:
        lines.append("Prior verification issues:")
        lines.extend(f"- {item}" for item in issues[:4])
    lines.append(
        "Resume from the smallest missing step. Do not repeat completed tools unless the prior evidence is stale or contradictory."
    )
    return "\n".join(line for line in lines if line.strip())


def _build_observation_state(inputs: PacketBuildInputs) -> str:
    lines: list[str] = []
    observation_status = str(inputs.observation_status or "").strip()
    observation_tools = [str(item).strip() for item in (inputs.observation_tools or []) if str(item).strip()]

    if observation_status:
        lines.append(f"Manager status: {observation_status}")
    if inputs.strategy.strip():
        lines.append(f"Strategy: {inputs.strategy.strip()}")
    if observation_tools:
        lines.append("Preferred next tools: " + ", ".join(observation_tools))
    if inputs.notes.strip():
        lines.append(f"Notes: {inputs.notes.strip()}")
    if inputs.tool_turn:
        lines.append(f"Tool turn: {inputs.tool_turn}")

    if inputs.tool_results:
        lines.append("Recent tool observations:")
        for item in inputs.tool_results[-4:]:
            lines.append(
                format_tool_result_line(
                    item,
                    preview_chars=180,
                    include_status_label=True,
                )
            )

    return "\n".join(line for line in lines if line.strip())


def _build_historical_context(inputs: PacketBuildInputs) -> str:
    if inputs.direct_fact_memory_only:
        return ""

    history = list(getattr(inputs.session, "full_history", []) or [])
    recent = list(getattr(inputs.session, "sliding_window", []) or [])
    if len(history) <= len(recent):
        return ""

    older_history = history[:-len(recent)] if recent else history[:-4]
    segments: list[str] = []
    for entry in older_history[-4:]:
        role = str(entry.get("role") or "")
        content = str(entry.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            segments.append(f"{role.upper()}: {_clip(content, 800)}")

    return "\n\n---\n".join(segments)


def _truncate_section(content: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(content) <= budget:
        return content
    suffix = "\n[...truncated...]"
    if budget <= len(suffix):
        return content[:budget]
    return content[: budget - len(suffix)].rstrip() + suffix


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
