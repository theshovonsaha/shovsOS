from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from engine.candidate_signals import parse_candidate_context
from engine.conversation_tension import ConversationTension
from engine.direct_fact_policy import should_answer_direct_fact_from_memory
from run_engine.evidence_lane import WorkingEvidenceSnapshot


@dataclass(frozen=True)
class MetaContextSnapshot:
    objective: str
    known_fact_count: int
    candidate_count: int
    evidence_count: int
    exact_match_count: int
    substantive_evidence_count: int
    verification_posture: str
    falsifier: str
    minimum_probe: str
    plan_rule: str
    contradiction_policy: str
    tool_economy: str
    memory_mode: str


def build_meta_context_snapshot(
    *,
    objective: str,
    allowed_tools: list[dict[str, Any]],
    current_facts: Optional[list[tuple[str, str, str]]] = None,
    candidate_context: str = "",
    evidence_snapshot: Optional[WorkingEvidenceSnapshot] = None,
    conversation_tension: Optional[ConversationTension] = None,
    observation_status: str = "",
) -> MetaContextSnapshot:
    normalized_objective = str(objective or "").strip()
    candidate_count = len(parse_candidate_context(candidate_context or ""))
    known_fact_count = len(list(current_facts or []))

    snapshot = evidence_snapshot or WorkingEvidenceSnapshot(
        objective=normalized_objective,
        exact_targets=tuple(),
        selected=tuple(),
    )
    allowed_tool_names = {
        str(tool.get("name") or "").strip()
        for tool in allowed_tools
        if isinstance(tool, dict)
    }
    allowed_tool_names.discard("")
    direct_fact_answerable = should_answer_direct_fact_from_memory(
        normalized_objective,
        list(current_facts or []),
    )

    if direct_fact_answerable:
        verification_posture = (
            "Current deterministic facts already answer the objective. Prefer them over retrieval or tools."
        )
    elif snapshot.exact_match_count > 0:
        verification_posture = (
            "Exact-target evidence exists. Treat it as primary and reject strong claims not supported there."
        )
    elif snapshot.substantive_count > 0:
        verification_posture = (
            "Evidence exists but remains indirect or partial. Verify strong claims before finalizing."
        )
    elif candidate_count > 0:
        verification_posture = (
            "Only candidate signals are present. Use them as hints, not as truth."
        )
    else:
        verification_posture = (
            "No substantive evidence is present yet. Gather one direct probe before making strong claims."
        )

    if conversation_tension and conversation_tension.should_challenge:
        falsifier = (
            "A direct user correction or contradiction against deterministic facts falsifies the current assumption set."
        )
    elif snapshot.exact_targets and "web_fetch" in allowed_tool_names and snapshot.exact_match_count == 0:
        falsifier = (
            f"An exact-target fetch for {snapshot.exact_targets[0]} that contradicts the current assumption should change the path."
        )
    elif candidate_count > 0:
        falsifier = (
            "Any direct source or explicit user correction that contradicts the candidate signals falsifies them."
        )
    else:
        falsifier = (
            "Any direct source result that conflicts with the current answer should override the weaker path."
        )

    if direct_fact_answerable:
        minimum_probe = "No probe is required. Answer from deterministic facts and keep superseded memory out."
    elif snapshot.exact_match_count > 0:
        minimum_probe = "No new probe is required unless verification fails or a contradiction appears."
    elif snapshot.exact_targets and "web_fetch" in allowed_tool_names:
        minimum_probe = f"Fetch the exact target directly: {snapshot.exact_targets[0]}."
    elif "web_search" in allowed_tool_names:
        clipped_objective = normalized_objective if len(normalized_objective) <= 120 else normalized_objective[:117].rstrip() + "..."
        minimum_probe = f"Run one targeted search for: {clipped_objective}"
    elif "query_memory" in allowed_tool_names:
        minimum_probe = "Query durable memory only if the objective depends on prior user facts or earlier work."
    else:
        minimum_probe = "Collect one direct observation before broadening the plan."

    if direct_fact_answerable:
        plan_rule = (
            "Do not broaden the plan. Use the existing deterministic facts and avoid unnecessary tool calls."
        )
    elif (
        snapshot.exact_match_count > 0
        or str(observation_status or "").strip().lower() in {"finalize", "complete", "done"}
    ):
        plan_rule = (
            "Current observations mostly change the act, not the plan. Synthesize from verified evidence."
        )
    else:
        plan_rule = (
            "Change the plan only if new evidence overturns the resolved objective, deterministic facts, or verification posture."
        )

    if conversation_tension and conversation_tension.conflicting_facts:
        contradiction_policy = (
            "Keep contradiction explicit. Prefer the latest user correction, retire superseded facts, and never present both as current."
        )
    elif conversation_tension and conversation_tension.stance_drifts:
        contradiction_policy = (
            "Preserve stance drift explicitly when it matters; do not flatten older and newer positions into one vague summary."
        )
    else:
        contradiction_policy = (
            "If no material contradiction exists, keep context compact and avoid speculative reconciliation."
        )

    if direct_fact_answerable:
        tool_economy = "Zero-tool turn preferred. Existing deterministic facts are sufficient."
    elif snapshot.exact_match_count > 0:
        tool_economy = "Enough evidence exists. Synthesize instead of adding more tools."
    elif snapshot.exact_targets and "web_fetch" in allowed_tool_names:
        tool_economy = "One exact fetch is better than multi-step broadening."
    elif snapshot.substantive_count > 0:
        tool_economy = "Use at most one more concrete tool if it closes a named evidence gap."
    else:
        tool_economy = "Take the smallest direct evidence step that can falsify or confirm the current path."

    if direct_fact_answerable:
        memory_mode = "deterministic_only"
    elif candidate_count > 0 and known_fact_count == 0:
        memory_mode = "candidate_cautious"
    elif known_fact_count > 0:
        memory_mode = "deterministic_plus_evidence"
    else:
        memory_mode = "evidence_first"

    return MetaContextSnapshot(
        objective=normalized_objective,
        known_fact_count=known_fact_count,
        candidate_count=candidate_count,
        evidence_count=len(snapshot.selected),
        exact_match_count=snapshot.exact_match_count,
        substantive_evidence_count=snapshot.substantive_count,
        verification_posture=verification_posture,
        falsifier=falsifier,
        minimum_probe=minimum_probe,
        plan_rule=plan_rule,
        contradiction_policy=contradiction_policy,
        tool_economy=tool_economy,
        memory_mode=memory_mode,
    )


def build_meta_context_block(snapshot: MetaContextSnapshot) -> str:
    if not snapshot.objective:
        return ""

    lines = [
        "Epistemic Posture:",
        f"- Known truth: {snapshot.known_fact_count} deterministic fact(s)",
        f"- Candidate signals: {snapshot.candidate_count} pending",
        (
            f"- Working evidence: {snapshot.evidence_count} selected "
            f"({snapshot.exact_match_count} exact-target, {snapshot.substantive_evidence_count} substantive)"
        ),
        "",
        "Verification Posture:",
        f"- {snapshot.verification_posture}",
        "",
        "Falsifier:",
        f"- {snapshot.falsifier}",
        "",
        "Minimum Next Probe:",
        f"- {snapshot.minimum_probe}",
        "",
        "Plan Discipline:",
        f"- {snapshot.plan_rule}",
        "",
        "Contradiction Policy:",
        f"- {snapshot.contradiction_policy}",
        "",
        "Tool Economy:",
        f"- {snapshot.tool_economy}",
        "",
        "Memory Mode:",
        f"- {snapshot.memory_mode}",
    ]
    return "\n".join(lines)
