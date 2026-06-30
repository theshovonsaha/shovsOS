from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


TURN_RELATION_VERSION = "turn-relation-v1"


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "for", "from", "get", "give", "have", "how", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "please", "that", "the", "this", "to",
    "use", "we", "what", "with", "you", "your",
}

_RESUME_RE = re.compile(
    r"\b(continue|resume|keep going|finish|carry on|pick up|complete (?:it|that|the plan)|next step)\b",
    re.IGNORECASE,
)
_ADDITIVE_RE = re.compile(r"\b(also|and also|include|add|what about|same thing|another|more)\b", re.IGNORECASE)
_SHORT_FOLLOWUP_RE = re.compile(
    r"^\s*(?:"
    r"what\s+about\s+.+|"
    r"how\s+about\s+.+|"
    r"same\s+(?:for|thing)\s+.+|"
    r"also\s+.+|"
    r"and\s+.+|"
    r"that\s+one|this\s+one|"
    r"compare\s+.+|"
    r"check\s+.+"
    r")\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_BARE_FOLLOWUP_BLOCK_RE = re.compile(
    r"^\s*(?:who|what|where|when|why|how|can|could|should|would|is|are|do|does|did)\b|"
    r"\b(search|find|fetch|lookup|look\s+up|research|investigate|compare|write|create|build|make|"
    r"generate|explain|summari[sz]e|recommend|suggest|buy|order|book|open|run|test|fix|implement)\b",
    re.IGNORECASE,
)
_CORRECTION_RE = re.compile(
    r"\b(actually|correction|correct|instead|not\b|no,|wrong|update|changed?|switched|rather than|i meant)\b",
    re.IGNORECASE,
)
_META_RE = re.compile(
    r"\b(be more|be less|concise|verbose|tone|style|format|do not use|don't use|always use|"
    r"when planning|use web|no web|frontend_shovs|ledger|memory|context|prompt|agent should)\b",
    re.IGNORECASE,
)
_NEW_TOPIC_RE = re.compile(
    r"\b(new topic|separate topic|different topic|forget that|leave that|switch topic|unrelated)\b",
    re.IGNORECASE,
)
_REFINE_RE = re.compile(
    r"\b(make it|change it|adjust|polish|tighten|simplify|expand|reduce|increase|replace|keep .* but)\b",
    re.IGNORECASE,
)
_RETURN_RE = re.compile(r"\b(return to|go back to|back to|revisit|resume the old|pick up the old)\b", re.IGNORECASE)


@dataclass(frozen=True)
class TurnRelation:
    relation: str
    confidence: float
    reason: str
    overlap: float
    anchors: list[str] = field(default_factory=list)
    required_context: list[str] = field(default_factory=list)
    blocked_context: list[str] = field(default_factory=list)
    memory_write_policy: str = "default"
    tool_policy: str = "planner_may_choose"
    proof_obligations: list[str] = field(default_factory=list)
    resolved_objective: str = ""
    carried_context: list[str] = field(default_factory=list)
    dropped_context: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["version"] = TURN_RELATION_VERSION
        return payload

    def render(self) -> str:
        lines = [
            "Turn Relation:",
            f"- relation: {self.relation}",
            f"- confidence: {self.confidence:.2f}",
            f"- reason: {self.reason}",
            f"- overlap: {self.overlap:.2f}",
        ]
        if self.anchors:
            lines.append("- anchors: " + ", ".join(self.anchors))
        if self.required_context:
            lines.append("- required context: " + ", ".join(self.required_context))
        if self.blocked_context:
            lines.append("- blocked context: " + ", ".join(self.blocked_context))
        lines.append(f"- memory policy: {self.memory_write_policy}")
        lines.append(f"- tool policy: {self.tool_policy}")
        if self.proof_obligations:
            lines.append("Proof obligations:")
            for item in self.proof_obligations:
                lines.append(f"- {item}")
        if self.resolved_objective:
            lines.append("Resolved objective:")
            lines.append(self.resolved_objective)
        if self.carried_context:
            lines.append("- carried context: " + ", ".join(self.carried_context))
        if self.dropped_context:
            lines.append("- dropped context: " + ", ".join(self.dropped_context))
        return "\n".join(lines)


def classify_turn_relation(
    *,
    user_message: str,
    continuation_state: dict[str, Any] | None = None,
    recent_turns: list[dict[str, Any]] | None = None,
    distant_memory_signals: list[str] | None = None,
    ambiguous_followup: bool = False,
) -> TurnRelation:
    """Classify how this turn is allowed to relate to prior context.

    This is deterministic by design. It does not infer truth; it gates which
    prior state may enter the next pass and what the verifier must prove.
    """

    current = str(user_message or "").strip()
    lowered = current.lower()
    continuation = continuation_state if isinstance(continuation_state, dict) else {}
    previous_objective = str(continuation.get("objective") or "").strip()
    overlap = _overlap(current, previous_objective)
    has_continuation = bool(previous_objective)
    recent_text = _recent_text(recent_turns or [])
    recent_overlap = _overlap(current, recent_text)
    memory_overlap = max((_overlap(current, signal) for signal in (distant_memory_signals or [])), default=0.0)
    recent_frame = _latest_user_frame(current, recent_turns or [])
    short_followup = _is_short_followup(current, recent_frame=recent_frame)

    if _NEW_TOPIC_RE.search(lowered):
        return TurnRelation(
            relation="fresh_topic",
            confidence=0.95,
            reason="explicit new-topic language",
            overlap=overlap,
            blocked_context=["previous_plan", "locked_entities", "continuation_state", "stale_memory"],
            memory_write_policy="normal_after_response",
            tool_policy="clear_continuation_then_plan",
            proof_obligations=["actor_context excludes stale workflow state"],
        )

    if _META_RE.search(lowered):
        return TurnRelation(
            relation="meta_instruction",
            confidence=0.88,
            reason="runtime/style instruction rather than domain content",
            overlap=overlap,
            anchors=["runtime_policy"],
            required_context=["current_runtime_settings"],
            blocked_context=["domain_fact_claims"],
            memory_write_policy="store_as_runtime_preference_only_if_stable",
            tool_policy="update_policy_before_tools",
            proof_obligations=["meta instruction is not used as factual answer content"],
        )

    if _CORRECTION_RE.search(lowered):
        relation = "correction" if not _REFINE_RE.search(lowered) else "refinement"
        return TurnRelation(
            relation=relation,
            confidence=0.86,
            reason="explicit correction/refinement signal",
            overlap=max(overlap, recent_overlap),
            anchors=["latest_user_turn", "superseded_fact_or_constraint"],
            required_context=["recent_turns", "current_facts", "contradiction_lane"],
            blocked_context=["older_conflicting_fact_as_current"],
            memory_write_policy="void_or_demote_then_store_with_provenance",
            tool_policy="patch_current_plan_before_new_search",
            proof_obligations=[
                "latest explicit correction dominates older conflicting context",
                "old fact is voided, demoted, or excluded from actor context",
            ],
            resolved_objective=_join_frame(recent_frame, current) if recent_frame else current,
            carried_context=["latest_user_turn", "active_constraints"] if recent_frame else [],
            dropped_context=["older_conflicting_fact_as_current"],
        )

    if recent_frame and short_followup:
        return TurnRelation(
            relation="refinement",
            confidence=0.84,
            reason="short follow-up needs the latest active task frame",
            overlap=max(overlap, recent_overlap),
            anchors=["latest_user_turn", "current_turn_patch"],
            required_context=["latest_user_turn", "active_constraints", "current_turn_update"],
            blocked_context=["older_unrelated_entities", "stale_completed_tool_results"],
            memory_write_policy="commit_only_if_user_preference_or_fact",
            tool_policy="patch_plan_then_continue",
            proof_obligations=[
                "current short follow-up is resolved against the latest active task frame",
                "tool query preserves latest active product/task constraints",
            ],
            resolved_objective=_join_frame(recent_frame, current),
            carried_context=["latest_user_turn", "active_constraints"],
            dropped_context=["older_unrelated_entities", "stale_completed_tool_results"],
        )

    if has_continuation and (ambiguous_followup or _RESUME_RE.search(lowered)):
        return TurnRelation(
            relation="direct_continuation",
            confidence=0.92,
            reason="explicit or ambiguous resume of active continuation",
            overlap=overlap,
            anchors=["continuation_state", "pending_steps"],
            required_context=["continuation_state", "locked_entities", "next_required_action"],
            blocked_context=["unrelated_memory", "old_completed_plans"],
            memory_write_policy="defer_until_task_progress",
            tool_policy="resume_next_required_action",
            proof_obligations=["ledger.next_required_action matches continuation pending step"],
            resolved_objective=previous_objective or current,
            carried_context=["continuation_state", "pending_steps"],
            dropped_context=["unrelated_memory", "old_completed_plans"],
        )

    if has_continuation and _ADDITIVE_RE.search(lowered) and overlap >= 0.12:
        return TurnRelation(
            relation="refinement",
            confidence=0.78,
            reason="additive related turn patches the active objective",
            overlap=overlap,
            anchors=["active_objective", "current_turn_patch"],
            required_context=["continuation_state", "pending_steps", "current_turn_update"],
            blocked_context=["unrelated_memory"],
            memory_write_policy="commit_only_if_user_preference_or_fact",
            tool_policy="patch_plan_then_continue",
            proof_obligations=["surviving plan constraints are compatible with current-turn patch"],
            resolved_objective=_join_frame(previous_objective, current),
            carried_context=["active_objective", "current_turn_patch"],
            dropped_context=["unrelated_memory"],
        )

    if has_continuation and overlap >= 0.45:
        return TurnRelation(
            relation="direct_continuation",
            confidence=0.74,
            reason="high token overlap with active objective",
            overlap=overlap,
            anchors=["active_objective"],
            required_context=["continuation_state", "pending_steps"],
            blocked_context=["unrelated_memory"],
            memory_write_policy="defer_until_task_progress",
            tool_policy="resume_or_patch_plan",
            proof_obligations=["current objective overlap justifies continuation context"],
            resolved_objective=_join_frame(previous_objective, current),
            carried_context=["active_objective", "pending_steps"],
            dropped_context=["unrelated_memory"],
        )

    if has_continuation and overlap > 0 and overlap < 0.45:
        return TurnRelation(
            relation="deviation",
            confidence=0.68,
            reason="some overlap with active objective but task appears changed",
            overlap=overlap,
            anchors=["shared_constraints"],
            required_context=["user_preferences", "hard_constraints"],
            blocked_context=["old_pending_steps", "old_locked_entities"],
            memory_write_policy="preserve_preferences_only",
            tool_policy="replan_with_shared_constraints",
            proof_obligations=["old workflow state is not treated as binding"],
            resolved_objective=current,
            carried_context=["user_preferences", "hard_constraints"],
            dropped_context=["old_pending_steps", "old_locked_entities"],
        )

    if not has_continuation and (recent_overlap >= 0.35 or memory_overlap >= 0.35 or (_RETURN_RE.search(lowered) and memory_overlap > 0)):
        return TurnRelation(
            relation="distant_resumption",
            confidence=0.7,
            reason="matches older turns or durable memory without active continuation",
            overlap=max(recent_overlap, memory_overlap),
            anchors=["semantic_capsules", "raw_refs"],
            required_context=["compact_memory_signal", "relevant_block_refs"],
            blocked_context=["full_raw_history_by_default"],
            memory_write_policy="no_write_until_user_confirms_or_task_completes",
            tool_policy="retrieve_then_validate_relevance",
            proof_obligations=["retrieved old context has raw refs and is still valid"],
        )

    return TurnRelation(
        relation="fresh_topic",
        confidence=0.72,
        reason="no active continuation or strong prior-context relation",
        overlap=overlap,
        required_context=["current_turn"],
        blocked_context=["previous_plan", "old_locked_entities", "stale_candidates"],
        memory_write_policy="normal_after_response",
        tool_policy="plan_from_current_turn",
        proof_obligations=["actor context is anchored to current turn"],
        resolved_objective=current,
    )


def simulate_turn_relation_cases() -> list[dict[str, Any]]:
    cases = [
        {
            "name": "fresh_topic",
            "user_message": "Write a Python script to rename files.",
            "continuation_state": {"objective": "Find a storage bin under $20 near Toronto."},
        },
        {
            "name": "direct_continuation",
            "user_message": "continue",
            "continuation_state": {"objective": "Fetch 9 source URLs for ROKU, TBN, SENEA.", "pending_steps": [{"id": "fetch"}]},
            "ambiguous_followup": True,
        },
        {
            "name": "distant_resumption",
            "user_message": "Let's go back to the sushi Toronto workflow.",
            "distant_memory_signals": ["top 3 sushi places in Toronto, search each, fetch 3 URLs each"],
        },
        {
            "name": "correction",
            "user_message": "No, I meant pink buttons, not purple.",
            "recent_turns": [{"role": "user", "content": "Make buttons purple."}],
        },
        {
            "name": "refinement",
            "user_message": "also check Walmart for the same storage bin",
            "continuation_state": {"objective": "Find a storage bin under $20 near Toronto at Canadian Tire."},
        },
        {
            "name": "short_followup_frame",
            "user_message": "what about razr",
            "recent_turns": [
                {"role": "user", "content": "i want to buy an ergonomic gaming race chair under 500 cad"},
            ],
        },
        {
            "name": "meta_instruction",
            "user_message": "When planning, be more concise and do not use web unless needed.",
        },
    ]
    results = []
    for case in cases:
        relation = classify_turn_relation(
            user_message=str(case.get("user_message") or ""),
            continuation_state=case.get("continuation_state") if isinstance(case.get("continuation_state"), dict) else None,
            recent_turns=case.get("recent_turns") if isinstance(case.get("recent_turns"), list) else None,
            distant_memory_signals=case.get("distant_memory_signals") if isinstance(case.get("distant_memory_signals"), list) else None,
            ambiguous_followup=bool(case.get("ambiguous_followup")),
        )
        results.append({"name": case["name"], **relation.to_dict()})
    return results


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(text or "").lower())
        if token not in _STOPWORDS
    }


def _overlap(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return round(len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens))), 4)


def _recent_text(turns: list[dict[str, Any]]) -> str:
    parts = []
    for item in turns[-8:]:
        if isinstance(item, dict):
            parts.append(str(item.get("content") or ""))
    return "\n".join(parts)


def _is_short_followup(text: str, *, recent_frame: str = "") -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return False
    tokens = _tokens(normalized)
    if bool(_SHORT_FOLLOWUP_RE.search(normalized)) and len(tokens) <= 8:
        return True
    # Bare elliptical refinements are common in real chat:
    #   "who is shawon saha" -> "york university one"
    # They are not self-contained tasks. Only treat them as follow-ups when a
    # usable prior frame exists, and block obvious fresh questions/commands.
    if not recent_frame or len(tokens) == 0 or len(tokens) > 5:
        return False
    if _BARE_FOLLOWUP_BLOCK_RE.search(normalized):
        return False
    return True


def _latest_user_frame(current: str, turns: list[dict[str, Any]]) -> str:
    skipped_current = False
    for item in reversed(turns or []):
        if not isinstance(item, dict) or str(item.get("role") or "") != "user":
            continue
        content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if not content:
            continue
        if not skipped_current and content == str(current or "").strip():
            skipped_current = True
            continue
        if _is_low_value_frame(content):
            continue
        if len(_tokens(content)) >= 2:
            return content
    return ""


def _is_low_value_frame(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return True
    if _RESUME_RE.search(lowered) or _META_RE.search(lowered):
        return True
    return bool(re.fullmatch(r"(hi|hello|hey|thanks|thank you|ok|okay|cool|yes|no|sure)[.! ]*", lowered))


def _join_frame(frame: str, update: str) -> str:
    frame_text = re.sub(r"\s+", " ", str(frame or "")).strip()
    update_text = re.sub(r"\s+", " ", str(update or "")).strip()
    if frame_text and update_text:
        return f"{frame_text}\nCurrent turn update: {update_text}"
    return frame_text or update_text
