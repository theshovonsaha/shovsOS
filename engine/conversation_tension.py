from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from engine.candidate_signals import extract_stance_signals


CORRECTION_SIGNAL_RE = re.compile(
    r"\b(actually|instead|correction|updated|changed|not .* anymore|moved to|call me)\b",
    re.IGNORECASE,
)
HEDGING_SIGNAL_RE = re.compile(
    r"\b(i think|maybe|perhaps|it seems|i guess|might|could be)\b",
    re.IGNORECASE,
)
CHALLENGE_REQUEST_RE = re.compile(
    r"\b(challenge me|push back|be critical|disagree|call out|poke holes|contradict)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConversationTension:
    summary: str = ""
    notes: str = ""
    challenge_level: str = "low"
    should_challenge: bool = False
    drift_detected: bool = False
    conflicting_facts: tuple[dict[str, str], ...] = ()
    stance_drifts: tuple[dict[str, str], ...] = ()
    unacknowledged_drift: bool = False


def analyze_conversation_tension(
    *,
    user_message: str,
    current_facts: Optional[Iterable[tuple[str, str, str]]],
    deterministic_keyed_facts: Optional[Iterable[dict]],
    session_history: Optional[list[dict]],
    candidate_signals: Optional[list[dict]] = None,
    current_stance_signals: Optional[list[dict]] = None,
) -> ConversationTension:
    conflicts: list[dict[str, str]] = []
    current_index: dict[tuple[str, str], set[str]] = {}
    for subject, predicate, object_ in current_facts or []:
        key = (str(subject).strip().lower(), str(predicate).strip().lower())
        current_index.setdefault(key, set()).add(str(object_).strip())

    for fact in deterministic_keyed_facts or []:
        subject = str(fact.get("subject") or "").strip()
        predicate = str(fact.get("predicate") or "").strip()
        object_ = str(fact.get("object") or "").strip()
        if not subject or not predicate or not object_:
            continue
        existing = current_index.get((subject.lower(), predicate.lower()), set())
        if existing and object_ not in existing:
            conflicts.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "previous": ", ".join(sorted(existing)),
                    "current": object_,
                }
            )

    correction_signal = bool(CORRECTION_SIGNAL_RE.search(user_message or ""))
    hedged = bool(HEDGING_SIGNAL_RE.search(user_message or ""))
    direct_push_request = bool(CHALLENGE_REQUEST_RE.search(user_message or ""))
    stance_drifts = detect_stance_drifts(
        new_stances=list(current_stance_signals or extract_stance_signals(user_message, turn_index=0)),
        candidate_signals=list(candidate_signals or []),
        correction_signal=correction_signal,
    )
    recent_user_messages = [
        str(item.get("content") or "").strip()
        for item in (session_history or [])[-6:]
        if str(item.get("role") or "") == "user" and str(item.get("content") or "").strip()
    ]
    drift_detected = bool(conflicts or stance_drifts)
    unacknowledged_drift = any(not bool(item.get("acknowledged")) for item in stance_drifts)

    if conflicts and correction_signal:
        summary = "Current turn revises earlier user-stated facts. Preserve the update explicitly instead of smoothing over the change."
        notes = "Acknowledge the revision, state what changed, and use the new fact as authoritative going forward."
        challenge_level = "medium"
        should_challenge = True
    elif conflicts:
        summary = "Current turn conflicts with earlier user-stated facts. Do not assume both are true at once."
        notes = "Surface the contradiction plainly and ask the user to reconcile it if the answer depends on that fact."
        challenge_level = "high" if not hedged else "medium"
        should_challenge = True
    elif stance_drifts and unacknowledged_drift:
        summary = "User's current position diverges from an earlier stated stance without explicit acknowledgment."
        drift = stance_drifts[0]
        notes = (
            f"Earlier stance on {drift.get('topic')}: '{drift.get('previous')}'. "
            f"Current turn implies: '{drift.get('current')}'. Surface this if the answer depends on which position holds."
        )
        challenge_level = "medium" if not hedged else "low"
        should_challenge = True
    elif stance_drifts:
        summary = "Current turn explicitly revises an earlier stated stance. Preserve the new stance and avoid treating both positions as active."
        drift = stance_drifts[0]
        notes = (
            f"Update stance on {drift.get('topic')} from '{drift.get('previous')}' to '{drift.get('current')}' going forward."
        )
        challenge_level = "low"
        should_challenge = False
    elif direct_push_request:
        summary = "User explicitly wants pushback instead of comfort or agreement."
        notes = "Challenge weak assumptions, point out drift, and prefer precision over reassurance."
        challenge_level = "medium"
        should_challenge = True
    elif recent_user_messages and not hedged:
        summary = "Track the user's trajectory across the conversation and preserve any meaningful drift in position or constraints."
        notes = "If the current answer depends on a changed premise, make that change explicit instead of flattening the conversation into the last message only."
        challenge_level = "low"
        should_challenge = False
    else:
        summary = ""
        notes = ""
        challenge_level = "low"
        should_challenge = False

    return ConversationTension(
        summary=summary,
        notes=notes,
        challenge_level=challenge_level,
        should_challenge=should_challenge,
        drift_detected=drift_detected,
        conflicting_facts=tuple(conflicts),
        stance_drifts=tuple(stance_drifts),
        unacknowledged_drift=unacknowledged_drift,
    )


def render_conversation_tension(tension: ConversationTension) -> str:
    if not tension.summary and not tension.conflicting_facts:
        if not tension.stance_drifts:
            return ""
    lines = [f"Summary: {tension.summary}"] if tension.summary else []
    if tension.notes:
        lines.append(f"Notes: {tension.notes}")
    lines.append(f"Challenge Level: {tension.challenge_level}")
    lines.append(f"Should Challenge: {'yes' if tension.should_challenge else 'no'}")
    for conflict in tension.conflicting_facts[:4]:
        lines.append(
            "- Drift: "
            f"{conflict.get('subject')} {conflict.get('predicate')} was '{conflict.get('previous')}' "
            f"but current turn implies '{conflict.get('current')}'"
        )
    for drift in tension.stance_drifts[:4]:
        prefix = "- Stance Drift" if not drift.get("acknowledged") else "- Stance Revision"
        lines.append(
            f"{prefix}: {drift.get('topic')} was '{drift.get('previous')}' "
            f"but current turn implies '{drift.get('current')}'"
        )
    return "\n".join(lines)


def detect_stance_drifts(
    *,
    new_stances: list[dict],
    candidate_signals: list[dict],
    correction_signal: bool,
) -> list[dict[str, str]]:
    prior_by_topic: dict[str, list[dict]] = {}
    for signal in candidate_signals or []:
        if str(signal.get("signal_type") or "") != "stance":
            continue
        topic = str(signal.get("topic") or "").strip().lower()
        if not topic:
            continue
        prior_by_topic.setdefault(topic, []).append(signal)

    drifts: list[dict[str, str]] = []
    for stance in new_stances or []:
        topic = str(stance.get("topic") or "").strip().lower()
        position = str(stance.get("position") or "").strip()
        if not topic or not position:
            continue
        priors = prior_by_topic.get(topic, [])
        if not priors:
            continue
        last_prior = None
        for signal in reversed(priors):
            if not bool(signal.get("superseded")):
                last_prior = signal
                break
        if last_prior is None:
            continue
        previous = str(last_prior.get("position") or "").strip()
        if not previous:
            continue
        similarity = _token_similarity(previous, position)
        if similarity >= 0.75:
            continue
        drifts.append(
            {
                "topic": str(stance.get("topic") or "").strip(),
                "previous": previous,
                "current": position,
                "acknowledged": bool(correction_signal),
            }
        )
    return drifts


def _token_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", (left or "").lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", (right or "").lower()))
    if not left_tokens or not right_tokens:
        return 1.0 if (left or "").strip().lower() == (right or "").strip().lower() else 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)