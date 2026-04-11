from __future__ import annotations

import re
from typing import Any, Optional


CANDIDATE_SIGNAL_MAX_LINES = 12
CANDIDATE_SIGNAL_MAX_STORED = 24
CANDIDATE_SIGNAL_MAX_AGE_TURNS = 10
CANDIDATE_SIGNAL_STALE_AFTER_TURNS = 4
CANDIDATE_SIGNAL_PROMOTION_SIGHTINGS = 2
STANCE_CORRECTION_SIGNAL_RE = re.compile(
    r"\b(actually|instead|correction|updated|changed|not .* anymore|moved to|call me)\b",
    re.IGNORECASE,
)
STANCE_HEDGING_SIGNAL_RE = re.compile(
    r"\b(i think|maybe|perhaps|it seems|i guess|might|could be)\b",
    re.IGNORECASE,
)
STANCE_EXTRACTION_PATTERNS = (
    re.compile(r"\b(?:i|we)\s+(?:prefer|want|need|support|oppose|recommend)\s+(?P<position>[^.!?;]+)", re.IGNORECASE),
    re.compile(r"\b(?:i|we)\s+(?:think|believe|feel)\s+(?P<position>[^.!?;]+)", re.IGNORECASE),
    re.compile(r"\b(?:this|that|it|we)\s+should\s+(?P<position>[^.!?;]+)", re.IGNORECASE),
    re.compile(r"\b(?:i|we)\s+(?:am|are|'m)\s+(?:for|against)\s+(?P<position>[^.!?;]+)", re.IGNORECASE),
)
STANCE_TOPIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "if", "in", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "should", "that", "the", "this", "to", "we", "with",
}


def parse_candidate_context(candidate_context: str) -> list[dict[str, str]]:
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
        signal = {"text": text, "reason": reason}
        if signal not in signals:
            signals.append(signal)
    return signals


def render_candidate_signals(signals: list[dict[str, str]], *, max_lines: int = CANDIDATE_SIGNAL_MAX_LINES) -> str:
    lines: list[str] = []
    for signal in _rank_candidate_signals(signals)[-max_lines:]:
        if bool(signal.get("superseded")) or bool(signal.get("expired")) or signal.get("prompt_eligible") is False:
            continue
        text = str(signal.get("text") or "").strip()
        if not text:
            continue
        reason = str(signal.get("reason") or "under_review").strip() or "under_review"
        lines.append(f"- Candidate: {text} (reason={reason})")
    return "\n".join(lines)


def blocked_fact_to_signal(record: dict[str, Any], *, source: str = "compression_filter") -> Optional[dict[str, str]]:
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
        return None
    signal = {
        "text": fact,
        "reason": str(record.get("grounding_reason") or "candidate").strip() or "candidate",
        "source": source,
    }
    return signal


def merge_candidate_signals(
    existing_signals: list[dict[str, str]],
    blocked_records: list[dict[str, Any]],
    *,
    extra_signals: Optional[list[dict[str, Any]]] = None,
    supersede_matching_stances: bool = False,
    max_lines: int = CANDIDATE_SIGNAL_MAX_LINES,
    current_turn: Optional[int] = None,
    max_age_turns: int = CANDIDATE_SIGNAL_MAX_AGE_TURNS,
    max_signals: int = CANDIDATE_SIGNAL_MAX_STORED,
) -> list[dict[str, str]]:
    merged: list[dict[str, Any]] = []
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    stance_topics_to_supersede = {
        str(signal.get("topic") or "").strip().lower()
        for signal in (extra_signals or [])
        if str(signal.get("signal_type") or "") == "stance" and str(signal.get("topic") or "").strip()
    }

    for signal in existing_signals or []:
        text = str(signal.get("text") or "").strip()
        reason = str(signal.get("reason") or "under_review").strip() or "under_review"
        if not text:
            continue
        normalized = dict(signal)
        if (
            supersede_matching_stances
            and str(normalized.get("signal_type") or "") == "stance"
            and str(normalized.get("topic") or "").strip().lower() in stance_topics_to_supersede
        ):
            normalized["superseded"] = True
        normalized["text"] = text
        normalized["reason"] = reason
        normalized = _normalize_candidate_signal(
            normalized,
            current_turn=current_turn,
            increment_seen=False,
        )
        key = _candidate_identity(normalized)
        by_identity[key] = normalized
        merged.append(normalized)

    for record in blocked_records or []:
        signal = blocked_fact_to_signal(record)
        if signal is None:
            continue
        signal = _normalize_candidate_signal(signal, current_turn=current_turn)
        key = _candidate_identity(signal)
        if key in by_identity:
            _merge_signal_update(by_identity[key], signal, current_turn=current_turn)
            continue
        by_identity[key] = signal
        merged.append(signal)

    for signal in extra_signals or []:
        text = str(signal.get("text") or "").strip()
        reason = str(signal.get("reason") or "under_review").strip() or "under_review"
        if not text:
            continue
        normalized = dict(signal)
        normalized["text"] = text
        normalized["reason"] = reason
        normalized = _normalize_candidate_signal(normalized, current_turn=current_turn)
        key = _candidate_identity(normalized)
        if key in by_identity:
            _merge_signal_update(by_identity[key], normalized, current_turn=current_turn)
            continue
        by_identity[key] = normalized
        merged.append(normalized)

    for signal in merged:
        _apply_candidate_lifecycle(
            signal,
            current_turn=current_turn,
            max_age_turns=max_age_turns,
        )

    active = [
        signal
        for signal in _rank_candidate_signals(merged)
        if not bool(signal.get("expired"))
    ]
    if max_signals > 0:
        active = active[-max_signals:]
    return [dict(signal) for signal in active]


def extract_stance_signals(user_message: str, *, turn_index: int) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw_clause in re.split(r"[\n.!?;]+", user_message or ""):
        clause = raw_clause.strip()
        if not clause:
            continue
        confidence = "hedged" if STANCE_HEDGING_SIGNAL_RE.search(clause) else "asserted"
        for pattern in STANCE_EXTRACTION_PATTERNS:
            match = pattern.search(clause)
            if not match:
                continue
            position = str(match.group("position") or "").strip(" ,")
            topic = _normalize_stance_topic(position)
            if not position or not topic:
                continue
            key = (topic, position.lower())
            if key in seen:
                continue
            seen.add(key)
            signals.append(
                {
                    "signal_type": "stance",
                    "topic": topic,
                    "position": position,
                    "confidence": confidence,
                    "turn_index": int(turn_index),
                    "raw_text": clause,
                    "superseded": False,
                    "source": "stance_extractor",
                    "reason": f"stance_{confidence}",
                    "text": f"Stance [{topic}]: {position}",
                }
            )
            break
    return signals


def has_correction_signal(user_message: str) -> bool:
    return bool(STANCE_CORRECTION_SIGNAL_RE.search(user_message or ""))


def _normalize_stance_topic(position: str) -> str:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", (position or "").lower())
        if token not in STANCE_TOPIC_STOPWORDS
    ]
    return " ".join(tokens[:4]).strip()


def _candidate_identity(signal: dict[str, Any]) -> tuple[str, str]:
    return (
        str(signal.get("text") or "").strip().lower(),
        str(signal.get("reason") or "under_review").strip().lower(),
    )


def _normalize_candidate_signal(
    signal: dict[str, Any],
    *,
    current_turn: Optional[int],
    increment_seen: bool = True,
) -> dict[str, Any]:
    normalized = dict(signal)
    text = str(normalized.get("text") or "").strip()
    reason = str(normalized.get("reason") or "under_review").strip() or "under_review"
    normalized["text"] = text
    normalized["reason"] = reason
    if current_turn is not None:
        first_seen = normalized.get("first_seen_turn")
        last_seen = normalized.get("last_seen_turn")
        turn_index = normalized.get("turn_index")
        normalized["first_seen_turn"] = int(
            first_seen if first_seen is not None else turn_index if turn_index is not None else current_turn
        )
        normalized["last_seen_turn"] = int(current_turn if increment_seen else (last_seen if last_seen is not None else normalized["first_seen_turn"]))
        normalized["age_turns"] = max(0, int(current_turn) - int(normalized["last_seen_turn"]))
    sightings = int(normalized.get("sightings") or 0)
    if increment_seen:
        sightings += 1
    if current_turn is not None:
        normalized["sightings"] = max(1, sightings)
    elif sightings:
        normalized["sightings"] = max(1, sightings)
    elif "sightings" in normalized:
        normalized.pop("sightings", None)
    return normalized


def _merge_signal_update(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    *,
    current_turn: Optional[int],
) -> None:
    existing["source"] = str(incoming.get("source") or existing.get("source") or "").strip() or existing.get("source")
    existing["reason"] = str(incoming.get("reason") or existing.get("reason") or "under_review").strip() or "under_review"
    existing["text"] = str(incoming.get("text") or existing.get("text") or "").strip()
    if incoming.get("signal_type"):
        existing["signal_type"] = incoming.get("signal_type")
    if incoming.get("topic"):
        existing["topic"] = incoming.get("topic")
    if incoming.get("position"):
        existing["position"] = incoming.get("position")
    if incoming.get("confidence"):
        existing["confidence"] = incoming.get("confidence")
    existing["superseded"] = bool(existing.get("superseded")) or bool(incoming.get("superseded"))
    if current_turn is not None:
        existing["last_seen_turn"] = int(current_turn)
        first_seen = existing.get("first_seen_turn")
        turn_index = existing.get("turn_index")
        existing["first_seen_turn"] = int(
            first_seen if first_seen is not None else turn_index if turn_index is not None else current_turn
        )
        existing["age_turns"] = 0
    existing["sightings"] = int(existing.get("sightings") or 1) + 1


def _apply_candidate_lifecycle(
    signal: dict[str, Any],
    *,
    current_turn: Optional[int],
    max_age_turns: int,
) -> None:
    if (
        current_turn is None
        and signal.get("last_seen_turn") is None
        and signal.get("first_seen_turn") is None
        and signal.get("turn_index") is None
    ):
        return
    if current_turn is not None and signal.get("last_seen_turn") is not None:
        signal["age_turns"] = max(0, int(current_turn) - int(signal.get("last_seen_turn") or current_turn))
    age_turns = int(signal.get("age_turns") or 0)
    sightings = int(signal.get("sightings") or 1)
    reason = str(signal.get("reason") or "under_review").strip().lower()
    signal_type = str(signal.get("signal_type") or "").strip().lower()
    provenance = _candidate_provenance_weight(signal)
    promoted = (
        sightings >= CANDIDATE_SIGNAL_PROMOTION_SIGHTINGS
        or signal_type == "stance"
        or reason in {"user_correction", "corrected"}
    )
    signal["promotion_state"] = "promoted" if promoted else "candidate"
    signal["freshness_state"] = (
        "fresh"
        if age_turns <= 1
        else "stale"
        if age_turns < CANDIDATE_SIGNAL_STALE_AFTER_TURNS
        else "aging"
    )
    decay_penalty = age_turns + (2 if bool(signal.get("superseded")) else 0)
    signal["promotion_score"] = max(0, sightings + provenance + (2 if promoted else 0) - decay_penalty)
    signal["expired"] = False
    expiry_threshold = max_age_turns * 2 if promoted else max_age_turns
    if expiry_threshold > 0 and age_turns >= expiry_threshold:
        signal["expired"] = True
    if reason in {"user_correction", "corrected"}:
        signal["expired"] = False
    signal["prompt_eligible"] = (
        not bool(signal.get("superseded"))
        and not bool(signal.get("expired"))
        and (
            promoted
            or age_turns < CANDIDATE_SIGNAL_STALE_AFTER_TURNS
            or provenance >= 2
        )
    )


def _candidate_provenance_weight(signal: dict[str, Any]) -> int:
    source = str(signal.get("source") or "").strip().lower()
    reason = str(signal.get("reason") or "").strip().lower()
    if source in {"compression_filter", "stance_extractor"}:
        return 2
    if source in {"user_message", "direct_observation"}:
        return 3
    if reason in {"user_correction", "corrected", "stance_asserted", "stance_hedged"}:
        return 2
    return 1


def _rank_candidate_signals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(signal) for signal in signals or [] if str(signal.get("text") or "").strip()],
        key=lambda signal: (
            1 if bool(signal.get("superseded")) else 0,
            1 if bool(signal.get("expired")) else 0,
            -int(signal.get("promotion_score") or 0),
            -int(signal.get("last_seen_turn") or signal.get("turn_index") or -1),
            -int(signal.get("sightings") or 0),
            str(signal.get("text") or "").lower(),
        ),
    )
