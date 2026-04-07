from __future__ import annotations

import re
from typing import Any, Optional


CANDIDATE_SIGNAL_MAX_LINES = 12
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
    for signal in signals[-max_lines:]:
        if bool(signal.get("superseded")):
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
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
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
        key = (text.lower(), reason.lower())
        if key in seen:
            continue
        seen.add(key)
        normalized["text"] = text
        normalized["reason"] = reason
        merged.append(normalized)

    for record in blocked_records or []:
        signal = blocked_fact_to_signal(record)
        if signal is None:
            continue
        key = (signal["text"].lower(), signal["reason"].lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(signal)

    for signal in extra_signals or []:
        text = str(signal.get("text") or "").strip()
        reason = str(signal.get("reason") or "under_review").strip() or "under_review"
        if not text:
            continue
        key = (text.lower(), reason.lower())
        if key in seen:
            continue
        seen.add(key)
        normalized = dict(signal)
        normalized["text"] = text
        normalized["reason"] = reason
        merged.append(normalized)

    return merged[-max_lines:]


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