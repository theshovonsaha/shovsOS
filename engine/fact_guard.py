from __future__ import annotations

import re
from typing import Optional


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._-]*", re.IGNORECASE)
_GENERIC_SUBJECTS = {
    "user",
    "assistant",
    "agent",
    "system",
    "session",
    "general",
    "context",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "up",
    "us",
    "was",
    "we",
    "what",
    "when",
    "where",
    "who",
    "will",
    "with",
    "you",
    "your",
}


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _TOKEN_RE.findall((text or "").lower()):
        if token in _STOPWORDS:
            continue
        if len(token) <= 1 and not token.isdigit():
            continue
        tokens.add(token)
    return tokens


def _phrase_in_evidence(phrase: str, evidence: str) -> bool:
    normalized_phrase = _normalize_text(phrase)
    if len(normalized_phrase) < 4:
        return False
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])")
    return bool(pattern.search(evidence))


def _is_generic_subject(subject: str) -> bool:
    return _normalize_text(subject) in _GENERIC_SUBJECTS


def is_grounded_fact_record(
    record: dict,
    *,
    user_message: str,
    grounding_text: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Decide whether a candidate memory fact is grounded in the user's message or
    verified tool evidence from the same turn.

    We intentionally do not use assistant_response as evidence, otherwise the
    model could ratify its own hallucinations into memory.
    """
    subject = str(record.get("subject") or "").strip()
    predicate = str(record.get("predicate") or "").strip()
    object_ = str(record.get("object") or record.get("object_") or "").strip()
    fact = str(record.get("fact") or " ".join(part for part in (subject, predicate, object_) if part)).strip()

    if not subject or not predicate:
        return False, "missing_subject_or_predicate"

    if _normalize_text(subject) == "session" and _normalize_text(predicate) == "start":
        return True, "session_anchor"

    evidence_text = "\n".join(
        part for part in ((user_message or "").strip(), (grounding_text or "").strip()) if part
    )
    evidence_norm = _normalize_text(evidence_text)
    evidence_tokens = _tokenize(evidence_text)

    if not evidence_norm:
        return False, "no_evidence"

    if object_ and _phrase_in_evidence(object_, evidence_norm):
        return True, "object_exact"
    if fact and _phrase_in_evidence(fact, evidence_norm):
        return True, "fact_exact"

    object_tokens = _tokenize(object_)
    fact_tokens = _tokenize(fact)
    subject_tokens = set() if _is_generic_subject(subject) else _tokenize(subject)

    if object_tokens and object_tokens.issubset(evidence_tokens):
        return True, "object_tokens"
    if subject_tokens and object_tokens and (subject_tokens | object_tokens).issubset(evidence_tokens):
        return True, "subject_object_tokens"
    if subject_tokens and not object_tokens and subject_tokens.issubset(evidence_tokens):
        return True, "subject_tokens"
    if fact_tokens and len(fact_tokens) <= 3 and fact_tokens.issubset(evidence_tokens):
        return True, "short_fact_tokens"

    return False, "not_grounded"


def filter_grounded_fact_records(
    records: list[dict],
    *,
    user_message: str,
    grounding_text: Optional[str] = None,
) -> tuple[list[dict], list[dict]]:
    allowed: list[dict] = []
    blocked: list[dict] = []
    for record in records or []:
        grounded, reason = is_grounded_fact_record(
            record,
            user_message=user_message,
            grounding_text=grounding_text,
        )
        enriched = dict(record)
        enriched["grounding_reason"] = reason
        if grounded:
            allowed.append(enriched)
        else:
            blocked.append(enriched)
    return allowed, blocked
