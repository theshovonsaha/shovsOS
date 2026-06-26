from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


MEMORY_GOVERNOR_VERSION = "memory-governor-v1"

# Junk values that must never be committed as a current fact (e.g. the actor
# storing ``User --is_named--> unknown`` before the user supplied a name).
_PLACEHOLDER_OBJECT_VALUES = {
    "unknown", "n/a", "na", "none", "null", "nil", "tbd", "?", "??",
    "undefined", "not provided", "not specified", "no name", "not sure",
    "i don't know", "i dont know", "idk", "unspecified", "value", "object",
}


@dataclass(frozen=True)
class MemoryDecision:
    action: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    reason: str = ""
    provenance: str = "runtime"
    confidence: float = 1.0
    voids: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GovernedMemoryCommit:
    facts: list[dict[str, Any]]
    voids: list[dict[str, Any]]
    decisions: list[MemoryDecision]

    def decision_payloads(self) -> list[dict[str, Any]]:
        return [decision.to_dict() for decision in self.decisions]


def govern_memory_commit(
    *,
    facts: list[dict[str, Any]],
    voids: list[dict[str, Any]],
    current_facts: Iterable[tuple[str, str, str]] | None,
) -> GovernedMemoryCommit:
    """Normalize and gate proposed memory writes before graph mutation.

    The governor is deliberately deterministic. It does not decide what the
    user meant; earlier extraction stages do that. It enforces memory-state
    invariants: no malformed current facts, no duplicate active fact writes,
    and no mismatched current facts without a corresponding void unless the
    write is explicitly conflict-traced.
    """
    current_by_key = {
        (_norm(subject), _norm(predicate)): str(object_).strip()
        for subject, predicate, object_ in (current_facts or [])
        if _norm(subject) and _norm(predicate)
    }
    requested_void_keys = {
        (_norm(item.get("subject")), _norm(item.get("predicate")))
        for item in (voids or [])
        if _norm(item.get("subject")) and _norm(item.get("predicate"))
    }
    decisions: list[MemoryDecision] = []
    governed_facts: list[dict[str, Any]] = []
    governed_voids: list[dict[str, Any]] = [
        dict(item) for item in (voids or [])
        if _norm(item.get("subject")) and _norm(item.get("predicate"))
    ]

    for fact in facts or []:
        normalized, reject_reason = _normalize_fact(fact)
        subject = str(normalized.get("subject") or "")
        predicate = str(normalized.get("predicate") or "")
        object_ = str(normalized.get("object") or "")
        key = (_norm(subject), _norm(predicate))

        if reject_reason:
            decisions.append(MemoryDecision(
                action="ignore",
                subject=subject,
                predicate=predicate,
                object=object_,
                reason=reject_reason,
                confidence=0.0,
            ))
            continue

        existing_object = current_by_key.get(key)
        if existing_object is not None and _norm(existing_object) == _norm(object_):
            decisions.append(MemoryDecision(
                action="ignore",
                subject=subject,
                predicate=predicate,
                object=object_,
                reason="current_fact_already_matches",
            ))
            continue

        if bool(normalized.get("conflict_trace")) or bool(normalized.get("prior_value_disputed")):
            governed_facts.append(normalized)
            decisions.append(MemoryDecision(
                action="store_current_with_conflict_trace",
                subject=subject,
                predicate=predicate,
                object=object_,
                reason=str(normalized.get("conflict_reason") or "conflict_provenance_present"),
                provenance="conflict_trace",
            ))
            continue

        if existing_object is not None and key not in requested_void_keys:
            void = {"subject": subject, "predicate": predicate, "void_source": "memory_governor"}
            governed_voids.append(void)
            requested_void_keys.add(key)
            governed_facts.append(normalized)
            decisions.append(MemoryDecision(
                action="void_and_replace",
                subject=subject,
                predicate=predicate,
                object=object_,
                reason=f"replaces_current_value:{existing_object}",
                provenance="current_fact_conflict",
                voids=(void,),
            ))
            continue

        governed_facts.append(normalized)
        decisions.append(MemoryDecision(
            action="store_current",
            subject=subject,
            predicate=predicate,
            object=object_,
            reason="eligible_current_fact",
        ))

    return GovernedMemoryCommit(
        facts=_dedupe_facts(governed_facts),
        voids=_dedupe_voids(governed_voids),
        decisions=decisions,
    )


def _normalize_fact(fact: dict[str, Any]) -> tuple[dict[str, Any], str]:
    raw_object = fact.get("object") if "object" in fact else fact.get("object_")
    subject = str(fact.get("subject") or "").strip()
    predicate = str(fact.get("predicate") or "").strip()
    if not subject or not predicate:
        return {"subject": subject, "predicate": predicate, "object": ""}, "missing_subject_or_predicate"
    if subject == "General":
        return {"subject": subject, "predicate": predicate, "object": str(raw_object or "").strip()}, "general_subject_not_current_memory"
    if not isinstance(raw_object, (str, int, float, bool)):
        return {"subject": subject, "predicate": predicate, "object": ""}, "non_scalar_object"
    object_ = str(raw_object).strip()
    if not object_:
        return {"subject": subject, "predicate": predicate, "object": ""}, "empty_object"
    if _norm(object_) in _PLACEHOLDER_OBJECT_VALUES:
        return {"subject": subject, "predicate": predicate, "object": object_}, "placeholder_object_value"
    # Inverted identity assertion, e.g. ``theshovonsaha --is name of--> user``.
    # A real user-name fact is ``User --preferred_name--> <name>``; the inverted
    # shape is an LLM-compression artifact (often inferring the name from the
    # account email/handle present in context), so reject it.
    if _norm(predicate) in {"is name of", "is the name of", "name of"} and _norm(object_) in {"user", "the user"}:
        return {"subject": subject, "predicate": predicate, "object": object_}, "inverted_identity_assertion"
    return {**fact, "subject": subject, "predicate": predicate, "object": object_}, ""


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _dedupe_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for fact in facts:
        key = (_norm(fact.get("subject")), _norm(fact.get("predicate")), _norm(fact.get("object")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def _dedupe_voids(voids: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for void in voids:
        key = (_norm(void.get("subject")), _norm(void.get("predicate")))
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        deduped.append(void)
    return deduped
