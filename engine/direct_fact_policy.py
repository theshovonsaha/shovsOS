from __future__ import annotations

from typing import Optional


def direct_fact_predicates(query: str) -> set[str]:
    q = (query or "").lower()
    predicates: set[str] = set()
    if any(token in q for token in ("name", "call me", "preferred name", "what do you call me")):
        predicates.add("preferred_name")
    if any(token in q for token in ("where", "location", "live", "based", "current location", "moved")):
        predicates.add("location")
    if any(token in q for token in ("time zone", "timezone", "tz")):
        predicates.add("timezone")
    if any(token in q for token in ("editor", "ide", "what do i use for coding")):
        predicates.add("preferred_editor")
    if any(token in q for token in ("package manager", "npm", "pnpm", "yarn", "bun")):
        predicates.add("package_manager")
    if any(token in q for token in ("primary language", "main language", "language do i use", "what language")):
        predicates.add("primary_language")
    return predicates


def should_answer_direct_fact_from_memory(
    query: str,
    current_facts: Optional[list[tuple[str, str, str]]],
) -> bool:
    predicates = direct_fact_predicates(query)
    if not predicates or not current_facts:
        return False
    available = {
        str(predicate or "").strip().lower()
        for (_, predicate, _) in current_facts
        if str(predicate or "").strip()
    }
    return predicates.issubset(available)
