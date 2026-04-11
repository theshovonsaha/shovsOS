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
    if any(token in q for token in ("verbosity", "concise", "detailed", "brief response", "response style")):
        predicates.add("response_verbosity")
    if any(token in q for token in ("operating system", "os do i use", "what os", "windows", "linux", "macos", "mac")):
        predicates.add("operating_system")
    if any(token in q for token in ("pronouns", "my pronoun", "what pronouns do i use")):
        predicates.add("pronouns")
    if any(token in q for token in ("environment", "prod", "production", "staging", "dev", "development", "test mode", "local mode")):
        predicates.add("environment_mode")
    if any(token in q for token in ("scope", "in scope", "out of scope", "only touch", "only modify")):
        predicates.add("scope_boundary")
    if any(token in q for token in ("budget", "time budget", "cost cap", "limit this to", "keep it under")):
        predicates.add("budget_limit")
    if any(token in q for token in ("constraint", "constraints", "must use", "do not use", "don't use", "avoid", "what did i say not to")):
        predicates.add("task_constraint")
    if any(token in q for token in ("directive", "follow up", "followup", "check back", "remind me")):
        predicates.add("followup_directive")
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
