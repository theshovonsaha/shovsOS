from __future__ import annotations

from typing import Optional


_PREDICATE_ALIASES: dict[str, set[str]] = {
    "preferred_name": {
        "preferred name",
        "name",
        "is the user's name",
        "user name",
    },
    "location": {
        "location",
        "lives in",
        "live in",
        "is based in",
        "based in",
        "moved to",
        "is staying in",
        "staying in",
        "current location",
    },
    "timezone": {
        "timezone",
        "time zone",
        "is in timezone",
    },
    "preferred_editor": {
        "preferred editor",
        "primary editor",
        "uses editor",
        "use editor",
        "editor",
        "editor preference",
        "ide preference",
        "has no editor preference for",
    },
    "package_manager": {
        "package manager",
        "uses package manager",
        "package tool",
    },
    "primary_language": {
        "primary language",
        "main language",
        "language",
    },
    "response_verbosity": {
        "response verbosity",
        "response style",
        "verbosity",
    },
    "operating_system": {
        "operating system",
        "os",
    },
    "pronouns": {
        "pronouns",
        "pronoun",
    },
    "environment_mode": {
        "environment mode",
        "environment",
    },
    "scope_boundary": {
        "scope boundary",
        "scope",
    },
    "budget_limit": {
        "budget limit",
        "budget",
        "time budget",
        "cost cap",
    },
    "task_constraint": {
        "task constraint",
        "constraint",
        "constraints",
    },
    "followup_directive": {
        "followup directive",
        "follow up directive",
        "directive",
    },
    "current_employer": {
        "employer",
        "current employer",
        "work at",
        "works at",
        "employed at",
        "company",
    },
    "current_project": {
        "project",
        "current project",
        "building project",
        "open source project",
    },
    "professional_role": {
        "role",
        "job title",
        "professional role",
        "work as",
    },
    "professional_focus": {
        "focus",
        "current focus",
        "professional focus",
        "role focus",
    },
    "years_experience": {
        "years experience",
        "experience",
        "years of experience",
    },
}


def normalize_memory_predicate(predicate: str) -> str:
    normalized = " ".join((predicate or "").strip().lower().replace("_", " ").replace("-", " ").split())
    if not normalized:
        return ""
    for canonical, aliases in _PREDICATE_ALIASES.items():
        if normalized == canonical.lower() or normalized in aliases:
            return canonical
    return normalized.replace(" ", "_")


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
    if any(token in q for token in ("employer", "work for", "company do i work", "where do i work")):
        predicates.add("current_employer")
    if any(token in q for token in ("project am i building", "what project", "current project", "open source project")):
        predicates.add("current_project")
    if any(token in q for token in ("my role", "job title", "professional role", "what do i work as")):
        predicates.add("professional_role")
    if any(token in q for token in ("current focus", "professional focus", "focused on", "ai integration or enterprise apps")):
        predicates.add("professional_focus")
    if any(token in q for token in ("years experience", "years of experience", "how much experience")):
        predicates.add("years_experience")
    return predicates


def should_answer_direct_fact_from_memory(
    query: str,
    current_facts: Optional[list[tuple[str, str, str]]],
) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if any(token in q for token in (
        "research",
        "investigate",
        "compare",
        "summarize the latest",
        "latest trends",
        "set up",
        "setup",
        "explain ",
        "recommend",
        "optimize",
        "how do i",
        "how would you",
    )):
        return False
    if any(token in q for token in ("actually", "correction:", "update:", "call me ")) and "?" not in q:
        return False
    predicates = direct_fact_predicates(query)
    if not predicates or not current_facts:
        return False
    available = {
        normalize_memory_predicate(str(predicate or ""))
        for (_, predicate, _) in current_facts
        if str(predicate or "").strip()
    }
    return predicates.issubset(available)
