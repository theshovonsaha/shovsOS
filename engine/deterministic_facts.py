from __future__ import annotations

import re
from typing import Iterable, Optional


_NAME_PATTERNS = [
    re.compile(
        r"\bcall me (?P<value>[a-z][a-z' -]{0,40}?)(?:\s+from now on|\s+going forward|\s+please|\s*$|[.!?,])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy name is (?P<value>[a-z][a-z' -]{0,40}?)(?:\s*$|[.!?,])",
        re.IGNORECASE,
    ),
]

_LOCATION_PATTERNS = [
    re.compile(r"\bi moved to (?P<value>[^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bi live in (?P<value>[^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bi am in (?P<value>[^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bi'm in (?P<value>[^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bi am based in (?P<value>[^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bi'm based in (?P<value>[^.!?\n]+)", re.IGNORECASE),
]

_TIMEZONE_PATTERNS = [
    re.compile(r"\bmy time ?zone is (?P<value>[^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bi(?:'| a)?m in the (?P<value>[^.!?\n]+ time(?: ?zone)?)", re.IGNORECASE),
]

_EDITOR_VALUES = (
    "vs code",
    "vscode",
    "cursor",
    "neovim",
    "vim",
    "emacs",
    "zed",
)
_EDITOR_PATTERNS = [
    re.compile(r"\bi use (?P<value>vs code|vscode|cursor|neovim|vim|emacs|zed)\b", re.IGNORECASE),
    re.compile(r"\bi prefer (?P<value>vs code|vscode|cursor|neovim|vim|emacs|zed)\b", re.IGNORECASE),
    re.compile(r"\bmy editor is (?P<value>vs code|vscode|cursor|neovim|vim|emacs|zed)\b", re.IGNORECASE),
]

_PACKAGE_MANAGER_PATTERNS = [
    re.compile(r"\bi use (?P<value>pnpm|npm|yarn|bun)\b(?:\s+as (?:my )?package manager)?", re.IGNORECASE),
    re.compile(r"\bmy package manager is (?P<value>pnpm|npm|yarn|bun)\b", re.IGNORECASE),
]

_LANGUAGE_PATTERNS = [
    re.compile(r"\bmy primary language is (?P<value>[^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bi mainly code in (?P<value>[^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bi primarily use (?P<value>[^.!?\n]+)(?:\s+for coding|\s+at work|\s*$|[.!?,])", re.IGNORECASE),
]

_RESPONSE_VERBOSITY_PATTERNS = [
    re.compile(
        r"\b(?:i (?:prefer|want)|please keep|keep|be)\s+(?:your\s+)?(?:responses?|answers?)?\s*(?P<value>concise|brief|short|detailed|verbose|in depth|in-depth)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i like)\s+(?P<value>concise|brief|short|detailed|verbose|in depth|in-depth)\s+(?:responses?|answers?)\b",
        re.IGNORECASE,
    ),
]

_OS_PATTERNS = [
    re.compile(r"\bi use (?P<value>macos|mac os|os x|mac|windows|linux)\b", re.IGNORECASE),
    re.compile(r"\bi(?:'m| am) on (?P<value>macos|mac os|os x|mac|windows|linux)\b", re.IGNORECASE),
    re.compile(r"\bmy (?:os|operating system) is (?P<value>macos|mac os|os x|mac|windows|linux)\b", re.IGNORECASE),
]

_PRONOUN_PATTERNS = [
    re.compile(r"\bmy pronouns are (?P<value>[a-z]+/[a-z]+(?:/[a-z]+)?)\b", re.IGNORECASE),
    re.compile(r"\bi use (?P<value>[a-z]+/[a-z]+(?:/[a-z]+)?) pronouns\b", re.IGNORECASE),
]
_ENVIRONMENT_PATTERNS = [
    re.compile(
        r"\buse (?P<value>production|prod|development|dev|staging|test|testing|local)\s+(?:not|instead of|over)\s+(?:production|prod|development|dev|staging|test|testing|local)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:use|target|run in|work in)\s+(?P<value>production|prod|development|dev|staging|test|testing|local)\s+(?:environment|mode)\b",
        re.IGNORECASE,
    ),
]
_SCOPE_PATTERNS = [
    re.compile(r"\b(?:keep|limit)\s+(?:the\s+)?scope\s+to\s+(?P<value>[^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bonly\s+(?:touch|modify|change|work in)\s+(?P<value>[^.!?\n]+)", re.IGNORECASE),
]
_BUDGET_PATTERNS = [
    re.compile(r"\b(?:my\s+|the\s+)?(?:budget|time budget|cost cap)\s+is\s+(?P<value>[^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bkeep\s+(?:it|this|the work)\s+under\s+(?P<value>[^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\blimit\s+(?:this|the work)\s+to\s+(?P<value>[^.!?\n]+)", re.IGNORECASE),
]
_TASK_CONSTRAINT_PATTERNS = [
    re.compile(
        r"\b(?:do not|don't|never|avoid)\s+(?P<value>(?:use|edit|change|touch|rewrite|refactor|browse|search|fetch|install|delete)\b[^.!?\n]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:must|need to|be sure to)\s+(?P<value>(?:use|preserve|keep|include|avoid|skip)\b[^.!?\n]*)",
        re.IGNORECASE,
    ),
]
_FOLLOWUP_DIRECTIVE_PATTERNS = [
    re.compile(r"\b(?:follow up|check back|revisit)\s+(?P<value>[^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bremind me to\s+(?P<value>[^.!?\n]+)", re.IGNORECASE),
]

_SPLIT_TRAILING_RE = re.compile(r"\b(?:but|and|because|so|what|where|who|please)\b", re.IGNORECASE)
_TRAILING_LOCATION_NOISE_RE = re.compile(r"(?:\s+(?:now|currently|today|instead|actually|too))+$", re.IGNORECASE)
_TRAILING_NAME_NOISE_RE = re.compile(r"(?:\s+(?:now|currently|today|instead|actually|too))+$", re.IGNORECASE)
_TRAILING_GENERIC_NOISE_RE = re.compile(r"(?:\s+(?:now|currently|today|instead|actually|too|for coding|at work))+$", re.IGNORECASE)

_LANGUAGE_NORMALIZATION = {
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "python": "Python",
    "go": "Go",
    "golang": "Go",
    "rust": "Rust",
    "java": "Java",
    "c#": "C#",
    "csharp": "C#",
    "c++": "C++",
}
_ENVIRONMENT_NORMALIZATION = {
    "prod": "production",
    "production": "production",
    "dev": "development",
    "development": "development",
    "staging": "staging",
    "test": "test",
    "testing": "test",
    "local": "local",
}
_MULTI_VALUE_PREDICATES = {"task_constraint", "followup_directive"}


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def _normalize_predicate(text: str) -> str:
    return _normalize((text or "").replace("_", " ").replace("-", " ")).lower()


def _clean_name(raw: str) -> str:
    text = _normalize(raw).strip(" .,!?:;")
    if not text:
        return ""
    text = _SPLIT_TRAILING_RE.split(text, maxsplit=1)[0]
    text = _TRAILING_NAME_NOISE_RE.sub("", text)
    text = _normalize(text.strip(" .,!?:;"))
    if not text:
        return ""
    words = []
    for chunk in text.split():
        if not chunk:
            continue
        words.append(chunk[:1].upper() + chunk[1:])
    return " ".join(words[:3]).strip()


def _clean_location(raw: str) -> str:
    text = _normalize(raw)
    if not text:
        return ""
    text = _SPLIT_TRAILING_RE.split(text, maxsplit=1)[0]
    text = _TRAILING_LOCATION_NOISE_RE.sub("", text)
    return _normalize(text.strip(" .,!?:;"))


def _clean_generic_value(raw: str) -> str:
    text = _normalize(raw)
    if not text:
        return ""
    text = _SPLIT_TRAILING_RE.split(text, maxsplit=1)[0]
    text = _TRAILING_GENERIC_NOISE_RE.sub("", text)
    return _normalize(text.strip(" .,!?:;"))


def _clean_editor(raw: str) -> str:
    value = _clean_generic_value(raw).lower()
    if not value:
        return ""
    if value == "vscode":
        return "VS Code"
    if value in {"vs code", "cursor"}:
        return value.title() if value == "cursor" else "VS Code"
    return value[:1].upper() + value[1:]


def _clean_package_manager(raw: str) -> str:
    value = _clean_generic_value(raw).lower()
    return value if value in {"pnpm", "npm", "yarn", "bun"} else ""


def _clean_language(raw: str) -> str:
    value = _clean_generic_value(raw).lower()
    return _LANGUAGE_NORMALIZATION.get(value, "")


def _clean_response_verbosity(raw: str) -> str:
    value = _clean_generic_value(raw).lower()
    if value in {"concise", "brief", "short"}:
        return "concise"
    if value in {"detailed", "verbose", "in depth", "in-depth"}:
        return "detailed"
    return ""


def _clean_operating_system(raw: str) -> str:
    value = _clean_generic_value(raw).lower()
    if value in {"macos", "mac os", "os x", "mac"}:
        return "macOS"
    if value == "windows":
        return "Windows"
    if value == "linux":
        return "Linux"
    return ""


def _clean_pronouns(raw: str) -> str:
    value = _clean_generic_value(raw).lower().replace(" ", "")
    if re.fullmatch(r"[a-z]+/[a-z]+(?:/[a-z]+)?", value):
        return value
    return ""


def _clean_environment_mode(raw: str) -> str:
    value = _clean_generic_value(raw).lower()
    return _ENVIRONMENT_NORMALIZATION.get(value, "")


def _build_fact(subject: str, predicate: str, object_: str) -> dict:
    fact_text = f"{subject} {predicate} {object_}".strip()
    return {
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "fact": fact_text,
        "key": f"{subject} {predicate}".strip(),
        "source": "user_stated",
    }


def merge_fact_records(*groups: Iterable[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for group in groups:
        for item in group or []:
            subject = str(item.get("subject") or "").strip()
            predicate = str(item.get("predicate") or "").strip()
            object_ = str(item.get("object") or item.get("object_") or "").strip()
            if not subject or not predicate:
                continue
            key = (subject.lower(), predicate.lower(), object_.lower())
            if key in seen:
                continue
            seen.add(key)
            enriched = dict(item)
            enriched.setdefault("fact", f"{subject} {predicate} {object_}".strip())
            enriched.setdefault("key", f"{subject} {predicate}".strip())
            merged.append(enriched)
    return merged


def merge_void_records(*groups: Iterable[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for item in group or []:
            subject = str(item.get("subject") or "").strip()
            predicate = str(item.get("predicate") or "").strip()
            if not subject or not predicate:
                continue
            key = (subject.lower(), predicate.lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append({"subject": subject, "predicate": predicate, **{k: v for k, v in item.items() if k not in {"subject", "predicate"}}})
    return merged


def extract_user_stated_fact_updates(
    user_message: str,
    *,
    current_facts: Optional[Iterable[tuple[str, str, str]]] = None,
) -> tuple[list[dict], list[dict]]:
    text = user_message or ""
    extracted: list[dict] = []

    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_name(match.group("value"))
            if value:
                extracted.append(_build_fact("User", "preferred_name", value))
            break

    for pattern in _LOCATION_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_location(match.group("value"))
            if value:
                extracted.append(_build_fact("User", "location", value))
            break

    for pattern in _TIMEZONE_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_generic_value(match.group("value"))
            if value:
                extracted.append(_build_fact("User", "timezone", value))
            break

    for pattern in _EDITOR_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_editor(match.group("value"))
            if value:
                extracted.append(_build_fact("User", "preferred_editor", value))
            break

    for pattern in _PACKAGE_MANAGER_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_package_manager(match.group("value"))
            if value:
                extracted.append(_build_fact("User", "package_manager", value))
            break

    for pattern in _LANGUAGE_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_language(match.group("value"))
            if value:
                extracted.append(_build_fact("User", "primary_language", value))
            break

    for pattern in _RESPONSE_VERBOSITY_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_response_verbosity(match.group("value"))
            if value:
                extracted.append(_build_fact("User", "response_verbosity", value))
            break

    for pattern in _OS_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_operating_system(match.group("value"))
            if value:
                extracted.append(_build_fact("User", "operating_system", value))
            break

    for pattern in _PRONOUN_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_pronouns(match.group("value"))
            if value:
                extracted.append(_build_fact("User", "pronouns", value))
            break

    for pattern in _ENVIRONMENT_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_environment_mode(match.group("value"))
            if value:
                extracted.append(_build_fact("Task", "environment_mode", value))
            break

    for pattern in _SCOPE_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_generic_value(match.group("value"))
            if value:
                extracted.append(_build_fact("Task", "scope_boundary", value))
            break

    for pattern in _BUDGET_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_generic_value(match.group("value"))
            if value:
                extracted.append(_build_fact("Task", "budget_limit", value))
            break

    for pattern in _TASK_CONSTRAINT_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_generic_value(match.group(0))
            if value:
                extracted.append(_build_fact("Task", "task_constraint", value))
            break

    for pattern in _FOLLOWUP_DIRECTIVE_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _clean_generic_value(match.group(0))
            if value:
                extracted.append(_build_fact("Task", "followup_directive", value))
            break

    current_index: dict[tuple[str, str], set[str]] = {}
    for subject, predicate, object_ in current_facts or []:
        key = (_normalize(subject).lower(), _normalize(predicate).lower())
        current_index.setdefault(key, set()).add(_normalize(object_).lower())

    new_facts: list[dict] = []
    voids: list[dict] = []
    for fact in merge_fact_records(extracted):
        key = (fact["subject"].lower(), fact["predicate"].lower())
        current_objects = current_index.get(key, set())
        new_object = _normalize(fact.get("object", "")).lower()
        if new_object and new_object in current_objects:
            continue
        if current_objects and fact["predicate"] not in _MULTI_VALUE_PREDICATES:
            voids.append({
                "subject": fact["subject"],
                "predicate": fact["predicate"],
                "source": "user_stated",
            })
        new_facts.append(fact)

    return new_facts, merge_void_records(voids)


def filter_redundant_user_alias_facts(
    records: Iterable[dict],
    *,
    deterministic_facts: Optional[Iterable[dict]] = None,
    current_facts: Optional[Iterable[tuple[str, str, str]]] = None,
) -> tuple[list[dict], list[dict]]:
    """
    Remove compression-side persona paraphrases once we already have canonical
    user facts such as `User preferred_name` and `User location`.
    """
    known_name = ""
    known_location = ""
    known_timezone = ""

    for item in deterministic_facts or []:
        predicate = _normalize_predicate(str(item.get("predicate") or ""))
        object_ = _normalize(str(item.get("object") or "")).lower()
        if predicate == "preferred name" and object_:
            known_name = object_
        elif predicate == "location" and object_:
            known_location = object_
        elif predicate == "timezone" and object_:
            known_timezone = object_

    for subject, predicate, object_ in current_facts or []:
        pred = _normalize_predicate(predicate)
        obj = _normalize(object_).lower()
        if pred == "preferred name" and obj:
            known_name = known_name or obj
        elif pred == "location" and obj:
            known_location = known_location or obj
        elif pred == "timezone" and obj:
            known_timezone = known_timezone or obj

    allowed: list[dict] = []
    blocked: list[dict] = []
    for record in records or []:
        subject = _normalize(str(record.get("subject") or "")).lower()
        predicate = _normalize_predicate(str(record.get("predicate") or ""))
        object_ = _normalize(str(record.get("object") or "")).lower()

        alias_noise = False
        if known_name and subject == known_name and predicate in {
            "lives in",
            "is the user's name",
            "name",
            "preferred name",
        }:
            alias_noise = True
        if known_name and object_ == known_name and predicate in {
            "is the user's name",
            "name",
            "preferred name",
        }:
            alias_noise = True
        if known_location and subject == known_name and object_ == known_location and predicate in {
            "lives in",
            "is based in",
            "based in",
            "location",
        }:
            alias_noise = True
        if known_timezone and subject == known_name and object_ == known_timezone and predicate in {
            "timezone",
            "is in timezone",
        }:
            alias_noise = True

        if alias_noise:
            enriched = dict(record)
            enriched["grounding_reason"] = "redundant_user_alias"
            blocked.append(enriched)
        else:
            allowed.append(record)

    return allowed, blocked


def is_redundant_user_alias_text(
    text: str,
    *,
    deterministic_facts: Optional[Iterable[dict]] = None,
    current_facts: Optional[Iterable[tuple[str, str, str]]] = None,
) -> bool:
    normalized_text = _normalize((text or "").replace("_", " ").replace("-", " ")).lower()
    if not normalized_text:
        return False

    known_name = ""
    known_location = ""
    known_timezone = ""

    for item in deterministic_facts or []:
        predicate = _normalize_predicate(str(item.get("predicate") or ""))
        object_ = _normalize(str(item.get("object") or "")).lower()
        if predicate == "preferred name" and object_:
            known_name = object_
        elif predicate == "location" and object_:
            known_location = object_
        elif predicate == "timezone" and object_:
            known_timezone = object_

    for _, predicate, object_ in current_facts or []:
        pred = _normalize_predicate(predicate)
        obj = _normalize(object_).lower()
        if pred == "preferred name" and obj:
            known_name = known_name or obj
        elif pred == "location" and obj:
            known_location = known_location or obj
        elif pred == "timezone" and obj:
            known_timezone = known_timezone or obj

    if known_name and known_location:
        if f"{known_name} lives in {known_location}" in normalized_text:
            return True
        if f"{known_name} based in {known_location}" in normalized_text:
            return True
    if known_name and f"{known_name} is the user s name" in normalized_text:
        return True
    if known_name and f"{known_name} preferred name" in normalized_text:
        return True
    if known_name and known_timezone and f"{known_name} timezone {known_timezone}" in normalized_text:
        return True
    return False
