from __future__ import annotations

import re
from dataclasses import dataclass, field


_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass(frozen=True)
class SourceContract:
    objective: str
    entity_count: int = 0
    urls_per_entity: int = 0
    total_urls: int = 0
    required_tools: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def final_allowed(self) -> bool:
        return not self.missing


def infer_source_contract(objective: str) -> SourceContract:
    """Infer the minimum source workflow from plain language.

    This is intentionally topic agnostic. Stocks, sushi, laptops, papers, and
    stores should all compile to the same execution shape when the user asks
    for N entities and M URLs each.
    """

    text = " ".join(str(objective or "").lower().split())
    entity_count = _number_after(r"\btop\s+({n})\b", text)
    each_count = _number_before(
        r"\b({n})\s+(?:relevant\s+)?(?:urls?|links?|results?|sources?|articles?)\s+(?:for\s+)?(?:each|per)\b",
        text,
    )
    total_urls = each_count * entity_count if each_count and entity_count else 0
    required_tools: list[str] = []
    if any(word in text for word in ("search", "find", "lookup")):
        required_tools.append("web_search")
    if any(word in text for word in ("fetch", "open", "read")):
        required_tools.append("web_fetch")

    missing: list[str] = []
    if "each" in text and not entity_count:
        missing.append("entity_count")
    if "each" in text and not each_count:
        missing.append("urls_per_entity")
    if "fetch" in text and "web_search" not in required_tools:
        required_tools.insert(0, "web_search")

    return SourceContract(
        objective=objective,
        entity_count=entity_count,
        urls_per_entity=each_count,
        total_urls=total_urls,
        required_tools=required_tools,
        missing=missing,
    )


def _number_after(pattern: str, text: str) -> int:
    match = re.search(pattern.format(n=_number_pattern()), text)
    if not match:
        return 0
    return _to_int(match.group(1))


def _number_before(pattern: str, text: str) -> int:
    return _number_after(pattern, text)


def _number_pattern() -> str:
    return r"\d+|" + "|".join(_WORDS)


def _to_int(value: str) -> int:
    if value.isdigit():
        return int(value)
    return _WORDS.get(value, 0)
