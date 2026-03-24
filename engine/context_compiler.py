from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from engine.context_schema import ContextItem, ContextKind, ContextPhase


PHASE_KIND_ORDER: dict[ContextPhase, tuple[ContextKind, ...]] = {
    ContextPhase.PLANNING: (
        ContextKind.INSTRUCTION,
        ContextKind.RUNTIME,
        ContextKind.OBJECTIVE,
        ContextKind.MEMORY,
        ContextKind.ENVIRONMENT,
    ),
    ContextPhase.ACTING: (
        ContextKind.INSTRUCTION,
        ContextKind.RUNTIME,
        ContextKind.OBJECTIVE,
        ContextKind.WORKING,
        ContextKind.MEMORY,
        ContextKind.ENVIRONMENT,
    ),
    ContextPhase.RESPONSE: (
        ContextKind.INSTRUCTION,
        ContextKind.RUNTIME,
        ContextKind.OBJECTIVE,
        ContextKind.WORKING,
        ContextKind.MEMORY,
    ),
    ContextPhase.MEMORY_COMMIT: (
        ContextKind.RUNTIME,
        ContextKind.OBJECTIVE,
        ContextKind.WORKING,
        ContextKind.MEMORY,
    ),
    ContextPhase.VERIFICATION: (
        ContextKind.INSTRUCTION,
        ContextKind.RUNTIME,
        ContextKind.OBJECTIVE,
        ContextKind.WORKING,
        ContextKind.MEMORY,
        ContextKind.ENVIRONMENT,
    ),
}

PHASE_ALLOWED_KINDS: dict[ContextPhase, frozenset[ContextKind]] = {
    phase: frozenset(order)
    for phase, order in PHASE_KIND_ORDER.items()
}


@dataclass(frozen=True)
class CompiledContextRecord:
    item_id: str
    kind: str
    title: str
    source: str
    reason: str
    allocated_chars: int
    used_chars: int
    trace_id: str | None
    stale: bool
    mutable: bool
    provenance: dict


@dataclass
class CompiledPhaseContext:
    phase: ContextPhase
    char_budget: int
    used_chars: int
    content: str
    included: list[CompiledContextRecord] = field(default_factory=list)
    excluded: list[CompiledContextRecord] = field(default_factory=list)

    def to_trace_payload(self) -> dict:
        return {
            "phase": self.phase.value,
            "char_budget": self.char_budget,
            "used_chars": self.used_chars,
            "included": [
                {
                    "item_id": record.item_id,
                    "kind": record.kind,
                    "title": record.title,
                    "source": record.source,
                    "reason": record.reason,
                    "allocated_chars": record.allocated_chars,
                    "used_chars": record.used_chars,
                    "trace_id": record.trace_id,
                    "stale": record.stale,
                    "mutable": record.mutable,
                    "provenance": record.provenance,
                }
                for record in self.included
            ],
            "excluded": [
                {
                    "item_id": record.item_id,
                    "kind": record.kind,
                    "title": record.title,
                    "source": record.source,
                    "reason": record.reason,
                    "allocated_chars": record.allocated_chars,
                    "used_chars": record.used_chars,
                    "trace_id": record.trace_id,
                    "stale": record.stale,
                    "mutable": record.mutable,
                    "provenance": record.provenance,
                }
                for record in self.excluded
            ],
            "summary": {
                "included_count": len(self.included),
                "excluded_count": len(self.excluded),
                "included_kinds": sorted({record.kind for record in self.included}),
                "excluded_kinds": sorted({record.kind for record in self.excluded}),
            },
        }


def _kind_rank(phase: ContextPhase, kind: ContextKind) -> int:
    order = PHASE_KIND_ORDER.get(phase, ())
    try:
        return order.index(kind)
    except ValueError:
        return len(order) + 1


def _render_item(item: ContextItem) -> str:
    if item.formatted:
        return item.content.strip()
    body = item.content.strip()
    if not body:
        return ""
    if not item.title:
        return body
    return f"--- {item.title} ---\n{body}\n--- End {item.title} ---"


def _record(item: ContextItem, reason: str, allocated_chars: int = 0, used_chars: int = 0) -> CompiledContextRecord:
    return CompiledContextRecord(
        item_id=item.item_id,
        kind=item.kind.value,
        title=item.title,
        source=item.source,
        reason=reason,
        allocated_chars=allocated_chars,
        used_chars=used_chars,
        trace_id=item.trace_id,
        stale=item.stale,
        mutable=item.mutable,
        provenance=dict(item.provenance or {}),
    )


def compile_context_items(
    items: list[ContextItem],
    *,
    phase: ContextPhase,
    char_budget: int,
    truncate_section: Callable[[str, int], str],
    min_section_budget: int = 120,
) -> CompiledPhaseContext:
    allowed_kinds = PHASE_ALLOWED_KINDS.get(phase, frozenset())
    ordered_items = sorted(
        items,
        key=lambda item: (_kind_rank(phase, item.kind), item.priority, item.item_id),
    )

    sections: list[str] = []
    included: list[CompiledContextRecord] = []
    excluded: list[CompiledContextRecord] = []
    used_chars = 0

    for item in ordered_items:
        if not (item.content or "").strip():
            excluded.append(_record(item, "empty"))
            continue
        if item.kind not in allowed_kinds:
            excluded.append(_record(item, "kind_not_allowed"))
            continue
        if not item.visible_in(phase):
            excluded.append(_record(item, "phase_hidden"))
            continue
        if item.stale or item.is_expired():
            excluded.append(_record(item, "stale"))
            continue

        separator_cost = 2 if sections else 0
        remaining = char_budget - used_chars - separator_cost
        if remaining < min_section_budget:
            excluded.append(_record(item, "budget_exhausted"))
            continue

        render_budget = min(remaining, item.max_chars or remaining)
        rendered = _render_item(item)
        if not rendered:
            excluded.append(_record(item, "empty"))
            continue
        if len(rendered) > render_budget:
            rendered = truncate_section(rendered, render_budget)
        if not rendered.strip():
            excluded.append(_record(item, "empty_after_truncation", render_budget, 0))
            continue

        sections.append(rendered)
        used = len(rendered)
        used_chars += separator_cost + used
        included.append(_record(item, "included", render_budget, used))

    return CompiledPhaseContext(
        phase=phase,
        char_budget=char_budget,
        used_chars=len("\n\n".join(sections)),
        content="\n\n".join(sections),
        included=included,
        excluded=excluded,
    )
