from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContextKind(str, Enum):
    INSTRUCTION = "instruction"
    OBJECTIVE = "objective"
    EVIDENCE = "evidence"
    WORKING = "working"
    MEMORY = "memory"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"


class ContextPhase(str, Enum):
    PLANNING = "planning"
    ACTING = "acting"
    RESPONSE = "response"
    MEMORY_COMMIT = "memory_commit"
    VERIFICATION = "verification"


ALL_CONTEXT_PHASES = frozenset({
    ContextPhase.PLANNING,
    ContextPhase.ACTING,
    ContextPhase.RESPONSE,
    ContextPhase.MEMORY_COMMIT,
    ContextPhase.VERIFICATION,
})


@dataclass(frozen=True)
class ContextItem:
    item_id: str
    kind: ContextKind
    title: str
    content: str
    source: str
    priority: int = 100
    max_chars: int | None = None
    ttl_turns: int | None = None
    stale: bool = False
    mutable: bool = False
    phase_visibility: frozenset[ContextPhase] = field(default_factory=lambda: ALL_CONTEXT_PHASES)
    trace_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    formatted: bool = False

    def visible_in(self, phase: ContextPhase) -> bool:
        return phase in self.phase_visibility

    def is_expired(self) -> bool:
        return self.ttl_turns is not None and self.ttl_turns <= 0

    def trace_payload(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "kind": self.kind.value,
            "title": self.title,
            "source": self.source,
            "priority": self.priority,
            "max_chars": self.max_chars,
            "ttl_turns": self.ttl_turns,
            "stale": self.stale,
            "mutable": self.mutable,
            "phase_visibility": sorted(phase.value for phase in self.phase_visibility),
            "trace_id": self.trace_id,
            "formatted": self.formatted,
            "provenance": dict(self.provenance or {}),
        }
