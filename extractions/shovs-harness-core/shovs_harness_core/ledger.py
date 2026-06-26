from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    created_at: str = field(default_factory=_now)


@dataclass(frozen=True)
class ToolResult:
    id: str
    call_id: str
    name: str
    ok: bool
    data: dict[str, Any]
    summary: str = ""
    created_at: str = field(default_factory=_now)


@dataclass
class Ledger:
    """The authority record for one task.

    The model can propose actions. Only this ledger can say an action happened.
    """

    objective: str
    allowed_tools: list[str]
    calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)

    def event(self, kind: str, **data: Any) -> dict[str, Any]:
        item = {"id": _id("evt"), "kind": kind, "at": _now(), **data}
        self.events.append(item)
        return item

    def add_call(self, name: str, args: dict[str, Any]) -> ToolCall:
        if name not in self.allowed_tools:
            self.event("tool_rejected", tool=name, reason="not_allowed")
            raise ValueError(f"tool not allowed: {name}")
        call = ToolCall(id=_id("call"), name=name, args=dict(args))
        self.calls.append(call)
        self.event("tool_call", call_id=call.id, tool=name)
        return call

    def add_result(self, call_id: str, ok: bool, data: dict[str, Any], summary: str = "") -> ToolResult:
        call = self.call_by_id(call_id)
        if call is None:
            self.event("tool_result_rejected", call_id=call_id, reason="orphan_result")
            raise ValueError(f"orphan tool result: {call_id}")
        result = ToolResult(
            id=_id("result"),
            call_id=call.id,
            name=call.name,
            ok=bool(ok),
            data=dict(data),
            summary=summary,
        )
        self.results.append(result)
        self.event("tool_result", result_id=result.id, call_id=call.id, ok=result.ok)
        return result

    def call_by_id(self, call_id: str) -> ToolCall | None:
        return next((call for call in self.calls if call.id == call_id), None)

    def result_by_id(self, result_id: str) -> ToolResult | None:
        return next((result for result in self.results if result.id == result_id), None)

    def successful_result_ids(self) -> set[str]:
        return {result.id for result in self.results if result.ok}

    def assert_claim_refs(self, refs: list[str]) -> None:
        valid = self.successful_result_ids()
        missing = [ref for ref in refs if ref not in valid]
        if missing:
            self.event("claim_rejected", missing_refs=missing)
            raise ValueError(f"claim cites non-successful results: {missing}")

    def trace(self) -> list[dict[str, Any]]:
        return list(self.events)
