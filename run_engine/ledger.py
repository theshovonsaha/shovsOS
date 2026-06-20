from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from engine.context_schema import ContextPhase


LEDGER_VERSION = "run-ledger-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass
class LedgerEvent:
    id: str
    phase: str
    event_type: str
    source: str
    status: str = "info"
    created_at: str = field(default_factory=_now)
    depends_on: list[str] = field(default_factory=list)
    raw_ref: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    id: str
    tool_name: str
    arguments: dict[str, Any]
    phase: str = "acting"
    source: str = "actor"
    status: str = "pending"
    created_at: str = field(default_factory=_now)
    depends_on: list[str] = field(default_factory=list)
    raw_ref: Optional[str] = None


@dataclass
class ToolResultRecord:
    id: str
    tool_call_id: str
    tool_name: str
    success: bool
    status: str
    summary: str
    phase: str = "acting"
    source: str = "tool_registry"
    created_at: str = field(default_factory=_now)
    depends_on: list[str] = field(default_factory=list)
    supports_claims: list[str] = field(default_factory=list)
    raw_ref: Optional[str] = None


@dataclass
class EvidenceRecord:
    id: str
    source: str
    summary: str
    status: str = "selected"
    phase: str = "acting"
    created_at: str = field(default_factory=_now)
    depends_on: list[str] = field(default_factory=list)
    supports_claims: list[str] = field(default_factory=list)
    raw_ref: Optional[str] = None


@dataclass
class MemoryWriteRecord:
    id: str
    status: str
    summary: str
    phase: str = "memory_commit"
    source: str = "memory_pipeline"
    created_at: str = field(default_factory=_now)
    depends_on: list[str] = field(default_factory=list)
    supports_claims: list[str] = field(default_factory=list)
    raw_ref: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationRecord:
    id: str
    supported: bool
    verdict: str
    status: str
    issues: list[str] = field(default_factory=list)
    confidence: Optional[float] = None
    phase: str = "verification"
    source: str = "verification_layer"
    created_at: str = field(default_factory=_now)
    depends_on: list[str] = field(default_factory=list)
    raw_ref: Optional[str] = None


@dataclass
class ContinuationRecord:
    id: str
    reason: str
    next_action: str
    status: str = "pending"
    phase: str = "continue_or_done"
    source: str = "run_engine"
    created_at: str = field(default_factory=_now)
    depends_on: list[str] = field(default_factory=list)
    raw_ref: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallDraft:
    tool_name: str
    arguments: dict[str, Any]
    raw: str = ""
    source: str = "actor"
    validation_error: str = ""

    @property
    def valid(self) -> bool:
        return bool(self.tool_name) and isinstance(self.arguments, dict) and not self.validation_error


@dataclass
class RunLedger:
    run_id: str
    session_id: str
    turn_id: str
    objective: str
    allowed_tools: list[str]
    owner_id: str = ""
    agent_id: str = "default"
    ledger_mode: str = "shadow"
    phase: str = "intake"
    version: str = LEDGER_VERSION
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    plan_steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    tool_results: list[ToolResultRecord] = field(default_factory=list)
    evidence_items: list[EvidenceRecord] = field(default_factory=list)
    memory_writes: list[MemoryWriteRecord] = field(default_factory=list)
    verification: Optional[VerificationRecord] = None
    continuation_state: Optional[ContinuationRecord] = None
    events: list[LedgerEvent] = field(default_factory=list)

    def set_phase(self, phase: ContextPhase | str, *, source: str = "run_engine") -> None:
        self.phase = phase.value if isinstance(phase, ContextPhase) else str(phase)
        self.updated_at = _now()
        self.append_event("phase_transition", source=source, status="info", data={"phase": self.phase})

    def append_event(
        self,
        event_type: str,
        *,
        source: str,
        status: str = "info",
        data: Optional[dict[str, Any]] = None,
        depends_on: Optional[list[str]] = None,
        raw_ref: Optional[str] = None,
    ) -> LedgerEvent:
        event = LedgerEvent(
            id=_id("evt"),
            phase=self.phase,
            event_type=event_type,
            source=source,
            status=status,
            depends_on=list(depends_on or []),
            raw_ref=raw_ref,
            data=dict(data or {}),
        )
        self.events.append(event)
        self.updated_at = _now()
        return event

    def set_plan(self, steps: list[dict[str, Any]], *, source: str = "planner") -> None:
        self.plan_steps = [dict(step) for step in steps if isinstance(step, dict)]
        self.append_event(
            "plan_steps",
            source=source,
            data={
                "step_count": len(self.plan_steps),
                "pending": len(self.pending_steps()),
            },
        )

    def update_plan_steps(self, steps: list[dict[str, Any]], *, source: str = "run_engine") -> None:
        self.plan_steps = [dict(step) for step in steps if isinstance(step, dict)]
        self.append_event(
            "plan_steps_updated",
            source=source,
            data={
                "step_count": len(self.plan_steps),
                "pending": len(self.pending_steps()),
                "done": sum(1 for step in self.plan_steps if step.get("status") in {"done", "completed"}),
                "failed": sum(1 for step in self.plan_steps if step.get("status") == "failed"),
            },
        )

    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        source: str = "actor",
        raw_ref: Optional[str] = None,
    ) -> ToolCallRecord:
        record = ToolCallRecord(
            id=_id("call"),
            tool_name=str(tool_name or ""),
            arguments=dict(arguments or {}),
            phase=self.phase,
            source=source,
            raw_ref=raw_ref,
        )
        self.tool_calls.append(record)
        self.append_event("tool_call", source=source, status="pending", depends_on=[record.id], data={
            "tool_call_id": record.id,
            "tool_name": record.tool_name,
        })
        return record

    def link_tool_result(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        success: bool,
        status: str,
        summary: str,
        raw_ref: Optional[str] = None,
    ) -> ToolResultRecord:
        if not any(call.id == tool_call_id for call in self.tool_calls):
            raise ValueError(f"Unknown tool_call_id for tool result: {tool_call_id}")
        record = ToolResultRecord(
            id=_id("result"),
            tool_call_id=tool_call_id,
            tool_name=str(tool_name or ""),
            success=bool(success),
            status=str(status or ("ok" if success else "failed")),
            summary=str(summary or ""),
            phase=self.phase,
            depends_on=[tool_call_id],
            raw_ref=raw_ref,
        )
        self.tool_results.append(record)
        for call in self.tool_calls:
            if call.id == tool_call_id:
                call.status = record.status
                break
        self.append_event("tool_result", source="tool_registry", status=record.status, depends_on=[tool_call_id, record.id], data={
            "tool_call_id": tool_call_id,
            "tool_result_id": record.id,
            "tool_name": record.tool_name,
            "success": record.success,
        })
        return record

    def add_evidence_from_result(self, result: ToolResultRecord) -> Optional[EvidenceRecord]:
        if not result.success:
            return None
        evidence = EvidenceRecord(
            id=_id("evidence"),
            source=f"tool:{result.tool_name}",
            summary=result.summary,
            phase=self.phase,
            depends_on=[result.tool_call_id, result.id],
            raw_ref=result.raw_ref or result.id,
        )
        self.evidence_items.append(evidence)
        self.append_event("evidence_selected", source=evidence.source, status="selected", depends_on=evidence.depends_on + [evidence.id])
        return evidence

    def add_memory_write(self, *, status: str, summary: str, data: Optional[dict[str, Any]] = None) -> MemoryWriteRecord:
        record = MemoryWriteRecord(
            id=_id("memory"),
            status=str(status or "committed"),
            summary=str(summary or ""),
            data=dict(data or {}),
            depends_on=[item.id for item in self.evidence_items[-4:]],
        )
        self.memory_writes.append(record)
        self.append_event("memory_write", source="memory_pipeline", status=record.status, depends_on=record.depends_on + [record.id], data=record.data)
        return record

    def set_verification(self, *, supported: bool, verdict: str, issues: list[str], confidence: Optional[float] = None) -> VerificationRecord:
        record = VerificationRecord(
            id=_id("verify"),
            supported=bool(supported),
            verdict=str(verdict or ("supported" if supported else "needs_redraft")),
            status="supported" if supported else "unsupported",
            issues=[str(item) for item in issues if str(item).strip()],
            confidence=confidence,
            depends_on=[item.id for item in self.evidence_items],
        )
        self.verification = record
        self.append_event("verification", source="verification_layer", status=record.status, depends_on=record.depends_on + [record.id], data={
            "supported": record.supported,
            "verdict": record.verdict,
            "issue_count": len(record.issues),
            "confidence": record.confidence,
        })
        return record

    def set_continuation(self, payload: dict[str, Any]) -> Optional[ContinuationRecord]:
        if not payload:
            self.continuation_state = None
            return None
        record = ContinuationRecord(
            id=_id("continue"),
            reason=str(payload.get("reason") or "unfinished_run"),
            next_action=str(payload.get("next_action") or ""),
            data=dict(payload),
            depends_on=[item.id for item in self.tool_results[-4:]],
        )
        self.continuation_state = record
        self.append_event("continuation_state", source="run_engine", status=record.status, depends_on=record.depends_on + [record.id], data=payload)
        return record

    def pending_steps(self) -> list[dict[str, Any]]:
        return [
            dict(step)
            for step in self.plan_steps
            if str(step.get("status") or "pending") not in {"done", "completed", "skipped"}
        ]

    def selected_evidence(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.evidence_items if item.status == "selected"]

    def missing_requirements(self) -> list[str]:
        missing: list[str] = []
        if self.phase in {"acting", "observe", "verification"} and self.tool_calls and not self.tool_results:
            missing.append("tool_results")
        if self.phase == "response" and self.pending_steps():
            missing.append("pending_plan_steps")
        if self.phase == "verification" and not self.evidence_items and self.tool_calls:
            missing.append("selected_evidence")
        if self.continuation_state and self.continuation_state.data.get("missing_evidence"):
            missing.append("missing_evidence")
        return missing

    def to_phase_packet(self, phase: ContextPhase | str) -> dict[str, Any]:
        phase_value = phase.value if isinstance(phase, ContextPhase) else str(phase)
        return {
            "version": self.version,
            "ledger_mode": self.ledger_mode,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "objective": self.objective,
            "phase": phase_value,
            "current_phase": self.phase,
            "allowed_tools": list(self.allowed_tools),
            "plan_steps": [dict(step) for step in self.plan_steps],
            "pending_steps": self.pending_steps(),
            "tool_calls": [asdict(item) for item in self.tool_calls],
            "tool_results": [asdict(item) for item in self.tool_results],
            "evidence_items": [asdict(item) for item in self.evidence_items],
            "memory_writes": [asdict(item) for item in self.memory_writes],
            "verification": asdict(self.verification) if self.verification else None,
            "continuation_state": asdict(self.continuation_state) if self.continuation_state else None,
            "missing_requirements": self.missing_requirements(),
            "summary": {
                "event_count": len(self.events),
                "tool_calls": len(self.tool_calls),
                "tool_results": len(self.tool_results),
                "evidence": len(self.evidence_items),
                "memory_writes": len(self.memory_writes),
                "pending_steps": len(self.pending_steps()),
                "tool_call_count": len(self.tool_calls),
                "tool_result_count": len(self.tool_results),
                "evidence_count": len(self.evidence_items),
                "memory_write_count": len(self.memory_writes),
                "pending_step_count": len(self.pending_steps()),
            },
        }

    def render_for_phase(self, phase: ContextPhase | str) -> str:
        packet = self.to_phase_packet(phase)
        lines = [
            "Canonical Run Ledger:",
            f"- objective: {self.objective or 'not recorded'}",
            f"- phase: {packet['phase']}",
            f"- mode: {self.ledger_mode}",
            f"- plan steps: {len(self.plan_steps)} total, {len(self.pending_steps())} pending",
            f"- tools: {len(self.tool_calls)} call(s), {len(self.tool_results)} result(s)",
            f"- selected evidence: {len(self.evidence_items)} item(s)",
            f"- memory writes: {len(self.memory_writes)} item(s)",
        ]
        if packet["missing_requirements"]:
            lines.append(f"- missing: {', '.join(packet['missing_requirements'])}")
        if self.pending_steps():
            lines.append("Pending steps:")
            for step in self.pending_steps()[:4]:
                tool = f" [{step.get('tool')}]" if step.get("tool") else ""
                lines.append(f"- {step.get('id', 'step')} ({step.get('status', 'pending')}){tool}: {step.get('description', '')}")
        if self.tool_results:
            lines.append("Latest tool results:")
            for result in self.tool_results[-4:]:
                lines.append(f"- {result.id} <- {result.tool_call_id}: {result.tool_name} {result.status} — {result.summary[:180]}")
        return "\n".join(lines)

    def to_trace_payload(self) -> dict[str, Any]:
        payload = self.to_phase_packet(self.phase)
        payload["events"] = [asdict(item) for item in self.events[-20:]]
        return payload
