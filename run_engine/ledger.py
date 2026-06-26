from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Optional
from uuid import uuid4

from engine.context_schema import ContextPhase
from run_engine.runtime_attention import RuntimeAttentionSnapshot, select_runtime_attention
from run_engine.pass_framework import (
    PassGraph,
    PassGraphExecution,
    complete_pass_node,
    fail_pass_node,
    initialize_pass_graph_execution,
    start_next_pass_node,
)
from run_engine.workflow_contracts import WorkflowContract
from run_engine.control_policies import ControlPolicy


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
class PolicyValidation:
    valid: bool
    issue: str = ""
    recovery_class: str = ""
    expected_tool: str = ""
    expected_arguments: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecoveryPolicy:
    recovery_class: str
    max_retries: int
    allowed_recovery_tools: list[str] = field(default_factory=list)
    requires_user_clarification: bool = False
    persist_continuation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RECOVERY_POLICIES: dict[str, RecoveryPolicy] = {
    "missing_input": RecoveryPolicy(
        recovery_class="missing_input",
        max_retries=0,
        requires_user_clarification=True,
        persist_continuation=True,
    ),
    "tool_validation_failed": RecoveryPolicy(
        recovery_class="tool_validation_failed",
        max_retries=1,
        persist_continuation=True,
    ),
    "duplicate_loop": RecoveryPolicy(
        recovery_class="duplicate_loop",
        max_retries=1,
        persist_continuation=True,
    ),
    "entity_drift": RecoveryPolicy(
        recovery_class="entity_drift",
        max_retries=1,
        allowed_recovery_tools=["web_search", "web_fetch"],
        persist_continuation=True,
    ),
    "fetch_failed": RecoveryPolicy(
        recovery_class="fetch_failed",
        max_retries=1,
        allowed_recovery_tools=["web_fetch", "web_search"],
        persist_continuation=True,
    ),
    "evidence_quota_missing": RecoveryPolicy(
        recovery_class="evidence_quota_missing",
        max_retries=1,
        allowed_recovery_tools=["web_search", "web_fetch"],
        persist_continuation=True,
    ),
    "unsupported_response_claim": RecoveryPolicy(
        recovery_class="unsupported_response_claim",
        max_retries=0,
        persist_continuation=True,
    ),
    "memory_conflict": RecoveryPolicy(
        recovery_class="memory_conflict",
        max_retries=0,
        requires_user_clarification=True,
        persist_continuation=True,
    ),
}


def recovery_policy_for(recovery_class: str) -> RecoveryPolicy:
    return RECOVERY_POLICIES.get(
        str(recovery_class or ""),
        RecoveryPolicy(recovery_class=str(recovery_class or "unknown"), max_retries=0, persist_continuation=True),
    )


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
    workflow_contract: Optional[WorkflowContract] = None
    pass_graph: Optional[PassGraph] = None
    pass_graph_execution: Optional[PassGraphExecution] = None
    pass_graph_phase_marks: list[str] = field(default_factory=list)
    control_policy: Optional[ControlPolicy] = None
    locked_entities: list[dict[str, Any]] = field(default_factory=list)
    source_contract: dict[str, Any] = field(default_factory=dict)
    policy_violations: list[dict[str, Any]] = field(default_factory=list)
    events: list[LedgerEvent] = field(default_factory=list)

    def attention_for_phase(self, phase: ContextPhase | str, *, limit: int = 8) -> RuntimeAttentionSnapshot:
        return select_runtime_attention(self, phase, limit=limit)

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
        payload = dict(data or {})
        if not self._memory_write_has_provenance(payload):
            raise ValueError("Memory write requires provenance, evidence, or explicit source metadata")
        record = MemoryWriteRecord(
            id=_id("memory"),
            status=str(status or "committed"),
            summary=str(summary or ""),
            data=payload,
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

    def set_workflow_contract(self, contract: Optional[WorkflowContract], *, source: str = "workflow_contracts") -> None:
        self.workflow_contract = contract
        if contract is None:
            return
        self.append_event(
            "workflow_contract",
            source=source,
            status="active",
            data=contract.to_dict(),
        )

    def set_pass_graph(self, graph: Optional[PassGraph], *, source: str = "pass_framework") -> None:
        self.pass_graph = graph
        self.pass_graph_execution = initialize_pass_graph_execution(graph) if graph is not None else None
        self.pass_graph_phase_marks = []
        if graph is None:
            return
        self.append_event(
            "pass_graph",
            source=source,
            status="active",
            data=graph.to_dict(),
        )
        self.append_event(
            "pass_graph_execution",
            source=source,
            status="pending",
            data=self.pass_graph_execution.to_dict() if self.pass_graph_execution else {},
        )

    def start_next_graph_pass(self, *, source: str = "graph_harness") -> dict[str, Any] | None:
        if self.pass_graph_execution is None:
            return None
        node = start_next_pass_node(self.pass_graph_execution)
        if node is None:
            return None
        payload = {
            "graph_id": self.pass_graph_execution.graph_id,
            "node": node.to_dict(),
            "execution": self.pass_graph_execution.to_dict(),
        }
        self.append_event("pass_node_started", source=source, status=node.status, data=payload)
        return payload

    def complete_graph_pass(
        self,
        node_id: str,
        *,
        outputs: Optional[dict[str, Any]] = None,
        source: str = "graph_harness",
    ) -> dict[str, Any] | None:
        if self.pass_graph_execution is None:
            return None
        node = complete_pass_node(self.pass_graph_execution, node_id, outputs=outputs)
        payload = {
            "graph_id": self.pass_graph_execution.graph_id,
            "node": node.to_dict(),
            "execution": self.pass_graph_execution.to_dict(),
        }
        self.append_event("pass_node_completed", source=source, status=node.status, data=payload)
        return payload

    def fail_graph_pass(
        self,
        node_id: str,
        *,
        issue: str,
        blocked: bool = False,
        source: str = "graph_harness",
    ) -> dict[str, Any] | None:
        if self.pass_graph_execution is None:
            return None
        node = fail_pass_node(self.pass_graph_execution, node_id, issue=issue, blocked=blocked)
        payload = {
            "graph_id": self.pass_graph_execution.graph_id,
            "node": node.to_dict(),
            "execution": self.pass_graph_execution.to_dict(),
        }
        self.append_event("pass_node_failed", source=source, status=node.status, data=payload)
        return payload

    def record_graph_phase(
        self,
        phase: ContextPhase | str,
        *,
        marker: str = "",
        source: str = "graph_harness",
    ) -> dict[str, Any] | None:
        if self.pass_graph_execution is None or self.pass_graph is None:
            return None
        phase_value = phase.value if isinstance(phase, ContextPhase) else str(phase)
        phase_marker = str(marker or phase_value).strip()
        if phase_marker and phase_marker in self.pass_graph_phase_marks:
            return None
        started = start_next_pass_node(self.pass_graph_execution)
        if started is None:
            return None
        if phase_marker:
            self.pass_graph_phase_marks.append(phase_marker)
        self.append_event(
            "pass_node_started",
            source=source,
            status=started.status,
            data={
                "graph_id": self.pass_graph_execution.graph_id,
                "node": started.to_dict(),
                "execution": self.pass_graph_execution.to_dict(),
            },
        )
        return self.complete_graph_pass(
            started.id,
            outputs={"phase": phase_value},
            source=source,
        )

    def set_control_policy(self, policy: Optional[ControlPolicy], *, source: str = "control_policy") -> None:
        self.control_policy = policy
        if policy is None:
            return
        self.append_event(
            "control_policy",
            source=source,
            status="active",
            data=policy.to_dict(),
        )

    def lock_entities(
        self,
        entities: list[str],
        *,
        entity_type: str = "entity",
        source: str = "workflow_plugin",
        status: str = "locked",
    ) -> None:
        existing = {str(item.get("value") or "").upper() for item in self.locked_entities}
        added: list[dict[str, Any]] = []
        for raw in entities or []:
            value = str(raw or "").strip()
            if not value:
                continue
            key = value.upper()
            if key in existing:
                continue
            existing.add(key)
            item = {
                "value": value,
                "entity_type": entity_type,
                "source": source,
                "status": status,
                "created_at": _now(),
            }
            self.locked_entities.append(item)
            added.append(item)
        if added:
            self.append_event(
                "entity_locked",
                source=source,
                status="active",
                data={"entities": added},
            )

    def set_source_contract(self, contract: dict[str, Any], *, source: str = "workflow_plugin") -> None:
        if not isinstance(contract, dict):
            return
        merged = dict(self.source_contract or {})
        merged.update({key: value for key, value in contract.items() if value not in (None, "", [], {})})
        self.source_contract = merged
        self.append_event(
            "source_contract",
            source=source,
            status="active",
            data=dict(self.source_contract),
        )

    def record_policy_violation(
        self,
        *,
        issue: str,
        recovery_class: str,
        tool_name: str = "",
        arguments: Optional[dict[str, Any]] = None,
        expected_tool: str = "",
        expected_arguments: Optional[dict[str, Any]] = None,
        message: str = "",
        source: str = "policy_gate",
    ) -> None:
        payload = {
            "issue": str(issue or ""),
            "recovery_class": str(recovery_class or ""),
            "recovery_policy": recovery_policy_for(recovery_class).to_dict(),
            "tool_name": str(tool_name or ""),
            "arguments": dict(arguments or {}),
            "expected_tool": str(expected_tool or ""),
            "expected_arguments": dict(expected_arguments or {}),
            "message": str(message or ""),
        }
        self.policy_violations.append(payload)
        self.append_event("policy_violation", source=source, status="blocked", data=payload)

    def next_required_action(self) -> dict[str, Any]:
        contract = dict(self.source_contract or {})
        has_evidence_contract = any(
            key in contract
            for key in ("total_urls", "total_fetches", "allowed_fetch_urls", "allowed_fetch_urls_by_entity")
        )
        if contract and has_evidence_contract:
            gate = self.completion_gate()
            if gate.get("final_answer_allowed", False):
                return {"tool": "", "arguments": {}, "reason": "source_contract_complete", "missing_slots": []}
        next_tool = str(contract.get("next_tool") or "").strip()
        next_arguments = contract.get("next_arguments") if isinstance(contract.get("next_arguments"), dict) else {}
        if next_tool:
            return {
                "tool": next_tool,
                "arguments": dict(next_arguments or {}),
                "reason": str(contract.get("next_reason") or "source_contract_next_action"),
                "missing_slots": list(contract.get("missing_slots") or []),
            }
        for step in self.pending_steps():
            tool = str(step.get("tool") or "").strip()
            if tool:
                return {
                    "tool": tool,
                    "arguments": {},
                    "reason": str(step.get("description") or "pending_plan_step"),
                    "missing_slots": [str(step.get("id") or "pending_step")],
                }
        gate = self.completion_gate()
        if not gate.get("final_answer_allowed", True):
            return {
                "tool": "",
                "arguments": {},
                "reason": str(gate.get("reason") or "completion_gate_blocked"),
                "missing_slots": list(gate.get("missing_slots") or []),
            }
        return {"tool": "", "arguments": {}, "reason": "none", "missing_slots": []}

    def validate_tool_call_against_policy(self, tool_name: str, arguments: dict[str, Any]) -> PolicyValidation:
        policy_id = str(getattr(self.control_policy, "id", "") or "")
        if policy_id != "plan_execute":
            return PolicyValidation(valid=True)
        tool = str(tool_name or "").strip()
        args = dict(arguments or {})
        expected = self.next_required_action()
        expected_tool = str(expected.get("tool") or "")
        if expected_tool and tool != expected_tool:
            return PolicyValidation(
                valid=False,
                issue="wrong_next_tool",
                recovery_class="missing_input",
                expected_tool=expected_tool,
                expected_arguments=dict(expected.get("arguments") or {}),
                message=f"Plan-execute expected {expected_tool}, got {tool or 'none'}",
            )
        locked = [str(item.get("value") or "").strip() for item in self.locked_entities if str(item.get("status") or "locked") == "locked"]
        contract = dict(self.source_contract or {})
        if tool == "web_search" and locked and bool(contract.get("forbid_unlocked_entity_drift", True)):
            query = str(args.get("query") or "").upper()
            if query and not any(re.search(rf"\b{re.escape(entity.upper())}\b", query) for entity in locked):
                return PolicyValidation(
                    valid=False,
                    issue="unlocked_entity_search",
                    recovery_class="entity_drift",
                    expected_tool=expected_tool,
                    expected_arguments=dict(expected.get("arguments") or {}),
                    message="Search query does not contain a locked entity.",
                )
        if tool == "web_fetch":
            url = str(args.get("url") or "").strip()
            allowed_urls = set(str(item) for item in contract.get("allowed_fetch_urls") or [])
            allowed_by_entity = contract.get("allowed_fetch_urls_by_entity")
            if isinstance(allowed_by_entity, dict):
                for urls in allowed_by_entity.values():
                    if isinstance(urls, list):
                        allowed_urls.update(str(item) for item in urls)
            discovery_url = str(contract.get("discovery_url") or "").strip()
            if discovery_url:
                allowed_urls.add(discovery_url)
            if allowed_urls and url not in allowed_urls:
                return PolicyValidation(
                    valid=False,
                    issue="off_contract_fetch",
                    recovery_class="entity_drift",
                    expected_tool=expected_tool,
                    expected_arguments=dict(expected.get("arguments") or {}),
                    message="Fetch URL is outside the locked source contract.",
                )
        return PolicyValidation(valid=True)

    def completion_gate(self) -> dict[str, Any]:
        contract = dict(self.source_contract or {})
        if contract:
            required_total = int(contract.get("total_urls") or contract.get("total_fetches") or 0)
            allowed_urls = set(str(item) for item in contract.get("allowed_fetch_urls") or [])
            allowed_by_entity = contract.get("allowed_fetch_urls_by_entity")
            required_per_entity = int(contract.get("urls_per_entity") or contract.get("per_entity") or 0)
            entity_observed: dict[str, int] = {}
            entity_required: dict[str, int] = {}
            if isinstance(allowed_by_entity, dict):
                for entity, urls in allowed_by_entity.items():
                    if isinstance(urls, list):
                        clean_urls = [str(item) for item in urls if str(item).strip()]
                        allowed_urls.update(clean_urls)
                        if clean_urls:
                            entity_required[str(entity)] = min(
                                len(clean_urls),
                                required_per_entity or len(clean_urls),
                            )
            fetched_urls = {
                str(call.arguments.get("url") or "").strip()
                for call in self.tool_calls
                for result in self.tool_results
                if result.tool_call_id == call.id
                and result.tool_name == "web_fetch"
                and result.success
                and str(call.arguments.get("url") or "").strip()
            }
            if allowed_urls:
                fetched_urls = {url for url in fetched_urls if url in allowed_urls}
            if isinstance(allowed_by_entity, dict):
                for entity, urls in allowed_by_entity.items():
                    if not isinstance(urls, list):
                        continue
                    clean_urls = [str(item) for item in urls if str(item).strip()]
                    entity_observed[str(entity)] = len([url for url in clean_urls if url in fetched_urls])
            missing_slots: list[str] = []
            if required_total and len(fetched_urls) < required_total:
                missing_slots.append(f"fetched_urls:{len(fetched_urls)}/{required_total}")
            for entity, required_count in entity_required.items():
                observed_count = entity_observed.get(entity, 0)
                if required_count and observed_count < required_count:
                    missing_slots.append(f"{entity}_fetched_urls:{observed_count}/{required_count}")
            final_allowed = not missing_slots
            return {
                "final_answer_allowed": final_allowed,
                "missing_slots": missing_slots,
                "reason": "source contract complete" if final_allowed else "source contract missing required evidence",
                "observed": {"fetched_url_count": len(fetched_urls), "by_entity": entity_observed},
                "required": {"total_urls": required_total, "by_entity": entity_required},
            }
        if self.workflow_contract is not None:
            contract_gate = self.workflow_contract.completion_gate
            if not contract_gate.final_answer_allowed:
                return {
                    "final_answer_allowed": False,
                    "missing_slots": list(contract_gate.missing_slots),
                    "reason": contract_gate.reason or "workflow_contract_incomplete",
                }
        return {"final_answer_allowed": True, "missing_slots": [], "reason": "no_source_contract"}

    def response_support_check(self, response_text: str) -> dict[str, Any]:
        gate = self.completion_gate()
        issues: list[str] = []
        text = str(response_text or "")
        if not gate.get("final_answer_allowed", True):
            issues.extend(str(item) for item in gate.get("missing_slots") or [])
        if re.search(r"\b(fetched|searched|read|opened|verified)\b", text, re.IGNORECASE):
            if not any(result.success for result in self.tool_results):
                issues.append("response_claims_tool_work_without_successful_tool_result")
        # Fetch-specific honesty gate: a claim that page content was actually
        # fetched/retrieved requires a SUCCESSFUL web_fetch result. A prior
        # web_search does not prove the page was read. This catches the
        # hallucinated "I fetched URL 1: the article discusses..." reports where
        # the model narrates a fetch it never performed.
        fetch_action_claim = bool(
            re.search(
                r"\b(i (?:have )?(?:now )?(?:successfully )?fetched|fetched (?:the|these|those|all)|"
                r"based on the fetched|retrieved the (?:content|page|url|article))\b",
                text,
                re.IGNORECASE,
            )
            or re.search(r"\bURL\s*\d+\s*:", text)
        )
        if fetch_action_claim:
            had_successful_fetch = any(
                result.success and str(getattr(result, "tool_name", "")) in {"web_fetch", "web_fetch_batch"}
                for result in self.tool_results
            )
            if not had_successful_fetch:
                issues.append("response_claims_fetch_without_successful_web_fetch")
        return {
            "supported": not issues,
            "issues": list(dict.fromkeys(issues)),
            "successful_tool_result_ids": [result.id for result in self.tool_results if result.success],
        }

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
        attention = self.attention_for_phase(phase_value)
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
            "workflow_contract": self.workflow_contract.to_dict() if self.workflow_contract else None,
            "pass_graph": self.pass_graph.to_dict() if self.pass_graph else None,
            "pass_graph_execution": self.pass_graph_execution.to_dict() if self.pass_graph_execution else None,
            "control_policy": self.control_policy.to_dict() if self.control_policy else None,
            "locked_entities": [dict(item) for item in self.locked_entities],
            "source_contract": dict(self.source_contract or {}),
            "policy_violations": [dict(item) for item in self.policy_violations],
            "next_required_action": self.next_required_action(),
            "completion_gate": self.completion_gate(),
            "missing_requirements": self.missing_requirements(),
            "runtime_attention": attention.to_dict(),
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
        if self.workflow_contract is not None:
            lines.append(self.workflow_contract.render())
        if self.pass_graph is not None:
            lines.append(self.pass_graph.render())
        if self.pass_graph_execution is not None:
            execution = self.pass_graph_execution.to_dict()
            summary = execution.get("summary") or {}
            lines.append(
                "Pass graph execution: "
                f"{execution.get('status')} current={execution.get('current_node_id') or 'none'} "
                f"completed={summary.get('completed', 0)} pending={summary.get('pending', 0)}"
            )
        if self.control_policy is not None:
            lines.append(self.control_policy.render())
        if self.locked_entities:
            lines.append("Locked entities:")
            for entity in self.locked_entities[:8]:
                lines.append(f"- {entity.get('value')} ({entity.get('entity_type', 'entity')})")
        next_action = self.next_required_action()
        if next_action.get("tool"):
            lines.append(f"Next required action: {next_action.get('tool')} {next_action.get('arguments') or {}}")
        gate = self.completion_gate()
        if not gate.get("final_answer_allowed", True):
            lines.append(f"Completion gate blocked: {gate.get('reason')}")
            if gate.get("missing_slots"):
                lines.append("- missing slots: " + ", ".join(str(item) for item in gate.get("missing_slots")[:6]))
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

    def render_attention_for_phase(self, phase: ContextPhase | str, *, limit: int = 6) -> str:
        return self.attention_for_phase(phase, limit=limit).render(max_items=limit)

    def to_trace_payload(self) -> dict[str, Any]:
        payload = self.to_phase_packet(self.phase)
        payload["events"] = [asdict(item) for item in self.events[-20:]]
        return payload

    def _memory_write_has_provenance(self, payload: dict[str, Any]) -> bool:
        if self.evidence_items:
            return True
        provenance_keys = {
            "path",
            "source",
            "provenance",
            "run_id",
            "fact_count",
            "candidate_signal_count",
            "indexed_fact_count",
        }
        return any(key in payload for key in provenance_keys)
