from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from engine.context_schema import ContextPhase


ATTENTION_VERSION = "runtime-attention-v1"


@dataclass(frozen=True)
class AttentionHeadScore:
    name: str
    score: float
    reason: str = ""


@dataclass(frozen=True)
class RuntimeAttentionItem:
    id: str
    kind: str
    title: str
    summary: str
    status: str = "info"
    source: str = ""
    phase: str = ""
    depends_on: list[str] = field(default_factory=list)
    raw_ref: str | None = None


@dataclass(frozen=True)
class RuntimeAttentionScore:
    item_id: str
    kind: str
    title: str
    summary: str
    status: str
    source: str
    final_score: float
    heads: dict[str, float]
    reasons: list[str]
    depends_on: list[str] = field(default_factory=list)
    raw_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeAttentionSnapshot:
    version: str
    phase: str
    items: list[RuntimeAttentionScore]
    omitted_count: int
    policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "phase": self.phase,
            "items": [item.to_dict() for item in self.items],
            "omitted_count": self.omitted_count,
            "policy": dict(self.policy),
        }

    def render(self, *, max_items: int = 6) -> str:
        if not self.items:
            return "Runtime Attention:\n- no scored context items"
        lines = [
            "Runtime Attention:",
            f"- phase: {self.phase}",
            f"- policy: {self.policy.get('name', 'default')}",
        ]
        for item in self.items[:max_items]:
            heads = ", ".join(
                f"{name}={score:.2f}"
                for name, score in sorted(item.heads.items())
                if abs(score) >= 0.01
            )
            reason = f" ({'; '.join(item.reasons[:2])})" if item.reasons else ""
            lines.append(
                f"- {item.item_id} [{item.kind}] score={item.final_score:.2f} "
                f"status={item.status}: {item.summary[:180]}{reason}"
            )
            if heads:
                lines.append(f"  heads: {heads}")
        if self.omitted_count:
            lines.append(f"- omitted: {self.omitted_count} lower-scored item(s)")
        return "\n".join(lines)


PHASE_HEAD_WEIGHTS: dict[str, dict[str, float]] = {
    "planning": {
        "objective": 1.0,
        "continuation": 0.9,
        "pending": 0.8,
        "tool": 0.25,
        "evidence": 0.35,
        "verification": 0.2,
        "memory": 0.25,
        "contract": 0.8,
        "pass_graph": 0.7,
        "control_policy": 0.75,
        "risk_penalty": 1.0,
    },
    "acting": {
        "objective": 0.55,
        "continuation": 0.7,
        "pending": 1.0,
        "tool": 0.95,
        "evidence": 0.45,
        "verification": 0.2,
        "memory": 0.15,
        "contract": 0.9,
        "pass_graph": 0.75,
        "control_policy": 0.85,
        "risk_penalty": 1.0,
    },
    "response": {
        "objective": 0.65,
        "continuation": 0.45,
        "pending": 0.85,
        "tool": 0.45,
        "evidence": 1.0,
        "verification": 0.7,
        "memory": 0.4,
        "contract": 1.0,
        "pass_graph": 0.7,
        "control_policy": 0.7,
        "risk_penalty": 1.0,
    },
    "verification": {
        "objective": 0.55,
        "continuation": 0.4,
        "pending": 0.9,
        "tool": 0.65,
        "evidence": 1.0,
        "verification": 1.0,
        "memory": 0.25,
        "contract": 1.0,
        "pass_graph": 0.8,
        "control_policy": 0.8,
        "risk_penalty": 1.0,
    },
    "memory_commit": {
        "objective": 0.25,
        "continuation": 0.25,
        "pending": 0.35,
        "tool": 0.25,
        "evidence": 0.75,
        "verification": 0.55,
        "memory": 1.0,
        "contract": 0.6,
        "pass_graph": 0.3,
        "control_policy": 0.3,
        "risk_penalty": 1.0,
    },
}


def select_runtime_attention(
    ledger: Any,
    phase: ContextPhase | str,
    *,
    limit: int = 8,
) -> RuntimeAttentionSnapshot:
    phase_value = phase.value if isinstance(phase, ContextPhase) else str(phase)
    weights = PHASE_HEAD_WEIGHTS.get(phase_value, PHASE_HEAD_WEIGHTS["planning"])
    candidates = _items_from_ledger(ledger)
    scored = [_score_item(item, phase_value=phase_value, weights=weights) for item in candidates]
    scored.sort(key=lambda item: (-item.final_score, item.kind, item.item_id))
    selected = [item for item in scored if item.final_score > 0][: max(0, limit)]
    return RuntimeAttentionSnapshot(
        version=ATTENTION_VERSION,
        phase=phase_value,
        items=selected,
        omitted_count=max(0, len(scored) - len(selected)),
        policy={
            "name": "ledger_phase_weighted",
            "limit": limit,
            "weights": dict(weights),
        },
    )


def _items_from_ledger(ledger: Any) -> list[RuntimeAttentionItem]:
    items: list[RuntimeAttentionItem] = []
    objective = str(getattr(ledger, "objective", "") or "").strip()
    if objective:
        items.append(
            RuntimeAttentionItem(
                id="objective",
                kind="objective",
                title="Objective",
                summary=objective,
                status="active",
                source="run_ledger",
                phase=str(getattr(ledger, "phase", "") or ""),
            )
        )

    for idx, step in enumerate(list(getattr(ledger, "plan_steps", []) or [])):
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or f"step_{idx + 1}")
        items.append(
            RuntimeAttentionItem(
                id=f"plan:{step_id}",
                kind="plan_step",
                title=step_id,
                summary=str(step.get("description") or step.get("task") or ""),
                status=str(step.get("status") or "pending"),
                source="planner",
                phase=str(getattr(ledger, "phase", "") or ""),
                raw_ref=step_id,
            )
        )

    for call in list(getattr(ledger, "tool_calls", []) or []):
        tool_name = str(getattr(call, "tool_name", "") or "")
        items.append(
            RuntimeAttentionItem(
                id=str(getattr(call, "id", "") or f"tool_call:{tool_name}"),
                kind="tool_call",
                title=tool_name,
                summary=f"{tool_name}({dict(getattr(call, 'arguments', {}) or {})})",
                status=str(getattr(call, "status", "") or "pending"),
                source=str(getattr(call, "source", "") or "actor"),
                phase=str(getattr(call, "phase", "") or ""),
                depends_on=list(getattr(call, "depends_on", []) or []),
                raw_ref=str(getattr(call, "raw_ref", "") or "") or None,
            )
        )

    for result in list(getattr(ledger, "tool_results", []) or []):
        items.append(
            RuntimeAttentionItem(
                id=str(getattr(result, "id", "") or "tool_result"),
                kind="tool_result",
                title=str(getattr(result, "tool_name", "") or "tool_result"),
                summary=str(getattr(result, "summary", "") or ""),
                status=str(getattr(result, "status", "") or ("success" if getattr(result, "success", False) else "failed")),
                source=str(getattr(result, "source", "") or "tool_registry"),
                phase=str(getattr(result, "phase", "") or ""),
                depends_on=list(getattr(result, "depends_on", []) or []),
                raw_ref=str(getattr(result, "raw_ref", "") or "") or None,
            )
        )

    for evidence in list(getattr(ledger, "evidence_items", []) or []):
        items.append(
            RuntimeAttentionItem(
                id=str(getattr(evidence, "id", "") or "evidence"),
                kind="evidence",
                title=str(getattr(evidence, "source", "") or "evidence"),
                summary=str(getattr(evidence, "summary", "") or ""),
                status=str(getattr(evidence, "status", "") or "selected"),
                source=str(getattr(evidence, "source", "") or "evidence"),
                phase=str(getattr(evidence, "phase", "") or ""),
                depends_on=list(getattr(evidence, "depends_on", []) or []),
                raw_ref=str(getattr(evidence, "raw_ref", "") or "") or None,
            )
        )

    for memory in list(getattr(ledger, "memory_writes", []) or []):
        items.append(
            RuntimeAttentionItem(
                id=str(getattr(memory, "id", "") or "memory"),
                kind="memory_write",
                title="Memory Write",
                summary=str(getattr(memory, "summary", "") or ""),
                status=str(getattr(memory, "status", "") or "committed"),
                source=str(getattr(memory, "source", "") or "memory_pipeline"),
                phase=str(getattr(memory, "phase", "") or ""),
                depends_on=list(getattr(memory, "depends_on", []) or []),
                raw_ref=str(getattr(memory, "raw_ref", "") or "") or None,
            )
        )

    verification = getattr(ledger, "verification", None)
    if verification is not None:
        items.append(
            RuntimeAttentionItem(
                id=str(getattr(verification, "id", "") or "verification"),
                kind="verification",
                title="Verification",
                summary=str(getattr(verification, "verdict", "") or ""),
                status=str(getattr(verification, "status", "") or ""),
                source=str(getattr(verification, "source", "") or "verification_layer"),
                phase=str(getattr(verification, "phase", "") or ""),
                depends_on=list(getattr(verification, "depends_on", []) or []),
                raw_ref=str(getattr(verification, "raw_ref", "") or "") or None,
            )
        )

    continuation = getattr(ledger, "continuation_state", None)
    if continuation is not None:
        items.append(
            RuntimeAttentionItem(
                id=str(getattr(continuation, "id", "") or "continuation"),
                kind="continuation",
                title="Continuation",
                summary=str(getattr(continuation, "next_action", "") or getattr(continuation, "reason", "") or ""),
                status=str(getattr(continuation, "status", "") or "pending"),
                source=str(getattr(continuation, "source", "") or "run_engine"),
                phase=str(getattr(continuation, "phase", "") or ""),
                depends_on=list(getattr(continuation, "depends_on", []) or []),
                raw_ref=str(getattr(continuation, "raw_ref", "") or "") or None,
            )
        )
    contract = getattr(ledger, "workflow_contract", None)
    if contract is not None:
        gate = getattr(contract, "completion_gate", None)
        missing_slots = list(getattr(gate, "missing_slots", []) or []) if gate is not None else []
        final_allowed = bool(getattr(gate, "final_answer_allowed", True)) if gate is not None else True
        items.append(
            RuntimeAttentionItem(
                id=str(getattr(contract, "id", "") or "workflow_contract"),
                kind="workflow_contract",
                title=str(getattr(contract, "workflow_shape", "") or "workflow_contract"),
                summary=(
                    f"{getattr(contract, 'workflow_shape', 'workflow')} contract; "
                    f"final_allowed={final_allowed}; missing={', '.join(missing_slots[:4]) or 'none'}"
                ),
                status="complete" if final_allowed else "pending",
                source="workflow_contracts",
                phase=str(getattr(ledger, "phase", "") or ""),
                raw_ref=str(getattr(contract, "id", "") or "") or None,
            )
        )
    pass_graph = getattr(ledger, "pass_graph", None)
    if pass_graph is not None:
        passes = list(getattr(pass_graph, "passes", []) or [])
        items.append(
            RuntimeAttentionItem(
                id=str(getattr(pass_graph, "id", "") or "pass_graph"),
                kind="pass_graph",
                title=str(getattr(pass_graph, "workflow_shape", "") or "pass_graph"),
                summary=(
                    f"{len(passes)} specialist pass(es); "
                    f"context={getattr(pass_graph, 'context_strategy', '')}; "
                    f"stop={getattr(pass_graph, 'stop_condition', '')}"
                ),
                status="active",
                source="pass_framework",
                phase=str(getattr(ledger, "phase", "") or ""),
                raw_ref=str(getattr(pass_graph, "id", "") or "") or None,
            )
        )
    control_policy = getattr(ledger, "control_policy", None)
    if control_policy is not None:
        items.append(
            RuntimeAttentionItem(
                id=str(getattr(control_policy, "id", "") or "control_policy"),
                kind="control_policy",
                title=str(getattr(control_policy, "label", "") or "Control Policy"),
                summary=str(getattr(control_policy, "reason", "") or getattr(control_policy, "loop_shape", "") or ""),
                status="active",
                source="control_policy",
                phase=str(getattr(ledger, "phase", "") or ""),
                raw_ref=str(getattr(control_policy, "id", "") or "") or None,
            )
        )
    return items


def _score_item(
    item: RuntimeAttentionItem,
    *,
    phase_value: str,
    weights: dict[str, float],
) -> RuntimeAttentionScore:
    heads: dict[str, float] = {
        "objective": 1.0 if item.kind == "objective" else 0.0,
        "continuation": 1.0 if item.kind == "continuation" else 0.0,
        "pending": _pending_score(item),
        "tool": 1.0 if item.kind in {"tool_call", "tool_result"} else 0.0,
        "evidence": 1.0 if item.kind == "evidence" else 0.0,
        "verification": 1.0 if item.kind == "verification" else 0.0,
        "memory": 1.0 if item.kind == "memory_write" else 0.0,
        "contract": 1.0 if item.kind == "workflow_contract" else 0.0,
        "pass_graph": 1.0 if item.kind == "pass_graph" else 0.0,
        "control_policy": 1.0 if item.kind == "control_policy" else 0.0,
        "risk_penalty": _risk_penalty(item),
    }
    reasons = _reasons(item, heads)
    raw_score = sum(heads[name] * weights.get(name, 0.0) for name in heads)
    final_score = max(0.0, min(1.0, raw_score / max(1.0, sum(v for k, v in weights.items() if k != "risk_penalty"))))
    return RuntimeAttentionScore(
        item_id=item.id,
        kind=item.kind,
        title=item.title,
        summary=item.summary,
        status=item.status,
        source=item.source,
        final_score=round(final_score, 4),
        heads={name: round(score, 4) for name, score in heads.items()},
        reasons=reasons,
        depends_on=list(item.depends_on),
        raw_ref=item.raw_ref,
    )


def _pending_score(item: RuntimeAttentionItem) -> float:
    status = item.status.strip().lower()
    if item.kind == "plan_step":
        if status in {"done", "completed", "skipped"}:
            return 0.2
        if status in {"failed", "blocked", "error"}:
            return 1.0
        return 0.95
    if item.kind == "tool_call" and status in {"pending", "running", "retrying"}:
        return 0.9
    if item.kind == "continuation" and status in {"pending", "open"}:
        return 1.0
    return 0.0


def _risk_penalty(item: RuntimeAttentionItem) -> float:
    status = item.status.strip().lower()
    if status in {"failed", "error", "unsupported", "blocked"}:
        return -0.6
    if item.kind == "tool_call" and status in {"pending", "running"}:
        return -0.1
    return 0.0


def _reasons(item: RuntimeAttentionItem, heads: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if heads["objective"]:
        reasons.append("anchors current user goal")
    if heads["continuation"]:
        reasons.append("resumes unfinished work")
    if heads["pending"] >= 0.8:
        reasons.append("pending or failed work needs attention")
    if heads["tool"]:
        reasons.append("tool state affects what can be claimed")
    if heads["evidence"]:
        reasons.append("successful evidence can ground response")
    if heads["verification"]:
        reasons.append("verification state controls final answer")
    if heads["memory"]:
        reasons.append("memory commit candidate")
    if heads.get("contract"):
        reasons.append("workflow contract controls completion")
    if heads.get("pass_graph"):
        reasons.append("specialist pass framework controls decomposition")
    if heads["risk_penalty"] < 0:
        reasons.append("status requires caution")
    return reasons
