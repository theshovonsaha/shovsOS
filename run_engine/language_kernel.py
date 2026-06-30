from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from engine.context_schema import ContextPhase
from run_engine.context_ladder import build_context_ladder, render_context_ladder


LANGUAGE_KERNEL_VERSION = "language-kernel-v1"


@dataclass(frozen=True)
class PromptContract:
    """Small role contract generated from runtime state, not a giant prompt."""

    phase: str
    role: str
    objective: str
    policy: str
    allowed_next_tool: str = ""
    allowed_next_arguments: dict[str, Any] = field(default_factory=dict)
    final_answer_allowed: bool = True
    missing_slots: list[str] = field(default_factory=list)
    locked_entities: list[str] = field(default_factory=list)
    successful_tool_result_ids: list[str] = field(default_factory=list)
    turn_relation: dict[str, Any] = field(default_factory=dict)
    rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [
            "Runtime Prompt Contract:",
            f"- role: {self.role}",
            f"- phase: {self.phase}",
            f"- objective: {self.objective or 'not recorded'}",
            f"- policy: {self.policy or 'default'}",
            f"- final answer allowed: {str(self.final_answer_allowed).lower()}",
        ]
        if self.allowed_next_tool:
            lines.append(f"- allowed next tool: {self.allowed_next_tool}")
            lines.append(f"- allowed arguments: {self.allowed_next_arguments or {}}")
        if self.locked_entities:
            lines.append("- locked entities: " + ", ".join(self.locked_entities))
        if self.turn_relation:
            lines.append(
                "- turn relation: "
                f"{self.turn_relation.get('relation', 'not_recorded')} "
                f"({self.turn_relation.get('reason', 'no reason recorded')})"
            )
            required_context = self.turn_relation.get("required_context")
            blocked_context = self.turn_relation.get("blocked_context")
            if isinstance(required_context, list) and required_context:
                lines.append("- relation required context: " + ", ".join(str(item) for item in required_context[:6]))
            if isinstance(blocked_context, list) and blocked_context:
                lines.append("- relation blocked context: " + ", ".join(str(item) for item in blocked_context[:6]))
            resolved_objective = str(self.turn_relation.get("resolved_objective") or "").strip()
            if resolved_objective and resolved_objective != self.objective:
                lines.append("- relation resolved objective: " + resolved_objective.replace("\n", " / "))
            carried_context = self.turn_relation.get("carried_context")
            if isinstance(carried_context, list) and carried_context:
                lines.append("- relation carried context: " + ", ".join(str(item) for item in carried_context[:6]))
        if self.missing_slots:
            lines.append("- missing slots: " + ", ".join(self.missing_slots))
        if self.successful_tool_result_ids:
            lines.append("- successful result ids: " + ", ".join(self.successful_tool_result_ids))
        if self.rules:
            lines.append("Rules:")
            for rule in self.rules:
                lines.append(f"- {rule}")
        return "\n".join(lines)


@dataclass(frozen=True)
class MicroAgentJob:
    id: str
    label: str
    input_kind: str
    output_schema: dict[str, Any]
    max_tokens: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KernelSnapshot:
    version: str
    phase: str
    prompt_contract: PromptContract
    context_ladder: dict[str, Any]
    attention: dict[str, Any]
    tool_gate: dict[str, Any]
    memory_immune_report: dict[str, Any]
    experience_graph: dict[str, Any]
    proxy_state_eval: dict[str, Any]
    micro_agent_jobs: list[MicroAgentJob]
    ui_run_map: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "phase": self.phase,
            "prompt_contract": self.prompt_contract.to_dict(),
            "context_ladder": dict(self.context_ladder),
            "attention": dict(self.attention),
            "tool_gate": dict(self.tool_gate),
            "memory_immune_report": dict(self.memory_immune_report),
            "experience_graph": dict(self.experience_graph),
            "proxy_state_eval": dict(self.proxy_state_eval),
            "micro_agent_jobs": [job.to_dict() for job in self.micro_agent_jobs],
            "ui_run_map": dict(self.ui_run_map),
        }

    def render_for_actor(self) -> str:
        return "\n\n".join(
            [
                self.prompt_contract.render(),
                render_context_ladder(self.context_ladder),
            ]
        )


def build_kernel_snapshot(
    ledger: Any,
    phase: ContextPhase | str,
    *,
    query: str = "",
    compact_memory: str = "",
    relevant_blocks: list[dict[str, Any]] | None = None,
    raw_payloads: list[dict[str, Any]] | None = None,
    include_raw: bool = False,
    response_text: str = "",
) -> KernelSnapshot:
    """Compose the seven ShovsOS research lanes into one typed runtime object.

    This is intentionally deterministic. It does not call a model. The snapshot
    can be handed to prompts, UI, evals, or tests without mutating the run.
    """

    phase_value = phase.value if isinstance(phase, ContextPhase) else str(phase)
    evidence_refs = _evidence_refs(ledger)
    context_ladder = build_context_ladder(
        query=query or str(getattr(ledger, "objective", "") or ""),
        compact_memory=compact_memory,
        relevant_blocks=relevant_blocks,
        evidence_refs=evidence_refs,
        raw_payloads=raw_payloads,
        include_raw=include_raw,
        limit=8,
    )
    attention = _attention_snapshot(ledger, phase_value)
    prompt_contract = build_prompt_contract(ledger, phase_value)
    tool_gate = build_tool_gate(ledger)
    memory_report = build_memory_immune_report(ledger)
    experience_graph = build_experience_graph(ledger)
    proxy_eval = build_proxy_state_eval(ledger, response_text=response_text)
    jobs = propose_micro_agent_jobs(
        ledger,
        phase_value,
        memory_report=memory_report,
        proxy_eval=proxy_eval,
    )
    return KernelSnapshot(
        version=LANGUAGE_KERNEL_VERSION,
        phase=phase_value,
        prompt_contract=prompt_contract,
        context_ladder=context_ladder,
        attention=attention,
        tool_gate=tool_gate,
        memory_immune_report=memory_report,
        experience_graph=experience_graph,
        proxy_state_eval=proxy_eval,
        micro_agent_jobs=jobs,
        ui_run_map=build_ui_run_map(ledger, prompt_contract=prompt_contract, proxy_eval=proxy_eval),
    )


def build_prompt_contract(ledger: Any, phase: str) -> PromptContract:
    gate = _completion_gate(ledger)
    next_action = _next_required_action(ledger)
    successful_ids = [
        str(getattr(result, "id", ""))
        for result in list(getattr(ledger, "tool_results", []) or [])
        if getattr(result, "success", False) and str(getattr(result, "id", ""))
    ]
    policy = getattr(getattr(ledger, "control_policy", None), "id", "") or "default"
    turn_relation = dict(getattr(ledger, "turn_relation", {}) or {})
    final_allowed = bool(gate.get("final_answer_allowed", True))
    rules = [
        "Do not claim a tool ran unless its successful result id is listed.",
        "Use the allowed next tool when one is provided.",
        "Do not search or fetch outside locked entities/source contracts.",
        "Do not write memory unless provenance or evidence is available.",
    ]
    relation_name = str(turn_relation.get("relation") or "")
    if relation_name == "fresh_topic":
        rules.append("Ignore old plans, locked entities, and stale candidate memory unless explicitly reintroduced.")
    elif relation_name == "direct_continuation":
        rules.append("Preserve continuation state and execute the next required action before replanning.")
    elif relation_name == "distant_resumption":
        rules.append("Use compact memory signals and raw refs; do not assume old context is still valid.")
    elif relation_name in {"correction", "refinement"}:
        rules.append("Treat the latest user correction/refinement as dominant over conflicting older context.")
    elif relation_name == "deviation":
        rules.append("Preserve stable preferences but replan workflow state from the current turn.")
    elif relation_name == "meta_instruction":
        rules.append("Apply this as runtime/style policy, not as domain factual content.")
    if not final_allowed:
        rules.append("Do not produce a final answer while missing slots remain.")
    return PromptContract(
        phase=phase,
        role=_role_for_phase(phase),
        objective=str(getattr(ledger, "objective", "") or ""),
        policy=str(policy),
        allowed_next_tool=str(next_action.get("tool") or ""),
        allowed_next_arguments=dict(next_action.get("arguments") or {}),
        final_answer_allowed=final_allowed,
        missing_slots=[str(item) for item in gate.get("missing_slots") or []],
        locked_entities=_locked_entity_values(ledger),
        successful_tool_result_ids=successful_ids,
        turn_relation=turn_relation,
        rules=rules,
    )


def build_tool_gate(ledger: Any) -> dict[str, Any]:
    next_action = _next_required_action(ledger)
    gate = _completion_gate(ledger)
    return {
        "policy": getattr(getattr(ledger, "control_policy", None), "id", "") or "default",
        "next_required_action": dict(next_action),
        "completion_gate": dict(gate),
        "locked_entities": _locked_entity_values(ledger),
        "source_contract": dict(getattr(ledger, "source_contract", {}) or {}),
        "policy_violations": [dict(item) for item in list(getattr(ledger, "policy_violations", []) or [])],
    }


def build_memory_immune_report(ledger: Any) -> dict[str, Any]:
    writes = list(getattr(ledger, "memory_writes", []) or [])
    issues: list[str] = []
    disputed: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for write in writes:
        data = dict(getattr(write, "data", {}) or {})
        status = str(getattr(write, "status", "") or "")
        has_provenance = bool(
            getattr(write, "depends_on", None)
            or data.get("source")
            or data.get("provenance")
            or data.get("path")
            or data.get("run_id")
        )
        payload = {
            "id": str(getattr(write, "id", "")),
            "status": status,
            "summary": str(getattr(write, "summary", "") or ""),
            "has_provenance": has_provenance,
        }
        if not has_provenance:
            issues.append(f"memory_without_provenance:{payload['id']}")
        if "conflict_trace" in data or status in {"disputed", "candidate", "demoted"}:
            disputed.append({**payload, "conflict_trace": data.get("conflict_trace")})
        else:
            eligible.append(payload)
    for violation in list(getattr(ledger, "policy_violations", []) or []):
        if str(violation.get("recovery_class") or "") == "memory_conflict":
            issues.append("memory_conflict_policy_violation")
    return {
        "version": "memory-immune-v1",
        "write_count": len(writes),
        "eligible_writes": eligible,
        "disputed_writes": disputed,
        "issues": list(dict.fromkeys(issues)),
        "safe_to_commit": not issues,
    }


def build_experience_graph(ledger: Any) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    objective = str(getattr(ledger, "objective", "") or "")
    nodes.append({"id": "objective", "kind": "objective", "label": objective[:120]})
    for step in list(getattr(ledger, "plan_steps", []) or []):
        if not isinstance(step, dict):
            continue
        node_id = str(step.get("id") or f"step_{len(nodes)}")
        nodes.append({
            "id": node_id,
            "kind": "plan_step",
            "label": str(step.get("description") or step.get("task") or node_id)[:160],
            "status": str(step.get("status") or "pending"),
        })
        edges.append({"from": "objective", "to": node_id, "kind": "plans"})
    for call in list(getattr(ledger, "tool_calls", []) or []):
        call_id = str(getattr(call, "id", "") or "")
        nodes.append({
            "id": call_id,
            "kind": "tool_call",
            "label": str(getattr(call, "tool_name", "") or ""),
            "status": str(getattr(call, "status", "") or "pending"),
        })
        edges.append({"from": "objective", "to": call_id, "kind": "acts"})
    for result in list(getattr(ledger, "tool_results", []) or []):
        result_id = str(getattr(result, "id", "") or "")
        call_id = str(getattr(result, "tool_call_id", "") or "")
        nodes.append({
            "id": result_id,
            "kind": "tool_result",
            "label": str(getattr(result, "summary", "") or "")[:160],
            "status": str(getattr(result, "status", "") or ""),
            "success": bool(getattr(result, "success", False)),
        })
        if call_id:
            edges.append({"from": call_id, "to": result_id, "kind": "observes"})
    for evidence in list(getattr(ledger, "evidence_items", []) or []):
        evidence_id = str(getattr(evidence, "id", "") or "")
        nodes.append({
            "id": evidence_id,
            "kind": "evidence",
            "label": str(getattr(evidence, "summary", "") or "")[:160],
            "status": str(getattr(evidence, "status", "") or ""),
        })
        for dep in list(getattr(evidence, "depends_on", []) or []):
            edges.append({"from": str(dep), "to": evidence_id, "kind": "supports"})
    return {
        "version": "experience-graph-v1",
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "successful_tools": sum(
                1 for result in list(getattr(ledger, "tool_results", []) or [])
                if getattr(result, "success", False)
            ),
            "failed_tools": sum(
                1 for result in list(getattr(ledger, "tool_results", []) or [])
                if not getattr(result, "success", False)
            ),
        },
    }


def build_proxy_state_eval(ledger: Any, *, response_text: str = "") -> dict[str, Any]:
    gate = _completion_gate(ledger)
    support = {}
    if response_text and hasattr(ledger, "response_support_check"):
        support = ledger.response_support_check(response_text)
    issues = [str(item) for item in gate.get("missing_slots") or []]
    issues.extend(str(item) for item in support.get("issues") or [])
    drift_count = len(list(getattr(ledger, "policy_violations", []) or []))
    if drift_count:
        issues.append(f"policy_violations:{drift_count}")
    return {
        "version": "proxy-state-eval-v1",
        "passed": not issues,
        "issues": list(dict.fromkeys(issues)),
        "metrics": {
            "entity_preservation": 1.0 if not any("entity" in issue for issue in issues) else 0.0,
            "tool_truthfulness": 1.0 if not support.get("issues") else 0.0,
            "evidence_complete": 1.0 if gate.get("final_answer_allowed", True) else 0.0,
            "drift_count": drift_count,
            "memory_safe": 1.0,
        },
        "completion_gate": gate,
        "response_support": support,
    }


def propose_micro_agent_jobs(
    ledger: Any,
    phase: str,
    *,
    memory_report: dict[str, Any],
    proxy_eval: dict[str, Any],
) -> list[MicroAgentJob]:
    jobs: list[MicroAgentJob] = []
    gate = _completion_gate(ledger)
    if not gate.get("final_answer_allowed", True):
        jobs.append(MicroAgentJob(
            id="gap_classifier",
            label="Classify missing runtime slots",
            input_kind="completion_gate",
            output_schema={"type": "object", "required": ["missing_slots", "next_action"]},
            max_tokens=160,
            reason="Completion gate is blocked.",
        ))
    if memory_report.get("write_count") or phase == "memory_commit":
        jobs.append(MicroAgentJob(
            id="memory_eligibility",
            label="Check memory write eligibility",
            input_kind="memory_candidates",
            output_schema={"type": "object", "required": ["eligible", "reason", "provenance"]},
            max_tokens=180,
            reason="Memory writes require provenance and conflict checks.",
        ))
    if not proxy_eval.get("passed", True):
        jobs.append(MicroAgentJob(
            id="verifier_precheck",
            label="Precheck response against proxy state",
            input_kind="ledger_eval",
            output_schema={"type": "object", "required": ["supported", "issues"]},
            max_tokens=220,
            reason="Proxy state has unresolved issues.",
        ))
    if list(getattr(ledger, "tool_results", []) or []):
        jobs.append(MicroAgentJob(
            id="evidence_summarizer",
            label="Compress tool evidence",
            input_kind="successful_tool_results",
            output_schema={"type": "object", "required": ["evidence_ids", "summary"]},
            max_tokens=240,
            reason="Successful tool results can be summarized before response.",
        ))
    return jobs


def build_ui_run_map(
    ledger: Any,
    *,
    prompt_contract: PromptContract,
    proxy_eval: dict[str, Any],
) -> dict[str, Any]:
    sections = [
        {
            "id": "objective",
            "label": "Objective",
            "status": "active" if getattr(ledger, "objective", "") else "not_recorded",
            "count": 1 if getattr(ledger, "objective", "") else 0,
        },
        {
            "id": "plan",
            "label": "Plan",
            "status": "active" if list(getattr(ledger, "plan_steps", []) or []) else "not_recorded",
            "count": len(list(getattr(ledger, "plan_steps", []) or [])),
        },
        {
            "id": "tools",
            "label": "Tools",
            "status": "active" if list(getattr(ledger, "tool_calls", []) or []) else "not_recorded",
            "count": len(list(getattr(ledger, "tool_calls", []) or [])),
        },
        {
            "id": "evidence",
            "label": "Evidence",
            "status": "active" if list(getattr(ledger, "evidence_items", []) or []) else "not_recorded",
            "count": len(list(getattr(ledger, "evidence_items", []) or [])),
        },
        {
            "id": "memory",
            "label": "Memory",
            "status": "active" if list(getattr(ledger, "memory_writes", []) or []) else "not_recorded",
            "count": len(list(getattr(ledger, "memory_writes", []) or [])),
        },
        {
            "id": "verification",
            "label": "Verification",
            "status": "passed" if proxy_eval.get("passed") else "blocked",
            "count": len(proxy_eval.get("issues") or []),
        },
        {
            "id": "response",
            "label": "Response",
            "status": "allowed" if prompt_contract.final_answer_allowed else "blocked",
            "count": 1 if prompt_contract.final_answer_allowed else 0,
        },
    ]
    return {
        "version": "agent-native-run-map-v1",
        "sections": sections,
        "next_focus": _next_focus(sections),
    }


def _next_focus(sections: list[dict[str, Any]]) -> str:
    for section in sections:
        if section.get("status") in {"blocked", "not_recorded"}:
            return str(section.get("id") or "")
    return "response"


def _role_for_phase(phase: str) -> str:
    if phase in {"planning", "plan"}:
        return "planner"
    if phase in {"acting", "act"}:
        return "actor"
    if phase in {"verification", "verify"}:
        return "verifier"
    if phase == "memory_commit":
        return "memory"
    return "observer"


def _locked_entity_values(ledger: Any) -> list[str]:
    return [
        str(item.get("value") or "").strip()
        for item in list(getattr(ledger, "locked_entities", []) or [])
        if isinstance(item, dict) and str(item.get("value") or "").strip()
    ]


def _completion_gate(ledger: Any) -> dict[str, Any]:
    if hasattr(ledger, "completion_gate"):
        return dict(ledger.completion_gate())
    return {"final_answer_allowed": True, "missing_slots": [], "reason": "not_recorded"}


def _next_required_action(ledger: Any) -> dict[str, Any]:
    if hasattr(ledger, "next_required_action"):
        return dict(ledger.next_required_action())
    return {"tool": "", "arguments": {}, "reason": "not_recorded", "missing_slots": []}


def _attention_snapshot(ledger: Any, phase: str) -> dict[str, Any]:
    if hasattr(ledger, "attention_for_phase"):
        return ledger.attention_for_phase(phase).to_dict()
    return {"version": "not_recorded", "phase": phase, "items": [], "omitted_count": 0, "policy": {}}


def _evidence_refs(ledger: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for evidence in list(getattr(ledger, "evidence_items", []) or []):
        refs.append({
            "id": str(getattr(evidence, "id", "") or ""),
            "source": str(getattr(evidence, "source", "") or ""),
            "summary": str(getattr(evidence, "summary", "") or ""),
            "raw_ref": str(getattr(evidence, "raw_ref", "") or ""),
        })
    return refs
