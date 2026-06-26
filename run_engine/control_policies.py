from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from run_engine.workflow_contracts import WorkflowContract, classify_workflow_shape


CONTROL_POLICY_VERSION = "control-policy-v1"


@dataclass(frozen=True)
class ControlPolicy:
    id: str
    version: str
    label: str
    loop_shape: str
    reason: str
    planner_required: bool
    observe_after_each_action: bool
    mutable_plan: bool
    max_recovery_rounds: int
    prompt_contract: str
    risk_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [
            "Control Policy:",
            f"- id: {self.id}",
            f"- loop shape: {self.loop_shape}",
            f"- reason: {self.reason}",
            f"- planner required: {self.planner_required}",
            f"- observe after each action: {self.observe_after_each_action}",
            f"- mutable plan: {self.mutable_plan}",
            f"- max recovery rounds: {self.max_recovery_rounds}",
            f"- prompt contract: {self.prompt_contract}",
        ]
        if self.risk_notes:
            lines.append("Risk notes:")
            lines.extend(f"- {note}" for note in self.risk_notes[:4])
        return "\n".join(lines)


def resolve_control_policy(
    objective: str,
    *,
    requested: str = "auto",
    workflow_contract: WorkflowContract | None = None,
    risk_policy: str = "standard",
    allowed_tools: list[str] | tuple[str, ...] | None = None,
) -> ControlPolicy:
    requested_id = _normalize_requested(requested)
    if requested_id != "auto":
        return _policy_by_id(requested_id, reason=f"explicit request: {requested_id}")

    shape = (
        workflow_contract.workflow_shape
        if workflow_contract is not None
        else classify_workflow_shape(objective)
    )
    text = str(objective or "").lower()
    tools = {str(tool or "") for tool in (allowed_tools or [])}
    risk = str(risk_policy or "standard").lower()
    time_sensitive_web = bool(
        {"web_search", "web_fetch"} & tools
        and any(term in text for term in ("today", "latest", "current", "price", "news", "search", "fetch", "url"))
    )

    if shape == "source_collection" or time_sensitive_web:
        return _policy_by_id(
            "plan_execute",
            reason="source/web workflow uses plan-then-execute to keep untrusted web content from redefining control flow",
        )
    if shape == "coding_change":
        return _policy_by_id(
            "graph_harness",
            reason="coding changes need inspect-plan-edit-test structure with bounded recovery",
        )
    if risk in {"strict", "high", "enterprise", "ledger_enforced"}:
        return _policy_by_id(
            "graph_harness",
            reason=f"risk policy '{risk}' prefers explicit graph/state gates",
        )
    if shape in {"simple_chat", "open_ended_chat"}:
        return _policy_by_id("react", reason="low-risk chat can use lightweight ReAct-style iteration")
    return _policy_by_id("plan_observe", reason=f"default for workflow shape '{shape}'")


def _normalize_requested(value: str) -> str:
    raw = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "managed": "auto",
        "single": "react",
        "paov": "plan_observe",
        "plan_act_observe": "plan_observe",
        "plan_then_execute": "plan_execute",
        "pte": "plan_execute",
        "sgh": "graph_harness",
        "structured_graph": "graph_harness",
        "structured_graph_harness": "graph_harness",
    }
    return aliases.get(raw, raw if raw in {"auto", "react", "plan_observe", "plan_execute", "graph_harness"} else "auto")


def _policy_by_id(policy_id: str, *, reason: str) -> ControlPolicy:
    if policy_id == "react":
        return ControlPolicy(
            id="react",
            version=CONTROL_POLICY_VERSION,
            label="ReAct",
            loop_shape="reason_act_observe",
            reason=reason,
            planner_required=False,
            observe_after_each_action=True,
            mutable_plan=True,
            max_recovery_rounds=2,
            prompt_contract="Actor can choose the next valid tool from current observation, but ledger verifies every claim.",
            risk_notes=["Best for bounded tasks and fast iteration.", "Can drift if untrusted observations redefine the goal."],
        )
    if policy_id == "plan_execute":
        return ControlPolicy(
            id="plan_execute",
            version=CONTROL_POLICY_VERSION,
            label="Plan Then Execute",
            loop_shape="plan_then_execute_with_state_eval",
            reason=reason,
            planner_required=True,
            observe_after_each_action=False,
            mutable_plan=False,
            max_recovery_rounds=1,
            prompt_contract="Plan is committed before untrusted tool content; observations fill slots but cannot redefine the task.",
            risk_notes=["Safer for web/source workflows.", "Needs typed tools and clear completion gates."],
        )
    if policy_id == "graph_harness":
        return ControlPolicy(
            id="graph_harness",
            version=CONTROL_POLICY_VERSION,
            label="Structured Graph Harness",
            loop_shape="typed_pass_graph",
            reason=reason,
            planner_required=True,
            observe_after_each_action=True,
            mutable_plan=False,
            max_recovery_rounds=1,
            prompt_contract="Run a pass graph with explicit node states, dependencies, and bounded recovery.",
            risk_notes=["Best for long-horizon or high-risk workflows.", "Less flexible than free-form ReAct."],
        )
    return ControlPolicy(
        id="plan_observe",
        version=CONTROL_POLICY_VERSION,
        label="Plan Act Observe",
        loop_shape="plan_act_observe_verify",
        reason=reason,
        planner_required=True,
        observe_after_each_action=True,
        mutable_plan=True,
        max_recovery_rounds=2,
        prompt_contract="Planner narrows tools, actor performs one action, observer decides continue or finalize.",
        risk_notes=["Current compatibility policy.", "Works broadly but can overreact to noisy observations."],
    )
