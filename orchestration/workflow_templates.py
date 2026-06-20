"""Workflow template registry for agent creation.

Templates are declarative defaults, not a second runtime. They give the UI and
profile store one shared source of truth for tools, prompt version, risk policy,
and ledger rollout mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from orchestration.workflow_patterns import get_workflow_pattern


@dataclass(frozen=True)
class WorkflowTemplate:
    id: str
    label: str
    description: str
    tools: tuple[str, ...]
    system_prompt: str
    default_use_planner: bool = True
    default_loop_mode: str = "managed"
    default_context_mode: str = "v3"
    risk_policy: str = "standard"
    prompt_version: str = "role_contracts_v1"
    ledger_mode: str = "shadow"
    workflow_pattern: str = ""
    ui_label: str = ""
    test_scenarios: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tools"] = list(self.tools)
        data["test_scenarios"] = list(self.test_scenarios)
        data["ui_label"] = self.ui_label or self.label
        pattern = get_workflow_pattern(self.workflow_pattern)
        data["workflow_pattern_detail"] = pattern.to_dict() if pattern else None
        return data


WORKFLOW_TEMPLATES: dict[str, WorkflowTemplate] = {
    "general_operator_v1": WorkflowTemplate(
        id="general_operator_v1",
        label="General Operator",
        description="Balanced assistant for chat, tools, memory, and monitored runtime execution.",
        tools=(
            "web_search",
            "web_fetch",
            "query_memory",
            "store_memory",
            "shovs_memory_query",
            "shovs_memory_store",
            "delegate_to_agent",
        ),
        system_prompt=(
            "You are Shovs, a clear runtime operator. Use tools only when they materially "
            "improve accuracy or complete a concrete task. Never claim a tool ran unless "
            "a successful ledger result exists."
        ),
        risk_policy="standard",
        test_scenarios=("tool_hallucination", "ambiguous_followup", "memory_candidate"),
    ),
    "research_agent_v1": WorkflowTemplate(
        id="research_agent_v1",
        label="Research Agent",
        description="Evidence-first web and memory research with source chaining and verification.",
        tools=("web_search", "web_fetch", "query_memory", "store_memory", "shovs_memory_query", "shovs_memory_store"),
        system_prompt=(
            "Act as a rigorous research agent. Prefer primary sources, preserve exact URLs, "
            "follow READ_MORE with a precise fetch, and distinguish evidence from inference."
        ),
        risk_policy="evidence_first",
        test_scenarios=("search_fetch_chain", "prompt_injection_tool_result", "unsupported_claim"),
    ),
    "shopping_advisor_v1": WorkflowTemplate(
        id="shopping_advisor_v1",
        label="Shopping Advisor",
        description="Consumer buyer assistant that verifies product pages before recommending.",
        tools=("shopping_advice", "web_search", "web_fetch", "query_memory", "store_memory"),
        system_prompt=(
            "Act as a calm consumer shopping advisor. Use shopping_advice for buying questions. "
            "Ask for or infer location when local stores matter. Compare relevant stores like Costco, Canadian Tire, "
            "Shoppers, Metro, Dollarama, Walmart, and Best Buy when useful. Recommend from verified URLs and extracted "
            "price/rating signals. Keep output short, useful, and explicit about what was not verified. Never invent "
            "prices, purchases, availability, or links."
        ),
        risk_policy="consumer_verified",
        prompt_version="shopping_patch_v1",
        workflow_pattern="diverge_converge_patch_v1",
        test_scenarios=("shopping_price_patch", "unverified_store_page", "failed_fetch_disclosed"),
    ),
    "coding_agent_v1": WorkflowTemplate(
        id="coding_agent_v1",
        label="Coding Agent",
        description="Workspace-grounded implementation agent focused on narrow edits and tests.",
        tools=("file_view", "file_create", "file_str_replace", "bash", "query_memory", "store_memory"),
        system_prompt=(
            "Act as a disciplined coding operator. Inspect local files first, preserve project "
            "conventions, keep changes scoped, and verify with relevant tests."
        ),
        risk_policy="change_controlled",
        test_scenarios=("plan_resume_after_stop", "failed_tool_reported", "contract_preservation"),
    ),
    "memory_agent_v1": WorkflowTemplate(
        id="memory_agent_v1",
        label="Memory Agent",
        description="Memory hygiene agent for facts, candidates, contradiction handling, and provenance.",
        tools=("query_memory", "store_memory", "shovs_memory_query", "shovs_memory_store", "shovs_list_loci"),
        system_prompt=(
            "Act as a memory steward. Commit only user-stated facts, verified evidence, or "
            "explicit candidates with provenance. Surface contradictions before storing."
        ),
        risk_policy="memory_strict",
        test_scenarios=("void_then_add", "profile_correction", "candidate_demoted"),
    ),
    "verifier_agent_v1": WorkflowTemplate(
        id="verifier_agent_v1",
        label="Verifier Agent",
        description="Ledger-grounded verifier that checks claims against successful results and evidence IDs.",
        tools=("query_memory", "web_fetch"),
        system_prompt=(
            "Act as a verifier. Check response claims against ledger evidence IDs and successful "
            "tool results. Mark unsupported claims instead of repairing them silently."
        ),
        default_use_planner=False,
        default_loop_mode="single",
        risk_policy="strict_verification",
        test_scenarios=("unsupported_response_claim", "orphaned_tool_result", "missing_required_slot"),
    ),
}


def get_workflow_template(template_id: str | None) -> WorkflowTemplate:
    key = str(template_id or "").strip() or "general_operator_v1"
    return WORKFLOW_TEMPLATES.get(key, WORKFLOW_TEMPLATES["general_operator_v1"])


def list_workflow_templates() -> list[dict[str, Any]]:
    return [template.to_dict() for template in WORKFLOW_TEMPLATES.values()]
