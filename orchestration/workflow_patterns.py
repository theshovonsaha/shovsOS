"""Reusable workflow meta-patterns.

These are compact contracts that can be injected into planning context and used
as deterministic fallback plan steps when a model omits a workflow spine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WorkflowPattern:
    id: str
    label: str
    purpose: str
    stages: tuple[str, ...]
    default_plan_steps: tuple[dict[str, Any], ...]
    response_contract: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stages"] = list(self.stages)
        data["default_plan_steps"] = [dict(step) for step in self.default_plan_steps]
        data["response_contract"] = list(self.response_contract)
        return data

    def render_for_prompt(self) -> str:
        lines = [
            f"Workflow Pattern: {self.label} ({self.id})",
            f"Purpose: {self.purpose}",
            "Stages:",
        ]
        lines.extend(f"- {stage}" for stage in self.stages)
        lines.append("Response contract:")
        lines.extend(f"- {item}" for item in self.response_contract)
        return "\n".join(lines)


DIVERGE_CONVERGE_PATCH_V1 = WorkflowPattern(
    id="diverge_converge_patch_v1",
    label="Diverge -> Verify -> Normalize -> Converge -> Patch",
    purpose=(
        "Use when the user needs a practical decision from many messy real-world options. "
        "Do not free-generate the answer first; gather compact facts, compare them, then patch the final response."
    ),
    stages=(
        "INTAKE: identify item, location, budget, constraints, preferred stores, and must-have qualities.",
        "DIVERGE: search across relevant stores/categories without over-broadening the user's actual need.",
        "VERIFY: fetch pages for candidate products; keep only URL-backed prices, ratings, and availability signals.",
        "NORMALIZE: convert page text into comparable fields: store, item, price, rating, quality signals, caveats, URL.",
        "CONVERGE: choose the best useful option and one backup; state why using normalized fields.",
        "PATCH: final answer is assembled from verified fields and explicit caveats, not invented prose.",
    ),
    default_plan_steps=(
        {
            "id": "intake_slots",
            "description": "Extract item, location, budget, priorities, and preferred stores from the user request.",
            "tool": None,
            "status": "pending",
            "risk": "read_only",
        },
        {
            "id": "diverge_store_candidates",
            "description": "Gather candidate products across relevant stores with location and budget hints.",
            "tool": "shopping_advice",
            "status": "pending",
            "risk": "read_only",
        },
        {
            "id": "verify_and_normalize",
            "description": "Use fetched product/store pages to normalize prices, ratings, tradeoffs, and verified URLs.",
            "tool": "shopping_advice",
            "status": "pending",
            "risk": "read_only",
        },
        {
            "id": "converge_recommendation",
            "description": "Pick the best verified option and a backup, or state what remains unverified.",
            "tool": None,
            "status": "pending",
            "risk": "read_only",
        },
        {
            "id": "patch_response",
            "description": "Generate the final answer only from the answer_patch fields and verified URLs.",
            "tool": None,
            "status": "pending",
            "risk": "read_only",
        },
    ),
    response_contract=(
        "Lead with the recommended store/product and why it fits the user's actual need.",
        "Include price/rating only when extracted from verified candidate fields.",
        "Include verified URLs, but do not dump a link list.",
        "Say when local availability, exact price, or quality could not be verified.",
        "Keep the answer compact; no generic shopping lecture.",
    ),
)


WORKFLOW_PATTERNS: dict[str, WorkflowPattern] = {
    DIVERGE_CONVERGE_PATCH_V1.id: DIVERGE_CONVERGE_PATCH_V1,
}


def get_workflow_pattern(pattern_id: str | None) -> WorkflowPattern | None:
    key = str(pattern_id or "").strip()
    if not key:
        return None
    return WORKFLOW_PATTERNS.get(key)
