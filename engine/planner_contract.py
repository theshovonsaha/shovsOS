"""
Planner Contract
----------------
The planner is the entry point to the whole phase-aware architecture.
When it silently fails and the runtime "continues with direct reasoning,"
every downstream guarantee (phase packets, meta-context, evidence lanes)
is bypassed — you are running a raw LLM with a context blob.

This module makes the planner boundary robust in three ways:

1. TIERED PROMPTS — small local models cannot reliably emit the complex
   plan JSON. They get a stripped-down contract. Frontier models get the
   full one.
2. TOOL PRE-FILTERING — 27 tools overwhelm a small model's selection.
   We filter to the handful relevant to the objective before the planner
   ever sees them.
3. LOUD FAILURE — when the planner produces nothing actionable, we return
   a typed PlannerOutcome with status=FAILED instead of silently degrading.
   The caller decides: retry simplified, surface to user, or stop. The one
   thing it must not do is pretend a plan exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# Models that cannot be trusted with the full structured-plan contract.
# Substring match against the model string (provider prefix tolerated).
_SMALL_MODEL_MARKERS = (
    "gemma", "llama3.2", "llama3:8b", "qwen2.5-coder-3b", "qwen2.5:3b",
    "phi3", "phi-3", "tinyllama", "1b", "3b", "mini", "small",
)

# Objective keyword → tool relevance. Used to pre-filter the registry so the
# planner sees a focused menu instead of all 27 tools.
_TOOL_RELEVANCE: dict[str, tuple[str, ...]] = {
    "fetch": ("web_fetch",),
    "url": ("web_fetch",),
    "http": ("web_fetch",),
    "website": ("web_fetch", "web_search"),
    "page": ("web_fetch",),
    "search": ("web_search",),
    "news": ("web_search",),
    "latest": ("web_search",),
    "find": ("web_search",),
    "image": ("image_search",),
    "picture": ("image_search",),
    "photo": ("image_search",),
    "file": ("file_create", "file_view", "bash"),
    "code": ("file_create", "bash", "file_str_replace"),
    "write": ("file_create",),
    "create": ("file_create",),
    "save": ("file_create", "shovs_memory_store"),
    "run": ("bash",),
    "script": ("bash", "file_create"),
    "remember": ("query_memory", "shovs_memory_query"),
    "recall": ("query_memory", "shovs_memory_query"),
    "earlier": ("query_memory",),
    "memory": ("query_memory", "shovs_memory_query", "shovs_memory_store"),
    "weather": ("weather_fetch",),
}

# Always offered as a floor so the planner is never toolless on a research turn.
_FLOOR_TOOLS = ("web_search",)

_MAX_PLANNER_TOOLS = 6


class PlannerStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    EMPTY_OBJECTIVE = "empty_objective"


@dataclass(frozen=True)
class PlannerOutcome:
    status: PlannerStatus
    route: str = ""
    tools: tuple[str, ...] = ()
    strategy: str = ""
    argument_clues: dict[str, str] = field(default_factory=dict)
    raw: str = ""
    reason: str = ""

    @property
    def actionable(self) -> bool:
        return self.status == PlannerStatus.OK and bool(self.tools)


def is_small_model(model: str) -> bool:
    m = (model or "").lower()
    return any(marker in m for marker in _SMALL_MODEL_MARKERS)


def filter_tools_for_planner(
    all_tools: list[str],
    objective: str,
    *,
    max_tools: int = _MAX_PLANNER_TOOLS,
) -> list[str]:
    """Reduce the full tool list to the most relevant for this objective.

    Small models choke on long tool menus; even frontier models plan more
    cleanly with a focused set. Order: relevance hits first, floor tools,
    then fill from the remainder up to max_tools.
    """
    obj = (objective or "").lower()
    available = {str(t).strip() for t in all_tools if str(t).strip()}

    picked: list[str] = []

    def _add(name: str) -> None:
        if name in available and name not in picked:
            picked.append(name)

    for keyword, tools in _TOOL_RELEVANCE.items():
        if keyword in obj:
            for tool in tools:
                _add(tool)

    for tool in _FLOOR_TOOLS:
        _add(tool)

    # Fill remaining slots deterministically (sorted for reproducible plans).
    for tool in sorted(available):
        if len(picked) >= max_tools:
            break
        _add(tool)

    return picked[:max_tools]


def build_planner_prompt(
    *,
    model: str,
    objective: str,
    tools: list[str],
    full_prompt_template: Optional[str] = None,
) -> str:
    """Return a planner prompt sized to the model's reliability.

    Small models get a minimal JSON-only contract. Frontier models get the
    caller's full template (or a solid default if none supplied).
    """
    tool_list = ", ".join(tools) if tools else "web_search"

    if is_small_model(model):
        return (
            "Output ONLY one JSON object. No markdown, no prose, no explanation.\n\n"
            f"Task: {objective}\n"
            f"Available tools (choose 1-3 that directly serve the task): {tool_list}\n\n"
            "Required exact shape:\n"
            '{"route": "tool_loop", "tools": ["<tool>"], '
            '"strategy": "<one short sentence>", '
            '"argument_clues": {"<tool>": "<exact argument value>"}}\n'
        )

    if full_prompt_template:
        return full_prompt_template.format(objective=objective, tool_list=tool_list)

    return (
        "You are the planning phase of an agentic runtime. Produce a single JSON "
        "object describing the plan for this objective. Do not include any text "
        "outside the JSON.\n\n"
        f"Objective:\n{objective}\n\n"
        f"Available tools: {tool_list}\n\n"
        "Schema:\n"
        "{\n"
        '  "route": "tool_loop" | "direct_answer",\n'
        '  "tools": [ordered tool names you intend to call],\n'
        '  "strategy": "one or two sentences on approach",\n'
        '  "argument_clues": {"tool_name": "concrete starting argument"}\n'
        "}\n\n"
        "Choose direct_answer with empty tools ONLY if the objective is fully "
        "answerable from context with no external lookup."
    )


def parse_planner_output(raw: str) -> PlannerOutcome:
    """Parse planner model output into a typed outcome.

    Tolerant of: clean JSON, JSON wrapped in markdown fences, JSON with
    leading/trailing prose. If nothing parseable is found, returns FAILED —
    never a silent empty success.
    """
    text = str(raw or "").strip()
    if not text:
        return PlannerOutcome(status=PlannerStatus.FAILED, raw=raw, reason="empty_output")

    payload = _extract_json_object(text)
    if payload is None:
        return PlannerOutcome(status=PlannerStatus.FAILED, raw=raw, reason="no_json_object")

    route = str(payload.get("route") or "").strip() or "tool_loop"
    tools_raw = payload.get("tools") or []
    if isinstance(tools_raw, str):
        tools_raw = [tools_raw]
    tools = tuple(str(t).strip() for t in tools_raw if str(t).strip())

    strategy = str(payload.get("strategy") or "").strip()
    clues_raw = payload.get("argument_clues") or {}
    argument_clues = {
        str(k).strip(): str(v).strip()
        for k, v in clues_raw.items()
        if isinstance(clues_raw, dict) and str(k).strip()
    }

    # A direct_answer route with no tools is a VALID plan (answer from context).
    if route == "direct_answer":
        return PlannerOutcome(
            status=PlannerStatus.OK,
            route=route,
            tools=(),
            strategy=strategy or "Answer directly from available context.",
            argument_clues=argument_clues,
            raw=raw,
            reason="direct_answer",
        )

    if not tools:
        return PlannerOutcome(
            status=PlannerStatus.FAILED,
            route=route,
            strategy=strategy,
            raw=raw,
            reason="tool_loop_without_tools",
        )

    return PlannerOutcome(
        status=PlannerStatus.OK,
        route=route,
        tools=tools,
        strategy=strategy,
        argument_clues=argument_clues,
        raw=raw,
        reason="parsed",
    )


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    # 1) direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) strip markdown fences
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    if fenced != text:
        try:
            obj = json.loads(fenced)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) first balanced {...} span
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def planner_failure_event(outcome: PlannerOutcome, *, model: str, objective: str) -> dict[str, Any]:
    """Build the loud, structured failure event the runtime should emit
    instead of silently falling back. This is what makes the failure
    visible in traces and to the operator."""
    return {
        "type": "planner_failure",
        "code": "PLANNER_FAIL",
        "model": model,
        "objective_preview": (objective or "")[:160],
        "reason": outcome.reason,
        "raw_preview": (outcome.raw or "")[:300],
        "message": (
            "Planner produced no actionable plan. Runtime stopped before "
            "acting rather than silently degrading to direct reasoning. "
            "Retry with a simpler objective or a larger planner model."
        ),
    }