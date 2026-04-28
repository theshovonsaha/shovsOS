from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from engine.tool_contract import format_tool_result_line, summarize_arguments
from plugins.tool_registry import ToolCall, ToolRegistry


def build_actor_request_content(
    *,
    user_message: str,
    effective_objective: str,
    session_first_message: str,
    allowed_tools: list[str],
    tool_results: list[dict[str, Any]],
    context_block: str,
    clip_text,
    argument_clues: Optional[dict[str, str]] = None,
) -> str:
    recent_results = "\n".join(
        format_tool_result_line(item, preview_chars=200)
        for item in tool_results[-4:]
    ) or "none"
    objective_block = f"Current user turn:\n{user_message}\n\n"
    if effective_objective and effective_objective.strip() != user_message.strip():
        objective_block += (
            "Resolved working objective:\n"
            f"{effective_objective}\n\n"
            "If the current user turn is brief confirmation or retry language, treat the resolved working objective as the operative instruction.\n\n"
        )

    # Planner argument clues — exact hints for what each tool should target.
    # Using these avoids the actor having to re-derive URLs, search terms, etc.
    clues_block = ""
    if argument_clues:
        relevant = {name: clue for name, clue in argument_clues.items() if name in allowed_tools and clue}
        if relevant:
            clue_lines = "\n".join(f"- {name}: {clue}" for name, clue in relevant.items())
            clues_block = f"Planner argument hints (use these as starting arguments for each tool):\n{clue_lines}\n\n"

    return (
        f"{objective_block}"
        "Decision policy:\n"
        "- Use the smallest sufficient next step.\n"
        "- If deterministic facts in context already answer the user, answer directly and do not call tools.\n"
        "- If one exact probe can close the gap, prefer that over broad multi-step exploration.\n"
        "- Do not repeat stale context or speculative candidate signals as if they were facts.\n"
        "- If planner argument hints are provided, use them as the exact argument values for those tools.\n\n"
        "The allowed tools below are available in this runtime right now. If a current-information request can be answered with an allowed tool, use that tool instead of claiming you lack access.\n\n"
        f"Allowed tools: {', '.join(allowed_tools)}\n\n"
        f"{clues_block}"
        f"Recent tool results:\n{recent_results}\n\n"
        f"Session first message: {session_first_message or 'none'}\n\n"
        f"Context block:\n{context_block or 'none'}"
    )


def extract_tool_call(raw: str, tool_registry: ToolRegistry) -> Optional[ToolCall]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    stripped = raw.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
            calls = payload.get("tool_calls") or []
            if calls:
                first = calls[0]
                function_block = first.get("function") if isinstance(first, dict) else None
                if isinstance(function_block, dict):
                    arguments = function_block.get("arguments") or {}
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except Exception:
                            arguments = {}
                    if isinstance(function_block.get("name"), str) and isinstance(arguments, dict):
                        return ToolCall(
                            tool_name=function_block["name"],
                            arguments=arguments,
                            raw_json=stripped,
                        )
    except Exception:
        pass

    direct = tool_registry.detect_tool_call(stripped)
    if direct is not None:
        return direct

    obj_match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if obj_match:
        direct = tool_registry.detect_tool_call(obj_match.group(0))
        if direct is not None:
            return direct
    return None


def fallback_tool_call(tool_name: str, user_message: str) -> Optional[ToolCall]:
    arguments: dict[str, Any]
    lowered = (user_message or "").strip()
    if tool_name in {"web_search", "rag_search"}:
        arguments = {"query": lowered}
    elif tool_name == "query_memory":
        arguments = {"topic": lowered}
    elif tool_name == "web_fetch":
        url_match = re.search(r"https?://\S+", lowered)
        if not url_match:
            return None
        arguments = {"url": url_match.group(0).rstrip('.,)')}
    elif tool_name == "weather_fetch":
        arguments = {"location": lowered}
    elif tool_name == "delegate_to_agent":
        arguments = {"task": lowered}
    else:
        return None
    return ToolCall(
        tool_name=tool_name,
        arguments=arguments,
        raw_json=json.dumps({"tool": tool_name, "arguments": arguments}),
    )
