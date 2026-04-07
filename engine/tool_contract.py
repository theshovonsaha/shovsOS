from __future__ import annotations

import json
from typing import Any


TOOL_ARG_PRIORITY = [
    "query",
    "url",
    "path",
    "title",
    "filename",
    "language",
    "command",
    "prompt",
]
CONTENT_SIZE_SUMMARY_KEYS = ["code", "html", "content", "svg", "script", "css", "markup"]


def clip_text(text: str, max_chars: int) -> str:
    text = str(text or "")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def format_tool_argument_value(key: str, value: Any) -> str:
    if isinstance(value, str):
        normalized = " ".join(value.split()).strip()
        if not normalized:
            return "empty"
        lowered_key = key.lower()
        if any(token in lowered_key for token in CONTENT_SIZE_SUMMARY_KEYS):
            return f"{len(normalized)} chars"
        return clip_text(normalized, 72)
    if isinstance(value, list):
        return f"{len(value)} item{'' if len(value) == 1 else 's'}"
    if isinstance(value, dict):
        return f"{len(value)} field{'' if len(value) == 1 else 's'}"
    return str(value)


def summarize_arguments(arguments: dict[str, Any] | None = None, *, max_items: int = 3) -> str:
    entries = list((arguments or {}).items())
    if not entries:
        return "starting..."
    sorted_entries = sorted(
        entries,
        key=lambda item: (
            TOOL_ARG_PRIORITY.index(item[0]) if item[0] in TOOL_ARG_PRIORITY else len(TOOL_ARG_PRIORITY),
            item[0],
        ),
    )
    displayed = [
        f"{key}: {format_tool_argument_value(key, value)}"
        for key, value in sorted_entries[:max_items]
    ]
    remaining = len(sorted_entries) - len(displayed)
    if remaining > 0:
        return f"{' · '.join(displayed)} · +{remaining} more"
    return " · ".join(displayed)


def canonical_tool_call(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    max_items: int = 3,
) -> dict[str, Any]:
    args = dict(arguments or {})
    normalized_name = str(tool_name or "unknown")
    return {
        "tool": normalized_name,
        "tool_name": normalized_name,
        "arguments": args,
        "arguments_summary": summarize_arguments(args, max_items=max_items),
    }


def tool_call_signature(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
    try:
        return json.dumps(
            {
                "tool": str(tool_name or ""),
                "arguments": arguments or {},
            },
            sort_keys=True,
            default=str,
        )
    except Exception:
        return f"{tool_name}:{repr(arguments)}"


def is_retry_sensitive_tool(tool_name: str) -> bool:
    return str(tool_name or "").lower() in {"web_search", "web_fetch"}


def canonical_tool_result(
    item: dict[str, Any],
    *,
    preview_chars: int = 220,
) -> dict[str, Any]:
    tool_name = str(item.get("tool_name") or item.get("tool") or "unknown")
    success = bool(item.get("success"))
    content = str(item.get("content") or item.get("preview") or "")
    arguments = item.get("arguments")
    normalized: dict[str, Any] = {
        "tool": tool_name,
        "tool_name": tool_name,
        "success": success,
        "status": "ok" if success else "failed",
        "preview": clip_text(content.strip(), preview_chars),
    }
    if isinstance(arguments, dict) and arguments:
        normalized["arguments"] = dict(arguments)
        normalized["arguments_summary"] = summarize_arguments(arguments)
    return normalized


def format_tool_result_line(
    item: dict[str, Any],
    *,
    preview_chars: int = 220,
    include_status_label: bool = False,
) -> str:
    summary = canonical_tool_result(item, preview_chars=preview_chars)
    if include_status_label:
        return f"- {summary['tool_name']} [{summary['status']}]: {summary['preview']}"
    return (
        f"- {summary['tool_name']}: success={str(summary['success']).lower()} "
        f"preview={summary['preview']}"
    )


def summarize_tool_results(
    tool_results: list[dict[str, Any]],
    *,
    limit: int = 4,
    preview_chars: int = 220,
) -> list[dict[str, Any]]:
    items = list(tool_results or [])[-max(limit, 0) :]
    return [canonical_tool_result(item, preview_chars=preview_chars) for item in items]