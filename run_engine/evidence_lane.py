from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from run_engine.tool_contract import canonical_tool_result, clip_text, format_tool_result_line


@dataclass(frozen=True)
class WorkingEvidenceSnapshot:
    objective: str
    exact_targets: tuple[str, ...]
    selected: tuple[dict[str, Any], ...]

    @property
    def exact_match_count(self) -> int:
        return sum(1 for item in self.selected if tool_result_matches_exact_target(item, list(self.exact_targets)))

    @property
    def substantive_count(self) -> int:
        return sum(1 for item in self.selected if is_substantive_tool_result(item))


def extract_exact_query_targets(user_message: str) -> list[str]:
    text = str(user_message or "").lower()
    targets = re.findall(r"\b[a-z0-9][a-z0-9.-]*\.[a-z]{2,}\b", text)
    seen: set[str] = set()
    ordered: list[str] = []
    for target in targets:
        normalized = target.strip().lower()
        if normalized and normalized not in seen:
            ordered.append(normalized)
            seen.add(normalized)
    return ordered


def tool_result_matches_exact_target(item: dict[str, Any], exact_targets: list[str]) -> bool:
    if not exact_targets:
        return False
    haystacks = [
        str(item.get("content") or "").lower(),
        json.dumps(item.get("arguments") or {}, ensure_ascii=False).lower(),
    ]
    return any(target in haystack for target in exact_targets for haystack in haystacks)


def is_substantive_tool_result(item: dict[str, Any]) -> bool:
    tool_name = str(item.get("tool_name") or "")
    return tool_name not in {"todo_write", "todo_update", "query_memory", "store_memory"}


def tool_kind_priority(tool_name: str) -> int:
    priority = {
        "web_fetch": 0,
        "file_view": 1,
        "web_search": 2,
        "image_search": 3,
        "query_memory": 5,
        "todo_update": 6,
        "todo_write": 7,
        "store_memory": 8,
    }
    return priority.get(tool_name, 4)


def select_working_evidence(
    tool_results: list[dict[str, Any]],
    *,
    user_message: str,
    max_results: int = 4,
) -> list[dict[str, Any]]:
    if not tool_results:
        return []

    exact_targets = extract_exact_query_targets(user_message)
    scored: list[tuple[tuple[int, int, int, int], int, dict[str, Any]]] = []
    for idx, item in enumerate(tool_results):
        tool_name = str(item.get("tool_name") or "unknown")
        success = bool(item.get("success"))
        substantive = is_substantive_tool_result(item)
        exact_match = tool_result_matches_exact_target(item, exact_targets)
        score = (
            0 if success else 1,
            0 if substantive else 1,
            0 if exact_match else 1,
            tool_kind_priority(tool_name),
        )
        scored.append((score, idx, item))

    scored.sort(key=lambda row: (row[0], -row[1]))

    selected: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, str]] = set()
    for _, _, item in scored:
        tool_name = str(item.get("tool_name") or "unknown")
        preview = clip_text(str(item.get("content") or "").strip(), 180)
        signature = (tool_name, preview[:180])
        if signature in seen_signatures:
            continue
        selected.append(item)
        seen_signatures.add(signature)
        if len(selected) >= max_results:
            break

    if any(bool(item.get("success")) for item in selected) and len(selected) < max_results:
        latest_failure = next(
            (item for item in reversed(tool_results) if not bool(item.get("success"))),
            None,
        )
        if latest_failure:
            tool_name = str(latest_failure.get("tool_name") or "unknown")
            preview = clip_text(str(latest_failure.get("content") or "").strip(), 180)
            signature = (tool_name, preview[:180])
            if signature not in seen_signatures:
                selected.append(latest_failure)

    return selected


def build_working_evidence_snapshot(
    tool_results: list[dict[str, Any]],
    *,
    user_message: str,
    max_results: int = 4,
) -> WorkingEvidenceSnapshot:
    objective = str(user_message or "").strip()
    exact_targets = tuple(extract_exact_query_targets(objective))
    selected = tuple(
        select_working_evidence(
            tool_results,
            user_message=objective,
            max_results=max_results,
        )
    )
    return WorkingEvidenceSnapshot(
        objective=objective,
        exact_targets=exact_targets,
        selected=selected,
    )


def build_working_evidence_block(
    tool_results: list[dict[str, Any]],
    *,
    user_message: str,
    max_results: int = 3,
    preview_chars: int = 180,
) -> str:
    snapshot = build_working_evidence_snapshot(
        tool_results,
        user_message=user_message,
        max_results=max_results,
    )
    if not snapshot.selected:
        return ""

    lines = [
        "Curated evidence most relevant to the current objective. Prefer these results over noisier or administrative tool outputs."
    ]
    for item in snapshot.selected:
        summary = canonical_tool_result(item, preview_chars=preview_chars)
        tags: list[str] = []
        if tool_result_matches_exact_target(item, list(snapshot.exact_targets)):
            tags.append("exact-target")
        if is_substantive_tool_result(item):
            tags.append("substantive")
        elif str(summary["tool_name"]) in {"query_memory", "todo_write", "todo_update", "store_memory"}:
            tags.append("administrative")
        label = f"- {summary['tool_name']} [{summary['status']}]"
        if tags:
            label += f" ({', '.join(tags)})"
        lines.append(f"{label}: {summary['preview']}")
    return "\n".join(lines)


def build_evidence_priority_reminder(user_message: str, tool_results: list[dict[str, Any]]) -> str:
    snapshot = build_working_evidence_snapshot(
        tool_results,
        user_message=user_message,
        max_results=4,
    )
    if any(
        str(item.get("tool_name") or "") == "web_fetch"
        and bool(item.get("success"))
        and tool_result_matches_exact_target(item, list(snapshot.exact_targets))
        for item in snapshot.selected
    ):
        return (
            "Prioritize verified exact-domain fetch evidence over noisier search results when deciding what is real, active, or trustworthy."
        )
    return ""


def build_evidence_focus_lines(
    user_message: str,
    tool_results: list[dict[str, Any]],
    *,
    max_results: int = 2,
    preview_chars: int = 140,
) -> list[str]:
    snapshot = build_working_evidence_snapshot(
        tool_results,
        user_message=user_message,
        max_results=max_results + 2,
    )
    if not snapshot.selected:
        return []
    substantive = [item for item in snapshot.selected if is_substantive_tool_result(item)]
    focus_items = substantive[:max_results] if substantive else list(snapshot.selected[:max_results])
    return [
        format_tool_result_line(item, preview_chars=preview_chars, include_status_label=True)
        for item in focus_items
    ]
