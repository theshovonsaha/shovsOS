from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from engine.tool_contract import tool_call_signature


def _parse_tool_payload(content: str) -> dict[str, Any]:
    raw = str(content or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def is_null_bash_result(tool_name: str, success: bool, content: str) -> bool:
    if str(tool_name or "").strip().lower() != "bash" or not success:
        return False

    payload = _parse_tool_payload(content)
    if payload:
        if str(payload.get("type") or "") != "bash_result":
            return False
        verification = payload.get("verification") if isinstance(payload.get("verification"), dict) else {}
        if verification.get("existing_paths"):
            return False
        if verification.get("missing_paths"):
            return False
        output = str(payload.get("output") or "").strip()
        return output == ""

    return str(content or "").strip() == ""


@dataclass
class ToolLoopGuard:
    null_bash_counts: dict[str, int] = field(default_factory=dict)

    def observe_result(
        self,
        *,
        tool_name: str,
        arguments: Optional[dict[str, Any]],
        success: bool,
        content: str,
    ) -> Optional[dict[str, Any]]:
        if str(tool_name or "").strip().lower() != "bash":
            return None

        signature = tool_call_signature("bash", arguments or {})
        if is_null_bash_result(tool_name, success, content):
            count = self.null_bash_counts.get(signature, 0) + 1
            self.null_bash_counts[signature] = count
            if count >= 2:
                command = str((arguments or {}).get("command") or "").strip()
                return {
                    "type": "logical_stall_alert",
                    "tool_name": "bash",
                    "signature": signature,
                    "command": command,
                    "message": (
                        "Logical_Stall_Alert: the same bash command returned no observable output twice. "
                        "Pivot to a different inspection step or explain the missing physical evidence."
                    ),
                    "suggested_pivot": "Prefer file_view, pwd, ls, or a command with explicit observable output.",
                }
        else:
            self.null_bash_counts.pop(signature, None)
        return None
