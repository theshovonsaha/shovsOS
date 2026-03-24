"""
Tool Audit Logger
-----------------
Writes a structured JSONL record for every tool call.
Logs BEFORE execution (intent) and AFTER (result).

Log file: ./logs/tool_audit.jsonl  (configurable)

Each line is valid JSON:
{
  "ts":         "2026-02-28T13:38:45Z",
  "event":      "TOOL_INTENT" | "TOOL_RESULT" | "TOOL_BLOCKED" | "TOOL_DENIED" | "TOOL_TIMEOUT",
  "call_id":    "a1b2c3d4",
  "session_id": "sess_xyz",
  "agent_id":   "default",
  "tool":       "bash",
  "arguments":  {...},
  "risk_level": "confirm",
  "preview":    "bash: ls -la",
  "result":     "...",   # only on TOOL_RESULT
  "success":    true,    # only on TOOL_RESULT
  "reason":     "..."    # only on BLOCKED/DENIED/TIMEOUT
}

Usage:
    logger = ToolAuditLogger()
    call_id = logger.log_intent(session_id, agent_id, tool, args, classification)
    logger.log_result(call_id, result)
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class ToolAuditLogger:
    """
    Thread-safe append-only JSONL audit logger.
    Safe to share across async tasks.
    """

    def __init__(self, log_path: str = "./logs/tool_audit.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Touch the file so it exists immediately
        self.log_path.touch(exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def log_intent(
        self,
        session_id: str,
        agent_id: str,
        tool: str,
        arguments: dict,
        classification,        # Classification dataclass from risk_classifier
    ) -> str:
        """
        Log that a tool call is about to be considered.
        Returns a call_id you should pass to log_result/log_blocked.
        """
        call_id = str(uuid.uuid4())[:8]
        self._write({
            "event":      "TOOL_INTENT",
            "call_id":    call_id,
            "session_id": session_id,
            "agent_id":   agent_id,
            "tool":       tool,
            "arguments":  self._sanitize(arguments),
            "risk_level": classification.level.value,
            "preview":    classification.preview,
            "reason":     classification.reason,
        })
        return call_id

    def log_result(
        self,
        call_id: str,
        tool: str,
        success: bool,
        content: str,
        duration_ms: Optional[float] = None,
    ):
        """Log the outcome after execution."""
        self._write({
            "event":       "TOOL_RESULT",
            "call_id":     call_id,
            "tool":        tool,
            "success":     success,
            "result":      content[:500],   # truncate large outputs
            "duration_ms": duration_ms,
        })

    def log_blocked(self, call_id: str, tool: str, reason: str):
        """Log a call that was hard-blocked by the classifier."""
        self._write({
            "event":   "TOOL_BLOCKED",
            "call_id": call_id,
            "tool":    tool,
            "reason":  reason,
        })

    def log_denied(self, call_id: str, tool: str, reason: str):
        """Log a call that was denied by the user."""
        self._write({
            "event":   "TOOL_DENIED",
            "call_id": call_id,
            "tool":    tool,
            "reason":  reason,
        })

    def log_timeout(self, call_id: str, tool: str):
        """Log a call that timed out waiting for confirmation."""
        self._write({
            "event":   "TOOL_TIMEOUT",
            "call_id": call_id,
            "tool":    tool,
        })

    def tail(self, n: int = 50) -> list[dict]:
        """Return the last n log entries as a list of dicts."""
        try:
            lines = self.log_path.read_text().strip().splitlines()
            return [json.loads(l) for l in lines[-n:] if l]
        except Exception:
            return []

    def get_session_trace(self, session_id: str) -> list[dict]:
        """Return all log entries for a given session_id."""
        try:
            lines = self.log_path.read_text().strip().splitlines()
            result = []
            for line in lines:
                try:
                    entry = json.loads(line)
                    if entry.get("session_id") == session_id:
                        result.append(entry)
                except Exception:
                    continue
            return result
        except Exception:
            return []

    # ── Internal ───────────────────────────────────────────────────────────────

    def _write(self, record: dict):
        record["ts"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[ToolAuditLogger] Failed to write log: {e}")

    def _sanitize(self, arguments: dict) -> dict:
        """Remove sensitive keys before logging."""
        sensitive = {"password", "token", "api_key", "secret", "auth"}
        return {
            k: ("***" if k.lower() in sensitive else v)
            for k, v in arguments.items()
        }
