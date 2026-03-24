"""
Risk Classifier
---------------
Pure rule-based classifier. No LLM. No async.
Maps every tool call to a RiskLevel before execution.

Levels:
  BLOCK   → Never execute. Return error to LLM immediately.
  CONFIRM → Pause. Show user what will run. Wait for approval.
  WARN    → Execute but log visibly.
  SAFE    → Execute silently.

Extend by:
  - Adding entries to TOOL_BASE_RISK
  - Adding patterns to BLOCK_PATTERNS / CONFIRM_PATTERNS
  - Subclassing and overriding classify()
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class RiskLevel(str, Enum):
    SAFE    = "safe"
    WARN    = "warn"
    CONFIRM = "confirm"
    BLOCK   = "block"


@dataclass
class Classification:
    level:   RiskLevel
    reason:  str
    tool:    str
    preview: str          # Human-readable summary of what will run


# ── Base risk per tool (override with patterns below) ─────────────────────────

TOOL_BASE_RISK: dict[str, RiskLevel] = {
    # Read-only — always safe
    "web_search":      RiskLevel.SAFE,
    "web_fetch":       RiskLevel.SAFE,
    "image_search":    RiskLevel.SAFE,
    "weather_fetch":   RiskLevel.SAFE,
    "query_memory":    RiskLevel.SAFE,
    "file_view":       RiskLevel.SAFE,

    # Write — warn by default, confirm if pattern matches
    "store_memory":    RiskLevel.WARN,
    "file_create":     RiskLevel.CONFIRM,
    "file_str_replace":RiskLevel.CONFIRM,
    "generate_app":    RiskLevel.WARN,
    "pdf_processor":   RiskLevel.WARN,
    "places_search":   RiskLevel.SAFE,
    "places_map":      RiskLevel.WARN,

    # High risk — always confirm
    "bash":              RiskLevel.CONFIRM,
    "delegate_to_agent": RiskLevel.CONFIRM,
}

# ── Patterns that escalate bash commands to BLOCK ─────────────────────────────

BLOCK_PATTERNS: list[str] = [
    "rm -rf /",
    "rm -rf ~",
    "sudo rm",
    "chmod 777 /",
    "chown -R root",
    "mkfs",
    "> /dev/",
    "dd if=",
    ":(){ :|:& };:",       # fork bomb
    "curl | sh",
    "curl | bash",
    "wget | sh",
    "wget | bash",
    "python -c",           # inline exec — flag for review
    "/etc/passwd",
    "/etc/shadow",
    "~/.ssh",
    "/Applications",
    "~/Library",
    "/System",
    "/usr/bin",
    "/usr/local/bin",
]

# ── Patterns that escalate bash to CONFIRM even if base is WARN ───────────────

CONFIRM_PATTERNS: list[str] = [
    "rm ",
    "mv ",
    "cp -r",
    "chmod",
    "chown",
    "kill ",
    "pkill",
    "sudo",
    "brew uninstall",
    "pip uninstall",
    "npm uninstall",
    "git reset --hard",
    "git clean -f",
    "truncate",
    "shred",
]


class RiskClassifier:
    """
    Classifies a tool call before execution.
    Instantiate once; call classify() per tool call.

    Args:
        require_confirmation_for: "all" | "confirm_and_above" | "block_only"
            "all"              → every tool pauses for confirmation
            "confirm_and_above"→ only CONFIRM/BLOCK tools pause (default)
            "block_only"       → only BLOCK tools are stopped, rest run freely
    """

    def __init__(
        self,
        require_confirmation_for: str = "all",
        extra_block_patterns: Optional[list[str]] = None,
        extra_confirm_patterns: Optional[list[str]] = None,
        custom_tool_risks: Optional[dict[str, RiskLevel]] = None,
    ):
        self.confirmation_mode = require_confirmation_for
        self._block_patterns   = BLOCK_PATTERNS + (extra_block_patterns or [])
        self._confirm_patterns = CONFIRM_PATTERNS + (extra_confirm_patterns or [])
        self._tool_risks       = {**TOOL_BASE_RISK, **(custom_tool_risks or {})}

    def classify(self, tool_name: str, arguments: dict) -> Classification:
        base = self._tool_risks.get(tool_name, RiskLevel.CONFIRM)

        # bash / shell commands get pattern analysis
        if tool_name == "bash":
            cmd = arguments.get("command", "")
            return self._classify_bash(cmd)

        # file operations — check path traversal
        if tool_name in ("file_create", "file_str_replace", "file_view"):
            path = arguments.get("path", arguments.get("filename", ""))
            if self._path_escapes_sandbox(path):
                return Classification(
                    level   = RiskLevel.BLOCK,
                    reason  = "Path traversal detected — target is outside sandbox.",
                    tool    = tool_name,
                    preview = f"{tool_name}(path='{path}')",
                )

        # delegation — always confirm, unknown agents are extra risky
        if tool_name == "delegate_to_agent":
            agent_id = arguments.get("target_agent_id", "unknown")
            task     = arguments.get("task", "")[:80]
            return Classification(
                level   = self._apply_mode(RiskLevel.CONFIRM),
                reason  = f"Delegation to sub-agent '{agent_id}'.",
                tool    = tool_name,
                preview = f"delegate → {agent_id}: \"{task}\"",
            )

        # build preview string
        preview = self._build_preview(tool_name, arguments)

        return Classification(
            level   = self._apply_mode(base),
            reason  = f"Base risk for '{tool_name}'.",
            tool    = tool_name,
            preview = preview,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _classify_bash(self, cmd: str) -> Classification:
        cmd_lower = cmd.lower()

        for pattern in self._block_patterns:
            if pattern.lower() in cmd_lower:
                return Classification(
                    level   = RiskLevel.BLOCK,
                    reason  = f"Blocked pattern detected: '{pattern}'",
                    tool    = "bash",
                    preview = f"bash: {cmd[:120]}",
                )

        for pattern in self._confirm_patterns:
            if pattern.lower() in cmd_lower:
                return Classification(
                    level   = self._apply_mode(RiskLevel.CONFIRM),
                    reason  = f"Risky pattern requires confirmation: '{pattern}'",
                    tool    = "bash",
                    preview = f"bash: {cmd[:120]}",
                )

        return Classification(
            level   = self._apply_mode(RiskLevel.CONFIRM),
            reason  = "All bash commands require confirmation.",
            tool    = "bash",
            preview = f"bash: {cmd[:120]}",
        )

    def _path_escapes_sandbox(self, path: str) -> bool:
        """Detect obvious path traversal attempts."""
        suspicious = ["..", "~", "/Users", "/home", "/etc", "/System", "/Applications"]
        return any(s in path for s in suspicious)

    def _apply_mode(self, base: RiskLevel) -> RiskLevel:
        """Escalate based on confirmation_mode setting."""
        if self.confirmation_mode == "all":
            # Everything at WARN or above becomes CONFIRM
            if base in (RiskLevel.SAFE, RiskLevel.WARN):
                return RiskLevel.CONFIRM
            return base

        if self.confirmation_mode == "block_only":
            # Only BLOCK stays blocked, everything else runs freely
            if base == RiskLevel.BLOCK:
                return RiskLevel.BLOCK
            return RiskLevel.SAFE

        # Default: "confirm_and_above" — keep as-is
        return base

    def _build_preview(self, tool_name: str, arguments: dict) -> str:
        parts = ", ".join(f"{k}='{str(v)[:40]}'" for k, v in arguments.items())
        return f"{tool_name}({parts})"
