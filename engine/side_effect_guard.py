"""Deterministic side-effect claim guard.

Detects when a final response asserts a side-effect (file written, command run,
package installed, etc.) and cross-checks tool_results for matching evidence.
Prevents the Replit/Gemini failure pattern where the model confabulates
"already created" before any tool ran, or after a denial/HARD_FAILURE.

This is a pre-LLM deterministic gate — runs before `verify_with_context`
so verification can inherit issues without round-tripping through a model.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

# Tool names that produce side-effects on disk / system state.
WRITE_TOOLS: frozenset[str] = frozenset({
    "bash",
    "file_create",
    "file_str_replace",
})

# Per-tool execution risk tier. Read by ``check_plan_for_side_effects`` so the
# planner's tool selection can be cross-checked against the user's intent
# *before* the tool runs — instead of only after the response is drafted.
# Tiers match the ``execution_risk_tier`` vocabulary in run_engine/code_intent.py.
TOOL_RISK_TIERS: dict[str, str] = {
    # Reads (default for anything not listed)
    "web_search": "read_only",
    "web_fetch": "read_only",
    "query_memory": "read_only",
    "shovs_memory_query": "read_only",
    # Writes that can mutate the user's repo / disk
    "file_create": "write",
    "file_str_replace": "write",
    # Arbitrary code execution
    "bash": "destructive",
}


def tool_risk_tier(tool_name: str) -> str:
    """Return the risk tier for a tool. Unknown tools default to ``read_only``
    so we don't block on tools the registry didn't explicitly classify."""
    return TOOL_RISK_TIERS.get(str(tool_name or "").lower(), "read_only")


# User-message phrases that authorize a write/destructive plan. Conservative
# on purpose — false positives (under-blocking) are worse than false negatives
# (extra warnings), but we don't want to demand explicit consent for the obvious
# "fix the bug" / "create the file" turns either.
_WRITE_AUTHORIZATION_RE = re.compile(
    r"\b("
    r"create|write|add|edit|update|modify|patch|fix|refactor|rename|delete|"
    r"remove|install|run|build|deploy|commit|push|migrate|generate|wire|"
    r"implement|hook[\s-]?up|set[\s-]?up"
    r")\b",
    re.IGNORECASE,
)


def check_plan_for_side_effects(
    *,
    user_message: str,
    selected_tools: list[str],
    declared_risk_tier: str = "",
) -> dict[str, Any]:
    """Sanity-check a *plan* before tool dispatch.

    Returns a non-blocking advisory: ``{"clear": bool, "warnings": [...],
    "max_tier": str}``. Caller decides whether to surface the warning to the
    user or just trace it. We never block dispatch from here — over-blocking
    breaks legitimate "fix this bug" turns where the write intent is implicit.
    The check exists so we can *flag* mismatches (e.g. plan picked ``bash``
    but user message has no write/run verbs) and pair them with the post-hoc
    response guard for full coverage.
    """
    tiers = [tool_risk_tier(name) for name in (selected_tools or [])]
    rank = {"read_only": 0, "write": 1, "destructive": 2}
    max_tier = "read_only"
    for tier in tiers:
        if rank.get(tier, 0) > rank.get(max_tier, 0):
            max_tier = tier

    warnings: list[str] = []
    if max_tier in ("write", "destructive"):
        text = (user_message or "").strip()
        has_authorization = bool(_WRITE_AUTHORIZATION_RE.search(text))
        # Only warn when both: plan is risky AND user message lacks any
        # action verb. If the user said "explain X" but the plan picked
        # bash, that's the signal worth surfacing.
        if not has_authorization:
            risky_tools = [
                name for name, tier in zip(selected_tools or [], tiers)
                if tier in ("write", "destructive")
            ]
            warnings.append(
                f"Plan selected {max_tier} tool(s) {risky_tools} but the user "
                "message contains no write/run/edit verb. Consider asking for "
                "explicit confirmation before mutating state."
            )
        # Cross-check vs declared risk tier from code_intent classifier.
        if (
            declared_risk_tier
            and declared_risk_tier in {"none", "read_only"}
            and max_tier in {"write", "destructive"}
        ):
            warnings.append(
                f"Plan picked {max_tier} tools but code_intent classified the "
                f"task as {declared_risk_tier}. Risk tiers disagree — verify "
                "intent before dispatch."
            )

    return {
        "clear": not warnings,
        "warnings": warnings,
        "max_tier": max_tier,
        "selected_tools": list(selected_tools or []),
    }

# First-person assertion patterns — past/perfect tense claims about side-effects.
# Kept narrow on purpose: only fires on direct claims of completed action,
# not on plans ("I will create…") or descriptions ("the file contains…").
_CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("file_write", re.compile(r"\bI\s+(?:have\s+)?(?:just\s+)?(?:created|wrote|written|saved|generated|added|made)\b[^.]*\b(?:file|script|module|component|page|folder|directory)\b", re.IGNORECASE)),
    ("file_write", re.compile(r"\b(?:created|wrote|written|saved|generated)\s+(?:the\s+|a\s+|an\s+)?(?:file|script|module|component|page)\b", re.IGNORECASE)),
    ("file_write", re.compile(r"\bfile\s+(?:has\s+been|was)\s+(?:created|written|saved|generated)\b", re.IGNORECASE)),
    ("bash_exec", re.compile(r"\bI\s+(?:have\s+)?(?:just\s+)?(?:ran|executed|installed|built|deployed|started|launched)\b", re.IGNORECASE)),
    ("bash_exec", re.compile(r"\b(?:ran|executed)\s+(?:the\s+)?command\b", re.IGNORECASE)),
    ("bash_exec", re.compile(r"\binstalled\s+(?:the\s+)?(?:package|dependency|dependencies|module)\b", re.IGNORECASE)),
    ("bash_exec", re.compile(r"\b(?:command|script)\s+(?:has\s+been|was)\s+(?:run|executed|completed)\b", re.IGNORECASE)),
    ("file_edit", re.compile(r"\bI\s+(?:have\s+)?(?:just\s+)?(?:updated|modified|edited|patched|replaced)\b[^.]*\b(?:file|line|function|method|variable)\b", re.IGNORECASE)),
)

# Negation patterns — if the response itself flags failure, skip the guard
# (the response is being honest about the failure already).
_HONEST_FAILURE = re.compile(
    r"\b(?:could not|couldn['’]t|failed to|unable to|did not|was not|wasn['’]t|"
    r"hard[_\s-]?failure|denied|blocked|error|exception)\b",
    re.IGNORECASE,
)


def _result_status(result: dict[str, Any]) -> str:
    """Extract status from a tool result. Empty string if unparseable."""
    content = result.get("content")
    if isinstance(content, str):
        try:
            payload = json.loads(content)
            if isinstance(payload, dict):
                status = payload.get("status")
                if isinstance(status, str):
                    return status.upper()
        except Exception:
            pass
    return ""


def _has_supporting_evidence(claim_kind: str, tool_results: list[dict[str, Any]]) -> bool:
    """Return True if at least one tool result supports the claim kind."""
    for result in tool_results:
        if not isinstance(result, dict):
            continue
        if not result.get("success"):
            continue
        tool_name = str(result.get("tool_name") or "")
        status = _result_status(result)
        # HARD_FAILURE / DENIED / FAILED never count as supporting evidence,
        # even if `success` was somehow set.
        if status in {"HARD_FAILURE", "DENIED", "FAILED"}:
            continue
        if claim_kind == "file_write" and tool_name in {"bash", "file_create"}:
            return True
        if claim_kind == "file_edit" and tool_name in {"bash", "file_str_replace", "file_create"}:
            return True
        if claim_kind == "bash_exec" and tool_name == "bash":
            return True
    return False


def _has_hard_failure(tool_results: Iterable[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Return (has_failure, [tool_names])."""
    failed: list[str] = []
    for result in tool_results or []:
        if not isinstance(result, dict):
            continue
        status = _result_status(result)
        if status in {"HARD_FAILURE", "DENIED"}:
            name = str(result.get("tool_name") or "tool")
            failed.append(f"{name}:{status}")
    return bool(failed), failed


def check_side_effect_claims(
    response: str,
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Inspect a final response for unsupported side-effect claims.

    Returns a dict with:
      - supported: bool
      - issues: list[str] — human-readable issue strings
      - claims: list[str] — claim kinds detected
      - hard_failures: list[str] — any tool results that returned HARD_FAILURE/DENIED
    """
    text = (response or "").strip()
    issues: list[str] = []
    claims: list[str] = []

    if not text:
        return {"supported": True, "issues": [], "claims": [], "hard_failures": []}

    has_failure, failure_names = _has_hard_failure(tool_results)

    # Detect first-person past-tense claims of completed action.
    for kind, pattern in _CLAIM_PATTERNS:
        if pattern.search(text):
            if kind not in claims:
                claims.append(kind)

    if not claims:
        # No side-effect claim — guard is silent.
        if has_failure:
            # Response made no claim, but a tool hard-failed. Not the guard's
            # job to escalate — verification will still see the failure.
            return {
                "supported": True,
                "issues": [],
                "claims": [],
                "hard_failures": failure_names,
            }
        return {"supported": True, "issues": [], "claims": [], "hard_failures": []}

    # If response itself flags the failure, it's being honest — skip guard.
    if _HONEST_FAILURE.search(text):
        return {
            "supported": True,
            "issues": [],
            "claims": claims,
            "hard_failures": failure_names,
        }

    # Hard failure present + claim made = unsupported, regardless of evidence.
    if has_failure:
        issues.append(
            f"Response claims completed side-effect ({', '.join(claims)}) but "
            f"tool(s) returned hard failure: {', '.join(failure_names)}."
        )
        return {
            "supported": False,
            "issues": issues,
            "claims": claims,
            "hard_failures": failure_names,
        }

    # Claim made — must have at least one supporting successful tool result.
    for kind in claims:
        if not _has_supporting_evidence(kind, tool_results):
            issues.append(
                f"Response claims '{kind}' completed but no successful tool "
                f"result supports it (no successful {', '.join(sorted(WRITE_TOOLS))} call)."
            )

    return {
        "supported": not issues,
        "issues": issues,
        "claims": claims,
        "hard_failures": failure_names,
    }
