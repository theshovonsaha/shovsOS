"""
Code Intent Classifier
-----------------------
Heuristic-first classification of whether a user message implies
code writing, what execution risk tier it carries, and whether
scope clarification is needed before acting.

Design rules:
- Regex/pattern first — no LLM call by default.
- Returns a typed dataclass, not raw strings.
- Empty/neutral return when no code intent detected — zero impact on non-code turns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ── Signals ──────────────────────────────────────────────────────────

# Explicit code creation words
EXPLICIT_CODE_SIGNAL = re.compile(
    r"\b(write|create|build|make|generate|scaffold|implement|code|develop|program)\b"
    r".*\b(script|function|class|module|app|application|program|file|component|api|endpoint|server|bot|tool|cli|page|website)\b",
    re.IGNORECASE | re.DOTALL,
)

# Implicit code intent — task that obviously needs code but doesn't say "create"
IMPLICIT_CODE_SIGNAL = re.compile(
    r"\b(analyze|parse|process|convert|transform|extract|scrape|crawl|automate|migrate|refactor|debug|fix|patch|deploy|test)\b"
    r".*\b(csv|json|xml|html|pdf|data|file|database|db|api|logs?|table|spreadsheet|image|video|audio)\b",
    re.IGNORECASE | re.DOTALL,
)

# Destruction signals — risk tier escalation
DESTRUCTIVE_SIGNAL = re.compile(
    r"\b(delete|remove|drop|truncate|purge|wipe|destroy|erase|overwrite|replace all|rm -rf|rm -r|rmdir|unlink)\b",
    re.IGNORECASE,
)

# Write signals — risk tier for file/database mutations
WRITE_SIGNAL = re.compile(
    r"\b(write|save|create|modify|update|insert|append|rename|move|copy|chmod|chown|pip install|npm install|apt install|brew install)\b",
    re.IGNORECASE,
)

# Ambiguity signals — scope is underspecified
AMBIGUOUS_SCOPE_SIGNAL = re.compile(
    r"^(?:write|create|build|make|generate)\s+(?:me\s+)?(?:a|an|some|the)?\s*(?:script|app|tool|program|bot|thing|something)\s*[.!?]?\s*$",
    re.IGNORECASE,
)

# Specific enough — has a concrete target/domain
SPECIFIC_SCOPE_SIGNAL = re.compile(
    r"\b(?:that|which|to|for|using|with|from|in|on)\b",
    re.IGNORECASE,
)

# File type mentions
FILE_TYPE_SIGNAL = re.compile(
    r"\b(?:\.py|\.js|\.ts|\.tsx|\.jsx|\.html|\.css|\.sh|\.bash|\.sql|\.go|\.rs|\.java|\.rb|\.php|\.yaml|\.yml|\.toml|\.ini|\.cfg)\b",
    re.IGNORECASE,
)

# ── Research / multi-step ambiguity signals ──────────────────────────────────

# "Research X" — vague enough to need scoping if X is a bare noun with no context
VAGUE_RESEARCH_SIGNAL = re.compile(
    r"^(?:research|investigate|analyze|evaluate|assess|look into|find out about|tell me about)\s+(?:a|an|the|some)?\s*\w[\w\s]{0,30}[.!?]?\s*$",
    re.IGNORECASE,
)

# Comparison with no named second party — "compare X" with nothing to compare to
VAGUE_COMPARE_SIGNAL = re.compile(
    r"^(?:compare|contrast)\s+(?:a|an|the|some)?\s*\w[\w\s]{0,30}[.!?]?\s*$",
    re.IGNORECASE,
)

# "Find the best X" — needs to know the evaluation criteria
BEST_OF_SIGNAL = re.compile(
    r"^(?:find|get|show|list|recommend)\s+(?:me\s+)?(?:the\s+)?best\s+\w[\w\s]{0,40}[.!?]?\s*$",
    re.IGNORECASE,
)

# Signals that the message already has enough scope to proceed
SCOPED_ENOUGH_SIGNAL = re.compile(
    r"\b(for\s+my|for\s+our|for\s+the|in\s+\w+|using\s+\w+|vs\.?\s+\w+|compared\s+to|between\s+\w+\s+and|criteria|requirements?|goal|budget|timeline)\b",
    re.IGNORECASE,
)


def check_research_ambiguity(message: str) -> Optional[str]:
    """
    Return a clarification question if the message is a vague research/comparison
    request that needs scoping before we waste tool calls on wrong targets.
    Returns None if message is specific enough to act on.
    """
    msg = (message or "").strip()
    if not msg or len(msg) > 280:
        # Long messages are usually specific enough
        return None
    if SCOPED_ENOUGH_SIGNAL.search(msg):
        return None

    if VAGUE_COMPARE_SIGNAL.match(msg):
        return (
            "What should I compare it against? "
            "A second product, approach, or set of criteria would help me give you a useful comparison."
        )
    if BEST_OF_SIGNAL.match(msg):
        return (
            "What are the most important criteria for 'best' here — "
            "price, features, ease of use, performance, or something else?"
        )
    if VAGUE_RESEARCH_SIGNAL.match(msg):
        return (
            "What specific aspect do you want to focus on? "
            "For example: pricing, technical details, reviews, competitors, or recent news."
        )
    return None


# ── Classifier ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class CodeIntent:
    """Classification of code-related intent in a user message."""

    code_warranted: bool
    reason: str
    execution_risk_tier: str    # "none" | "read_only" | "write" | "destructive"
    missing_context: Optional[str]  # one clarification question if scope is ambiguous
    fallback_if_failed: str     # next best action if primary approach fails

    def to_phase_note(self) -> str:
        """Render as a compact note for phase_guidance injection."""
        if not self.code_warranted:
            return ""
        parts = [f"Code intent detected: {self.reason}."]
        if self.execution_risk_tier in ("write", "destructive"):
            parts.append(f"Execution risk: {self.execution_risk_tier}.")
        if self.fallback_if_failed:
            parts.append(f"Fallback if primary approach fails: {self.fallback_if_failed}")
        return " ".join(parts)

    def to_risk_note(self) -> str:
        """Render as a constraint note for loop_contract injection."""
        if self.execution_risk_tier == "destructive":
            return (
                "WARNING: This task involves potentially destructive operations. "
                "Confirm exact targets before executing. Prefer dry-run or preview first."
            )
        if self.execution_risk_tier == "write":
            return (
                "This task involves file or system writes. "
                "Verify target paths and expected outcomes before execution."
            )
        return ""


_NEUTRAL = CodeIntent(
    code_warranted=False,
    reason="",
    execution_risk_tier="none",
    missing_context=None,
    fallback_if_failed="",
)


def classify_code_intent(
    user_message: str,
    effective_objective: str = "",
    tool_results: Optional[list[dict]] = None,
) -> CodeIntent:
    """Classify code intent from user message using heuristic signals.

    This is deliberately regex-first to avoid adding LLM latency.
    Returns a neutral CodeIntent when no code signals are detected.
    """
    msg = (user_message or "").strip()
    obj = (effective_objective or msg).strip()
    combined = f"{msg} {obj}"

    if not msg:
        return _NEUTRAL

    # ── Detect code intent ──
    explicit_match = EXPLICIT_CODE_SIGNAL.search(combined)
    implicit_match = IMPLICIT_CODE_SIGNAL.search(combined)
    file_type_match = FILE_TYPE_SIGNAL.search(combined)

    if not explicit_match and not implicit_match and not file_type_match:
        return _NEUTRAL

    # ── Classify reason ──
    if explicit_match:
        reason = "explicit code creation request"
    elif file_type_match:
        reason = f"file type reference ({file_type_match.group(0)}) implies code task"
    else:
        reason = "task domain implies code is needed"

    # ── Classify risk tier ──
    if DESTRUCTIVE_SIGNAL.search(combined):
        risk_tier = "destructive"
    elif WRITE_SIGNAL.search(combined):
        risk_tier = "write"
    else:
        risk_tier = "read_only"

    # ── Check for ambiguous scope ──
    missing_context: Optional[str] = None
    if explicit_match and AMBIGUOUS_SCOPE_SIGNAL.search(msg):
        if not SPECIFIC_SCOPE_SIGNAL.search(msg) and not file_type_match:
            missing_context = (
                "What specific task should this script accomplish? "
                "A concrete goal will produce much better code."
            )

    # ── Determine fallback ──
    if risk_tier == "destructive":
        fallback = "Show a preview/dry-run of what would be affected instead of executing."
    elif risk_tier == "write":
        fallback = "Output the code in chat for review instead of writing directly to disk."
    elif implicit_match:
        fallback = "Explain the approach and ask if the user wants code generated."
    else:
        fallback = "Provide a minimal working example before building the full solution."

    return CodeIntent(
        code_warranted=True,
        reason=reason,
        execution_risk_tier=risk_tier,
        missing_context=missing_context,
        fallback_if_failed=fallback,
    )
