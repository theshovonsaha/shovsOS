from __future__ import annotations

import re


SOCIAL_TURN_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|ok|okay|cool|nice|great|got it|understood|makes sense)"
    r"(?:\s+(again|there|that helps|so much|thanks|ok|okay|got it|understood|cool|nice|great|today))*\s*[!.?]*\s*$",
    re.IGNORECASE,
)

ACTION_TERMS = {
    "search", "find", "fetch", "research", "analyze", "analyse", "compare",
    "summarize", "summarise", "write", "draft", "make", "create", "implement",
    "fix", "remember", "call", "use", "prefer", "update", "change", "continue",
    "resume", "go", "open", "read", "calculate", "compute",
}


def is_low_value_social_turn(user_message: str) -> bool:
    """Return True for greetings/acks that should not become durable memory."""
    text = str(user_message or "").strip()
    lowered = text.lower()
    if not text:
        return True
    tokens = [
        re.sub(r"[^a-z0-9_]", "", token)
        for token in lowered.replace("-", " ").split()
    ]
    tokens = [token for token in tokens if token]
    if any(token in ACTION_TERMS for token in tokens):
        return False
    if len(tokens) > 6:
        return False
    return bool(SOCIAL_TURN_PATTERN.match(text))


def should_skip_memory_compression(user_message: str, assistant_response: str = "") -> bool:
    """Compression gate for low-value turns.

    The assistant response is intentionally ignored for social turns. A verbose
    greeting response should not turn "hi again" into durable context.
    """
    return is_low_value_social_turn(user_message)
