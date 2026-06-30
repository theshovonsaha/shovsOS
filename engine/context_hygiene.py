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

MEMORY_SIGNAL_PATTERN = re.compile(
    r"\b("
    r"remember|forget|actually|correction|correct(?:ion)?|update|changed?|switched|"
    r"prefer|preference|call me|my name|i am|i'm|i live|i work|i use|i like|"
    r"i don't|i do not|allergic|budget|timezone|location|based in|moving|"
    r"save this|store this|keep this"
    r")\b",
    re.IGNORECASE,
)


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


def has_durable_memory_signal(user_message: str) -> bool:
    """Return True when a turn is likely worth durable memory work."""

    return bool(MEMORY_SIGNAL_PATTERN.search(str(user_message or "")))


def should_run_llm_memory_compression(
    user_message: str,
    assistant_response: str = "",
    *,
    is_first_exchange: bool = False,
    deterministic_fact_count: int = 0,
    void_count: int = 0,
    candidate_signal_count: int = 0,
    turn: int | None = None,
    interval: int = 6,
    mode: str = "adaptive",
) -> bool:
    """Gate expensive LLM summarization separately from deterministic memory.

    Deterministic fact extraction and governed memory commits can run cheaply.
    Calling a chat model just to recompress context should be rarer: only when
    there is a durable user signal, correction/void, candidate stance signal, or
    a periodic maintenance interval. ``mode`` supports operational overrides:
    ``always`` preserves old behavior, ``off`` disables LLM compression, and
    ``adaptive`` is the default.
    """

    normalized_mode = str(mode or "adaptive").strip().lower()
    if normalized_mode in {"always", "force", "on"}:
        return not should_skip_memory_compression(user_message, assistant_response)
    if normalized_mode in {"off", "none", "skip"}:
        return False
    if should_skip_memory_compression(user_message, assistant_response):
        return False
    if deterministic_fact_count > 0 or void_count > 0 or candidate_signal_count > 0:
        return True
    if has_durable_memory_signal(user_message):
        return True
    if is_first_exchange:
        text = str(user_message or "").strip()
        return len(text.split()) >= 8 or len(text) >= 60
    if interval > 0 and turn is not None and turn > 0:
        return turn % interval == 0
    return False
