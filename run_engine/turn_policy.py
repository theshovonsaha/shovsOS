"""Turn policy — the single deterministic routing gate.

One place that decides, from the user's intent, *what kind of turn this is* and
therefore: which tools the planner may pick, whether the planner runs at all, and
whether the answer should come straight from memory. This converges the routing,
context-poisoning, and looping concerns into one auditable decision instead of
scattered ad-hoc checks across the engine:

  - **Correct routing** — memory / identity / "read recent chat" intents are
    routed to memory tools, never web search.
  - **No context poisoning** — memory/disclosure intents forbid web tools, so raw
    web junk can't enter context for a "remember my name" turn.
  - **No looping** — direct-fact and simple-chat turns take zero tools and answer
    immediately; personal disclosures store + acknowledge instead of looping.

It is intentionally conservative: a turn only gets constrained when the intent is
*clear*. Anything ambiguous (or anything with a research/request verb) returns an
unconstrained policy so genuine web/research intents keep their tools.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

from run_engine.workflow_contracts import classify_workflow_shape


_MEMORY_TOOLS = frozenset({
    "query_memory", "store_memory", "update_memory",
    "shovs_memory_query", "shovs_memory_store", "rag_search",
})
_RECALL_TOOLS = frozenset({"query_memory", "shovs_memory_query", "rag_search"})
_STORE_TOOLS = frozenset({"query_memory", "store_memory", "update_memory", "shovs_memory_store"})
_SOURCE_TOOLS = frozenset({
    "web_search", "web_fetch", "web_fetch_batch",
    "source_collect", "source_contract", "source_select",
    "source_next_action", "source_coverage",
})

# A research / request verb anywhere disables the memory & disclosure routes, so
# "I'm looking for the best DAW" or "what's the latest on X" keep their web tools.
_RESEARCH_INTENT_RE = re.compile(
    r"\b(search|find|look(?:ing)?\s*(?:up|for)|lookup|google|research|investigate|compare|"
    r"recommend|suggest|need|want|how\s+(?:do|to|can|should)|what(?:'s|\s+is|\s+are|\s+should)\s+the\b|"
    r"where|when|who|which|why|best|top|news|price|prices|latest|current|today|trending|"
    r"help\s+me|show\s+me|give\s+me|fetch|get\s+me|tell\s+me\s+about|ideas|options|resources|"
    r"tutorial|guide|tips|advice)\b",
    re.IGNORECASE,
)
# Narrow "external information" signal for the memory routes — only clearly
# outward-facing terms disable identity recall/store (so "who am i" stays a
# memory recall, but "remember to search the web for the weather" does not).
_EXTERNAL_INFO_RE = re.compile(
    r"\b(search|web|google|fetch|url|https?://|online|latest|news|price|prices|"
    r"stock|stocks|market|weather|forecast|current\s+events|today'?s)\b",
    re.IGNORECASE,
)
_DISCLOSURE_OPENER_RE = re.compile(
    r"^\s*(?:i\s*am|i'?m|i\s+have|i\s+like|i\s+love|i\s+prefer|i\s+enjoy|"
    r"my\s+\w+(?:\s+\w+){0,3}\s+(?:is|are|was)|call\s+me)\b",
    re.IGNORECASE,
)
# "remember my name", "forget that", "remember that I ...", "update my ..."
_MEMORY_STORE_RE = re.compile(
    r"\b(remember|forget|update|store|save)\b.{0,30}\b(my|me|that|this|i\b)",
    re.IGNORECASE,
)
# "what's my name", "what do you remember/know about me", "who am i"
_MEMORY_RECALL_RE = re.compile(
    r"\bwhat(?:'?s| is| do you (?:remember|know))\b.{0,30}\b(my|me|about me)\b|"
    r"\bwho\s+am\s+i\b|\bwhat'?s\s+my\s+name\b|\bdo you (?:remember|know) (?:my|me)\b",
    re.IGNORECASE,
)
# "read recent chat", "what did we talk about", "summarize our conversation"
_CONVERSATION_RECALL_RE = re.compile(
    r"\b(read|recall|summari[sz]e|show|recap)\b[\w\s]{0,20}\b(recent|last|previous|prior|our|the|this)\b"
    r"[\w\s]{0,12}\b(chat|conversation|conversations|messages|discussion|thread|history)\b|"
    r"\bwhat did (?:we|i|you)\b|\b(chat|conversation)\s+history\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TurnPolicy:
    intent: str
    reason: str
    # None  -> no constraint (all allowed tools available to the planner)
    # frozenset (possibly empty) -> constrain the planner to exactly these tools
    tool_whitelist: Optional[frozenset[str]] = None
    use_planner: bool = True
    answer_from_memory: bool = False
    forbid_web: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tool_whitelist"] = sorted(self.tool_whitelist) if self.tool_whitelist is not None else None
        return data

    def constrain(self, allowed_tool_names: Iterable[str]) -> list[str]:
        """Return the allowed tool names filtered by this policy (order preserved)."""
        names = [str(n or "").strip() for n in allowed_tool_names if str(n or "").strip()]
        if self.tool_whitelist is None:
            return names
        return [n for n in names if n in self.tool_whitelist]


def _has_research_verb(text: str) -> bool:
    return bool(_RESEARCH_INTENT_RE.search(text))


def _is_disclosure(text: str) -> bool:
    if "?" in text or not _DISCLOSURE_OPENER_RE.search(text):
        return False
    if _has_research_verb(text):
        return False
    return len(text.split()) <= 12


def resolve_turn_policy(
    objective: str,
    *,
    user_message: str,
    allowed_tools: list[str] | tuple[str, ...] | None = None,
    direct_fact_answerable: bool = False,
    workflow_shape: str = "",
) -> TurnPolicy:
    """Resolve the single routing decision for this turn (deterministic, no LLM)."""
    text = str(user_message or objective or "").strip()
    shape = (workflow_shape or classify_workflow_shape(objective)).strip()

    # 1) Already answerable from clean deterministic facts → zero tools, no loop.
    if direct_fact_answerable:
        return TurnPolicy(
            intent="direct_fact",
            reason="deterministic facts already answer the objective",
            tool_whitelist=_RECALL_TOOLS,
            use_planner=False,
            answer_from_memory=True,
            forbid_web=True,
        )

    # 2) Conversation / history meta ("read recent chat") → memory, never web.
    if _CONVERSATION_RECALL_RE.search(text):
        return TurnPolicy(
            intent="conversation_recall",
            reason="meta request about the conversation history",
            tool_whitelist=_RECALL_TOOLS,
            use_planner=False,
            forbid_web=True,
        )

    # 3) Identity recall ("what's my name", "what do you know about me", "who am i").
    if _MEMORY_RECALL_RE.search(text) and not _EXTERNAL_INFO_RE.search(text):
        return TurnPolicy(
            intent="memory_recall",
            reason="recall of stored user facts",
            tool_whitelist=_RECALL_TOOLS,
            use_planner=False,
            forbid_web=True,
        )

    # 4) Memory store ("remember my name", "update my ...").
    if _MEMORY_STORE_RE.search(text) and not _EXTERNAL_INFO_RE.search(text):
        return TurnPolicy(
            intent="memory_store",
            reason="user asked to remember/update a fact",
            tool_whitelist=_STORE_TOOLS,
            use_planner=False,  # deterministic extractor + bootstrap handle it; no LLM planner
            forbid_web=True,
        )

    # 5) Personal disclosure ("I am a photographer", "I like blue").
    if _is_disclosure(text):
        return TurnPolicy(
            intent="disclosure",
            reason="first-person fact/preference to store, not research",
            tool_whitelist=_STORE_TOOLS,
            use_planner=False,  # the fact is captured deterministically; just acknowledge
            forbid_web=True,
        )

    # 6) Source collection → keep the web/source palette; the contract gate
    #    bounds the loop. (Don't constrain it away.)
    if shape == "source_collection":
        return TurnPolicy(
            intent="source_collection",
            reason="multi-source collection workflow",
            tool_whitelist=_SOURCE_TOOLS,
        )

    # 7) Trivial / simple chat → no tools.
    if shape == "simple_chat":
        return TurnPolicy(
            intent="simple_chat",
            reason="greeting / acknowledgement",
            tool_whitelist=frozenset(),
            use_planner=False,
        )

    # 8) Research / coding / open-ended → unconstrained (all allowed tools).
    return TurnPolicy(
        intent=shape or "open_ended",
        reason="no deterministic constraint; planner chooses from allowed tools",
        tool_whitelist=None,
    )
