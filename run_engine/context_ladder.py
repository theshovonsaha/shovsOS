from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


CONTEXT_LADDER_VERSION = "context-ladder-v1"


@dataclass(frozen=True)
class ContextLadderStep:
    level: str
    title: str
    content: str
    source: str
    priority: int
    raw_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_context_ladder(
    *,
    query: str,
    compact_memory: str = "",
    relevant_blocks: list[dict[str, Any]] | None = None,
    evidence_refs: list[dict[str, Any]] | None = None,
    raw_payloads: list[dict[str, Any]] | None = None,
    include_raw: bool = False,
    limit: int = 8,
) -> dict[str, Any]:
    """Build a small, ordered retrieval ladder for phase packets and UI.

    The ladder keeps the actor on compact, high-signal context by default.
    Raw payloads are represented as references unless explicitly requested.
    """

    terms = _keywords(query)
    steps: list[ContextLadderStep] = []
    if compact_memory.strip():
        steps.append(ContextLadderStep(
            level="keyword_hint",
            title="Compact memory signal",
            content=_clip(_best_lines(compact_memory, terms), 700),
            source="memory",
            priority=60,
        ))

    for block in relevant_blocks or []:
        content = str(block.get("content") or block.get("text") or "").strip()
        if not content:
            continue
        steps.append(ContextLadderStep(
            level="relevant_block",
            title=str(block.get("title") or block.get("id") or "Relevant block"),
            content=_clip(_best_lines(content, terms), 900),
            source=str(block.get("source") or "context"),
            priority=70,
            raw_ref=str(block.get("id") or block.get("raw_ref") or ""),
        ))

    for evidence in evidence_refs or []:
        summary = str(evidence.get("summary") or evidence.get("content") or "").strip()
        if not summary:
            continue
        steps.append(ContextLadderStep(
            level="evidence_reference",
            title=str(evidence.get("id") or evidence.get("title") or "Evidence"),
            content=_clip(summary, 700),
            source=str(evidence.get("source") or "evidence"),
            priority=95,
            raw_ref=str(evidence.get("raw_ref") or evidence.get("id") or ""),
        ))

    for raw in raw_payloads or []:
        raw_ref = str(raw.get("id") or raw.get("raw_ref") or raw.get("url") or "").strip()
        content = str(raw.get("content") or raw.get("text") or raw.get("payload") or "").strip()
        if not raw_ref and not content:
            continue
        steps.append(ContextLadderStep(
            level="raw_payload" if include_raw else "raw_payload_ref",
            title=str(raw.get("title") or raw_ref or "Raw payload"),
            content=_clip(content, 1200) if include_raw else "Raw payload available on demand.",
            source=str(raw.get("source") or "raw"),
            priority=20 if include_raw else 35,
            raw_ref=raw_ref,
        ))

    steps = sorted(steps, key=lambda item: -item.priority)[: max(1, int(limit or 8))]
    return {
        "version": CONTEXT_LADDER_VERSION,
        "query_keywords": terms,
        "include_raw": bool(include_raw),
        "steps": [step.to_dict() for step in steps],
    }


def render_context_ladder(ladder: dict[str, Any]) -> str:
    lines = ["Context Ladder:"]
    for step in ladder.get("steps") or []:
        raw_ref = f" ({step.get('raw_ref')})" if step.get("raw_ref") else ""
        lines.append(f"- {step.get('level')}: {step.get('title')}{raw_ref}")
        content = str(step.get("content") or "").strip()
        if content:
            lines.append(f"  {content}")
    return "\n".join(lines)


def _keywords(text: str) -> list[str]:
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", str(text or ""))
        if word.lower() not in {"the", "and", "for", "with", "that", "this", "from", "into"}
    ]
    return list(dict.fromkeys(words))[:12]


def _best_lines(text: str, terms: list[str]) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    if not terms:
        return "\n".join(lines[:5])
    scored = []
    for index, line in enumerate(lines):
        low = line.lower()
        score = sum(1 for term in terms if term in low)
        if score:
            scored.append((-score, index, line))
    if not scored:
        return "\n".join(lines[:5])
    return "\n".join(item[2] for item in sorted(scored)[:5])


def _clip(text: str, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"
