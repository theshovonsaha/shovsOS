"""Proposers — where a model (or a script) proposes the actions to take.

The harness core stays pure: it never imports an LLM SDK. A proposer is any
object with ``async propose(...) -> list[ProposedAction]``. The real-model
proposer takes an *injected* adapter (anything with ``async complete(model,
messages) -> str``), so the main project's `llm/` adapters plug straight in
without coupling the core to them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .contract import SourceContract


@dataclass(frozen=True)
class ProposedAction:
    action: str            # "lock" | "fetch" | "respond"
    entity: str = ""
    url: str = ""
    entities: tuple[str, ...] = ()


class Proposer(Protocol):
    async def propose(self, contract: SourceContract, world: dict[str, Any]) -> list[ProposedAction]:
        ...


def _parse_actions(text: str) -> list[ProposedAction]:
    """Pull the first JSON array of actions out of a (possibly chatty) reply."""
    match = re.search(r"\[.*\]", str(text or ""), re.DOTALL)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0))
    except Exception:
        return []
    out: list[ProposedAction] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        out.append(ProposedAction(
            action=str(item.get("action") or "").strip().lower(),
            entity=str(item.get("entity") or "").strip(),
            url=str(item.get("url") or "").strip(),
            entities=tuple(str(e).strip() for e in (item.get("entities") or []) if str(e).strip()),
        ))
    return out


class ScriptedProposer:
    """Deterministic, offline 'model' that drifts on purpose — used so the eval
    and CI run without any API key. It under-fetches and chases an off-list
    entity, exactly the failure the harness must catch."""

    async def propose(self, contract: SourceContract, world: dict[str, Any]) -> list[ProposedAction]:
        entities = list(world.get("entities") or [])
        correct = entities[: max(1, contract.entity_count or 1)]
        drift = world.get("drift_entity") or "OFF_LIST"
        actions: list[ProposedAction] = [ProposedAction("lock", entities=tuple(correct))]
        if correct:
            # under-fetches the first entity (1 of M) ...
            urls = (world.get("urls") or {}).get(correct[0], [])
            if urls:
                actions.append(ProposedAction("fetch", entity=correct[0], url=urls[0]))
            # ... then drifts to an off-list entity and stops early.
            actions.append(ProposedAction("fetch", entity=drift, url=f"https://{drift.lower()}.test/0"))
        actions.append(ProposedAction("respond"))
        return actions


class LLMProposer:
    """Real-model proposer. ``adapter`` is any object exposing
    ``async complete(model, messages, temperature, max_tokens) -> str`` — e.g.
    the main project's GeminiAdapter via ``create_adapter('gemini:...')``."""

    def __init__(self, adapter: Any, model: str):
        self.adapter = adapter
        self.model = model

    async def propose(self, contract: SourceContract, world: dict[str, Any]) -> list[ProposedAction]:
        entities = list(world.get("entities") or [])
        urls = world.get("urls") or {}
        prompt = (
            f"Task: {contract.objective}\n"
            f"Lock the top {contract.entity_count} entities, then FETCH "
            f"{contract.urls_per_entity} URLs for EACH locked entity.\n\n"
            f"Search returned these candidate entities (pick the top "
            f"{contract.entity_count}; the rest are off-topic):\n{entities}\n\n"
            f"Candidate URLs per entity:\n{json.dumps(urls)}\n\n"
            "Output ONLY a JSON array of the tool actions you would take, in order. "
            "Each action is one of:\n"
            '  {"action":"lock","entities":["E1","E2",...]}\n'
            '  {"action":"fetch","entity":"E1","url":"https://..."}\n'
            '  {"action":"respond"}\n'
            "No prose, JSON array only."
        )
        text = await self.adapter.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700,
        )
        return _parse_actions(text)
