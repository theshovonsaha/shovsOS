"""Kernel-driven source collection — the deterministic control plane.

This is the "one architectural move" in miniature: the *kernel* (contract +
ledger) owns control flow; the model is a typed slot-filler called at exactly
two points — (1) lock the N entities from the discovery search, and (2)
synthesize the final answer. Everything else — which tool, when to search, how
many to fetch, when to stop — is deterministic and contract-enforced.

For a "top N entities, M URLs each" task that is:
    1 discovery search  (deterministic query)
  + 1 LLM call          (extract/lock N entities)
  + N entity searches   (deterministic topic-aware query)
  + N*M fetches         (deterministic; contract decides the count)
  + 1 LLM call          (synthesize TLDR)
  = exactly **2 LLM calls**, vs the orchestrator loop's ~2T+3.

The core stays pure: tools and the model are injected as callables, so the main
project's real `web_search`/`web_fetch` and adapter plug in without coupling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .contract import SourceContract, infer_source_contract
from .evals import TraceEval, evaluate_trace
from .ledger import Ledger

SearchFn = Callable[[str], Awaitable[tuple[list[str], str]]]      # query -> (urls, raw_text)
FetchFn = Callable[[str], Awaitable[tuple[bool, str]]]            # url -> (ok, content)
ExtractFn = Callable[[str, str, int], Awaitable[list[str]]]       # (objective, discovery_text, n) -> entities
SynthFn = Callable[[str, list[dict[str, Any]]], Awaitable[str]]   # (objective, sources) -> answer

_TRAILING_WORKFLOW_RE = re.compile(
    r"\b(?:then|and then|after that|web\s*search|search\s+each|search\s+those|search\s+these|web\s*fetch|fetch|capture|analy[sz]e|"
    r"write|produce|report|tldr|tl;dr|summary table|one by one|separately|each)\b.*",
    re.IGNORECASE,
)
_LEADING_SEARCH_RE = re.compile(r"^\s*(?:please\s+)?(?:web\s*search|search\s+for|search|find|look\s*up)\s+", re.IGNORECASE)
_NUMBER_RE = r"\d+|one|two|three|four|five|six|seven|eight|nine|ten"


def discovery_query(objective: str) -> str:
    """Deterministically compile the discovery probe (no LLM): keep the topic,
    drop the workflow verbs/quotas."""
    q = _TRAILING_WORKFLOW_RE.sub("", str(objective or ""))
    q = _LEADING_SEARCH_RE.sub("", q)
    q = re.sub(rf"\b(top\s+(?:{_NUMBER_RE})|those|these|(?:{_NUMBER_RE})\s+relevant|all\s+(?:{_NUMBER_RE})\s+urls?)\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip(" .,:;")
    return q or str(objective or "")[:80]


def entity_search_query(objective: str, entity: str) -> str:
    """Build the per-entity source query without assuming the topic is stocks."""

    topic = str(objective or "").lower()
    name = str(entity or "").strip()
    if not name:
        return "sources"
    if any(term in topic for term in ("stock", "stocks", "ticker", "market", "gainer", "gainers", "equity")):
        return f"{name} stock news"
    if any(term in topic for term in ("restaurant", "restaurants", "sushi", "coffee", "bar", "hotel", "place", "places", "near me")):
        return f"{name} reviews sources"
    if any(term in topic for term in ("price", "prices", "shopping", "product", "products", "store", "stores", "walmart", "costco", "bestbuy")):
        return f"{name} reviews price sources"
    if any(term in topic for term in ("paper", "papers", "research", "study", "studies", "arxiv")):
        return f"{name} research paper sources"
    return f"{name} sources"


@dataclass
class KernelRunResult:
    objective: str
    contract: SourceContract
    entities: list[str]
    fetched: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    llm_calls: int = 0
    tool_calls: int = 0
    eval: TraceEval | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)


async def run_source_collection(
    objective: str,
    *,
    search_fn: SearchFn,
    fetch_fn: FetchFn,
    extract_entities_fn: ExtractFn,
    synth_fn: SynthFn,
) -> KernelRunResult:
    contract = infer_source_contract(objective)
    n = contract.entity_count or 3
    m = contract.urls_per_entity or 3
    ledger = Ledger(objective=objective, allowed_tools=["web_search", "web_fetch"])
    events: list[dict[str, Any]] = []
    llm_calls = 0

    # 1) Discovery search — deterministic query, no model.
    dq = discovery_query(objective)
    disc_urls, disc_text = await search_fn(dq)
    _record(ledger, events, "web_search", {"query": dq}, bool(disc_urls), {"urls": disc_urls})

    # 2) LLM slot-fill #1 — lock the N entities from the discovery results.
    raw_entities = await extract_entities_fn(objective, disc_text, n)
    llm_calls += 1
    entities: list[str] = []
    for e in raw_entities:
        e = str(e or "").strip().upper()
        if e and e not in entities:
            entities.append(e)
    entities = entities[:n]
    for e in entities:
        events.append({"kind": "entity_locked", "entity": e})
        ledger.event("entity_locked", entity=e)

    # 3) Per-entity: search each (deterministic), fetch M each. The contract
    #    decides the quota; the loop cannot drift or under-fetch by construction.
    fetched: list[dict[str, Any]] = []
    for e in entities:
        q = entity_search_query(objective, e)
        urls, _raw = await search_fn(q)
        _record(ledger, events, "web_search", {"query": q, "entity": e}, bool(urls), {"urls": urls}, entity=e)
        got = 0
        for u in urls:
            if got >= m:
                break
            ok, content = await fetch_fn(u)
            _record(ledger, events, "web_fetch", {"url": u, "entity": e}, ok, {"len": len(content or "")}, entity=e)
            if ok and content:
                fetched.append({"entity": e, "url": u, "content": content})
                got += 1

    # 4) LLM slot-fill #2 — synthesize the TLDR from the fetched, ledger-linked sources.
    answer = await synth_fn(objective, fetched)
    llm_calls += 1

    return KernelRunResult(
        objective=objective,
        contract=contract,
        entities=entities,
        fetched=fetched,
        answer=answer,
        llm_calls=llm_calls,
        tool_calls=sum(1 for event in events if event.get("tool") in {"web_search", "web_fetch"}),
        eval=evaluate_trace(contract, events),
        trace=ledger.trace(),
    )


def _record(ledger: Ledger, events: list, tool: str, args: dict, ok: bool, data: dict, *, entity: str = "") -> None:
    call = ledger.add_call(tool, args)
    ledger.add_result(call.id, ok, data, f"{tool} {entity}".strip())
    ev = {"tool": tool, "ok": ok}
    if entity:
        ev["entity"] = entity
    events.append(ev)
