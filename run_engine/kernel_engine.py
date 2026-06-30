"""KernelRunEngine — the deterministic kernel as a real platform runtime.

The maximum-impact revamp: instead of the orchestrator owning the loop (an LLM
call to plan, to observe, to verify — ~2T+3 calls per run), a deterministic
kernel owns control flow and the model is called at exactly two points:

    1. lock the N entities from the discovery search   (model slot #1)
    2. synthesize the final answer                      (model slot #2)

Everything else — which tool, how many to fetch, when to stop — is decided by
the workflow contract (completion grounding) and the run ledger (evidence
grounding). The model cannot drift, under-fetch, stop early, or claim a fetch
that did not happen.

This is a NEW runtime that reuses the existing organs (`infer_workflow_contract`,
`update_contract_from_tool_results`, `RunLedger`, `turn_policy`, the real
`web_search`/`web_fetch` tools). It does not touch `RunEngine`; select it
explicitly so the two can be A/B'd on one task.
"""

from __future__ import annotations

import dataclasses
import json
import re
import uuid
from typing import Any, AsyncIterator

from llm.adapter_factory import create_adapter, strip_provider_prefix
from run_engine.ledger import RunLedger
from run_engine.search_query import compile_web_search_query
from run_engine.turn_policy import resolve_turn_policy
from run_engine.types import RunEngineRequest
from run_engine.workflow_contracts import (
    EntityLock,
    infer_workflow_contract,
    update_contract_from_tool_results,
)


_QUERY_STOP = {
    "compare", "comparison", "versus", "difference", "differences", "between", "which",
    "better", "best", "build", "decision", "table", "cite", "every", "claim", "claims",
    "write", "summary", "tldr", "report", "search", "fetch", "analyze", "each", "those",
    "these", "top", "relevant", "results", "url", "urls", "the", "and", "for", "with",
    "into", "that", "this", "from", "you", "are", "was", "were", "will", "now", "should",
    "would", "could", "capture", "separately", "one", "all", "major", "web", "today",
}


def _salient_topic(objective: str, others: tuple[str, ...] | list[str] = ()) -> str:
    """Pull the comparison dimensions / salient topic words from the objective
    (no LLM), dropping the option names and workflow verbs. Topic-agnostic."""
    s = str(objective or "").lower()
    for name in others:
        s = s.replace(str(name).lower(), " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    out: list[str] = []
    seen: set[str] = set()
    for w in s.split():
        if len(w) > 2 and not w.isdigit() and w not in _QUERY_STOP and w not in seen:
            seen.add(w); out.append(w)
    return " ".join(out[:6])


def _entity_query(objective: str, entity: str, others: tuple[str, ...] | list[str] = ()) -> str:
    """Topic-aware per-entity probe (no LLM). Explicit verticals first, then a
    generic 'entity + salient dimensions' query so any topic stays on-subject."""
    t = str(objective or "").lower()
    e = str(entity or "").strip()
    # Stock requires a real stock signal — "job market"/"housing market" must NOT match.
    if re.search(r"\bstocks?\b|\btickers?\b|\bgainers?\b|\bequit|\bshares\b|stock market", t):
        return f"{e} stock news"
    if any(w in t for w in ("restaurant", "sushi", "cafe", "coffee", "bar", "hotel", "near me")):
        return f"{e} reviews"
    if any(w in t for w in ("paper", "arxiv")):
        return f"{e} research paper"
    topic = _salient_topic(objective, [e, *others])
    return f"{e} {topic}".strip() if topic else f"{e} overview"


def _urls_from_search(raw: str) -> tuple[list[str], str]:
    urls: list[str] = []
    text = raw
    try:
        d = json.loads(raw)
        parts = []
        for r in (d.get("results") or []):
            u = r.get("url") or r.get("normalized_url")
            if u:
                urls.append(u)
            parts.append(f"{r.get('title', '')} — {r.get('snippet', '')}".strip(" —"))
        if parts:
            text = "\n".join(p for p in parts if p)
    except Exception:
        urls = re.findall(r"https?://[^\s\"'<>)]+", raw)
    seen, out = set(), []
    for u in urls:
        if u not in seen and "google.com/search" not in u:
            seen.add(u); out.append(u)
    return out, text


def _content_from_fetch(raw: str) -> tuple[bool, str]:
    try:
        d = json.loads(raw)
    except Exception:
        return (False, "")
    if d.get("error") or str(d.get("type", "")).endswith("error"):
        return (False, "")
    return (True, str(d.get("content") or "")[:3000])


_COMPARE_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|difference[s]? between|which is better)\b", re.IGNORECASE)
_VS_RE = re.compile(r"\b(?:vs\.?|versus)\b", re.IGNORECASE)
_QUESTION_RE = re.compile(
    r"^\s*(what|who|whom|whose|when|where|which|why|how|is|are|was|were|does|do|did|can|will|should)\b",
    re.IGNORECASE,
)
_WEB_INTENT_RE = re.compile(r"\b(search|find|look\s*up|research|latest|current|today|news|price|who\s+won)\b", re.IGNORECASE)
_COMPARE_LEAD_RE = re.compile(
    r"^\s*(please\s+)?(compare|comparison of|difference between|differences between|how (?:does|do)|which is better[:,]?)\s+",
    re.IGNORECASE,
)
_COMPARE_TAIL_RE = re.compile(
    r"\b(?:then|and then|web\s*search|web\s*fetch|fetch|search|summari[sz]e|write|in terms of|regarding|"
    r"which|what|that|who|whose|whom|for|on|about|across)\b.*", re.IGNORECASE)
_LEAD_ARTICLE_RE = re.compile(r"^\s*(the|a|an)\s+", re.IGNORECASE)
_GENERIC_CLAIM_WORDS = {
    "ticker", "source", "sources", "note", "notes", "why", "moved", "name", "entity",
    "summary", "reason", "comparison", "feature", "features", "aspect", "verdict",
    "winner", "pros", "cons", "takeaway", "overall", "https", "http", "www", "com",
}


def _parse_entities(raw: str, fallback_urls: list[str], n: int) -> list[str]:
    """Parse the entity-lock model reply into N unique short names (deterministic)."""
    cand: list[str] = []
    mjson = re.search(r"\[.*\]", raw or "", re.DOTALL)
    if mjson:
        try:
            cand = [str(x).strip().upper() for x in json.loads(mjson.group(0))]
        except Exception:
            cand = []
    if not cand:
        cand = re.findall(r"\b[A-Z][A-Z0-9]{1,5}\b", raw or "") or list(fallback_urls[:n])
    out: list[str] = []
    seen: set[str] = set()
    for e in cand:
        if e and e not in seen:
            seen.add(e); out.append(e)
    return out[:n]


def classify_kernel_shape(objective: str, contract) -> str:
    """Pick the kernel flow for a request. Topic-agnostic, no LLM."""
    if getattr(contract, "workflow_shape", "") == "source_collection" \
            and int((getattr(contract, "metadata", None) or {}).get("entity_count") or 0) > 0:
        return "source_collection"
    if getattr(contract, "workflow_shape", "") == "simple_chat":
        return "simple"
    low = str(objective or "").lower()
    if _COMPARE_RE.search(low):
        return "comparison"
    if _QUESTION_RE.search(low) or low.rstrip().endswith("?") or _WEB_INTENT_RE.search(low):
        return "single_fact"
    return "simple"


def comparison_entities(objective: str) -> list[str]:
    """Deterministically pull the named things to compare from the query (no LLM)."""
    s = _COMPARE_LEAD_RE.sub("", str(objective or "").strip())
    parts = _VS_RE.split(s)
    if len(parts) < 2:
        head = _COMPARE_TAIL_RE.split(s)[0]
        parts = re.split(r"\s*(?:,| and )\s*", head)
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        p = _COMPARE_TAIL_RE.sub("", p)
        p = _LEAD_ARTICLE_RE.sub("", p).strip(" .,?-\"'")
        if p and 1 <= len(p) <= 60 and p.lower() not in seen:
            seen.add(p.lower()); out.append(p)
    return out[:4]


def check_citation_grounding(answer: str, fetched: list[dict[str, Any]], *, min_overlap: int = 1) -> dict[str, Any]:
    """Per-claim citation grounding (the frontier gap): a cited URL must have been
    actually fetched, AND the claim's content tokens must appear in that URL's
    fetched content. Catches the "cite a real-looking URL for an invented fact"
    failure that fetch-existence checks miss."""
    by_url = {s.get("url"): str(s.get("content") or "").lower() for s in (fetched or [])}
    url_re = re.compile(r"https?://[^\s)\]|>]+")
    cited = list(dict.fromkeys(url_re.findall(answer or "")))
    fabricated = [u for u in cited if u not in by_url]
    rows_checked = 0
    ungrounded_rows: list[str] = []
    for line in str(answer or "").splitlines():
        if line.count("|") < 2 or "---" in line:
            continue
        row_urls = url_re.findall(line)
        if not row_urls:
            continue
        rows_checked += 1
        # Skip the first cell (row key / entity name) and URL cells: the entity
        # name trivially appears in its own source, which would let a fabricated
        # claim ("invented spaceship division") pass just because "Tesla" matched.
        cells = [c.strip() for c in line.split("|") if c.strip()]
        claim_cells = [c for i, c in enumerate(cells) if i > 0 and not url_re.search(c)]
        claim = " ".join(claim_cells)
        claim_tokens = {w for w in re.findall(r"[a-zA-Z]{4,}", claim.lower())} - _GENERIC_CLAIM_WORDS
        supported = False
        for u in row_urls:
            content = by_url.get(u)
            if content is None:
                continue  # cited a URL we never fetched
            if not claim_tokens or sum(1 for w in claim_tokens if w in content) >= min_overlap:
                supported = True
                break
        if not supported:
            ungrounded_rows.append(claim.strip(" |"))
    return {
        "grounded": (not fabricated) and (not ungrounded_rows),
        "fabricated_urls": fabricated,
        "rows_checked": rows_checked,
        "ungrounded_rows": ungrounded_rows,
    }


class KernelRunEngine:
    """Deterministic-kernel runtime. Same construction surface as RunEngine; only
    `tool_registry`, `sessions`, and the resolved adapter are used."""

    def __init__(self, *, adapter=None, sessions=None, tool_registry=None,
                 run_store=None, trace_store=None, orchestrator=None,
                 context_engine=None, graph=None):
        self.adapter = adapter
        self.sessions = sessions
        self.tool_registry = tool_registry
        self.run_store = run_store
        self.trace_store = trace_store
        self.graph = graph

    def _resolve_adapter(self, model: str):
        if ":" in str(model):
            return create_adapter(provider=str(model).split(":", 1)[0].lower())
        return self.adapter or create_adapter()

    async def _llm(self, adapter, model: str, prompt: str, max_tokens: int = 700) -> str:
        return await adapter.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )

    async def _run_tool(self, name: str, **args) -> str:
        tool = self.tool_registry.get(name) if self.tool_registry else None
        if tool is None:
            return json.dumps({"type": "error", "error": f"tool not registered: {name}"})
        return await tool.handler(**args)

    async def _do_search(self, ledger: RunLedger, query: str, entity: str = ""):
        """Run one real search; return (urls, snippet_text, events, result_dict)."""
        args = {"query": query, **({"entity": entity} if entity else {})}
        call = ledger.add_tool_call("web_search", args, source="kernel")
        raw = await self._run_tool("web_search", query=query, num_results=10)
        urls, text = _urls_from_search(raw)
        res = ledger.link_tool_result(
            tool_call_id=call.id, tool_name="web_search", success=bool(urls),
            status="ok" if urls else "error", summary=f"web_search '{query}' -> {len(urls)} urls")
        events = [
            {"type": "tool_call", "tool_name": "web_search", "tool_call_id": call.id, "data": args},
            {"type": "tool_result", "tool_name": "web_search", "tool_call_id": call.id,
             "tool_result_id": res.id, "status": "ok" if urls else "failed"},
        ]
        rdict = {"tool_name": "web_search", "success": bool(urls), "arguments": args, "extracted_urls": urls}
        return urls, text, events, rdict

    async def _fetch_urls(self, ledger, cand_urls, m, entity, *, fetched, tool_result_dicts):
        """Fetch up to `m` URLs (deterministic, contract-bounded). Yields tool
        events; appends accepted sources to `fetched` and dicts to
        `tool_result_dicts` in place."""
        got = 0
        for u in cand_urls:
            if got >= m:
                break
            fargs = {"url": u, **({"entity": entity} if entity else {})}
            fcall = ledger.add_tool_call("web_fetch", fargs, source="kernel")
            yield {"type": "tool_call", "tool_name": "web_fetch", "tool_call_id": fcall.id, "data": fargs}
            ok, content = _content_from_fetch(await self._run_tool("web_fetch", url=u, max_chars=3000))
            fres = ledger.link_tool_result(
                tool_call_id=fcall.id, tool_name="web_fetch", success=ok,
                status="ok" if ok else "error", summary=f"web_fetch {u} ({len(content or '')} chars)")
            if ok:
                ledger.add_evidence_from_result(fres)
            yield {"type": "tool_result", "tool_name": "web_fetch", "tool_call_id": fcall.id,
                   "tool_result_id": fres.id, "status": "ok" if ok else "failed"}
            tool_result_dicts.append({"tool_name": "web_fetch", "success": ok, "arguments": fargs})
            if ok and content:
                fetched.append({"entity": entity or "topic", "url": u, "content": content}); got += 1

    async def stream(self, request: RunEngineRequest) -> AsyncIterator[dict[str, Any]]:
        objective = request.user_message
        model_str = request.model or "gemini-2.5-flash"
        adapter = self._resolve_adapter(model_str)
        model = strip_provider_prefix(model_str)
        run_id = uuid.uuid4().hex[:12]
        session_id = request.session_id
        yield {"type": "session", "run_id": run_id, "session_id": session_id}

        contract = infer_workflow_contract(objective, allowed_tools=list(request.allowed_tools) or ["web_search", "web_fetch"])
        shape = classify_kernel_shape(objective, contract)
        meta = contract.metadata or {}
        n = max(0, int(meta.get("entity_count") or 0))
        m = max(1, int(meta.get("results_per_entity") or 1))

        ledger = RunLedger(run_id=run_id, session_id=session_id, turn_id=f"{session_id}:0",
                           objective=objective, allowed_tools=["web_search", "web_fetch"])
        ledger.set_workflow_contract(contract)
        policy = resolve_turn_policy(objective, user_message=objective,
                                     allowed_tools=["web_search", "web_fetch", "query_memory"])
        yield {"type": "turn_policy", "intent": policy.intent, "reason": policy.reason}

        llm_calls = 0
        tool_result_dicts: list[dict[str, Any]] = []
        fetched: list[dict[str, Any]] = []
        entities: list[str] = []
        total = 0
        gate_ok = True

        if shape == "source_collection":
            yield {"type": "plan", "strategy": f"kernel: source collection ({n}x{m})",
                   "tools": ["web_search", "web_fetch"], "confidence": 1.0}
            total = int(meta.get("total_fetches") or (n * m))
            # 1) discovery search (deterministic query)
            urls, disc_text, evs, rdict = await self._do_search(ledger, compile_web_search_query(objective))
            for ev in evs:
                yield ev
            tool_result_dicts.append(rdict)
            # 2) lock entities (model slot #1)
            if n > 0 and disc_text:
                raw = await self._llm(adapter, model,
                    f"From these search results, list the {n} most prominent entity names/tickers for: "
                    f"{objective!r}.\nResults:\n{disc_text[:4000]}\n\n"
                    f'Output ONLY a JSON array of {n} short names, e.g. ["AAA","BBB"].', max_tokens=120)
                llm_calls += 1
                entities = _parse_entities(raw, urls, n)
            if entities:
                ledger.lock_entities(entities, source="kernel")
                contract = dataclasses.replace(
                    contract,
                    entity_locks=[EntityLock(value=e, status="locked", source="kernel") for e in entities],
                )
            for e in entities:
                ledger.append_event("entity_locked", source="kernel", data={"entity": e})
                yield {"type": "entity_locked", "entity": e}
            # 3) per-entity search + fetch M each (deterministic; contract-bounded)
            for e in (entities or [None]):
                cand_urls = urls
                if e is not None:
                    others = [x for x in entities if x != e]
                    cand_urls, _t, evs, rdict = await self._do_search(ledger, _entity_query(objective, e, others), entity=e)
                    for ev in evs:
                        yield ev
                    tool_result_dicts.append(rdict)
                async for ev in self._fetch_urls(ledger, cand_urls, m, e, fetched=fetched, tool_result_dicts=tool_result_dicts):
                    yield ev
            # completion grounding: update the contract from real results, read its gate
            contract = update_contract_from_tool_results(
                contract, tool_result_dicts, tool_turn=len(tool_result_dicts), max_tool_turns=24)
            ledger.set_workflow_contract(contract)
            gate_ok = bool(contract.completion_gate.final_answer_allowed)

        elif shape == "comparison":
            entities = comparison_entities(objective)
            if len(entities) < 2:
                shape = "single_fact"  # not actually a comparison; degrade
            else:
                per = 2  # sources per side
                total = len(entities) * per
                yield {"type": "plan", "strategy": f"kernel: comparison ({len(entities)} items x {per})",
                       "tools": ["web_search", "web_fetch"], "confidence": 1.0}
                ledger.lock_entities(entities, source="kernel")
                for e in entities:
                    ledger.append_event("entity_locked", source="kernel", data={"entity": e})
                    yield {"type": "entity_locked", "entity": e}
                for e in entities:
                    others = [x for x in entities if x != e]
                    cand_urls, _t, evs, rdict = await self._do_search(ledger, _entity_query(objective, e, others), entity=e)
                    for ev in evs:
                        yield ev
                    tool_result_dicts.append(rdict)
                    async for ev in self._fetch_urls(ledger, cand_urls, per, e, fetched=fetched, tool_result_dicts=tool_result_dicts):
                        yield ev
                # completion: at least one source per side
                covered = {s["entity"] for s in fetched}
                gate_ok = all(e in covered for e in entities)

        if shape == "single_fact":
            per = 3
            total = per
            yield {"type": "plan", "strategy": "kernel: single-fact lookup (1 search, fetch top K)",
                   "tools": ["web_search", "web_fetch"], "confidence": 1.0}
            urls, _disc, evs, rdict = await self._do_search(ledger, compile_web_search_query(objective))
            for ev in evs:
                yield ev
            tool_result_dicts.append(rdict)
            async for ev in self._fetch_urls(ledger, urls, per, None, fetched=fetched, tool_result_dicts=tool_result_dicts):
                yield ev
            gate_ok = len(fetched) >= 1

        elif shape == "simple":
            yield {"type": "plan", "strategy": "kernel: direct answer (no tools)",
                   "tools": [], "confidence": 1.0}
            gate_ok = True

        # ---- synthesize (model slot: #2 for source_collection, #1 for the rest) ----
        answer = await self._synthesize(adapter, model, objective, shape, fetched)
        llm_calls += 1

        # ---- evidence grounding (fetch existence) ----
        support = ledger.response_support_check(answer)
        if not support.get("supported", True):
            answer = ("[evidence gate] The draft claimed work not backed by a successful tool result "
                      f"({support.get('issues')}). Honest result: {len(fetched)} sources fetched.")

        # ---- per-claim citation grounding (claims must trace to fetched content) ----
        citation = check_citation_grounding(answer, fetched)
        if fetched and not citation["grounded"]:
            notes = []
            if citation["fabricated_urls"]:
                notes.append("cited URL(s) never fetched: " + ", ".join(citation["fabricated_urls"][:3]))
            if citation["ungrounded_rows"]:
                notes.append(f"{len(citation['ungrounded_rows'])} claim(s) not found in their cited source")
            answer = answer.rstrip() + "\n\n> ⚠️ Citation gate: " + "; ".join(notes) + "."
            yield {"type": "citation_grounding", **citation}

        for chunk in re.findall(r".{1,400}(?:\s|$)", answer, re.DOTALL) or [answer]:
            if chunk:
                yield {"type": "token", "content": chunk}

        yield {"type": "kernel_metrics", "shape": shape, "llm_calls": llm_calls, "fetched": len(fetched),
               "contract_total": total, "completion_gate_open": gate_ok,
               "evidence_supported": bool(support.get("supported", True)),
               "citations_grounded": bool(citation["grounded"]),
               "citation_rows_checked": citation["rows_checked"],
               "ungrounded_claims": len(citation["ungrounded_rows"]) + len(citation["fabricated_urls"]),
               "tool_calls": len(ledger.tool_calls)}
        if self.sessions is not None:
            try:
                self.sessions.append_message(session_id, "assistant", answer)
            except Exception:
                pass
        yield {"type": "done", "run_id": run_id, "session_id": session_id}

    async def _synthesize(self, adapter, model, objective, shape, fetched):
        if shape == "simple" or not fetched:
            if shape == "simple":
                return await self._llm(adapter, model,
                    f"Answer concisely and directly. If this is a greeting or chit-chat, respond "
                    f"naturally without tools.\n\n{objective}", max_tokens=400)
            return "I could not fetch any sources, so I won't fabricate an answer."
        blob = "\n\n".join(f"[{s['entity']}] {s['url']}\n{s['content'][:1200]}" for s in fetched)[:14000]
        if shape == "comparison":
            instr = ("produce a markdown comparison table (one row per dimension, one column per "
                     "item) and cite the source URL for each claim")
        else:
            instr = ("answer in 2-4 sentences and cite the source URL(s) inline for every factual claim")
        return await self._llm(adapter, model,
            f"Using ONLY the fetched sources below, {instr}. Do NOT invent facts, figures, or URLs "
            f"not present in the sources.\n\nTask: {objective}\n\nSources:\n{blob}", max_tokens=900)
