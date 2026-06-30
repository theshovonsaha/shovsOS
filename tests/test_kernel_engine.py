"""Offline proof for the kernel-driven runtime (run_engine/kernel_engine.py).

No network, no real model: mock tools + a prompt-aware mock adapter. Asserts the
kernel's invariants across shapes — minimal LLM calls, contract quota fetched,
completion + evidence + per-claim citation grounding, standard event stream.
"""

import json
import re

import pytest

from plugins.tool_registry import Tool, ToolRegistry
from run_engine.kernel_engine import KernelRunEngine
from run_engine.types import RunEngineRequest

SOURCE_TASK = ("Search top 3 stocks today with major jumps, web search those 3 separately, "
               "fetch 3 urls each, analyze and write a tldr summary table.")


def _tok(url: str) -> str:
    parts = [p for p in url.split("/") if p]
    return parts[-2].upper() if len(parts) >= 2 else "X"


class MockAdapter:
    """Prompt-aware: returns entity JSON for the lock prompt, a grounded table
    for synthesis (citing exactly the URLs present in the sources block, with
    claim words that appear in the mock fetched content), a greeting for chat."""

    def __init__(self):
        self.calls = 0

    async def complete(self, *, model, messages, **kw):
        self.calls += 1
        prompt = messages[-1]["content"]
        if "JSON array" in prompt:
            return '["AAA", "BBB", "CCC"]'
        if "respond naturally" in prompt:
            return "Hello! How can I help you today?"
        urls = list(dict.fromkeys(re.findall(r"https?://[^\s\n]+", prompt)))
        rows = "\n".join(f"| {_tok(u)} | earnings beat upgrade | {u} |" for u in urls)
        return "| Item | Note | Source |\n| --- | --- | --- |\n" + rows


class FabricatingAdapter(MockAdapter):
    """Synthesis cites a URL that was never fetched — the citation gate must catch it."""

    async def complete(self, *, model, messages, **kw):
        self.calls += 1
        if "JSON array" in messages[-1]["content"]:
            return '["AAA", "BBB", "CCC"]'
        return ("| Item | Note | Source |\n| --- | --- | --- |\n"
                "| AAA | secret merger | https://fake.test/not-fetched |")


def _mock_registry():
    async def web_search(*, query, **kw):
        token = (query.split() or ["X"])[0].upper()
        results = [{"url": f"https://src.test/{token}/{i}",
                    "title": f"{token} result {i}",
                    "snippet": f"{token} news AAA BBB CCC mover today"} for i in range(4)]
        return json.dumps({"results": results})

    async def web_fetch(*, url, **kw):
        tok = _tok(url)
        body = f"{tok} report: earnings beat upgrade buyback rally surge analyst price target. " * 8
        return json.dumps({"content": body + url, "url": url})

    reg = ToolRegistry()
    reg.register(Tool(name="web_search", description="search", parameters={}, handler=web_search))
    reg.register(Tool(name="web_fetch", description="fetch", parameters={}, handler=web_fetch))
    return reg


def _req(session_id, task):
    return RunEngineRequest(
        session_id=session_id, owner_id="o1", agent_id="a1",
        user_message=task, model="mock-model",
        allowed_tools=("web_search", "web_fetch"),
    )


async def _run(adapter, task, session_id="s"):
    engine = KernelRunEngine(adapter=adapter, tool_registry=_mock_registry())
    events = [ev async for ev in engine.stream(_req(session_id, task))]
    metrics = next(e for e in events if e["type"] == "kernel_metrics")
    return events, metrics


@pytest.mark.asyncio
async def test_source_collection_two_calls_grounded():
    adapter = MockAdapter()
    events, m = await _run(adapter, SOURCE_TASK, "s1")
    types = [e["type"] for e in events]
    assert types[0] == "session" and types[-1] == "done"
    assert m["shape"] == "source_collection"
    assert adapter.calls == 2 and m["llm_calls"] == 2          # lock + synth
    locked = [e["entity"] for e in events if e["type"] == "entity_locked"]
    assert locked == ["AAA", "BBB", "CCC"]
    assert m["fetched"] == 9 and m["contract_total"] == 9
    assert m["completion_gate_open"] is True
    assert m["evidence_supported"] is True
    assert m["citations_grounded"] is True and m["ungrounded_claims"] == 0


@pytest.mark.asyncio
async def test_source_collection_drift_is_structurally_impossible():
    events, _ = await _run(MockAdapter(), SOURCE_TASK, "s2")
    fetch_urls = [e["data"]["url"] for e in events
                  if e["type"] == "tool_call" and e["tool_name"] == "web_fetch"]
    assert len(fetch_urls) == 9
    assert all(any(f"/{tok}/" in u for tok in ("AAA", "BBB", "CCC")) for u in fetch_urls)


@pytest.mark.asyncio
async def test_comparison_shape_one_call_no_discovery():
    adapter = MockAdapter()
    events, m = await _run(adapter, "Compare AAA versus BBB on price and growth", "s3")
    assert m["shape"] == "comparison"
    # entities are parsed from the query (no discovery search, no extract call)
    assert adapter.calls == 1 and m["llm_calls"] == 1
    locked = [e["entity"] for e in events if e["type"] == "entity_locked"]
    assert locked == ["AAA", "BBB"]
    covered = {u.split("/")[-2] for u in
               (e["data"]["url"] for e in events if e["type"] == "tool_call" and e["tool_name"] == "web_fetch")}
    assert covered == {"AAA", "BBB"}            # both sides sourced
    assert m["completion_gate_open"] is True
    assert m["citations_grounded"] is True


@pytest.mark.asyncio
async def test_single_fact_shape_one_search_one_call():
    adapter = MockAdapter()
    events, m = await _run(adapter, "What were the top earnings beats today?", "s4")
    assert m["shape"] == "single_fact"
    assert adapter.calls == 1 and m["llm_calls"] == 1
    searches = [e for e in events if e["type"] == "tool_call" and e["tool_name"] == "web_search"]
    assert len(searches) == 1                    # single discovery search, no per-entity fan-out
    assert m["fetched"] >= 1 and m["completion_gate_open"] is True


@pytest.mark.asyncio
async def test_simple_shape_no_tools():
    adapter = MockAdapter()
    events, m = await _run(adapter, "hello there", "s5")
    assert m["shape"] == "simple"
    assert adapter.calls == 1 and m["llm_calls"] == 1
    assert m["tool_calls"] == 0 and m["fetched"] == 0


@pytest.mark.asyncio
async def test_citation_gate_flags_fabricated_source():
    events, m = await _run(FabricatingAdapter(), SOURCE_TASK, "s6")
    assert m["citations_grounded"] is False
    assert m["ungrounded_claims"] >= 1
    cite_ev = next(e for e in events if e["type"] == "citation_grounding")
    assert "https://fake.test/not-fetched" in cite_ev["fabricated_urls"]
    answer = "".join(e["content"] for e in events if e["type"] == "token")
    assert "Citation gate" in answer
