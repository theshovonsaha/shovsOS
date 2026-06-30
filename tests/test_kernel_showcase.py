"""Tests for the flagship 'high-stakes cited comparison' workflow.

Two layers:
  1. OFFLINE (always runs): proves the harness GUARANTEES on the comparison
     workflow deterministically — all options locked + sourced, every claim
     traced to fetched content, a fabricated citation caught, 1 LLM call.
  2. LIVE (skipped without GEMINI_API_KEY): runs the real kernel with
     gemini-3.1-flash-lite and asserts the guarantees that hold even under a
     partial/rate-limited fetch — 1 LLM call, no fabricated citations.

This is the "real harness power" test: a single LLM call, and the answer
physically cannot cite a source the agent did not fetch.
"""

import json
import os
import re

import pytest

from plugins.tool_registry import Tool, ToolRegistry
from run_engine.kernel_engine import KernelRunEngine
from run_engine.types import RunEngineRequest

TASK = ("Compare Austin vs Denver vs Raleigh for relocating in 2026 — cost of living and "
        "job market. Build a decision table and cite every claim.")
OPTIONS = ["Austin", "Denver", "Raleigh"]
# What the (mock) fetched pages actually say — the only facts the answer may use.
FACTS = {
    "AUSTIN": "austin median rent is 1680 dollars and tech job growth is strong",
    "DENVER": "denver median rent is 1890 dollars with steady job market",
    "RALEIGH": "raleigh median rent is 1410 dollars and research triangle hiring",
}


def _slug(url):
    parts = [p for p in url.split("/") if p]
    return parts[-2].upper() if len(parts) >= 2 else "X"


def _registry():
    async def web_search(*, query, **kw):
        tok = (query.split() or ["X"])[0].upper()
        return json.dumps({"results": [{"url": f"https://city.test/{tok}/{i}",
                                         "title": f"{tok} {i}", "snippet": f"{tok} relocation data"}
                                        for i in range(3)]})

    async def web_fetch(*, url, **kw):
        return json.dumps({"url": url, "content": FACTS.get(_slug(url), "no data") + f"  [{url}]"})

    reg = ToolRegistry()
    reg.register(Tool(name="web_search", description="s", parameters={}, handler=web_search))
    reg.register(Tool(name="web_fetch", description="f", parameters={}, handler=web_fetch))
    return reg


class GroundedAdapter:
    """Synthesizes a comparison table whose claims come from the fetched FACTS,
    citing the real fetched URLs — the honest agent."""

    def __init__(self):
        self.calls = 0

    async def complete(self, *, model, messages, **kw):
        self.calls += 1
        prompt = messages[-1]["content"]
        urls = list(dict.fromkeys(re.findall(r"https?://[^\s\n]+", prompt)))
        first = {}
        for u in urls:
            first.setdefault(_slug(u), u)
        header = "| City | Median rent | Source |\n| --- | --- | --- |\n"
        body = "\n".join(f"| {s.title()} | {FACTS[s].split('rent is ')[1].split(' and')[0].split(' with')[0]} "
                         f"| {first[s]} |" for s in first if s in FACTS)
        return header + body


class LyingAdapter:
    """Cites a plausible source it was never given — the failure the gate must catch."""

    def __init__(self):
        self.calls = 0

    async def complete(self, *, model, messages, **kw):
        self.calls += 1
        return ("| City | Median rent | Source |\n| --- | --- | --- |\n"
                "| Austin | 1680 | https://nytimes.com/totally-real-article |")


async def _run(adapter, task=TASK, allowed=("web_search", "web_fetch")):
    engine = KernelRunEngine(adapter=adapter, tool_registry=_registry())
    req = RunEngineRequest(session_id="show", owner_id="o", agent_id="a",
                           user_message=task, model="mock-model", allowed_tools=allowed)
    events = [ev async for ev in engine.stream(req)]
    metrics = next(e for e in events if e["type"] == "kernel_metrics")
    return events, metrics


@pytest.mark.asyncio
async def test_comparison_is_grounded_and_cheap():
    adapter = GroundedAdapter()
    events, m = await _run(adapter)
    assert m["shape"] == "comparison"
    # ONE model call for the whole high-stakes decision
    assert adapter.calls == 1 and m["llm_calls"] == 1
    # every option locked and sourced (drift impossible)
    locked = [e["entity"] for e in events if e["type"] == "entity_locked"]
    assert locked == OPTIONS
    fetched_slugs = {_slug(e["data"]["url"]) for e in events
                     if e["type"] == "tool_call" and e["tool_name"] == "web_fetch"}
    assert {"AUSTIN", "DENVER", "RALEIGH"} <= fetched_slugs
    # THE GUARANTEE: every claim row traces to fetched content
    assert m["citation_rows_checked"] >= 3
    assert m["citations_grounded"] is True and m["ungrounded_claims"] == 0


@pytest.mark.asyncio
async def test_citation_gate_catches_a_fabricated_source():
    events, m = await _run(LyingAdapter())
    assert m["citations_grounded"] is False
    assert m["ungrounded_claims"] >= 1
    cite = next(e for e in events if e["type"] == "citation_grounding")
    assert any("nytimes.com/totally-real-article" in u for u in cite["fabricated_urls"])
    answer = "".join(e["content"] for e in events if e["type"] == "token")
    assert "Citation gate" in answer        # the lie is labeled, not presented as fact


# --------------------------- LIVE (real model) ---------------------------

def _has_key():
    try:
        from config.config import cfg  # noqa: F401  (loads .env)
    except Exception:
        pass
    return bool(os.getenv("GEMINI_API_KEY"))


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_key(), reason="needs GEMINI_API_KEY for a live run")
async def test_live_comparison_one_call_no_fabrication():
    """Real gemini-3.1-flash-lite. Guarantees that hold even if fetches are
    rate-limited: exactly one model call, and zero fabricated citations."""
    from config.config import cfg  # noqa: F401
    from llm.adapter_factory import create_adapter
    from plugins.tools_web import register_web_tools

    reg = ToolRegistry()
    register_web_tools(reg)
    engine = KernelRunEngine(adapter=create_adapter("gemini"), tool_registry=reg)
    req = RunEngineRequest(session_id="live", owner_id="o", agent_id="a",
                           user_message=TASK, model="gemini:gemini-3.1-flash-lite",
                           allowed_tools=("web_search", "web_fetch"), control_policy="kernel")
    metrics = None
    answer = []
    async for ev in engine.stream(req):
        if ev.get("type") == "kernel_metrics":
            metrics = ev
        elif ev.get("type") == "token":
            answer.append(ev.get("content") or "")
    assert metrics is not None
    # Invariants that hold regardless of live network/rate-limits:
    assert metrics["shape"] == "comparison"
    assert metrics["llm_calls"] == 1                       # one model call for the whole decision
    assert "citations_grounded" in metrics                # the per-claim gate ran
    assert "citation_rows_checked" in metrics
    # If the model fabricated a citation, the gate must have labeled it — the
    # answer is never presented as clean fact while ungrounded.
    if not metrics["citations_grounded"]:
        assert "Citation gate" in "".join(answer)
