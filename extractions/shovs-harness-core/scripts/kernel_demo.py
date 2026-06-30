"""Real kernel-driven run: the deterministic kernel drives live web_search/web_fetch;
Gemini is called only to (1) lock entities and (2) synthesize. Compare the LLM
call count against the orchestrator loop's ~2T+3.

Usage:  python scripts/kernel_demo.py [gemini-model]
"""
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from config.config import cfg  # noqa: E402  (triggers load_dotenv)
from llm.adapter_factory import create_adapter, strip_provider_prefix  # noqa: E402
from plugins.tools_web import _web_search, _web_fetch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the package
from shovs_harness_core import run_source_collection  # noqa: E402

OBJ = ("Search top 3 stocks today with major jumps, web search those 3 stocks separately and "
       "capture the 3 relevant results for each, web fetch all 9 urls one by one, analyze each "
       "and write a tldr summary table.")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemini-2.5-flash"
_adapter = create_adapter("gemini")
_calls = {"n": 0}


def _urls_from_search(raw: str) -> list[str]:
    urls: list[str] = []
    try:
        d = json.loads(raw)
        for r in (d.get("results") or []):
            u = r.get("url") or r.get("normalized_url")
            if u:
                urls.append(u)
    except Exception:
        pass
    if not urls:
        urls = re.findall(r"https?://[^\s\"'<>)]+", raw)
    seen, out = set(), []
    for u in urls:
        if u not in seen and "google.com/search" not in u:
            seen.add(u); out.append(u)
    return out


async def search_fn(query):
    raw = await _web_search(query, num_results=10)
    urls = _urls_from_search(raw)
    text = raw
    try:
        d = json.loads(raw)
        parts = [f"{r.get('title','')} — {r.get('snippet','')}".strip(" —") for r in (d.get("results") or [])[:10]]
        parts = [p for p in parts if p]
        if parts:
            text = "\n".join(parts)
    except Exception:
        pass
    return urls, text


async def fetch_fn(url):
    try:
        raw = await _web_fetch(url, max_chars=3000)
        d = json.loads(raw)
    except Exception as e:
        return (False, "")
    if d.get("error") or str(d.get("type", "")).endswith("error"):
        return (False, "")
    return (True, str(d.get("content") or "")[:3000])


async def _llm(prompt: str, max_tokens: int = 600) -> str:
    _calls["n"] += 1
    return await _adapter.complete(
        model=strip_provider_prefix(MODEL),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
    )


_STOP = {"THE", "AND", "FOR", "USA", "CEO", "ETF", "NYSE", "USD", "NEWS", "TOP", "INC",
         "LLC", "API", "IPO", "SEC", "EPS", "CNBC", "WSJ", "ET", "EST", "UTC", "YTD",
         "Q1", "Q2", "Q3", "Q4", "AI", "US", "PR", "ESG"}


async def extract_entities_fn(objective, discovery_text, n):
    txt = await _llm(
        f"From these stock-market search results, identify the {n} US stock TICKER symbols of "
        f"today's biggest gainers/movers.\nResults:\n{discovery_text[:4000]}\n\n"
        f'Output ONLY a JSON array of exactly {n} tickers like ["AAA","BBB","CCC"]. '
        f"If unsure, pick the {n} most prominent tickers mentioned.",
        max_tokens=120,
    )
    out: list[str] = []
    m = re.search(r"\[.*\]", txt, re.DOTALL)
    if m:
        try:
            out = [str(x).strip().upper() for x in json.loads(m.group(0))]
        except Exception:
            out = []
    if not out:
        out = re.findall(r"\b[A-Z]{2,5}\b", txt)                      # from the LLM reply
    if not out:
        out = re.findall(r"\(([A-Z]{2,5})\)", discovery_text) or \
            re.findall(r"\b[A-Z]{2,5}\b", discovery_text)             # deterministic fallback
    seen, res = set(), []
    for t in out:
        if t and t not in _STOP and t not in seen:
            seen.add(t); res.append(t)
    print(f"[extract] discovery={len(discovery_text)}ch llm={txt[:60]!r} -> {res[:n]}")
    return res[:n]


async def synth_fn(objective, sources):
    if not sources:
        return ("(No sources were successfully fetched — nothing to synthesize. The harness "
                "reports this honestly instead of inventing data.)")
    blob = "\n\n".join(f"[{s['entity']}] {s['url']}\n{s['content'][:1200]}" for s in sources)[:14000]
    return await _llm(
        "You are a careful equity research assistant. Using ONLY the fetched sources below, "
        "write a concise TLDR markdown table with columns: Ticker | Why it moved | Source. "
        "One row per ticker. Then one sentence of overall takeaway. Do NOT invent tickers, "
        "URLs, or facts not present in the sources.\n\n"
        f"Task: {objective}\n\nSources:\n{blob}",
        max_tokens=900,
    )


async def main():
    print(f"=== KERNEL-DRIVEN REAL RUN — model={MODEL} ===\nTASK: {OBJ}\n")
    t0 = time.time()
    r = await run_source_collection(
        OBJ, search_fn=search_fn, fetch_fn=fetch_fn,
        extract_entities_fn=extract_entities_fn, synth_fn=synth_fn,
    )
    dt = time.time() - t0
    print(f"contract      : {r.contract.entity_count} entities x {r.contract.urls_per_entity} urls = {r.contract.total_urls}")
    print(f"locked        : {r.entities}")
    print(f"fetched ok    : {len(r.fetched)}/{r.contract.total_urls}")
    print(f"tool calls    : {r.tool_calls}  (deterministic)")
    print(f"LLM calls     : {r.llm_calls}   (vs orchestrator loop ~2*{r.contract.total_urls}+3 = {2*r.contract.total_urls+3})")
    print(f"contract eval : score={r.eval.score} ok={r.eval.ok} failures={r.eval.failures}")
    print(f"wall time     : {dt:.1f}s\n")
    print("=== ANSWER ===")
    print(r.answer.strip() or "(empty)")


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), timeout=220))
