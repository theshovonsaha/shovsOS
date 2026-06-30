"""Kernel-driven source collection: deterministic structure, 2 LLM calls."""

import pytest

from shovs_harness_core import discovery_query, entity_search_query, run_source_collection


def test_discovery_query_strips_workflow_words():
    q = discovery_query("Search top 3 stocks today with major jumps, web search those 3 separately, fetch 3 urls each and write a tldr table.")
    assert "fetch" not in q.lower() and "tldr" not in q.lower() and "top 3" not in q.lower()
    assert "stock" in q.lower()
    word_number = discovery_query("Find top three laptops for students, search each, fetch three articles each.")
    assert "top three" not in word_number.lower()
    assert word_number == "laptops for students"


def test_entity_search_query_is_topic_agnostic():
    assert entity_search_query("Search top 3 stocks today, fetch 3 URLs each.", "ROKU") == "ROKU stock news"
    assert entity_search_query("Find top 3 sushi places in Toronto, fetch 3 URLs each.", "Miku") == "Miku reviews sources"
    assert entity_search_query("Compare 3 laptops by price and reviews.", "ThinkPad") == "ThinkPad reviews price sources"
    assert entity_search_query("Find top 3 papers about agent traces.", "AgentTrace") == "AgentTrace research paper sources"


@pytest.mark.asyncio
async def test_kernel_runs_full_source_collection_with_two_llm_calls():
    OBJ = "Search top 3 stocks today, search each separately, fetch 3 urls each, write a tldr table."
    llm = {"n": 0}

    async def search_fn(query):
        # entity searches return 4 candidate urls; discovery returns the movers
        e = query.split()[0]
        return ([f"https://src.test/{e}/{i}" for i in range(4)], f"movers: AAA BBB CCC for '{query}'")

    async def fetch_fn(url):
        return (True, f"content of {url}")

    async def extract_entities_fn(objective, discovery_text, n):
        llm["n"] += 1
        return ["AAA", "BBB", "CCC", "DDD"]  # runner clamps to n

    async def synth_fn(objective, sources):
        llm["n"] += 1
        return f"| ticker | sources |\n" + "\n".join(f"| {s['entity']} | {s['url']} |" for s in sources)

    r = await run_source_collection(
        OBJ, search_fn=search_fn, fetch_fn=fetch_fn,
        extract_entities_fn=extract_entities_fn, synth_fn=synth_fn,
    )

    assert r.contract.entity_count == 3 and r.contract.urls_per_entity == 3 and r.contract.total_urls == 9
    assert r.entities == ["AAA", "BBB", "CCC"]            # locked, clamped to N
    assert len(r.fetched) == 9                            # exactly the quota, no drift
    assert {s["entity"] for s in r.fetched} == {"AAA", "BBB", "CCC"}
    assert r.llm_calls == 2 and llm["n"] == 2             # only extract + synthesize
    assert r.tool_calls == 1 + 3 + 9                      # contract calls: discovery + per-entity search + fetches
    assert r.eval is not None and r.eval.ok and r.eval.score == 1.0
    assert "| AAA |" in r.answer


@pytest.mark.asyncio
async def test_kernel_cannot_drift_to_off_list_entity():
    OBJ = "Find top 2 stocks, search each, fetch 2 urls each."

    async def search_fn(query):
        return ([f"https://x.test/{query.split()[0]}/{i}" for i in range(3)], "AAA BBB CCC")

    async def fetch_fn(url):
        return (True, "ok")

    async def extract_entities_fn(objective, discovery_text, n):
        return ["AAA", "BBB"]

    async def synth_fn(objective, sources):
        return "table"

    r = await run_source_collection(
        OBJ, search_fn=search_fn, fetch_fn=fetch_fn,
        extract_entities_fn=extract_entities_fn, synth_fn=synth_fn,
    )
    # every fetched source belongs to a locked entity — drift is structurally impossible
    assert {s["entity"] for s in r.fetched} <= {"AAA", "BBB"}
    assert len(r.fetched) == 4  # 2 entities x 2 urls
    assert r.eval.ok


@pytest.mark.asyncio
async def test_kernel_local_places_do_not_use_stock_news_queries():
    OBJ = "Find top 2 sushi places in Toronto, search each separately, fetch 2 urls each."
    searched: list[str] = []

    async def search_fn(query):
        searched.append(query)
        if len(searched) == 1:
            return (["https://places.test/discovery"], "Miku Sushi Masaki Saito")
        slug = query.split()[0]
        return ([f"https://places.test/{slug}/{i}" for i in range(2)], query)

    async def fetch_fn(url):
        return (True, "review source")

    async def extract_entities_fn(objective, discovery_text, n):
        return ["Miku", "Masaki Saito"]

    async def synth_fn(objective, sources):
        return "done"

    r = await run_source_collection(
        OBJ,
        search_fn=search_fn,
        fetch_fn=fetch_fn,
        extract_entities_fn=extract_entities_fn,
        synth_fn=synth_fn,
    )

    assert all("stock news" not in query.lower() for query in searched)
    assert searched[1:] == ["MIKU reviews sources", "MASAKI SAITO reviews sources"]
    assert len(r.fetched) == 4
    assert r.eval.ok
