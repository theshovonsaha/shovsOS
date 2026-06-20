import json

import pytest

from plugins.tool_registry import ToolRegistry
from plugins.tools import (
    _format_search_results,
    _normalize_search_results,
    _source_contract,
    _source_next_action,
    _source_select,
    register_tools,
)
from plugins import tools_web


def test_normalize_search_results_dedupes_and_fills_missing_snippets():
    raw = [
        {"title": "Result 1", "url": "https://example.com/a", "snippet": "", "source": "duckduckgo"},
        {"title": "Result 1 duplicate", "url": "https://example.com/a/", "snippet": "Duplicate", "source": "duckduckgo"},
        {"title": "", "url": "https://example.com/b", "snippet": "  Useful   context   text  ", "source": "duckduckgo"},
    ]

    normalized = _normalize_search_results(raw, max_results=5)

    assert len(normalized) == 2
    assert normalized[0]["snippet"] == "Result 1"
    assert normalized[1]["snippet"] == "Useful context text"


def test_format_search_results_includes_engine_and_context_summary():
    raw = [
        {"title": "R1", "url": "https://example.com/1", "snippet": "Alpha fact", "source": "duckduckgo"},
        {"title": "R2", "url": "https://example.com/2", "snippet": "Beta fact", "source": "duckduckgo"},
    ]

    payload = json.loads(_format_search_results("test query", raw, engine="duckduckgo"))

    assert payload["type"] == "web_search_results"
    assert payload["engine"] == "duckduckgo"
    assert "Alpha fact" in payload["context_summary"]
    assert len(payload["results"]) == 2


@pytest.mark.asyncio
async def test_source_contract_extracts_multi_entity_fetch_workflow():
    raw = await _source_contract(
        "Find top 3 stocks today, search each separately, capture 3 relevant URLs for each, fetch all 9 URLs.",
        topic="stock news June 13 2026",
    )
    payload = json.loads(raw)

    assert payload["type"] == "source_contract"
    assert payload["contract"]["entity_count"] == 3
    assert payload["contract"]["results_per_entity"] == 3
    assert payload["contract"]["total_fetches"] == 9
    assert payload["contract"]["needs_separate_queries"] is True
    assert "fetch_selected_urls" in payload["contract"]["workflow"]


@pytest.mark.asyncio
async def test_source_select_groups_urls_by_locked_entities_and_dedupes():
    search_payloads = [
        {
            "query": "ROKU stock news June 13 2026",
            "results": [
                {"title": "Roku 1", "url": "https://news.example/roku-1", "host": "news.example", "rank": 1},
                {"title": "Roku duplicate", "url": "https://news.example/roku-1", "host": "news.example", "rank": 2},
                {"title": "Roku 2", "url": "https://analysis.example/roku-2", "host": "analysis.example", "rank": 3},
            ],
        },
        {
            "query": "TBN stock news June 13 2026",
            "results": [
                {"title": "TBN 1", "url": "https://news.example/tbn-1", "host": "news.example", "rank": 1},
            ],
        },
    ]

    payload = json.loads(await _source_select(search_payloads=search_payloads, entities=["ROKU", "TBN"], per_entity=2))

    assert payload["type"] == "source_selection"
    assert payload["success"] is True
    assert payload["selected_by_entity"]["ROKU"][0]["url"] == "https://news.example/roku-1"
    assert len(payload["selected_by_entity"]["ROKU"]) == 2
    assert payload["coverage"]["missing_entities"] == ["TBN"]
    assert payload["selected_urls"].count("https://news.example/roku-1") == 1


@pytest.mark.asyncio
async def test_source_next_action_moves_from_search_to_fetch_to_finalize():
    objective = "Search each of 2 products and fetch 1 URL per product."
    contract = {"entity_count": 2, "results_per_entity": 1, "total_fetches": 2}

    next_search = json.loads(await _source_next_action(
        objective,
        contract=contract,
        entities=["vacuum", "mop"],
        searched_queries=["vacuum reviews"],
        topic="reviews",
    ))
    assert next_search["next_tool"] == "web_search"
    assert next_search["arguments"]["query"] == "mop reviews"

    next_fetch = json.loads(await _source_next_action(
        objective,
        contract=contract,
        entities=["vacuum", "mop"],
        searched_queries=["vacuum reviews", "mop reviews"],
        selected_urls=["https://example.com/a", "https://example.com/b"],
        fetched_urls=["https://example.com/a"],
        topic="reviews",
    ))
    assert next_fetch["next_tool"] == "web_fetch"
    assert next_fetch["arguments"]["url"] == "https://example.com/b"

    done = json.loads(await _source_next_action(
        objective,
        contract=contract,
        entities=["vacuum", "mop"],
        searched_queries=["vacuum reviews", "mop reviews"],
        selected_urls=["https://example.com/a", "https://example.com/b"],
        fetched_urls=["https://example.com/a", "https://example.com/b"],
        topic="reviews",
    ))
    assert done["status"] == "finalize"


def test_kernel_source_tools_register_by_name():
    registry = ToolRegistry()
    register_tools(registry, "source_contract", "source_select", "source_next_action")

    assert registry.get("source_contract") is not None
    assert registry.get("source_select") is not None
    assert registry.get("source_next_action") is not None


def test_web_search_curation_adds_stable_result_metadata():
    results, summary = tools_web._curate_results(
        [
            {"title": "Alpha", "url": "https://Example.com/post?utm_source=x&id=1#top", "snippet": "Useful"},
            {"title": "Beta", "url": "https://other.example/item", "snippet": "Useful"},
        ],
        2,
    )

    assert summary["curated_results"] == 2
    assert results[0]["rank"] == 1
    assert results[0]["host"] == "example.com"
    assert results[0]["normalized_url"] == "https://example.com/post?id=1"
