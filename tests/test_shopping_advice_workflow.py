import json
from unittest.mock import AsyncMock

import pytest

from orchestration.capability_cards import render_capability_cards
from orchestration.workflow_templates import get_workflow_template
from orchestration.workflow_patterns import get_workflow_pattern
from plugins.tools import SHOPPING_ADVICE_TOOL, _shopping_advice
from run_engine.engine import RunEngine


@pytest.mark.asyncio
async def test_shopping_advice_returns_verified_answer_patch(monkeypatch):
    async def fake_search(query: str, num_results: int = 8, **kwargs):
        return json.dumps({
            "type": "web_search_results",
            "query": query,
            "results": [
                {
                    "title": "Example Laptop Pro",
                    "url": "https://shop.example/laptop-pro",
                    "snippet": "Example Laptop Pro under $900 with 4.6/5 stars.",
                }
            ],
        })

    async def fake_fetch(url: str, max_chars: int = 8000, **kwargs):
        return json.dumps({
            "type": "web_fetch_result",
            "url": url,
            "content": "Example Laptop Pro costs $849. It has 4.6/5 stars, long battery life, and limited warranty.",
        })

    monkeypatch.setattr("plugins.tools._web_search", fake_search)
    monkeypatch.setattr("plugins.tools._web_fetch", fake_fetch)

    raw = await _shopping_advice(
        query="laptop for school",
        budget="$900",
        priorities=["battery life", "warranty"],
        region="US",
    )
    payload = json.loads(raw)

    assert payload["type"] == "shopping_advice_result"
    assert payload["success"] is True
    assert payload["verified_urls"] == ["https://shop.example/laptop-pro"]
    assert payload["answer_patch"]["recommendation"]["prices"] == ["$849"]
    assert payload["answer_patch"]["recommendation"]["ratings"] == ["4.6/5"]
    assert any("Verified URL" in line for line in payload["answer_patch"]["must_say"])


@pytest.mark.asyncio
async def test_shopping_advice_scopes_to_location_and_preferred_canadian_stores(monkeypatch):
    seen_queries: list[str] = []

    async def fake_search(query: str, num_results: int = 8, **kwargs):
        seen_queries.append(query)
        if "canadiantire.ca" in query:
            url = "https://www.canadiantire.ca/en/pdp/storage-bin.html"
            title = "Canadian Tire Storage Bin"
        elif "walmart.ca" in query:
            url = "https://www.walmart.ca/en/ip/storage-bin"
            title = "Walmart Storage Bin"
        else:
            url = "https://www.canadiantire.ca/en/pdp/storage-bin.html"
            title = "Storage bin"
        return json.dumps({
            "type": "web_search_results",
            "query": query,
            "results": [{"title": title, "url": url, "snippet": "Storage bin $12.99 with 4.4/5 rating."}],
        })

    async def fake_fetch(url: str, max_chars: int = 8000, **kwargs):
        return json.dumps({
            "type": "web_fetch_result",
            "url": url,
            "title": url,
            "content": "Storage bin price $12.99. Customer rating 4.4/5. Return policy available.",
        })

    monkeypatch.setattr("plugins.tools._web_search", fake_search)
    monkeypatch.setattr("plugins.tools._web_fetch", fake_fetch)

    payload = json.loads(await _shopping_advice(
        query="storage bin",
        budget="$20",
        location="Toronto",
        region="Canada",
        stores=["Canadian Tire", "Walmart"],
        max_candidates=3,
    ))

    assert any("near Toronto" in query for query in seen_queries)
    assert any("site:canadiantire.ca" in query for query in seen_queries)
    assert any("site:walmart.ca" in query for query in seen_queries)
    assert payload["stores_requested"] == ["Canadian Tire", "Walmart Canada"]
    assert payload["answer_patch"]["comparison_table"]
    assert payload["answer_patch"]["comparison_table"][0]["store"] in {"Canadian Tire", "Walmart Canada"}


def test_shopping_advisor_template_uses_patch_workflow_tool():
    template = get_workflow_template("shopping_advisor_v1")

    assert "shopping_advice" in template.tools
    assert template.prompt_version == "shopping_patch_v1"
    assert template.risk_policy == "consumer_verified"
    assert template.workflow_pattern == "diverge_converge_patch_v1"


def test_diverge_converge_patch_pattern_has_runtime_plan_spine():
    pattern = get_workflow_pattern("diverge_converge_patch_v1")

    assert pattern is not None
    assert any("DIVERGE" in stage for stage in pattern.stages)
    assert any(step["tool"] == "shopping_advice" for step in pattern.default_plan_steps)
    assert "verified URLs" in " ".join(pattern.response_contract)


def test_capability_cards_render_shopping_pattern_for_llm():
    text = render_capability_cards(
        allowed_tools=["shopping_advice"],
        workflow_template="shopping_advisor_v1",
    )

    assert "Local Store Shopping Advisor" in text
    assert "DIVERGE" in text
    assert "answer_patch" in text
    assert "verified_urls" in text


def test_run_engine_repairs_shopping_arguments_from_user_objective():
    args = RunEngine._normalize_shopping_arguments(
        {"query": "storage bin"},
        objective="Find a storage bin under $20 near Toronto at Canadian Tire, Walmart, or Dollarama.",
    )

    assert args["budget"] == "$20"
    assert args["location"] == "Toronto"
    assert args["region"] == "Canada"
    assert args["stores"] == ["Canadian Tire", "Dollarama", "Walmart"]


def test_run_engine_extracts_shopping_answer_patch():
    payload = {
        "type": "shopping_advice_result",
        "verified_urls": ["https://shop.example/laptop-pro"],
        "warnings": [],
        "answer_patch": {
            "format": "concise_buyer_advice_v1",
            "recommendation": {"title": "Example Laptop Pro", "prices": ["$849"]},
            "must_say": ["Best verified lead: Example Laptop Pro"],
        },
    }

    patch = RunEngine._extract_answer_patch([
        {
            "tool_name": "shopping_advice",
            "success": True,
            "content": json.dumps(payload),
        }
    ])

    assert patch is not None
    assert patch["verified_urls"] == ["https://shop.example/laptop-pro"]
    assert patch["recommendation"]["prices"] == ["$849"]


def test_shopping_advice_tool_contract_is_json_tool():
    assert SHOPPING_ADVICE_TOOL.name == "shopping_advice"
    assert SHOPPING_ADVICE_TOOL.response_format == "json"
    assert "query" in SHOPPING_ADVICE_TOOL.parameters["required"]
