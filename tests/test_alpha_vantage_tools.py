import json

import pytest

from plugins.finance_alpha_vantage import (
    ALPHA_VANTAGE_TOOLS,
    alpha_vantage_movers,
    finance_snapshot,
)
from plugins.tool_registry import ToolCall, ToolRegistry
from plugins.tools import register_all_tools
from run_engine.engine import RunEngine
from run_engine.tool_selection import fallback_tool_call
from run_engine.turn_policy import resolve_turn_policy


@pytest.mark.asyncio
async def test_finance_snapshot_builds_report_patch(monkeypatch):
    async def fake_alpha_get(function, *, api_key="", **params):
        if function == "GLOBAL_QUOTE":
            return {
                "_success": True,
                "Global Quote": {
                    "01. symbol": params["symbol"],
                    "05. price": "132.50",
                    "10. change percent": "2.10%",
                    "06. volume": "123456",
                    "07. latest trading day": "2026-06-30",
                },
            }
        if function == "OVERVIEW":
            return {
                "_success": True,
                "Symbol": params["symbol"],
                "Name": "Airbnb Inc",
                "Sector": "Consumer Cyclical",
                "Industry": "Travel Services",
                "MarketCapitalization": "86000000000",
            }
        if function == "NEWS_SENTIMENT":
            return {
                "_success": True,
                "feed": [
                    {
                        "title": "Airbnb test news",
                        "url": "https://example.com/abnb",
                        "source": "Example",
                        "summary": "A concise test summary.",
                    }
                ],
            }
        return {"_success": False, "_error": "unexpected"}

    monkeypatch.setattr("plugins.finance_alpha_vantage._alpha_get", fake_alpha_get)

    payload = json.loads(await finance_snapshot("ABNB", news_limit=1))

    assert payload["success"] is True
    assert payload["symbol"] == "ABNB"
    assert payload["quote"]["price"] == "132.50"
    assert payload["overview"]["Name"] == "Airbnb Inc"
    assert payload["answer_patch"]["source_urls"] == ["https://example.com/abnb"]
    assert any("last Alpha Vantage quote" in line for line in payload["answer_patch"]["must_say"])


@pytest.mark.asyncio
async def test_alpha_vantage_movers_locks_top_gainers(monkeypatch):
    async def fake_alpha_get(function, *, api_key="", **params):
        assert function == "TOP_GAINERS_LOSERS"
        return {
            "_success": True,
            "last_updated": "2026-06-30 16:15:00",
            "top_gainers": [
                {"ticker": "AAA", "price": "10", "change_percentage": "20%"},
                {"ticker": "BBB", "price": "11", "change_percentage": "18%"},
                {"ticker": "CCC", "price": "12", "change_percentage": "15%"},
            ],
        }

    monkeypatch.setattr("plugins.finance_alpha_vantage._alpha_get", fake_alpha_get)

    payload = json.loads(await alpha_vantage_movers(limit=3))

    assert payload["success"] is True
    assert [row["symbol"] for row in payload["top_gainers"]] == ["AAA", "BBB", "CCC"]
    assert payload["answer_patch"]["locked_entities"] == ["AAA", "BBB", "CCC"]


def test_alpha_vantage_tools_are_registered():
    registry = ToolRegistry()
    register_all_tools(registry)
    names = {tool["name"] for tool in registry.list_tools()}

    assert {tool.name for tool in ALPHA_VANTAGE_TOOLS}.issubset(names)


def test_finance_fallback_prefers_deterministic_tools_before_web():
    ranked = RunEngine._rank_fallback_tool_candidates(
        "ABNB fundamentals report with news",
        ["web_search", "finance_snapshot", "alpha_vantage_quote", "alpha_vantage_overview", "alpha_vantage_news"],
    )

    assert ranked[:2] == ["finance_snapshot", "alpha_vantage_quote"]
    assert "web_search" in ranked


def test_finance_snapshot_phrase_prefers_deterministic_tool():
    ranked = RunEngine._rank_fallback_tool_candidates(
        "Create a compact ABNB finance snapshot",
        ["web_search", "finance_snapshot"],
    )

    assert ranked[0] == "finance_snapshot"


def test_top_stock_movers_policy_keeps_alpha_vantage_tools():
    policy = resolve_turn_policy(
        "Search top 3 stocks today with major jumps and fetch 3 URLs each",
        user_message="Search top 3 stocks today with major jumps and fetch 3 URLs each",
        workflow_shape="source_collection",
    )
    constrained = policy.constrain(["web_search", "web_fetch", "alpha_vantage_movers", "finance_snapshot"])

    assert "alpha_vantage_movers" in constrained
    assert "web_search" in constrained


def test_finance_fallback_tool_call_extracts_symbol():
    call = fallback_tool_call("finance_snapshot", "Create an ABNB fundamentals report")

    assert isinstance(call, ToolCall)
    assert call.arguments == {"symbol": "ABNB"}


def test_finance_snapshot_answer_patch_is_extracted():
    patch = RunEngine._extract_answer_patch([
        {
            "tool_name": "finance_snapshot",
            "content": json.dumps({
                "provider": "alpha_vantage",
                "symbol": "ABNB",
                "answer_patch": {
                    "format": "finance_snapshot_v1",
                    "must_say": ["ABNB quote available."],
                    "source_urls": ["https://example.com/news"],
                },
            }),
        }
    ])

    assert patch is not None
    assert patch["symbol"] == "ABNB"
    assert patch["source_urls"] == ["https://example.com/news"]
