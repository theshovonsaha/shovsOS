import json

from run_engine.workflow_plugins import (
    WORKFLOW_PLUGINS,
    select_workflow_override,
    select_workflow_plugin_contract,
    source_collection_contract_from_objective,
)


def test_workflow_plugins_register_stock_movers_without_engine_hardcoding():
    assert any(plugin.id == "stock_movers_source_collection" for plugin in WORKFLOW_PLUGINS)


def test_select_workflow_override_uses_plugin_registry():
    objective = (
        "Search top 3 stocks today with major jumps web search those 3 stocks separately "
        "and capture the 3 relevant for each results web fetch all 9 urls one by one."
    )
    search_result = {
        "tool_name": "web_search",
        "success": True,
        "arguments": {"query": "top 3 stocks today with major jumps"},
        "content": json.dumps({
            "type": "web_search_results",
            "results": [
                {
                    "title": "Top Gaining US Stocks - TradingView",
                    "url": "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers",
                    "snippet": "MEIMethode Electronics, Inc. D · +21.18%",
                }
            ],
        }),
    }

    override = select_workflow_override(
        objective=objective,
        allowed_tools=[{"name": "web_search"}, {"name": "web_fetch"}],
        tool_results=[search_result],
    )

    assert override["plugin_id"] == "stock_movers_source_collection"
    assert override["selected_tools"] == ["web_fetch"]
    assert override["argument_clues"]["web_fetch"] == "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers"


def test_source_contract_remains_topic_agnostic_in_plugin_layer():
    assert source_collection_contract_from_objective(
        "Find top 3 sushi places, search each, fetch 3 URLs each."
    ) == {
        "entity_count": 3,
        "urls_per_entity": 3,
        "total_urls": 9,
    }


def test_local_and_comparison_plugins_expose_typed_contracts():
    local = select_workflow_plugin_contract("top 3 sushi places in Toronto, search each, fetch 3 URLs each")
    comparison = select_workflow_plugin_contract("compare 3 products, search each, fetch 2 URLs each")

    assert local["plugin_id"] == "local_place_source_collection"
    assert local["preferred_policy"] == "plan_execute"
    assert local["entity_lock_rules"]["forbid_unlocked_entity_drift"] is True
    assert comparison["plugin_id"] == "comparison_source_collection"
    assert comparison["completion_evaluator"] == "source_collection_scenario"


def test_generic_source_collection_discovery_query_is_compact():
    override = select_workflow_override(
        objective=(
            "Search top 3 sushi places in Toronto, then search each separately, "
            "capture 3 relevant results for each, web fetch all 9 URLs, and write a TLDR table."
        ),
        allowed_tools=[{"name": "web_search"}, {"name": "web_fetch"}],
        tool_results=[],
    )

    assert override["plugin_id"] == "local_place_source_collection"
    assert override["selected_tools"] == ["web_search"]
    assert override["argument_clues"]["web_search"] == "top 3 sushi places in Toronto"
