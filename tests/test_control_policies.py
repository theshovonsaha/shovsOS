from run_engine.control_policies import resolve_control_policy
from run_engine.ledger import RunLedger
from run_engine.workflow_contracts import infer_workflow_contract
from run_engine.workflow_plugins import stock_source_workflow_override
from run_engine.engine import _canonical_tool_arguments_for_loop


def test_source_collection_auto_uses_plan_execute_policy():
    contract = infer_workflow_contract(
        "Search top 3 stocks today, search each, fetch 3 URLs each.",
        allowed_tools=["web_search", "web_fetch"],
    )

    policy = resolve_control_policy(
        contract.objective,
        requested="auto",
        workflow_contract=contract,
        allowed_tools=["web_search", "web_fetch"],
    )

    assert policy.id == "plan_execute"
    assert policy.mutable_plan is False
    assert policy.planner_required is True


def test_explicit_react_policy_is_supported_for_lightweight_runs():
    policy = resolve_control_policy("hi", requested="react")

    assert policy.id == "react"
    assert policy.loop_shape == "reason_act_observe"
    assert policy.planner_required is False


def test_high_risk_auto_uses_graph_harness():
    policy = resolve_control_policy(
        "Implement this backend change and run tests.",
        requested="auto",
        risk_policy="high",
        allowed_tools=["file_view", "file_str_replace", "bash"],
    )

    assert policy.id == "graph_harness"
    assert policy.max_recovery_rounds == 1


def test_control_policy_is_in_ledger_packet_and_attention():
    ledger = RunLedger(
        run_id="run-policy",
        session_id="session-policy",
        turn_id="turn-policy",
        objective="Search top 3 stocks today, search each, fetch 3 URLs each.",
        allowed_tools=["web_search", "web_fetch"],
    )
    contract = infer_workflow_contract(ledger.objective, allowed_tools=ledger.allowed_tools)
    policy = resolve_control_policy(ledger.objective, workflow_contract=contract, allowed_tools=ledger.allowed_tools)

    ledger.set_workflow_contract(contract)
    ledger.set_control_policy(policy)
    packet = ledger.to_phase_packet("planning")

    assert packet["control_policy"]["id"] == "plan_execute"
    assert any(item["kind"] == "control_policy" for item in packet["runtime_attention"]["items"])


def test_stock_workflow_pivots_after_unproductive_movers_fetch():
    override = stock_source_workflow_override(
        objective=(
            "Search top 3 stocks today with major jumps web search those 3 stocks separately "
            "and capture the 3 relevant for each results web fetch all 9 urls."
        ),
        allowed_tools=[{"name": "web_search"}, {"name": "web_fetch"}],
        tool_results=[
            {
                "tool_name": "web_search",
                "success": True,
                "arguments": {"query": "top 3 stocks today with major jumps"},
                "content": (
                    '{"type":"web_search_results","results":['
                    '{"title":"Top Gaining US Stocks - TradingView","url":"https://www.tradingview.com/markets/stocks-usa/market-movers-gainers"},'
                    '{"title":"Top Stock Gainers Today - Yahoo Finance","url":"https://finance.yahoo.com/markets/stocks/gainers"},'
                    '{"title":"Today Top Stock Gainers - Stock Analysis","url":"https://stockanalysis.com/markets/gainers"}'
                    ']}'
                ),
                "extracted_urls": [
                    "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers",
                    "https://finance.yahoo.com/markets/stocks/gainers",
                    "https://stockanalysis.com/markets/gainers",
                ],
            },
            {
                "tool_name": "web_fetch",
                "success": True,
                "arguments": {"url": "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers"},
                "content": '{"type":"web_fetch_result","url":"https://www.tradingview.com/markets/stocks-usa/market-movers-gainers","content":"dynamic page no ticker table captured"}',
            },
        ],
    )

    assert override["selected_tools"] == ["web_fetch"]
    assert override["argument_clues"]["web_fetch"] == "https://finance.yahoo.com/markets/stocks/gainers"
    assert "pivot" in override["notes"]


def test_ledger_completion_gate_requires_per_entity_fetch_coverage():
    ledger = RunLedger(
        run_id="run-coverage",
        session_id="session-coverage",
        turn_id="turn-coverage",
        objective="Search top 3 stocks, search each, fetch 3 URLs each.",
        allowed_tools=["web_search", "web_fetch"],
        ledger_mode="ledger_enforced",
    )
    ledger.set_source_contract(
        {
            "total_urls": 6,
            "urls_per_entity": 3,
            "allowed_fetch_urls_by_entity": {
                "ROKU": ["https://news.example/roku-1", "https://news.example/roku-2", "https://news.example/roku-3"],
                "TBN": ["https://news.example/tbn-1", "https://news.example/tbn-2", "https://news.example/tbn-3"],
            },
        }
    )
    for url in [
        "https://news.example/roku-1",
        "https://news.example/roku-2",
        "https://news.example/roku-3",
        "https://news.example/tbn-1",
    ]:
        call = ledger.add_tool_call("web_fetch", {"url": url})
        ledger.link_tool_result(
            tool_call_id=call.id,
            tool_name="web_fetch",
            success=True,
            status="ok",
            summary=url,
        )

    gate = ledger.completion_gate()

    assert gate["final_answer_allowed"] is False
    assert "TBN_fetched_urls:1/3" in gate["missing_slots"]
    assert ledger.next_required_action()["missing_slots"] == gate["missing_slots"]


def test_fetch_url_loop_arguments_are_canonicalized():
    args = _canonical_tool_arguments_for_loop(
        "web_fetch",
        {"url": "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/"},
    )

    assert args["url"] == "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers"


def test_web_search_loop_arguments_are_compiled_from_workflow_request():
    args = _canonical_tool_arguments_for_loop(
        "web_search",
        {
            "query": (
                "Search top 3 stocks today with major jumps web search those 3 stocks separately "
                "and capture the 3 relevant for each results web fetch all 9 urls one by one "
                "analyze report and write a tldr summary table."
            )
        },
    )

    assert args["query"] == "top 3 stocks today with major jumps"
