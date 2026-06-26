from shovs_harness_core import HarnessKernel, ToolCall, evaluate_trace, infer_source_contract, select_attention
from shovs_harness_core.attention import AttentionItem
from shovs_harness_core.ledger import Ledger


def test_source_contract_is_topic_agnostic():
    cases = [
        "Search top 3 stocks today, then search each, fetch 3 URLs each.",
        "Find top 3 sushi places in Toronto, search each, fetch 3 URLs each.",
        "Find top three laptops for students, search each, fetch three articles each.",
    ]
    for text in cases:
        contract = infer_source_contract(text)
        assert contract.entity_count == 3
        assert contract.urls_per_entity == 3
        assert contract.total_urls == 9
        assert contract.required_tools == ["web_search", "web_fetch"]


def test_ledger_rejects_tool_result_without_tool_call():
    ledger = Ledger(objective="test", allowed_tools=["web_search"])
    try:
        ledger.add_result("call_fake", True, {"x": 1})
    except ValueError as exc:
        assert "orphan tool result" in str(exc)
    else:
        raise AssertionError("orphan tool result was accepted")


def test_ledger_rejects_claims_without_successful_result_refs():
    ledger = Ledger(objective="test", allowed_tools=["web_search"])
    call = ledger.add_call("web_search", {"query": "x"})
    result = ledger.add_result(call.id, False, {"error": "rate_limited"})
    try:
        ledger.assert_claim_refs([result.id])
    except ValueError as exc:
        assert "non-successful" in str(exc)
    else:
        raise AssertionError("failed tool result was claimable")


def test_attention_changes_by_phase_and_risk():
    items = [
        AttentionItem("objective", "objective", "fetch 9 URLs"),
        AttentionItem("contract", "contract", "3 entities x 3 urls"),
        AttentionItem("gap", "pending", "need 6 more fetches", "missing"),
        AttentionItem("old", "evidence", "old evidence", "done"),
    ]
    act = select_attention(items, "act")
    respond = select_attention(items, "respond")
    assert act[0][1].id == "gap"
    assert respond[0][1].id == "gap"
    assert all(item.id != "old" or score < 0.5 for score, item in respond)


def test_kernel_continues_until_fetch_quota_is_met():
    kernel = HarnessKernel("Find top 3 sushi places, search each, fetch 3 URLs each.")
    first = kernel.decide()
    assert first.state == "act"
    assert first.next_tool == "web_search"
    kernel.add_tool_result("web_search", {"query": "sushi"}, True, {"results": []}, "searched sushi")
    second = kernel.decide()
    assert second.state == "act"
    assert second.next_tool == "web_fetch"
    for idx in range(9):
        kernel.add_tool_result("web_fetch", {"url": f"https://source.test/{idx}"}, True, {"body": "ok"}, f"fetched {idx}")
    final = kernel.decide()
    assert final.state == "respond"
    assert final.reason == "source quota met"


def test_trace_eval_catches_entity_drift_and_missing_fetches():
    contract = infer_source_contract("Search top 3 stocks, search each, fetch 3 URLs each.")
    bad_trace = [
        {"kind": "entity_locked", "entity": "ROKU"},
        {"kind": "entity_locked", "entity": "TBN"},
        {"kind": "entity_locked", "entity": "SENEA"},
        {"tool": "web_search", "entity": "EPAM"},
        {"tool": "web_fetch", "entity": "ROKU"},
    ]
    report = evaluate_trace(contract, bad_trace)
    assert not report.ok
    assert "missing_fetch_quota" in report.failures
    assert "entity_drift:EPAM" in report.failures


def test_trace_eval_accepts_correct_source_collection_shape():
    contract = infer_source_contract("Search top 3 stocks, search each, fetch 3 URLs each.")
    trace = [
        {"kind": "entity_locked", "entity": "ROKU"},
        {"kind": "entity_locked", "entity": "TBN"},
        {"kind": "entity_locked", "entity": "SENEA"},
        {"tool": "web_search", "entity": "ROKU"},
        {"tool": "web_search", "entity": "TBN"},
        {"tool": "web_search", "entity": "SENEA"},
    ]
    for entity in ("ROKU", "TBN", "SENEA"):
        for idx in range(3):
            trace.append({"tool": "web_fetch", "entity": entity, "url": f"https://source.test/{entity}/{idx}"})
    report = evaluate_trace(contract, trace)
    assert report.ok
    assert report.score == 1.0
