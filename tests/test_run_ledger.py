import pytest

from engine.context_schema import ContextPhase
from run_engine.ledger import RunLedger
from run_engine.control_policies import resolve_control_policy
from run_engine.pass_framework import build_pass_graph
from run_engine.workflow_contracts import infer_workflow_contract


def test_ledger_rejects_orphaned_tool_results():
    ledger = RunLedger(
        run_id="run-1",
        session_id="session-1",
        turn_id="turn-1",
        objective="Test orphaned results",
        allowed_tools=["web_search"],
    )

    with pytest.raises(ValueError, match="Unknown tool_call_id"):
        ledger.link_tool_result(
            tool_call_id="missing",
            tool_name="web_search",
            success=True,
            status="success",
            summary="ok",
        )


def test_ledger_links_tool_result_and_evidence():
    ledger = RunLedger(
        run_id="run-2",
        session_id="session-2",
        turn_id="turn-1",
        objective="Research exact source",
        allowed_tools=["web_search"],
    )

    call = ledger.add_tool_call(tool_name="web_search", arguments={"query": "shovsOS"})
    result = ledger.link_tool_result(
        tool_call_id=call.id,
        tool_name="web_search",
        success=True,
        status="success",
        summary="Source https://example.com",
    )
    evidence = ledger.add_evidence_from_result(result)

    assert result.tool_call_id == call.id
    assert evidence.source == "tool:web_search"
    assert evidence.raw_ref == result.id
    assert ledger.selected_evidence()[0]["id"] == evidence.id


def test_phase_packet_is_deterministic_from_same_ledger():
    ledger = RunLedger(
        run_id="run-3",
        session_id="session-3",
        turn_id="turn-1",
        objective="Summarize current state",
        allowed_tools=["web_search"],
    )
    ledger.set_plan([
        {"description": "Search for sources", "tool": "web_search", "status": "pending"},
        {"description": "Answer from evidence", "status": "pending"},
    ])
    call = ledger.add_tool_call(tool_name="web_search", arguments={"query": "state"})
    result = ledger.link_tool_result(
        tool_call_id=call.id,
        tool_name="web_search",
        success=True,
        status="success",
        summary="state found",
    )
    ledger.add_evidence_from_result(result)

    first = ledger.to_phase_packet(ContextPhase.RESPONSE)
    second = ledger.to_phase_packet(ContextPhase.RESPONSE)

    assert first == second
    assert first["objective"] == "Summarize current state"
    assert first["tool_results"][0]["tool_call_id"] == call.id
    assert first["evidence_items"][0]["raw_ref"] == result.id
    assert first["runtime_attention"] == second["runtime_attention"]
    assert first["runtime_attention"]["version"] == "runtime-attention-v1"


def test_continuation_state_surfaces_pending_steps_and_missing_requirements():
    ledger = RunLedger(
        run_id="run-4",
        session_id="session-4",
        turn_id="turn-1",
        objective="Fetch exact pricing",
        allowed_tools=["web_search", "web_fetch"],
    )
    ledger.set_plan([
        {"id": "step_1", "description": "Search pricing", "tool": "web_search", "status": "done"},
        {"id": "step_2", "description": "Fetch pricing page", "tool": "web_fetch", "status": "pending"},
    ])
    ledger.set_continuation({
        "objective": "Fetch exact pricing",
        "pending_steps": ["Fetch pricing page"],
        "missing_evidence": ["first-party pricing page"],
    })

    packet = ledger.to_phase_packet(ContextPhase.RESPONSE)

    assert ledger.pending_steps()[0]["id"] == "step_2"
    assert "missing_evidence" in ledger.missing_requirements()
    assert packet["continuation_state"]["data"]["objective"] == "Fetch exact pricing"


def test_runtime_attention_is_phase_weighted():
    ledger = RunLedger(
        run_id="run-attn",
        session_id="session-attn",
        turn_id="turn-1",
        objective="Search the selected ticker and fetch exact sources",
        allowed_tools=["web_search", "web_fetch"],
    )
    ledger.set_plan([
        {"id": "search", "description": "Search ROKU stock news", "tool": "web_search", "status": "pending"},
        {"id": "answer", "description": "Answer only from fetched sources", "status": "pending"},
    ])
    call = ledger.add_tool_call("web_search", {"query": "ROKU stock news June 13 2026"})
    result = ledger.link_tool_result(
        tool_call_id=call.id,
        tool_name="web_search",
        success=True,
        status="success",
        summary="Found ROKU source candidates",
    )
    evidence = ledger.add_evidence_from_result(result)

    acting = ledger.attention_for_phase(ContextPhase.ACTING).to_dict()
    response = ledger.attention_for_phase(ContextPhase.RESPONSE).to_dict()

    acting_top_kinds = [item["kind"] for item in acting["items"][:3]]
    response_top_ids = [item["item_id"] for item in response["items"][:3]]

    assert "plan_step" in acting_top_kinds
    assert evidence.id in response_top_ids
    assert acting["policy"]["name"] == "ledger_phase_weighted"


def test_plan_execute_rejects_unlocked_entity_search_and_suggests_next_action():
    ledger = RunLedger(
        run_id="run-policy-gate",
        session_id="session-policy-gate",
        turn_id="turn-1",
        objective="Search ROKU, TBN, SENEA separately and fetch sources",
        allowed_tools=["web_search", "web_fetch"],
        ledger_mode="ledger_enforced",
    )
    ledger.set_control_policy(resolve_control_policy(ledger.objective, requested="plan_execute"))
    ledger.lock_entities(["ROKU", "TBN", "SENEA"])
    ledger.set_source_contract({
        "next_tool": "web_search",
        "next_arguments": {"query": "ROKU stock news June 13 2026"},
        "forbid_unlocked_entity_drift": True,
    })

    validation = ledger.validate_tool_call_against_policy("web_search", {"query": "EPAM stock news June 13 2026"})

    assert validation.valid is False
    assert validation.issue == "unlocked_entity_search"
    assert validation.recovery_class == "entity_drift"
    assert validation.expected_arguments["query"] == "ROKU stock news June 13 2026"


def test_plan_execute_rejects_off_contract_fetch():
    ledger = RunLedger(
        run_id="run-fetch-gate",
        session_id="session-fetch-gate",
        turn_id="turn-1",
        objective="Fetch locked source URLs",
        allowed_tools=["web_fetch"],
    )
    ledger.set_control_policy(resolve_control_policy(ledger.objective, requested="plan_execute"))
    ledger.lock_entities(["ROKU"])
    ledger.set_source_contract({
        "next_tool": "web_fetch",
        "next_arguments": {"url": "https://news.example/roku-1"},
        "allowed_fetch_urls": ["https://news.example/roku-1"],
        "total_urls": 1,
    })

    validation = ledger.validate_tool_call_against_policy("web_fetch", {"url": "https://news.example/epam-1"})

    assert validation.valid is False
    assert validation.issue == "off_contract_fetch"
    assert validation.expected_tool == "web_fetch"


def test_completion_gate_blocks_until_required_fetches_are_recorded():
    ledger = RunLedger(
        run_id="run-completion-gate",
        session_id="session-completion-gate",
        turn_id="turn-1",
        objective="Fetch two URLs",
        allowed_tools=["web_fetch"],
    )
    ledger.set_source_contract({
        "allowed_fetch_urls": ["https://news.example/a", "https://news.example/b"],
        "total_urls": 2,
    })
    first = ledger.add_tool_call("web_fetch", {"url": "https://news.example/a"})
    ledger.link_tool_result(
        tool_call_id=first.id,
        tool_name="web_fetch",
        success=True,
        status="ok",
        summary="A fetched",
    )

    blocked = ledger.completion_gate()
    assert blocked["final_answer_allowed"] is False
    assert blocked["missing_slots"] == ["fetched_urls:1/2"]

    second = ledger.add_tool_call("web_fetch", {"url": "https://news.example/b"})
    ledger.link_tool_result(
        tool_call_id=second.id,
        tool_name="web_fetch",
        success=True,
        status="ok",
        summary="B fetched",
    )

    assert ledger.completion_gate()["final_answer_allowed"] is True


def test_memory_write_requires_provenance_without_evidence():
    ledger = RunLedger(
        run_id="run-memory-provenance",
        session_id="session-memory-provenance",
        turn_id="turn-1",
        objective="Remember fact",
        allowed_tools=[],
    )

    with pytest.raises(ValueError, match="provenance"):
        ledger.add_memory_write(status="committed", summary="unsupported")

    record = ledger.add_memory_write(
        status="committed",
        summary="deterministic governor",
        data={"path": "deterministic_governor", "fact_count": 1},
    )
    assert record.data["path"] == "deterministic_governor"


def test_graph_harness_phase_recording_advances_pass_graph_execution():
    contract = infer_workflow_contract(
        "Implement this backend change and run tests.",
        allowed_tools=["file_view", "file_str_replace", "bash"],
    )
    graph = build_pass_graph(contract)
    ledger = RunLedger(
        run_id="run-graph-execution",
        session_id="session-graph-execution",
        turn_id="turn-1",
        objective=contract.objective,
        allowed_tools=["file_view", "file_str_replace", "bash"],
    )
    ledger.set_workflow_contract(contract)
    ledger.set_pass_graph(graph)
    ledger.set_control_policy(resolve_control_policy(ledger.objective, requested="graph_harness"))

    payload = ledger.record_graph_phase(ContextPhase.PLANNING)
    packet = ledger.to_phase_packet(ContextPhase.ACTING)

    assert payload is not None
    assert payload["node"]["status"] == "completed"
    assert packet["pass_graph_execution"]["summary"]["completed"] == 1
    assert any(event.event_type == "pass_node_completed" for event in ledger.events)


def test_graph_harness_phase_recording_is_idempotent_per_marker():
    contract = infer_workflow_contract(
        "Implement this backend change and run tests.",
        allowed_tools=["file_view", "file_str_replace", "bash"],
    )
    ledger = RunLedger(
        run_id="run-graph-idempotent",
        session_id="session-graph-idempotent",
        turn_id="turn-1",
        objective=contract.objective,
        allowed_tools=["file_view", "file_str_replace", "bash"],
    )
    ledger.set_pass_graph(build_pass_graph(contract))

    first = ledger.record_graph_phase(ContextPhase.PLANNING, marker="planning:0")
    second = ledger.record_graph_phase(ContextPhase.PLANNING, marker="planning:0")

    assert first is not None
    assert second is None
    assert ledger.pass_graph_execution is not None
    assert ledger.pass_graph_execution.to_dict()["summary"]["completed"] == 1
    assert sum(1 for event in ledger.events if event.event_type == "pass_node_completed") == 1
