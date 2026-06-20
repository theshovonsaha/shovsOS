import pytest

from engine.context_schema import ContextPhase
from run_engine.ledger import RunLedger


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
