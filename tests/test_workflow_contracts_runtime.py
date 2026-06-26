from dataclasses import replace
from types import SimpleNamespace

from engine.context_schema import ContextPhase
from run_engine.context_packets import PacketBuildInputs, build_phase_packet
from run_engine.ledger import RunLedger
from run_engine.types import RunEngineRequest
from run_engine.workflow_contracts import (
    EntityLock,
    classify_workflow_shape,
    infer_workflow_contract,
    update_contract_from_tool_results,
)


def test_workflow_contract_infers_topic_agnostic_source_collection():
    objective = (
        "Find top 4 budget laptops, search each separately, collect 2 review URLs "
        "for each, web fetch all 8 URLs, then summarize."
    )

    contract = infer_workflow_contract(objective, allowed_tools=["web_search", "web_fetch", "bash"])

    assert contract.workflow_shape == "source_collection"
    assert contract.metadata["entity_count"] == 4
    assert contract.metadata["results_per_entity"] == 2
    assert contract.metadata["total_fetches"] == 8
    assert contract.allowed_tools == ["web_search", "web_fetch"]
    assert contract.completion_gate.final_answer_allowed is False
    assert "locked_entities" in contract.completion_gate.missing_slots


def test_workflow_contract_treats_fetch_urls_each_as_per_entity_quota():
    contract = infer_workflow_contract(
        "top 3 sushi places in toronto, then search each, fetch 3 URLs each to get intel",
        allowed_tools=["web_search", "web_fetch"],
    )

    assert contract.workflow_shape == "source_collection"
    assert contract.metadata["entity_count"] == 3
    assert contract.metadata["results_per_entity"] == 3
    assert contract.metadata["total_fetches"] == 9


def test_workflow_shape_keeps_simple_chat_out_of_heavy_contracts():
    assert classify_workflow_shape("hi") == "simple_chat"
    assert classify_workflow_shape("hi again") == "simple_chat"
    assert classify_workflow_shape("thanks that helps") == "simple_chat"
    assert classify_workflow_shape("ok got it") == "simple_chat"
    assert classify_workflow_shape("hello search Toronto sushi") != "simple_chat"
    contract = infer_workflow_contract("hi", allowed_tools=["web_search"])
    assert contract.workflow_shape == "simple_chat"
    assert contract.allowed_tools == []
    assert contract.completion_gate.final_answer_allowed is True


def test_simple_chat_phase_packet_suppresses_context_poisoning_lanes():
    request = RunEngineRequest(
        session_id="simple-packet",
        owner_id="owner-1",
        agent_id="default",
        user_message="hi again",
        model="llama3.2",
        system_prompt="You are Shovs.",
        allowed_tools=("web_search", "web_fetch"),
    )
    ledger = RunLedger(
        run_id="run-simple",
        session_id=request.session_id,
        turn_id="turn-1",
        objective=request.user_message,
        allowed_tools=["web_search", "web_fetch"],
    )
    ledger.set_workflow_contract(infer_workflow_contract(request.user_message, allowed_tools=ledger.allowed_tools))
    ledger.set_plan([
        {"id": "stale_step", "description": "Search old stock movers", "tool": "web_search", "status": "pending"},
    ])

    packet = build_phase_packet(
        context_engine=None,
        inputs=PacketBuildInputs(
            request=request,
            session=SimpleNamespace(
                first_message="Search stocks from yesterday.",
                sliding_window=[
                    {"role": "user", "content": "Search top stocks."},
                    {"role": "assistant", "content": "I will search ROKU and TBN."},
                ],
                message_count=20,
            ),
            phase=ContextPhase.ACTING,
            system_prompt=request.system_prompt,
            current_context="Old compressed memory: user keeps saying hi multiple times.",
            allowed_tools=[{"name": "web_search"}, {"name": "web_fetch"}],
            tool_results=[{"tool_name": "web_search", "success": True, "content": "stale result"}],
            run_ledger=ledger,
        ),
    )

    included_ids = {item["item_id"] for item in packet.trace["included"]}
    assert "current_objective" in included_ids
    assert "canonical_run_ledger" not in included_ids
    assert "runtime_attention" not in included_ids
    assert "context_ladder" not in included_ids
    assert "historical_context" not in included_ids
    assert "working_state" not in included_ids
    assert "Search old stock movers" not in packet.content
    assert "user keeps saying hi" not in packet.content


def test_workflow_contract_updates_from_tool_results_without_new_entity_drift():
    contract = infer_workflow_contract(
        "Search top 3 stocks separately and collect 3 URLs for each, fetch all 9 URLs.",
        allowed_tools=["web_search", "web_fetch"],
    )
    contract = replace(
        contract,
        entity_locks=[
            EntityLock(value="ROKU", entity_type="ticker", source="morningstar", status="locked"),
            EntityLock(value="TBN", entity_type="ticker", source="morningstar", status="locked"),
            EntityLock(value="SENEA", entity_type="ticker", source="morningstar", status="locked"),
        ],
    )

    updated = update_contract_from_tool_results(
        contract,
        [
            {
                "tool_name": "web_search",
                "success": True,
                "arguments": {"query": "ROKU stock news June 13 2026"},
                "content": "https://news.example/roku-a https://news.example/roku-b https://news.example/roku-c",
            },
            {
                "tool_name": "web_search",
                "success": True,
                "arguments": {"query": "EPAM stock major jump today"},
                "content": "https://news.example/epam-a",
            },
        ],
    )

    assert "ROKU_search_results" not in updated.completion_gate.missing_slots
    assert "TBN_search_results" in updated.completion_gate.missing_slots
    assert "SENEA_search_results" in updated.completion_gate.missing_slots
    assert "EPAM" not in updated.metadata["searched_entities"]
    assert updated.completion_gate.final_answer_allowed is False


def test_ledger_phase_packet_exposes_workflow_contract_and_attention():
    request = RunEngineRequest(
        session_id="contract-packet",
        owner_id="owner-1",
        agent_id="default",
        user_message="Find top 3 tools, search each separately, collect 2 URLs for each, fetch all 6 URLs.",
        model="llama3.2",
        system_prompt="You are Shovs.",
        allowed_tools=("web_search", "web_fetch"),
    )
    ledger = RunLedger(
        run_id="run-contract",
        session_id=request.session_id,
        turn_id="turn-1",
        objective=request.user_message,
        allowed_tools=["web_search", "web_fetch"],
    )
    ledger.set_workflow_contract(infer_workflow_contract(request.user_message, allowed_tools=ledger.allowed_tools))

    packet = build_phase_packet(
        context_engine=None,
        inputs=PacketBuildInputs(
            request=request,
            session=SimpleNamespace(first_message=request.user_message, sliding_window=[]),
            phase=ContextPhase.ACTING,
            system_prompt=request.system_prompt,
            current_context="",
            allowed_tools=[{"name": "web_search"}, {"name": "web_fetch"}],
            tool_results=[],
            run_ledger=ledger,
        ),
    )

    phase_packet = ledger.to_phase_packet(ContextPhase.ACTING)
    attention_kinds = [item["kind"] for item in phase_packet["runtime_attention"]["items"]]

    assert phase_packet["workflow_contract"]["workflow_shape"] == "source_collection"
    assert "workflow_contract" in attention_kinds
    assert "Workflow Contract:" in packet.content
    assert "Runtime Attention:" in packet.content
