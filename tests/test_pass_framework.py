from engine.context_schema import ContextPhase
from run_engine.ledger import RunLedger
from run_engine.pass_framework import (
    build_pass_graph,
    complete_pass_node,
    initialize_pass_graph_execution,
    start_next_pass_node,
)
from run_engine.workflow_contracts import infer_workflow_contract


def test_source_collection_pass_graph_has_specialist_roles_and_stop_gate():
    contract = infer_workflow_contract(
        "top 3 sushi places in toronto, then search each, fetch 3 URLs each",
        allowed_tools=["web_search", "web_fetch", "source_select"],
    )

    graph = build_pass_graph(contract)

    roles = [item.role for item in graph.passes]
    assert graph.workflow_shape == "source_collection"
    assert graph.context_strategy == "local_retrieval_plus_contract_eval"
    assert "retrieval_agent" in roles
    assert "scoring_agent" in roles
    assert "evaluation_agent" in roles
    assert "orchestration_agent" in roles
    assert "workflow_contract.completion_gate" in graph.stop_condition
    assert graph.max_rounds >= 9


def test_research_report_pass_graph_uses_local_and_global_reasoning():
    contract = infer_workflow_contract(
        "Research whether this proposal is coherent and summarize the risks.",
        allowed_tools=["web_search"],
    )

    graph = build_pass_graph(contract)
    strategies = [item.context_strategy for item in graph.passes]

    assert graph.workflow_shape == "research_report"
    assert graph.context_strategy == "local_plus_global_reasoning"
    assert "chunk_wise" in strategies
    assert "global_reasoning" in strategies


def test_pass_graph_is_visible_in_ledger_and_attention():
    contract = infer_workflow_contract(
        "Find top 3 tools, search each separately, fetch 2 URLs each.",
        allowed_tools=["web_search", "web_fetch"],
    )
    graph = build_pass_graph(contract)
    ledger = RunLedger(
        run_id="run-pass",
        session_id="session-pass",
        turn_id="turn-1",
        objective=contract.objective,
        allowed_tools=["web_search", "web_fetch"],
    )
    ledger.set_workflow_contract(contract)
    ledger.set_pass_graph(graph)

    packet = ledger.to_phase_packet(ContextPhase.PLANNING)
    attention_kinds = [item["kind"] for item in packet["runtime_attention"]["items"]]

    assert packet["pass_graph"]["id"] == "pass_graph_source_collection_v1"
    assert packet["pass_graph_execution"]["status"] == "pending"
    assert "pass_graph" in attention_kinds
    assert "Agent Pass Framework:" in ledger.render_for_phase(ContextPhase.PLANNING)


def test_pass_graph_execution_runs_nodes_in_dependency_order():
    contract = infer_workflow_contract(
        "Search top 3 stocks, search each, fetch 3 URLs each.",
        allowed_tools=["web_search", "web_fetch"],
    )
    graph = build_pass_graph(contract)
    execution = initialize_pass_graph_execution(graph)

    first = start_next_pass_node(execution)
    assert first is not None
    assert first.id == "contract"
    assert first.status == "running"

    complete_pass_node(execution, first.id, outputs={"workflow_contract": True})
    second = start_next_pass_node(execution)

    assert second is not None
    assert second.id == "retrieve"
    assert execution.status == "running"
