import json

from engine.response_guard import guard_final_response
from memory.semantic_graph import SemanticGraph
from run_engine.ledger import RunLedger
from run_engine.scenario_eval import SourceCollectionScenario, evaluate_source_collection_trace
from shovs_memory import ShovsMemory


def _record(scenario, passed, score=1.0, issues=None, summary="", evidence=None):
    return {
        "suite": "agent_harness_core",
        "scenario": scenario,
        "passed": bool(passed),
        "score": float(score),
        "issues": list(issues or []),
        "summary": summary,
        "evidence": dict(evidence or {}),
    }


def test_agent_harness_core_benchmark_records_are_machine_readable(tmp_path):
    records = []

    guarded = guard_final_response(
        '{"tool":"name_lookup","arguments":{"preferred_name":"Shovon"}}',
        user_message="What should you call me?",
        model="ollama:llama3.2:3b",
    )
    records.append(
        _record(
            "response_guard_tool_json",
            guarded.changed and guarded.text == "You should be called Shovon.",
            issues=guarded.issues,
            summary="Tool-call-shaped JSON was converted into clean user-facing text.",
            evidence={"text": guarded.text},
        )
    )

    allowed = {
        "ROKU": [
            "https://news.example.com/roku-1",
            "https://news.example.com/roku-2",
            "https://news.example.com/roku-3",
        ],
        "TBN": [
            "https://news.example.com/tbn-1",
            "https://news.example.com/tbn-2",
            "https://news.example.com/tbn-3",
        ],
        "SENEA": [
            "https://news.example.com/senea-1",
            "https://news.example.com/senea-2",
            "https://news.example.com/senea-3",
        ],
    }
    scenario = SourceCollectionScenario(
        objective="Find Morningstar top gainers, search each ticker, and fetch 3 URLs per ticker.",
        entities=["ROKU", "TBN", "SENEA"],
        urls_per_entity=3,
        total_urls=9,
        discovery_url="https://www.morningstar.com/markets",
        query_template="{entity} stock news June 13 2026",
        forbidden_query_terms=["EPAM", "ARKO"],
        allowed_fetch_urls_by_entity=allowed,
    )
    good_calls = [
        {"tool": "web_fetch", "url": "https://www.morningstar.com/markets"},
        {"tool": "web_search", "query": "ROKU stock news June 13 2026"},
        {"tool": "web_search", "query": "TBN stock news June 13 2026"},
        {"tool": "web_search", "query": "SENEA stock news June 13 2026"},
    ] + [{"tool": "web_fetch", "url": url} for urls in allowed.values() for url in urls]
    good_eval = evaluate_source_collection_trace(scenario=scenario, tool_calls=good_calls)
    records.append(
        _record(
            "source_collection_contract",
            good_eval.passed,
            score=good_eval.score,
            issues=good_eval.issues,
            summary=good_eval.detail,
            evidence={"entities": good_eval.state["entities"], "entity_fetch_count": good_eval.state["entity_fetch_count"]},
        )
    )

    bad_calls = [
        {"tool": "web_fetch", "url": "https://www.morningstar.com/markets"},
        {"tool": "web_search", "query": "EPAM stock major jump today"},
    ]
    bad_eval = evaluate_source_collection_trace(scenario=scenario, tool_calls=bad_calls)
    records.append(
        _record(
            "source_collection_drift_negative_control",
            (not bad_eval.passed) and "forbidden_query_drift" in bad_eval.issues,
            score=1.0 if "forbidden_query_drift" in bad_eval.issues else 0.0,
            issues=bad_eval.issues,
            summary="The evaluator caught unrelated entity drift.",
            evidence={"searches": bad_eval.state["searches"]},
        )
    )

    ledger = RunLedger(
        run_id="bench-run",
        session_id="bench-session",
        turn_id="bench-turn",
        objective="Reject orphan tool results.",
        allowed_tools=["web_search"],
    )
    try:
        ledger.link_tool_result(
            tool_call_id="missing-call",
            tool_name="web_search",
            success=True,
            status="success",
            summary="This should be rejected.",
        )
    except ValueError as exc:
        orphan_rejected = "Unknown tool_call_id" in str(exc)
    else:
        orphan_rejected = False
    records.append(
        _record(
            "ledger_orphan_tool_result_rejected",
            orphan_rejected,
            summary="The ledger rejected a tool result without a matching tool call.",
        )
    )

    graph = SemanticGraph(db_path=str(tmp_path / "memory_graph.db"))
    memory = ShovsMemory(session_id="bench-memory", owner_id="bench-owner", graph=graph)
    memory.store_fact(subject="User", predicate="location", object_="Vancouver", turn=1)
    try:
        memory.store_fact(subject="User", predicate="location", object_=object(), turn=2)
    except AttributeError:
        pass
    records.append(
        _record(
            "memory_replacement_rollback",
            memory.current_facts() == [("User", "location", "Vancouver")],
            summary="A failed fact replacement preserved the previous current fact.",
            evidence={"current_facts": memory.current_facts()},
        )
    )

    report = {"suite": "agent_harness_core", "version": 1, "results": records}
    out = tmp_path / "agent_harness_core_results.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))

    loaded = json.loads(out.read_text())
    assert loaded["suite"] == "agent_harness_core"
    assert all(item["passed"] for item in loaded["results"])
    assert {item["scenario"] for item in loaded["results"]} == {
        "response_guard_tool_json",
        "source_collection_contract",
        "source_collection_drift_negative_control",
        "ledger_orphan_tool_result_rejected",
        "memory_replacement_rollback",
    }
