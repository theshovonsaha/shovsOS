from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from fastapi import APIRouter, Body

from engine.response_guard import guard_final_response
from memory.semantic_graph import SemanticGraph
from run_engine.ledger import RunLedger
from run_engine.control_policies import resolve_control_policy
from run_engine.workflow_contracts import infer_workflow_contract
from run_engine.workflow_plugins import source_collection_contract_from_objective
from run_engine.pass_framework import build_pass_graph
from run_engine.scenario_eval import SourceCollectionScenario, evaluate_policy_trace, evaluate_source_collection_trace
from shovs_memory import ShovsMemory


PAPER_REFERENCES = [
    {
        "id": "agenttrace",
        "title": "AgentTrace",
        "url": "https://arxiv.org/abs/2602.10133",
        "alignment": "Operational, cognitive, and contextual traces should be first-class runtime objects.",
    },
    {
        "id": "proxy-state-eval",
        "title": "Proxy State-Based Evaluation",
        "url": "https://arxiv.org/abs/2602.16246",
        "alignment": "Agent claims should be checked against scenario state, not trusted from final prose alone.",
    },
    {
        "id": "toolspec",
        "title": "ToolSpec",
        "url": "https://arxiv.org/abs/2604.13519",
        "alignment": "Tool traces can be structured enough for schema-aware execution and validation.",
    },
    {
        "id": "tool-chain-risk",
        "title": "Tool-chain resource amplification risk",
        "url": "https://arxiv.org/abs/2601.10955",
        "alignment": "Tool chains need bounded execution, traceability, and failure controls.",
    },
]


WEDGES = [
    {
        "id": "run_ledger",
        "label": "Run Ledger",
        "plain": "Keeps the task state outside the model.",
        "impact": "Stops later phases from inventing what happened earlier.",
        "limitation": "Only useful if phases read and write through it.",
        "test": "ledger_orphan_tool_result_rejected",
    },
    {
        "id": "phase_packets",
        "label": "Phase Packets",
        "plain": "Gives each model step the right slice of context.",
        "impact": "Reduces prompt drift and noisy carryover.",
        "limitation": "Packet quality depends on the ledger and context compiler.",
        "test": "runtime_e2e_diagnostics",
    },
    {
        "id": "source_contract",
        "label": "Source Contract",
        "plain": "Locks selected entities and required URLs during research workflows.",
        "impact": "Catches wrong searches even when the final answer sounds plausible.",
        "limitation": "Currently strongest for source-collection workflows.",
        "test": "source_collection_contract",
    },
    {
        "id": "memory_lanes",
        "label": "Memory Lanes",
        "plain": "Separates current facts, superseded facts, candidates, and disputes.",
        "impact": "Prevents stale or guessed memory from becoming hidden truth.",
        "limitation": "Needs good extraction and explicit commit policy.",
        "test": "memory_replacement_rollback",
    },
    {
        "id": "response_guard",
        "label": "Response Guard",
        "plain": "Checks final user-visible text before it reaches chat.",
        "impact": "Blocks tool JSON leaks and unsupported runtime language.",
        "limitation": "It is a final guard, not a replacement for correct planning.",
        "test": "response_guard_tool_json",
    },
    {
        "id": "trace_replay",
        "label": "Trace Replay",
        "plain": "Turns logs into a replayable workflow story.",
        "impact": "Makes failures inspectable instead of mysterious.",
        "limitation": "Missing backend fields still appear as not recorded.",
        "test": "trace_replay_api",
    },
]


MODES = [
    {
        "id": "plain_model",
        "label": "Plain Model",
        "included_wedges": [],
        "best_for": "Simple chat and low-risk brainstorming.",
        "expected_failure": "Can sound confident without proof of tool work or memory state.",
    },
    {
        "id": "model_plus_tools",
        "label": "Model + Tools",
        "included_wedges": ["response_guard"],
        "best_for": "Short tasks where tool access helps but deep traceability is not needed.",
        "expected_failure": "Can still drift across multi-step workflows because state lives mostly in text.",
    },
    {
        "id": "shovs_observed",
        "label": "Shovs Observed",
        "included_wedges": ["run_ledger", "phase_packets", "trace_replay", "response_guard"],
        "best_for": "Operator-supervised agents and debugging.",
        "expected_failure": "Some ledger checks may be shadow-mode until promoted to enforcement.",
    },
    {
        "id": "shovs_react",
        "label": "Shovs ReAct",
        "included_wedges": ["run_ledger", "phase_packets", "response_guard", "trace_replay"],
        "best_for": "Low-risk iterative tool use where observations can steer the next action.",
        "expected_failure": "Can drift on source workflows because the plan remains mutable.",
    },
    {
        "id": "shovs_plan_execute",
        "label": "Shovs Plan-Execute",
        "included_wedges": ["run_ledger", "phase_packets", "source_contract", "response_guard", "trace_replay"],
        "best_for": "Web/source workflows where entities, URLs, and quotas must stay fixed.",
        "expected_failure": "May block completion when evidence is missing instead of guessing.",
    },
    {
        "id": "shovs_graph_harness",
        "label": "Shovs Graph Harness",
        "included_wedges": [
            "run_ledger",
            "phase_packets",
            "source_contract",
            "memory_lanes",
            "response_guard",
            "trace_replay",
        ],
        "best_for": "Long-horizon, coding, memory-sensitive, or enterprise workflows.",
        "expected_failure": "Less flexible than a free-form loop but easier to inspect and recover.",
    },
    {
        "id": "shovs_enforced",
        "label": "Shovs Enforced",
        "included_wedges": [
            "run_ledger",
            "phase_packets",
            "source_contract",
            "memory_lanes",
            "response_guard",
            "trace_replay",
        ],
        "best_for": "Research, shopping advice, coding, and workflows where the path matters.",
        "expected_failure": "May stop or ask for missing inputs instead of guessing.",
    },
]


def _record(
    scenario: str,
    passed: bool,
    *,
    score: float = 1.0,
    issues: Optional[list[str]] = None,
    summary: str = "",
    evidence: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "passed": bool(passed),
        "score": float(score),
        "issues": list(issues or []),
        "summary": summary,
        "evidence": dict(evidence or {}),
    }


def _slug(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or ""))
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-") or "entity"


def _allowed_urls_for_entities(entities: list[str], per_entity: int, *, topic: str) -> dict[str, list[str]]:
    return {
        entity: [
            f"https://sources.example.com/{topic}/{_slug(entity)}-{index}"
            for index in range(1, per_entity + 1)
        ]
        for entity in entities
    }


def _demo_source_scenario(objective: str) -> SourceCollectionScenario:
    text = str(objective or "").lower()
    contract = source_collection_contract_from_objective(objective) or {
        "entity_count": 3,
        "urls_per_entity": 3,
        "total_urls": 9,
    }
    entity_count = max(1, min(int(contract.get("entity_count") or 3), 5))
    urls_per_entity = max(1, min(int(contract.get("urls_per_entity") or 3), 5))
    total_urls = max(entity_count * urls_per_entity, int(contract.get("total_urls") or entity_count * urls_per_entity))

    if any(term in text for term in ("sushi", "restaurant", "restaurants", "places", "place", "coffee", "toronto")):
        entities = ["Sushi Masaki Saito", "Yasu", "Sushi Kaji", "Miku", "Aburi Hana"][:entity_count]
        return SourceCollectionScenario(
            objective=objective,
            entities=entities,
            urls_per_entity=urls_per_entity,
            total_urls=total_urls,
            discovery_url="https://local.example.com/toronto/best-sushi",
            query_template="{entity} reviews Toronto sources",
            forbidden_query_terms=["Vancouver", "Montreal", "EPAM"],
            allowed_fetch_urls_by_entity=_allowed_urls_for_entities(entities, urls_per_entity, topic="local"),
        )

    if any(term in text for term in ("shopping", "stores", "store", "price", "prices", "compare", "comparison", "product", "products")):
        entities = ["Costco", "Walmart", "Best Buy", "Canadian Tire", "Dollarama"][:entity_count]
        return SourceCollectionScenario(
            objective=objective,
            entities=entities,
            urls_per_entity=urls_per_entity,
            total_urls=total_urls,
            discovery_url="https://shopping.example.com/store-comparison",
            query_template="{entity} product price review sources",
            forbidden_query_terms=["ROKU", "ARKO", "unrelated store"],
            allowed_fetch_urls_by_entity=_allowed_urls_for_entities(entities, urls_per_entity, topic="shopping"),
        )

    entities = ["ROKU", "TBN", "SENEA", "LEU", "SNDK"][:entity_count]
    return SourceCollectionScenario(
        objective=objective,
        entities=entities,
        urls_per_entity=urls_per_entity,
        total_urls=total_urls,
        discovery_url="https://www.morningstar.com/markets",
        query_template="{entity} stock news June 13 2026",
        forbidden_query_terms=["EPAM", "ARKO"],
        allowed_fetch_urls_by_entity=_allowed_urls_for_entities(entities, urls_per_entity, topic="stocks"),
    )


def _mode_trace(mode: str, scenario: SourceCollectionScenario) -> list[dict[str, Any]]:
    allowed = scenario.allowed_fetch_urls_by_entity
    locked_label = ", ".join(scenario.entities)
    drift_term = scenario.forbidden_query_terms[0] if scenario.forbidden_query_terms else "unrelated"
    if mode == "plain_model":
        return [
            {"phase": "intake", "actor": "model", "action": "read_task", "status": "ok", "summary": "Task accepted as plain text."},
            {"phase": "response", "actor": "model", "action": "answer_from_prior", "status": "unsupported", "summary": "Produces a plausible answer without executable tool evidence."},
        ]
    if mode == "model_plus_tools":
        return [
            {"phase": "tool", "actor": "tool", "tool": "web_fetch", "url": scenario.discovery_url, "status": "success", "summary": "Fetched mover discovery page."},
            {"phase": "tool", "actor": "tool", "tool": "web_search", "query": f"{drift_term} broad search", "status": "success", "summary": "Drifted to an unrelated entity after discovery."},
            {"phase": "tool", "actor": "tool", "tool": "web_fetch", "url": "https://generic.example.com/broad-list", "status": "success", "summary": "Fetched generic article outside the requested entity set."},
            {"phase": "response", "actor": "model", "action": "summarize", "status": "warning", "summary": "Answer can sound useful, but the path no longer matches the task."},
        ]
    if mode == "shovs_react":
        return [
            {"phase": "policy", "actor": "shovs", "action": "select_policy", "status": "active", "control_policy": "react", "summary": "ReAct lets observations steer the next action."},
            {"phase": "tool", "actor": "tool", "tool": "web_fetch", "url": scenario.discovery_url, "status": "success", "summary": "Fetched discovery page."},
            {"phase": "tool", "actor": "tool", "tool": "web_search", "query": f"{drift_term} broad search", "status": "success", "summary": "Observation-driven loop drifted to a noisy entity."},
            {"phase": "policy_violation", "actor": "shovs", "action": "detect_drift", "status": "warning", "summary": "Ledger can see drift, but ReAct policy is not source-contract enforced."},
        ]
    calls = [
        {"phase": "policy", "actor": "shovs", "action": "select_policy", "status": "active", "control_policy": "graph_harness" if mode == "shovs_graph_harness" else "plan_execute", "ledger_mode": "ledger_enforced", "summary": "Policy selected from workflow shape."},
        {"phase": "tool", "actor": "tool", "tool": "web_fetch", "url": scenario.discovery_url, "status": "success", "summary": "Fetched discovery page before locking entities."},
        {"phase": "contract", "actor": "shovs", "action": "lock_entities", "status": "success", "summary": f"Locked {locked_label} as the source contract entities.", "entities": scenario.entities},
        {"phase": "contract", "actor": "shovs", "action": "set_quota", "status": "success", "summary": f"Required {scenario.urls_per_entity} URLs per entity, {scenario.total_urls} total.", "source_contract": {"total_urls": scenario.total_urls}},
    ]
    for entity in scenario.entities:
        calls.append({
            "phase": "tool",
            "actor": "tool",
            "tool": "web_search",
            "query": scenario.query_template.format(entity=entity),
            "status": "success",
            "summary": f"Searched exact query for {entity}.",
        })
    for entity, urls in allowed.items():
        for url in urls:
            calls.append({
                "phase": "tool",
                "actor": "tool",
                "tool": "web_fetch",
                "url": url,
                "status": "success",
                "summary": f"Fetched selected source for {entity}.",
            })
    calls.append({"phase": "verify", "actor": "shovs", "action": "state_eval", "status": "success", "summary": "Verified entity locks and fetch quota before response."})
    if mode == "shovs_graph_harness":
        calls.insert(1, {"phase": "pass_graph_execution", "event_type": "pass_node_started", "actor": "shovs", "action": "node_start", "status": "success", "summary": "Pass graph executes discover -> lock -> collect -> fetch -> verify nodes."})
        calls.append({"phase": "pass_graph_execution", "event_type": "pass_node_completed", "actor": "shovs", "action": "node_complete", "status": "success", "summary": "Graph completion gate passed."})
    return calls


def _tool_calls_from_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for event in trace:
        if not isinstance(event, dict) or not event.get("tool"):
            continue
        call = {"tool": event.get("tool")}
        if event.get("query"):
            call["query"] = event.get("query")
        if event.get("url"):
            call["url"] = event.get("url")
        calls.append(call)
    return calls


def run_mode_comparison(objective: str, *, modes: Optional[list[str]] = None) -> dict[str, Any]:
    task = str(objective or "").strip() or "Search top 3 stocks today, then search each, fetch 3 URLs each."
    selected_modes = modes or ["plain_model", "model_plus_tools", "shovs_react", "shovs_plan_execute", "shovs_graph_harness"]
    scenario = _demo_source_scenario(task)
    contract = infer_workflow_contract(task, allowed_tools=["web_search", "web_fetch"])
    pass_graph = build_pass_graph(contract)
    results = []
    for mode in selected_modes:
        trace = _mode_trace(mode, scenario)
        eval_result = evaluate_source_collection_trace(
            scenario=scenario,
            tool_calls=_tool_calls_from_trace(trace),
            source_contract_events=trace if mode in {"shovs_stateful", "shovs_plan_execute", "shovs_graph_harness"} else [],
        )
        policy_eval = evaluate_policy_trace(
            trace_events=trace,
            expected_policy={
                "plain_model": "none",
                "model_plus_tools": "none",
                "shovs_react": "react",
                "shovs_plan_execute": "plan_execute",
                "shovs_graph_harness": "graph_harness",
                "shovs_stateful": "plan_execute",
            }.get(mode, ""),
            expected_ledger_mode="ledger_enforced" if mode in {"shovs_plan_execute", "shovs_graph_harness", "shovs_stateful"} else "",
            require_completion_gate=mode in {"shovs_plan_execute", "shovs_graph_harness", "shovs_stateful"},
            require_recovery_for_violations=mode in {"shovs_plan_execute", "shovs_graph_harness", "shovs_stateful"},
            require_graph_nodes=mode == "shovs_graph_harness",
        )
        if mode == "plain_model":
            score = 0.0
            passed = False
            issues = ["no_tool_evidence", "no_ledger_authority", "no_source_contract"]
            summary = "Plain model mode cannot prove search/fetch work. It may answer, but it has no executable trace."
        elif mode == "model_plus_tools":
            score = eval_result.score
            passed = eval_result.passed
            issues = eval_result.issues
            summary = "Tool mode executed tools, but state-based eval exposes drift and missing per-entity coverage."
        elif mode == "shovs_react":
            score = eval_result.score
            passed = False
            issues = list(dict.fromkeys(eval_result.issues + ["mutable_plan_drift_risk"]))
            summary = "Shovs ReAct keeps trace state, but source workflows still need plan-execute enforcement."
        else:
            score = eval_result.score
            passed = eval_result.passed
            issues = eval_result.issues
            if mode == "shovs_graph_harness":
                summary = "Shovs graph harness adds explicit node structure over the same source contract and verification gates."
            else:
                summary = "Shovs plan-execute mode locks entities, uses exact queries, fetches required URLs, and verifies before answer."
        results.append({
            "mode": mode,
            "label": {
                "plain_model": "Plain Model",
                "model_plus_tools": "Model + Tools",
                "shovs_stateful": "ShovsOS Stateful",
                "shovs_react": "ShovsOS ReAct",
                "shovs_plan_execute": "ShovsOS Plan-Execute",
                "shovs_graph_harness": "ShovsOS Graph Harness",
            }.get(mode, mode),
            "control_policy": {
                "plain_model": "none",
                "model_plus_tools": "none",
                "shovs_react": "react",
                "shovs_plan_execute": "plan_execute",
                "shovs_graph_harness": "graph_harness",
                "shovs_stateful": "plan_execute",
            }.get(mode, "unknown"),
            "ledger_mode": "ledger_enforced" if mode in {"shovs_plan_execute", "shovs_graph_harness", "shovs_stateful"} else "shadow",
            "passed": passed,
            "score": round(float(score), 4),
            "issues": issues,
            "summary": summary,
            "trace": trace,
            "trace_summary": {
                "event_count": len(trace),
                "policy_violations": len([event for event in trace if event.get("phase") == "policy_violation"]),
                "tool_events": len([event for event in trace if event.get("tool")]),
                "completion_gate": "passed" if passed else "blocked_or_unproven",
            },
            "state_eval": eval_result.to_dict(),
            "policy_eval": policy_eval.to_dict(),
        })
    return {
        "suite": "shovs_mode_comparison",
        "objective": task,
        "live_model": False,
        "deterministic": True,
        "contract": contract.to_dict(),
        "pass_graph": pass_graph.to_dict(),
        "results": results,
        "takeaway": "The difference is not better prose. The difference is whether the run can prove what happened and refuse completion when required state is missing.",
    }


def get_harness_lab_catalog() -> dict[str, Any]:
    return {
        "title": "Harness Lab",
        "subtitle": "Compare plain model behavior with ShovsOS runtime wedges.",
        "workflow": [
            "intake",
            "plan",
            "context build",
            "tool call",
            "evidence",
            "verify",
            "memory commit",
            "response",
        ],
        "modes": MODES,
        "wedges": WEDGES,
        "papers": PAPER_REFERENCES,
        "benchmark": {
            "suite": "agent_harness_core",
            "endpoint": "/harness-lab/benchmark/run",
            "frontend_endpoint": "/api/harness-lab/benchmark/run",
            "deterministic": True,
            "live_llm_required": False,
            "network_required": False,
        },
        "comparison": {
            "suite": "shovs_mode_comparison",
            "endpoint": "/harness-lab/compare/run",
            "frontend_endpoint": "/api/harness-lab/compare/run",
            "modes": ["plain_model", "model_plus_tools", "shovs_react", "shovs_plan_execute", "shovs_graph_harness"],
            "live_model_required": False,
            "network_required": False,
        },
        "runtime": {
            "suite": "shovs_runtime_harness",
            "endpoint": "/harness-lab/runtime/run",
            "frontend_endpoint": "/api/harness-lab/runtime/run",
            "deterministic": True,
            "live_model_required": False,
            "network_required": False,
        },
    }


def run_harness_core_benchmark() -> dict[str, Any]:
    records: list[dict[str, Any]] = []

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
            evidence={
                "entities": good_eval.state["entities"],
                "entity_fetch_count": good_eval.state["entity_fetch_count"],
            },
        )
    )

    bad_eval = evaluate_source_collection_trace(
        scenario=scenario,
        tool_calls=[
            {"tool": "web_fetch", "url": "https://www.morningstar.com/markets"},
            {"tool": "web_search", "query": "EPAM stock major jump today"},
        ],
    )
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

    policy_eval = evaluate_policy_trace(
        trace_events=[
            {"phase": "policy", "control_policy": "plan_execute", "ledger_mode": "ledger_enforced", "status": "active"},
            {"phase": "policy_violation", "type": "policy_violation", "issue": "unlocked_entity_search", "recovery_class": "entity_drift"},
            {"phase": "recovery_started", "type": "recovery_started", "recovered_tool": "web_search"},
            {"phase": "tool", "tool": "web_search", "query": "ROKU stock news June 13 2026", "status": "success"},
            {"phase": "completion_gate", "action": "completion_gate", "status": "blocked"},
        ],
        expected_policy="plan_execute",
        expected_ledger_mode="ledger_enforced",
        require_completion_gate=True,
        require_recovery_for_violations=True,
    )
    records.append(
        _record(
            "policy_trace_recovery_contract",
            policy_eval.passed,
            score=policy_eval.score,
            issues=policy_eval.issues,
            summary=policy_eval.detail,
            evidence=policy_eval.state,
        )
    )

    ledger = RunLedger(
        run_id="harness-lab-run",
        session_id="harness-lab-session",
        turn_id="harness-lab-turn",
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

    with tempfile.TemporaryDirectory(prefix="shovs-harness-lab-") as tmp:
        graph = SemanticGraph(db_path=str(Path(tmp) / "memory_graph.db"))
        memory = ShovsMemory(session_id="harness-lab-memory", owner_id="harness-lab-owner", graph=graph)
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

    passed = all(item["passed"] for item in records)
    return {
        "suite": "agent_harness_core",
        "version": 1,
        "passed": passed,
        "score": round(sum(item["score"] for item in records) / max(1, len(records)), 4),
        "results": records,
        "mode_implications": [
            {
                "mode": mode["id"],
                "label": mode["label"],
                "coverage": len(mode["included_wedges"]),
                "included_wedges": mode["included_wedges"],
            }
            for mode in MODES
        ],
    }


class _RuntimeHarnessAsyncIter:
    def __init__(self, items: list[str]):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


class _RuntimeHarnessAdapter:
    async def complete(self, **_kwargs):
        return json.dumps(
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "image_generate",
                            "arguments": json.dumps({"prompt": "a calm product mockup"}),
                        }
                    }
                ]
            }
        )

    def stream(self, **_kwargs):
        return _RuntimeHarnessAsyncIter(["Generated image ready."])

    async def list_models(self):
        return ["runtime-harness-small-model"]

    async def health(self):
        return True


class _RuntimeHarnessContext:
    def build_context_block(self, *_args, **_kwargs):
        return ""

    async def compress_exchange(self, *_args, **_kwargs):
        return "", [], []


class _RuntimeHarnessTraceStore:
    def __init__(self):
        self.events: list[dict[str, Any]] = []

    def append_event(self, agent_id, session_id, event_type, data, **kwargs):
        event = {
            "agent_id": agent_id,
            "session_id": session_id,
            "event_type": event_type,
            "data": data,
            **kwargs,
        }
        self.events.append(event)
        return event


async def _run_image_generation_runtime_probe(tmp: Path) -> dict[str, Any]:
    from orchestration.run_store import RunStore
    from orchestration.session_manager import SessionManager
    from plugins.tool_registry import Tool, ToolRegistry
    from run_engine.engine import RunEngine
    from run_engine.types import RunEngineRequest

    async def image_generate(prompt: str, **_kwargs):
        return json.dumps(
            {
                "type": "image_generation_result",
                "url": "/sandbox/generated/images/runtime-harness.png",
                "path": "generated/images/runtime-harness.png",
                "prompt": prompt,
                "model": "runtime-harness-image-model",
                "bytes": 12,
            }
        )

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="image_generate",
            description="Generate a deterministic image artifact for runtime harness testing.",
            parameters={
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
            handler=image_generate,
            response_format="json",
        )
    )
    trace_store = _RuntimeHarnessTraceStore()
    engine = RunEngine(
        adapter=_RuntimeHarnessAdapter(),
        sessions=SessionManager(db_path=str(tmp / "sessions.db")),
        tool_registry=registry,
        run_store=RunStore(db_path=str(tmp / "runs.db")),
        trace_store=trace_store,
        orchestrator=None,
        context_engine=_RuntimeHarnessContext(),
        graph=SemanticGraph(db_path=str(tmp / "memory.db")),
    )
    events = [
        event
        async for event in engine.stream(
            RunEngineRequest(
                session_id="runtime-harness-image",
                owner_id="runtime-harness-owner",
                agent_id="default",
                user_message="Generate an image of a calm product mockup",
                model="llama3.2",
                system_prompt="You are Shovs.",
                allowed_tools=("image_generate",),
                use_planner=False,
            )
        )
    ]
    tool_call = next((event for event in events if event.get("type") == "tool_call"), {})
    tool_result = next((event for event in events if event.get("type") == "tool_result"), {})
    payload = {}
    try:
        payload = json.loads(str(tool_result.get("content") or "{}"))
    except Exception:
        payload = {}
    passed = (
        tool_call.get("tool_name") == "image_generate"
        and tool_result.get("success") is True
        and payload.get("type") == "image_generation_result"
        and any(event.get("type") == "done" for event in events)
    )
    return _record(
        "runtime_image_generation_tool_loop",
        passed,
        summary="Actual RunEngine stream selected image_generate, executed the tool, and returned a structured image artifact.",
        evidence={
            "event_types": [event.get("type") for event in events],
            "tool_call": tool_call,
            "tool_result_payload": payload,
            "trace_event_count": len(trace_store.events),
        },
    )


def _run_fake_tool_claim_probe() -> dict[str, Any]:
    ledger = RunLedger(
        run_id="runtime-fake-claim",
        session_id="runtime-fake-claim-session",
        turn_id="turn-1",
        objective="Detect unsupported tool work claims.",
        allowed_tools=["web_search"],
        ledger_mode="ledger_enforced",
    )
    support = ledger.response_support_check("I searched the web and verified the answer.")
    return _record(
        "runtime_ledger_rejects_fake_tool_claim",
        support.get("supported") is False
        and "response_claims_tool_work_without_successful_tool_result" in support.get("issues", []),
        summary="Ledger support check rejected a final answer that claimed tool work without a successful tool result.",
        evidence=support,
    )


def _run_entity_drift_recovery_probe(tmp: Path) -> dict[str, Any]:
    from orchestration.run_store import RunStore
    from orchestration.session_manager import SessionManager
    from plugins.tool_registry import ToolCall, ToolRegistry
    from run_engine.engine import RunEngine

    trace_store = _RuntimeHarnessTraceStore()
    engine = RunEngine(
        adapter=_RuntimeHarnessAdapter(),
        sessions=SessionManager(db_path=str(tmp / "drift_sessions.db")),
        tool_registry=ToolRegistry(),
        run_store=RunStore(db_path=str(tmp / "drift_runs.db")),
        trace_store=trace_store,
        orchestrator=None,
        context_engine=_RuntimeHarnessContext(),
        graph=SemanticGraph(db_path=str(tmp / "drift_memory.db")),
    )
    ledger = RunLedger(
        run_id="runtime-drift",
        session_id="runtime-drift-session",
        turn_id="turn-1",
        objective="Search ROKU, TBN, SENEA separately and fetch sources.",
        allowed_tools=["web_search", "web_fetch"],
        ledger_mode="ledger_enforced",
    )
    ledger.set_control_policy(resolve_control_policy(ledger.objective, requested="plan_execute"))
    ledger.lock_entities(["ROKU", "TBN", "SENEA"])
    ledger.set_source_contract(
        {
            "next_tool": "web_search",
            "next_arguments": {"query": "ROKU stock news June 13 2026"},
            "forbid_unlocked_entity_drift": True,
        },
        source="runtime_harness",
    )
    recovered, event = engine._gate_tool_call_with_ledger(
        request=SimpleNamespace(session_id="runtime-drift-session", agent_id="default", owner_id="runtime-harness-owner"),
        run_id="runtime-drift",
        ledger=ledger,
        tool_call=ToolCall("web_search", {"query": "EPAM stock news June 13 2026"}, "{}"),
        tool_turn=1,
    )
    passed = (
        recovered is not None
        and recovered.tool_name == "web_search"
        and recovered.arguments == {"query": "ROKU stock news June 13 2026"}
        and event
        and event.get("type") == "recovery_started"
    )
    return _record(
        "runtime_policy_gate_recovers_entity_drift",
        passed,
        summary="Plan-execute ledger gate recovered an off-entity search into the next required locked-entity search.",
        evidence={
            "recovered_tool": recovered.tool_name if recovered else "",
            "recovered_arguments": recovered.arguments if recovered else {},
            "gate_event": event or {},
            "trace_events": trace_store.events,
        },
    )


async def run_runtime_harness_tests() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="shovs-runtime-harness-") as raw_tmp:
        tmp = Path(raw_tmp)
        records = [
            await _run_image_generation_runtime_probe(tmp),
            _run_fake_tool_claim_probe(),
            _run_entity_drift_recovery_probe(tmp),
        ]
    passed = all(item["passed"] for item in records)
    return {
        "suite": "shovs_runtime_harness",
        "version": 1,
        "deterministic": True,
        "live_model_required": False,
        "network_required": False,
        "passed": passed,
        "score": round(sum(item["score"] for item in records) / max(1, len(records)), 4),
        "results": records,
        "coverage": {
            "entrypoints": ["RunEngine.stream", "RunLedger.response_support_check", "RunEngine._gate_tool_call_with_ledger"],
            "failure_modes": ["tool_not_called", "fake_tool_claim", "entity_drift"],
            "frontend_contract": ["tool_call", "tool_result", "image_generation_result", "done"],
        },
        "operator_guidance": [
            "If runtime_image_generation_tool_loop fails, the model/tool router is not selecting or executing creation tools.",
            "If runtime_ledger_rejects_fake_tool_claim fails, final answers can claim tool work without evidence.",
            "If runtime_policy_gate_recovers_entity_drift fails, locked-entity workflows can drift to unrelated searches.",
        ],
    }


def make_harness_lab_router() -> APIRouter:
    router = APIRouter(prefix="/harness-lab", tags=["harness-lab"])

    @router.get("/catalog")
    async def catalog():
        return get_harness_lab_catalog()

    @router.post("/benchmark/run")
    async def benchmark_run(_payload: Optional[dict] = Body(default=None)):
        return run_harness_core_benchmark()

    @router.post("/compare/run")
    async def compare_run(payload: Optional[dict] = Body(default=None)):
        data = payload or {}
        modes = data.get("modes") if isinstance(data.get("modes"), list) else None
        return run_mode_comparison(str(data.get("objective") or ""), modes=modes)

    @router.post("/runtime/run")
    async def runtime_run(_payload: Optional[dict] = Body(default=None)):
        return await run_runtime_harness_tests()

    return router
