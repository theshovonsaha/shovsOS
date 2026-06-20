from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body

from engine.response_guard import guard_final_response
from memory.semantic_graph import SemanticGraph
from run_engine.ledger import RunLedger
from run_engine.scenario_eval import SourceCollectionScenario, evaluate_source_collection_trace
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


def make_harness_lab_router() -> APIRouter:
    router = APIRouter(prefix="/harness-lab", tags=["harness-lab"])

    @router.get("/catalog")
    async def catalog():
        return get_harness_lab_catalog()

    @router.post("/benchmark/run")
    async def benchmark_run(_payload: Optional[dict] = Body(default=None)):
        return run_harness_core_benchmark()

    return router
