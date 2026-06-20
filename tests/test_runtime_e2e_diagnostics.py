import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.semantic_graph import SemanticGraph
from orchestration.run_store import RunStore
from orchestration.session_manager import SessionManager
from plugins.tool_registry import Tool, ToolRegistry
from run_engine.engine import RunEngine
from run_engine.scenario_eval import SourceCollectionScenario, evaluate_source_collection_trace
from run_engine.types import RunEngineRequest


class AsyncIter:
    def __init__(self, items):
        self.items = list(items)

    async def __aiter__(self):
        for item in self.items:
            yield item


class DiagnosticTraceStore:
    def __init__(self):
        self.events = []

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


def _search_payload(query: str) -> dict:
    query_upper = query.upper()
    if "MORNINGSTAR" in query_upper or "TOP 3" in query_upper:
        return {
            "type": "web_search_results",
            "query": query,
            "results": [
                {
                    "title": "Morningstar Market Movers",
                    "url": "https://www.morningstar.com/markets/movers",
                    "snippet": "Top stock market gainers, losers, and active stocks.",
                },
                {
                    "title": "EPAM stock major jump today",
                    "url": "https://noise.example/epam",
                    "snippet": "A noisy search result that must not become a selected entity.",
                },
                {
                    "title": "US PM market recap",
                    "url": "https://noise.example/us-pm",
                    "snippet": "Uppercase tokens that must not become selected entities.",
                },
            ],
        }

    ticker = query_upper.split()[0]
    return {
        "type": "web_search_results",
        "query": query,
        "results": [
            {
                "title": f"{ticker} company update",
                "url": f"https://news.example/{ticker.lower()}-1",
                "snippet": f"{ticker} announces operational update after market move.",
            },
            {
                "title": f"{ticker} analyst reaction",
                "url": f"https://news.example/{ticker.lower()}-2",
                "snippet": f"Analysts discuss {ticker} price action and volume.",
            },
            {
                "title": f"{ticker} market context",
                "url": f"https://news.example/{ticker.lower()}-3",
                "snippet": f"Market report covering {ticker} and sector context.",
            },
        ],
    }


def _fetch_payload(url: str) -> dict:
    if url == "https://www.morningstar.com/markets/movers":
        return {
            "type": "web_fetch_result",
            "url": url,
            "title": "Top Stock Market Gainers, Losers, and Most Active Stocks",
            "content": """
## Gainers
| 1-Day Chart | Stock | Price | Volume/Average | Status |
| --- | --- | --- | --- | --- |
| | [Roku Inc](https://www.morningstar.com/stocks/xnas/roku/quote) ROKU | $143.66 +24.02 (20.08%) | 15M 3M | Closed |
| | [Tamboran Resources Corp](https://www.morningstar.com/stocks/xnys/tbn/quote) TBN | $40.37 +6.71 (19.93%) | 509,942 222,049 | Closed |
| | [Seneca Foods Corp](https://www.morningstar.com/stocks/xnas/senea/quote) SENEA | $175.37 +26.16 (17.53%) | 538,088 133,799 | Closed |
## Losers
""",
        }
    ticker = Path(url).name.split("-")[0].upper()
    return {
        "type": "web_fetch_result",
        "url": url,
        "title": f"{ticker} fetched source",
        "content": f"{ticker} source content from {url}. This is deterministic article text for synthesis.",
    }


@pytest.mark.asyncio
async def test_managed_runtime_source_collection_e2e_diagnostic_report(tmp_path):
    """End-to-end managed-runtime diagnostic without live LLM calls.

    This touches the practical seams that have been drifting in real runs:
    planner, actor fallback, tools, observation, source contract traces,
    run ledger, run store, verifier, memory skip policy, and report artifact.
    """

    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value="{}")
    adapter.stream = MagicMock(
        return_value=AsyncIter([
            "Ticker | URLs fetched | Summary\n",
            "ROKU | 3 | Verified from fetched sources.\n",
            "TBN | 3 | Verified from fetched sources.\n",
            "SENEA | 3 | Verified from fetched sources.\n",
        ])
    )

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(
        return_value={
            "strategy": "Identify market movers, then collect three source URLs per locked ticker.",
            "tools": [
                {
                    "name": "web_search",
                    "priority": "high",
                    "reason": "Find a market movers source.",
                    "target_argument_clue": "Morningstar market movers top stock gainers today",
                }
            ],
            "plan_steps": [
                {"id": "discover", "description": "Find market movers source", "tool": "web_search", "status": "pending"},
                {"id": "lock", "description": "Fetch movers table and lock top entities", "tool": "web_fetch", "status": "pending"},
                {"id": "search", "description": "Search each locked ticker separately", "tool": "web_search", "status": "pending"},
                {"id": "fetch", "description": "Fetch three URLs per locked ticker", "tool": "web_fetch", "status": "pending"},
            ],
            "confidence": 0.95,
            "risk_tier": "read_only",
        }
    )
    orchestrator.observe_with_context = AsyncMock(
        return_value={
            "status": "finalize",
            "strategy": "Model observation intentionally says finalize; deterministic source contract must override until complete.",
            "tools": [],
            "notes": "Finalize only if source contract is satisfied.",
            "confidence": 0.4,
        }
    )
    orchestrator.verify_with_context = AsyncMock(
        return_value={"supported": True, "verdict": "supported", "issues": [], "confidence": 0.99}
    )

    registry = ToolRegistry()
    call_log = []

    async def web_search(query: str, **kwargs):
        call_log.append({"tool": "web_search", "query": query})
        return _search_payload(query)

    async def web_fetch(url: str, **kwargs):
        call_log.append({"tool": "web_fetch", "url": url})
        return _fetch_payload(url)

    registry.register(
        Tool(
            name="web_search",
            description="Search the web.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            handler=web_search,
            response_format="json",
        )
    )
    registry.register(
        Tool(
            name="web_fetch",
            description="Fetch a URL.",
            parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            handler=web_fetch,
            response_format="json",
        )
    )

    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    runs = RunStore(db_path=str(tmp_path / "runs.db"))
    traces = DiagnosticTraceStore()
    context_engine = MagicMock()
    context_engine.compress_exchange = AsyncMock(return_value=("diagnostic ctx", [], []))
    engine = RunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=registry,
        run_store=runs,
        trace_store=traces,
        orchestrator=orchestrator,
        context_engine=context_engine,
        graph=SemanticGraph(db_path=str(tmp_path / "memory.db")),
    )

    request = RunEngineRequest(
        session_id="diagnostic-source-contract",
        owner_id="owner-diagnostic",
        agent_id="default",
        user_message=(
            "Search top 3 stocks today with major jumps web search those 3 stocks separately "
            "and capture the 3 relevant for each results web fetch all 9 urls that was found "
            "one by one analyze report for each and write a tldr summary table."
        ),
        model="llama3.2",
        system_prompt="You are Shovs.",
        allowed_tools=("web_search", "web_fetch"),
        use_planner=True,
    )

    events = [event async for event in engine.stream(request)]
    run_id = next(event["run_id"] for event in events if event["type"] == "session")

    searched_queries = [item["query"] for item in call_log if item["tool"] == "web_search"]
    fetched_urls = [item["url"] for item in call_log if item["tool"] == "web_fetch"]
    ticker_fetches = [url for url in fetched_urls if url.startswith("https://news.example/")]
    source_contract_events = [event for event in traces.events if event["event_type"] == "source_contract"]
    tool_result_events = [event for event in traces.events if event["event_type"] == "tool_result"]
    phase_packet_events = [event for event in traces.events if event["event_type"] == "phase_packet"]
    run_ledger_events = [event for event in traces.events if event["event_type"] == "run_ledger"]
    pass_records = runs.list_passes(run_id)

    diagnostics = {
        "run_id": run_id,
        "searched_queries": searched_queries,
        "fetched_urls": fetched_urls,
        "ticker_fetches": ticker_fetches,
        "source_contract_events": [
            {
                "status": event["data"].get("status"),
                "strategy": event["data"].get("strategy"),
                "selected_tools": event["data"].get("selected_tools"),
                "entities": event["data"].get("entities"),
                "source_contract": event["data"].get("source_contract"),
                "missing_slots": event["data"].get("missing_slots"),
            }
            for event in source_contract_events
        ],
        "trace_event_counts": {
            "source_contract": len(source_contract_events),
            "tool_result": len(tool_result_events),
            "phase_packet": len(phase_packet_events),
            "run_ledger": len(run_ledger_events),
        },
        "pass_phases": [record.phase for record in pass_records],
        "non_blocking_findings": [],
    }
    if any("EPAM" in query or query.startswith("US ") or query.startswith("PM ") for query in searched_queries):
        diagnostics["non_blocking_findings"].append("noisy ticker query escaped entity lock")
        if len(ticker_fetches) != 9:
            diagnostics["non_blocking_findings"].append(f"expected 9 ticker fetches, saw {len(ticker_fetches)}")

    scenario_eval = evaluate_source_collection_trace(
        scenario=SourceCollectionScenario(
            objective=request.user_message,
            entities=["ROKU", "TBN", "SENEA"],
            urls_per_entity=3,
            total_urls=9,
            discovery_url="https://www.morningstar.com/markets/movers",
            forbidden_query_terms=["EPAM", "ARKO", "US ", "PM "],
            allowed_fetch_urls_by_entity={
                "ROKU": [
                    "https://news.example/roku-1",
                    "https://news.example/roku-2",
                    "https://news.example/roku-3",
                ],
                "TBN": [
                    "https://news.example/tbn-1",
                    "https://news.example/tbn-2",
                    "https://news.example/tbn-3",
                ],
                "SENEA": [
                    "https://news.example/senea-1",
                    "https://news.example/senea-2",
                    "https://news.example/senea-3",
                ],
            },
        ),
        tool_calls=call_log,
        source_contract_events=source_contract_events,
    )
    diagnostics["scenario_eval"] = scenario_eval.to_dict()

    report_path = tmp_path / "diagnostic_source_contract_report.json"
    report_text = json.dumps(diagnostics, indent=2, sort_keys=True)
    report_path.write_text(report_text)
    runs.save_artifact(
        run_id=run_id,
        session_id=request.session_id,
        owner_id=request.owner_id,
        artifact_type="diagnostic_report",
        label="Source contract E2E diagnostic report",
        storage_path=str(report_path),
        content_hash=hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
        size_bytes=len(report_text.encode("utf-8")),
        preview=report_text[:500],
        metadata={
            "source_contract_events": len(source_contract_events),
            "ticker_fetch_count": len(ticker_fetches),
            "non_blocking_findings": diagnostics["non_blocking_findings"],
        },
    )
    stored_eval = runs.save_eval(
        run_id=run_id,
        session_id=request.session_id,
        owner_id=request.owner_id,
        eval_type="source_collection_scenario",
        phase="end_to_end",
        passed=scenario_eval.passed,
        score=scenario_eval.score,
        detail=scenario_eval.detail,
        metadata=scenario_eval.to_dict(),
    )

    assert "ROKU stock news June 13 2026" in searched_queries
    assert "TBN stock news June 13 2026" in searched_queries
    assert "SENEA stock news June 13 2026" in searched_queries
    assert not any("EPAM" in query or query.startswith("US ") or query.startswith("PM ") for query in searched_queries)
    assert len(ticker_fetches) == 9
    assert "https://www.morningstar.com/markets/movers" in fetched_urls
    assert source_contract_events
    assert any(event["data"].get("entities") == ["ROKU", "TBN", "SENEA"] for event in source_contract_events)
    assert any(event["data"].get("source_contract", {}).get("total_urls") == 9 for event in source_contract_events)
    assert any(event["data"]["summary"]["tool_results"] >= 10 for event in run_ledger_events)
    assert {record.phase for record in pass_records} >= {"planning", "acting", "observation", "response", "verification"}
    assert runs.list_artifacts(run_id)[-1].artifact_type == "diagnostic_report"
    assert stored_eval.passed is True
    assert runs.list_evals(run_id)[-1].eval_type == "source_collection_scenario"
    assert diagnostics["non_blocking_findings"] == []


def test_source_collection_scenario_eval_catches_entity_drift():
    scenario_eval = evaluate_source_collection_trace(
        scenario=SourceCollectionScenario(
            objective="Collect three source URLs for each locked entity.",
            entities=["ROKU", "TBN", "SENEA"],
            urls_per_entity=3,
            total_urls=9,
            discovery_url="https://www.morningstar.com/markets/movers",
            forbidden_query_terms=["EPAM", "ARKO"],
            allowed_fetch_urls_by_entity={
                "ROKU": ["https://news.example/roku-1", "https://news.example/roku-2", "https://news.example/roku-3"],
                "TBN": ["https://news.example/tbn-1", "https://news.example/tbn-2", "https://news.example/tbn-3"],
                "SENEA": ["https://news.example/senea-1", "https://news.example/senea-2", "https://news.example/senea-3"],
            },
        ),
        tool_calls=[
            {"tool": "web_search", "query": "Morningstar market movers top stock gainers today"},
            {"tool": "web_fetch", "url": "https://www.morningstar.com/markets/movers"},
            {"tool": "web_search", "query": "EPAM stock news June 13 2026"},
            {"tool": "web_fetch", "url": "https://news.example/epam-1"},
        ],
        source_contract_events=[],
    )

    assert scenario_eval.passed is False
    assert "forbidden_query_drift" in scenario_eval.issues
    assert "missing_targeted_search:ROKU" in scenario_eval.issues
    assert any(issue.startswith("missing_total_fetch_quota") for issue in scenario_eval.issues)
