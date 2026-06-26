"""One clean, domain-agnostic end-to-end test of the dynamic agent loop.

It proves the runtime's *architecture* — not any domain — by driving a single
turn through the full cycle the user cares about:

    understand intent -> plan -> execute -> failure -> heal/retry -> figure it
    out -> finalize -> respond

The tool, plan, and observations are generic ("fetch_data" from a primary then
a fallback source). The model + orchestrator are scripted so the test is
deterministic and provider-independent: it exercises the *loop machinery*, which
is exactly what was failing live (planner-failure -> raw output, no recovery).
"""

from unittest.mock import MagicMock

import pytest

from memory.semantic_graph import SemanticGraph
from orchestration.run_store import RunStore
from orchestration.session_manager import SessionManager
from plugins.tool_registry import Tool, ToolRegistry
from run_engine.engine import RunEngine
from run_engine.types import RunEngineRequest


class AsyncIter:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


class FakeTraceStore:
    def __init__(self):
        self.events = []

    def append_event(self, agent_id, session_id, event_type, data, **kwargs):
        event = {"event_type": event_type, "data": data, **kwargs}
        self.events.append(event)
        return event


@pytest.mark.asyncio
async def test_dynamic_loop_plans_executes_heals_and_completes(tmp_path):
    # --- scripted model: returns a fetch_data tool call, primary first then
    #     fallback on the second acting turn (adaptive recovery, not blind repeat).
    complete_calls = {"n": 0}

    async def _complete(**kwargs):
        complete_calls["n"] += 1
        source = "primary" if complete_calls["n"] == 1 else "fallback"
        return (
            '{"tool_calls": [{"function": {"name": "fetch_data", '
            '"arguments": "{\\"source\\": \\"%s\\"}"}}]}' % source
        )

    adapter = MagicMock()
    adapter.complete = _complete
    adapter.stream = MagicMock(return_value=AsyncIter(["Done — ", "recovered ", "the data."]))

    # --- scripted orchestrator: plan once; observe says CONTINUE after the
    #     failure (heal), then FINALIZE after the success (figure it out).
    observe_calls = {"n": 0}

    async def _plan(**kwargs):
        return {
            "strategy": "Fetch the data from the primary source, then answer.",
            "tools": [{"name": "fetch_data", "priority": "high", "reason": "need the data"}],
            "confidence": 0.9,
        }

    async def _observe(**kwargs):
        observe_calls["n"] += 1
        if observe_calls["n"] == 1:
            return {
                "status": "continue",
                "strategy": "Primary source failed (503); recover via the fallback source.",
                "tools": [{"name": "fetch_data", "priority": "high", "reason": "retry via fallback"}],
                "notes": "Transient failure — switch source and retry.",
                "confidence": 0.6,
            }
        return {
            "status": "finalize",
            "strategy": "Fallback succeeded; answer from the recovered data.",
            "tools": [],
            "notes": "Have the data now.",
            "confidence": 0.9,
        }

    async def _verify(**kwargs):
        return {"supported": True, "issues": [], "confidence": 0.95}

    orchestrator = MagicMock()
    orchestrator.plan_with_context = _plan
    orchestrator.observe_with_context = _observe
    orchestrator.verify_with_context = _verify

    # --- generic tool: fails on the primary source, succeeds on the fallback.
    sources_tried: list[str] = []

    async def fetch_data(source: str = "primary", **kwargs):
        sources_tried.append(source)
        if source == "primary":
            return {"type": "error", "success": False, "error": "primary source unavailable (HTTP 503)"}
        return {"type": "result", "success": True, "output": f"data from {source}"}

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="fetch_data",
            description="Fetch a record from a named source.",
            parameters={
                "type": "object",
                "properties": {"source": {"type": "string", "description": "Source name."}},
                "required": [],
            },
            handler=fetch_data,
            response_format="json",
        )
    )

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""

    async def _compress(*args, **kwargs):
        return ("ctx", [], [])

    context_engine.compress_exchange = _compress

    sessions = SessionManager(db_path=str(tmp_path / "sessions.db"))
    runs = RunStore(db_path=str(tmp_path / "runs.db"))
    traces = FakeTraceStore()
    graph = SemanticGraph(db_path=str(tmp_path / "memory.db"))

    engine = RunEngine(
        adapter=adapter,
        sessions=sessions,
        tool_registry=registry,
        run_store=runs,
        trace_store=traces,
        orchestrator=orchestrator,
        context_engine=context_engine,
        graph=graph,
    )
    engine._context_governor._v3_engine = context_engine

    request = RunEngineRequest(
        session_id="dyn-loop-1",
        owner_id="owner-dyn",
        agent_id="default",
        user_message="Get me the data and summarize it.",
        model="llama3.2",
        system_prompt="You are a careful assistant.",
        allowed_tools=("fetch_data",),
        use_planner=True,
        max_turns=4,
    )

    events = [event async for event in engine.stream(request)]
    event_types = [e["type"] for e in events]

    # 1) understood intent + planned
    assert "plan" in event_types

    # 2) executed, then HEALED: tried primary, it failed, switched to fallback.
    assert sources_tried == ["primary", "fallback"], sources_tried
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_calls) >= 2, f"expected a retry, got {len(tool_calls)} tool calls"

    # 3) the first attempt is recorded as a failure, the last as a success
    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert len(tool_results) >= 2
    assert any(str(r.get("status")) == "failed" for r in tool_results), tool_results
    assert str(tool_results[-1].get("status")) == "ok", tool_results[-1]

    # 4) figured it out and completed with a real answer
    assert "done" in event_types
    final = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert final == "Done — recovered the data."

    # 5) the dynamic transitions are recorded: observe drove continue -> finalize,
    #    verification passed, response completed.
    run_id = next(e["run_id"] for e in events if e["type"] == "session")
    pass_phases = {r.phase for r in runs.list_passes(run_id)}
    assert {"planning", "acting", "observation", "response", "verification"} <= pass_phases
    assert observe_calls["n"] >= 2  # continue (heal) then finalize (figure out)
    assert any(
        r.phase == "observation" and r.status == "finalize" for r in runs.list_passes(run_id)
    )
    assert any(
        r.phase == "verification" and r.status == "verified" for r in runs.list_passes(run_id)
    )
