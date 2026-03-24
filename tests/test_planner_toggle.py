import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from engine.core import AgentCore
from orchestration.session_manager import SessionManager
from plugins.tool_registry import ToolRegistry, Tool


class AsyncIter:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


@pytest.mark.asyncio
async def test():
    """
    Planner/direct-mode event behavior:
    - use_planner=True      -> emits plan when planner returns tools
    - use_planner=False     -> emits heuristic plan for factual query
    - use_planner='false'   -> still parses bool and uses direct-mode hints
    """
    adapter = MagicMock()
    adapter.stream = MagicMock(side_effect=lambda *a, **k: AsyncIter(["hello"]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    orchestrator = MagicMock()
    orchestrator.plan_with_context = AsyncMock(return_value={
        "strategy": "Use web_search",
        "tools": [{"name": "web_search", "priority": "high", "reason": "factual"}],
        "force_memory": False,
        "confidence": 0.8,
    })
    orchestrator.set_adapter = MagicMock()

    registry = ToolRegistry()
    registry.register(Tool(
        name="web_search",
        description="search web",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda **kwargs: "",
    ))
    registry.register(Tool(
        name="query_memory",
        description="query memory",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda **kwargs: "",
    ))

    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=registry,
        orchestrator=orchestrator,
    )

    with patch("engine.core.VectorEngine") as mock_ve:
        ve = MagicMock()
        ve.query = AsyncMock(return_value=[])
        mock_ve.return_value = ve

        events_true = [ev async for ev in core.chat_stream("research ai news then summarize it", use_planner=True)]
        assert any(ev.get("type") == "plan" for ev in events_true)

        orchestrator.plan_with_context.reset_mock()
        events_false = [ev async for ev in core.chat_stream("latest ai news", use_planner=False)]
        assert any(ev.get("type") == "plan" for ev in events_false)
        orchestrator.plan_with_context.assert_not_called()

        orchestrator.plan_with_context.reset_mock()
        events_false_str = [ev async for ev in core.chat_stream("latest ai news", use_planner="false")]
        assert any(ev.get("type") == "plan" for ev in events_false_str)
        orchestrator.plan_with_context.assert_not_called()
