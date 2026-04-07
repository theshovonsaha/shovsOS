import pytest
from unittest.mock import AsyncMock, MagicMock

from plugins.tool_registry import ToolRegistry
from plugins.tools import register_all_tools


@pytest.mark.asyncio
async def test_delegation():
    """
    Delegation tool should call AgentManager.run_agent_task with correct parent_id wiring.
    Kept hermetic: no live LLM/network dependency.
    """
    registry = ToolRegistry()
    manager = MagicMock()
    manager.run_agent_task = AsyncMock(return_value="The delegation was successful!")

    register_all_tools(registry, agent_manager=manager)
    delegate_tool = registry.get("delegate_to_agent")
    assert delegate_tool is not None

    result = await delegate_tool.handler(target_agent_id="default", task="Say success")
    assert "successful" in result.lower()
    manager.run_agent_task.assert_awaited_once_with(
        "default",
        "Say success",
        parent_id=None,
        parent_run_id=None,
        owner_id=None,
        runtime_kind_override="managed",
    )

    manager.run_agent_task.reset_mock()
    await delegate_tool.handler(target_agent_id="coder", task="Write script", _session_id="parent_123")
    manager.run_agent_task.assert_awaited_once_with(
        "coder",
        "Write script",
        parent_id="parent_123",
        parent_run_id=None,
        owner_id=None,
        runtime_kind_override="managed",
    )


@pytest.mark.asyncio
async def test_delegation_from_run_engine_uses_managed_runtime_override():
    registry = ToolRegistry()
    manager = MagicMock()
    manager.run_agent_task = AsyncMock(return_value="managed delegation ok")

    register_all_tools(registry, agent_manager=manager)
    delegate_tool = registry.get("delegate_to_agent")
    assert delegate_tool is not None

    await delegate_tool.handler(
        target_agent_id="researcher",
        task="Collect recent pricing data",
        _session_id="parent_456",
        _run_id="run_123",
        _owner_id="owner_abc",
        _runtime_path="run_engine",
    )

    manager.run_agent_task.assert_awaited_once_with(
        "researcher",
        "Collect recent pricing data",
        parent_id="parent_456",
        parent_run_id="run_123",
        owner_id="owner_abc",
        runtime_kind_override="managed",
    )
